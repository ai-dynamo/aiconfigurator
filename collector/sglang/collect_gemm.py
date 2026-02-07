# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pkg_resources
import torch
import torch.nn.functional as F
from common_test_cases import get_gemm_common_test_cases
from sgl_kernel import (
    fp8_scaled_mm,
    sgl_per_token_quant_fp8,
)

try:
    from flashinfer import fp4_quantize as flashinfer_fp4_quantize
    from flashinfer import mm_fp4 as flashinfer_mm_fp4
    from flashinfer import shuffle_matrix_sf_a as flashinfer_shuffle_sf_a

    HAS_FLASHINFER_FP4 = True
except ImportError:
    HAS_FLASHINFER_FP4 = False

from sglang.srt.layers.deep_gemm_wrapper import (
    DEEPGEMM_SCALE_UE8M0,
    gemm_nt_f8f8bf16,
)
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

from helper import benchmark_with_power, get_sm_version, log_perf

compatible_versions = ["0.5.5.post2", "0.5.6.post2", "0.5.8"]


def get_gemm_test_cases():
    test_cases = []

    # fp8_block (DeepGEMM) requires SM90+ for TMA support
    sm_version = get_sm_version()
    if sm_version < 90:
        # SM89 (L40S) and earlier don't have TMA - skip fp8_block
        gemm_list = ["float16", "fp8"]
    elif sm_version < 100:
        # Hopper supports fp8_block
        gemm_list = ["fp8_block", "float16", "fp8"]
    else:
        # Blackwell supports nvfp4
        gemm_list = ["fp8_block", "float16", "fp8", "nvfp4"]

    for gemm_common_testcase in get_gemm_common_test_cases():
        x = gemm_common_testcase.x
        n = gemm_common_testcase.n
        k = gemm_common_testcase.k
        for gemm_type in gemm_list:
            if (gemm_type == "nvfp4" or gemm_type == "fp8_block") and (n < 128 or k < 128):
                continue

            test_cases.append([gemm_type, x, n, k, "gemm_perf.txt"])

    return test_cases


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


def fp8_gemm_deepgemm(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    y_fp8: torch.Tensor,
    y_scale: torch.Tensor,
    out: torch.Tensor,
    m: int,
    n: int,
    k: int,
):
    """
    DeepGEMM implementation of FP8 GEMM
    It maps to a specific commit for each SGLang release.
    Check the commit tag in sglang/sgl-kernel/CMakeLists.txt, repo-deepgemm
    """

    # Run DeepGEMM kernel
    gemm_nt_f8f8bf16((x_fp8, x_scale), (y_fp8, y_scale), out)
    return out


def scale_shape(shape, group_shape):
    assert len(shape) == len(group_shape)
    return tuple(cdiv(shape[i], group_shape[i]) for i in range(len(group_shape)))


def per_token_quant_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize fp32/fp16/bf16 tensor to int8 with per-token scaling"""
    # Calculate per-row (per-token) scaling factor
    x_fp32 = x.to(torch.float32) if x.dtype != torch.float32 else x
    absmax = torch.max(torch.abs(x_fp32), dim=-1, keepdim=True)[0].clamp(min=1e-10)
    scale = absmax / 127.0

    # Quantize to int8
    x_scaled = x_fp32 / scale
    x_int8 = torch.round(x_scaled).clamp(-128, 127).to(torch.int8)

    # Return int8 tensor and scale (squeeze the last dimension for scale)
    return x_int8, scale.squeeze(-1)


def run_gemm(gemm_type, batch_size, N, K, perf_filename, device):  # noqa: N803
    assert gemm_type in [
        "fp8_block",
        "fp8",
        "float16",
        "int8_wo",
        "nvfp4",
    ], "not support gemm type"
    torch.cuda.set_device(device)
    M = batch_size  # noqa: N806

    def round_up(x, m):
        return (x + m - 1) // m * m

    def create_gemm():
        dtype = torch.bfloat16
        if gemm_type == "nvfp4":
            if not HAS_FLASHINFER_FP4:
                return None

            # Prepare source data: Activation A [M, K] in BF16
            a_bf16 = torch.randn((M, K), device=device, dtype=dtype)

            # Prepare Weight B [N, K] and its dummy scale
            b_bf16_dummy = torch.randn((N, K), device=device, dtype=dtype)

            # Global scales
            a_global_scale = torch.tensor([1.0], device=device, dtype=torch.float32)
            b_global_scale = torch.tensor([1.0], device=device, dtype=torch.float32)
            alpha = 1.0 / (a_global_scale * b_global_scale)

            # Pre-quantize and Shuffle Weight B (load time process)
            b_fp4_linear, b_sf_linear = flashinfer_fp4_quantize(
                b_bf16_dummy, b_global_scale, is_sf_swizzled_layout=False
            )

            from flashinfer import shuffle_matrix_a as flashinfer_shuffle_a

            epilogue_tile_m = 128
            b_fp4_shuffled = flashinfer_shuffle_a(b_fp4_linear, epilogue_tile_m)
            b_sf_shuffled = flashinfer_shuffle_sf_a(b_sf_linear.view(torch.uint8), epilogue_tile_m).view(
                torch.float8_e4m3fn
            )
            b_fp4_final = b_fp4_shuffled.t()
            b_sf_final = b_sf_shuffled.t()

            out = torch.empty((M, round_up(N, 128)), device=device, dtype=dtype)

            def gemm_op():
                # Dynamic Quantization of Activation + GEMM
                a_fp4_dynamic, a_sf_dynamic = flashinfer_fp4_quantize(
                    a_bf16, a_global_scale, is_sf_swizzled_layout=True
                )
                return flashinfer_mm_fp4(
                    a_fp4_dynamic, b_fp4_final, a_sf_dynamic, b_sf_final, alpha, dtype, backend="cutlass", out=out
                )

            return gemm_op

        elif gemm_type == "fp8_block":
            fp8_info = torch.finfo(torch.float8_e4m3fn)
            a_bf16 = torch.randn(M, K, dtype=dtype, device=device)
            b_fp32 = (torch.rand(N, K, device=device) - 0.5) * 2 * fp8_info.max
            b_fp8 = b_fp32.clamp(min=fp8_info.min, max=fp8_info.max).to(torch.float8_e4m3fn)
            scale_b = torch.randn(scale_shape(b_fp8.shape, (128, 128)), device=device, dtype=torch.float32)
            out = torch.empty((M, N), device=device, dtype=dtype)

            def gemm_op():
                a_fp8, scale_a = sglang_per_token_group_quant_fp8(
                    a_bf16,
                    group_size=128,
                    column_major_scales=True,
                    scale_tma_aligned=True,
                    scale_ue8m0=DEEPGEMM_SCALE_UE8M0,
                )
                return fp8_gemm_deepgemm(a_fp8, scale_a, b_fp8, scale_b, out, M, N, K)

            return gemm_op

        elif gemm_type == "fp8":
            fp8_info = torch.finfo(torch.float8_e4m3fn)
            a_bf16 = torch.randn(M, K, dtype=dtype, device=device)
            b_fp32 = (torch.rand(N, K, device=device) - 0.5) * 2 * fp8_info.max
            b_fp8 = b_fp32.clamp(min=fp8_info.min, max=fp8_info.max).to(torch.float8_e4m3fn).t()
            scale_b = torch.randn((N,), device=device, dtype=torch.float32)
            output_fp8 = torch.empty_like(a_bf16, dtype=torch.float8_e4m3fn)
            scale_a = torch.empty((M, 1), device=device, dtype=torch.float32)

            def gemm_op():
                sgl_per_token_quant_fp8(a_bf16, output_fp8, scale_a)
                return fp8_scaled_mm(output_fp8, b_fp8, scale_a, scale_b, dtype)

            return gemm_op

        elif gemm_type == "float16":
            a_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
            b_fp16 = torch.randn(N, K, dtype=torch.float16, device=device)

            def gemm_op():
                return F.linear(a_fp16, b_fp16, None)

            return gemm_op

    outside_loop_count = 6
    op_list = []
    for _ in range(outside_loop_count):
        op = create_gemm()
        if op is not None:
            op_list.append(op)

    if not op_list:
        print(f"No ops created for {gemm_type}, skipping.")
        return

    def kernel_func():
        for op in op_list:
            op()

    # Use benchmark_with_power context manager
    nvtx_tag = f"{gemm_type}_m{M}_n{N}_k{K}"
    torch.cuda.nvtx.range_push(nvtx_tag)

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=3,
        num_runs=6,
        repeat_n=1,
    ) as results:
        pass

    torch.cuda.nvtx.range_pop()

    log_perf(
        item_list=[
            {"gemm_dtype": gemm_type, "m": M, "n": N, "k": K, "latency": results["latency_ms"] / outside_loop_count}
        ],
        framework="SGLang",
        version=pkg_resources.get_distribution("sglang").version,
        device_name=torch.cuda.get_device_name(device),
        op_name="gemm",
        kernel_source="sglang",
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )

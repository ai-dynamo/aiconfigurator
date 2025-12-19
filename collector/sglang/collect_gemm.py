# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pkg_resources
import torch
import torch.nn.functional as F
from common_test_cases import get_gemm_common_test_cases
from sgl_kernel import fp8_scaled_mm, int8_scaled_mm, sgl_per_tensor_quant_fp8, sgl_per_token_quant_fp8
from sglang.srt.layers.deep_gemm_wrapper import (
    DEEPGEMM_SCALE_UE8M0,
    gemm_nt_f8f8bf16,
)
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

from helper import benchmark_with_power, get_sm_version, log_perf

compatible_versions = ["0.5.5.post2", "0.5.5.post3"]


def get_gemm_test_cases():
    test_cases = []

    # fp8_block (DeepGEMM) requires SM90+ for TMA support
    sm_version = get_sm_version()
    if sm_version < 90:
        # SM89 (L40S) and earlier don't have TMA - skip fp8_block
        gemm_list = ["int8_wo", "float16", "fp8"]
    else:
        gemm_list = ["int8_wo", "fp8_block", "float16", "fp8"]

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
    ], "not support gemm type"
    torch.cuda.set_device(device)
    M = batch_size  # noqa: N806

    if gemm_type == "fp8_block" or gemm_type == "fp8":
        fp8_info = torch.finfo(torch.float8_e4m3fn)
        fp8_max, fp8_min = fp8_info.max, fp8_info.min

        a_fp32 = (torch.rand(M, K, dtype=torch.float32, device=device) - 0.5) * 2 * fp8_max
        b_fp32 = (torch.rand(N, K, dtype=torch.float32, device=device) - 0.5) * 2 * fp8_max

        # Use bf16 as input source for activation to simulate realistic scenario
        a_bf16 = a_fp32.to(torch.bfloat16)

        if gemm_type == "fp8_block":
            # Quantize B (weights) outside
            b_fp8 = b_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

            scale_b_group_shape = (128, 128)
            scale_b_shape = scale_shape(b_fp8.shape, scale_b_group_shape)
            scale_b = torch.randn(scale_b_shape, device=device, dtype=torch.float32)

            out = torch.empty((M, N), device=device, dtype=torch.bfloat16)

            def gemm_op():
                # Use the real SGLang kernel to quantize A dynamically
                # This computes the scale and quantizes A in one fused kernel
                a_fp8, scale_a_col_major = sglang_per_token_group_quant_fp8(
                    a_bf16,
                    group_size=128,
                    column_major_scales=True,
                    scale_tma_aligned=True,
                    scale_ue8m0=DEEPGEMM_SCALE_UE8M0,
                )
                return fp8_gemm_deepgemm(
                    x_fp8=a_fp8, x_scale=scale_a_col_major, y_fp8=b_fp8, y_scale=scale_b, out=out, m=M, n=N, k=K
                )
        else:

            def sglang_scaled_fp8_quant(
                input_tensor: torch.Tensor,
                scale: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                fp8_type_: torch.dtype = torch.float8_e4m3fn
                output = torch.empty_like(input_tensor, device=input_tensor.device, dtype=fp8_type_)

                if scale is None:
                    # Dynamic per-token quantization
                    scale = torch.empty((input_tensor.shape[0], 1), device=input_tensor.device, dtype=torch.float32)
                    sgl_per_token_quant_fp8(input_tensor, output, scale)
                    return output, scale

                # Static per-tensor quantization
                sgl_per_tensor_quant_fp8(input_tensor, output, scale, True)
                return output, scale

            scale_b = torch.randn((N,), device=device, dtype=torch.float32)
            # Quantize B (weights) outside
            b_fp8, scale_b_fp8 = sglang_scaled_fp8_quant(b_fp32, scale_b)
            b_fp8 = b_fp8.t()

            def gemm_op():
                # Dynamic quantization for A (per-token by default)
                scale_a = None
                # Uncomment to use static per-tensor quantization
                # scale_a = torch.ones(1, device=device, dtype=torch.float32)

                a_fp8, scale_a_fp8 = sglang_scaled_fp8_quant(a_bf16, scale=scale_a)
                return fp8_scaled_mm(a_fp8, b_fp8, scale_a_fp8, scale_b_fp8, torch.bfloat16)

    elif gemm_type == "float16":
        fp16_info = torch.finfo(torch.float16)
        fp16_max, fp16_min = fp16_info.max, fp16_info.min

        a_fp32 = (torch.rand(M, K, dtype=torch.float32, device=device) - 0.5) * 2 * fp16_max
        a_fp16 = a_fp32.clamp(min=fp16_min, max=fp16_max).to(torch.float16)

        b_fp32 = (torch.rand(N, K, dtype=torch.float32, device=device) - 0.5) * 2 * fp16_max
        b_fp16 = b_fp32.clamp(min=fp16_min, max=fp16_max).to(torch.float16)

        def gemm_op():
            return F.linear(a_fp16, b_fp16, None)

    elif gemm_type == "int8_wo":
        # Use SGLang's native int8_scaled_mm kernel for int8 weight-only
        fp16_info = torch.finfo(torch.float16)
        fp16_max, fp16_min = fp16_info.max, fp16_info.min

        # Create activation tensor (fp16)
        a_fp32 = (torch.rand(M, K, dtype=torch.float32, device=device) - 0.5) * 2 * fp16_max
        a_fp16 = a_fp32.clamp(min=fp16_min, max=fp16_max).to(torch.float16)

        # Create weight tensor (int8 with per-channel scaling)
        b_fp32 = (torch.rand(N, K, dtype=torch.float32, device=device) - 0.5) * 2 * fp16_max
        b_fp16 = b_fp32.clamp(min=fp16_min, max=fp16_max).to(torch.float16)

        # Quantize weight to int8 with per-channel (per-row) scaling
        # Note: b_int8 will be [N, K], then we transpose to [K, N] for column-major
        b_int8, scale_b = per_token_quant_int8(b_fp16)
        b_int8 = b_int8.t()  # Transpose to column-major format [K, N]

        def gemm_op():
            # Dynamically quantize activation, then run int8 GEMM
            # a_int8: [M, K], b_int8: [K, N] (column-major)
            a_int8, scale_a = per_token_quant_int8(a_fp16)
            return int8_scaled_mm(a_int8, b_int8, scale_a, scale_b, torch.bfloat16)
    else:
        raise ValueError(f"Unsupported gemm type: {gemm_type}")

    # Use benchmark_with_power context manager
    nvtx_tag = f"{gemm_type}_m{M}_n{N}_k{K}"
    torch.cuda.nvtx.range_push(nvtx_tag)

    # TODO: to study repeat_n impact, whether it can avoid L2 cache hit
    with benchmark_with_power(
        device=device,
        kernel_func=gemm_op,
        repeat_n=5,
    ) as results:
        pass

    torch.cuda.nvtx.range_pop()

    log_perf(
        item_list=[{"gemm_dtype": gemm_type, "m": M, "n": N, "k": K, "latency": results["latency_ms"]}],
        framework="SGLang",
        version=pkg_resources.get_distribution("sglang").version,
        device_name=torch.cuda.get_device_name(device),
        op_name="gemm",
        kernel_source="sglang",
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang GEMM collector.

Benchmarks SGLang/sgl-kernel matrix multiplication paths for shared GEMM case
specs, including BF16, FP8, FP8 block/DeepGEMM, and FP4-capable kernels where
available. The module owns SGLang-specific kernel selection, quantization
helpers, SM filters, and perf logging.
"""

__compat__ = "sglang==0.5.14"

import os
import random

# SGLang reads this flag while importing ``deep_gemm_wrapper.compile_utils``.
# Set it before any SGLang imports so a task compiles only its requested M.
os.environ.setdefault("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "0")

import pkg_resources
import torch
import torch.nn.functional as F
from sgl_kernel import (
    fp8_scaled_mm,
    sgl_per_token_quant_fp8,
)

from collector.case_generator import get_gemm_case_specs

try:
    from flashinfer import fp4_quantize as flashinfer_fp4_quantize
    from flashinfer import mm_fp4 as flashinfer_mm_fp4

    HAS_FLASHINFER_FP4 = True
except ImportError:
    HAS_FLASHINFER_FP4 = False

from sglang.srt.layers.deep_gemm_wrapper import (
    DEEPGEMM_SCALE_UE8M0,
    gemm_nt_f8f8bf16,
)
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

from collector.helper import benchmark_with_power, get_sm_version, log_perf


def get_gemm_test_cases():
    test_cases = []

    sm_version = get_sm_version()
    if sm_version < 89:
        gemm_list = ["bfloat16"]
    elif sm_version < 90:
        # SM89 (L40S) and earlier don't have TMA - skip fp8_block
        gemm_list = ["bfloat16", "fp8"]
    elif sm_version < 100:
        # Hopper supports fp8_block
        # fp8_block (DeepGEMM) requires SM90+ for TMA support
        gemm_list = ["fp8_block", "bfloat16", "fp8"]
    elif sm_version < 110:
        # SM100/SM103 (B100/B200 datacenter Blackwell): fp8_block + nvfp4
        gemm_list = ["fp8_block", "bfloat16", "fp8", "nvfp4"]
    else:
        # SM120+ (RTX PRO 6000 Blackwell workstation): no DeepGEMM recipe for fp8_block
        gemm_list = ["bfloat16", "fp8", "nvfp4"]

    requested_gemm_types = os.environ.get("AIC_COLLECT_GEMM_TYPES")
    if requested_gemm_types:
        requested = {item.strip() for item in requested_gemm_types.split(",") if item.strip()}
        gemm_list = [gemm_type for gemm_type in gemm_list if gemm_type in requested]

    for gemm_common_testcase in get_gemm_case_specs():
        x = gemm_common_testcase.x
        n = gemm_common_testcase.n
        k = gemm_common_testcase.k
        for gemm_type in gemm_list:
            if (gemm_type == "nvfp4" or gemm_type == "fp8_block") and (n < 128 or k < 128):
                continue
            if gemm_type == "nvfp4" and (n % 128 != 0 or k % 64 != 0):
                # The 0.5.14 FlashInfer path uses the modelopt 128x4
                # block-scale layout without synthetic padding.
                continue

            test_cases.append([gemm_type, x, n, k])

    # Try to optimize number of JIT precompile cache hits by shuffling test cases.
    random.seed(42)
    random.shuffle(test_cases)

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


def run_gemm(gemm_type, batch_size, N, K, *, perf_filename, device="cuda:0"):  # noqa: N803
    assert gemm_type in [
        "fp8_block",
        "fp8",
        "bfloat16",
        "int8_wo",
        "nvfp4",
    ], "not support gemm type"
    torch.cuda.set_device(device)
    M = batch_size  # noqa: N806
    sm_version = get_sm_version()
    fp4_backend = None
    if gemm_type == "nvfp4":
        if not HAS_FLASHINFER_FP4:
            raise RuntimeError("SGLang NVFP4 GEMM requires FlashInfer FP4 quantization support")
        if sm_version in {100, 103}:
            fp4_backend = "cute-dsl"
        elif sm_version == 120:
            fp4_backend = "cutlass"
        else:
            raise ValueError(f"SGLang NVFP4 dense GEMM is not implemented for SM{sm_version}")

    def create_gemm():
        dtype = torch.bfloat16
        if gemm_type == "nvfp4":
            # Prepare source data: Activation A [M, K] in BF16
            a_bf16 = torch.randn((M, K), device=device, dtype=dtype)

            # Prepare Weight B [N, K] and its dummy scale
            b_bf16_dummy = torch.randn((N, K), device=device, dtype=dtype)

            # Global scales
            a_global_scale = torch.tensor([1.0], device=device, dtype=torch.float32)
            b_global_scale = torch.tensor([1.0], device=device, dtype=torch.float32)
            alpha = 1.0 / (a_global_scale * b_global_scale)

            # SGLang 0.5.14 selects CuTeDSL on SM100/103 and CUTLASS on
            # SM120. Both consume the modelopt swizzled scale layout; the
            # shuffle_matrix_a/sf_a layout belongs to the TRTLLM backend.
            b_fp4, b_sf = flashinfer_fp4_quantize(
                b_bf16_dummy,
                b_global_scale,
                is_sf_swizzled_layout=True,
            )
            del b_bf16_dummy
            b_fp4 = b_fp4.t()
            b_sf = b_sf.t()
            out = torch.empty((M, N), device=device, dtype=dtype)

            def gemm_op():
                # Dynamic Quantization of Activation + GEMM
                a_fp4_dynamic, a_sf_dynamic = flashinfer_fp4_quantize(
                    a_bf16, a_global_scale, is_sf_swizzled_layout=True
                )
                return flashinfer_mm_fp4(
                    a_fp4_dynamic,
                    b_fp4,
                    a_sf_dynamic,
                    b_sf,
                    alpha,
                    dtype,
                    backend=fp4_backend,
                    out=out,
                )

            return gemm_op

        elif gemm_type == "fp8_block":
            fp8_info = torch.finfo(torch.float8_e4m3fn)
            a_bf16 = torch.randn(M, K, dtype=dtype, device=device)
            b_fp32 = (torch.rand(N, K, device=device) - 0.5) * 2 * fp8_info.max
            b_fp8 = b_fp32.clamp(min=fp8_info.min, max=fp8_info.max).to(torch.float8_e4m3fn)
            del b_fp32
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
            del b_fp32
            scale_b = torch.randn((N,), device=device, dtype=torch.float32)
            output_fp8 = torch.empty_like(a_bf16, dtype=torch.float8_e4m3fn)
            scale_a = torch.empty((M, 1), device=device, dtype=torch.float32)

            def gemm_op():
                sgl_per_token_quant_fp8(a_bf16, output_fp8, scale_a)
                return fp8_scaled_mm(output_fp8, b_fp8, scale_a, scale_b, dtype)

            return gemm_op

        elif gemm_type == "bfloat16":
            a_bfloat16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
            b_bfloat16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)

            def gemm_op():
                return F.linear(a_bfloat16, b_bfloat16, None)

            return gemm_op

    op_list = []
    try:
        # Scale loop count down for large matrices to avoid OOM. FP8 setup briefly
        # materializes FP32 tensors before retaining quantized weights, so account
        # for both persistent and transient pressure instead of only steady state.
        _gpu_mem = torch.cuda.get_device_properties(device).total_memory
        _persistent_bytes = M * K * 2 + M * N * 2
        _transient_bytes = 0
        if gemm_type == "bfloat16":
            _persistent_bytes += N * K * 2
        elif gemm_type == "fp8":
            _persistent_bytes += N * K + M * K + N * 4 + M * 4
            _transient_bytes = N * K * 4
        elif gemm_type == "fp8_block":
            _persistent_bytes += N * K + cdiv(N, 128) * cdiv(K, 128) * 4
            _transient_bytes = N * K * 4
        elif gemm_type == "nvfp4":
            _persistent_bytes += int(N * K * 0.75)
            _transient_bytes = N * K * 2

        _budget = int(_gpu_mem * 0.45)
        if _persistent_bytes + _transient_bytes >= _budget:
            outside_loop_count = 1
        else:
            outside_loop_count = max(1, min(6, (_budget - _transient_bytes) // max(_persistent_bytes, 1)))

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

        try:
            with benchmark_with_power(
                device=device,
                kernel_func=kernel_func,
                num_warmups=3,
                num_runs=6,
                repeat_n=1,
            ) as results:
                pass
        finally:
            torch.cuda.nvtx.range_pop()

        kernel_source = "sglang"
        if gemm_type == "nvfp4":
            kernel_source = (
                "sglang_flashinfer_cutedsl_nvfp4" if fp4_backend == "cute-dsl" else "sglang_flashinfer_cutlass_nvfp4"
            )
        log_perf(
            item_list=[
                {"gemm_dtype": gemm_type, "m": M, "n": N, "k": K, "latency": results["latency_ms"] / len(op_list)}
            ],
            framework="SGLang",
            version=pkg_resources.get_distribution("sglang").version,
            device_name=torch.cuda.get_device_name(device),
            op_name="gemm",
            kernel_source=kernel_source,
            perf_filename=perf_filename,
            power_stats=results["power_stats"],
        )
    finally:
        op_list.clear()
        torch.cuda.empty_cache()

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import deep_gemm
import pkg_resources
import torch
import torch.nn.functional as F
from deep_gemm import get_col_major_tma_aligned_tensor
from sgl_kernel import fp8_scaled_mm, int8_scaled_mm, sgl_per_tensor_quant_fp8

from helper import log_perf


def get_gemm_test_cases():
    x_list = [
        1,
        2,
        4,
        8,
        16,
        32,
        48,
        64,
        80,
        96,
        128,
        160,
        192,
        256,
        384,
        512,
        768,
        1024,
        2048,
        4096,
        8192,
    ]
    nk_list = [
        64,
        128,
        256,
        512,
        768,
        1024,
        1536,
        2048,
        2560,
        3072,
        3584,
        4096,
        5120,
        6144,
        7168,
        8192,
        10240,
        12288,
    ]
    nk_list_ext = [16384, 65536]  # for coverage and interp purpose
    gemm_list = ["int8_wo", "int4_wo", "fp8_block", "float16", "fp8"]

    test_cases = []
    for gemm_type in gemm_list:
        # x_list_orig+add+ext  <==> nk_list+ext
        for x in sorted(x_list, reverse=True):
            for n in sorted(nk_list + nk_list_ext, reverse=True):
                for k in sorted(nk_list + nk_list_ext, reverse=True):
                    if n * k == 65536 * 65536:
                        continue
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
    m: int,
    n: int,
    k: int,
):
    """DeepGEMM implementation of FP8 GEMM"""
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    # Run DeepGEMM kernel
    deep_gemm.gemm_fp8_fp8_bf16_nt((x_fp8, x_scale), (y_fp8, y_scale), out)
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
        "int4_wo",
    ], "not support gemm type"
    torch.cuda.set_device(device)
    M = batch_size  # noqa: N806

    if gemm_type == "fp8_block" or gemm_type == "fp8":
        fp8_info = torch.finfo(torch.float8_e4m3fn)
        fp8_max, fp8_min = fp8_info.max, fp8_info.min

        a_fp32 = (torch.rand(M, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_max
        b_fp32 = (torch.rand(N, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_max

        if gemm_type == "fp8_block":
            a_fp8 = a_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
            b_fp8 = b_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

            scale_a_group_shape = (1, 128)
            scale_b_group_shape = (128, 128)
            scale_a_shape = scale_shape(a_fp8.shape, scale_a_group_shape)
            scale_b_shape = scale_shape(b_fp8.shape, scale_b_group_shape)

            scale_a = torch.randn(scale_a_shape, device="cuda", dtype=torch.float32)
            scale_b = torch.randn(scale_b_shape, device="cuda", dtype=torch.float32)

            scale_a_col_major = get_col_major_tma_aligned_tensor(scale_a.clone())

            repeat_n = 5

            def gemm_op():
                return fp8_gemm_deepgemm(a_fp8, scale_a_col_major, b_fp8, scale_b, M, N, K)
        else:

            def sglang_scaled_fp8_quant(
                input_tensor: torch.Tensor,
                scale: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                fp8_type_: torch.dtype = torch.float8_e4m3fn
                output = torch.empty_like(input_tensor, device=input_tensor.device, dtype=fp8_type_)
                is_static = True
                if scale is None:
                    scale = torch.zeros(1, device=input_tensor.device, dtype=torch.float32)
                    is_static = False
                sgl_per_tensor_quant_fp8(input_tensor, output, scale, is_static)

                return output, scale

            scale_a = torch.randn((M,), device="cuda", dtype=torch.float32)
            scale_b = torch.randn((N,), device="cuda", dtype=torch.float32)
            a_fp8, scale_a_fp8 = sglang_scaled_fp8_quant(a_fp32, scale_a)
            b_fp8, scale_b_fp8 = sglang_scaled_fp8_quant(b_fp32, scale_b)
            b_fp8 = b_fp8.t()

            repeat_n = 5

            def gemm_op():
                return fp8_scaled_mm(a_fp8, b_fp8, scale_a_fp8, scale_b_fp8, torch.bfloat16)

    elif gemm_type == "float16":
        fp16_info = torch.finfo(torch.float16)
        fp16_max, fp16_min = fp16_info.max, fp16_info.min

        a_fp32 = (torch.rand(M, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp16_max
        a_fp16 = a_fp32.clamp(min=fp16_min, max=fp16_max).to(torch.float16)

        b_fp32 = (torch.rand(N, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp16_max
        b_fp16 = b_fp32.clamp(min=fp16_min, max=fp16_max).to(torch.float16)

        repeat_n = 5

        def gemm_op():
            return F.linear(a_fp16, b_fp16, None)

    elif gemm_type == "int8_wo":
        # Use SGLang's native int8_scaled_mm kernel for int8 weight-only
        fp16_info = torch.finfo(torch.float16)
        fp16_max, fp16_min = fp16_info.max, fp16_info.min

        # Create activation tensor (fp16)
        a_fp32 = (torch.rand(M, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp16_max
        a_fp16 = a_fp32.clamp(min=fp16_min, max=fp16_max).to(torch.float16)

        # Create weight tensor (int8 with per-channel scaling)
        b_fp32 = (torch.rand(N, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp16_max
        b_fp16 = b_fp32.clamp(min=fp16_min, max=fp16_max).to(torch.float16)

        # Quantize weight to int8 with per-channel (per-row) scaling
        # Note: b_int8 will be [N, K], then we transpose to [K, N] for column-major
        b_int8, scale_b = per_token_quant_int8(b_fp16)
        b_int8 = b_int8.t()  # Transpose to column-major format [K, N]

        repeat_n = 5

        def gemm_op():
            # Dynamically quantize activation, then run int8 GEMM
            # a_int8: [M, K], b_int8: [K, N] (column-major)
            a_int8, scale_a = per_token_quant_int8(a_fp16)
            return int8_scaled_mm(a_int8, b_int8, scale_a, scale_b, torch.bfloat16)

    else:
        # int4_wo: Use torchao since SGLang doesn't have native int4 kernel
        from torchao.quantization import (
            int4_weight_only,
            quantize_,
        )

        # Define fp16 range for tensor initialization
        fp16_info = torch.finfo(torch.float16)
        fp16_max, fp16_min = fp16_info.max, fp16_info.min

        # Use torch.nn.Linear with bfloat16 dtype (required by TorchAO int4)
        linear = torch.nn.Linear(
            in_features=K,
            out_features=N,
            bias=False,
            device="cuda",
            dtype=torch.bfloat16,
        )
        quantize_(linear, int4_weight_only(group_size=128), filter_fn=None)

        a_fp32 = (torch.rand(M, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp16_max
        a_bf16 = a_fp32.clamp(min=fp16_min, max=fp16_max).to(torch.bfloat16)

        repeat_n = 5

        def gemm_op():
            return linear(a_bf16)

    num_warmups = 3
    num_runs = 6

    # Warmup outside of graph capture
    torch.cuda.synchronize()
    for _ in range(num_warmups):
        gemm_op()
    torch.cuda.synchronize()

    # Capture the graph with repeated operations
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(repeat_n):
            gemm_op()
    torch.cuda.synchronize()

    # Time the graph replay
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(num_runs):
        g.replay()
    end_event.record()
    torch.cuda.synchronize()
    latency = start_event.elapsed_time(end_event) / num_runs / repeat_n

    log_perf(
        item_list=[{"gemm_dtype": gemm_type, "m": M, "n": N, "k": K, "latency": latency}],
        framework="SGLang",
        version=pkg_resources.get_distribution("sglang").version,
        device_name=torch.cuda.get_device_name(device),
        op_name="gemm",
        kernel_source="sglang",
        perf_filename=perf_filename,
    )

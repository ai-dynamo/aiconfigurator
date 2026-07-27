# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure SGLang FP8 activation quantization overhead for static-FP8 GEMM."""

__compat__ = "sglang==0.5.14"

import pkg_resources
import torch
from sgl_kernel import sgl_per_token_quant_fp8

from collector.case_generator import get_compute_scale_case_specs
from collector.helper import benchmark_with_power, get_sm_version, log_perf


def get_computescale_test_cases():
    if get_sm_version() <= 86:
        return []
    return [[case.m, case.k] for case in get_compute_scale_case_specs()]


def _static_quantize_e4m3_per_tensor(x: torch.Tensor, scale: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    out.copy_((x / scale).clamp(min=fp8_info.min, max=fp8_info.max).to(torch.float8_e4m3fn))
    return out


def run_computescale(m, k, *, perf_filename, device="cuda:0"):
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    x = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    dynamic_out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    dynamic_scale = torch.empty((m, 1), dtype=torch.float32, device=device)
    static_out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    static_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    outside_loop_count = 5

    def dynamic_kernel_func():
        for _ in range(outside_loop_count):
            sgl_per_token_quant_fp8(x, dynamic_out, dynamic_scale)

    with benchmark_with_power(device=device, kernel_func=dynamic_kernel_func, repeat_n=1) as dynamic_results:
        pass

    dynamic_latency = dynamic_results["latency_ms"] / outside_loop_count

    def static_kernel_func():
        for _ in range(outside_loop_count):
            _static_quantize_e4m3_per_tensor(x, static_scale, static_out)

    with benchmark_with_power(device=device, kernel_func=static_kernel_func, repeat_n=1) as static_results:
        pass

    static_latency = static_results["latency_ms"] / outside_loop_count
    compute_scale_latency = max(0.0, dynamic_latency - static_latency)
    version = pkg_resources.get_distribution("sglang").version

    if not log_perf(
        item_list=[{"m": m, "k": k, "quant_dtype": "fp8", "latency": compute_scale_latency}],
        framework="SGLang",
        version=version,
        device_name=torch.cuda.get_device_name(device),
        op_name="compute_scale",
        kernel_source="sglang",
        perf_filename=perf_filename,
        power_stats=dynamic_results["power_stats"],
    ):
        raise RuntimeError(f"Failed to persist SGLang compute scale performance row to {perf_filename}")
    if not log_perf(
        item_list=[{"m": m, "k": k, "quant_dtype": "fp8", "latency": static_latency}],
        framework="SGLang",
        version=version,
        device_name=torch.cuda.get_device_name(device),
        op_name="scale_matrix",
        kernel_source="sglang",
        perf_filename="scale_matrix_perf.txt",
        power_stats=static_results["power_stats"],
    ):
        raise RuntimeError("Failed to persist SGLang scale matrix performance row to scale_matrix_perf.txt")

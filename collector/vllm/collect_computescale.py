# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure vLLM FP8 activation quantization overhead for static-FP8 GEMM."""

__compat__ = "vllm>=0.14.0"

import torch
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.version import __version__ as vllm_version

from collector.case_generator import get_compute_scale_case_specs
from collector.helper import benchmark_with_power, delta_latency_power_stats, get_sm_version, log_perf
from collector.vllm.utils import setup_distributed


def get_computescale_test_cases():
    if get_sm_version() <= 86:
        return []
    return [[case.m, case.k] for case in get_compute_scale_case_specs()]


def run_computescale(m, k, *, perf_filename, device="cuda:0"):
    setup_distributed(device)
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    x = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    static_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    outside_loop_count = 5

    def dynamic_kernel_func():
        for _ in range(outside_loop_count):
            ops.scaled_fp8_quant(x, scale=None, use_per_token_if_dynamic=True)

    with benchmark_with_power(device=device, kernel_func=dynamic_kernel_func, repeat_n=1) as dynamic_results:
        pass

    dynamic_latency = dynamic_results["latency_ms"] / outside_loop_count

    def static_kernel_func():
        for _ in range(outside_loop_count):
            ops.scaled_fp8_quant(
                x,
                scale=static_scale,
                group_shape=(GroupShape.PER_TENSOR.row, GroupShape.PER_TENSOR.col),
            )

    with benchmark_with_power(device=device, kernel_func=static_kernel_func, repeat_n=1) as static_results:
        pass

    static_latency = static_results["latency_ms"] / outside_loop_count
    compute_scale_latency, compute_scale_power_stats = delta_latency_power_stats(
        dynamic_latency,
        static_latency,
        dynamic_results["power_stats"],
        static_results["power_stats"],
    )

    log_perf(
        item_list=[{"m": m, "k": k, "quant_dtype": "fp8", "latency": compute_scale_latency}],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(device),
        op_name="compute_scale",
        kernel_source="vllm_default",
        perf_filename=perf_filename,
        power_stats=compute_scale_power_stats,
    )
    log_perf(
        item_list=[{"m": m, "k": k, "quant_dtype": "fp8", "latency": static_latency}],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(device),
        op_name="scale_matrix",
        kernel_source="vllm_default",
        perf_filename="scale_matrix_perf.txt",
        power_stats=static_results["power_stats"],
    )

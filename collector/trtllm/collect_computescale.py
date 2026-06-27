# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure TensorRT-LLM FP8 compute-scale overhead.

The compute-scale collector compares dynamic FP8 quantization against static
quantization for each YAML-backed shape. The reported latency isolates the
scale-computation portion so support matrix data can track the cost separately
from the static quantize kernel.
"""

import tensorrt_llm
import torch
from case_generator import get_compute_scale_case_specs

from helper import benchmark_with_power, delta_latency_power_stats, get_sm_version, log_perf


def get_computescale_test_cases():
    # compute_scale only applies to fp8 quantization (SM > 86)
    if get_sm_version() <= 86:
        return []

    test_cases = []
    for compute_scale_common_testcase in get_compute_scale_case_specs():
        test_cases.append([compute_scale_common_testcase.m, compute_scale_common_testcase.k])

    return test_cases


def run_computescale(m, k, *, perf_filename, device="cuda:0"):
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    dtype = torch.bfloat16
    x = torch.randn((m, k), dtype=dtype).to(torch.device(device))
    scale = torch.tensor([1.0], dtype=torch.float32, device=device)

    outside_loop_count = 5  # to reduce impact of L2 cache hit

    # Build op lists for both dynamic and static quantization
    dynamic_op_list = []
    static_op_list = []
    for _ in range(outside_loop_count):
        dynamic_op_list.append(torch.ops.tensorrt_llm.quantize_e4m3_per_tensor)
        static_op_list.append(torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor)

    # Benchmark dynamic quantization (compute scale + scale matrix)
    def dynamic_kernel_func():
        for op in dynamic_op_list:
            op(x)

    with benchmark_with_power(
        device=device,
        kernel_func=dynamic_kernel_func,
        repeat_n=1,
    ) as dynamic_results:
        pass

    dynamic_latency = dynamic_results["latency_ms"] / outside_loop_count

    # Benchmark static quantization (scale matrix only)
    def static_kernel_func():
        for op in static_op_list:
            op(x, scale)

    with benchmark_with_power(
        device=device,
        kernel_func=static_kernel_func,
        repeat_n=1,
    ) as static_results:
        pass

    static_latency = static_results["latency_ms"] / outside_loop_count

    # Encode the incremental energy (dynamic - static), not dynamic power
    # multiplied by the latency delta.
    compute_scale_latency, compute_scale_power_stats = delta_latency_power_stats(
        dynamic_latency,
        static_latency,
        dynamic_results["power_stats"],
        static_results["power_stats"],
    )

    # Log compute_scale performance
    log_perf(
        item_list=[
            {
                "m": m,
                "k": k,
                "quant_dtype": "fp8",
                "latency": compute_scale_latency,
            }
        ],
        framework="TRTLLM",
        version=tensorrt_llm.__version__,
        device_name=torch.cuda.get_device_name(device),
        op_name="compute_scale",
        kernel_source="torch_ops",
        perf_filename=perf_filename,
        power_stats=compute_scale_power_stats,
    )

    # Log scale_matrix performance
    log_perf(
        item_list=[
            {
                "m": m,
                "k": k,
                "quant_dtype": "fp8",
                "latency": static_latency,
            }
        ],
        framework="TRTLLM",
        version=tensorrt_llm.__version__,
        device_name=torch.cuda.get_device_name(device),
        op_name="scale_matrix",
        kernel_source="torch_ops",
        perf_filename="scale_matrix_perf.txt",
        power_stats=static_results["power_stats"],
    )

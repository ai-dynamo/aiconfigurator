# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import tensorrt_llm
import torch
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

from helper import (
    get_dtype_size,
    get_gpu_specs_from_device,
    get_sm_version,
    log_perf,
    measure_kernel_power,
)


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
        32,
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
    gemm_list = ["float16"]
    if get_sm_version() > 86:
        gemm_list += ["fp8"]
        if get_sm_version() < 100:
            gemm_list += ["fp8_block"]
    if get_sm_version() >= 100:
        gemm_list += ["nvfp4"]

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


def is_gemm_compute_bound(m, n, k, dtype, device_name):
    """
    Determine if a GEMM operation is compute-bound.

    Args:
        m, n, k: GEMM dimensions (C = A @ B, A is mxk, B is kxn)
        dtype: Data type (e.g., 'float16', 'fp8')
        device_name: GPU device name

    Returns:
        True if compute-bound, False if memory-bound
    """
    gpu_specs = get_gpu_specs_from_device(device_name)
    dtype_size = get_dtype_size(dtype)

    # Hardware intensity (FLOPs per byte)
    if "fp8" in dtype.lower():
        hardware_tflops = gpu_specs["fp8_tflops"]
    else:
        hardware_tflops = gpu_specs["float16_tflops"]

    hardware_intensity = (hardware_tflops * 1e12) / (gpu_specs["mem_bw_gbs"] * 1e9)

    # GEMM arithmetic intensity
    total_flops = 2 * m * n * k
    memory_bytes = dtype_size * (m * k + k * n + m * n)
    arithmetic_intensity = total_flops / memory_bytes

    # Compute-bound if arithmetic intensity > hardware intensity
    return arithmetic_intensity > hardware_intensity


def run_gemm(
    gemm_type,
    m,
    n,
    k,
    perf_filename,
    device="cuda:0",
    power_monitor=None,
    power_limit=None,
    measure_power=False,
    kernel_power_measurement_duration=3.0,
):
    """
    Run GEMM benchmark with optional power measurement.

    Args:
        gemm_type: GEMM quantization type
        m, n, k: Matrix dimensions
        perf_filename: Output CSV filename
        device: CUDA device
        power_monitor: NVMLPowerMonitor instance (optional)
        power_limit: GPU power limit in Watts (optional)
        measure_power: Whether to measure power consumption
        kernel_power_measurement_duration: Target duration for memory-bound benchmarks (seconds)
    """
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    device_name = torch.cuda.get_device_name(device)
    dtype = torch.bfloat16
    x = torch.randn((m, k), dtype=dtype).to(torch.device(device))

    if gemm_type == "fp8":
        qc = QuantConfig(quant_algo=QuantAlgo.FP8)
    elif gemm_type == "fp8_block":
        group_size = 128
        qc = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=group_size)
    elif gemm_type == "nvfp4":
        group_size = 128
        qc = QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=group_size)
    else:
        qc = None

    repeat_n = 5  # to reduce impact of L2 cache hit
    op_list = []
    for i in range(repeat_n):
        gemm = Linear(
            k,
            n,
            bias=False,
            dtype=dtype,
            quant_config=qc,
        )

        if gemm_type == "fp8":
            weights = {
                "weight": torch.randn((n, k), dtype=torch.bfloat16, device=torch.device(device)).to(
                    dtype=torch.float8_e4m3fn
                ),
                "weight_scale": torch.randn(1, dtype=torch.float32, device=torch.device(device)),
            }
        elif gemm_type == "fp8_block":
            weights = {
                "weight": torch.randn((n, k), dtype=torch.bfloat16, device=torch.device(device)).to(
                    dtype=torch.float8_e4m3fn
                ),
                "weight_scale": torch.randn(
                    (math.ceil(n / group_size), math.ceil(k / group_size)),
                    dtype=torch.float32,
                    device=torch.device(device),
                ),
            }
        elif gemm_type == "nvfp4":
            # From trtllm test case
            x_sf_global = (448 * 6) / x.abs().max().float()
            w = torch.randn((n, k), dtype=torch.float16, device=torch.device(device))
            w_sf_global = (448 * 6) / w.abs().max().float()
            w_fp4, w_sf_block = torch.ops.trtllm.fp4_quantize(w, w_sf_global, 16, False)
            w_sf_block_unswizzled = torch.ops.trtllm.nvfp4_block_scale_interleave_reverse(w_sf_block.cpu().view(k, -1))
            weights = {
                "weight": w_fp4.cpu(),
                "weight_scale": w_sf_block_unswizzled.view(torch.float8_e4m3fn),
                "weight_scale_2": 1.0 / w_sf_global.cpu(),
                "input_scale": 1.0 / x_sf_global.cpu(),
            }
        else:
            weights = {"weight": torch.randn((n, k), dtype=torch.bfloat16, device=torch.device(device))}

        gemm.load_weights([weights])
        gemm.to(torch.device(device))
        gemm.forward(x)  # dry run to init
        op_list.append(gemm)

    num_warmups = 3
    num_runs = 6

    # capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for op in op_list:
            op.forward(x)

    # Determine if compute-bound
    compute_bound = is_gemm_compute_bound(m, n, k, gemm_type, device_name)

    # Benchmarking
    if measure_power and power_monitor is not None and not compute_bound:
        latency, power = measure_kernel_power(power_monitor, g.replay, num_warmups, kernel_power_measurement_duration)
        latency /= len(op_list)
    else:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        for _ in range(num_warmups):
            g.replay()
        torch.cuda.synchronize()

        start_event.record()
        for _ in range(num_runs):
            g.replay()
        end_event.record()
        torch.cuda.synchronize()

        latency = start_event.elapsed_time(end_event) / num_runs / len(op_list)
        power = power_limit or None

    # Build result item
    item = {
        "gemm_dtype": gemm_type,
        "m": m,
        "n": n,
        "k": k,
        "latency": latency,
    }

    if power is not None:
        item["power_limit"] = power_limit
        item["power"] = power
        item["compute_bound"] = int(compute_bound)

    log_perf(
        item_list=[item],
        framework="TRTLLM",
        version=tensorrt_llm.__version__,
        device_name=device_name,
        op_name="gemm",
        kernel_source="torch_flow",
        perf_filename=perf_filename,
    )

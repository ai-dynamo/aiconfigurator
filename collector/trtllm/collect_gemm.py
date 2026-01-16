# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import tensorrt_llm
import torch
from common_test_cases import get_gemm_common_test_cases
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

from helper import benchmark_with_power, get_sm_version, log_perf


def pad_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def get_gemm_test_cases():
    gemm_list = ["float16"]
    if get_sm_version() > 86:
        gemm_list += ["fp8"]
        if get_sm_version() < 100:
            gemm_list += ["fp8_block"]
    if get_sm_version() >= 100:
        gemm_list += ["nvfp4"]

    test_cases = []
    for gemm_common_testcase in get_gemm_common_test_cases():
        x = gemm_common_testcase.x
        n = gemm_common_testcase.n
        k = gemm_common_testcase.k
        for gemm_type in gemm_list:
            if (gemm_type == "nvfp4" or gemm_type == "fp8_block") and (n < 128 or k < 128):
                continue
            test_cases.append([gemm_type, x, n, k, "gemm_perf.txt"])

    return test_cases


def run_gemm(gemm_type, m, n, k, perf_filename, device="cuda:0"):
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

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

    outside_loop_count = 5  # to reduce impact of L2 cache hit
    op_list = []
    for i in range(outside_loop_count):
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
            if tensorrt_llm.__version__.startswith(("1.1.0", "1.2.0")):
                w_sf_block_unswizzled = torch.ops.trtllm.block_scale_interleave_reverse(
                    w_sf_block.cpu().view(pad_up(n, 128), -1)
                )
            else:
                w_sf_block_unswizzled = torch.ops.trtllm.nvfp4_block_scale_interleave_reverse(
                    w_sf_block.cpu().view(k, -1)
                )
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

    # Use benchmark_with_power context manager
    def kernel_func():
        for op in op_list:
            op.forward(x)

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        repeat_n=1,  # Already repeating inside kernel_func via op_list
    ) as results:
        pass

    log_perf(
        item_list=[
            {"gemm_dtype": gemm_type, "m": m, "n": n, "k": k, "latency": results["latency_ms"] / outside_loop_count}
        ],
        framework="TRTLLM",
        version=tensorrt_llm.__version__,
        device_name=torch.cuda.get_device_name(device),
        op_name="gemm",
        kernel_source="torch_flow",
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )

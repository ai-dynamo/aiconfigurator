# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os

import torch
from common_test_cases import get_gemm_common_test_cases
from vllm.distributed import (
    init_distributed_environment,
)
from vllm.distributed.parallel_state import ensure_model_parallel_initialized
from vllm.model_executor.layers.linear import (
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.version import __version__ as vllm_version

from helper import benchmark_with_power, get_sm_version, log_perf

compatible_version = ["0.11.0", "0.12.0"]


@functools.cache  # only run once per process
def setup_distributed(device):
    # Each process needs to use a different port.
    device_idx = torch.device(device).index
    port = 8889 + device_idx
    print(device, device_idx, port)

    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_distributed_environment()
    ensure_model_parallel_initialized(1, 1)


def get_gemm_test_cases():
    gemm_list = ["float16"]
    if get_sm_version() > 86:
        gemm_list += ["fp8"]
        # gemm_list += ["fp8_block"] # TODO: broken

    # if get_sm_version() >= 100:
    #     gemm_list += ["nvfp4"]

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
    setup_distributed(device)

    torch.set_default_dtype(torch.float16)
    torch.cuda.set_device(device)

    dtype = torch.float16
    x = torch.randn((m, k), dtype=dtype, device=torch.device(device))

    if gemm_type == "fp8":
        qc = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="static",
            ignored_layers=None,
            weight_block_size=None,
        )
    elif gemm_type == "fp8_block":
        qc = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
    else:
        qc = None

    gemm = RowParallelLinear(
        input_size=k,
        output_size=n,
        bias=False,
        skip_bias_add=True,
        params_dtype=dtype,
        quant_config=qc,
        prefix="",
        return_bias=True,
        disable_tp=True,
    )
    # TODO, to evaluate random weights impact
    gemm.to(torch.device(device))

    if gemm_type == "fp8" and hasattr(gemm, "weight"):
        new_weight = gemm.weight.data.t()
        # print("new_weight stride:", new_weight.stride())
        # mnk = 1,128,128   weight stride = (128,1)  - transpose to (1,128) for fp8 cutlass limit
        gemm.weight = torch.nn.Parameter(new_weight)
        # print("after fix, weight stride:", gemm.weight.data.stride())

    gemm.forward(x)  # dry run to init

    # Use benchmark_with_power context manager
    def kernel_func():
        for _ in range(6):
            gemm.forward(x)

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=3,
        num_runs=6,
        repeat_n=1,
    ) as results:
        pass

    log_perf(
        item_list=[{"gemm_dtype": gemm_type, "m": m, "n": n, "k": k, "latency": results["latency_ms"]}],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(device),
        op_name="gemm",
        kernel_source="vllm_default",
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


if __name__ == "__main__":
    test_cases = get_gemm_test_cases()
    test_cases = test_cases[:10]
    for tc in test_cases:
        print(f"Running test case: {tc}")
        run_gemm(*tc)

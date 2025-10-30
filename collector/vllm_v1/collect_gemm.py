# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os

import torch
from vllm.distributed import (
    init_distributed_environment,
)
from vllm.distributed.parallel_state import ensure_model_parallel_initialized
from vllm.model_executor.layers.linear import (
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.version import __version__ as vllm_version

from helper import get_sm_version, log_perf


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


def get_gemm_test_cases(is_unit_test=False):
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
        128,
        256,
        512,
        1024,
        1536,
        2048,
        2560,
        3072,
        3584,
        4096,
        5120,
        7168,
        8192,
        10240,
        12288,
    ]
    nk_list_ext = [16384, 65536]  # for coverage and interp purpose
    # gemm_list = ['float16']
    gemm_list = ["awq", "gptq"]
    if get_sm_version() < 86:
        gemm_list += ["fp8", "fp8_block", "awq", "gptq"]

    if is_unit_test:
        x_list = [1, 2, 4, 8]
        nk_list = [128]
        nk_list_ext = []
        gemm_list = ["float16"]

    test_cases = []

    for gemm_type in gemm_list:
        # x_list_orig+add+ext  <==> nk_list+ext
        for x in sorted(x_list, reverse=True):
            for n in sorted(nk_list + nk_list_ext, reverse=True):
                for k in sorted(nk_list + nk_list_ext, reverse=True):
                    if n * k == 65536 * 65536:
                        continue
                    test_cases.append([gemm_type, x, n, k, "gemm_perf_vllm.txt"])
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
    elif gemm_type == "awq":
        qc = AWQConfig(weight_bits=4, group_size=128, zero_point=True, modules_to_not_convert=None)
    elif gemm_type == "gptq":
        qc = GPTQConfig(weight_bits=8, group_size=128, desc_act=False, lm_head_quantized=False, dynamic={})
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

    num_warmups = 3
    num_runs = 6

    # capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(num_runs):
            gemm.forward(x)
    # warmup
    for i in range(num_warmups):
        g.replay()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(num_runs):
        g.replay()
    end_event.record()
    torch.cuda.synchronize()
    latency = start_event.elapsed_time(end_event) / (num_runs * num_runs)

    log_perf(
        item_list=[{"gemm_dtype": gemm_type, "m": m, "n": n, "k": k, "latency": latency}],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(device),
        op_name="gemm",
        kernel_source="vllm_default",
        perf_filename=perf_filename,
    )


if __name__ == "__main__":
    test_cases = get_gemm_test_cases()
    test_cases = test_cases[:10]
    for tc in test_cases:
        print(f"Running test case: {tc}")
        run_gemm(*tc)

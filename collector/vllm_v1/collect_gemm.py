# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import torch
from vllm.distributed import (
    init_distributed_environment,
)
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
)
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.version import __version__ as vllm_version

from helper import get_sm_version, log_perf


# If we want to use advanced linear implementations like MergedColumnParallelLinear and
# RowParallelLinear, we need to unit and destroy TP and rank group before and after each test case.
def setup_distributed():
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8889"
    init_distributed_environment()


def destroy_distributed():
    torch.distributed.destroy_process_group()


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
    torch.set_default_dtype(torch.float16)
    torch.cuda.set_device(device)
    dtype = torch.float16
    x = torch.randn((m, k), dtype=dtype).to(torch.device(device))

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
    # print(f"dtype: {dtype}, type: {type(dtype)}")
    # print(f"qc: {qc}, type: {type(qc)}")
    gemm = ReplicatedLinear(
        input_size=k,
        output_size=n,
        bias=False,
        skip_bias_add=True,
        params_dtype=dtype,
        quant_config=qc,
        prefix="",
        return_bias=True,
    )
    # TODO, to evaluate random weights impact
    gemm.to(torch.device(device))
    # print(dir(gemm)) # print all attributes of gemm
    # print(gemm.weight.data.stride())

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

    prefix = f"VLLM,{vllm_version},{torch.cuda.get_device_name(device)},gemm_torch,{gemm_type},{m},{n},{k}"

    fd = os.open(perf_filename, os.O_APPEND | os.O_WRONLY | os.O_CREAT)
    content = prefix + f",gemm_vllm,{latency}\n"
    os.write(fd, content.encode())
    os.close(fd)

    log_perf(
        item_list=[{"gemm_dtype": gemm_type, "m": m, "n": n, "k": k, "latency": latency}],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(device),
        op_name="gemm",
        kernel_source="vllm_default",
        perf_filename=perf_filename,
    )

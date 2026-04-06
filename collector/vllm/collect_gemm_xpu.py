# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__compat__ = "vllm>=0.11.0"

import os

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

try:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        maybe_post_process_fp8_weight_block,
    )
except Exception:
    print("No maybe_post_process_fp8_weight_block found, please check your vLLM version.")
from vllm.utils.deep_gemm import per_block_cast_to_fp8
from vllm.version import __version__ as vllm_version

from collector.common_test_cases import GemmCommonTestCase
from collector.helper import benchmark_with_power, get_device_module, log_perf
from collector.vllm.utils import setup_distributed, with_exit_stack

FP8_BLOCK_SHAPE = (128, 128)


def get_gemm_xpu_test_cases() -> list[GemmCommonTestCase]:
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
    # narrow down the search space for xpu currently
    nk_list_ext = []

    test_cases = []
    # x_list_orig+add+ext  <==> nk_list+ext
    for x in sorted(x_list, reverse=True):
        for n in sorted(nk_list + nk_list_ext, reverse=True):
            for k in sorted(nk_list + nk_list_ext, reverse=True):
                if n * k == 65536 * 65536:
                    continue
                test_cases.append(GemmCommonTestCase(x=x, n=n, k=k))

    return test_cases


def get_gemm_test_cases():
    gemm_list = ["float16", "fp8"]

    test_cases = []

    for gemm_common_testcase in get_gemm_xpu_test_cases():
        x = gemm_common_testcase.x
        n = gemm_common_testcase.n
        k = gemm_common_testcase.k
        for gemm_type in gemm_list:
            test_cases.append([gemm_type, x, n, k, "gemm_perf.txt"])

    return test_cases


@with_exit_stack
def run_gemm(exit_stack, gemm_type, m, n, k, perf_filename, device="xpu:0"):
    # Force DeepGEMM path when available to capture the intended kernel.
    os.environ["VLLM_USE_DEEP_GEMM"] = "1"

    setup_distributed(device)

    dtype = torch.float16
    torch.set_default_dtype(dtype)
    get_device_module().set_device(device)

    x = torch.randn((m, k), dtype=dtype, device=torch.device(device))

    if gemm_type == "fp8":
        qc = Fp8Config(
            is_checkpoint_fp8_serialized=False,  # dynamic quant after creation
            activation_scheme="dynamic",
            ignored_layers=None,
            weight_block_size=None,
        )
    elif gemm_type == "fp8_block":
        qc = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=list(FP8_BLOCK_SHAPE),
        )
    else:
        qc = None

    def create_gemm():
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
        # vLLM >=0.16 creates quantized layers on meta device;
        # use to_empty() then fill with random data.
        try:
            gemm.to(torch.device(device))
        except NotImplementedError:
            gemm = gemm.to_empty(device=torch.device(device))
            with torch.no_grad():
                for param in gemm.parameters():
                    if param.dtype.is_floating_point:
                        param.normal_()
                    else:
                        param.zero_()

        if gemm_type == "fp8" and hasattr(gemm, "weight"):
            # Use process_weights_after_loading() to quantize the weights after creation
            if hasattr(gemm, "quant_method") and gemm.quant_method is not None:
                quant_method = gemm.quant_method
                if hasattr(quant_method, "process_weights_after_loading"):
                    quant_method.process_weights_after_loading(gemm)
        elif gemm_type == "fp8_block":
            block_n, block_k = FP8_BLOCK_SHAPE
            with torch.no_grad():
                # Blockwise quantize a random weight to provide valid scales.
                raw_weight = torch.randn((n, k), dtype=torch.float32, device=device)
                q_weight, weight_scale = per_block_cast_to_fp8(raw_weight, [block_n, block_k], use_ue8m0=False)
                if hasattr(gemm, "weight"):
                    gemm.weight.copy_(q_weight)
                if hasattr(gemm, "weight_scale_inv"):
                    gemm.weight_scale_inv.copy_(weight_scale.contiguous().to(torch.float32))
                    # Some versions expect `weight_scale` even for block quant.
                    if not hasattr(gemm, "weight_scale"):
                        gemm.weight_scale = gemm.weight_scale_inv

                # Support both old (layer-only) and new (layer, cutlass_supported)
                # signatures for maybe_post_process_fp8_weight_block.
                try:
                    maybe_post_process_fp8_weight_block(gemm)
                except TypeError:
                    maybe_post_process_fp8_weight_block(gemm, cutlass_block_fp8_supported=True)

        gemm.forward(x)  # dry run to init

        return gemm

    exit_stack.enter_context(set_current_vllm_config(VllmConfig()))

    outside_loop_count = 6
    op_list = []
    for i in range(outside_loop_count):
        op_list.append(create_gemm())

    def kernel_func():
        for op in op_list:
            op.forward(x)

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=3,
        num_runs=6,
        repeat_n=1,
    ) as results:
        pass

    log_perf(
        item_list=[
            {
                "gemm_dtype": gemm_type,
                "m": m,
                "n": n,
                "k": k,
                "latency": results["latency_ms"] / outside_loop_count,
            }
        ],
        framework="VLLM",
        version=vllm_version,
        device_name=get_device_module().get_device_name(device),
        op_name="gemm",
        kernel_source="vllm_default",
        perf_filename=perf_filename,
        power_stats=None,
    )


if __name__ == "__main__":
    test_cases = get_gemm_test_cases()
    for test_case in test_cases[:10]:
        run_gemm(*test_case)

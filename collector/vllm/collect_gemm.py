# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM GEMM collector for CUDA backends.

Builds vLLM RowParallelLinear layers with synthetic weights to benchmark BF16,
FP8, FP8 block, and FP4-style paths where available. Shared GEMM shapes come
from `case_generator.py`; this file handles vLLM config contexts, distributed setup,
quantized-weight preparation, and backend-specific skips.
"""

__compat__ = "vllm==0.24.0"

import os
from types import SimpleNamespace

import torch
from vllm._custom_ops import scaled_fp4_quant as _scaled_fp4_quant
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig as _CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.utils.deep_gemm import per_block_cast_to_fp8
from vllm.version import __version__ as vllm_version

from collector.case_generator import get_gemm_case_specs
from collector.helper import benchmark_with_power, get_sm_version, log_perf
from collector.vllm.utils import setup_distributed, with_exit_stack

FP8_BLOCK_SHAPE = (128, 128)

# NVFP4 is source-supported by vLLM 0.24.0 on datacenter and RTX Blackwell.
# These SM100/SM103/SM120 paths remain hardware-unvalidated in this effort.
BLACKWELL_SMS = (100, 103, 120)

_NVFP4_QUANT_ARGS = {
    "num_bits": 4,
    "type": "float",
    "strategy": "tensor_group",
    "group_size": 16,
    "symmetric": True,
    "dynamic": False,
}


def get_gemm_test_cases():
    sm = get_sm_version()

    gemm_list = ["bfloat16"]
    if sm in (89, 90, *BLACKWELL_SMS):
        gemm_list += ["fp8"]
    # Blockwise FP8 kernels are available on Hopper/Blackwell+
    if sm in (90, *BLACKWELL_SMS):
        gemm_list += ["fp8_block"]

    if sm in BLACKWELL_SMS:
        gemm_list += ["nvfp4"]

    requested_gemm_types = os.environ.get("AIC_COLLECT_GEMM_TYPES")
    if requested_gemm_types:
        requested = {item.strip() for item in requested_gemm_types.split(",") if item.strip()}
        gemm_list = [gemm_type for gemm_type in gemm_list if gemm_type in requested]

    test_cases = []

    for gemm_common_testcase in get_gemm_case_specs():
        x = gemm_common_testcase.x
        n = gemm_common_testcase.n
        k = gemm_common_testcase.k
        for gemm_type in gemm_list:
            if gemm_type in ("nvfp4", "fp8_block") and (n < 128 or k < 128):
                continue
            if gemm_type == "nvfp4" and ((n % 16) != 0 or (k % 16) != 0):
                continue
            if gemm_type == "fp8_block":
                block_n, block_k = FP8_BLOCK_SHAPE
                # Block-wise kernels expect dimensions that align with the block.
                if (n % block_n) != 0 or (k % block_k) != 0:
                    continue
                # Blackwell block kernel currently prefers m divisible by 4.
                if sm in BLACKWELL_SMS and (x % 4) != 0:
                    continue

            test_cases.append([gemm_type, x, n, k])

    return test_cases


@with_exit_stack
def run_gemm(exit_stack, gemm_type, m, n, k, *, perf_filename, device="cuda:0"):
    setup_distributed(device)

    dtype = torch.bfloat16
    torch.set_default_dtype(dtype)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    x = torch.randn((m, k), dtype=dtype, device=torch.device(device))

    if gemm_type == "fp8":
        qc = Fp8Config(
            is_checkpoint_fp8_serialized=True,
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
    elif gemm_type == "nvfp4":
        qc = _CompressedTensorsConfig.from_config(
            {
                "quant_type": "compressed-tensors",
                "format": "nvfp4-pack-quantized",
                "global_compression_ratio": 1.0,
                "config_groups": {
                    "group_0": {
                        "weights": _NVFP4_QUANT_ARGS,
                        "input_activations": _NVFP4_QUANT_ARGS,
                        "targets": ["Linear"],
                        "output_activations": None,
                    }
                },
            }
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
        # TODO, to evaluate random weights impact
        gemm.to(torch.device(device))

        if gemm_type == "fp8":
            with torch.no_grad():
                gemm.weight.fill_(0.01)
                gemm.weight_scale.fill_(1.0)
            gemm.quant_method.process_weights_after_loading(gemm)
        elif gemm_type == "fp8_block":
            block_n, block_k = FP8_BLOCK_SHAPE
            with torch.no_grad():
                # Blockwise quantize a random weight to provide valid scales.
                raw_weight = torch.randn((n, k), dtype=torch.float32, device=device)
                q_weight, weight_scale = per_block_cast_to_fp8(raw_weight, [block_n, block_k], use_ue8m0=False)
                gemm.weight.copy_(q_weight)
                gemm.weight_scale_inv.copy_(weight_scale.contiguous().to(torch.float32))
                gemm.quant_method.process_weights_after_loading(gemm)
        elif gemm_type == "nvfp4":
            with torch.no_grad():
                weight_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device=device)
                w_gscale_val = float(weight_bf16.abs().max()) / 6.0
                weight_fp4, weight_scale_fp8 = _scaled_fp4_quant(
                    weight_bf16,
                    torch.tensor(1.0 / w_gscale_val, dtype=torch.float32, device=device),
                    is_sf_swizzled_layout=False,
                )
                in_gscale_val = float(x.abs().max()) / 6.0
                gemm.weight_packed.data.copy_(weight_fp4)
                gemm.weight_scale.data.copy_(weight_scale_fp8.to(torch.float8_e4m3fn))
                # CT convention: global_scale parameters store 1/actual_scale.
                gemm.weight_global_scale.data.fill_(1.0 / w_gscale_val)
                gemm.input_global_scale.data.fill_(1.0 / in_gscale_val)
            gemm.scheme.process_weights_after_loading(gemm)

        gemm.forward(x)  # dry run to init

        return gemm

    vllm_config = VllmConfig()
    if vllm_config.model_config is None:
        vllm_config.model_config = SimpleNamespace(
            dtype=dtype,
            hf_text_config=SimpleNamespace(model_type=""),
            model="collector_dummy",
        )
    exit_stack.enter_context(set_current_vllm_config(vllm_config))

    outside_loop_count = 1 if gemm_type in ("fp8_block", "nvfp4") else 6
    op_list = []
    for i in range(outside_loop_count):
        op_list.append(create_gemm())

    if gemm_type in {"fp8", "fp8_block"}:
        kernel_sources = {type(op.quant_method.fp8_linear).__name__ for op in op_list}
    elif gemm_type == "nvfp4":
        kernel_sources = {type(op.scheme.kernel).__name__ for op in op_list}
    else:
        kernel_sources = {"torch.nn.functional.linear"}
    if len(kernel_sources) != 1:
        raise RuntimeError(f"vLLM selected inconsistent GEMM kernels: {sorted(kernel_sources)}")
    kernel_source = kernel_sources.pop()

    def kernel_func():
        for op in op_list:
            op.forward(x)

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=3,
        num_runs=6,
        repeat_n=1,
        use_cuda_graph=gemm_type != "fp8_block",
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
        device_name=torch.cuda.get_device_name(device),
        op_name="gemm",
        kernel_source=kernel_source,
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    test_cases = get_gemm_test_cases()
    for test_case in test_cases[:10]:
        run_gemm(*test_case, perf_filename=PerfFile.GEMM)

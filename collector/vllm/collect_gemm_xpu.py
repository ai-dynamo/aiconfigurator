# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM GEMM collector for XPU devices.

This is the XPU counterpart to the CUDA vLLM GEMM collector. It builds
RowParallelLinear layers, prepares supported FP8 paths, expands YAML-backed
matrix shapes, and logs perf rows using XPU-aware device helpers.
"""

__compat__ = "vllm==0.26.0"

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

from collector.case_generator import get_gemm_case_specs, get_gemm_type_specs
from collector.helper import benchmark_with_power, get_device_module, log_perf
from collector.vllm.utils_xpu import create_vllm_config, setup_distributed, with_exit_stack

FP8_BLOCK_SHAPE = (128, 128)

# Cases whose footprint exceeds this fraction of device memory purge the
# allocator cache after running.
_CLEANUP_FOOTPRINT_FRACTION = 0.05

# Max device-memory fraction for outside_loop_count op copies.
_LOOP_MEM_BUDGET_FRACTION = 0.45

# Cap on outside_loop_count (upper bound; may drop lower if memory budget is
# tight for the shape). Value inherited from the CUDA collector.
_MAX_OUTSIDE_LOOP_COUNT = 6


def _gemm_peak_footprint_bytes(gemm_type: str, m: int, n: int, k: int, copies: int = 1) -> int:
    """Peak GEMM case footprint (bytes): input + copies*(weight+output) bf16,
    plus fp8_block float32 staging (n*k*4) or fp8 int8 staging (n*k).

    ``copies=0`` returns just the copy-independent portion (input + staging).
    Over-reserves for fp8 paths (weight/output sized as bf16) on purpose.
    """
    input_bytes = m * k * 2
    per_copy = (n * k + m * n) * 2
    footprint = input_bytes + per_copy * max(copies, 0)
    if gemm_type == "fp8_block":
        footprint += n * k * 4
    elif gemm_type == "fp8":
        footprint += n * k
    return footprint


def get_gemm_test_cases():
    gemm_list = get_gemm_type_specs("vllm_xpu")
    if not gemm_list:
        raise RuntimeError("collector/cases/base_ops/gemm.yaml must define vllm_xpu gemm_types")

    test_cases = []

    for gemm_common_testcase in get_gemm_case_specs("vllm_xpu"):
        x = gemm_common_testcase.x
        n = gemm_common_testcase.n
        k = gemm_common_testcase.k
        for gemm_type in gemm_list:
            test_cases.append([gemm_type, x, n, k])

    return test_cases


@with_exit_stack
def run_gemm(exit_stack, gemm_type, m, n, k, *, perf_filename, device="xpu:0"):
    # Force DeepGEMM path when available to capture the intended kernel.
    os.environ["VLLM_USE_DEEP_GEMM"] = "1"

    setup_distributed(device)

    dtype = torch.bfloat16
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

        gemm.forward(x)  # noqa: F821  # dry run to init

        return gemm

    # vLLM >=0.20 requires model_config.dtype in VllmConfig for FP8 layers;
    # fall back to bare VllmConfig() for older versions where the vLLM config
    # APIs used by create_vllm_config() may be incompatible.
    try:
        model = os.path.join(os.path.dirname(__file__), "fake_hf_model")
        vllm_config = create_vllm_config(model_name=model, dtype=dtype)
    except Exception as exc:
        print(f"create_vllm_config failed, falling back to VllmConfig(): {exc}")
        vllm_config = VllmConfig()
    exit_stack.enter_context(set_current_vllm_config(vllm_config))

    # Cap loop copies to fit memory; shares sizing with the cleanup gate below.
    total_mem = get_device_module().get_device_properties(device).total_memory
    per_copy = (n * k + m * n) * 2
    fixed_bytes = _gemm_peak_footprint_bytes(gemm_type, m, n, k, copies=0)
    budget = int(total_mem * _LOOP_MEM_BUDGET_FRACTION)
    outside_loop_count = max(1, min(_MAX_OUTSIDE_LOOP_COUNT, (budget - fixed_bytes) // max(per_copy, 1)))

    op_list = []
    try:
        for i in range(outside_loop_count):
            op_list.append(create_gemm())

        def kernel_func():
            for op in op_list:  # noqa: F821
                op.forward(x)  # noqa: F821

        with benchmark_with_power(
            device=device,
            kernel_func=kernel_func,
            num_warmups=3,
            num_runs=6,
            repeat_n=1,
        ) as results:
            pass
    finally:
        # Free this case before the next one runs.
        del op_list
        del x
        import gc

        gc.collect()
        # Purge cache only when the case actually held enough memory to matter;
        # small cases keep the warm allocator pool.
        case_peak = _gemm_peak_footprint_bytes(gemm_type, m, n, k, copies=outside_loop_count)
        if case_peak > total_mem * _CLEANUP_FOOTPRINT_FRACTION:
            get_device_module().empty_cache()

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
    from collector.registry_types import PerfFile

    test_cases = get_gemm_test_cases()
    for test_case in test_cases[:10]:
        run_gemm(*test_case, perf_filename=PerfFile.GEMM)

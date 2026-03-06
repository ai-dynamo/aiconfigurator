# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__compat__ = "vllm>=0.11.0"

import os

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    maybe_post_process_fp8_weight_block,
)
from vllm.utils.deep_gemm import per_block_cast_to_fp8
from vllm.version import __version__ as vllm_version

try:
    from flashinfer import fp4_quantize as flashinfer_fp4_quantize
    from flashinfer import mm_fp4 as flashinfer_mm_fp4
    from flashinfer import shuffle_matrix_a as flashinfer_shuffle_a
    from flashinfer import shuffle_matrix_sf_a as flashinfer_shuffle_sf_a

    HAS_FLASHINFER_FP4 = True
except ImportError:
    HAS_FLASHINFER_FP4 = False

from collector.common_test_cases import get_gemm_common_test_cases
from collector.helper import benchmark_with_power, get_sm_version, log_perf
from collector.vllm.utils import setup_distributed, with_exit_stack

FP8_BLOCK_SHAPE = (128, 128)


def _round_up(x: int, y: int) -> int:
    return (x + y - 1) // y * y


class _FlashInferNvFP4Op:
    """Wraps FlashInfer NVFP4 GEMM to match RowParallelLinear's .forward(x) interface."""

    def __init__(self, m, n, k, device, dtype):
        self.dtype = dtype
        self.device = device

        a_global_scale = torch.tensor([1.0], device=device, dtype=torch.float32)
        b_global_scale = torch.tensor([1.0], device=device, dtype=torch.float32)
        self.alpha = 1.0 / (a_global_scale * b_global_scale)
        self.a_global_scale = a_global_scale

        b_bf16 = torch.randn((n, k), device=device, dtype=dtype)
        b_fp4_linear, b_sf_linear = flashinfer_fp4_quantize(
            b_bf16, b_global_scale, is_sf_swizzled_layout=False
        )

        epilogue_tile_m = 128
        b_fp4_shuffled = flashinfer_shuffle_a(b_fp4_linear, epilogue_tile_m)
        b_sf_shuffled = flashinfer_shuffle_sf_a(
            b_sf_linear.view(torch.uint8), epilogue_tile_m
        ).view(torch.float8_e4m3fn)
        self.b_fp4 = b_fp4_shuffled.t()
        self.b_sf = b_sf_shuffled.t()

        self.out = torch.empty(
            (m, _round_up(n, 128)), device=device, dtype=dtype
        )

    def forward(self, x):
        a_fp4, a_sf = flashinfer_fp4_quantize(
            x, self.a_global_scale, is_sf_swizzled_layout=True
        )
        return flashinfer_mm_fp4(
            a_fp4,
            self.b_fp4,
            a_sf,
            self.b_sf,
            self.alpha,
            self.dtype,
            backend="cutlass",
            out=self.out,
        )


def get_gemm_test_cases():
    sm = get_sm_version()

    gemm_list = ["float16"]
    if sm > 86:
        gemm_list += ["fp8"]
    # Blockwise FP8 kernels are available on Hopper/Blackwell+
    if sm >= 90:
        gemm_list += ["fp8_block"]

    if sm >= 100:
        gemm_list += ["nvfp4"]

    test_cases = []

    for gemm_common_testcase in get_gemm_common_test_cases():
        x = gemm_common_testcase.x
        n = gemm_common_testcase.n
        k = gemm_common_testcase.k
        for gemm_type in gemm_list:
            if gemm_type in ("nvfp4", "fp8_block") and (n < 128 or k < 128):
                continue
            if gemm_type == "fp8_block":
                block_n, block_k = FP8_BLOCK_SHAPE
                # Block-wise kernels expect dimensions that align with the block.
                if (n % block_n) != 0 or (k % block_k) != 0:
                    continue
                # Blackwell block kernel currently prefers m divisible by 4.
                if sm >= 100 and (x % 4) != 0:
                    continue

            test_cases.append([gemm_type, x, n, k, "gemm_perf.txt"])

    return test_cases


@with_exit_stack
def run_gemm(exit_stack, gemm_type, m, n, k, perf_filename, device="cuda:0"):
    # Force DeepGEMM path when available to capture the intended kernel.
    os.environ["VLLM_USE_DEEP_GEMM"] = "1"

    setup_distributed(device)

    dtype = torch.float16
    torch.set_default_dtype(dtype)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

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
            weight_block_size=list(FP8_BLOCK_SHAPE),
        )
    else:
        qc = None

    def create_gemm():
        if gemm_type == "nvfp4":
            if not HAS_FLASHINFER_FP4:
                return None
            op = _FlashInferNvFP4Op(m, n, k, torch.device(device), dtype)
            op.forward(x)  # dry run
            return op

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
            # mnk = 1,128,128 weight stride = (128,1)
            # transpose to (1,128) for fp8 cutlass limit
            gemm.weight = torch.nn.Parameter(new_weight)
            # print("after fix, weight stride:", gemm.weight.data.stride())
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
        op = create_gemm()
        if op is not None:
            op_list.append(op)

    if not op_list:
        print(f"Skipping {gemm_type}: required dependencies not available")
        return

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
        device_name=torch.cuda.get_device_name(device),
        op_name="gemm",
        kernel_source="vllm_default",
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


if __name__ == "__main__":
    test_cases = get_gemm_test_cases()
    for test_case in test_cases[:10]:
        run_gemm(*test_case)

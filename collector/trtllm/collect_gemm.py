# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import math
from collections import defaultdict

import tensorrt_llm
import torch
import torch.nn.functional as F
from common_test_cases import get_gemm_common_test_cases
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

from helper import benchmark_with_power, get_sm_version, log_perf


def pad_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def _compute_fp8_block_weight_scales(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """Compute per-block scales from weight tensor (shape: [n, k])."""
    n, k = weight.shape
    n_blocks = math.ceil(n / group_size)
    k_blocks = math.ceil(k / group_size)
    pad_n = n_blocks * group_size - n
    pad_k = k_blocks * group_size - k

    weight_abs = weight.abs()
    if pad_n or pad_k:
        weight_abs = F.pad(weight_abs, (0, pad_k, 0, pad_n))

    block_max = weight_abs.view(n_blocks, group_size, k_blocks, group_size).amax(dim=(1, 3))
    return (block_max / 448.0).clamp_min(1e-6).to(dtype=torch.float32)


# Per-worker-process weight cache for large shapes (outside_loop_count == 1).
# Keyed by (gemm_type, n, k).  Only the most recently used entry is kept so
# that GPU memory is not accumulated across (n, k) groups.
_weight_cache: dict = {}


def _get_l2_cache_bytes(device_id: int = 0) -> int:
    """Query the GPU L2 cache size via the CUDA runtime API."""
    cuda_dev_attr_l2_cache_size = 38
    libcudart = ctypes.CDLL("libcudart.so")
    value = ctypes.c_int()
    ret = libcudart.cudaDeviceGetAttribute(ctypes.byref(value), cuda_dev_attr_l2_cache_size, device_id)
    if ret != 0 or value.value <= 0:
        raise RuntimeError(f"Failed to query L2 cache size (cudaError={ret}, value={value.value})")
    return value.value


def _build_weights(gemm_type: str, n: int, k: int, device, dtype, group_size, x) -> dict:
    """Allocate and return the weight tensors for one GEMM copy."""
    if gemm_type == "fp8":
        return {
            "weight": torch.randn((n, k), dtype=torch.bfloat16, device=device).to(dtype=torch.float8_e4m3fn),
            "weight_scale": torch.randn(1, dtype=torch.float32, device=device),
        }
    elif gemm_type == "fp8_block":
        fp_weight = torch.randn((n, k), dtype=torch.bfloat16, device=device)
        return {
            "weight": fp_weight.to(dtype=torch.float8_e4m3fn),
            "weight_scale": _compute_fp8_block_weight_scales(fp_weight, group_size),
        }
    elif gemm_type == "nvfp4":
        w = torch.randn((n, k), dtype=torch.float16, device=device)
        w_sf_global = (448 * 6) / w.abs().max().float()
        w_fp4, w_sf_block = torch.ops.trtllm.fp4_quantize(w, w_sf_global, 16, False)
        if tensorrt_llm.__version__.startswith(("1.1.0", "1.2.0", "1.3.0")):
            w_sf_block_unswizzled = torch.ops.trtllm.block_scale_interleave_reverse(
                w_sf_block.cpu().view(pad_up(n, 128), -1)
            )
        else:
            w_sf_block_unswizzled = torch.ops.trtllm.nvfp4_block_scale_interleave_reverse(w_sf_block.cpu().view(k, -1))
        x_sf_global = (448 * 6) / x.abs().max().float()
        return {
            "weight": w_fp4.cpu(),
            "weight_scale": w_sf_block_unswizzled.view(torch.float8_e4m3fn),
            "weight_scale_2": 1.0 / w_sf_global.cpu(),
            "input_scale": 1.0 / x_sf_global.cpu(),
        }
    else:  # float16
        return {"weight": torch.randn((n, k), dtype=torch.bfloat16, device=device)}


def get_gemm_test_cases():
    gemm_list = ["float16"]
    sm_version = get_sm_version()
    if sm_version > 86:
        gemm_list += ["fp8"]
        # SM90 (Hopper) and SM100 (Blackwell) both support fp8_block.
        # SM90 uses CUTLASS backend with FP32 scale.
        # SM100 uses TRTLLM/DeepGEMM backend with UE8M0 scale (MXFP8 style).
        gemm_list += ["fp8_block"]
    if sm_version >= 100:
        gemm_list += ["nvfp4"]

    # Group (x, n, k) triples by (n, k) so that all batch sizes for the same
    # weight shape run consecutively.  This allows the per-process weight cache
    # in run_gemm to amortise the expensive torch.randn weight allocation across
    # all 75 x values instead of reallocating on every call.
    all_cases = get_gemm_common_test_cases()
    nk_to_x: dict = defaultdict(list)
    for c in all_cases:
        nk_to_x[(c.n, c.k)].append(c.x)

    # Sort: largest (n, k) first so that the most expensive shapes are processed
    # early (same behaviour as before for progress visibility).
    nk_pairs_sorted = sorted(nk_to_x.keys(), key=lambda nk: (-nk[0], -nk[1]))

    test_cases = []
    for n, k in nk_pairs_sorted:
        x_list = sorted(nk_to_x[(n, k)], reverse=True)  # largest batch size first
        for gemm_type in gemm_list:
            if (gemm_type == "nvfp4" or gemm_type == "fp8_block") and (n < 128 or k < 128):
                continue
            for x in x_list:
                test_cases.append([gemm_type, x, n, k, "gemm_perf.txt"])

    return test_cases


def run_gemm(gemm_type, m, n, k, perf_filename, device="cuda:0"):
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    sm_version = get_sm_version()

    dtype = torch.bfloat16
    x = torch.randn((m, k), dtype=dtype).to(torch.device(device))

    if gemm_type == "fp8":
        group_size = None
        qc = QuantConfig(quant_algo=QuantAlgo.FP8)
    elif gemm_type == "fp8_block":
        group_size = 128
        qc = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=group_size)
    elif gemm_type == "nvfp4":
        group_size = 128
        qc = QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=group_size)
    else:
        qc = None
        group_size = None

    _l2_cache_bytes = _get_l2_cache_bytes(device.index or 0)
    _bytes_per_elem = {"float16": 2, "fp8": 1, "fp8_block": 1, "nvfp4": 0.5}
    _weight_bytes = int(n * k * _bytes_per_elem[gemm_type])
    outside_loop_count = max(1, min(5, math.ceil(_l2_cache_bytes / _weight_bytes)))
    op_list = []

    if outside_loop_count == 1:
        # Large weight (exceeds GPU L2): reuse cached tensor across consecutive
        # tasks with the same (gemm_type, n, k) to avoid re-running the slow
        # torch.randn call for every batch-size variant.
        cache_key = (gemm_type, n, k)
        if cache_key not in _weight_cache:
            _weight_cache.clear()  # release the previous (n,k) weight
            _weight_cache[cache_key] = _build_weights(gemm_type, n, k, device, dtype, group_size, x)

        weights = _weight_cache[cache_key]

        # For nvfp4 the input_scale is x-dependent; recompute it cheaply.
        if gemm_type == "nvfp4":
            x_sf_global = (448 * 6) / x.abs().max().float()
            weights = {**weights, "input_scale": 1.0 / x_sf_global.cpu()}

        gemm = Linear(k, n, bias=False, dtype=dtype, quant_config=qc)
        gemm.load_weights([weights])
        gemm.to(torch.device(device))
        gemm.forward(x)  # dry run to init
        op_list.append(gemm)
    else:
        # Small weight: create distinct copies so that repeated calls span more
        # than the GPU L2 cache, giving a realistic memory-pressure measurement.
        for _i in range(outside_loop_count):
            weights = _build_weights(gemm_type, n, k, device, dtype, group_size, x)
            gemm = Linear(k, n, bias=False, dtype=dtype, quant_config=qc)
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

    kernel_source = "deepgemm" if gemm_type == "fp8_block" and sm_version >= 100 else "torch_flow"

    log_perf(
        item_list=[
            {"gemm_dtype": gemm_type, "m": m, "n": n, "k": k, "latency": results["latency_ms"] / outside_loop_count}
        ],
        framework="TRTLLM",
        version=tensorrt_llm.__version__,
        device_name=torch.cuda.get_device_name(device),
        op_name="gemm",
        kernel_source=kernel_source,
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


if __name__ == "__main__":
    for test_case in get_gemm_test_cases():
        run_gemm(*test_case)

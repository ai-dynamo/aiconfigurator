# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__compat__ = "vllm>=0.11.0"

import os

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
from vllm.model_executor.layers.fused_moe.layer import determine_expert_map
from vllm.model_executor.layers.fused_moe.utils import disable_inplace
from vllm.version import __version__ as vllm_version

# Compatibility: block FP8 helpers may differ by version.
# Priority: vllm.utils.deep_gemm -> deep_gemm extension -> None.
try:
    from vllm.utils.deep_gemm import per_block_cast_to_fp8
except Exception:
    try:
        import deep_gemm  # type: ignore

        per_block_cast_to_fp8 = getattr(deep_gemm, "per_block_cast_to_fp8", None)
    except Exception:
        per_block_cast_to_fp8 = None  # type: ignore[assignment]

# nvfp4 MoE helpers (Blackwell SM 100+).
# Note: fused_experts does not support nvfp4; use flashinfer_cutlass_fused_moe directly.
try:
    from flashinfer.fused_moe.core import ActivationType as _ActivationType
    from vllm._custom_ops import scaled_fp4_quant as _scaled_fp4_quant
    from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
        swizzle_blockscale as _swizzle_blockscale,
    )
    from vllm.utils.flashinfer import flashinfer_cutlass_fused_moe as _flashinfer_cutlass_fused_moe

    _nvfp4_moe_available = True
except Exception:
    _nvfp4_moe_available = False

from collector.common_test_cases import get_common_moe_test_cases
from collector.helper import balanced_logits, benchmark_with_power, get_sm_version, log_perf, power_law_logits_v3

aic_debug = int(os.getenv("aic_moe_debug", "0"))  # noqa: SIM112


def get_moe_test_cases():
    """Generate MoE test cases"""

    # Quantization types supported by vLLM
    moe_list = ["float16"]
    if get_sm_version() > 86:
        moe_list += ["fp8"]
    if get_sm_version() >= 90 and per_block_cast_to_fp8 is not None:
        moe_list += ["fp8_block"]
    if get_sm_version() >= 100 and _nvfp4_moe_available:
        moe_list += ["nvfp4"]

    test_cases = []

    for common_moe_testcase in get_common_moe_test_cases():
        if common_moe_testcase.token_expert_distribution != "power_law":
            continue

        model_name = common_moe_testcase.model_name
        if model_name in ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
            continue

        # vllm does not support TP when EP is enabled.
        if common_moe_testcase.tp > 1 and common_moe_testcase.ep > 1:
            continue

        local_inter_size = common_moe_testcase.inter_size // common_moe_testcase.tp

        for moe_type in moe_list:
            # fp8_block requires hidden_size and local_inter_size divisible by 128.
            if moe_type == "fp8_block" and (common_moe_testcase.hidden_size % 128 != 0 or local_inter_size % 128 != 0):
                continue

            # nvfp4 requires hidden_size and local_inter_size divisible by 16 (group_size).
            # flashinfer_cutlass_fused_moe (the nvfp4 kernel) does not support EP > 1.
            if moe_type == "nvfp4" and (
                common_moe_testcase.hidden_size % 16 != 0 or local_inter_size % 16 != 0 or common_moe_testcase.ep > 1
            ):
                continue

            test_cases.append(
                [
                    moe_type,
                    common_moe_testcase.num_tokens_list,
                    common_moe_testcase.hidden_size,
                    common_moe_testcase.inter_size,
                    common_moe_testcase.topk,
                    common_moe_testcase.num_experts,
                    common_moe_testcase.tp,
                    common_moe_testcase.ep,
                    common_moe_testcase.model_name,
                    "moe_perf.txt",
                    common_moe_testcase.token_expert_distribution,
                    common_moe_testcase.power_law_alpha,
                ]
            )

    return test_cases


def run_moe_torch(
    moe_type,
    num_tokens_lists,
    hidden_size,
    inter_size,
    topk,
    num_experts,
    moe_tp_size,
    moe_ep_size,
    model_name,
    perf_filename,
    distributed="power_law",
    power_law_alpha=0.0,
    device="cuda:0",
):
    """Run vLLM MoE performance benchmarking"""
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    # Configure quantization parameters
    dtype = torch.float16
    quant_config = None
    block_shape: list[int] | None = None
    a1_scale = None
    a2_scale = None

    # Calculate local number of experts
    local_inter_size = inter_size // moe_tp_size
    expert_map_result = determine_expert_map(moe_ep_size, 0, num_experts)
    if isinstance(expert_map_result, tuple) and len(expert_map_result) == 3:
        local_num_experts, expert_map, _ = expert_map_result
    else:
        # Backward compatibility with older determine_expert_map signatures
        # that return only (local_num_experts, expert_map)
        local_num_experts, expert_map = expert_map_result  # type: ignore[misc]

    # Create weight tensors
    # w1: gate + up projection weights [num_experts, 2 * inter_size, hidden_size]
    # w2: down projection weights [num_experts, hidden_size, inter_size]
    w1 = torch.randn(
        local_num_experts,
        2 * local_inter_size,
        hidden_size,
        dtype=torch.float16,
        device=device,
    )
    w2 = torch.randn(
        local_num_experts,
        hidden_size,
        local_inter_size,
        dtype=torch.float16,
        device=device,
    )

    if moe_type in ["fp8", "fp8_block"]:
        dtype = torch.float8_e4m3fn
        if moe_type == "fp8_block":
            block_shape = [128, 128]

            if per_block_cast_to_fp8 is None:
                raise ImportError("per_block_cast_to_fp8 is unavailable; fp8_block requires a newer vLLM build.")

            w1_scale_list = []
            w2_scale_list = []
            w1_q = torch.empty_like(w1, dtype=dtype)
            w2_q = torch.empty_like(w2, dtype=dtype)
            for i in range(local_num_experts):
                w1_q[i], w1_scale_i = per_block_cast_to_fp8(w1[i], block_size=block_shape, use_ue8m0=True)
                w2_q[i], w2_scale_i = per_block_cast_to_fp8(w2[i], block_size=block_shape, use_ue8m0=True)
                w1_scale_list.append(w1_scale_i)
                w2_scale_list.append(w2_scale_i)
            w1 = w1_q
            w2 = w2_q
            w1_scale = torch.stack(w1_scale_list)
            w2_scale = torch.stack(w2_scale_list)
        else:
            w1_scale = torch.randn(local_num_experts, dtype=torch.float32, device=device)
            w2_scale = torch.randn(local_num_experts, dtype=torch.float32, device=device)
            a1_scale = torch.randn(1, dtype=torch.float32, device=device)
            a2_scale = torch.randn(1, dtype=torch.float32, device=device)

        quant_config = fp8_w8a8_moe_quant_config(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=block_shape,
        )

    if dtype == torch.float8_e4m3fn:
        w1 = w1.to(dtype)
        w2 = w2.to(dtype)

    use_inplace = not disable_inplace()

    # nvfp4 MoE: quantize weights to fp4 and precompute quant scales.
    # fused_experts does not support nvfp4; use flashinfer_cutlass_fused_moe directly.
    nvfp4_w1_fp4 = nvfp4_w2_fp4 = None
    nvfp4_quant_scales: list | None = None
    if moe_type == "nvfp4":
        dtype = torch.bfloat16  # nvfp4 uses bfloat16 activations

        def _quant_expert_weights_nvfp4(w_bf16: torch.Tensor):
            E = w_bf16.shape[0]  # noqa: N806
            gscales = w_bf16.view(E, -1).abs().max(dim=1).values.float() / 6.0
            fp4_list, sf_list = [], []
            for i in range(E):
                gscale_inv = torch.tensor(1.0 / gscales[i].item(), dtype=torch.float32, device=w_bf16.device)
                w_fp4, w_sf = _scaled_fp4_quant(
                    w_bf16[i].contiguous(), gscale_inv, is_sf_swizzled_layout=False, backend="cutlass"
                )
                fp4_list.append(w_fp4)
                sf_list.append(w_sf)
            fp4 = torch.stack(fp4_list)
            scale = _swizzle_blockscale(torch.stack(sf_list))
            return fp4, scale, gscales

        nvfp4_w1_fp4, nvfp4_w1_scale, w1_gscales = _quant_expert_weights_nvfp4(w1.to(torch.bfloat16))
        nvfp4_w2_fp4, nvfp4_w2_scale, w2_gscales = _quant_expert_weights_nvfp4(w2.to(torch.bfloat16))

        # Input activation global scale (CT convention: store 1/actual_scale).
        x_actual_scale = 1.0 / 6.0
        a1_gscale = torch.full((local_num_experts,), 1.0 / x_actual_scale, dtype=torch.float32, device=device)
        a2_gscale = a1_gscale.clone()
        # Combined dequantization alpha: w_scale * x_scale.
        g1_alphas = (w1_gscales * x_actual_scale).to(torch.float32)
        g2_alphas = (w2_gscales * x_actual_scale).to(torch.float32)
        nvfp4_quant_scales = [
            a1_gscale,
            nvfp4_w1_scale.view(torch.int32),
            g1_alphas,
            a2_gscale,
            nvfp4_w2_scale.view(torch.int32),
            g2_alphas,
        ]

    # Performance testing for each token count
    for num_tokens_idx, num_tokens in enumerate(num_tokens_lists):
        print("num_tokens", num_tokens)
        print("topk", topk)
        act_dtype = torch.bfloat16 if moe_type == "nvfp4" else torch.float16
        hidden_states = torch.randn([num_tokens, hidden_size], dtype=act_dtype, device=device)

        # Generate topk_weights and topk_ids
        num_iter = 10 if distributed == "power_law" else 1
        if distributed == "power_law":
            topk_weights_list = []
            topk_ids_list = []

            for _ in range(num_iter):
                logits = (
                    power_law_logits_v3(
                        num_tokens,
                        num_experts,
                        topk,
                        moe_ep_size,
                        power_law_alpha,
                    )
                    .half()
                    .to(device)
                )
                weights, ids = torch.topk(logits, topk, dim=-1)
                topk_weights_list.append(F.softmax(weights, dim=-1))
                topk_ids_list.append(ids)

        elif distributed == "balanced":
            actual_logits = balanced_logits(num_tokens, num_experts, topk).half().to(device)
            topk_weights, topk_ids = torch.topk(actual_logits, topk, dim=-1)
            topk_weights = F.softmax(topk_weights, dim=-1)

        else:
            raise ValueError(f"Unsupported distributed mode: {distributed}")

        num_warmups = 3
        num_runs = 6
        if distributed == "power_law":
            num_warmups = 1
            num_runs = 1

        if moe_type == "nvfp4":

            def run_single_iteration():
                if distributed == "power_law":
                    for tw, ti in zip(topk_weights_list, topk_ids_list):
                        local_num_tokens = tw.shape[0]
                        out = torch.empty(local_num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
                        _ = _flashinfer_cutlass_fused_moe(
                            input=hidden_states[:local_num_tokens],
                            token_selected_experts=ti.to(torch.int),
                            token_final_scales=tw,
                            fc1_expert_weights=nvfp4_w1_fp4.view(torch.long),
                            fc2_expert_weights=nvfp4_w2_fp4.view(torch.long),
                            output_dtype=torch.bfloat16,
                            quant_scales=nvfp4_quant_scales,
                            input_sf=None,
                            tp_size=1,
                            tp_rank=0,
                            ep_size=1,
                            ep_rank=0,
                            output=out,
                            activation_type=_ActivationType.Swiglu,
                        )
                else:
                    out = torch.empty(hidden_states.shape[0], hidden_size, dtype=torch.bfloat16, device=device)
                    _ = _flashinfer_cutlass_fused_moe(
                        input=hidden_states,
                        token_selected_experts=topk_ids.to(torch.int),
                        token_final_scales=topk_weights.float(),
                        fc1_expert_weights=nvfp4_w1_fp4.view(torch.long),
                        fc2_expert_weights=nvfp4_w2_fp4.view(torch.long),
                        output_dtype=torch.bfloat16,
                        quant_scales=nvfp4_quant_scales,
                        input_sf=None,
                        tp_size=1,
                        tp_rank=0,
                        ep_size=1,
                        ep_rank=0,
                        output=out,
                        activation_type=_ActivationType.Swiglu,
                    )
        else:

            def run_single_iteration():
                if distributed == "power_law":
                    for i, (tw, ti) in enumerate(zip(topk_weights_list, topk_ids_list)):
                        local_num_tokens = tw.shape[0]
                        _ = fused_experts(
                            hidden_states[:local_num_tokens],
                            w1,
                            w2,
                            tw,
                            ti,
                            inplace=use_inplace,
                            quant_config=quant_config,
                            global_num_experts=num_experts,
                            expert_map=expert_map,
                        )
                else:
                    _ = fused_experts(
                        hidden_states,
                        w1,
                        w2,
                        topk_weights,
                        topk_ids,
                        inplace=use_inplace,
                        quant_config=quant_config,
                        global_num_experts=num_experts,
                        expert_map=expert_map,
                    )

        def run_iterations():
            # Use benchmark_with_power context manager
            with benchmark_with_power(
                device=device,
                kernel_func=run_single_iteration,
                num_warmups=num_warmups,
                num_runs=num_runs,
                repeat_n=1,
            ) as results:
                pass

            return results["latency_ms"] / num_iter, results["power_stats"]

        try:
            latency, power_stats = run_iterations()
        except torch.OutOfMemoryError:
            # If OOM, check if we had at least one successful run.
            if num_tokens_idx > 0:
                break
            raise

        print(f"moe latency: {latency}")

        source = "flashinfer_cutlass_fused_moe" if moe_type == "nvfp4" else "vllm_fused_moe"

        log_perf(
            item_list=[
                {
                    "moe_dtype": moe_type,
                    "num_tokens": num_tokens,
                    "hidden_size": hidden_size,
                    "inter_size": inter_size,
                    "topk": topk,
                    "num_experts": num_experts,
                    "moe_tp_size": moe_tp_size,
                    "moe_ep_size": moe_ep_size,
                    "distribution": "power_law_" + str(power_law_alpha) if distributed == "power_law" else distributed,
                    "latency": latency,
                }
            ],
            framework="VLLM",
            version=vllm_version,
            device_name=torch.cuda.get_device_name(device),
            op_name="moe",
            kernel_source=source,
            perf_filename=perf_filename,
            power_stats=power_stats,
        )


if __name__ == "__main__":
    test_cases = get_moe_test_cases()
    print(f"Total test cases: {len(test_cases)}")

    for test_case in test_cases[:4]:
        print(f"Running test case: {test_case}")
        try:
            run_moe_torch(*test_case)
        except Exception as e:
            print(f"Test case failed: {test_case}")
            print(f"Error: {e}")
            continue

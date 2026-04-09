# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__compat__ = "vllm>=0.17.0"

import os

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
from vllm.model_executor.layers.fused_moe.layer import determine_expert_map
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

# vLLM >= 0.14.0 raises AssertionError in get_current_vllm_config() when called
# outside a set_current_vllm_config() context (https://github.com/vllm-project/vllm/pull/31747).
# vLLM's custom ops (e.g. _vllm_ops.scaled_fp4_quant) requires vllm config to decide how to dispatch.
from vllm.config import VllmConfig, set_current_vllm_config

# NVFP4 support: requires Blackwell (SM>=100) and FlashInfer TRTLLM FP4 kernel.
trtllm_fp4_block_scale_routed_moe = None
_vllm_ops = None
prepare_static_weights_for_trtllm_fp4_moe = None
_nvfp4_available = False
try:
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe  # type: ignore[assignment]
    from vllm import _custom_ops as _vllm_ops  # type: ignore[assignment]
    from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
        prepare_static_weights_for_trtllm_fp4_moe,  # type: ignore[assignment]
    )

    _nvfp4_available = True
except Exception:
    trtllm_fp4_block_scale_routed_moe = None
    _vllm_ops = None
    prepare_static_weights_for_trtllm_fp4_moe = None

# MXFP4 support: uses vLLM's high-level FusedMoE module with Mxfp4Config.
# This lets vLLM handle backend selection (FlashInfer/Triton/Marlin) and
# weight swizzle internally, so one code path works on all GPUs.
_mxfp4_available = False
try:
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config

    _mxfp4_available = True
except Exception:
    pass

from vllm.forward_context import set_forward_context

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
    if get_sm_version() >= 100 and _nvfp4_available:
        moe_list += ["nvfp4"]
    if _mxfp4_available:
        moe_list += ["w4a16_mxfp4"]

    _gpt_oss_models = {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}

    test_cases = []

    for common_moe_testcase in get_common_moe_test_cases():
        model_name = common_moe_testcase.model_name

        # vllm does not support TP when EP is enabled.
        if common_moe_testcase.tp > 1 and common_moe_testcase.ep > 1:
            continue

        for moe_type in moe_list:
            # GPT-OSS models only use mxfp4 quantization in production;
            # skip them for other quant types.
            if model_name in _gpt_oss_models and moe_type != "w4a16_mxfp4":
                continue
            # Conversely, mxfp4 is only collected for GPT-OSS models.
            if moe_type == "w4a16_mxfp4" and model_name not in _gpt_oss_models:
                continue

            # fp8_block requires hidden_size divisible by block group_size (128)
            if moe_type == "fp8_block" and (
                common_moe_testcase.hidden_size % 128 != 0
                or (common_moe_testcase.inter_size // common_moe_testcase.tp) % 128 != 0
            ):
                continue

            # nvfp4 uses TRTLLM FP4 kernel which has stricter constraints:
            # - hidden_size must be divisible by 512 (GEMM tiling requirement)
            # - local_inter_size (inter_size // tp) must be divisible by 64
            #   (GEMM1 N = 2*local_inter must be multiple of 128, GEMM2 K must be multiple of 64)
            # - topk must be <= 10 (MaxNumTopExperts in routing kernel)
            if moe_type == "nvfp4" and (
                common_moe_testcase.hidden_size % 512 != 0
                or (common_moe_testcase.inter_size // common_moe_testcase.tp) % 64 != 0
                or common_moe_testcase.topk > 10
            ):
                continue

            # w4a16_mxfp4 requires dimensions aligned to group_size (32)
            if moe_type == "w4a16_mxfp4" and (
                common_moe_testcase.hidden_size % 32 != 0
                or (common_moe_testcase.inter_size // common_moe_testcase.tp) % 32 != 0
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
    local_num_experts, expert_map, _ = determine_expert_map(moe_ep_size, 0, num_experts)

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

    # MXFP4 path: uses vLLM's high-level FusedMoE module with Mxfp4Config.
    # vLLM handles backend selection (FlashInfer/Triton/Marlin) and weight swizzle.
    #
    # We keep a reference to the VllmConfig used during construction because
    # vLLM 0.17.0's MoERunner (vllm-project/vllm#32344) calls
    # get_forward_context() → get_layer_from_name() during forward, which
    # looks up the module in static_forward_context.  FusedMoE registers
    # itself there during __init__, so we must pass the *same* config to
    # set_forward_context() at benchmark time.
    use_mxfp4 = moe_type == "w4a16_mxfp4"
    moe_module = None
    mxfp4_vllm_cfg = None

    if use_mxfp4:
        if not _mxfp4_available:
            raise ImportError("MXFP4 MoE requires vllm >= 0.17.0 with Mxfp4Config support.")

        mxfp4_quant_config = Mxfp4Config()

        # pcp_size=1: vLLM 0.17.0 added prefill context parallel to FusedMoE
        # (vllm-project/vllm#32344); without it, __init__ calls get_pcp_group()
        # which requires distributed init.
        mxfp4_vllm_cfg = VllmConfig()
        with set_current_vllm_config(mxfp4_vllm_cfg):
            moe_module = FusedMoE(
                num_experts=num_experts,
                top_k=topk,
                hidden_size=hidden_size,
                intermediate_size=inter_size,
                reduce_results=False,
                renormalize=True,
                quant_config=mxfp4_quant_config,
                tp_size=moe_tp_size,
                dp_size=1,
                ep_size=moe_ep_size,
                prefix="",
                has_bias=True,  # GPT-OSS uses bias
                activation="swigluoai",  # GPT-OSS activation
                pcp_size=1,
            )
        moe_module.to(device)
        moe_module.eval()
        moe_module.requires_grad_(False)

        # Fill synthetic mxfp4 weights (uint8 packed, E2M1 format)
        with torch.no_grad():
            moe_module.w13_weight.data.random_(0, 255)
            moe_module.w2_weight.data.random_(0, 255)
            moe_module.w13_weight_scale.data.random_(0, 255)
            moe_module.w2_weight_scale.data.random_(0, 255)
            if hasattr(moe_module, "w13_bias"):
                moe_module.w13_bias.data.normal_()
            if hasattr(moe_module, "w2_bias"):
                moe_module.w2_bias.data.normal_()

        # Trigger backend selection + weight swizzle for current GPU
        moe_module.quant_method.process_weights_after_loading(moe_module)

        # Free float16 weights; not used for mxfp4.
        del w1, w2

    # NVFP4 path: uses FlashInfer TRTLLM FP4 monolithic kernel (not fused_experts).
    use_nvfp4 = moe_type == "nvfp4"
    nvfp4_data: dict | None = None

    if use_nvfp4:
        _missing = [
            name
            for name, obj in [
                ("trtllm_fp4_block_scale_routed_moe", trtllm_fp4_block_scale_routed_moe),
                ("_vllm_ops", _vllm_ops),
                ("prepare_static_weights_for_trtllm_fp4_moe", prepare_static_weights_for_trtllm_fp4_moe),
            ]
            if obj is None
        ]
        if _missing:
            raise ImportError(
                f"NVFP4 MoE requires flashinfer and vllm >= 0.14.0 with FP4 support, but the following "
                f"could not be imported: {', '.join(_missing)}. "
                f"Install a compatible flashinfer build and ensure vllm >= 0.14.0 with FP4 support."
            )

        # Raw packed FP4 weights and block scales
        w1_raw = torch.randint(
            0, 255, (local_num_experts, 2 * local_inter_size, hidden_size // 2), dtype=torch.uint8, device=device
        )
        w2_raw = torch.randint(
            0, 255, (local_num_experts, hidden_size, local_inter_size // 2), dtype=torch.uint8, device=device
        )
        w1_scale_raw = torch.ones(
            local_num_experts, 2 * local_inter_size, hidden_size // 16, dtype=torch.float8_e4m3fn, device=device
        )
        w2_scale_raw = torch.ones(
            local_num_experts, hidden_size, local_inter_size // 16, dtype=torch.float8_e4m3fn, device=device
        )

        # Shuffle weights and scales for TRTLLM kernel layout
        w1_shuf, w1_scale_shuf, w2_shuf, w2_scale_shuf = prepare_static_weights_for_trtllm_fp4_moe(
            w1_raw,
            w2_raw,
            w1_scale_raw,
            w2_scale_raw,
            hidden_size=hidden_size,
            intermediate_size=local_inter_size,
            num_experts=local_num_experts,
            is_gated_activation=True,
        )
        del w1_raw, w2_raw, w1_scale_raw, w2_scale_raw

        # Per-expert scales
        a13_scale = torch.ones(local_num_experts, dtype=torch.float32, device=device)
        a2_scale_nvfp4 = torch.ones(local_num_experts, dtype=torch.float32, device=device)
        w13_scale_2 = torch.ones(local_num_experts, dtype=torch.float32, device=device)
        w2_scale_2 = torch.ones(local_num_experts, dtype=torch.float32, device=device)

        nvfp4_data = dict(
            w1=w1_shuf,
            w1_scale=w1_scale_shuf,
            w2=w2_shuf,
            w2_scale=w2_scale_shuf,
            g1_scale_c=a13_scale * w13_scale_2 / a2_scale_nvfp4,
            a1_gscale=1.0 / a13_scale,
            g1_alphas=a13_scale * w13_scale_2,
            g2_alphas=a2_scale_nvfp4 * w2_scale_2,
        )
        # Free the float16 weights; they are not used for nvfp4.
        del w1, w2

    elif moe_type in ["fp8", "fp8_block"]:
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

    if not use_mxfp4 and dtype == torch.float8_e4m3fn:
        w1 = w1.to(dtype)
        w2 = w2.to(dtype)

    # Performance testing for each token count
    for num_tokens_idx, num_tokens in enumerate(num_tokens_lists):
        print("num_tokens", num_tokens)
        print("topk", topk)
        hs_dtype = torch.bfloat16 if use_mxfp4 else torch.float16
        hidden_states = torch.randn([num_tokens, hidden_size], dtype=hs_dtype, device=device)

        # Generate routing inputs.
        # mxfp4 path uses FusedMoE.forward(hidden_states, router_logits) which does
        # routing internally; other paths need pre-computed topk_weights/topk_ids.
        num_iter = 5 if distributed == "power_law" else 1
        if use_mxfp4:
            # FusedMoE.forward() takes raw router logits (num_tokens, num_experts)
            if distributed == "power_law":
                actual_logits_list = [
                    power_law_logits_v3(num_tokens, num_experts, topk, moe_ep_size, power_law_alpha)
                    .to(torch.bfloat16)
                    .to(device)
                    for _ in range(num_iter)
                ]
            elif distributed == "balanced":
                actual_logits = balanced_logits(num_tokens, num_experts, topk).to(torch.bfloat16).to(device)
            else:
                raise ValueError(f"Unsupported distributed mode: {distributed}")
        elif distributed == "power_law":
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

            print("actual num_tokens: ", [topk_ids.shape[0] for topk_ids in topk_ids_list])

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

        def _run_nvfp4_once(hs, tw, ti):
            """Run a single nvfp4 MoE iteration via FlashInfer TRTLLM FP4 kernel."""
            # Quantize input to FP4
            x_fp4, x_scale = _vllm_ops.scaled_fp4_quant(
                hs.to(torch.bfloat16),
                nvfp4_data["a1_gscale"][0:1],
                is_sf_swizzled_layout=False,
            )
            num_tok = x_fp4.shape[0]
            scale_cols = hs.shape[1] // 16
            # Pack topk: (expert_id << 16) | bf16_weight_as_int16
            packed = (ti.to(torch.int32) << 16) | tw.to(torch.bfloat16).view(torch.int16).to(torch.int32)
            trtllm_fp4_block_scale_routed_moe(
                topk_ids=packed,
                routing_bias=None,
                hidden_states=x_fp4,
                hidden_states_scale=x_scale.view(num_tok, scale_cols).to(torch.float8_e4m3fn),
                gemm1_weights=nvfp4_data["w1"],
                gemm1_weights_scale=nvfp4_data["w1_scale"].view(torch.float8_e4m3fn),
                gemm1_bias=None,
                gemm1_alpha=None,
                gemm1_beta=None,
                gemm1_clamp_limit=None,
                gemm2_weights=nvfp4_data["w2"],
                gemm2_weights_scale=nvfp4_data["w2_scale"].view(torch.float8_e4m3fn),
                gemm2_bias=None,
                output1_scale_scalar=nvfp4_data["g1_scale_c"],
                output1_scale_gate_scalar=nvfp4_data["g1_alphas"],
                output2_scale_scalar=nvfp4_data["g2_alphas"],
                num_experts=num_experts,
                top_k=topk,
                n_group=0,
                topk_group=0,
                intermediate_size=local_inter_size,
                local_expert_offset=0,
                local_num_experts=local_num_experts,
                routed_scaling_factor=None,
                routing_method_type=1,  # Renormalize
                do_finalize=True,
            )

        def run_single_iteration():
            if use_mxfp4:
                # FusedMoE.forward(hidden_states, router_logits) does routing internally.
                if distributed == "power_law":
                    for logits in actual_logits_list:
                        moe_module.forward(hidden_states[: logits.shape[0]], logits[: logits.shape[0]])
                else:
                    moe_module.forward(hidden_states, actual_logits)
            elif use_nvfp4:
                if distributed == "power_law":
                    for tw, ti in zip(topk_weights_list, topk_ids_list, strict=True):
                        _run_nvfp4_once(hidden_states[: tw.shape[0]], tw, ti)
                else:
                    _run_nvfp4_once(hidden_states, topk_weights, topk_ids)
            elif distributed == "power_law":
                for i, (tw, ti) in enumerate(zip(topk_weights_list, topk_ids_list, strict=True)):
                    local_num_tokens = tw.shape[0]
                    _ = fused_experts(
                        hidden_states[:local_num_tokens],
                        w1,
                        w2,
                        tw,
                        ti,
                        inplace=False,
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
                    inplace=False,
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
                allow_graph_fail=True,
            ) as results:
                pass

            return results["latency_ms"] / num_iter, results["power_stats"]

        try:
            vllm_cfg = mxfp4_vllm_cfg if use_mxfp4 else VllmConfig()
            with set_current_vllm_config(vllm_cfg), set_forward_context({}, vllm_cfg):
                latency, power_stats = run_iterations()
        except torch.OutOfMemoryError:
            # If OOM, check if we had at least one successful run.
            if num_tokens_idx > 0:
                break
            raise

        print(f"moe latency: {latency}")

        if use_mxfp4:
            source = "vllm_mxfp4_moe"
        elif use_nvfp4:
            source = "vllm_flashinfer_trtllm_moe_fp4"
        else:
            source = "vllm_fused_moe"

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

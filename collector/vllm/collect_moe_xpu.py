# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__compat__ = "vllm>=0.11.0"

import itertools
import os

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.fused_moe.layer import determine_expert_map
from vllm.version import __version__ as vllm_version

from collector.common_test_cases import MoeCommonTestCase
from collector.helper import (
    balanced_logits,
    benchmark_with_power,
    get_device_module,
    log_perf,
    power_law_logits_v3,
)

if torch.xpu.is_available():
    try:
        from vllm_xpu_kernels.fused_moe_interface import xpu_fused_moe
    except Exception as e:
        print(f"Please refer to vllm_xpu_kernels for MoE on XPU, \n{e}")

aic_debug = int(os.getenv("aic_moe_debug", "0"))  # noqa: SIM112


def resolve_moe_activation(model_name: str) -> str:
    """Resolve MoE activation by model name.

    Priority:
    1) explicit env override via AIC_COLLECTOR_MOE_ACTIVATION
    2) model-family heuristic
    3) default silu
    """
    env_activation = os.getenv("AIC_COLLECTOR_MOE_ACTIVATION")
    if env_activation:
        return env_activation.strip().lower()

    name = model_name.lower()
    if any(key in name for key in ["qwen", "mixtral", "deepseek", "llama"]):
        return "silu"
    if "gemma" in name:
        return "gelu"
    if "gpt-oss" in name:
        return "swigluoai"

    return "silu"


def get_moe_xpu_test_cases():
    num_tokens = [
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
        320,
        384,
        512,
        768,
        1024,
        1536,
        2048,
        3072,
        4096,
        6144,
        8192,
        12288,
        16384,
    ]
    tp_list = [1, 2, 4, 8, 16, 32]
    ep_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    num_gpu_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    # previous commit's smaller configs for non-GPT-OSS models
    tp_list_small = [1, 2, 4, 8]
    ep_list_small = [1, 2, 4, 8]
    num_gpu_list_small = [1, 2, 4, 8]

    gpt_oss_model_names = {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}

    token_distributions = [
        ("balanced", 0.0),
        ("power_law", 1.01),
        ("power_law", 1.2),
    ]

    # hidden_size,inter_s,topk,num_expert
    model_config_list = [
        [2048, 1408, 4, 60, "Qwen/Qwen1.5-MoE-A2.7B"],
        [2048, 768, 8, 128, "Qwen/Qwen3-30B-A3B"],
        [8192, 5120, 1, 16, "meta-llama/Llama-4-Scout-17B-16E"],
        [4096, 1536, 8, 128, "Qwen/Qwen3-235B-A22B-Instruct-2507"],
        [2880, 2880, 4, 128, "openai/gpt-oss-120b"],
        [2880, 2880, 4, 32, "openai/gpt-oss-20b"],
    ]

    test_cases: list[MoeCommonTestCase] = []

    for model_config in model_config_list:
        hs, inter_s, topk, num_experts, model_name = model_config

        # Use large configs for GPT-OSS models, small configs for commit's models
        if model_name in gpt_oss_model_names:
            cur_tp_list = tp_list
            cur_ep_list = ep_list
            cur_num_gpu_list = num_gpu_list
        else:
            cur_tp_list = tp_list_small
            cur_ep_list = ep_list_small
            cur_num_gpu_list = num_gpu_list_small

        for (
            num_gpu,
            tp,
            ep,
            (token_distribution, power_law_alpha),
        ) in itertools.product(
            cur_num_gpu_list,
            cur_tp_list,
            cur_ep_list,
            token_distributions,
        ):
            # Qwen3-30B-A3B: exclude tp >= 8 as they are not used for actual deployments
            if model_name == "Qwen/Qwen3-30B-A3B" and tp >= 8:
                continue

            if tp * ep != num_gpu:
                continue
            if ep > num_experts:
                continue
            if num_experts % ep != 0:
                continue
            # we need to ensure inter_s can be divided by tp.
            if inter_s % tp != 0:
                continue

            test_cases.append(
                MoeCommonTestCase(
                    num_tokens_list=num_tokens,
                    hidden_size=hs,
                    inter_size=inter_s,
                    topk=topk,
                    num_experts=num_experts,
                    tp=tp,
                    ep=ep,
                    model_name=model_name,
                    token_expert_distribution=token_distribution,
                    power_law_alpha=power_law_alpha,
                )
            )

    return test_cases


def get_moe_test_cases():
    """Generate MoE test cases"""

    # Quantization types supported by vLLM
    moe_list = ["float16"]
    if hasattr(torch, "float8_e4m3fn"):
        moe_list += ["fp8"]
    moe_list += ["w4a16_mxfp4"]

    gpt_oss_models = {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}

    test_cases = []

    for common_moe_testcase in get_moe_xpu_test_cases():
        if common_moe_testcase.token_expert_distribution != "power_law":
            continue

        model_name = common_moe_testcase.model_name

        # vllm does not support TP when EP is enabled.
        if common_moe_testcase.tp > 1 and common_moe_testcase.ep > 1:
            continue

        for moe_type in moe_list:
            # GPT-OSS models only use mxfp4 quantization in production;
            # skip them for other quant types.
            if model_name in gpt_oss_models and moe_type != "w4a16_mxfp4":
                continue
            # Conversely, mxfp4 is only collected for GPT-OSS models.
            if moe_type == "w4a16_mxfp4" and model_name not in gpt_oss_models:
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


def quantize_fp8_per_expert(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_dtype = torch.float8_e4m3fn
    fp8_info = torch.finfo(fp8_dtype)
    fp32_weights = weights.to(torch.float32)

    num_experts_local = weights.shape[0]
    random_exponents = torch.randint(-3, 4, (num_experts_local,), device=weights.device)
    scales = torch.pow(2.0, random_exponents.float())

    qweights = (fp32_weights / scales.view(-1, 1, 1)).clamp(min=fp8_info.min, max=fp8_info.max).to(fp8_dtype)
    return qweights, scales


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
    device="xpu:0",
):
    """Run vLLM MoE performance benchmarking"""
    get_device_module().set_device(device)
    torch.set_default_device(device)

    use_mxfp4 = moe_type == "w4a16_mxfp4"
    is_fp8 = moe_type == "fp8"
    activation_name = resolve_moe_activation(model_name)

    # Calculate local number of experts
    local_inter_size = inter_size // moe_tp_size
    expert_map_result = determine_expert_map(moe_ep_size, 0, num_experts)
    if isinstance(expert_map_result, tuple) and len(expert_map_result) == 3:
        local_num_experts, expert_map, _ = expert_map_result
    else:
        # Backward compatibility with older determine_expert_map signatures
        # that return only (local_num_experts, expert_map)
        local_num_experts, expert_map = expert_map_result  # type: ignore[misc]

    if use_mxfp4:
        if aic_debug:
            print(f"Using xpu_fused_moe (is_mxfp4=True) for {model_name}")
        w1, w2, w13_scales, w2_scales, w13_bias, w2_bias, local_num_experts, padded_hidden = create_mxfp4_weights_xpu(
            num_experts, hidden_size, inter_size, moe_tp_size, moe_ep_size, device
        )
    else:
        padded_hidden = hidden_size
        w13_scales = w2_scales = None
        w13_bias = w2_bias = None

        # Create weight tensors in xpu_fused_moe layout
        # w1: [num_experts, 2 * inter_size, hidden_size]
        # w2: [num_experts, hidden_size, inter_size]
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

        if is_fp8:
            w1, w13_scales = quantize_fp8_per_expert(w1)
            w2, w2_scales = quantize_fp8_per_expert(w2)

    # Performance testing for each token count
    for num_tokens_idx, num_tokens in enumerate(num_tokens_lists):
        print("num_tokens", num_tokens)
        print("topk", topk)

        # bfloat16 + padded hidden for mxfp4; float16 + original hidden otherwise
        hs_dtype = torch.bfloat16 if use_mxfp4 else torch.float16
        hidden_states = torch.randn([num_tokens, padded_hidden], dtype=hs_dtype, device=device)

        # Generate topk_weights and topk_ids
        num_iter = 5 if distributed == "power_law" else 1
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
                if not use_mxfp4:
                    # non-mxfp4 path: xpu topk weights must be fp32
                    logits = logits.to(torch.float32)
                weights, ids = torch.topk(logits, topk, dim=-1)
                topk_weights_list.append(F.softmax(weights, dim=-1).float())
                topk_ids_list.append(ids)

            print("actual num_tokens: ", [topk_ids.shape[0] for topk_ids in topk_ids_list])

        elif distributed == "balanced":
            actual_logits = balanced_logits(num_tokens, num_experts, topk).half().to(device)
            topk_weights, topk_ids = torch.topk(actual_logits, topk, dim=-1)
            topk_weights = F.softmax(topk_weights, dim=-1).float()

        else:
            raise ValueError(f"Unsupported distributed mode: {distributed}")

        num_warmups = 3
        num_runs = 6
        if distributed == "power_law":
            num_warmups = 1
            num_runs = 1

        def run_single_iteration():
            if distributed == "power_law":
                for i, (tw, ti) in enumerate(zip(topk_weights_list, topk_ids_list, strict=True)):
                    local_num_tokens = tw.shape[0]
                    if use_mxfp4:
                        _ = xpu_fused_moe(
                            hidden_states=hidden_states[:local_num_tokens],
                            w13=w1,
                            w13_scales=w13_scales,
                            w13_bias=w13_bias,
                            w2=w2,
                            w2_scales=w2_scales,
                            w2_bias=w2_bias,
                            topk_weights=tw,
                            topk_ids=ti,
                            n_experts_per_token=topk,
                            activation=activation_name,
                            num_experts=local_num_experts,
                            is_mxfp4=True,
                        )
                    else:
                        _ = xpu_fused_moe(
                            hidden_states=hidden_states[:local_num_tokens],
                            w13=w1,
                            w13_scales=w13_scales,
                            w13_bias=None,
                            w2=w2,
                            w2_scales=w2_scales,
                            w2_bias=None,
                            topk_weights=tw,
                            topk_ids=ti,
                            n_experts_per_token=topk,
                            activation=activation_name,
                            num_experts=local_num_experts,
                            is_fp8=is_fp8,
                        )
            else:
                if use_mxfp4:
                    _ = xpu_fused_moe(
                        hidden_states=hidden_states,
                        w13=w1,
                        w13_scales=w13_scales,
                        w13_bias=w13_bias,
                        w2=w2,
                        w2_scales=w2_scales,
                        w2_bias=w2_bias,
                        topk_weights=topk_weights,
                        topk_ids=topk_ids,
                        n_experts_per_token=topk,
                        activation=activation_name,
                        num_experts=local_num_experts,
                        is_mxfp4=True,
                    )
                else:
                    _ = xpu_fused_moe(
                        hidden_states=hidden_states,
                        w13=w1,
                        w13_scales=w13_scales,
                        w13_bias=None,
                        w2=w2,
                        w2_scales=w2_scales,
                        w2_bias=None,
                        topk_weights=topk_weights,
                        topk_ids=topk_ids,
                        n_experts_per_token=topk,
                        activation=activation_name,
                        num_experts=local_num_experts,
                        is_fp8=is_fp8,
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
            latency, power_stats = run_iterations()
        except torch.OutOfMemoryError:
            # If OOM, check if we had at least one successful run.
            if num_tokens_idx > 0:
                break
            raise

        print(f"moe latency: {latency}")

        source = "vllm_xpu_moe_mxfp4" if use_mxfp4 else "vllm_xpu_moe"

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
            device_name=get_device_module().get_device_name(),
            op_name="moe",
            kernel_source=source,
            perf_filename=perf_filename,
            power_stats=power_stats,
        )


def round_up(x: int, y: int) -> int:
    """Round up x to the nearest multiple of y."""
    return ((x + y - 1) // y) * y


def create_mxfp4_weights_xpu(
    num_experts,
    hidden_size,
    inter_size,
    moe_tp_size,
    moe_ep_size,
    device,
):
    """
    Create fake MXFP4 weights for XPU benchmarking.

    On XPU, weights stay in raw uint8 format (no Marlin repacking).
    xpu_fused_moe handles the MXFP4 dequantisation internally.
    Padding: hidden_size -> round_up(128), inter_size -> round_up(128).
    """
    mxfp4_block = 32
    local_inter_size = inter_size // moe_tp_size

    padded_inter = round_up(local_inter_size, 128)
    padded_hidden = round_up(hidden_size, 128)

    # Determine local number of experts for EP
    expert_map_result = determine_expert_map(moe_ep_size, 0, num_experts)
    if isinstance(expert_map_result, tuple) and len(expert_map_result) == 3:
        local_num_experts, expert_map, _ = expert_map_result
    else:
        local_num_experts, expert_map = expert_map_result

    # w13 = fused gate_up_proj: [local_experts, 2*inter, hidden//2] (packed uint8)
    w13 = torch.randint(
        0, 255, (local_num_experts, 2 * padded_inter, padded_hidden // 2), dtype=torch.uint8, device=device
    )
    # w2 = down_proj: [local_experts, hidden, inter//2] (packed uint8)
    w2 = torch.randint(0, 255, (local_num_experts, padded_hidden, padded_inter // 2), dtype=torch.uint8, device=device)

    # Scales: [local_experts, n_dim, k_dim // mxfp4_block]
    w13_scales = torch.randint(
        1, 255, (local_num_experts, 2 * padded_inter, padded_hidden // mxfp4_block), dtype=torch.uint8, device=device
    )
    w2_scales = torch.randint(
        1, 255, (local_num_experts, padded_hidden, padded_inter // mxfp4_block), dtype=torch.uint8, device=device
    )

    # Biases (GPT-OSS uses biased SwiGLU)
    w13_bias = torch.randn(local_num_experts, 2 * padded_inter, dtype=torch.bfloat16, device=device)
    w2_bias = torch.randn(local_num_experts, padded_hidden, dtype=torch.bfloat16, device=device)

    return w13, w2, w13_scales, w2_scales, w13_bias, w2_bias, local_num_experts, padded_hidden


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

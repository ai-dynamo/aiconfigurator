# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import os

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
from vllm.version import __version__ as vllm_version

from helper import get_sm_version, log_perf

# def get_sm_version():
#     return 86

# def log_perf(*args, **kwargs):
#     pass

aic_debug = int(os.getenv("aic_moe_debug", "0"))  # noqa: SIM112


def balanced_logits(num_tokens, num_experts, topk):
    """Generate balanced distribution router logits"""
    h_selected_experts = -torch.ones([num_tokens, topk])
    stride = math.ceil(num_experts / topk)

    for token_i in range(num_tokens):
        for i in range(topk):
            if num_tokens >= stride:
                h_selected_experts[token_i][i] = (token_i + i * stride) % num_experts
            else:
                h_selected_experts[token_i][i] = (token_i * stride / num_tokens + i * stride) % num_experts

    expert_map = F.one_hot(h_selected_experts.long(), num_classes=num_experts).sum(1)
    router_logits = F.softmax(expert_map.half(), dim=1)
    return router_logits


def sample_power_law(size, alpha, xmin, xmax):
    """Sample from power law distribution"""
    u = torch.rand(size)
    inv_cdf = ((xmax ** (1 - alpha) - xmin ** (1 - alpha)) * u + xmin ** (1 - alpha)) ** (1 / (1 - alpha))
    return inv_cdf


def power_law_logits_v3(num_tokens, num_experts, topk, ep, alpha, return_first_gpu_only=False):
    """Generate power law distributed router logits (simulating real-world load imbalance scenarios)"""
    if num_tokens * topk > num_experts:
        num_tokens_per_expert = sample_power_law(num_experts, alpha, 1, num_tokens * 0.8)
    else:
        num_tokens_per_expert = sample_power_law(num_experts, alpha, 0.01, 2)

    target_sum = num_tokens * topk

    original_distribution = num_tokens_per_expert / num_tokens_per_expert.sum()

    target_distribution = original_distribution * target_sum

    num_tokens_per_expert = torch.round(target_distribution).to(torch.int64)

    current_sum = num_tokens_per_expert.sum().item()
    delta = target_sum - current_sum
    if delta != 0:
        sorted_indices = torch.argsort(num_tokens_per_expert, descending=True)

        if delta > 0:
            for i in range(delta):
                expert_idx = sorted_indices[i % len(sorted_indices)]
                num_tokens_per_expert[expert_idx] += 1
        else:
            for i in range(-delta):
                expert_idx = sorted_indices[-(i % len(sorted_indices)) - 1]
                if num_tokens_per_expert[expert_idx] > 0:
                    num_tokens_per_expert[expert_idx] -= 1
                else:
                    num_tokens_per_expert[torch.argmax(num_tokens_per_expert)] -= 1

    if len(num_tokens_per_expert) > 1:
        sorted_tokens = torch.sort(num_tokens_per_expert, descending=True)[0]
        assert sorted_tokens[0] >= sorted_tokens[-1], "Power law distribution pattern disrupted"

    # Ensure the busiest expert group in EP dimension is placed on the first rank
    with torch.no_grad():
        conv1d = torch.nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=num_experts // ep,
            stride=num_experts // ep,
            padding=0,
            bias=False,
        )
        conv1d_weights = torch.tensor([1 for _ in range(num_experts // ep)])
        conv1d.weight.copy_(conv1d_weights)

    res = conv1d(num_tokens_per_expert.unsqueeze(0).unsqueeze(0).float())
    max_ep_idx = torch.argmax(res).item()

    if max_ep_idx != 0:
        ep_group_size = num_experts // ep
        num_tokens_per_expert_reshaped = num_tokens_per_expert.view(ep, ep_group_size)
        num_tokens_per_expert_reshaped[0], num_tokens_per_expert_reshaped[max_ep_idx] = (
            num_tokens_per_expert_reshaped[max_ep_idx].clone(),
            num_tokens_per_expert_reshaped[0].clone(),
        )
        num_tokens_per_expert = num_tokens_per_expert_reshaped.view(-1)

    revised_num_tokens = num_tokens
    revised_topk = topk
    if return_first_gpu_only:
        # Number of experts per GPU
        ep_group_size = num_experts // ep

        # How many experts will be run on the first GPU.
        # Can't exceed the number of experts per GPU.
        revised_topk = min(topk, ep_group_size)

        # Only generate token -> expert assignments for the first GPU.
        num_tokens_per_expert = num_tokens_per_expert[:ep_group_size]

        # Bump up the total number of tokens on the first GPU
        # to be a multiple of revised_topk.
        tokens_on_first_gpu = torch.sum(num_tokens_per_expert).item()
        num_extra_tokens = (revised_topk - (tokens_on_first_gpu % revised_topk)) % revised_topk
        for i in range(num_extra_tokens):
            num_tokens_per_expert[i % len(num_tokens_per_expert)] += 1
        tokens_on_first_gpu = torch.sum(num_tokens_per_expert).item()
        assert tokens_on_first_gpu % revised_topk == 0

        # Now revised_num_tokens represents only the tokens on the first GPU.
        revised_num_tokens = tokens_on_first_gpu // revised_topk

    if aic_debug == 2:
        print("num_tokens_per_expert", num_tokens_per_expert, num_tokens_per_expert.sum().item())

    _, num_tokens_per_expert_sorted_index = torch.sort(num_tokens_per_expert, descending=True)
    expert_assignments = []
    num_tokens_per_expert_sorted_index_lists = num_tokens_per_expert_sorted_index.tolist()
    for expert_id in num_tokens_per_expert_sorted_index_lists:
        expert_assignments.extend([expert_id] * num_tokens_per_expert[expert_id])

    expert_assignments = torch.tensor(expert_assignments, dtype=torch.long)
    h_selected_experts = expert_assignments.reshape(revised_topk, revised_num_tokens).T

    expert_map = F.one_hot(h_selected_experts.long(), num_classes=num_experts).sum(1)
    router_logits = F.softmax(expert_map.half(), dim=1)
    return router_logits


def get_moe_test_cases():
    """Generate MoE test cases"""
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
        20480,
        32768,
        65536,
    ]
    tp_list = [1, 2, 4, 8, 16, 32]
    ep_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    num_gpu_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    alpha_list = [1.01, 1.2]

    # Model configurations: [hidden_size, inter_size, topk, num_experts, model_name]
    model_config_list = [
        [4096, 14336, 2, 8, "MOE_Mixtral8x7B"],  # mixtral_8x7b
        [6144, 16384, 2, 8, "MOE_Mixtral8x22B"],  # mixtral_8x22b
        [7168, 2048, 8, 256, "DEEPSEEK_V3"],  # deepseekv3
        [2048, 768, 8, 128, "QWEN3_30B_A3B"],  # qwen3-moe, 30b-a3b
        [4096, 1536, 8, 128, "QWEN3_235B"],  # qwen3-moe, 235b-a22b
        [6144, 2560, 8, 160, "QWEN3_480B"],  # qwen3-moe, 480b-a35b
        [7168, 2048, 8, 384, "KIMI_K2"],  # kimi k2
    ]

    # Quantization types supported by vLLM
    moe_list = ["float16"]

    if get_sm_version() > 86:
        moe_list += ["fp8"]

    test_cases = []

    for num_gpu in num_gpu_list:
        for moe_type in moe_list:
            for model_config in model_config_list:
                hs, inter_s, topk, num_experts, model_name = model_config
                for tp in tp_list:
                    # QWEN3_30B_A3B: exclude tp >= 8 as they are not used in actual deployments
                    if model_name == "QWEN3_30B_A3B" and tp >= 8:
                        continue
                    for ep in ep_list:
                        if tp * ep != num_gpu:
                            continue
                        if ep > num_experts:
                            continue
                        if num_experts % ep != 0:
                            continue
                        # Ensure inter_s can be divided by tp
                        if inter_s % tp != 0:
                            continue

                        # vllm does not support TP when EP is enabled.
                        if tp > 1 and ep > 1:
                            continue

                        for power_law_alpha in alpha_list:
                            test_cases.append(
                                [
                                    moe_type,
                                    num_tokens,
                                    hs,
                                    inter_s,
                                    topk,
                                    num_experts,
                                    tp,
                                    ep,
                                    model_name,
                                    "moe_perf.txt",
                                    "power_law",
                                    power_law_alpha,
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

    if moe_type == "fp8":
        w1_scale = torch.randn(num_experts, dtype=torch.float32)
        w2_scale = torch.randn(num_experts, dtype=torch.float32)
        a1_scale = torch.randn(1, dtype=torch.float32)
        a2_scale = torch.randn(1, dtype=torch.float32)
        block_shape = None

        quant_config = fp8_w8a8_moe_quant_config(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=block_shape,
        )
        dtype = torch.float8_e4m3fn

    # Calculate local number of experts
    local_num_experts = num_experts // moe_ep_size
    local_inter_size = inter_size // moe_tp_size

    # How many experts will be run on this GPU
    local_topk = min(topk, local_num_experts)

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

    if dtype == torch.float8_e4m3fn:
        w1 = w1.to(dtype)
        w2 = w2.to(dtype)

    # Performance testing for each token count
    for num_tokens_idx, num_tokens in enumerate(num_tokens_lists):
        print("num_tokens", num_tokens)
        print("topk", topk)
        hidden_states = torch.randn([num_tokens, hidden_size]).half().to(device)

        # Generate topk_weights and topk_ids
        num_iter = 10 if distributed == "power_law" else 1
        if distributed == "power_law":
            topk_weights_list = []
            topk_ids_list = []

            for _ in range(num_iter):
                logits = (
                    power_law_logits_v3(
                        num_tokens, num_experts, topk, moe_ep_size, power_law_alpha, return_first_gpu_only=True
                    )
                    .half()
                    .to(device)
                )
                weights, ids = torch.topk(logits, local_topk, dim=-1)
                topk_weights_list.append(F.softmax(weights, dim=-1))
                topk_ids_list.append(ids)

            print("actual num_tokens: ", [topk_ids.shape[0] for topk_ids in topk_ids_list])

        elif distributed == "balanced":
            local_num_tokens = math.ceil(num_tokens / moe_ep_size)
            actual_logits = balanced_logits(local_num_tokens, local_num_experts, local_topk).half().to(device)
            topk_weights, topk_ids = torch.topk(actual_logits, local_topk, dim=-1)
            topk_weights = F.softmax(topk_weights, dim=-1)
            print("actual num_tokens: ", actual_logits.shape[0])

        else:
            raise ValueError(f"Unsupported distributed mode: {distributed}")

        num_warmups = 3
        num_runs = 6
        if distributed == "power_law":
            num_warmups = 1
            num_runs = 1

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
                        inplace=True,
                        quant_config=quant_config,
                    )
            else:
                _ = fused_experts(
                    hidden_states,
                    w1,
                    w2,
                    topk_weights,
                    topk_ids,
                    inplace=True,
                    quant_config=quant_config,
                )

        def run_iterations(use_cuda_graph=False):
            g = torch.cuda.CUDAGraph()
            if use_cuda_graph:
                # CUDA Graph capture
                with torch.cuda.graph(g):
                    run_single_iteration()

            # Warmup
            for i in range(num_warmups):
                if use_cuda_graph:
                    g.replay()
                else:
                    run_single_iteration()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # Performance measurement
            start_event.record()
            for i in range(num_runs):
                if use_cuda_graph:
                    g.replay()
                else:
                    run_single_iteration()
            end_event.record()
            torch.cuda.synchronize()

            return start_event.elapsed_time(end_event) / num_runs / num_iter

        try:
            latency = run_iterations(use_cuda_graph=False)
        except torch.OutOfMemoryError:
            # If OOM, check if we had at least one successful run.
            if num_tokens_idx > 0:
                break
            raise

        print(f"moe latency: {latency}")

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
        )


if __name__ == "__main__":
    test_cases = get_moe_test_cases()
    print(f"Total test cases: {len(test_cases)}")

    for test_case in test_cases:
        print(f"Running test case: {test_case}")
        try:
            run_moe_torch(*test_case)
        except Exception as e:
            print(f"Test case failed: {test_case}")
            print(f"Error: {e}")
            continue

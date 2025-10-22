# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math
import os
from typing import TypedDict

import pkg_resources
import torch
import torch.nn.functional as F
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
    fused_moe,
    get_config_dtype_str,
    get_default_config,
    get_moe_configs,
)
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.utils import is_hip

from helper import log_perf

_is_hip = is_hip()


def get_moe_test_cases():
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
    ]
    tp_list = [1, 2, 4, 8, 16, 32]
    ep_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    num_gpu_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    alpha_list = [1.01, 1.2]
    # hidden_size,inter_s,topk,num_expert, gated act
    # [15360,30720,2,16],# GPT-MOE-1.8T
    # [15360,3840,16,128],# GPT-MOE-1.8T-FineGrained
    # [3584,2560,8,64],# Qwen2-57B
    # [2048,1408,4,60], #qwen1.5_moe
    # [2048,1408,6,64], #deepseekv1_moe
    # [5120,1536,6,160], #deepseekv2
    model_config_list = [
        [4096, 14336, 2, 8, "MOE_Mixtral8x7B"],  # mixtral_8x7b
        [6144, 16384, 2, 8, "MOE_Mixtral8x22B"],  # mixtral_8x22b
        [7168, 2048, 8, 256, "DEEPSEEK_V3"],  # deepseekv3, will have 1 shared expert
        [2048, 768, 8, 128, "QWEN3_30B_A3B"],  # qwen3-moe, 30b-a3b
        [4096, 1536, 8, 128, "QWEN3_235B"],  # qwen3-moe, 235b-a22b
        [6144, 2560, 8, 160, "QWEN3_480B"],  # qwen3-moe, 480b-a35b
        [7168, 2048, 8, 384, "KIMI_K2"],  # kimi k2
    ]
    moe_list = ["float16", "fp8_block"]

    test_cases = []

    for num_gpu in num_gpu_list:  # starting from fewer gpus. workaround for potential buffer bug in moe impl.
        for moe_type in moe_list:
            for num_token in num_tokens:
                for model_config in model_config_list:
                    hs, inter_s, topk, num_experts, model_name = model_config
                    for tp in tp_list:
                        # QWEN3_30B_A3B: exclude tp >= 8 as they are not used for actual deployments
                        if model_name == "QWEN3_30B_A3B" and tp >= 8:
                            continue
                        for ep in ep_list:
                            if tp * ep != num_gpu:
                                continue
                            if ep > num_experts:
                                continue
                            # we need to ensure inter_s can be divided by tp.
                            if inter_s % tp != 0:
                                continue
                            for power_law_alpha in alpha_list:
                                test_cases.append(
                                    [
                                        moe_type,
                                        num_token,
                                        hs,
                                        inter_s,
                                        topk,
                                        num_experts,
                                        tp,
                                        ep,
                                        False,
                                        model_name,
                                        "moe_perf.txt",
                                        "power_law",
                                        power_law_alpha,
                                    ]
                                )
                            # test_cases.append(
                            #     [
                            #         moe_type,
                            #         num_token,
                            #         hs,
                            #         inter_s,
                            #         topk,
                            #         num_experts,
                            #         tp,
                            #         ep,
                            #         False,
                            #         model_name,
                            #         "moe_perf.txt",
                            #         "balanced",
                            #         0,
                            #     ]
                            # )
                            # test_cases.append(
                            #     [
                            #         moe_type,
                            #         num_token,
                            #         hs,
                            #         inter_s,
                            #         topk,
                            #         num_experts,
                            #         tp,
                            #         ep,
                            #         False,
                            #         model_name,
                            #         "moe_perf.txt",
                            #         "uniform",
                            #         0,
                            #     ]
                            # )
    return test_cases


def balanced_logits(num_tokens, num_experts, topk):
    h_selected_experts = -torch.ones([num_tokens, topk])
    stride = math.ceil(num_experts / topk)

    for token_i in range(num_tokens):
        for i in range(topk):
            if num_tokens >= stride:
                h_selected_experts[token_i][i] = (token_i + i * stride) % num_experts
            else:
                h_selected_experts[token_i][i] = (token_i * stride / num_tokens + i * stride) % num_experts

    expert_map = F.one_hot(h_selected_experts.long(), num_classes=num_experts).sum(1)
    router_logits = F.softmax(expert_map.bfloat16(), dim=1)
    return router_logits


def sample_power_law(size, alpha, xmin, xmax):
    u = torch.rand(size)
    inv_cdf = ((xmax ** (1 - alpha) - xmin ** (1 - alpha)) * u + xmin ** (1 - alpha)) ** (1 / (1 - alpha))
    return inv_cdf


def power_law_logits_v3(num_tokens, num_experts, topk, ep, alpha):
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

    pet_debug = int(os.getenv("PET_DEBUG", "0"))
    if pet_debug == 1:
        print("num_tokens_per_expert", num_tokens_per_expert, num_tokens_per_expert.sum().item())

    _, num_tokens_per_expert_sorted_index = torch.sort(num_tokens_per_expert, descending=True)
    expert_assignments = []
    num_tokens_per_expert_sorted_index_lists = num_tokens_per_expert_sorted_index.tolist()
    for expert_id in num_tokens_per_expert_sorted_index_lists:
        expert_assignments.extend([expert_id] * num_tokens_per_expert[expert_id])

    expert_assignments = torch.tensor(expert_assignments, dtype=torch.long)
    h_selected_experts = expert_assignments.reshape(topk, num_tokens).T

    expert_map = F.one_hot(h_selected_experts.long(), num_classes=num_experts).sum(1)
    router_logits = F.softmax(expert_map.bfloat16(), dim=1)
    return router_logits


class BenchmarkConfig(TypedDict):
    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    BLOCK_SIZE_K: int
    GROUP_SIZE_M: int
    num_warps: int
    num_stages: int


def benchmark_config(
    config: BenchmarkConfig,
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    block_shape: list[int] | None = None,
    num_iters: int = 10,
    distributed: str = "power_law",
    power_law_alpha: float = 0,
) -> float:
    init_dtype = torch.float16 if use_fp8_w8a8 else dtype
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    if use_int8_w8a16 or use_int8_w8a8:
        w1 = torch.randint(
            -127,
            127,
            (
                num_experts,
                shard_intermediate_size,
                hidden_size,
            ),
            dtype=torch.int8,
        )
        w2 = torch.randint(
            -127,
            127,
            (
                num_experts,
                hidden_size,
                shard_intermediate_size // 2,
            ),
            dtype=torch.int8,
        )
    else:
        w1 = torch.randn(num_experts, shard_intermediate_size, hidden_size, dtype=init_dtype)
        w2 = torch.randn(num_experts, hidden_size, shard_intermediate_size // 2, dtype=init_dtype)
    if distributed == "uniform":
        gating_output = torch.randn(num_iters, num_tokens, num_experts, dtype=torch.float32)
    elif distributed == "balanced":
        gating_output = [balanced_logits(num_tokens, num_experts, topk) for _ in range(num_iters)]
    elif distributed == "power_law":
        # only support ep=1 for sglang
        gating_output = [
            power_law_logits_v3(num_tokens, num_experts, topk, 1, power_law_alpha) for _ in range(num_iters)
        ]
    else:
        raise ValueError(f"Unsupported distributed mode: {distributed}")

    w1_scale = None
    w2_scale = None
    a1_scale = None
    a2_scale = None
    if use_int8_w8a16:
        w1_scale = torch.randn((num_experts, 2 * shard_intermediate_size), dtype=torch.float32)
        w2_scale = torch.randn((hidden_size, num_experts), dtype=torch.float32)
    if use_fp8_w8a8 or use_int8_w8a8:
        if use_int8_w8a8 and block_shape is None:
            w1_scale = torch.randn(num_experts, shard_intermediate_size, dtype=torch.float32)
            w2_scale = torch.randn(num_experts, hidden_size, dtype=torch.float32)
        elif block_shape is None:
            w1_scale = torch.randn(num_experts, dtype=torch.float32)
            w2_scale = torch.randn(num_experts, dtype=torch.float32)
            a1_scale = torch.randn(1, dtype=torch.float32)
            a2_scale = torch.randn(1, dtype=torch.float32)
        else:
            block_n, block_k = block_shape[0], block_shape[1]
            n_tiles_w1 = (shard_intermediate_size + block_n - 1) // block_n
            n_tiles_w2 = (hidden_size + block_n - 1) // block_n
            k_tiles_w1 = (hidden_size + block_k - 1) // block_k
            k_tiles_w2 = (shard_intermediate_size // 2 + block_k - 1) // block_k
            w1_scale = torch.rand((num_experts, n_tiles_w1, k_tiles_w1), dtype=torch.float32)
            w2_scale = torch.rand((num_experts, n_tiles_w2, k_tiles_w2), dtype=torch.float32)

    if use_fp8_w8a8:
        w1 = w1.to(torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn)
        w2 = w2.to(torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn)

    input_gating = torch.randn(num_tokens, num_experts, dtype=torch.float32)
    topk_output = select_experts(x, input_gating, TopKConfig(top_k=topk))

    def prepare(i: int):
        input_gating = gating_output[i]
        new_topk_output = select_experts(x, input_gating, TopKConfig(top_k=topk))
        topk_output.topk_weights.copy_(new_topk_output.topk_weights)
        topk_output.topk_ids.copy_(new_topk_output.topk_ids)
        topk_output.router_logits.copy_(new_topk_output.router_logits)

    def run():
        from sglang.srt.layers.moe.fused_moe_triton import override_config

        with override_config(config):
            fused_moe(
                x,
                w1,
                w2,
                topk_output,
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a8=use_int8_w8a8,
                use_int8_w8a16=use_int8_w8a16,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                block_shape=block_shape,
            )

    # JIT compilation & warmup
    run()
    torch.cuda.synchronize()

    # Capture 10 invocations with CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            run()
    torch.cuda.synchronize()

    # Warmup
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies: list[float] = []
    for i in range(num_iters):
        prepare(i)
        torch.cuda.synchronize()

        start_event.record()
        graph.replay()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    avg = sum(latencies) / (num_iters * 10)
    graph.reset()
    return avg


def benchmark(
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    block_shape: list[int],
    distributed: str = "power_law",
    power_law_alpha: float = 0,
) -> tuple[dict[str, int], float]:
    torch.cuda.manual_seed_all(0)
    dtype_str = get_config_dtype_str(dtype, use_int8_w8a16=use_int8_w8a16, use_fp8_w8a8=use_fp8_w8a8)
    # NOTE(woosuk): The current naming convention uses w2.shape[2], which
    # is the intermediate size after silu_and_mul.
    block_n = block_shape[0] if block_shape else 0
    block_k = block_shape[1] if block_shape else 0
    op_config = get_moe_configs(num_experts, shard_intermediate_size // 2, dtype_str, block_n, block_k)
    if op_config is None:
        config = get_default_config(
            num_tokens,
            num_experts,
            shard_intermediate_size,
            hidden_size,
            topk,
            dtype_str,
            False,
            block_shape,
        )
    else:
        config = op_config[min(op_config.keys(), key=lambda x: abs(x - num_tokens))]
    kernel_time = benchmark_config(
        config,
        num_tokens,
        num_experts,
        shard_intermediate_size,
        hidden_size,
        topk,
        dtype,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        block_shape,
        distributed=distributed,
        power_law_alpha=power_law_alpha,
    )
    return kernel_time


def run_moe_torch(
    moe_type,
    num_tokens,
    hidden_size,
    inter_size,
    topk,
    num_experts,
    moe_tp_size,
    moe_ep_size,
    cutlass_min_latency_mode,
    model_name,
    perf_filename,
    distributed="power_law",
    power_law_alpha=0,
    device="cuda:0",
):
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    assert moe_ep_size == 1, "only support moe ep size = 1"
    assert moe_type == "fp8_block" or moe_type == "float16", "only support moe type = fp8_block or float16"
    assert inter_size % moe_tp_size == 0, "inter_size % moe_tp_size must be 0"

    latency = benchmark(
        num_tokens,
        num_experts,
        2 * inter_size // moe_tp_size,
        hidden_size,
        topk,
        torch.bfloat16,
        moe_type == "fp8_block",
        False,
        False,
        None,
        distributed=distributed,
        power_law_alpha=power_law_alpha,
    )

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
        framework="SGLang",
        version=pkg_resources.get_distribution("sglang").version,
        device_name=torch.cuda.get_device_name(device),
        op_name="moe",
        kernel_source="sglang_fused_moe_triton",
        perf_filename=perf_filename,
    )

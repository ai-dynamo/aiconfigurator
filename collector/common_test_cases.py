# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import itertools
from typing import Optional


@dataclasses.dataclass
class MoeCommonTestCase:
    num_tokens_list: list[int]
    hidden_size: int
    inter_size: int
    topk: int
    num_experts: int
    tp: int
    ep: int
    model_name: str
    token_expert_distribution: str
    power_law_alpha: Optional[float]


def get_common_moe_test_cases():
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

    token_distributions = [
        ("balanced", 0.0),
        ("power_law", 1.01),
        ("power_law", 1.2),
    ]

    # alpha_list = [1.01, 1.2]
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
        [2880, 2880, 4, 128, "GPT_OSS_120B"],
        [2880, 2880, 4, 32, "GPT_OSS_20B"],
    ]

    test_cases: list[MoeCommonTestCase] = []

    for (
        num_gpu,  # starting from fewer gpus. workaround for potential buffer bug in moe impl.
        model_config,
        tp,
        ep,
        (token_distribution, power_law_alpha),
    ) in itertools.product(
        num_gpu_list,
        model_config_list,
        tp_list,
        ep_list,
        token_distributions,
    ):
        hs, inter_s, topk, num_experts, model_name = model_config

        # QWEN3_30B_A3B: exclude tp >= 8 as they are not used for actual deployments
        if model_name == "QWEN3_30B_A3B" and tp >= 8:
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

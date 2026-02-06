# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import itertools
from typing import Optional

import torch


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
    if torch.xpu.is_available():
        # narrow down the search space for xpu currently
        num_tokens = [ntok for ntok in num_tokens if ntok < 128]
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
        [4096, 14336, 2, 8, "mistralai/Mixtral-8x7B-v0.1"],  # mixtral_8x7b
        [6144, 16384, 2, 8, "mistralai/Mixtral-8x22B-v0.1"],  # mixtral_8x22b
        [7168, 2048, 8, 256, "deepseek-ai/DeepSeek-V3"],  # deepseekv3, will have 1 shared expert
        [2048, 768, 8, 128, "Qwen/Qwen3-30B-A3B"],  # qwen3-moe, 30b-a3b
        [4096, 1536, 8, 128, "Qwen/Qwen3-235B-A22B"],  # qwen3-moe, 235b-a22b
        [6144, 2560, 8, 160, "Qwen/Qwen3-Coder-480B-A35B-Instruct"],  # qwen3-moe, 480b-a35b
        [7168, 2048, 8, 384, "moonshotai/Kimi-K2-Instruct"],  # kimi k2
        [2880, 2880, 4, 128, "openai/gpt-oss-120b"],
        [2880, 2880, 4, 32, "openai/gpt-oss-20b"],
        [2688, 1856, 6, 128, "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"],  # nemotron-3 nano (uses relu2, non-gated)
        [
            4096,
            2688,
            22,
            512,
            "nvidia/NVIDIA-Nemotron-3-Super-120B-NVFP4-FP8KV",
        ],  # nemotron-3 super (uses relu2, non-gated)
    ]
    if torch.xpu.is_available():
        # More configs will be validated/added soon for XPU
        model_config_list = [
            [2048, 1408, 4, 60, "QWEN1.5_MOE"],
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


@dataclasses.dataclass
class GemmCommonTestCase:
    x: int
    n: int
    k: int


def get_gemm_common_test_cases() -> list[GemmCommonTestCase]:
    x_list = [
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
        384,
        512,
        768,
        1024,
        2048,
        4096,
        8192,
    ]
    nk_list = [
        32,
        64,
        128,
        256,
        512,
        768,
        1024,
        1536,
        2048,
        2560,
        3072,
        3584,
        4096,
        5120,
        6144,
        7168,
        8192,
        10240,
        12288,
    ]
    if torch.cuda.is_available():
        nk_list_ext = [16384, 65536]  # for coverage and interp purpose
    elif torch.xpu.is_available():  # narrow down the search space for xpu currently
        nk_list_ext = []

    test_cases = []
    # x_list_orig+add+ext  <==> nk_list+ext
    for x in sorted(x_list, reverse=True):
        for n in sorted(nk_list + nk_list_ext, reverse=True):
            for k in sorted(nk_list + nk_list_ext, reverse=True):
                if n * k == 65536 * 65536:
                    continue
                test_cases.append(GemmCommonTestCase(x=x, n=n, k=k))

    return test_cases


@dataclasses.dataclass
class MLACommonTestCase:
    num_heads: int
    batch_size: int
    input_len: int
    is_context_phase: bool
    kv_cache_block_size: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    model_name: str


def _get_mla_common_test_cases(is_context: bool):
    test_cases = []

    # num_heads, q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim
    model_config_list = [
        [128, 1536, 512, 128, 64, 128, "deepseek-ai/DeepSeek-V3"],
    ]

    if is_context:
        b_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        s_list = [1, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384, 32768]
    else:
        b_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        s_list = [
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
            131072,
        ]  # [target token s] is equivalent to [in: s-1, step=1]
    kv_cache_block_size_list = [64]

    for (
        s,
        b,
        kv_cache_block_size,
        model_config,
    ) in itertools.product(
        s_list,
        b_list,
        kv_cache_block_size_list,
        model_config_list,
    ):
        if is_context:
            if b * s > 65536:
                continue
        else:
            if b * s > 1024 * 4096 * 2 * 2:
                continue

        test_cases.append(
            MLACommonTestCase(
                num_heads=model_config[0],
                input_len=s if is_context else s - 1,
                batch_size=b,
                is_context_phase=is_context,
                kv_cache_block_size=kv_cache_block_size,
                q_lora_rank=model_config[1],
                kv_lora_rank=model_config[2],
                qk_nope_head_dim=model_config[3],
                qk_rope_head_dim=model_config[4],
                v_head_dim=model_config[5],
                model_name=model_config[6],
            )
        )

    return test_cases


def get_context_mla_common_test_cases():
    return _get_mla_common_test_cases(is_context=True)


def get_generation_mla_common_test_cases():
    return _get_mla_common_test_cases(is_context=False)

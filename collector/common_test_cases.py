# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    # Qwen3-32B targeted test cases for FFN1/FFN2 gap analysis.
    # Model config: hidden_size=5120, intermediate_size=25600, ISL=512.
    #
    # gate_ffn1 GEMM shape (from LLAMAModel):
    #   N = 2 * intermediate_size / tp_size  (fused gate+up projection)
    #   K = hidden_size = 5120
    #
    # ffn2 GEMM shape (from LLAMAModel):
    #   N = hidden_size = 5120
    #   K = intermediate_size / tp_size  (down projection)
    #
    # TP=1: gate_ffn1 M=512,N=51200,K=5120 | ffn2 M=512,N=5120,K=25600
    # TP=2: gate_ffn1 M=512,N=25600,K=5120 | ffn2 M=512,N=5120,K=12800
    num_repeats = 10
    shapes = [
        (512, 51200, 5120),  # TP=1 gate_ffn1
        (512, 5120, 25600),  # TP=1 ffn2
        (512, 25600, 5120),  # TP=2 gate_ffn1
        (512, 5120, 12800),  # TP=2 ffn2
    ]
    return [GemmCommonTestCase(x=x, n=n, k=k) for x, n, k in shapes for _ in range(num_repeats)]


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


# =============================================================================
# Mamba2 SSM Test Cases
# =============================================================================


@dataclasses.dataclass
class Mamba2CommonTestCase:
    """Test case configuration for Mamba2 SSM benchmarking."""

    phase: str  # "context" or "generation"
    d_model: int  # hidden_size
    d_state: int  # SSM state dimension
    d_conv: int  # Conv1d kernel size
    nheads: int  # Number of Mamba heads
    head_dim: int  # Dimension per head
    n_groups: int  # Number of groups for B, C matrices
    chunk_size: int  # Chunk size for SSM scan
    num_tokens_list: Optional[list[int]]  # For context phase (continuous batching)
    batch_size_list: Optional[list[int]]  # For generation phase, or context static batching
    seq_len_list: Optional[list[int]]  # For context phase with static batching
    model_name: str


def get_common_mamba2_test_cases() -> list[Mamba2CommonTestCase]:
    """
    Generate common test cases for Mamba2 SSM benchmarking.

    Includes configurations for:
    - Nemotron-H 3-30B (primary target)
    - Other potential Mamba2-based models

    Returns:
        List of Mamba2CommonTestCase configurations
    """
    test_cases: list[Mamba2CommonTestCase] = []

    # Sequence lengths for context (prefill) phase
    context_seq_lens = [
        1,
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
    ]

    # Batch sizes for context phase
    context_batch_sizes = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
    ]

    # Batch sizes for generation (decode) phase
    generation_batch_sizes = [
        1,
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
    ]

    # Model configurations:
    # [d_model, d_state, d_conv, nheads, head_dim, n_groups, chunk_size, model_name]
    model_config_list = [
        # Nemotron-H 3-Nano
        # hidden_size=2688, ssm_state_size=128, conv_kernel=4,
        # mamba_num_heads=64, mamba_head_dim=64, n_groups=8, chunk_size=128
        [2688, 128, 4, 64, 64, 8, 128, "NEMOTRON_H_3_Nano"],
        # Nemotron-H 3-Super
        # hidden_size=4096, ssm_state_size=128, conv_kernel=4,
        # mamba_num_heads=128, mamba_head_dim=64, n_groups=8, chunk_size=128
        [4096, 128, 4, 128, 64, 8, 128, "NEMOTRON_H_3_Super"],
        # Generic Mamba2 configuration for interpolation coverage
        [8192, 128, 4, 64, 64, 8, 256, "MAMBA2_GENERIC_4K"],
        [1024, 64, 4, 16, 64, 4, 128, "MAMBA2_GENERIC_1K"],
    ]

    for model_config in model_config_list:
        d_model, d_state, d_conv, nheads, head_dim, n_groups, chunk_size, model_name = model_config

        # Context (prefill) test case
        test_cases.append(
            Mamba2CommonTestCase(
                phase="context",
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                nheads=nheads,
                head_dim=head_dim,
                n_groups=n_groups,
                chunk_size=chunk_size,
                num_tokens_list=None,  # Not used for static batching
                batch_size_list=context_batch_sizes,
                seq_len_list=context_seq_lens,
                model_name=model_name,
            )
        )

        # Generation (decode) test case
        test_cases.append(
            Mamba2CommonTestCase(
                phase="generation",
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                nheads=nheads,
                head_dim=head_dim,
                n_groups=n_groups,
                chunk_size=chunk_size,
                num_tokens_list=None,
                batch_size_list=generation_batch_sizes,
                seq_len_list=None,  # Not used for generation
                model_name=model_name,
            )
        )

    return test_cases

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import itertools
import os
import sys
from typing import Optional


def _get_model_path_filter() -> str | None:
    """Return the model-path filter from the environment, or None for 'all'."""
    val = os.environ.get("COLLECTOR_MODEL_PATH", "").strip()
    return val if val else None


def _filter_model_config_list(model_config_list: list[list]) -> list[list]:
    """Filter a model_config_list to only the entry matching COLLECTOR_MODEL_PATH.

    Each entry's last element is assumed to be the model name.
    Returns the full list when no filter is set.
    Returns an empty list when the filter doesn't match any entry in this
    particular list (the model may exist in a different op's list).
    Upfront validation against all known models is done in collect.py.
    """
    model_path = _get_model_path_filter()
    if model_path is None:
        return model_config_list
    return [cfg for cfg in model_config_list if cfg[-1] == model_path]


_WIDEEP_MOE_MODEL_NAMES: set[str] = {
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-V3.2",
    "zai-org/GLM-5",
    "MiniMaxAI/MiniMax-M2.5",
    "moonshotai/Kimi-K2-Instruct",
    "moonshotai/Kimi-K2.5",
    "nvidia/Kimi-K2.5-NVFP4",
}


def is_wideep_moe_model(model_name: str) -> bool:
    """Return True if *model_name* needs wideep MoE collection (DEEPSEEK / DEEPSEEKV32 family)."""
    return model_name in _WIDEEP_MOE_MODEL_NAMES


# Raw model config lists — module-level so get_all_model_names() can read them
# without instantiating test case objects or calling generator functions.

# MoE: [hidden_size, inter_size, topk, num_experts, model_name]
_MOE_MODEL_CONFIGS: list[list] = [
    [4096, 14336, 2, 8, "mistralai/Mixtral-8x7B-v0.1"],  # mixtral_8x7b
    [6144, 16384, 2, 8, "mistralai/Mixtral-8x22B-v0.1"],  # mixtral_8x22b
    [7168, 2048, 8, 256, "deepseek-ai/DeepSeek-V3"],  # deepseekv3, will have 1 shared expert, dsv32
    [6144, 2048, 8, 256, "zai-org/GLM-5"],  # glm-5 (DEEPSEEKV32 family, different hidden_size)
    [2048, 768, 8, 128, "Qwen/Qwen3-30B-A3B"],  # qwen3-moe, 30b-a3b
    [4096, 1536, 8, 128, "Qwen/Qwen3-235B-A22B"],  # qwen3-moe, 235b-a22b
    [6144, 2560, 8, 160, "Qwen/Qwen3-Coder-480B-A35B-Instruct"],  # qwen3-moe, 480b-a35b
    [7168, 2048, 8, 384, "moonshotai/Kimi-K2-Instruct"],  # kimi k2
    [7168, 2048, 8, 384, "nvidia/Kimi-K2.5-NVFP4"],  # kimi k2.5 nvfp4 (same shape, separate collection for fp4 kernels)
    [3072, 1536, 8, 256, "MiniMaxAI/MiniMax-M2.5"],  # minimax m2.5 (also covers nvidia/MiniMax-M2.5-NVFP4)
    [2880, 2880, 4, 128, "openai/gpt-oss-120b"],
    [2880, 2880, 4, 32, "openai/gpt-oss-20b"],
    [2688, 1856, 6, 128, "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"],  # nemotron-3 nano (uses relu2, non-gated)
    [
        4096,
        2688,
        22,
        512,
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    ],  # nemotron-3 super (uses relu2, non-gated)
    [4096, 1024, 10, 512, "Qwen/Qwen3.5-397B-A17B"],  # qwen3.5-moe, 397b-a17b
]

# MLA: [num_heads, q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, model_name]
_MLA_MODEL_CONFIGS: list[list] = [
    [128, 1536, 512, 128, 64, 128, "deepseek-ai/DeepSeek-V3"],
    [64, 1536, 512, 128, 64, 128, "moonshotai/Kimi-K2.5"],  # kimi k2.5: same MLA dims as DSV3 except num_heads=64
]

# MLA module: models from collect_mla_module.py's SUPPORTED_MODELS that are not
# already covered by _MLA_MODEL_CONFIGS above.
_MLA_MODULE_MODEL_NAMES: list[str] = [
    "deepseek-ai/DeepSeek-V3.2",
    "deepseek-ai/DeepSeek-V4-Flash",
    "zai-org/GLM-5",
]

# MHC (DeepSeek-V4 Hash-Compressed attention) module:
# [hidden_size, hc_mult, model_name]
# Only AIC-cached model_configs/<id>_config.json are usable; both sgl-project
# FP8 and deepseek-ai non-FP8 namespaces point at the same pre/post kernels.
_MHC_MODEL_CONFIGS: list[list] = [
    [4096, 4, "deepseek-ai/DeepSeek-V4-Flash"],
    [7168, 4, "deepseek-ai/DeepSeek-V4-Pro"],
    [4096, 4, "sgl-project/DeepSeek-V4-Flash-FP8"],
    [7168, 4, "sgl-project/DeepSeek-V4-Pro-FP8"],
]

# GDN (Gated DeltaNet): [d_model, d_conv, num_k_heads, head_k_dim, num_v_heads, head_v_dim, model_name]
# Covers all 8 unique dimension sets across the full Qwen3.5 collection.
# d_conv=4, head_k_dim=128, head_v_dim=128, num_k_heads=16 are constant across all models.
_GDN_MODEL_CONFIGS: list[list] = [
    [1024, 4, 16, 128, 16, 128, "Qwen/Qwen3.5-0.8B"],
    [2048, 4, 16, 128, 16, 128, "Qwen/Qwen3.5-2B"],
    [2560, 4, 16, 128, 32, 128, "Qwen/Qwen3.5-4B"],
    [4096, 4, 16, 128, 32, 128, "Qwen/Qwen3.5-9B"],
    [5120, 4, 16, 128, 48, 128, "Qwen/Qwen3.5-27B"],
    [2048, 4, 16, 128, 32, 128, "Qwen/Qwen3.5-35B-A3B"],  # same d_model as 2B, different num_v_heads
    [3072, 4, 16, 128, 64, 128, "Qwen/Qwen3.5-122B-A10B"],
    [4096, 4, 16, 128, 64, 128, "Qwen/Qwen3.5-397B-A17B"],  # same d_model as 9B, different num_v_heads
]

# Mamba2: [d_model, d_state, d_conv, nheads, head_dim, n_groups, chunk_size, model_name]
_MAMBA2_MODEL_CONFIGS: list[list] = [
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


def get_all_model_names() -> list[str]:
    """Return all known model names across all op types.

    Reads directly from the raw config list data — does not instantiate test
    case objects or call generator functions, so pruning logic in the generators
    cannot accidentally exclude models from the allowlist.
    """
    all_configs = (
        _MOE_MODEL_CONFIGS + _MLA_MODEL_CONFIGS + _MAMBA2_MODEL_CONFIGS + _GDN_MODEL_CONFIGS + _MHC_MODEL_CONFIGS
    )
    return [cfg[-1] for cfg in all_configs] + _MLA_MODULE_MODEL_NAMES


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
    model_config_list = _filter_model_config_list(_MOE_MODEL_CONFIGS)

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
    x_list = list(range(1, 16))
    x_list += list(range(16, 128, 16)) + [i + 1 for i in range(16, 128, 16)]
    x_list += list(range(128, 256, 32)) + [i + 1 for i in range(128, 256, 32)]
    for x in range(256, 4096 + 257, 256):
        x_list.append(x)
        x_list.append(x + 1)
        # after 4096, the zig-zag pattern can be ignored

    x = 8192
    while x <= 32768:
        x_list.append(x)
        x *= 2
    x_list.sort()  # sort the x_list to make it easier to debug
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
    nk_list_ext = [16384, 51200, 65536]  # for coverage and interp purpose

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
    model_config_list = _filter_model_config_list(_MLA_MODEL_CONFIGS)

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
    model_config_list = _filter_model_config_list(_MAMBA2_MODEL_CONFIGS)

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


# =============================================================================
# GDN (Gated DeltaNet) Test Cases  — Qwen3.5 linear_attention layers
# =============================================================================


@dataclasses.dataclass
class GdnCommonTestCase:
    """Test case configuration for GDN (Gated DeltaNet) kernel benchmarking."""

    phase: str  # "context" or "generation"
    d_model: int  # hidden_size
    d_conv: int  # Conv1d kernel size
    num_k_heads: int  # Number of GDN key heads
    head_k_dim: int  # Key head dimension
    num_v_heads: int  # Number of GDN value heads
    head_v_dim: int  # Value head dimension
    batch_size_list: Optional[list[int]]
    seq_len_list: Optional[list[int]]  # For context phase; None for generation
    model_name: str


# =============================================================================
# MHC (DeepSeek-V4 Hash-Compressed attention) Test Cases
# =============================================================================


@dataclasses.dataclass
class MhcCommonTestCase:
    """Test case configuration for DeepSeek-V4 mHC pre/post kernel benchmarking."""

    phase: str  # "pre" or "post"
    hidden_size: int
    hc_mult: int
    num_tokens_list: list[int]
    model_name: str


def get_common_mhc_test_cases() -> list[MhcCommonTestCase]:
    """Generate common test cases for mHC (pre/post) kernel benchmarking."""
    num_tokens_list = [
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
        49152,
        65536,
        98304,
        131072,
        262144,
        524288,
    ]

    model_config_list = _filter_model_config_list(_MHC_MODEL_CONFIGS)

    test_cases: list[MhcCommonTestCase] = []
    for model_config in model_config_list:
        hidden_size, hc_mult, model_name = model_config
        for phase in ("pre", "post"):
            test_cases.append(
                MhcCommonTestCase(
                    phase=phase,
                    hidden_size=hidden_size,
                    hc_mult=hc_mult,
                    num_tokens_list=num_tokens_list,
                    model_name=model_name,
                )
            )
    return test_cases


def get_common_gdn_test_cases() -> list[GdnCommonTestCase]:
    """
    Generate common test cases for GDN (Gated DeltaNet) kernel benchmarking.

    Covers all 8 unique dimension sets across the full Qwen3.5 collection
    for both context (prefill) and generation (decode) phases.
    """
    test_cases: list[GdnCommonTestCase] = []

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

    model_config_list = _filter_model_config_list(_GDN_MODEL_CONFIGS)

    for model_config in model_config_list:
        d_model, d_conv, num_k_heads, head_k_dim, num_v_heads, head_v_dim, model_name = model_config

        # Context (prefill) test case
        test_cases.append(
            GdnCommonTestCase(
                phase="context",
                d_model=d_model,
                d_conv=d_conv,
                num_k_heads=num_k_heads,
                head_k_dim=head_k_dim,
                num_v_heads=num_v_heads,
                head_v_dim=head_v_dim,
                batch_size_list=context_batch_sizes,
                seq_len_list=context_seq_lens,
                model_name=model_name,
            )
        )

        # Generation (decode) test case
        test_cases.append(
            GdnCommonTestCase(
                phase="generation",
                d_model=d_model,
                d_conv=d_conv,
                num_k_heads=num_k_heads,
                head_k_dim=head_k_dim,
                num_v_heads=num_v_heads,
                head_v_dim=head_v_dim,
                batch_size_list=generation_batch_sizes,
                seq_len_list=None,
                model_name=model_name,
            )
        )

    return test_cases


# ═══════════════════════════════════════════════════════════════════════
# DeepSeek-V4-Flash test cases
# ═══════════════════════════════════════════════════════════════════════
# Used by ``collector.sglang.collect_dsv4_flash_attn`` (full-module bench)
# and ``collector.sglang.deepseekv4_sparse_modules`` (sparse kernel bench).
# Both backends re-export the relevant ``get_*`` functions so collect.py
# can resolve them via getattr on each per-backend module.

_DSV4_FLASH_DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
DSV4_FLASH_ATTN_KINDS = ("swa", "csa", "hca")


def _dsv4_flash_active() -> bool:
    """Honour the ``--model-path`` (``COLLECTOR_MODEL_PATH``) filter.

    All V4-Flash test cases are V4-Flash-only by construction.  When the
    user filters to a different model, every V4-Flash op must emit zero
    cases so the collector skips it.
    """
    filt = _get_model_path_filter()
    return filt is None or filt == _DSV4_FLASH_DEFAULT_MODEL


# --- Module-level (full self_attn) sweep ---
_DSV4_FLASH_MODULE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# V4-Flash supports max_position_embeddings=1048576 (1M).  In generation
# mode the position of the new token equals the history length, so the max
# valid seq_len is 1048575 (= 1M - 1); using 1048576 indexes one past the
# rope tables and triggers cudaErrorIllegalAddress in fused_rope.
_DSV4_FLASH_MODULE_SEQ_LENGTHS = [
    1,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    10240,
    12288,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048575,
]

# TP sizes — single-process simulation via _tp_load_model_patch in
# collect_dsv4_flash_attn.  Projection ColumnParallel/RowParallel weights
# allocate at 1/N shards; FMLA always sees h_q=64 because V4 zero-pads Q.
_DSV4_FLASH_MODULE_TP_SIZES = [1, 2, 4, 8]


def _dsv4_flash_module_precision_combos(phase: str):
    """``(compute_dtype, kv_cache_dtype, gemm_type)`` triples.

    DeepseekV4ForCausalLM rejects bfloat16 KV cache (asserts at load time),
    so we only emit fp8 KV.  ``gemm_type`` switches projection dispatch:
      * ``bfloat16``  — projections through cuBLASLt nvjet kernels
      * ``fp8_block`` — fp8 block-quantised weights → DeepGEMM
                        ``sm90_fp8_gemm_1d2d_impl`` (matches production)
    """
    del phase
    return [
        ("bfloat16", "fp8", "bfloat16"),
        ("bfloat16", "fp8", "fp8_block"),
    ]


def _dsv4_flash_module_filter_pairs(mode: str, batch_sizes, seq_lens):
    """Drop ``(bs, sl)`` pairs that exceed KV pool / kernel limits.

    Context (b * s):
        ≤ 8192 — matches sglang's default ``chunked_prefill_size``.
    Generation (b * s):
        ≤ 1M overall, with per-sl batch caps for long contexts (sl≥8192→bs≤64,
        sl≥32768→bs≤16, sl≥65536→bs≤8, sl≥131072→bs≤4, sl≥262144→bs≤2,
        sl≥524288→bs==1).  Ensures bs=1 is always allowed at every sl.
    """
    is_context = mode == "context"
    pairs = []
    for bs in batch_sizes:
        for sl in seq_lens:
            if is_context:
                if bs * sl > 8192:
                    continue
            else:
                if bs * sl > 1024 * 1024:
                    continue
                if sl >= 524288 and bs > 1:
                    continue
                if sl >= 262144 and bs > 2:
                    continue
                if sl >= 131072 and bs > 4:
                    continue
                if sl >= 65536 and bs > 8:
                    continue
                if sl >= 32768 and bs > 16:
                    continue
                if sl >= 8192 and bs > 64:
                    continue
            pairs.append((bs, sl))
    return pairs


def _build_dsv4_flash_module_test_cases(mode: str, attn_kinds=DSV4_FLASH_ATTN_KINDS):
    """One case per ``(attn_kind, tp_size, gemm_type, batch_size)``.

    Test case shape (9 elements; ``perf_filename`` is bound by collect.py
    via OpEntry, NOT in the tuple)::

        [0, batch_size, tp_size, kv_cache_dtype, compute_dtype, gemm_type,
         model_path, attn_kind, attention_backend]

    Each spawned subprocess builds ONE ``ModelRunner`` for ``(bs, max_sl)``
    and sweeps every valid sl for that bs internally.
    """
    pairs = _dsv4_flash_module_filter_pairs(mode, _DSV4_FLASH_MODULE_BATCH_SIZES, _DSV4_FLASH_MODULE_SEQ_LENGTHS)
    bs_set = sorted({bs for bs, _ in pairs})

    cases: list[list] = []
    for attn_kind in attn_kinds:
        for compute_dtype, kv_dtype, gemm_type in _dsv4_flash_module_precision_combos(mode):
            for tp_size in _DSV4_FLASH_MODULE_TP_SIZES:
                for bs in bs_set:
                    cases.append(
                        [
                            0,
                            bs,
                            tp_size,
                            kv_dtype,
                            compute_dtype,
                            gemm_type,
                            _DSV4_FLASH_DEFAULT_MODEL,
                            attn_kind,
                            None,
                        ]
                    )
    return cases


def get_dsv4_flash_csa_context_test_cases():
    if not _dsv4_flash_active():
        return []
    return _build_dsv4_flash_module_test_cases("context", ("csa",))


def get_dsv4_flash_hca_context_test_cases():
    if not _dsv4_flash_active():
        return []
    return _build_dsv4_flash_module_test_cases("context", ("hca",))


def get_dsv4_flash_swa_context_test_cases():
    if not _dsv4_flash_active():
        return []
    return _build_dsv4_flash_module_test_cases("context", ("swa",))


def get_dsv4_flash_csa_generation_test_cases():
    if not _dsv4_flash_active():
        return []
    return _build_dsv4_flash_module_test_cases("generation", ("csa",))


def get_dsv4_flash_hca_generation_test_cases():
    if not _dsv4_flash_active():
        return []
    return _build_dsv4_flash_module_test_cases("generation", ("hca",))


def get_dsv4_flash_swa_generation_test_cases():
    if not _dsv4_flash_active():
        return []
    return _build_dsv4_flash_module_test_cases("generation", ("swa",))


# --- Sparse-kernel sweep (paged_mqa_logits / hca_attn) ---
# Strict superset of the module collector's (bs, sl) coverage.  Every
# (bs, isl) the module collector exercises is included with past_kv=0;
# on top of that we add past_kv>0 variants for kernel-level Δ correction.
_DSV4_FLASH_SPARSE_BS_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
_DSV4_FLASH_SPARSE_ISL_LIST = [
    1,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
]
_DSV4_FLASH_SPARSE_PAST_KV_LIST = [
    0,
    1,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    10240,
    12288,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048575,
]
_DSV4_FLASH_SPARSE_CHUNK_PREFILL_SIZE = 8192  # bs x isl per-chunk new-token budget
_DSV4_FLASH_SPARSE_MAX_FULL_S = 1048576  # max_position_embeddings
# Sparse kernels are TP-invariant:
#   * paged_mqa_logits: ReplicatedLinear indexer (no sharding)
#   * hca_attn:       FlashMLA gets h_q=64 on every rank (sglang zero-pads
#                     Q to N_HEADS_Q before the kernel call), no GEMM in
#                     the bench path → kernel time is identical across TP
# Module-level ops still sweep all 4 TP values because their projections
# (q_a/q_b/o_proj) are 1/N sharded by ColumnParallel/RowParallel.
_DSV4_FLASH_SPARSE_TP_LIST_ATTN = [1]
_DSV4_FLASH_SPARSE_TP_LIST_INDEXER = [1]

# Bench-sampled sparse kernels.  topk_512 + csa_attn are modeled analytically
# in perf_database — see KERNELS comment in deepseekv4_sparse_modules.py.
DSV4_FLASH_SPARSE_KERNELS = ("paged_mqa_logits", "hca_attn")


def _build_dsv4_flash_sparse_test_cases(
    kernels=DSV4_FLASH_SPARSE_KERNELS,
    bs_list=None,
    isl_list=None,
    past_kv_list=None,
    tp_list_attn=None,
    tp_list_indexer=None,
):
    """Generate ``(bs, isl, past_kv, tp_size, kernel, model)`` tuples.

    Filters mirror sglang prefill scheduler:
      * bs x isl ≤ chunked_prefill_size = 8192   — new-token budget per chunk
      * bs x (isl + past_kv) ≤ 1M                — model context cap
    """
    bs_list = list(bs_list) if bs_list is not None else list(_DSV4_FLASH_SPARSE_BS_LIST)
    isl_list = list(isl_list) if isl_list is not None else list(_DSV4_FLASH_SPARSE_ISL_LIST)
    past_kv_list = list(past_kv_list) if past_kv_list is not None else list(_DSV4_FLASH_SPARSE_PAST_KV_LIST)
    tp_list_attn = list(tp_list_attn) if tp_list_attn is not None else list(_DSV4_FLASH_SPARSE_TP_LIST_ATTN)
    tp_list_indexer = list(tp_list_indexer) if tp_list_indexer is not None else list(_DSV4_FLASH_SPARSE_TP_LIST_INDEXER)

    cases = []
    for kernel in kernels:
        tp_list = tp_list_attn if kernel == "hca_attn" else tp_list_indexer
        for tp_size in tp_list:
            for bs in bs_list:
                for isl in isl_list:
                    if bs * isl > _DSV4_FLASH_SPARSE_CHUNK_PREFILL_SIZE:
                        continue
                    for past_kv in past_kv_list:
                        if bs * (isl + past_kv) > _DSV4_FLASH_SPARSE_MAX_FULL_S:
                            continue
                        full_s = isl + past_kv
                        if kernel == "paged_mqa_logits" and full_s < 4:
                            continue
                        if kernel == "hca_attn" and full_s < 64:
                            continue
                        cases.append(
                            [
                                bs,
                                isl,
                                past_kv,
                                tp_size,
                                kernel,
                                _DSV4_FLASH_DEFAULT_MODEL,
                            ]
                        )
    return cases


def _dsv4_flash_sparse_smoke_or_full(kernel: str):
    if not _dsv4_flash_active():
        return []
    if "--smoke" in sys.argv:
        return [[1, 1024, 8192, 1, kernel, _DSV4_FLASH_DEFAULT_MODEL]]
    return _build_dsv4_flash_sparse_test_cases(kernels=(kernel,))


def get_dsv4_flash_paged_mqa_logits_test_cases():
    """paged_mqa_logits sparse-kernel sweep (CSA indexer scoring)."""
    return _dsv4_flash_sparse_smoke_or_full("paged_mqa_logits")


def get_dsv4_flash_hca_attn_test_cases():
    """hca_attn sparse-kernel sweep (HCA c128 sparse FMLA)."""
    return _dsv4_flash_sparse_smoke_or_full("hca_attn")

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Declarative registry mapping ops to collector modules for SGLang.

No version forks exist yet. When SGLang API changes require a fork,
add a ``versions`` tuple following the trtllm registry pattern.
"""

from collector.registry_types import OpEntry

REGISTRY: list[OpEntry] = [
    OpEntry(
        op="gemm",
        module="collector.sglang.collect_gemm",
        get_func="get_gemm_test_cases",
        run_func="run_gemm",
    ),
    OpEntry(
        op="mla_context",
        module="collector.sglang.collect_mla",
        get_func="get_context_mla_test_cases",
        run_func="run_mla",
    ),
    OpEntry(
        op="mla_generation",
        module="collector.sglang.collect_mla",
        get_func="get_generation_mla_test_cases",
        run_func="run_mla",
    ),
    OpEntry(
        op="mla_bmm_gen_pre",
        module="collector.sglang.collect_mla_bmm",
        get_func="get_mla_gen_pre_test_cases",
        run_func="run_mla_gen_pre",
    ),
    OpEntry(
        op="mla_bmm_gen_post",
        module="collector.sglang.collect_mla_bmm",
        get_func="get_mla_gen_post_test_cases",
        run_func="run_mla_gen_post",
    ),
    OpEntry(
        op="moe",
        module="collector.sglang.collect_moe",
        get_func="get_moe_test_cases",
        run_func="run_moe_torch",
    ),
    OpEntry(
        op="attention_context",
        module="collector.sglang.collect_attn",
        get_func="get_context_attention_test_cases",
        run_func="run_attention_torch",
    ),
    OpEntry(
        op="attention_generation",
        module="collector.sglang.collect_attn",
        get_func="get_generation_attention_test_cases",
        run_func="run_attention_torch",
    ),
    OpEntry(
        op="wideep_mla_context",
        module="collector.sglang.collect_wideep_attn",
        get_func="get_wideep_mla_context_test_cases",
        run_func="run_wideep_mla_context",
    ),
    OpEntry(
        op="wideep_mla_generation",
        module="collector.sglang.collect_wideep_attn",
        get_func="get_wideep_mla_generation_test_cases",
        run_func="run_wideep_mla_generation",
    ),
    OpEntry(
        op="wideep_moe",
        module="collector.sglang.collect_wideep_deepep_moe",
        get_func="get_wideep_moe_test_cases",
        run_func="run_wideep_moe",
    ),
    OpEntry(
        op="gdn",
        module="collector.sglang.collect_gdn",
        get_func="get_gdn_test_cases",
        run_func="run_gdn_torch",
    ),
]

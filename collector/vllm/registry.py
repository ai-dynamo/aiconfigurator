# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Declarative registry mapping ops to collector modules for vLLM.

No version forks exist yet. When vLLM API changes require a fork,
add a ``versions`` tuple following the trtllm registry pattern.
"""

from collector.registry_types import OpEntry

REGISTRY: list[OpEntry] = [
    OpEntry(
        op="gemm",
        module="collector.vllm.collect_gemm",
        get_func="get_gemm_test_cases",
        run_func="run_gemm",
    ),
    OpEntry(
        op="attention_context",
        module="collector.vllm.collect_attn",
        get_func="get_context_attention_test_cases",
        run_func="run_attention_torch",
    ),
    OpEntry(
        op="attention_generation",
        module="collector.vllm.collect_attn",
        get_func="get_generation_attention_test_cases",
        run_func="run_attention_torch",
    ),
    OpEntry(
        op="moe",
        module="collector.vllm.collect_moe",
        get_func="get_moe_test_cases",
        run_func="run_moe_torch",
    ),
    OpEntry(
        op="mla_context",
        module="collector.vllm.collect_mla",
        get_func="get_context_mla_test_cases",
        run_func="run_attention_torch",
    ),
    OpEntry(
        op="mla_generation",
        module="collector.vllm.collect_mla",
        get_func="get_generation_mla_test_cases",
        run_func="run_attention_torch",
    ),
    OpEntry(
        op="mla_context_module",
        module="collector.vllm.collect_mla_module",
        get_func="get_mla_context_module_test_cases",
        run_func="run_mla_module_worker",
    ),
    OpEntry(
        op="mla_generation_module",
        module="collector.vllm.collect_mla_module",
        get_func="get_mla_generation_module_test_cases",
        run_func="run_mla_module_worker",
    ),
    OpEntry(
        op="dsa_context_module",
        module="collector.vllm.collect_mla_module",
        get_func="get_dsa_context_module_test_cases",
        run_func="run_mla_module_worker",
    ),
    OpEntry(
        op="dsa_generation_module",
        module="collector.vllm.collect_mla_module",
        get_func="get_dsa_generation_module_test_cases",
        run_func="run_mla_module_worker",
    ),
    OpEntry(
        op="gdn",
        module="collector.vllm.collect_gdn",
        get_func="get_gdn_test_cases",
        run_func="run_gdn_torch",
    ),
]

REGISTRY_XPU: list[OpEntry] = [
    OpEntry(
        op="gemm",
        module="collector.vllm.collect_gemm_xpu",
        get_func="get_gemm_test_cases",
        run_func="run_gemm",
    ),
    OpEntry(
        op="attention_context",
        module="collector.vllm.collect_attn_xpu",
        get_func="get_context_attention_test_cases",
        run_func="run_attention_torch",
    ),
    OpEntry(
        op="attention_generation",
        module="collector.vllm.collect_attn_xpu",
        get_func="get_generation_attention_test_cases",
        run_func="run_attention_torch",
    ),
    OpEntry(
        op="moe",
        module="collector.vllm.collect_moe_xpu",
        get_func="get_moe_test_cases",
        run_func="run_moe_torch",
    ),
]

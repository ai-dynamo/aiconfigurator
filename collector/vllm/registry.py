# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Declarative registry mapping ops to collector modules for vLLM.

For versioned entries, ``versions`` is a tuple of :class:`VersionRoute` in
**descending** order. The resolver picks the first route whose min_version
is <= the runtime version. To add support for a new vLLM version:
  add a new VersionRoute at the top of the versions tuple.
"""

from collector.registry_types import OpEntry, VersionRoute

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
        get_func="get_moe_test_cases",
        run_func="run_moe_torch",
        versions=(
            VersionRoute("0.17.0", "collector.vllm.collect_moe_v2"),
            VersionRoute("0.0.0", "collector.vllm.collect_moe_v1"),
        ),
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
        get_func="get_mla_context_module_test_cases",
        run_func="run_mla_module_worker",
        versions=(
            VersionRoute("0.17.0", "collector.vllm.collect_mla_module_v2"),
            VersionRoute("0.0.0", "collector.vllm.collect_mla_module_v1"),
        ),
    ),
    OpEntry(
        op="mla_generation_module",
        get_func="get_mla_generation_module_test_cases",
        run_func="run_mla_module_worker",
        versions=(
            VersionRoute("0.17.0", "collector.vllm.collect_mla_module_v2"),
            VersionRoute("0.0.0", "collector.vllm.collect_mla_module_v1"),
        ),
    ),
    OpEntry(
        op="dsa_context_module",
        get_func="get_dsa_context_module_test_cases",
        run_func="run_mla_module_worker",
        versions=(
            VersionRoute("0.17.0", "collector.vllm.collect_mla_module_v2"),
            VersionRoute("0.0.0", "collector.vllm.collect_mla_module_v1"),
        ),
    ),
    OpEntry(
        op="dsa_generation_module",
        get_func="get_dsa_generation_module_test_cases",
        run_func="run_mla_module_worker",
        versions=(
            VersionRoute("0.17.0", "collector.vllm.collect_mla_module_v2"),
            VersionRoute("0.0.0", "collector.vllm.collect_mla_module_v1"),
        ),
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

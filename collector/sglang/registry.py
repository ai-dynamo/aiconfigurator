# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Declarative registry mapping ops to collector modules for SGLang.

No version forks exist yet. When SGLang API changes require a fork,
add a ``versions`` tuple following the trtllm registry pattern.
"""

from collector.registry_types import OpEntry, PerfFile

REGISTRY: list[OpEntry] = [
    OpEntry(
        op="gemm",
        module="collector.sglang.collect_gemm",
        get_func="get_gemm_test_cases",
        run_func="run_gemm",
        perf_filename=PerfFile.GEMM,
    ),
    OpEntry(
        op="mla_context",
        module="collector.sglang.collect_mla",
        get_func="get_context_mla_test_cases",
        run_func="run_mla",
        perf_filename=PerfFile.CONTEXT_MLA,
    ),
    OpEntry(
        op="mla_generation",
        module="collector.sglang.collect_mla",
        get_func="get_generation_mla_test_cases",
        run_func="run_mla",
        perf_filename=PerfFile.GENERATION_MLA,
    ),
    OpEntry(
        op="mla_bmm_gen_pre",
        module="collector.sglang.collect_mla_bmm",
        get_func="get_mla_gen_pre_test_cases",
        run_func="run_mla_gen_pre",
        perf_filename=PerfFile.MLA_BMM,
    ),
    OpEntry(
        op="mla_bmm_gen_post",
        module="collector.sglang.collect_mla_bmm",
        get_func="get_mla_gen_post_test_cases",
        run_func="run_mla_gen_post",
        perf_filename=PerfFile.MLA_BMM,
    ),
    OpEntry(
        op="moe",
        module="collector.sglang.collect_moe",
        get_func="get_moe_test_cases",
        run_func="run_moe_torch",
        perf_filename=PerfFile.MOE,
    ),
    OpEntry(
        op="attention_context",
        module="collector.sglang.collect_attn",
        get_func="get_context_attention_test_cases",
        run_func="run_attention_torch",
        perf_filename=PerfFile.CONTEXT_ATTENTION,
    ),
    OpEntry(
        op="attention_generation",
        module="collector.sglang.collect_attn",
        get_func="get_generation_attention_test_cases",
        run_func="run_attention_torch",
        perf_filename=PerfFile.GENERATION_ATTENTION,
    ),
    OpEntry(
        op="wideep_mla_context",
        module="collector.sglang.collect_mla_module",
        get_func="get_wideep_mla_context_test_cases",
        run_func="run_mla_module_worker",
        perf_filename=PerfFile.WIDEEP_CONTEXT_MLA,
    ),
    OpEntry(
        op="wideep_mla_generation",
        module="collector.sglang.collect_mla_module",
        get_func="get_wideep_mla_generation_test_cases",
        run_func="run_mla_module_worker",
        perf_filename=PerfFile.WIDEEP_GENERATION_MLA,
    ),
    OpEntry(
        op="dsa_context_module",
        module="collector.sglang.collect_mla_module",
        get_func="get_dsa_context_module_test_cases",
        run_func="run_mla_module_worker",
        perf_filename=PerfFile.DSA_CONTEXT_MODULE,
    ),
    OpEntry(
        op="dsa_generation_module",
        module="collector.sglang.collect_mla_module",
        get_func="get_dsa_generation_module_test_cases",
        run_func="run_mla_module_worker",
        perf_filename=PerfFile.DSA_GENERATION_MODULE,
    ),
    OpEntry(
        op="wideep_moe",
        module="collector.sglang.collect_wideep_deepep_moe",
        get_func="get_wideep_moe_test_cases",
        run_func="run_wideep_moe",
        perf_filename=PerfFile.WIDEEP_MOE,
    ),
    OpEntry(
        op="gdn",
        module="collector.sglang.collect_gdn",
        get_func="get_gdn_test_cases",
        run_func="run_gdn_torch",
        perf_filename=PerfFile.GDN,
    ),
]

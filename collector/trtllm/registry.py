# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Declarative registry mapping ops to collector modules.

For versioned entries, ``versions`` is a tuple of :class:`VersionRoute` in
DESCENDING order. The resolver picks the first entry where
``min_version <= runtime``.  Each module file declares its own precise
``__compat__`` constraint (e.g. ``__compat__ = "trtllm>=0.21.0,<1.1.0"``),
which is validated at runtime.

To add support for a new framework version:
- API unchanged: nothing to do (the latest matching entry covers it).
- API changed: create collect_{op}_v{N+1}.py with the right __compat__,
  add a new VersionRoute at the top of the versions tuple.
"""

from collector.registry_types import OpEntry, PerfFile, VersionRoute

REGISTRY: list[OpEntry] = [
    OpEntry(
        op="gemm",
        module="collector.trtllm.collect_gemm",
        get_func="get_gemm_test_cases",
        run_func="run_gemm",
        perf_filename=PerfFile.GEMM,
    ),
    OpEntry(
        op="compute_scale",
        module="collector.trtllm.collect_computescale",
        get_func="get_computescale_test_cases",
        run_func="run_computescale",
        perf_filename=PerfFile.COMPUTESCALE,
    ),
    OpEntry(
        op="mla_context",
        get_func="get_context_mla_test_cases",
        run_func="run_mla",
        perf_filename=PerfFile.CONTEXT_MLA,
        versions=(
            VersionRoute("1.1.0", "collector.trtllm.collect_mla_v2"),
            VersionRoute("0.0.0", "collector.trtllm.collect_mla_v1"),
        ),
    ),
    OpEntry(
        op="mla_generation",
        get_func="get_generation_mla_test_cases",
        run_func="run_mla",
        perf_filename=PerfFile.GENERATION_MLA,
        versions=(
            VersionRoute("1.1.0", "collector.trtllm.collect_mla_v2"),
            VersionRoute("0.0.0", "collector.trtllm.collect_mla_v1"),
        ),
    ),
    OpEntry(
        op="attention_context",
        module="collector.trtllm.collect_attn",
        get_func="get_context_attention_test_cases",
        run_func="run_attention_torch",
        perf_filename=PerfFile.CONTEXT_ATTENTION,
    ),
    OpEntry(
        op="attention_generation",
        module="collector.trtllm.collect_attn",
        get_func="get_generation_attention_test_cases",
        run_func="run_attention_torch",
        perf_filename=PerfFile.GENERATION_ATTENTION,
    ),
    OpEntry(
        op="mla_bmm_gen_pre",
        module="collector.trtllm.collect_mla_bmm",
        get_func="get_mla_gen_pre_test_cases",
        run_func="run_mla_gen_pre",
        perf_filename=PerfFile.MLA_BMM,
    ),
    OpEntry(
        op="mla_bmm_gen_post",
        module="collector.trtllm.collect_mla_bmm",
        get_func="get_mla_gen_post_test_cases",
        run_func="run_mla_gen_post",
        perf_filename=PerfFile.MLA_BMM,
    ),
    OpEntry(
        op="moe",
        get_func="get_moe_test_cases",
        run_func="run_moe_torch",
        perf_filename=PerfFile.MOE,
        versions=(
            VersionRoute("1.1.0", "collector.trtllm.collect_moe_v3"),
            VersionRoute("0.21.0", "collector.trtllm.collect_moe_v2"),
            VersionRoute("0.20.0", "collector.trtllm.collect_moe_v1"),
        ),
    ),
    OpEntry(
        op="trtllm_moe_wideep",
        module="collector.trtllm.collect_wideep_moe_compute",
        get_func="get_wideep_moe_compute_all_test_cases",
        run_func="run_wideep_moe_compute",
        perf_filename=PerfFile.WIDEEP_MOE,
    ),
    OpEntry(
        op="mamba2",
        module="collector.trtllm.collect_mamba2",
        get_func="get_mamba2_test_cases",
        run_func="run_mamba2_torch",
        perf_filename=PerfFile.MAMBA2,
    ),
    OpEntry(
        op="gdn",
        module="collector.trtllm.collect_gdn",
        get_func="get_gdn_test_cases",
        run_func="run_gdn_torch",
        perf_filename=PerfFile.GDN,
    ),
    OpEntry(
        op="mla_context_module",
        module="collector.trtllm.collect_mla_module",
        get_func="get_mla_context_module_test_cases",
        run_func="run_mla_module_worker",
        perf_filename=PerfFile.MLA_CONTEXT_MODULE,
    ),
    OpEntry(
        op="mla_generation_module",
        module="collector.trtllm.collect_mla_module",
        get_func="get_mla_generation_module_test_cases",
        run_func="run_mla_module_worker",
        perf_filename=PerfFile.MLA_GENERATION_MODULE,
    ),
    OpEntry(
        op="dsa_context_module",
        module="collector.trtllm.collect_mla_module",
        get_func="get_dsa_context_module_test_cases",
        run_func="run_mla_module_worker",
        perf_filename=PerfFile.DSA_CONTEXT_MODULE,
    ),
    OpEntry(
        op="dsa_generation_module",
        module="collector.trtllm.collect_mla_module",
        get_func="get_dsa_generation_module_test_cases",
        run_func="run_mla_module_worker",
        perf_filename=PerfFile.DSA_GENERATION_MODULE,
    ),
]

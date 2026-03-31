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

from collector.registry_types import OpEntry, VersionRoute

REGISTRY: list[OpEntry] = [
    OpEntry(
        op="gemm",
        module="collector.trtllm.collect_gemm",
        get_func="get_gemm_test_cases",
        run_func="run_gemm",
    ),
    OpEntry(
        op="compute_scale",
        module="collector.trtllm.collect_computescale",
        get_func="get_computescale_test_cases",
        run_func="run_computescale",
    ),
    OpEntry(
        op="mla_context",
        get_func="get_context_mla_test_cases",
        run_func="run_mla",
        versions=(
            VersionRoute("1.1.0", "collector.trtllm.collect_mla_v2"),
            VersionRoute("0.0.0", "collector.trtllm.collect_mla_v1"),
        ),
    ),
    OpEntry(
        op="mla_generation",
        get_func="get_generation_mla_test_cases",
        run_func="run_mla",
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
    ),
    OpEntry(
        op="attention_generation",
        module="collector.trtllm.collect_attn",
        get_func="get_generation_attention_test_cases",
        run_func="run_attention_torch",
    ),
    OpEntry(
        op="mla_bmm_gen_pre",
        module="collector.trtllm.collect_mla_bmm",
        get_func="get_mla_gen_pre_test_cases",
        run_func="run_mla_gen_pre",
    ),
    OpEntry(
        op="mla_bmm_gen_post",
        module="collector.trtllm.collect_mla_bmm",
        get_func="get_mla_gen_post_test_cases",
        run_func="run_mla_gen_post",
    ),
    OpEntry(
        op="moe",
        get_func="get_moe_test_cases",
        run_func="run_moe_torch",
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
    ),
    OpEntry(
        op="mamba2",
        module="collector.trtllm.collect_mamba2",
        get_func="get_mamba2_test_cases",
        run_func="run_mamba2_torch",
    ),
    OpEntry(
        op="gdn",
        module="collector.trtllm.collect_gdn",
        get_func="get_gdn_test_cases",
        run_func="run_gdn_torch",
    ),
    OpEntry(
        op="mla_context_module",
        module="collector.trtllm.collect_mla_module",
        get_func="get_mla_context_module_test_cases",
        run_func="run_mla_module_worker",
    ),
    OpEntry(
        op="mla_generation_module",
        module="collector.trtllm.collect_mla_module",
        get_func="get_mla_generation_module_test_cases",
        run_func="run_mla_module_worker",
    ),
    OpEntry(
        op="dsa_context_module",
        module="collector.trtllm.collect_mla_module",
        get_func="get_dsa_context_module_test_cases",
        run_func="run_mla_module_worker",
    ),
    OpEntry(
        op="dsa_generation_module",
        module="collector.trtllm.collect_mla_module",
        get_func="get_dsa_generation_module_test_cases",
        run_func="run_mla_module_worker",
    ),
]

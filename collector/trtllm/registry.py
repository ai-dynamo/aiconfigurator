# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Declarative registry mapping ops to collector modules.

For versioned entries, `versions` is a list of (min_version, module_path) tuples
in DESCENDING order. The resolver picks the first entry where min_version <= runtime.
Each module file declares its own precise ``__compat__`` constraint (e.g.
``__compat__ = "trtllm>=0.21.0,<1.1.0"``), which is validated at runtime.

To add support for a new framework version:
- API unchanged: nothing to do (the latest matching entry covers it).
- API changed: create collect_{op}_v{N+1}.py with the right __compat__,
  add a new tuple at the top of the versions list.
"""

REGISTRY = [
    {
        "op": "gemm",
        "module": "collector.trtllm.collect_gemm",
        "get_func": "get_gemm_test_cases",
        "run_func": "run_gemm",
    },
    {
        "op": "compute_scale",
        "module": "collector.trtllm.collect_computescale",
        "get_func": "get_computescale_test_cases",
        "run_func": "run_computescale",
    },
    {
        "op": "mla_context",
        "get_func": "get_context_mla_test_cases",
        "run_func": "run_mla",
        "versions": [
            ("1.1.0", "collector.trtllm.collect_mla_v2"),
            ("0.0.0", "collector.trtllm.collect_mla_v1"),
        ],
    },
    {
        "op": "mla_generation",
        "get_func": "get_generation_mla_test_cases",
        "run_func": "run_mla",
        "versions": [
            ("1.1.0", "collector.trtllm.collect_mla_v2"),
            ("0.0.0", "collector.trtllm.collect_mla_v1"),
        ],
    },
    {
        "op": "attention_context",
        "module": "collector.trtllm.collect_attn",
        "get_func": "get_context_attention_test_cases",
        "run_func": "run_attention_torch",
    },
    {
        "op": "attention_generation",
        "module": "collector.trtllm.collect_attn",
        "get_func": "get_generation_attention_test_cases",
        "run_func": "run_attention_torch",
    },
    {
        "op": "mla_bmm_gen_pre",
        "module": "collector.trtllm.collect_mla_bmm",
        "get_func": "get_mla_gen_pre_test_cases",
        "run_func": "run_mla_gen_pre",
    },
    {
        "op": "mla_bmm_gen_post",
        "module": "collector.trtllm.collect_mla_bmm",
        "get_func": "get_mla_gen_post_test_cases",
        "run_func": "run_mla_gen_post",
    },
    {
        "op": "moe",
        "get_func": "get_moe_test_cases",
        "run_func": "run_moe_torch",
        "versions": [
            ("1.1.0", "collector.trtllm.collect_moe_v3"),
            ("0.21.0", "collector.trtllm.collect_moe_v2"),
            ("0.20.0", "collector.trtllm.collect_moe_v1"),
        ],
    },
    {
        "op": "moe_eplb",
        "get_func": "get_moe_eplb_test_cases",
        "run_func": "run_moe_torch",
        "versions": [
            ("1.1.0", "collector.trtllm.collect_moe_v3"),
        ],
    },
    {
        "op": "trtllm_moe_wideep",
        "module": "collector.trtllm.collect_wideep_moe_compute",
        "get_func": "get_wideep_moe_compute_all_test_cases",
        "run_func": "run_wideep_moe_compute",
    },
    {
        "op": "mamba2",
        "module": "collector.trtllm.collect_mamba2",
        "get_func": "get_mamba2_test_cases",
        "run_func": "run_mamba2_torch",
    },
]

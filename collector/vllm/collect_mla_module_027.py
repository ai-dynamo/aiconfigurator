# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM 0.27.0 lane for the MLA module collector.

`collector.vllm.collect_mla_module` stays pinned to the manifest default
(0.24.0) because it also hosts the DSA ops whose family (`sparse_attention`)
collects on that runtime. The `mla` family moved to the 0.27.0 K3 serving
lane image; per the one-runtime-per-module invariant those ops resolve here.

The bodies are thin re-dispatch wrappers around the base collector: the
0.27.0 audit (GB300 probe, 2026-08-17) confirmed
`DeepseekV2MLAAttention.__init__` params, the `glm_moe_dsa` config-registry
gap patch, `backend_supports_prefill_query_quantization`, and the prefill
selector surface are all unchanged from the 0.24.0 citations documented in
collect_mla_module.py, so the wrappers change neither WHICH kernel runs nor
HOW a case is timed.
"""

__compat__ = "vllm==0.27.0"

from collector.vllm import collect_mla_module as _impl


def get_mla_context_module_test_cases():
    return _impl.get_mla_context_module_test_cases()


def get_mla_generation_module_test_cases():
    return _impl.get_mla_generation_module_test_cases()


def run_mla_module_worker(*args, **kwargs):
    return _impl.run_mla_module_worker(*args, **kwargs)

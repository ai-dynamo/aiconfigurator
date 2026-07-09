# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WideEP collector registry for SGLang.

These ops require the WideEP/SGLang runtime declared in
``collector/framework_manifest.yaml``. They stay out of the stock SGLang
registry so normal SGLang collection does not accidentally use a special image.
"""

from collector.registry_types import OpEntry, PerfFile

REGISTRY: list[OpEntry] = [
    OpEntry(
        op="wideep_mla_context",
        module="collector.wideep.sglang.collect_mla_module",
        get_func="get_wideep_mla_context_test_cases",
        run_func="run_mla_module_worker",
        perf_filename=PerfFile.WIDEEP_CONTEXT_MLA,
    ),
    OpEntry(
        op="wideep_mla_generation",
        module="collector.wideep.sglang.collect_mla_module",
        get_func="get_wideep_mla_generation_test_cases",
        run_func="run_mla_module_worker",
        perf_filename=PerfFile.WIDEEP_GENERATION_MLA,
    ),
    OpEntry(
        op="wideep_moe",
        module="collector.wideep.sglang.collect_deepep_moe",
        get_func="get_wideep_moe_test_cases",
        run_func="run_wideep_moe",
        perf_filename=PerfFile.WIDEEP_MOE,
    ),
]

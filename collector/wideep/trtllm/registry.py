# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WideEP collector registry for TensorRT-LLM."""

from collector.registry_types import OpEntry, PerfFile

REGISTRY: list[OpEntry] = [
    OpEntry(
        op="trtllm_moe_wideep",
        module="collector.wideep.trtllm.collect_moe_compute",
        get_func="get_wideep_moe_compute_all_test_cases",
        run_func="run_wideep_moe_compute",
        perf_filename=PerfFile.WIDEEP_MOE,
    ),
]

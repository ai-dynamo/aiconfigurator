# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WideEP collector registry for TensorRT-LLM.

The registered expert-compute op keeps the same runtime as stock TensorRT-LLM.
Distributed ``moe_a2a`` uses its separately attested standalone source build.
"""

from collector.registry_types import OpEntry, PerfFile

REGISTRY: list[OpEntry] = [
    OpEntry(
        op="moe_ep",
        module="collector.wideep.trtllm.collect_moe_compute",
        get_func="get_moe_ep_test_cases",
        run_func="run_moe_ep",
        perf_filename=PerfFile.MOE_EXPERT_COMPUTE,
    ),
]

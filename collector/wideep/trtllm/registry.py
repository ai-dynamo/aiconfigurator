# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WideEP collector registry for TensorRT-LLM.

Unlike ``wideep_sglang``, the ``wideep_trtllm`` manifest entry pins the SAME
image and version as stock trtllm (``framework_manifest.yaml``), so
``require_collector_runtime`` accepts these ops mixed into a stock trtllm run
and the model case plans activate ``moe_ep`` by default for wideep-declared
models.
"""

from collector.registry_types import OpEntry, PerfFile

REGISTRY: list[OpEntry] = [
    OpEntry(
        op="moe_ep",
        module="collector.wideep.trtllm.collect_moe_compute",
        get_func="get_moe_ep_test_cases",
        run_func="run_moe_ep",
        perf_filename=PerfFile.MOE_EP,
    ),
]

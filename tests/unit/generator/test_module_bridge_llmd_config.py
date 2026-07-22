# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The bridge must preserve LlmdConfig overrides for llm-d artifacts.

cli default / cli sweep go through the Task->generator bridge. It handled the
other config sections explicitly but dropped LlmdConfig, so llm-d artifacts
ignored user-selected image / kustomize base and fell back to template defaults.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from aiconfigurator.generator.module_bridge import task_config_to_generator_config


def _task() -> SimpleNamespace:
    return SimpleNamespace(
        primary_backend_name="vllm",
        primary_system_name="h200_sxm",
        primary_backend_version="0.20.1",
        primary_model_path="Qwen/Qwen3-32B",
        prefix=0,
        is_moe=False,
        nextn=0,
        nextn_accepted=None,
        serving_mode="disagg",
        total_gpus=0,
        system_name="h200_sxm",
        isl=4000,
        osl=1000,
        ttft=2000.0,
        tpot=50.0,
    )


def test_llmd_config_overrides_are_preserved():
    overrides = {
        "LlmdConfig": {
            "vllm_image": "nvcr.io/nvstaging/ai-dynamo/vllm-runtime:1.3.0-rc1",
            "kustomize_base_path": "./guides/pd-disaggregation/modelserver/gpu/vllm/base",
        }
    }
    row = pd.Series({"(p)workers": 1, "(p)tp": 1, "(d)workers": 1, "(d)tp": 1})

    result = task_config_to_generator_config(_task(), row, generator_overrides=overrides, num_gpus_per_node=8)

    assert result["LlmdConfig"] == overrides["LlmdConfig"]


def test_llmd_config_absent_when_not_requested():
    row = pd.Series({"(p)workers": 1, "(p)tp": 1, "(d)workers": 1, "(d)tp": 1})

    result = task_config_to_generator_config(_task(), row, num_gpus_per_node=8)

    assert "LlmdConfig" not in result

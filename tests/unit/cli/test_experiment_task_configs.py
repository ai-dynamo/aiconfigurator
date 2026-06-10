# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""build_experiment_task_configs builds v2 Tasks from (legacy or flat) experiment YAML.

Legacy V1 experiment dicts (``mode`` / nested ``config`` / ``profiles``) are
auto-converted to the flat V2 schema by ``Task.from_yaml``; these tests check
that top-level fields survive the conversion onto the flat ``Task``.
"""

import pytest

from aiconfigurator.cli.main import build_experiment_task_configs

pytestmark = pytest.mark.unit


def test_build_experiment_preserves_top_level_runtime_fields():
    task_configs = build_experiment_task_configs(
        config={
            "exps": ["exp_agg_trtllm"],
            "exp_agg_trtllm": {
                "mode": "patch",  # legacy marker -> auto-converted
                "serving_mode": "agg",
                "model_path": "Qwen/Qwen3-32B",
                "total_gpus": 8,
                "system_name": "h200_sxm",
                "backend_name": "trtllm",
                "isl": 8000,
                "osl": 1000,
                "prefix": 5600,
                "ttft": 1000.0,
                "tpot": 20.0,
                "request_latency": 25000.0,
            },
        }
    )

    task = task_configs["exp_agg_trtllm"]
    assert task.serving_mode == "agg"
    assert task.isl == 8000
    assert task.osl == 1000
    assert task.prefix == 5600
    assert task.ttft == 1000.0
    assert task.tpot == 20.0
    assert task.request_latency == 25000.0


def test_build_experiment_keeps_prefix_at_top_level():
    task_configs = build_experiment_task_configs(
        config={
            "exps": ["exp_agg"],
            "exp_agg": {
                "mode": "patch",
                "serving_mode": "agg",
                "model_path": "nvidia/Kimi-K2.5-NVFP4",
                "total_gpus": 8,
                "system_name": "b200_sxm",
                "backend_name": "trtllm",
                "database_mode": "HYBRID",
                "isl": 4000,
                "osl": 1000,
                "prefix": 1000,
            },
        }
    )

    task = task_configs["exp_agg"]
    assert task.prefix == 1000
    assert task.database_mode == "HYBRID"


def test_build_experiment_forwards_top_level_moe_backend():
    task_configs = build_experiment_task_configs(
        config={
            "exps": ["exp_megamoe"],
            "exp_megamoe": {
                "mode": "patch",
                "serving_mode": "agg",
                "model_path": "deepseek-ai/DeepSeek-V4-Pro",
                "total_gpus": 32,
                "system_name": "gb200",
                "backend_name": "sglang",
                "backend_version": "0.5.10",
                "database_mode": "HYBRID",
                "moe_backend": "megamoe",
            },
        }
    )

    task = task_configs["exp_megamoe"]
    assert task.serving_mode == "agg"
    assert task.moe_backend == "megamoe"

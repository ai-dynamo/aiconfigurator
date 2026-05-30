# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiconfigurator.cli.main import build_experiment_task_configs

pytestmark = pytest.mark.unit


def test_build_experiment_task_configs_preserves_flat_runtime_fields():
    """Top-level workload fields land as flat Task fields, not nested."""
    task_configs = build_experiment_task_configs(
        config={
            "exps": ["exp_agg_trtllm"],
            "exp_agg_trtllm": {
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
                "nextn": 0,
            },
        }
    )

    task = task_configs["exp_agg_trtllm"]
    assert task.isl == 8000
    assert task.osl == 1000
    assert task.prefix == 5600
    assert task.ttft == 1000.0
    assert task.tpot == 20.0
    assert task.request_latency == 25000.0
    assert task.serving_mode == "agg"
    assert task.model_path == "Qwen/Qwen3-32B"


def test_build_experiment_task_configs_propagates_database_mode():
    task_configs = build_experiment_task_configs(
        config={
            "exps": ["exp_agg"],
            "exp_agg": {
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

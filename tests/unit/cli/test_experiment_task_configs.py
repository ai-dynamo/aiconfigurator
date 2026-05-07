# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import yaml

from aiconfigurator.cli.main import build_experiment_task_configs

pytestmark = pytest.mark.unit


def test_build_experiment_task_configs_preserves_top_level_runtime_fields_with_config_patch():
    task_configs = build_experiment_task_configs(
        config={
            "exps": ["exp_agg_trtllm"],
            "exp_agg_trtllm": {
                "mode": "patch",
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
                "config": {
                    "nextn": 0,
                },
            },
        }
    )

    task_config = task_configs["exp_agg_trtllm"]
    runtime_config = task_config.config.runtime_config

    assert runtime_config.isl == 8000
    assert runtime_config.osl == 1000
    assert runtime_config.prefix == 5600
    assert runtime_config.ttft == 1000.0
    assert runtime_config.tpot == 20.0
    assert runtime_config.request_latency == 25000.0
    assert "_8000_1000_5600_1000.0_20.0" in task_config.task_name


def test_build_experiment_task_configs_keeps_no_config_prefix_out_of_yaml_patch():
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

    task_config = task_configs["exp_agg"]

    assert task_config.config.runtime_config.prefix == 1000
    assert task_config.yaml_patch == {}

    exported = next(iter(yaml.safe_load(task_config.to_yaml()).values()))
    assert exported["prefix"] == 1000
    assert "prefix" not in exported.get("config", {})

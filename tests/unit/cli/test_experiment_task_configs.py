# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import yaml

import aiconfigurator.sdk.task as task_module
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


def test_build_experiment_task_configs_validates_silicon_backend_without_explicit_version(monkeypatch):
    calls = []

    def record_backend_check(system_name, backend_name, backend_version=None):
        calls.append((system_name, backend_name, backend_version))

    monkeypatch.setattr("aiconfigurator.cli.main._ensure_backend_version_available", record_backend_check)

    build_experiment_task_configs(
        config={
            "exps": ["exp_agg"],
            "exp_agg": {
                "serving_mode": "agg",
                "model_path": "Qwen/Qwen3-32B",
                "total_gpus": 8,
                "system_name": "h200_sxm",
                "backend_name": "trtllm",
            },
        }
    )

    assert calls == [("h200_sxm", "trtllm", None)]


def test_build_experiment_task_configs_forwards_top_level_moe_backend_before_yaml_patch(monkeypatch):
    class FakeDatabase:
        def __init__(self):
            self.system_spec = {"gpu": {"sm_version": 100}}
            self.supported_quant_mode = {
                "gemm": ["fp8_block"],
                "moe": ["bfloat16"],
                "dsv4_megamoe_module": ["w4a8_mxfp4_mxfp8"],
                "deepseek_v4_context_module": ["bfloat16"],
                "deepseek_v4_generation_module": ["fp8"],
            }

    def fake_get_database(*args, **kwargs):
        return FakeDatabase()

    monkeypatch.setattr(task_module, "get_database", fake_get_database)

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

    task_config = task_configs["exp_megamoe"]
    worker_config = task_config.config.worker_config

    assert task_config.moe_backend == "megamoe"
    assert task_config.config.moe_backend == "megamoe"
    assert task_config.yaml_patch == {}
    assert worker_config.num_gpu_per_worker == [4, 8, 16, 32]
    assert worker_config.moe_tp_list == [1]
    assert worker_config.moe_ep_list == [4, 8, 16, 32]


def test_build_experiment_task_configs_expands_list_valued_top_level_fields():
    task_configs = build_experiment_task_configs(
        config={
            "exps": ["exp_qwen"],
            "exp_qwen": {
                "mode": "patch",
                "serving_mode": ["agg", "disagg"],
                "model_path": "Qwen/Qwen3-32B",
                "total_gpus": [4, 8],
                "system_name": "h200_sxm",
                "backend_name": "trtllm",
                "isl": [4000],
                "config": {
                    "nextn": 0,
                    "worker_config": {
                        "tp_list": [1, 2],
                    },
                },
            },
        }
    )

    assert list(task_configs) == [
        "exp_qwen__mode-agg__gpus-4",
        "exp_qwen__mode-agg__gpus-8",
        "exp_qwen__mode-disagg__gpus-4",
        "exp_qwen__mode-disagg__gpus-8",
    ]
    assert task_configs["exp_qwen__mode-agg__gpus-4"].serving_mode == "agg"
    assert task_configs["exp_qwen__mode-disagg__gpus-8"].serving_mode == "disagg"
    assert task_configs["exp_qwen__mode-disagg__gpus-8"].total_gpus == 8
    assert task_configs["exp_qwen__mode-agg__gpus-4"].yaml_patch["worker_config"]["tp_list"] == [1, 2]


def test_build_experiment_task_configs_expands_runtime_knob_lists_without_yaml_leakage():
    task_configs = build_experiment_task_configs(
        config={
            "exps": ["exp_runtime"],
            "exp_runtime": {
                "mode": "patch",
                "serving_mode": "agg",
                "model_path": "Qwen/Qwen3-32B",
                "total_gpus": 8,
                "system_name": "h200_sxm",
                "backend_name": "trtllm",
                "free_gpu_memory_fraction": [0.8, 1.0],
                "max_seq_len": [4096],
                "enable_chunked_prefill": [False],
            },
        }
    )

    assert list(task_configs) == [
        "exp_runtime__mem_frac-0.8",
        "exp_runtime__mem_frac-1.0",
    ]
    assert task_configs["exp_runtime__mem_frac-0.8"].free_gpu_memory_fraction == 0.8
    assert task_configs["exp_runtime__mem_frac-0.8"].max_seq_len == 4096
    assert task_configs["exp_runtime__mem_frac-0.8"].config.enable_chunked_prefill is False
    assert task_configs["exp_runtime__mem_frac-0.8"].yaml_patch == {}

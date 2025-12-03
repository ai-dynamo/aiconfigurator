# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pathlib
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import aiconfigurator.sdk.task as task_module
from aiconfigurator.sdk.task import TaskConfig, TaskRunner


@pytest.fixture(autouse=True)
def stub_perf_database(monkeypatch):
    class FakeDatabase:
        def __init__(self, sm_version: int):
            self.system_spec = {"gpu": {"sm_version": sm_version}}

    def fake_get_database(system, backend, version):
        if "b200" in system:
            sm = 100
        elif "h200" in system:
            sm = 90
        else:
            sm = 89
        return FakeDatabase(sm)

    def fake_latest_version(system, backend):
        return "stub-version"

    monkeypatch.setattr(task_module, "get_database", fake_get_database)
    monkeypatch.setattr(task_module, "get_latest_database_version", fake_latest_version)


@pytest.fixture(autouse=True)
def stub_pareto_analysis(monkeypatch):
    def fake_get_pareto_front(df, x_col, y_col):
        return df

    stub_module = SimpleNamespace(
        enumerate_parallel_config=lambda **kwargs: [kwargs],
        agg_pareto=lambda **kwargs: pd.DataFrame({"tokens/s/user": [1.0], "tokens/s/gpu": [0.5]}),
        disagg_pareto=lambda **kwargs: pd.DataFrame({"tokens/s/user": [0.8], "tokens/s/gpu": [0.4]}),
        get_pareto_front=fake_get_pareto_front,
    )

    monkeypatch.setitem(sys.modules, "aiconfigurator.sdk.pareto_analysis", stub_module)
    import aiconfigurator.sdk as sdk_pkg
    import aiconfigurator.sdk.utils as sdk_utils

    monkeypatch.setattr(sdk_pkg, "pareto_analysis", stub_module, raising=False)
    monkeypatch.setattr(sdk_utils, "enumerate_parallel_config", stub_module.enumerate_parallel_config)


def _enum_name(value):
    return value.name if hasattr(value, "name") else value


def test_taskconfig_agg_default():
    task = TaskConfig(serving_mode="agg", model_name="QWEN3_32B", system_name="h200_sxm")
    cfg = task.config

    assert cfg.worker_config.system_name == "h200_sxm"
    assert _enum_name(cfg.worker_config.gemm_quant_mode) == "fp8_block"
    assert cfg.worker_config.num_gpu_per_worker == [1, 2, 4, 8]
    assert cfg.applied_layers == ["base-common", "agg-defaults"]


def test_taskconfig_disagg_default():
    task = TaskConfig(serving_mode="disagg", model_name="QWEN3_32B", system_name="h200_sxm")
    cfg = task.config

    assert cfg.prefill_worker_config.system_name == "h200_sxm"
    assert cfg.decode_worker_config.system_name == "h200_sxm"
    assert cfg.replica_config.max_gpu_per_replica == 128
    assert cfg.advanced_tuning_config.prefill_max_batch_size == 1
    assert cfg.advanced_tuning_config.decode_max_batch_size == 512
    assert cfg.advanced_tuning_config.prefill_latency_correction_scale == 1.1
    assert cfg.advanced_tuning_config.decode_latency_correction_scale == 1.08
    assert "disagg-defaults" in cfg.applied_layers


def test_taskconfig_profile_application():
    task = TaskConfig(
        serving_mode="agg",
        model_name="QWEN3_32B",
        system_name="h200_sxm",
        profiles=["fp8_default"],
    )
    cfg = task.config

    assert _enum_name(cfg.worker_config.gemm_quant_mode) == "fp8"
    assert any(layer.startswith("profile:fp8_default") for layer in cfg.applied_layers)


def test_taskconfig_total_gpus_limits_agg_workers():
    task = TaskConfig(
        serving_mode="agg",
        model_name="QWEN3_32B",
        system_name="h200_sxm",
        total_gpus=2,
    )
    cfg = task.config

    assert cfg.worker_config.num_gpu_per_worker == [1, 2]


def test_taskconfig_agg_yaml_patch_overrides():
    task = TaskConfig(
        serving_mode="agg",
        model_name="QWEN3_32B",
        system_name="h200_sxm",
        yaml_config={
            "mode": "patch",
            "config": {
                "worker_config": {
                    "system_name": "b200_sxm",
                    "backend_version": "patched-version",
                    "num_gpu_per_worker": [1, 2, 4],
                    "tp_list": [1, 2, 4],
                }
            },
        },
    )
    cfg = task.config

    assert cfg.worker_config.system_name == "b200_sxm"
    assert cfg.worker_config.backend_version == "patched-version"
    assert cfg.worker_config.num_gpu_per_worker == [1, 2, 4]


def test_taskconfig_yaml_file_profiles_and_patch(tmp_path):
    yaml_payload = {
        "mode": "patch",
        "profiles": ["float16_default"],
        "config": {
            "worker_config": {
                "tp_list": [1, 2],
            }
        },
    }
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(yaml.safe_dump(yaml_payload), encoding="utf-8")

    loaded_yaml = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    task = TaskConfig(
        serving_mode="agg",
        model_name="QWEN3_32B",
        system_name="h200_sxm",
        profiles=["fp8_default"],
        yaml_config=loaded_yaml,
    )
    cfg = task.config

    assert cfg.worker_config.tp_list == [1, 2]
    assert _enum_name(cfg.worker_config.gemm_quant_mode) == "float16"
    assert cfg.applied_layers[-1] == "yaml_patch"


def test_taskconfig_disagg_profile_patch_expands_replica():
    task = TaskConfig(
        serving_mode="disagg",
        model_name="QWEN3_32B",
        system_name="b200_sxm",
        profiles=["fp8_default"],
        yaml_config={
            "mode": "patch",
            "config": {
                "prefill_worker_config": {"num_gpu_per_worker": [2, 4, 8]},
                "decode_worker_config": {"num_gpu_per_worker": [2, 4, 8]},
                "replica_config": {"max_gpu_per_replica": 256},
            },
        },
    )
    cfg = task.config

    assert cfg.replica_config.max_gpu_per_replica == 256
    assert cfg.replica_config.num_gpu_per_replica[-1] == 256
    assert cfg.prefill_worker_config.num_gpu_per_worker == [2, 4, 8]
    assert cfg.decode_worker_config.num_gpu_per_worker == [2, 4, 8]


def test_taskconfig_disagg_total_gpus_caps_replica():
    task = TaskConfig(
        serving_mode="disagg",
        model_name="QWEN3_32B",
        system_name="h200_sxm",
        total_gpus=16,
    )
    cfg = task.config

    assert cfg.replica_config.max_gpu_per_replica == 16


def test_taskconfig_disagg_total_gpus_with_patch():
    task = TaskConfig(
        serving_mode="disagg",
        model_name="QWEN3_32B",
        system_name="h200_sxm",
        total_gpus=32,
        profiles=["fp8_default"],
        yaml_config={
            "mode": "patch",
            "config": {"replica_config": {"max_gpu_per_replica": 256}},
        },
    )
    cfg = task.config

    assert cfg.replica_config.max_gpu_per_replica == 32


def test_taskconfig_disagg_wideep_expands_lists():
    task = TaskConfig(
        serving_mode="disagg",
        model_name="DEEPSEEK_V3",
        system_name="gb200_sxm",
        enable_wideep=True,
    )
    cfg = task.config

    assert cfg.is_moe is True
    assert cfg.replica_config.max_gpu_per_replica == 512
    assert cfg.prefill_worker_config.num_gpu_per_worker[-1] == 32
    assert cfg.decode_worker_config.num_gpu_per_worker[-1] == 64


def test_taskconfig_agg_total_gpus_negative_rejected():
    with pytest.raises(ValueError):
        TaskConfig(
            serving_mode="agg",
            model_name="QWEN3_32B",
            system_name="h200_sxm",
            total_gpus=-1,
        )


def test_taskconfig_disagg_total_gpus_small_rejected():
    with pytest.raises(ValueError):
        TaskConfig(
            serving_mode="disagg",
            model_name="QWEN3_32B",
            system_name="h200_sxm",
            total_gpus=1,
        )


def test_taskrunner_runs_agg_and_disagg():
    agg_task = TaskConfig(serving_mode="agg", model_name="QWEN3_32B", system_name="h200_sxm")
    disagg_task = TaskConfig(serving_mode="disagg", model_name="QWEN3_32B", system_name="h200_sxm", total_gpus=8)

    runner = TaskRunner()

    agg_result = runner.run(agg_task)
    disagg_result = runner.run(disagg_task)

    assert isinstance(agg_result["pareto_df"], pd.DataFrame)
    assert isinstance(disagg_result["pareto_df"], pd.DataFrame)


def test_sglang_moe_configs():
    """Test sglang MoE configurations for different scenarios."""
    # Test 1: sglang + MoE + wideep + disagg
    task_wideep = TaskConfig(
        serving_mode="disagg",
        model_name="DEEPSEEK_V3",
        system_name="h200_sxm",
        backend_name="sglang",
        enable_wideep=True,
        total_gpus=64,
    )

    prefill_cfg = task_wideep.config.prefill_worker_config
    decode_cfg = task_wideep.config.decode_worker_config

    # Verify prefill config
    assert prefill_cfg.num_gpu_per_worker == [8, 16, 32], f"Expected [8, 16, 32], got {prefill_cfg.num_gpu_per_worker}"
    assert prefill_cfg.tp_list == [1, 2, 4, 8], f"Expected [1, 2, 4, 8], got {prefill_cfg.tp_list}"
    assert prefill_cfg.dp_list == [1, 2, 4, 8, 16, 32], f"Expected [1, 2, 4, 8, 16, 32], got {prefill_cfg.dp_list}"
    assert prefill_cfg.moe_tp_list == [1], f"Expected [1], got {prefill_cfg.moe_tp_list}"
    assert prefill_cfg.moe_ep_list == [8, 16, 32], f"Expected [8, 16, 32], got {prefill_cfg.moe_ep_list}"

    # Verify decode config
    assert decode_cfg.num_gpu_per_worker == [8, 16, 32, 64], (
        f"Expected [8, 16, 32, 64], got {decode_cfg.num_gpu_per_worker}"
    )
    assert decode_cfg.tp_list == [1, 2, 4, 8], f"Expected [1, 2, 4, 8], got {decode_cfg.tp_list}"
    assert decode_cfg.dp_list == [1, 2, 4, 8, 16, 32, 64], (
        f"Expected [1, 2, 4, 8, 16, 32, 64], got {decode_cfg.dp_list}"
    )
    assert decode_cfg.moe_tp_list == [1], f"Expected [1], got {decode_cfg.moe_tp_list}"
    assert decode_cfg.moe_ep_list == [8, 16, 32, 64], f"Expected [8, 16, 32, 64], got {decode_cfg.moe_ep_list}"

    # Test 2: sglang + MoE (non-wideep) + disagg
    task_no_wideep = TaskConfig(
        serving_mode="disagg",
        model_name="DEEPSEEK_V3",
        system_name="h200_sxm",
        backend_name="sglang",
        enable_wideep=False,
        total_gpus=8,
    )

    prefill_cfg2 = task_no_wideep.config.prefill_worker_config
    decode_cfg2 = task_no_wideep.config.decode_worker_config

    # Verify prefill config
    assert prefill_cfg2.num_gpu_per_worker == [1, 2, 4, 8], (
        f"Expected [1, 2, 4, 8], got {prefill_cfg2.num_gpu_per_worker}"
    )
    assert prefill_cfg2.tp_list == [1, 2, 4, 8], f"Expected [1, 2, 4, 8], got {prefill_cfg2.tp_list}"
    assert prefill_cfg2.dp_list == [1, 2, 4, 8], f"Expected [1, 2, 4, 8], got {prefill_cfg2.dp_list}"
    assert prefill_cfg2.moe_tp_list == [1, 2, 4, 8], f"Expected [1, 2, 4, 8], got {prefill_cfg2.moe_tp_list}"
    assert prefill_cfg2.moe_ep_list == [1], f"Expected [1], got {prefill_cfg2.moe_ep_list}"

    # Verify decode config
    assert decode_cfg2.num_gpu_per_worker == [1, 2, 4, 8], (
        f"Expected [1, 2, 4, 8], got {decode_cfg2.num_gpu_per_worker}"
    )
    assert decode_cfg2.tp_list == [1, 2, 4, 8], f"Expected [1, 2, 4, 8], got {decode_cfg2.tp_list}"
    assert decode_cfg2.dp_list == [1, 2, 4, 8], f"Expected [1, 2, 4, 8], got {decode_cfg2.dp_list}"
    assert decode_cfg2.moe_tp_list == [1, 2, 4, 8], f"Expected [1, 2, 4, 8], got {decode_cfg2.moe_tp_list}"
    assert decode_cfg2.moe_ep_list == [1], f"Expected [1], got {decode_cfg2.moe_ep_list}"

    # Test 3: trtllm + MoE + wideep (should use previous logic)
    task_trtllm_wideep = TaskConfig(
        serving_mode="disagg",
        model_name="DEEPSEEK_V3",
        system_name="h200_sxm",
        backend_name="trtllm",
        enable_wideep=True,
        total_gpus=64,
    )

    prefill_cfg3 = task_trtllm_wideep.config.prefill_worker_config
    decode_cfg3 = task_trtllm_wideep.config.decode_worker_config

    # Verify trtllm uses previous wideep logic
    assert prefill_cfg3.num_gpu_per_worker == [1, 2, 4, 8, 16, 32], (
        f"Expected [1, 2, 4, 8, 16, 32], got {prefill_cfg3.num_gpu_per_worker}"
    )
    assert prefill_cfg3.moe_ep_list == [1, 2, 4, 8, 16, 32], (
        f"Expected [1, 2, 4, 8, 16, 32], got {prefill_cfg3.moe_ep_list}"
    )
    assert decode_cfg3.num_gpu_per_worker == [1, 2, 4, 8, 16, 32, 64], (
        f"Expected [1, 2, 4, 8, 16, 32, 64], got {decode_cfg3.num_gpu_per_worker}"
    )
    assert decode_cfg3.moe_ep_list == [1, 2, 4, 8, 16, 32, 64], (
        f"Expected [1, 2, 4, 8, 16, 32, 64], got {decode_cfg3.moe_ep_list}"
    )

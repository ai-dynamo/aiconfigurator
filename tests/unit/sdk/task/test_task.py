# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pathlib
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

pytestmark = pytest.mark.unit


def _find_repo_root(start: pathlib.Path) -> pathlib.Path:
    """Find repository root.

    In the Docker test image we copy `src/` and `tests/` into `/workspace/` but do
    not copy `pyproject.toml`, so we detect the repo root via `src/aiconfigurator/`.
    """
    start = start.resolve()
    for parent in [start, *start.parents]:
        if (parent / "src" / "aiconfigurator").is_dir():
            return parent
    raise RuntimeError("Cannot find repository root (expected src/aiconfigurator/)")


_SRC = _find_repo_root(pathlib.Path(__file__)) / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

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
    task = TaskConfig(serving_mode="agg", model_path="Qwen/Qwen3-32B", system_name="h200_sxm")
    cfg = task.config

    assert cfg.worker_config.system_name == "h200_sxm"
    assert _enum_name(cfg.worker_config.gemm_quant_mode) == "float16"
    assert cfg.worker_config.num_gpu_per_worker == [1, 2, 4, 8]
    assert cfg.applied_layers == ["base-common", "agg-defaults"]


def test_taskconfig_disagg_default():
    task = TaskConfig(serving_mode="disagg", model_path="Qwen/Qwen3-32B", system_name="h200_sxm")
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
        model_path="Qwen/Qwen3-32B",
        system_name="h200_sxm",
        profiles=["fp8"],
    )
    cfg = task.config

    assert _enum_name(cfg.worker_config.gemm_quant_mode) == "fp8"
    assert any(layer.startswith("profile:fp8") for layer in cfg.applied_layers)


def test_taskconfig_fp8_static_requires_trtllm_backend():
    with pytest.raises(ValueError, match=r"fp8_static"):
        TaskConfig(
            serving_mode="agg",
            model_path="Qwen/Qwen3-32B",
            system_name="h200_sxm",
            backend_name="sglang",
            yaml_config={
                "mode": "patch",
                "config": {
                    "worker_config": {
                        "gemm_quant_mode": "fp8_static",
                    }
                },
            },
        )


def test_taskconfig_total_gpus_limits_agg_workers():
    task = TaskConfig(
        serving_mode="agg",
        model_path="Qwen/Qwen3-32B",
        system_name="h200_sxm",
        total_gpus=2,
    )
    cfg = task.config

    assert cfg.worker_config.num_gpu_per_worker == [1, 2]


def test_taskconfig_agg_yaml_patch_overrides():
    task = TaskConfig(
        serving_mode="agg",
        model_path="Qwen/Qwen3-32B",
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
        "profiles": ["float16"],
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
        model_path="Qwen/Qwen3-32B",
        system_name="h200_sxm",
        profiles=["fp8"],
        yaml_config=loaded_yaml,
    )
    cfg = task.config

    assert cfg.worker_config.tp_list == [1, 2]
    assert _enum_name(cfg.worker_config.gemm_quant_mode) == "float16"
    assert cfg.applied_layers[-1] == "yaml_patch"


def test_taskconfig_disagg_profile_patch_expands_replica():
    task = TaskConfig(
        serving_mode="disagg",
        model_path="Qwen/Qwen3-32B",
        system_name="b200_sxm",
        profiles=["fp8"],
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
        model_path="Qwen/Qwen3-32B",
        system_name="h200_sxm",
        total_gpus=16,
    )
    cfg = task.config

    assert cfg.replica_config.max_gpu_per_replica == 16


def test_taskconfig_disagg_total_gpus_with_patch():
    task = TaskConfig(
        serving_mode="disagg",
        model_path="Qwen/Qwen3-32B",
        system_name="h200_sxm",
        total_gpus=32,
        profiles=["fp8"],
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
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="gb200",
        enable_wideep=True,
    )
    cfg = task.config

    assert cfg.is_moe is True
    assert cfg.replica_config.max_gpu_per_replica == 512
    assert cfg.prefill_worker_config.num_gpu_per_worker[-1] == 32
    assert cfg.decode_worker_config.num_gpu_per_worker[-1] == 64


def test_taskconfig_disagg_decode_system_name_override():
    task = TaskConfig(
        serving_mode="disagg",
        model_path="Qwen/Qwen3-32B",
        system_name="h200_sxm",
        decode_system_name="b200_sxm",
    )
    cfg = task.config

    assert cfg.prefill_worker_config.system_name == "h200_sxm"
    assert cfg.decode_worker_config.system_name == "b200_sxm"


def test_taskconfig_agg_total_gpus_negative_rejected():
    with pytest.raises(ValueError):
        TaskConfig(
            serving_mode="agg",
            model_path="Qwen/Qwen3-32B",
            system_name="h200_sxm",
            total_gpus=-1,
        )


def test_taskconfig_disagg_total_gpus_small_rejected():
    with pytest.raises(ValueError):
        TaskConfig(
            serving_mode="disagg",
            model_path="Qwen/Qwen3-32B",
            system_name="h200_sxm",
            total_gpus=1,
        )


def test_taskconfig_yaml_replace_uses_full_config():
    base = TaskConfig(serving_mode="agg", model_path="Qwen/Qwen3-32B", system_name="h200_sxm")
    replaced_config = base.config.toDict()
    replaced_config["worker_config"]["system_name"] = "b200_sxm"
    replaced_config.pop("applied_layers", None)

    task = TaskConfig(
        serving_mode="agg",
        model_path="Qwen/Qwen3-32B",
        system_name="h200_sxm",
        yaml_config={"mode": "replace", "config": replaced_config},
    )
    cfg = task.config

    assert cfg.worker_config.system_name == "b200_sxm"
    assert cfg.applied_layers[-1] == "yaml_replace"


def test_taskconfig_rejects_unsupported_quant_mode(monkeypatch):
    class FakeDatabase:
        def __init__(self):
            self.system_spec = {"gpu": {"sm_version": 90}}
            self.supported_quant_mode = {"gemm": ["float16"]}

    def fake_get_database(system, backend, version):
        return FakeDatabase()

    monkeypatch.setattr(task_module, "get_database", fake_get_database)

    with pytest.raises(ValueError, match=r"Unsupported gemm quant mode"):
        TaskConfig(
            serving_mode="agg",
            model_path="Qwen/Qwen3-32B",
            system_name="h200_sxm",
            profiles=["fp8"],
        )


def test_taskconfig_quant_merge_uses_model_info_when_missing(monkeypatch):
    class FakeDatabase:
        def __init__(self):
            self.system_spec = {"gpu": {"sm_version": 90}}
            self.supported_quant_mode = {
                "gemm": ["float16"],
                "moe": ["float16"],
                "context_attention": ["float16"],
                "generation_attention": ["float16"],
            }

    def fake_get_database(system, backend, version):
        return FakeDatabase()

    def fake_model_info(_path):
        return {
            "raw_config": {"quant_algo": "fp8", "quant_dynamic": True},
            "architecture": "LlamaForCausalLM",
        }

    monkeypatch.setattr(task_module, "get_database", fake_get_database)
    monkeypatch.setattr(task_module, "get_model_config_from_model_path", fake_model_info)

    with pytest.raises(ValueError, match=r"Unsupported gemm quant mode 'fp8'"):
        TaskConfig(
            serving_mode="agg",
            model_path="Qwen/Qwen3-32B",
            system_name="h200_sxm",
            backend_name="trtllm",
        )


def test_taskconfig_quant_merge_preserves_explicit_values(monkeypatch):
    class FakeDatabase:
        def __init__(self):
            self.system_spec = {"gpu": {"sm_version": 90}}
            self.supported_quant_mode = {
                "gemm": ["float16"],
                "moe": ["float16"],
                "context_attention": ["float16"],
                "generation_attention": ["float16"],
            }

    def fake_get_database(system, backend, version):
        return FakeDatabase()

    def fake_model_info(_path):
        return {
            "raw_config": {"quant_algo": "fp8", "quant_dynamic": True},
            "architecture": "LlamaForCausalLM",
        }

    monkeypatch.setattr(task_module, "get_database", fake_get_database)
    monkeypatch.setattr(task_module, "get_model_config_from_model_path", fake_model_info)

    TaskConfig(
        serving_mode="agg",
        model_path="Qwen/Qwen3-32B",
        system_name="h200_sxm",
        backend_name="trtllm",
        yaml_config={
            "mode": "patch",
            "config": {
                "worker_config": {
                    "gemm_quant_mode": "float16",
                    "moe_quant_mode": "float16",
                    "kvcache_quant_mode": "float16",
                    "fmha_quant_mode": "float16",
                }
            },
        },
    )


def test_taskconfig_quant_merge_deepseek_fmha_fallback(monkeypatch):
    class FakeDatabase:
        def __init__(self):
            self.system_spec = {"gpu": {"sm_version": 90}}
            self.supported_quant_mode = {
                "gemm": ["fp8"],
                "moe": ["fp8"],
                "context_mla": ["float16"],
                "generation_mla": ["fp8"],
            }

    def fake_get_database(system, backend, version):
        return FakeDatabase()

    def fake_model_info(_path):
        return {
            "raw_config": {"quant_algo": "fp8", "quant_dynamic": True},
            "architecture": "DeepseekV3ForCausalLM",
        }

    monkeypatch.setattr(task_module, "get_database", fake_get_database)
    monkeypatch.setattr(task_module, "get_model_config_from_model_path", fake_model_info)

    TaskConfig(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
        backend_name="trtllm",
    )


def test_taskrunner_runs_agg_and_disagg():
    agg_task = TaskConfig(serving_mode="agg", model_path="Qwen/Qwen3-32B", system_name="h200_sxm")
    disagg_task = TaskConfig(serving_mode="disagg", model_path="Qwen/Qwen3-32B", system_name="h200_sxm", total_gpus=8)

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
        model_path="deepseek-ai/DeepSeek-V3",
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
        model_path="deepseek-ai/DeepSeek-V3",
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
    # task_trtllm_wideep = TaskConfig(
    #     serving_mode="disagg",
    #     model_path="deepseek-ai/DeepSeek-V3",
    #     system_name="h200_sxm",
    #     backend_name="trtllm",
    #     enable_wideep=True,
    #     total_gpus=64,
    # )

    # prefill_cfg3 = task_trtllm_wideep.config.prefill_worker_config
    # decode_cfg3 = task_trtllm_wideep.config.decode_worker_config

    # # Verify trtllm uses previous wideep logic
    # assert prefill_cfg3.num_gpu_per_worker == [1, 2, 4, 8, 16, 32], (
    #     f"Expected [1, 2, 4, 8, 16, 32], got {prefill_cfg3.num_gpu_per_worker}"
    # )
    # assert prefill_cfg3.moe_ep_list == [1, 2, 4, 8, 16, 32], (
    #     f"Expected [1, 2, 4, 8, 16, 32], got {prefill_cfg3.moe_ep_list}"
    # )
    # assert decode_cfg3.num_gpu_per_worker == [1, 2, 4, 8, 16, 32, 64], (
    #     f"Expected [1, 2, 4, 8, 16, 32, 64], got {decode_cfg3.num_gpu_per_worker}"
    # )
    # assert decode_cfg3.moe_ep_list == [1, 2, 4, 8, 16, 32, 64], (
    #     f"Expected [1, 2, 4, 8, 16, 32, 64], got {decode_cfg3.moe_ep_list}"
    # )


@pytest.mark.unit
def test_trtllm_moe_configs():
    """Test trtllm MoE WideEP configurations on gb200."""
    # Test 1: trtllm + MoE + wideep + agg on gb200
    task_agg = TaskConfig(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="gb200",
        backend_name="trtllm",
        enable_wideep=True,
        total_gpus=64,
    )

    agg_cfg = task_agg.config.worker_config

    # WideEP on gb200: dp > 1, moe_ep > 1
    assert agg_cfg.num_gpu_per_worker == [2, 4, 8, 16, 32, 64], (
        f"Expected [2, 4, 8, 16, 32, 64], got {agg_cfg.num_gpu_per_worker}"
    )
    assert agg_cfg.dp_list == [2, 4, 8, 16, 32, 64], f"Expected [2, 4, 8, 16, 32, 64], got {agg_cfg.dp_list}"
    assert agg_cfg.moe_tp_list == [1], f"Expected [1], got {agg_cfg.moe_tp_list}"
    assert agg_cfg.moe_ep_list == [2, 4, 8, 16, 32, 64], f"Expected [2, 4, 8, 16, 32, 64], got {agg_cfg.moe_ep_list}"

    # Test 2: trtllm + MoE + wideep + disagg on gb200
    task_disagg = TaskConfig(
        serving_mode="disagg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="gb200",
        backend_name="trtllm",
        enable_wideep=True,
        total_gpus=64,
    )

    prefill_cfg = task_disagg.config.prefill_worker_config
    decode_cfg = task_disagg.config.decode_worker_config

    # Verify prefill config
    assert prefill_cfg.num_gpu_per_worker == [4, 8, 16, 32], (
        f"Expected [4, 8, 16, 32], got {prefill_cfg.num_gpu_per_worker}"
    )
    assert prefill_cfg.dp_list == [4, 8, 16, 32], f"Expected [4, 8, 16, 32], got {prefill_cfg.dp_list}"
    assert prefill_cfg.moe_tp_list == [1], f"Expected [1], got {prefill_cfg.moe_tp_list}"
    assert prefill_cfg.moe_ep_list == [4, 8, 16, 32], f"Expected [4, 8, 16, 32], got {prefill_cfg.moe_ep_list}"

    # Verify decode config
    assert decode_cfg.num_gpu_per_worker == [4, 8, 16, 32, 64], (
        f"Expected [4, 8, 16, 32, 64], got {decode_cfg.num_gpu_per_worker}"
    )
    assert decode_cfg.dp_list == [4, 8, 16, 32, 64], f"Expected [4, 8, 16, 32, 64], got {decode_cfg.dp_list}"
    assert decode_cfg.moe_tp_list == [1], f"Expected [1], got {decode_cfg.moe_tp_list}"
    assert decode_cfg.moe_ep_list == [4, 8, 16, 32, 64], f"Expected [4, 8, 16, 32, 64], got {decode_cfg.moe_ep_list}"


def _make_capturing_disagg_pareto(captured: dict):
    """Helper: return a fake disagg_pareto that records its kwargs."""

    def _fn(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame({"tokens/s/user": [0.8], "tokens/s/gpu": [0.4]})

    return _fn


def test_sglang_non_wideep_disagg_requires_same_tp(monkeypatch):
    """SGLang non-wideep disagg must pass require_same_tp=True to disagg_pareto."""
    captured = {}
    pa_stub = sys.modules["aiconfigurator.sdk.pareto_analysis"]
    monkeypatch.setattr(pa_stub, "disagg_pareto", _make_capturing_disagg_pareto(captured))

    task = TaskConfig(
        serving_mode="disagg",
        model_path="Qwen/Qwen3-32B",
        system_name="h200_sxm",
        backend_name="sglang",
        enable_wideep=False,
        total_gpus=8,
    )
    TaskRunner().run(task)

    assert captured.get("require_same_tp") is True, (
        f"Expected require_same_tp=True for sglang non-wideep disagg, got {captured.get('require_same_tp')}"
    )


def test_sglang_wideep_disagg_does_not_require_same_tp(monkeypatch):
    """SGLang wideep disagg should NOT enforce same TP."""
    captured = {}
    pa_stub = sys.modules["aiconfigurator.sdk.pareto_analysis"]
    monkeypatch.setattr(pa_stub, "disagg_pareto", _make_capturing_disagg_pareto(captured))

    task = TaskConfig(
        serving_mode="disagg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
        backend_name="sglang",
        enable_wideep=True,
        total_gpus=64,
    )
    TaskRunner().run(task)

    assert captured.get("require_same_tp") is False, (
        f"Expected require_same_tp=False for sglang wideep disagg, got {captured.get('require_same_tp')}"
    )


def test_trtllm_disagg_does_not_require_same_tp(monkeypatch):
    """TRT-LLM disagg should NOT enforce same TP."""
    captured = {}
    pa_stub = sys.modules["aiconfigurator.sdk.pareto_analysis"]
    monkeypatch.setattr(pa_stub, "disagg_pareto", _make_capturing_disagg_pareto(captured))

    task = TaskConfig(
        serving_mode="disagg",
        model_path="Qwen/Qwen3-32B",
        system_name="h200_sxm",
        backend_name="trtllm",
        total_gpus=8,
    )
    TaskRunner().run(task)

    assert captured.get("require_same_tp") is False, (
        f"Expected require_same_tp=False for trtllm disagg, got {captured.get('require_same_tp')}"
    )


def test_vllm_disagg_does_not_require_same_tp(monkeypatch):
    """vLLM disagg should NOT enforce same TP."""
    captured = {}
    pa_stub = sys.modules["aiconfigurator.sdk.pareto_analysis"]
    monkeypatch.setattr(pa_stub, "disagg_pareto", _make_capturing_disagg_pareto(captured))

    task = TaskConfig(
        serving_mode="disagg",
        model_path="Qwen/Qwen3-32B",
        system_name="h200_sxm",
        backend_name="vllm",
        total_gpus=8,
    )
    TaskRunner().run(task)

    assert captured.get("require_same_tp") is False, (
        f"Expected require_same_tp=False for vllm disagg, got {captured.get('require_same_tp')}"
    )

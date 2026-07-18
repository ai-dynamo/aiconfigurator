# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from collector.fpm_forward.config import reject_fpm_arguments_without_fpm
from collector.fpm_forward.entry import _load_generator_overrides, resolve_inputs, run_resolved

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]


def _generator_args(**overrides):
    values = {
        "generator_config": None,
        "generator_set": None,
        "generator_dynamo_version": None,
        "generated_config_version": None,
        "namespace": None,
        "transport": None,
        "image_pull_secret": None,
        "model_cache": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_fpm_generator_config_accepts_only_deployment_fields(tmp_path):
    config = tmp_path / "deployment.yaml"
    config.write_text(
        """K8sConfig:
  k8s_image: example/vllm-runtime:test
  k8s_pvc_name: model-cache
  k8s_pvc_mount_path: /models
  k8s_model_path_in_pvc: glm
  fpm_shared_memory_size: 200Gi
"""
    )

    resolved = _load_generator_overrides(
        _generator_args(generator_config=str(config), generator_dynamo_version="1.2.0")
    )

    assert resolved["K8sConfig"]["k8s_image"] == "example/vllm-runtime:test"
    assert resolved["generator_dynamo_version"] == "1.2.0"


def test_fpm_generator_config_rejects_collector_owned_engine_fields(tmp_path):
    config = tmp_path / "invalid.yaml"
    config.write_text("params:\n  agg:\n    gpu_memory_utilization: 0.86\n")

    with pytest.raises(ValueError, match="deployment-only"):
        _load_generator_overrides(_generator_args(generator_config=str(config)))

    with pytest.raises(ValueError, match="resolves generated_config_version"):
        _load_generator_overrides(_generator_args(generated_config_version="0.20.1"))


def test_normal_op_arguments_do_not_activate_fpm():
    args = argparse.Namespace(
        ops=["gemm"],
        fpm_max_gpus=None,
        fpm_gpu_counts=None,
        fpm_weight_quantizations=None,
        fpm_kv_cache_dtypes=None,
        fpm_tp_sizes=None,
        fpm_pp_sizes=None,
        fpm_dp_sizes=None,
        fpm_moe_tp_sizes=None,
        fpm_database_root=None,
        fpm_moe_ep_sizes=None,
        fpm_cp_sizes=None,
    )
    reject_fpm_arguments_without_fpm(args)


def test_fpm_only_arguments_are_rejected_for_normal_ops():
    args = argparse.Namespace(
        ops=["gemm"],
        fpm_max_gpus=4,
        fpm_gpu_counts=None,
        fpm_weight_quantizations=None,
        fpm_kv_cache_dtypes=None,
        fpm_tp_sizes=None,
        fpm_pp_sizes=None,
        fpm_dp_sizes=None,
        fpm_moe_tp_sizes=None,
        fpm_database_root=None,
        fpm_moe_ep_sizes=None,
        fpm_cp_sizes=None,
    )
    with pytest.raises(ValueError, match="require --ops fpm_forward"):
        reject_fpm_arguments_without_fpm(args)


def test_fpm_model_config_is_rejected_for_normal_ops():
    args = argparse.Namespace(ops=["gemm"], fpm_model_config="/tmp/config.json")

    with pytest.raises(ValueError, match="--fpm-model-config"):
        reject_fpm_arguments_without_fpm(args)


def test_resolve_inputs_passes_explicit_model_config_to_planner(monkeypatch):
    captured = {}

    def build_plan(**kwargs):
        captured.update(kwargs)
        return "plan"

    monkeypatch.setattr("collector.fpm_forward.entry.build_collection_plan", build_plan)
    args = _generator_args()
    args.backend = "vllm"
    args.gpu = "b200_sxm"
    args.fpm_max_gpus = 4
    args.fpm_model_config = "/configs/private-model.json"
    case_plan = SimpleNamespace(
        model_path="private-org/model",
        model_architecture="UnknownForCausalLM",
        selected_ops={"attention_context", "attention_generation"},
        model_cases_paths=[],
    )

    plan, overrides = resolve_inputs(args, case_plan)

    assert plan == "plan"
    assert overrides == {}
    assert captured["model_config_path"] == "/configs/private-model.json"


@pytest.mark.parametrize("plan_only", [False, True])
def test_fpm_entry_reports_missing_generator_config_without_traceback(tmp_path, plan_only):
    command = [
        sys.executable,
        str(REPO_ROOT / "collector" / "collect.py"),
        "--backend",
        "vllm",
        "--ops",
        "fpm_forward",
        "--model-path",
        "nvidia/GLM-5.2-NVFP4",
        "--gpu",
        "b200_sxm",
        "--fpm-max-gpus",
        "4",
        "--generator-config",
        str(tmp_path / "missing.yaml"),
    ]
    if plan_only:
        command.append("--plan-only")

    completed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30, check=False)

    assert completed.returncode == 2
    assert "collect.py: error:" in completed.stderr
    assert "missing.yaml" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_resolved_execution_preserves_runtime_failure(monkeypatch):
    def fail_collection(*args, **kwargs):
        raise RuntimeError("cluster execution failed")

    monkeypatch.setattr("collector.fpm_forward.runner.run_collection", fail_collection)
    args = SimpleNamespace(
        checkpoint_dir="checkpoint",
        fpm_artifact_root=None,
        resume=False,
        resume_retry_failed=False,
        smoke=False,
        limit=None,
        fpm_database_root=None,
    )

    with pytest.raises(RuntimeError, match="cluster execution failed"):
        run_resolved(args, (object(), {}))

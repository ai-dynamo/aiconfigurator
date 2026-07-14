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
from collector.fpm_forward.entry import run_resolved

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]


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
        fpm_smoke_points=None,
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
        fpm_smoke_points=None,
        fpm_database_root=None,
        fpm_moe_ep_sizes=None,
        fpm_cp_sizes=None,
    )
    with pytest.raises(ValueError, match="require --ops fpm_forward"):
        reject_fpm_arguments_without_fpm(args)


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

    completed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)

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
        run_resolved(args, (object(), {}, None))

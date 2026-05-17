# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration parity test: legacy CLI path vs new sweep path.

Verifies that the new sdk/sweep.py + sdk/task_config.py pipeline produces
the same Pareto DataFrame as the legacy
sdk.task.TaskRunner -> sdk.pareto_analysis.agg_pareto/disagg_pareto
pipeline when fed the same YAML configuration.

This test requires real perf databases on disk and (on first run) network
access to fetch HuggingFace model configs.  It is marked `integration`
so it does not run in the default unit-test pass.

To run:
    pytest tests/integration/test_old_vs_new_parity.py -m integration

If HF_TOKEN is not set or the perf DB is unavailable, the test is skipped.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.integration


# Two small, tightly-bounded YAML scenarios — keep runtime under a couple
# of minutes each by constraining the parallel search space.
AGG_YAML: dict = {
    "serving_mode": "agg",
    "model_path": "Qwen/Qwen3-32B",
    "system_name": "h200_sxm",
    "backend_name": "trtllm",
    "backend_version": "1.3.0rc10",
    "total_gpus": 8,
    "isl": 4000,
    "osl": 500,
    "ttft": 1000.0,
    "tpot": 50.0,
    "config": {
        "worker_config": {
            "num_gpu_per_worker": [1, 2],
            "tp_list": [1, 2],
            "pp_list": [1],
            "dp_list": [1],
            "moe_tp_list": [1],
            "moe_ep_list": [1],
        },
    },
}


DISAGG_YAML: dict = {
    "serving_mode": "disagg",
    "model_path": "Qwen/Qwen3-32B",
    "system_name": "h200_sxm",
    "backend_name": "trtllm",
    "backend_version": "1.3.0rc10",
    "total_gpus": 16,
    "isl": 4000,
    "osl": 500,
    "ttft": 1000.0,
    "tpot": 50.0,
    "config": {
        "prefill_worker_config": {
            "num_gpu_per_worker": [1, 2],
            "tp_list": [1, 2],
            "pp_list": [1],
            "dp_list": [1],
            "moe_tp_list": [1],
            "moe_ep_list": [1],
        },
        "decode_worker_config": {
            "num_gpu_per_worker": [1, 2],
            "tp_list": [1, 2],
            "pp_list": [1],
            "dp_list": [1],
            "moe_tp_list": [1],
            "moe_ep_list": [1],
        },
        "replica_config": {
            "num_gpu_per_replica": [2, 4, 8],
            "max_gpu_per_replica": 8,
            "max_prefill_worker": 2,
            "max_decode_worker": 2,
        },
    },
}


def _skip_if_no_db(system: str, backend: str, version: str) -> None:
    """Skip the test when the requested perf DB is not available locally."""
    db_path = Path(__file__).parents[2] / "src" / "aiconfigurator" / "systems" / "data" / system / backend / version
    if not db_path.exists():
        pytest.skip(f"perf database not available: {db_path}")


def _run_old_path_agg(yaml_data: dict) -> pd.DataFrame:
    """Invoke the legacy TaskConfig + TaskRunner path programmatically."""
    from aiconfigurator.sdk.task import TaskConfig as V1TaskConfig
    from aiconfigurator.sdk.task import TaskRunner

    v1 = V1TaskConfig(
        serving_mode=yaml_data["serving_mode"],
        model_path=yaml_data["model_path"],
        system_name=yaml_data["system_name"],
        backend_name=yaml_data.get("backend_name", "trtllm"),
        backend_version=yaml_data.get("backend_version"),
        total_gpus=yaml_data["total_gpus"],
        isl=yaml_data.get("isl"),
        osl=yaml_data.get("osl"),
        ttft=yaml_data.get("ttft"),
        tpot=yaml_data.get("tpot"),
        yaml_config={"config": yaml_data.get("config", {})},
    )
    runner = TaskRunner()
    result = runner.run(v1)
    return result["pareto_df"]


def _run_old_path_disagg(yaml_data: dict) -> pd.DataFrame:
    from aiconfigurator.sdk.task import TaskConfig as V1TaskConfig
    from aiconfigurator.sdk.task import TaskRunner

    v1 = V1TaskConfig(
        serving_mode=yaml_data["serving_mode"],
        model_path=yaml_data["model_path"],
        system_name=yaml_data["system_name"],
        decode_system_name=yaml_data["system_name"],
        backend_name=yaml_data.get("backend_name", "trtllm"),
        backend_version=yaml_data.get("backend_version"),
        total_gpus=yaml_data["total_gpus"],
        isl=yaml_data.get("isl"),
        osl=yaml_data.get("osl"),
        ttft=yaml_data.get("ttft"),
        tpot=yaml_data.get("tpot"),
        yaml_config={"config": yaml_data.get("config", {})},
    )
    runner = TaskRunner()
    result = runner.run(v1)
    return result["pareto_df"]


def _run_new_path_agg(yaml_data: dict) -> pd.DataFrame:
    """Invoke the new TaskConfig + sweep_agg path programmatically.

    Uses the same wide tpot sweep list TaskRunner sets internally, to
    match the legacy path's parallel + tpot enumeration.
    """
    from aiconfigurator.sdk.perf_database import get_database
    from aiconfigurator.sdk.sweep import sweep_agg
    from aiconfigurator.sdk.task_config import TaskConfig

    task = TaskConfig.from_yaml(yaml_data)
    db = get_database(task.system_name, task.backend_name, task.backend_version)
    kwargs = task.sweep_agg_kwargs(database=db)
    # Match legacy TaskRunner.run_agg's hardcoded tpot list.
    legacy_tpot_list = list(range(1, 20, 1)) + list(range(20, 300, 5))
    kwargs["runtime_config"].tpot = legacy_tpot_list
    return sweep_agg(**kwargs)


def _run_new_path_disagg(yaml_data: dict) -> pd.DataFrame:
    from aiconfigurator.sdk.perf_database import get_database
    from aiconfigurator.sdk.sweep import sweep_disagg
    from aiconfigurator.sdk.task_config import TaskConfig

    task = TaskConfig.from_yaml(yaml_data)
    prefill_db = get_database(task.prefill_system_name, task.prefill_backend_name, task.prefill_backend_version)
    decode_db = get_database(task.decode_system_name, task.decode_backend_name, task.decode_backend_version)
    kwargs = task.sweep_disagg_kwargs(prefill_database=prefill_db, decode_database=decode_db)
    legacy_tpot_list = list(range(1, 20, 1)) + list(range(20, 300, 5))
    kwargs["runtime_config"].tpot = legacy_tpot_list
    return sweep_disagg(**kwargs)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Drop object columns and sort by a stable key for byte-comparison."""
    drop_cols = [c for c in df.columns if c.startswith("_")]
    out = df.drop(columns=drop_cols, errors="ignore").copy()
    # Pick the first available stable sort key set.
    for key_set in (
        ["parallel", "bs", "global_bs", "tpot"],
        ["(p)parallel", "(d)parallel", "(p)bs", "(d)bs", "(p)workers", "(d)workers"],
    ):
        if all(c in out.columns for c in key_set):
            out = out.sort_values(key_set).reset_index(drop=True)
            break
    return out.round(3)


def test_old_vs_new_pareto_parity_agg():
    yaml_data = copy.deepcopy(AGG_YAML)
    _skip_if_no_db(yaml_data["system_name"], yaml_data["backend_name"], yaml_data["backend_version"])

    old_df = _run_old_path_agg(yaml_data)
    new_df = _run_new_path_agg(yaml_data)

    assert old_df is not None and len(old_df) > 0, "legacy path returned empty DataFrame"
    assert len(new_df) > 0, "new path returned empty DataFrame"

    old_norm = _normalize(old_df)
    new_norm = _normalize(new_df)
    pd.testing.assert_frame_equal(old_norm, new_norm, check_like=True)


def test_old_vs_new_pareto_parity_disagg():
    yaml_data = copy.deepcopy(DISAGG_YAML)
    _skip_if_no_db(yaml_data["system_name"], yaml_data["backend_name"], yaml_data["backend_version"])

    old_df = _run_old_path_disagg(yaml_data)
    new_df = _run_new_path_disagg(yaml_data)

    assert old_df is not None and len(old_df) > 0, "legacy path returned empty DataFrame"
    assert len(new_df) > 0, "new path returned empty DataFrame"

    old_norm = _normalize(old_df)
    new_norm = _normalize(new_df)
    pd.testing.assert_frame_equal(old_norm, new_norm, check_like=True)

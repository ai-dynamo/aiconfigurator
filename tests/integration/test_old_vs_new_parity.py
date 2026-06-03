# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration parity test: legacy CLI path vs new sweep path.

Verifies that the new sdk/sweep.py + sdk/task_v2.py pipeline produces
the same Pareto DataFrame as the legacy
sdk.task.TaskRunner -> sdk.pareto_analysis.agg_pareto/disagg_pareto
pipeline when describing the SAME task.

Each side is constructed via its native interface:
- Old: V1 TaskConfig + yaml_config nested dict, executed by TaskRunner.run
- New: Task (flat fields), executed by task.run() which dispatches internally

The two sides MUST produce byte-equal Pareto DataFrames (after dropping
object columns, sorting on a stable key, and rounding to 3 decimals).

This test requires real perf databases on disk and (on first run) network
access to fetch HuggingFace model configs.  Marked pytest.mark.integration
so it does not run in the default unit-test pass.

Run with:
    pytest tests/integration/test_old_vs_new_parity.py -m integration
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.integration


# Shared scenario parameters in role-prefixed form.  Each side translates
# this dict to its own native YAML/kwarg shape.

AGG_PARAMS: dict = {
    "model_path": "Qwen/Qwen3-32B",
    "system_name": "h200_sxm",
    "backend_name": "trtllm",
    "backend_version": "1.3.0rc10",
    "total_gpus": 8,
    "isl": 4000,
    "osl": 500,
    "ttft": 1000.0,
    "tpot": 50.0,
    # Tight search space to keep runtime <30s
    "agg_num_gpu_candidates": [1, 2],
    "agg_tp_candidates": [1, 2],
    "agg_pp_candidates": [1],
    "agg_dp_candidates": [1],
    "agg_moe_tp_candidates": [1],
    "agg_moe_ep_candidates": [1],
}


DISAGG_PARAMS: dict = {
    "model_path": "Qwen/Qwen3-32B",
    "system_name": "h200_sxm",
    "backend_name": "trtllm",
    "backend_version": "1.3.0rc10",
    "total_gpus": 16,
    "isl": 4000,
    "osl": 500,
    "ttft": 1000.0,
    "tpot": 50.0,
    # Tight search space
    "prefill_num_gpu_candidates": [1, 2],
    "prefill_tp_candidates": [1, 2],
    "prefill_pp_candidates": [1],
    "prefill_dp_candidates": [1],
    "prefill_moe_tp_candidates": [1],
    "prefill_moe_ep_candidates": [1],
    "decode_num_gpu_candidates": [1, 2],
    "decode_tp_candidates": [1, 2],
    "decode_pp_candidates": [1],
    "decode_dp_candidates": [1],
    "decode_moe_tp_candidates": [1],
    "decode_moe_ep_candidates": [1],
    "num_gpu_per_replica": [2, 4, 8],
    "max_gpu_per_replica": 8,
    "max_prefill_workers": 2,
    "max_decode_workers": 2,
}


def _skip_if_no_db(system: str, backend: str, version: str) -> None:
    db_path = Path(__file__).parents[2] / "src" / "aiconfigurator" / "systems" / "data" / system / backend / version
    if not db_path.exists():
        pytest.skip(f"perf database not available: {db_path}")


# ---------------------------------------------------------------------------
# Old path adapters — translate flat params into V1 TaskConfig + yaml_config
# ---------------------------------------------------------------------------


def _old_path_agg(params: dict) -> pd.DataFrame:
    from aiconfigurator.sdk.task import TaskConfig as V1TaskConfig
    from aiconfigurator.sdk.task import TaskRunner

    yaml_config = {
        "config": {
            "worker_config": {
                "num_gpu_per_worker": params["agg_num_gpu_candidates"],
                "tp_list": params["agg_tp_candidates"],
                "pp_list": params["agg_pp_candidates"],
                "dp_list": params["agg_dp_candidates"],
                "moe_tp_list": params["agg_moe_tp_candidates"],
                "moe_ep_list": params["agg_moe_ep_candidates"],
            },
        },
    }
    v1 = V1TaskConfig(
        serving_mode="agg",
        model_path=params["model_path"],
        system_name=params["system_name"],
        backend_name=params["backend_name"],
        backend_version=params["backend_version"],
        total_gpus=params["total_gpus"],
        isl=params["isl"],
        osl=params["osl"],
        ttft=params["ttft"],
        tpot=params["tpot"],
        yaml_config=yaml_config,
    )
    return TaskRunner().run(v1)["pareto_df"]


def _old_path_disagg(params: dict) -> pd.DataFrame:
    from aiconfigurator.sdk.task import TaskConfig as V1TaskConfig
    from aiconfigurator.sdk.task import TaskRunner

    yaml_config = {
        "config": {
            "prefill_worker_config": {
                "num_gpu_per_worker": params["prefill_num_gpu_candidates"],
                "tp_list": params["prefill_tp_candidates"],
                "pp_list": params["prefill_pp_candidates"],
                "dp_list": params["prefill_dp_candidates"],
                "moe_tp_list": params["prefill_moe_tp_candidates"],
                "moe_ep_list": params["prefill_moe_ep_candidates"],
            },
            "decode_worker_config": {
                "num_gpu_per_worker": params["decode_num_gpu_candidates"],
                "tp_list": params["decode_tp_candidates"],
                "pp_list": params["decode_pp_candidates"],
                "dp_list": params["decode_dp_candidates"],
                "moe_tp_list": params["decode_moe_tp_candidates"],
                "moe_ep_list": params["decode_moe_ep_candidates"],
            },
            "replica_config": {
                "num_gpu_per_replica": params["num_gpu_per_replica"],
                "max_gpu_per_replica": params["max_gpu_per_replica"],
                "max_prefill_worker": params["max_prefill_workers"],
                "max_decode_worker": params["max_decode_workers"],
            },
        },
    }
    v1 = V1TaskConfig(
        serving_mode="disagg",
        model_path=params["model_path"],
        system_name=params["system_name"],
        decode_system_name=params["system_name"],
        backend_name=params["backend_name"],
        backend_version=params["backend_version"],
        total_gpus=params["total_gpus"],
        isl=params["isl"],
        osl=params["osl"],
        ttft=params["ttft"],
        tpot=params["tpot"],
        yaml_config=yaml_config,
    )
    return TaskRunner().run(v1)["pareto_df"]


# ---------------------------------------------------------------------------
# New path adapters — flat YAML into Task, then task.run()
# ---------------------------------------------------------------------------


_LEGACY_TPOT_SWEEP = list(range(1, 20, 1)) + list(range(20, 300, 5))


def _new_path_agg(params: dict) -> pd.DataFrame:
    from aiconfigurator.sdk.task_v2 import Task

    yaml_data = {"serving_mode": "agg", **params}
    task = Task.from_yaml(yaml_data)
    # Match legacy TaskRunner.run_agg's hardcoded tpot sweep list so the
    # candidate set covers the same SLA domain.  task.tpot accepts either
    # a single float or a list -- sweep_agg's tpot expansion handles both.
    task.tpot = _LEGACY_TPOT_SWEEP
    return task.run()


def _new_path_disagg(params: dict) -> pd.DataFrame:
    from aiconfigurator.sdk.task_v2 import Task

    # Legacy disagg uses a single shared model_path + system_name; the new
    # Task requires explicit prefill_/decode_ fields.
    yaml_data: dict = {
        "serving_mode": "disagg",
        "isl": params["isl"],
        "osl": params["osl"],
        "ttft": params["ttft"],
        "tpot": params["tpot"],
        "total_gpus": params["total_gpus"],
        "num_gpu_per_replica": params["num_gpu_per_replica"],
        "max_gpu_per_replica": params["max_gpu_per_replica"],
        "max_prefill_workers": params["max_prefill_workers"],
        "max_decode_workers": params["max_decode_workers"],
    }
    for role in ("prefill", "decode"):
        yaml_data[f"{role}_model_path"] = params["model_path"]
        yaml_data[f"{role}_system_name"] = params["system_name"]
        yaml_data[f"{role}_backend_name"] = params["backend_name"]
        yaml_data[f"{role}_backend_version"] = params["backend_version"]
        for dim in ("num_gpu", "tp", "pp", "dp", "moe_tp", "moe_ep"):
            yaml_data[f"{role}_{dim}_candidates"] = params[f"{role}_{dim}_candidates"]

    task = Task.from_yaml(yaml_data)
    task.tpot = _LEGACY_TPOT_SWEEP
    return task.run()


# ---------------------------------------------------------------------------
# Comparison helper + tests
# ---------------------------------------------------------------------------


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in df.columns if c.startswith("_")]
    out = df.drop(columns=drop_cols, errors="ignore").copy()
    for key_set in (
        ["parallel", "bs", "global_bs", "tpot"],
        ["(p)parallel", "(d)parallel", "(p)bs", "(d)bs", "(p)workers", "(d)workers"],
    ):
        if all(c in out.columns for c in key_set):
            out = out.sort_values(key_set).reset_index(drop=True)
            break
    return out.round(3)


def test_old_vs_new_pareto_parity_agg():
    _skip_if_no_db(AGG_PARAMS["system_name"], AGG_PARAMS["backend_name"], AGG_PARAMS["backend_version"])

    old_df = _old_path_agg(AGG_PARAMS)
    new_df = _new_path_agg(AGG_PARAMS)

    assert old_df is not None and len(old_df) > 0, "legacy path returned empty DataFrame"
    assert len(new_df) > 0, "new path returned empty DataFrame"

    pd.testing.assert_frame_equal(_normalize(old_df), _normalize(new_df), check_like=True)


def test_old_vs_new_pareto_parity_disagg():
    _skip_if_no_db(DISAGG_PARAMS["system_name"], DISAGG_PARAMS["backend_name"], DISAGG_PARAMS["backend_version"])

    old_df = _old_path_disagg(DISAGG_PARAMS)
    new_df = _new_path_disagg(DISAGG_PARAMS)

    assert old_df is not None and len(old_df) > 0, "legacy path returned empty DataFrame"
    assert len(new_df) > 0, "new path returned empty DataFrame"

    pd.testing.assert_frame_equal(_normalize(old_df), _normalize(new_df), check_like=True)


# ---------------------------------------------------------------------------
# Extended parity tests: request_latency mode, autoscale, hetero-disagg
# ---------------------------------------------------------------------------


def _old_path_agg_request_latency(params: dict, request_latency: float) -> pd.DataFrame:
    """Old path with request_latency-driven (ttft, tpot) constraint pairs."""
    from aiconfigurator.sdk.task import TaskConfig as V1TaskConfig
    from aiconfigurator.sdk.task import TaskRunner

    yaml_config = {
        "config": {
            "worker_config": {
                "num_gpu_per_worker": params["agg_num_gpu_candidates"],
                "tp_list": params["agg_tp_candidates"],
                "pp_list": params["agg_pp_candidates"],
                "dp_list": params["agg_dp_candidates"],
                "moe_tp_list": params["agg_moe_tp_candidates"],
                "moe_ep_list": params["agg_moe_ep_candidates"],
            },
        },
    }
    v1 = V1TaskConfig(
        serving_mode="agg",
        model_path=params["model_path"],
        system_name=params["system_name"],
        backend_name=params["backend_name"],
        backend_version=params["backend_version"],
        total_gpus=params["total_gpus"],
        isl=params["isl"],
        osl=params["osl"],
        ttft=params["ttft"],
        tpot=params["tpot"],
        request_latency=request_latency,
        yaml_config=yaml_config,
    )
    return TaskRunner().run(v1)["pareto_df"]


def _new_path_agg_request_latency(params: dict, request_latency: float) -> pd.DataFrame:
    from aiconfigurator.sdk.task_v2 import Task

    yaml_data = {"serving_mode": "agg", **params, "request_latency": request_latency}
    task = Task.from_yaml(yaml_data)
    # In request_latency mode the SDK derives (ttft, tpot) pairs internally;
    # do NOT override task.tpot here.
    return task.run()


def test_old_vs_new_pareto_parity_agg_request_latency():
    """Verify request_latency-driven (ttft, tpot) constraint enumeration parity."""
    _skip_if_no_db(AGG_PARAMS["system_name"], AGG_PARAMS["backend_name"], AGG_PARAMS["backend_version"])
    request_latency = 10000.0  # 10s end-to-end

    old_df = _old_path_agg_request_latency(AGG_PARAMS, request_latency)
    new_df = _new_path_agg_request_latency(AGG_PARAMS, request_latency)

    assert old_df is not None and len(old_df) > 0, "legacy path returned empty DataFrame"
    assert len(new_df) > 0, "new path returned empty DataFrame"

    pd.testing.assert_frame_equal(_normalize(old_df), _normalize(new_df), check_like=True)


def _old_path_disagg_autoscale(params: dict) -> pd.DataFrame:
    """Old disagg autoscale path (picks prefill and decode independently)."""
    from aiconfigurator.sdk.task import TaskConfig as V1TaskConfig
    from aiconfigurator.sdk.task import TaskRunner

    yaml_config = {
        "config": {
            "prefill_worker_config": {
                "num_gpu_per_worker": params["prefill_num_gpu_candidates"],
                "tp_list": params["prefill_tp_candidates"],
                "pp_list": params["prefill_pp_candidates"],
                "dp_list": params["prefill_dp_candidates"],
                "moe_tp_list": params["prefill_moe_tp_candidates"],
                "moe_ep_list": params["prefill_moe_ep_candidates"],
            },
            "decode_worker_config": {
                "num_gpu_per_worker": params["decode_num_gpu_candidates"],
                "tp_list": params["decode_tp_candidates"],
                "pp_list": params["decode_pp_candidates"],
                "dp_list": params["decode_dp_candidates"],
                "moe_tp_list": params["decode_moe_tp_candidates"],
                "moe_ep_list": params["decode_moe_ep_candidates"],
            },
            "replica_config": {
                "num_gpu_per_replica": params["num_gpu_per_replica"],
                "max_gpu_per_replica": params["max_gpu_per_replica"],
                "max_prefill_worker": params["max_prefill_workers"],
                "max_decode_worker": params["max_decode_workers"],
            },
        },
    }
    v1 = V1TaskConfig(
        serving_mode="disagg",
        model_path=params["model_path"],
        system_name=params["system_name"],
        decode_system_name=params["system_name"],
        backend_name=params["backend_name"],
        backend_version=params["backend_version"],
        total_gpus=params["total_gpus"],
        isl=params["isl"],
        osl=params["osl"],
        ttft=params["ttft"],
        tpot=params["tpot"],
        yaml_config=yaml_config,
    )
    return TaskRunner().run(v1, autoscale=True)["pareto_df"]


def _new_path_disagg_autoscale(params: dict) -> pd.DataFrame:
    from aiconfigurator.sdk.task_v2 import Task

    yaml_data: dict = {
        "serving_mode": "disagg",
        "isl": params["isl"],
        "osl": params["osl"],
        "ttft": params["ttft"],
        "tpot": params["tpot"],
        "total_gpus": params["total_gpus"],
        "num_gpu_per_replica": params["num_gpu_per_replica"],
        "max_gpu_per_replica": params["max_gpu_per_replica"],
        "max_prefill_workers": params["max_prefill_workers"],
        "max_decode_workers": params["max_decode_workers"],
    }
    for role in ("prefill", "decode"):
        yaml_data[f"{role}_model_path"] = params["model_path"]
        yaml_data[f"{role}_system_name"] = params["system_name"]
        yaml_data[f"{role}_backend_name"] = params["backend_name"]
        yaml_data[f"{role}_backend_version"] = params["backend_version"]
        for dim in ("num_gpu", "tp", "pp", "dp", "moe_tp", "moe_ep"):
            yaml_data[f"{role}_{dim}_candidates"] = params[f"{role}_{dim}_candidates"]

    task = Task.from_yaml(yaml_data)
    task.tpot = _LEGACY_TPOT_SWEEP
    return task.run(autoscale=True)


def test_old_vs_new_pareto_parity_disagg_autoscale():
    """Verify autoscale path (picks P and D independently, no rate matching)."""
    _skip_if_no_db(DISAGG_PARAMS["system_name"], DISAGG_PARAMS["backend_name"], DISAGG_PARAMS["backend_version"])

    old_df = _old_path_disagg_autoscale(DISAGG_PARAMS)
    new_df = _new_path_disagg_autoscale(DISAGG_PARAMS)

    assert old_df is not None and len(old_df) > 0, "legacy path returned empty DataFrame"
    assert len(new_df) > 0, "new path returned empty DataFrame"

    pd.testing.assert_frame_equal(_normalize(old_df), _normalize(new_df), check_like=True)


HETERO_DISAGG_PARAMS = {
    **DISAGG_PARAMS,
    "prefill_system": "h200_sxm",
    "decode_system": "h100_sxm",
}


def _old_path_hetero_disagg(params: dict) -> pd.DataFrame:
    """Hetero-disagg: prefill on one system, decode on another."""
    from aiconfigurator.sdk.task import TaskConfig as V1TaskConfig
    from aiconfigurator.sdk.task import TaskRunner

    yaml_config = {
        "config": {
            "prefill_worker_config": {
                "num_gpu_per_worker": params["prefill_num_gpu_candidates"],
                "tp_list": params["prefill_tp_candidates"],
                "pp_list": params["prefill_pp_candidates"],
                "dp_list": params["prefill_dp_candidates"],
                "moe_tp_list": params["prefill_moe_tp_candidates"],
                "moe_ep_list": params["prefill_moe_ep_candidates"],
            },
            "decode_worker_config": {
                "num_gpu_per_worker": params["decode_num_gpu_candidates"],
                "tp_list": params["decode_tp_candidates"],
                "pp_list": params["decode_pp_candidates"],
                "dp_list": params["decode_dp_candidates"],
                "moe_tp_list": params["decode_moe_tp_candidates"],
                "moe_ep_list": params["decode_moe_ep_candidates"],
            },
            "replica_config": {
                "num_gpu_per_replica": params["num_gpu_per_replica"],
                "max_gpu_per_replica": params["max_gpu_per_replica"],
                "max_prefill_worker": params["max_prefill_workers"],
                "max_decode_worker": params["max_decode_workers"],
            },
        },
    }
    v1 = V1TaskConfig(
        serving_mode="disagg",
        model_path=params["model_path"],
        system_name=params["prefill_system"],
        decode_system_name=params["decode_system"],
        backend_name=params["backend_name"],
        backend_version=params["backend_version"],
        total_gpus=params["total_gpus"],
        isl=params["isl"],
        osl=params["osl"],
        ttft=params["ttft"],
        tpot=params["tpot"],
        yaml_config=yaml_config,
    )
    return TaskRunner().run(v1)["pareto_df"]


def _new_path_hetero_disagg(params: dict) -> pd.DataFrame:
    from aiconfigurator.sdk.task_v2 import Task

    yaml_data: dict = {
        "serving_mode": "disagg",
        "isl": params["isl"],
        "osl": params["osl"],
        "ttft": params["ttft"],
        "tpot": params["tpot"],
        "total_gpus": params["total_gpus"],
        "num_gpu_per_replica": params["num_gpu_per_replica"],
        "max_gpu_per_replica": params["max_gpu_per_replica"],
        "max_prefill_workers": params["max_prefill_workers"],
        "max_decode_workers": params["max_decode_workers"],
        "prefill_model_path": params["model_path"],
        "prefill_system_name": params["prefill_system"],
        "prefill_backend_name": params["backend_name"],
        "prefill_backend_version": params["backend_version"],
        "decode_model_path": params["model_path"],
        "decode_system_name": params["decode_system"],
        "decode_backend_name": params["backend_name"],
        "decode_backend_version": params["backend_version"],
    }
    for role in ("prefill", "decode"):
        for dim in ("num_gpu", "tp", "pp", "dp", "moe_tp", "moe_ep"):
            yaml_data[f"{role}_{dim}_candidates"] = params[f"{role}_{dim}_candidates"]

    task = Task.from_yaml(yaml_data)
    task.tpot = _LEGACY_TPOT_SWEEP
    return task.run()


def test_old_vs_new_pareto_parity_hetero_disagg():
    """Verify hetero-disagg path: prefill on h200_sxm, decode on h100_sxm."""
    _skip_if_no_db(
        HETERO_DISAGG_PARAMS["prefill_system"],
        HETERO_DISAGG_PARAMS["backend_name"],
        HETERO_DISAGG_PARAMS["backend_version"],
    )
    _skip_if_no_db(
        HETERO_DISAGG_PARAMS["decode_system"],
        HETERO_DISAGG_PARAMS["backend_name"],
        HETERO_DISAGG_PARAMS["backend_version"],
    )

    old_df = _old_path_hetero_disagg(HETERO_DISAGG_PARAMS)
    new_df = _new_path_hetero_disagg(HETERO_DISAGG_PARAMS)

    assert old_df is not None and len(old_df) > 0, "legacy path returned empty DataFrame"
    assert len(new_df) > 0, "new path returned empty DataFrame"

    pd.testing.assert_frame_equal(_normalize(old_df), _normalize(new_df), check_like=True)

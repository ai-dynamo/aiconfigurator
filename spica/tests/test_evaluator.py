# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ReplayEvaluator dispatch: static->plain, scaling->bridge (dynamo stubbed)."""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("dynamo.mocker")

import dynamo.mocker  # noqa: E402
import dynamo.replay.api  # noqa: E402
import dynamo.replay.main  # noqa: E402

from spica.config import OptimizationGoal, OptimizationTarget, SLATarget, Workload  # noqa: E402
from spica.deploy import DeploymentPlan  # noqa: E402
from spica.evaluator import ReplayEvaluator  # noqa: E402


class _FakeArgs:
    @classmethod
    def from_json(cls, s):
        return ("ARGS", json.loads(s))


def _wl():
    return Workload(trace_path="/tmp/t.jsonl")


def _agg_plan(static):
    return DeploymentPlan(
        deployment_mode="agg",
        is_static=static,
        agg_engine_args={"aic_tp_size": 4, "max_num_seqs": 512},
        prefill_engine_args=None,
        decode_engine_args=None,
        num_workers=2,
        num_prefill_workers=0,
        num_decode_workers=0,
        router_mode="round_robin",
        router_config=None,
        planner_config=None if static else {"mode": "agg", "optimization_target": "sla"},
    )


def test_static_agg_uses_plain_path(monkeypatch):
    monkeypatch.setattr(dynamo.mocker, "MockEngineArgs", _FakeArgs)
    rec = {}
    monkeypatch.setattr(
        dynamo.replay.api, "run_trace_replay", lambda **kw: rec.update(kw) or {"output_throughput_tok_s": 42.0}
    )
    ev = ReplayEvaluator(_wl(), OptimizationGoal(target=OptimizationTarget.THROUGHPUT))
    report = ev.evaluate(_agg_plan(static=True))
    assert report["output_throughput_tok_s"] == 42.0
    assert rec["num_workers"] == 2 and rec["trace_file"] == "/tmp/t.jsonl" and rec["router_mode"] == "round_robin"


def test_scaling_agg_uses_bridge_with_goodput_sla(monkeypatch):
    monkeypatch.setattr(dynamo.mocker, "MockEngineArgs", _FakeArgs)
    rec = {}
    monkeypatch.setattr(
        dynamo.replay.main,
        "_run_planner_replay",
        lambda **kw: (
            rec.update(kw) or SimpleNamespace(trace_report={"gpu_hours": 2.0, "goodput_output_throughput_tok_s": 100.0})
        ),
    )
    goal = OptimizationGoal(target=OptimizationTarget.GOODPUT_PER_GPU_HOUR, sla=SLATarget(ttft_ms=2000.0, itl_ms=30.0))
    report = ReplayEvaluator(_wl(), goal).evaluate(_agg_plan(static=False))
    assert report["gpu_hours"] == 2.0
    # goodput SLA threaded to the bridge; planner config carried as inline JSON
    assert rec["sla_ttft_ms"] == 2000.0 and rec["sla_itl_ms"] == 30.0
    assert json.loads(rec["planner_config_arg"])["optimization_target"] == "sla"
    assert rec["num_workers"] == 2


def test_requires_trace_workload():
    with pytest.raises(ValueError, match="trace-based"):
        ReplayEvaluator(Workload(isl=128, osl=128, concurrency=1.0, request_count=10), OptimizationGoal())

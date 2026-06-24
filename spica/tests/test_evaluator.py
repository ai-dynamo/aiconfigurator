# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ReplayEvaluator dispatch across 3 load shapes x {static, planner} (dynamo stubbed)."""

import dataclasses
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


def _disagg_plan(static):
    return DeploymentPlan(
        deployment_mode="disagg",
        is_static=static,
        agg_engine_args=None,
        prefill_engine_args={"aic_tp_size": 2, "max_num_seqs": 256},
        decode_engine_args={"aic_tp_size": 4, "max_num_seqs": 512},
        num_workers=0,
        num_prefill_workers=3,
        num_decode_workers=5,
        router_mode="round_robin",
        router_config=None,
        planner_config=None if static else {"mode": "disagg", "optimization_target": "sla"},
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
    assert "sla_ttft_ms" not in rec  # no SLA on a throughput goal -> none threaded


def test_static_path_threads_goodput_sla(monkeypatch):
    # the mocker is SLA-aware on the plain path too, so a static/disabled candidate
    # still receives the goodput SLA and can emit goodput.
    monkeypatch.setattr(dynamo.mocker, "MockEngineArgs", _FakeArgs)
    rec = {}
    monkeypatch.setattr(
        dynamo.replay.api,
        "run_trace_replay",
        lambda **kw: rec.update(kw) or {"goodput_output_throughput_tok_s": 100.0, "gpu_hours": 1.0},
    )
    goal = OptimizationGoal(target=OptimizationTarget.GOODPUT_PER_GPU_HOUR, sla=SLATarget(ttft_ms=2000.0, itl_ms=30.0))
    report = ReplayEvaluator(_wl(), goal).evaluate(_agg_plan(static=True))
    assert report["goodput_output_throughput_tok_s"] == 100.0
    assert rec["sla_ttft_ms"] == 2000.0 and rec["sla_itl_ms"] == 30.0  # SLA threaded to the plain path


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


def test_kv_router_config_is_built_and_passed(monkeypatch):
    # the searched kv-router weights must reach the replay as a real KvRouterConfig
    # (round_robin -> None). Regression for the "router_config=None" stub.
    from dynamo._core import KvRouterConfig

    monkeypatch.setattr(dynamo.mocker, "MockEngineArgs", _FakeArgs)
    rec = {}
    monkeypatch.setattr(
        dynamo.replay.api, "run_trace_replay", lambda **kw: rec.update(kw) or {"output_throughput_tok_s": 1.0}
    )
    ev = ReplayEvaluator(_wl(), OptimizationGoal(target=OptimizationTarget.THROUGHPUT))

    # round_robin -> no router config
    ev.evaluate(_agg_plan(static=True))
    assert rec["router_mode"] == "round_robin" and rec["router_config"] is None

    # kv_router with weights -> a KvRouterConfig carrying them
    plan = dataclasses.replace(
        _agg_plan(static=True),
        router_mode="kv_router",
        router_config={"overlap_score_credit": 0.5, "router_temperature": 0.2},
    )
    ev.evaluate(plan)
    assert rec["router_mode"] == "kv_router"
    assert isinstance(rec["router_config"], KvRouterConfig)


def test_static_trace_threads_replay_concurrency(monkeypatch):
    # a closed-loop concurrency cap on a trace reaches run_trace_replay as
    # replay_concurrency (run_trace_replay defaults replay_mode='offline').
    monkeypatch.setattr(dynamo.mocker, "MockEngineArgs", _FakeArgs)
    rec = {}
    monkeypatch.setattr(
        dynamo.replay.api, "run_trace_replay", lambda **kw: rec.update(kw) or {"output_throughput_tok_s": 1.0}
    )
    wl = Workload(trace_path="/tmp/t.jsonl", replay_concurrency=32)
    ReplayEvaluator(wl, OptimizationGoal(target=OptimizationTarget.THROUGHPUT)).evaluate(_agg_plan(static=True))
    assert rec["replay_concurrency"] == 32


def test_scaling_trace_threads_replay_concurrency(monkeypatch):
    # closed-loop concurrency over a trace + planner now works (the bridge takes a cap):
    # the evaluator threads replay_concurrency into _run_planner_replay.
    monkeypatch.setattr(dynamo.mocker, "MockEngineArgs", _FakeArgs)
    rec = {}
    monkeypatch.setattr(
        dynamo.replay.main,
        "_run_planner_replay",
        lambda **kw: rec.update(kw) or SimpleNamespace(trace_report={"gpu_hours": 1.0}),
    )
    wl = Workload(trace_path="/tmp/t.jsonl", replay_concurrency=32)
    goal = OptimizationGoal(target=OptimizationTarget.GOODPUT_PER_GPU_HOUR, sla=SLATarget(ttft_ms=2000.0, itl_ms=30.0))
    ReplayEvaluator(wl, goal).evaluate(_agg_plan(static=False))
    assert rec["replay_concurrency"] == 32 and rec["trace_file"] == "/tmp/t.jsonl"


def _syn_wl(**kw):
    base = dict(isl=128, osl=64, request_count=100)
    base.update(kw)
    return Workload(**base)


def test_synthetic_static_uses_from_synthetic_bridge(monkeypatch):
    # synthetic + static -> PlannerReplayBridge.from_synthetic, driven to completion with
    # no scaling; goodput SLA threaded; closed-loop in-flight cap = concurrency.
    monkeypatch.setattr(dynamo.mocker, "MockEngineArgs", _FakeArgs)
    rec = {}

    class _Bridge:
        def advance_to(self, until_ms):
            return {"is_done": True}

        def finalize(self):
            return {"goodput_output_throughput_tok_s": 50.0, "gpu_hours": 0.5}

    class _BridgeFactory:
        @staticmethod
        def from_synthetic(**kw):
            rec.update(kw)
            return _Bridge()

    monkeypatch.setattr(dynamo.mocker, "PlannerReplayBridge", _BridgeFactory)
    goal = OptimizationGoal(target=OptimizationTarget.GOODPUT_PER_GPU_HOUR, sla=SLATarget(ttft_ms=2000.0, itl_ms=30.0))
    report = ReplayEvaluator(_syn_wl(concurrency=4.0), goal).evaluate(_agg_plan(static=True))
    assert report["goodput_output_throughput_tok_s"] == 50.0
    assert rec["input_tokens"] == 128 and rec["output_tokens"] == 64 and rec["request_count"] == 100
    assert rec["replay_concurrency"] == 4  # closed-loop cap from concurrency
    assert rec["sla_ttft_ms"] == 2000.0


def test_synthetic_planner_uses_run_planner_replay(monkeypatch):
    # synthetic + planner -> _run_planner_replay(synthetic=SyntheticWorkload(...), trace_file=None).
    monkeypatch.setattr(dynamo.mocker, "MockEngineArgs", _FakeArgs)
    monkeypatch.setattr(dynamo.replay.main, "SyntheticWorkload", lambda **kw: ("SYN", kw), raising=False)
    rec = {}
    monkeypatch.setattr(
        dynamo.replay.main,
        "_run_planner_replay",
        lambda **kw: rec.update(kw) or SimpleNamespace(trace_report={"gpu_hours": 2.0}),
    )
    goal = OptimizationGoal(target=OptimizationTarget.GOODPUT_PER_GPU_HOUR, sla=SLATarget(ttft_ms=1500.0, itl_ms=50.0))
    # request-rate workload -> open-loop (no cap); arrival_interval derived from the rate
    ReplayEvaluator(_syn_wl(request_rate=20.0), goal).evaluate(_agg_plan(static=False))
    assert rec["trace_file"] is None and rec["replay_concurrency"] is None
    assert rec["synthetic"][0] == "SYN"
    assert rec["synthetic"][1]["arrival_interval_ms"] == 50.0  # 1000 / 20
    assert rec["sla_ttft_ms"] == 1500.0


def test_static_trace_disagg_uses_plain_path(monkeypatch):
    # disagg + trace + static -> run_trace_replay gets prefill/decode engine args
    # and the per-role worker counts (no agg num_workers/extra_engine_args).
    monkeypatch.setattr(dynamo.mocker, "MockEngineArgs", _FakeArgs)
    rec = {}
    monkeypatch.setattr(
        dynamo.replay.api, "run_trace_replay", lambda **kw: rec.update(kw) or {"output_throughput_tok_s": 7.0}
    )
    report = ReplayEvaluator(_wl(), OptimizationGoal(target=OptimizationTarget.THROUGHPUT)).evaluate(
        _disagg_plan(static=True)
    )
    assert report["output_throughput_tok_s"] == 7.0
    assert rec["prefill_engine_args"][1] == {"aic_tp_size": 2, "max_num_seqs": 256}
    assert rec["decode_engine_args"][1] == {"aic_tp_size": 4, "max_num_seqs": 512}
    assert rec["num_prefill_workers"] == 3 and rec["num_decode_workers"] == 5
    assert "num_workers" not in rec and "extra_engine_args" not in rec


def test_scaling_trace_disagg_uses_run_planner_replay(monkeypatch):
    # disagg + trace + planner -> _run_planner_replay with disagg args (num_workers=0,
    # per-role workers + engine args) and the disagg planner config carried as JSON.
    monkeypatch.setattr(dynamo.mocker, "MockEngineArgs", _FakeArgs)
    rec = {}
    monkeypatch.setattr(
        dynamo.replay.main,
        "_run_planner_replay",
        lambda **kw: (
            rec.update(kw) or SimpleNamespace(trace_report={"gpu_hours": 3.0, "goodput_output_throughput_tok_s": 90.0})
        ),
    )
    goal = OptimizationGoal(target=OptimizationTarget.GOODPUT_PER_GPU_HOUR, sla=SLATarget(ttft_ms=2000.0, itl_ms=30.0))
    report = ReplayEvaluator(_wl(), goal).evaluate(_disagg_plan(static=False))
    assert report["gpu_hours"] == 3.0
    assert rec["extra_engine_args"] is None and rec["num_workers"] == 0
    assert rec["prefill_engine_args"][1] == {"aic_tp_size": 2, "max_num_seqs": 256}
    assert rec["decode_engine_args"][1] == {"aic_tp_size": 4, "max_num_seqs": 512}
    assert rec["num_prefill_workers"] == 3 and rec["num_decode_workers"] == 5
    assert rec["sla_ttft_ms"] == 2000.0 and rec["sla_itl_ms"] == 30.0
    assert json.loads(rec["planner_config_arg"])["mode"] == "disagg"


def test_synthetic_static_disagg_uses_from_synthetic_disagg(monkeypatch):
    # disagg + synthetic + static -> PlannerReplayBridge.from_synthetic_disagg, driven to
    # completion; goodput SLA threaded; closed-loop in-flight cap = concurrency.
    monkeypatch.setattr(dynamo.mocker, "MockEngineArgs", _FakeArgs)
    rec = {}

    class _Bridge:
        def advance_to(self, until_ms):
            return {"is_done": True}

        def finalize(self):
            return {"goodput_output_throughput_tok_s": 60.0, "gpu_hours": 0.75}

    class _BridgeFactory:
        @staticmethod
        def from_synthetic_disagg(**kw):
            rec.update(kw)
            return _Bridge()

    monkeypatch.setattr(dynamo.mocker, "PlannerReplayBridge", _BridgeFactory)
    goal = OptimizationGoal(target=OptimizationTarget.GOODPUT_PER_GPU_HOUR, sla=SLATarget(ttft_ms=2000.0, itl_ms=30.0))
    report = ReplayEvaluator(_syn_wl(concurrency=4.0), goal).evaluate(_disagg_plan(static=True))
    assert report["goodput_output_throughput_tok_s"] == 60.0
    assert rec["prefill_engine_args"][1] == {"aic_tp_size": 2, "max_num_seqs": 256}
    assert rec["decode_engine_args"][1] == {"aic_tp_size": 4, "max_num_seqs": 512}
    assert rec["num_prefill_workers"] == 3 and rec["num_decode_workers"] == 5
    assert rec["input_tokens"] == 128 and rec["output_tokens"] == 64 and rec["request_count"] == 100
    assert rec["replay_concurrency"] == 4  # closed-loop cap from concurrency
    assert rec["sla_ttft_ms"] == 2000.0


def test_synthetic_planner_disagg_uses_run_planner_replay(monkeypatch):
    # disagg + synthetic + planner -> _run_planner_replay(synthetic=…, prefill/decode).
    monkeypatch.setattr(dynamo.mocker, "MockEngineArgs", _FakeArgs)
    monkeypatch.setattr(dynamo.replay.main, "SyntheticWorkload", lambda **kw: ("SYN", kw), raising=False)
    rec = {}
    monkeypatch.setattr(
        dynamo.replay.main,
        "_run_planner_replay",
        lambda **kw: rec.update(kw) or SimpleNamespace(trace_report={"gpu_hours": 2.0}),
    )
    goal = OptimizationGoal(target=OptimizationTarget.GOODPUT_PER_GPU_HOUR, sla=SLATarget(ttft_ms=1500.0, itl_ms=50.0))
    # request-rate workload -> open-loop (no cap); arrival_interval derived from the rate
    ReplayEvaluator(_syn_wl(request_rate=20.0), goal).evaluate(_disagg_plan(static=False))
    assert rec["trace_file"] is None and rec["replay_concurrency"] is None
    assert rec["extra_engine_args"] is None and rec["num_workers"] == 0
    assert rec["prefill_engine_args"][1] == {"aic_tp_size": 2, "max_num_seqs": 256}
    assert rec["decode_engine_args"][1] == {"aic_tp_size": 4, "max_num_seqs": 512}
    assert rec["num_prefill_workers"] == 3 and rec["num_decode_workers"] == 5
    assert rec["synthetic"][0] == "SYN"
    assert rec["synthetic"][1]["arrival_interval_ms"] == 50.0  # 1000 / 20
    assert rec["sla_ttft_ms"] == 1500.0


def test_drive_static_bridge_loops_until_done(monkeypatch):
    # the drive loop keeps advancing while is_done is False, then reaches finalize().
    monkeypatch.setattr(dynamo.mocker, "MockEngineArgs", _FakeArgs)

    class _Bridge:
        def __init__(self):
            self.advances = 0
            self.finalized = False

        def advance_to(self, until_ms):
            self.advances += 1
            # not done on the first advance, done on the second
            return {"is_done": self.advances >= 2}

        def finalize(self):
            self.finalized = True
            return {"goodput_output_throughput_tok_s": 11.0, "gpu_hours": 0.25}

    bridge = _Bridge()

    class _BridgeFactory:
        @staticmethod
        def from_synthetic(**kw):
            return bridge

    monkeypatch.setattr(dynamo.mocker, "PlannerReplayBridge", _BridgeFactory)
    goal = OptimizationGoal(target=OptimizationTarget.GOODPUT_PER_GPU_HOUR, sla=SLATarget(ttft_ms=2000.0, itl_ms=30.0))
    # synthetic + concurrency + static -> from_synthetic bridge driven to completion
    report = ReplayEvaluator(_syn_wl(concurrency=4.0), goal).evaluate(_agg_plan(static=True))
    assert bridge.advances == 2  # one not-done transition then done -> loop terminates
    assert bridge.finalized and report["goodput_output_throughput_tok_s"] == 11.0


def test_workload_validation():
    with pytest.raises(ValueError, match="positive integer"):  # trace + bad cap
        Workload(trace_path="/tmp/t.jsonl", replay_concurrency=0)
    with pytest.raises(ValueError, match="for trace workloads"):  # synthetic can't use replay_concurrency
        _syn_wl(concurrency=1.0, replay_concurrency=8)
    with pytest.raises(ValueError, match="exactly one of request_rate or concurrency"):
        _syn_wl()  # synthetic needs a load shape
    with pytest.raises(ValueError, match="exactly one of request_rate or concurrency"):
        _syn_wl(request_rate=10.0, concurrency=4.0)  # not both
    with pytest.raises(ValueError, match="must not set synthetic fields"):
        Workload(trace_path="/tmp/t.jsonl", isl=128)

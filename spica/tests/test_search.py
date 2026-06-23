# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Orchestration loop. enumerate_branches / sweep_load_predictor / backend-version
are stubbed and a fake sampler + evaluator are injected, so this exercises the
suggest->unroll->deploy->evaluate->score->observe->rank loop without Vizier or
real replay."""

import pytest

import spica.search as search_mod
from spica.config import SmartSearchConfig
from spica.load_predictor_sweep import LoadPredictorResult
from spica.parallel_enum import ParallelShape, ReplicaParallelConfig
from spica.sampler import Suggestion
from spica.search import run_smart_search
from spica.search_space import BranchSpace


def _config(gpu_budget=32):
    return SmartSearchConfig(
        search_space={
            "model_name": "deepseek-ai/DeepSeek-V3",
            "hardware_sku": "gb200",
            "backend": ["trtllm"],
            "deployment_mode": ["agg"],
            "gpu_budget": gpu_budget,
        },
        workload={"trace_path": "/tmp/t.jsonl"},
        sweep={"max_rounds": 1, "candidates_per_round": 3, "parallel_evals": 1},  # sequential (fakes)
        goal={"target": "throughput"},
    )


class _FakeSampler:
    """Suggests `count` agg candidates with increasing agg_max_num_seqs."""

    def __init__(self, branch, study_id):
        self.branch = branch
        self.scored: list = []

    def suggest(self, count):
        out = []
        for i in range(count):
            sel = {
                "deployment_mode": "agg",
                "backend": "trtllm",
                "router_mode": "round_robin",
                "planner_scaling_policy": "disabled",
                "planner_fpm_sampling": "default",
                "planner_load_sensitivity": "default",
                "agg_max_num_batched_tokens": 8192,
                "agg_max_num_seqs": 256 * (i + 1),
            }
            out.append(Suggestion(selection=sel, parallel_config=self.branch.parallel_configs[0], handle=sel))
        return out

    def observe(self, suggestion, score):
        self.scored.append(score)

    def observe_infeasible(self, suggestion, reason):
        self.scored.append(("infeasible", reason))


class _FakeEvaluator:
    """trace_report throughput == the plan's max_num_seqs (so higher seqs wins)."""

    def __init__(self):
        self.calls = 0

    def evaluate(self, plan):
        self.calls += 1
        return {"output_throughput_tok_s": float(plan.agg_engine_args["max_num_seqs"]), "gpu_hours": 1.0}


def _branch(parallel_config):
    return BranchSpace(
        deployment_mode="agg",
        parallel_configs=(parallel_config,),
        supported_backends={parallel_config: frozenset({"trtllm"})},
        knob_choices={"backend": ["trtllm"]},
    )


def _stub(monkeypatch, branch):
    monkeypatch.setattr(search_mod, "enumerate_branches", lambda config: [branch])
    monkeypatch.setattr(search_mod, "sweep_load_predictor", lambda config: LoadPredictorResult(reason="static"))
    monkeypatch.setattr(search_mod, "resolve_backend_version", lambda hw, be: "1.3.0rc10")


def test_ranks_feasible_best_first(monkeypatch):
    branch = _branch(ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2))  # 8 GPUs
    _stub(monkeypatch, branch)
    cands = run_smart_search(_config(), evaluator=_FakeEvaluator(), sampler_factory=_FakeSampler)
    assert [c.score for c in cands] == [768.0, 512.0, 256.0]  # throughput, best first
    assert all(c.used_gpus == 8 for c in cands)
    assert cands[0].metrics["gpu_hours"] == 1.0


def test_over_budget_candidates_dropped(monkeypatch):
    branch = _branch(ReplicaParallelConfig(ParallelShape(tp=16, dp=1, moe_tp=1, moe_ep=16), replicas=4))  # 64 GPUs
    _stub(monkeypatch, branch)
    sampler_seen = {}

    def factory(b, study_id):
        s = _FakeSampler(b, study_id)
        sampler_seen["s"] = s
        return s

    cands = run_smart_search(_config(gpu_budget=32), evaluator=_FakeEvaluator(), sampler_factory=factory)
    assert cands == []  # 64 GPUs > 32 budget -> all infeasible, dropped
    assert len(sampler_seen["s"].scored) == 3  # still scored/observed so the optimizer learns


def test_eval_failure_marked_infeasible(monkeypatch):
    branch = _branch(ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2))
    _stub(monkeypatch, branch)
    sampler_seen = {}

    class _Boom:
        def evaluate(self, plan):
            raise RuntimeError("replay blew up")

    def factory(b, study_id):
        s = _FakeSampler(b, study_id)
        sampler_seen["s"] = s
        return s

    cands = run_smart_search(_config(), evaluator=_Boom(), sampler_factory=factory)
    assert cands == []
    assert all(isinstance(x, tuple) and x[0] == "infeasible" for x in sampler_seen["s"].scored)


def test_study_id_unique_per_run(monkeypatch):
    # Vizier persists studies by id; a fixed id makes a later run inherit a stale
    # study (and its old param space). run_smart_search must use a fresh id per run.
    branch = _branch(ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2))
    _stub(monkeypatch, branch)
    seen: list[str] = []

    def factory(b, study_id):
        seen.append(study_id)
        return _FakeSampler(b, study_id)

    run_smart_search(_config(), evaluator=_FakeEvaluator(), sampler_factory=factory, show_progress=False)
    run_smart_search(_config(), evaluator=_FakeEvaluator(), sampler_factory=factory, show_progress=False)
    assert len(seen) == 2 and seen[0] != seen[1]  # fresh study per run, no stale reuse
    assert all(s.startswith("spica_agg_") for s in seen)  # study id is per-mode (backend is a knob)


def _config_with_policies(policies, target="throughput"):
    return SmartSearchConfig(
        search_space={
            "model_name": "deepseek-ai/DeepSeek-V3",
            "hardware_sku": "gb200",
            "backend": ["trtllm"],
            "deployment_mode": ["agg"],
            "gpu_budget": 32,
            "planner_scaling_policy": policies,
        },
        workload={"trace_path": "/tmp/t.jsonl"},
        sweep={"max_rounds": 1, "candidates_per_round": 2, "parallel_evals": 1},  # sequential (fakes)
        goal={"target": target},
    )


def test_non_goodput_sweep_rejects_all_throughput_scaling_policies(monkeypatch):
    # a throughput sweep can't use predictive throughput scaling (no SLA); if EVERY
    # policy enables it, there's nothing to search -> a clear error.
    branch = _branch(ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2))
    _stub(monkeypatch, branch)
    cfg = _config_with_policies(["throughput_180_5", "hybrid_600_5"], target="throughput")
    with pytest.raises(ValueError, match="throughput scaling"):
        run_smart_search(cfg, evaluator=_FakeEvaluator(), sampler_factory=_FakeSampler, show_progress=False)


def test_non_goodput_sweep_drops_throughput_scaling_and_proceeds(monkeypatch):
    # mixed list -> the throughput-scaling entry is dropped, the rest still run.
    branch = _branch(ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2))
    _stub(monkeypatch, branch)
    cfg = _config_with_policies(["disabled", "throughput_180_5", "load_180_5"], target="throughput")
    cands = run_smart_search(cfg, evaluator=_FakeEvaluator(), sampler_factory=_FakeSampler, show_progress=False)
    assert [c.score for c in cands] == [512.0, 256.0]  # ran fine (throughput == max_num_seqs)

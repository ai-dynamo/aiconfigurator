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

    def __init__(self, branch, study_id, objectives=None):
        self.branch = branch
        self.objectives = objectives
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

    def observe(self, suggestion, metrics):
        self.scored.append(metrics)

    def observe_infeasible(self, suggestion, reason):
        self.scored.append(("infeasible", reason))


class _FakeEvaluator:
    """trace_report throughput == the plan's max_num_seqs (so higher seqs wins)."""

    def __init__(self):
        self.calls = 0

    def evaluate(self, plan, *, concurrency_override=None):
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
    monkeypatch.setattr(search_mod, "enumerate_branches", lambda config, *, max_seq_len=None: [branch])
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

    def factory(b, study_id, objectives=None):
        s = _FakeSampler(b, study_id)
        sampler_seen["s"] = s
        return s

    cands = run_smart_search(_config(gpu_budget=32), evaluator=_FakeEvaluator(), sampler_factory=factory)
    assert cands == []  # 64 GPUs > 32 budget -> all infeasible, dropped
    # Over-budget trials are told to the optimizer as INFEASIBLE (observe_infeasible),
    # never fed back as a high objective score (which would steer it into the infeasible
    # region). _FakeSampler records observe_infeasible as ("infeasible", reason) tuples.
    scored = sampler_seen["s"].scored
    assert len(scored) == 3  # still reported so the optimizer learns to avoid the region
    assert all(isinstance(x, tuple) and x[0] == "infeasible" for x in scored)
    assert all("over gpu_budget" in x[1] for x in scored)  # reason carries the budget breach


def test_eval_failure_marked_infeasible(monkeypatch):
    branch = _branch(ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2))
    _stub(monkeypatch, branch)
    sampler_seen = {}

    class _Boom:
        def evaluate(self, plan, *, concurrency_override=None):
            raise RuntimeError("replay blew up")

    def factory(b, study_id, objectives=None):
        s = _FakeSampler(b, study_id)
        sampler_seen["s"] = s
        return s

    cands = run_smart_search(_config(), evaluator=_Boom(), sampler_factory=factory)
    assert cands == []
    assert all(isinstance(x, tuple) and x[0] == "infeasible" for x in sampler_seen["s"].scored)


def test_unsupported_backend_config_pair_marked_unsupported(monkeypatch):
    # A (backend, parallel_config) pair the backend can't run is split off on the main
    # process: it's told observe_infeasible ("does not support") and never evaluated.
    pc = ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2)
    branch = BranchSpace(
        deployment_mode="agg",
        parallel_configs=(pc,),
        supported_backends={pc: frozenset({"vllm"})},  # NOT trtllm (what _FakeSampler suggests)
        knob_choices={"backend": ["vllm", "trtllm"]},
    )
    _stub(monkeypatch, branch)
    sampler_seen = {}

    class _NeverCalled:
        def evaluate(self, plan, *, concurrency_override=None):
            raise AssertionError("evaluator must not run for an unsupported (backend, config) pair")

    def factory(b, study_id, objectives=None):
        s = _FakeSampler(b, study_id)
        sampler_seen["s"] = s
        return s

    cands = run_smart_search(_config(), evaluator=_NeverCalled(), sampler_factory=factory)
    assert cands == []  # nothing evaluated -> no feasible candidate
    scored = sampler_seen["s"].scored
    assert len(scored) == 3  # all three suggestions told infeasible (none evaluated)
    assert all(isinstance(x, tuple) and x[0] == "infeasible" for x in scored)
    assert all("does not support" in x[1] for x in scored)


def test_study_id_unique_per_run(monkeypatch):
    # Vizier persists studies by id; a fixed id makes a later run inherit a stale
    # study (and its old param space). run_smart_search must use a fresh id per run.
    branch = _branch(ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2))
    _stub(monkeypatch, branch)
    seen: list[str] = []

    def factory(b, study_id, objectives=None):
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


def test_e2e_only_goodput_drops_planner_scaling_and_proceeds(monkeypatch):
    # e2e-only SLA is valid for goodput, but cannot seed the planner's ttft/itl target.
    # Scaling policies are filtered out before Vizier can sample a build-time-invalid plan.
    branch = _branch(ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2))
    seen = {}

    def fake_enumerate_branches(config, *, max_seq_len=None):
        seen["policies"] = list(config.search_space.planner_scaling_policy)
        return [branch]

    monkeypatch.setattr(search_mod, "enumerate_branches", fake_enumerate_branches)
    monkeypatch.setattr(search_mod, "sweep_load_predictor", lambda config: LoadPredictorResult(reason="static"))
    monkeypatch.setattr(search_mod, "resolve_backend_version", lambda hw, be: "1.3.0rc10")
    cfg = SmartSearchConfig(
        search_space={
            "model_name": "deepseek-ai/DeepSeek-V3",
            "hardware_sku": "gb200",
            "backend": ["trtllm"],
            "deployment_mode": ["agg"],
            "gpu_budget": 32,
            "planner_scaling_policy": ["disabled", "throughput_180_5", "load_180_5", "hybrid_180_5"],
        },
        workload={"trace_path": "/tmp/t.jsonl"},
        sweep={"max_rounds": 1, "candidates_per_round": 1, "parallel_evals": 1},
        goal={"target": "goodput_per_gpu", "sla": {"e2e_ms": 5000.0}},
    )

    cands = run_smart_search(cfg, evaluator=_FakeEvaluator(), sampler_factory=_FakeSampler, show_progress=False)

    assert seen["policies"] == ["disabled"]
    assert len(cands) == 1


def test_candidate_build_error_is_reported_not_raised(monkeypatch):
    branch = _branch(ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2))
    _stub(monkeypatch, branch)
    sampler_seen = {}

    class BadSampler(_FakeSampler):
        def suggest(self, count):
            sel = {
                "deployment_mode": "agg",
                "backend": "trtllm",
                "router_mode": "round_robin",
                "planner_scaling_policy": "disabled",
                "planner_fpm_sampling": "default",
                "planner_load_sensitivity": "default",
                "agg_max_num_batched_tokens": 8192,
                # missing agg_max_num_seqs -> unroll/build failure
            }
            return [Suggestion(selection=sel, parallel_config=self.branch.parallel_configs[0], handle=sel)]

    def factory(b, study_id, objectives=None):
        s = BadSampler(b, study_id)
        sampler_seen["s"] = s
        return s

    cands = run_smart_search(_config(), evaluator=_FakeEvaluator(), sampler_factory=factory, show_progress=False)

    assert cands == []
    scored = sampler_seen["s"].scored
    assert len(scored) == 1
    assert scored[0][0] == "infeasible"
    assert "candidate build failed" in scored[0][1]


# --- pareto (multi-objective) sweep over a swept concurrency ---


def _pareto_config():
    # synthetic concurrency sweep (list) under a pareto goal -> the InferenceX-style front
    return SmartSearchConfig(
        search_space={
            "model_name": "deepseek-ai/DeepSeek-V3",
            "hardware_sku": "gb200",
            "backend": ["trtllm"],
            "deployment_mode": ["agg"],
            "gpu_budget": 32,
        },
        workload={"isl": 1024, "osl": 1024, "concurrency": [4, 8, 16], "num_request_ratio": 10},
        sweep={"max_rounds": 1, "candidates_per_round": 3, "parallel_evals": 1},
        goal={"target": "pareto"},
    )


# per-concurrency (aggregate throughput, per-user throughput): higher concurrency trades
# more aggregate throughput for less per-user interactivity -> all three are non-dominated.
_PARETO_POINTS = {4: (100.0, 40.0), 8: (150.0, 25.0), 16: (180.0, 12.0)}


class _ParetoSampler:
    """Suggests one candidate per swept concurrency (4/8/16), records observed metrics."""

    def __init__(self, branch, study_id, objectives=None):
        self.branch = branch
        self.objectives = objectives
        self.observed: list = []

    def suggest(self, count):
        out = []
        for c in (4, 8, 16):
            sel = {
                "deployment_mode": "agg",
                "backend": "trtllm",
                "router_mode": "round_robin",
                "planner_scaling_policy": "disabled",
                "planner_fpm_sampling": "default",
                "planner_load_sensitivity": "default",
                "agg_max_num_batched_tokens": 8192,
                "agg_max_num_seqs": 256,
                "concurrency": c,  # the swept Pareto dimension
            }
            out.append(Suggestion(selection=sel, parallel_config=self.branch.parallel_configs[0], handle=sel))
        return out

    def observe(self, suggestion, metrics):
        self.observed.append(metrics)

    def observe_infeasible(self, suggestion, reason):
        self.observed.append(("infeasible", reason))


class _ParetoEvaluator:
    def evaluate(self, plan, *, concurrency_override=None):
        tput, user = _PARETO_POINTS[concurrency_override]
        # avg_gpu = gpu_hours / (duration_ms / 3.6e6) = 1.0 / 1.0 = 1.0 -> throughput_per_gpu == tput
        return {
            "output_throughput_tok_s": tput,
            "mean_output_token_throughput_per_user": user,
            "gpu_hours": 1.0,
            "duration_ms": 3_600_000.0,
        }


def test_pareto_sweep_returns_non_dominated_front(monkeypatch):
    branch = _branch(ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2))  # 8 GPUs
    _stub(monkeypatch, branch)
    seen = {}

    def factory(b, study_id, objectives=None):
        s = _ParetoSampler(b, study_id, objectives)
        seen["s"] = s
        return s

    front = run_smart_search(
        _pareto_config(), evaluator=_ParetoEvaluator(), sampler_factory=factory, show_progress=False
    )
    # all three concurrency points are mutually non-dominated -> full front, sorted by the
    # x-axis (per-user throughput) ascending.
    assert [c.objectives["throughput_per_user"] for c in front] == [12.0, 25.0, 40.0]
    assert [c.objectives["throughput_per_gpu"] for c in front] == [180.0, 150.0, 100.0]
    assert {c.config["concurrency"] for c in front} == {4, 8, 16}  # each point recorded its concurrency
    # the sampler was built multi-objective (one (name, maximize) per objective) and fed raw vectors
    assert seen["s"].objectives == [("throughput_per_gpu", True), ("throughput_per_user", True)]
    assert all(set(m) == {"throughput_per_gpu", "throughput_per_user"} for m in seen["s"].observed)

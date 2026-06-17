# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The smart sweep: SearchSpace -> ranked candidates (best-first).

Per ``(deployment_mode, backend)`` branch, a Vizier study searches the parallel
config + knob space; each suggestion is unrolled, translated to a deployment, and
evaluated by replay; the score is fed back to the optimizer; feasible candidates
are ranked across branches.

The load-predictor winner is resolved once (the independent sub-sweep) and
injected into every candidate's unroll. Evaluation is sequential in v1
(``SweepConfig.parallel_evals`` is not yet parallelized).

``evaluator`` and ``sampler_factory`` are injectable so the loop is unit-testable
without real replay / Vizier.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from .config import Candidate, OptimizationGoal, SmartSearchConfig
from .deploy import build_deployment
from .evaluator import ReplayEvaluator
from .kv_estimate import resolve_backend_version
from .load_predictor_sweep import LoadPredictorResult, sweep_load_predictor
from .sample import unroll_sample
from .sampler import BranchSampler, Suggestion, make_branch_sampler
from .score import is_feasible, make_candidate, rank, score_report
from .search_space import BranchSpace, enumerate_branches


class _Evaluator(Protocol):
    def evaluate(self, plan: Any) -> dict[str, float]: ...


def _evaluate_suggestion(
    suggestion: Suggestion,
    *,
    config: SmartSearchConfig,
    goal: OptimizationGoal,
    backend_version: str,
    evaluator: _Evaluator,
    sampler: BranchSampler,
    load_predictor: LoadPredictorResult,
) -> Candidate | None:
    """Unroll -> deploy -> replay -> score one suggestion; tell the sampler; return
    a Candidate when feasible (else None). Replay failures are reported to the
    sampler as infeasible so the study keeps moving."""
    sample = unroll_sample(
        search_space=config.search_space,
        selection=suggestion.selection,
        parallel_config=suggestion.parallel_config,
        load_predictor=load_predictor,
    )
    plan = build_deployment(sample, backend_version=backend_version, planner_sla=goal.sla)
    try:
        report = evaluator.evaluate(plan)
    except Exception as exc:  # a single candidate failing must not abort the sweep
        sampler.observe_infeasible(suggestion, f"replay failed: {type(exc).__name__}: {exc}")
        return None
    sampler.observe(suggestion, score_report(report, goal.target))
    if is_feasible(report, int(sample["used_gpus"]), goal, config.search_space.gpu_budget):
        return make_candidate(sample, report, goal.target)
    return None


def run_smart_search(
    config: SmartSearchConfig,
    *,
    evaluator: _Evaluator | None = None,
    sampler_factory: Callable[..., BranchSampler] = make_branch_sampler,
) -> list[Candidate]:
    """Run the sweep and return feasible candidates sorted best-first.

    ``evaluator`` defaults to a :class:`ReplayEvaluator` over the workload+goal;
    inject a fake to test the loop without replay. ``sampler_factory`` defaults to
    the Vizier-backed sampler.
    """
    goal = config.goal
    branches: list[BranchSpace] = enumerate_branches(config)
    load_predictor = sweep_load_predictor(config)
    if evaluator is None:
        evaluator = ReplayEvaluator(config.workload, goal)

    sweep = config.sweep
    per_round = sweep.candidates_per_round or sweep.parallel_evals
    candidates: list[Candidate] = []
    for branch in branches:
        backend_version = resolve_backend_version(config.search_space.hardware_sku, branch.backend)
        sampler = sampler_factory(branch, study_id=f"spica_{branch.deployment_mode}_{branch.backend}")
        for _ in range(sweep.max_rounds):
            for suggestion in sampler.suggest(per_round):
                candidate = _evaluate_suggestion(
                    suggestion,
                    config=config,
                    goal=goal,
                    backend_version=backend_version,
                    evaluator=evaluator,
                    sampler=sampler,
                    load_predictor=load_predictor,
                )
                if candidate is not None:
                    candidates.append(candidate)
    return rank(candidates)

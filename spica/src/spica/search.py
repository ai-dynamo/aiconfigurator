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

import uuid
from collections.abc import Callable
from typing import Any, Protocol

from tqdm import tqdm

from .config import Candidate, OptimizationGoal, SmartSearchConfig
from .deploy import build_deployment
from .evaluator import ReplayEvaluator
from .kv_estimate import resolve_backend_version
from .load_predictor_sweep import LoadPredictorResult, sweep_load_predictor
from .planner import filter_scaling_policies
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
) -> tuple[Candidate | None, str]:
    """Unroll -> deploy -> replay -> score one suggestion; tell the sampler. Returns
    ``(candidate, outcome)`` where outcome is ``"feasible"`` (with the Candidate),
    ``"infeasible"`` (scored but gated out), or ``"failed"`` (replay raised). Replay
    failures are reported to the sampler as infeasible so the study keeps moving; the
    outcome lets the caller tally them instead of silently dropping both as None."""
    sample = unroll_sample(
        search_space=config.search_space,
        selection=suggestion.selection,
        parallel_config=suggestion.parallel_config,
        load_predictor=load_predictor,
    )
    plan = build_deployment(
        sample,
        backend_version=backend_version,
        optimization_target=goal.target.planner_optimization_target,
        planner_sla=goal.sla,
    )
    try:
        report = evaluator.evaluate(plan)
    except Exception as exc:  # a single candidate failing must not abort the sweep
        sampler.observe_infeasible(suggestion, f"replay failed: {type(exc).__name__}: {exc}")
        return None, "failed"
    sampler.observe(suggestion, score_report(report, goal.target))
    if is_feasible(int(sample["used_gpus"]), config.search_space.gpu_budget):
        return make_candidate(sample, report, goal.target), "feasible"
    return None, "infeasible"


def run_smart_search(
    config: SmartSearchConfig,
    *,
    evaluator: _Evaluator | None = None,
    sampler_factory: Callable[..., BranchSampler] = make_branch_sampler,
    show_progress: bool = True,
) -> list[Candidate]:
    """Run the sweep and return feasible candidates sorted best-first.

    ``evaluator`` defaults to a :class:`ReplayEvaluator` over the workload+goal;
    inject a fake to test the loop without replay. ``sampler_factory`` defaults to
    the Vizier-backed sampler. ``show_progress`` draws a tqdm bar over the
    candidate evaluations (live feasible/failed tally + best score) and prints a
    one-line summary at the end; set False for quiet/non-interactive runs.
    """
    goal = config.goal
    # Predictive throughput scaling only works under the planner's "sla" target
    # (a goodput sweep). For throughput/latency sweeps, drop the throughput-scaling
    # policies up front so neither the sampler nor the load-predictor sub-sweep sees
    # them. (Disabled / load_* still run; static-path goodput is fine once the mocker
    # is SLA-aware.)
    kept, dropped = filter_scaling_policies(
        config.search_space.planner_scaling_policy,
        allow_throughput=(goal.target.planner_optimization_target == "sla"),
    )
    if dropped:
        if not kept:
            raise ValueError(
                f"every planner_scaling_policy enables throughput scaling, which a "
                f"'{goal.target.value}' sweep can't use (it has no SLA — use a goodput target, "
                f"or include 'disabled' / a load_* policy)"
            )
        if show_progress:
            tqdm.write(
                f"smart-sweep: dropped {len(dropped)} throughput-scaling policy option(s) "
                f"for target={goal.target.value} (needs SLA): {dropped}"
            )
        config = config.model_copy(
            update={"search_space": config.search_space.model_copy(update={"planner_scaling_policy": kept})}
        )

    branches: list[BranchSpace] = enumerate_branches(config)
    load_predictor = sweep_load_predictor(config)
    if evaluator is None:
        evaluator = ReplayEvaluator(config.workload, goal)

    sweep = config.sweep
    per_round = sweep.candidates_per_round or sweep.parallel_evals
    # Upper bound on evaluations (a Vizier round may return fewer than per_round).
    total = len(branches) * sweep.max_rounds * per_round
    candidates: list[Candidate] = []
    tally = {"feasible": 0, "infeasible": 0, "failed": 0}
    # Unique per run: Vizier's datastore persists studies by id, so a fixed id would
    # make a later run inherit a stale study (and its old param space) -> decode crash.
    run_nonce = uuid.uuid4().hex[:8]

    def _best() -> float | None:
        return max((c.score for c in candidates), default=None)

    with tqdm(total=total, desc="smart-sweep", unit="eval", disable=not show_progress) as bar:
        for branch in branches:
            backend_version = resolve_backend_version(config.search_space.hardware_sku, branch.backend)
            study_id = f"spica_{branch.deployment_mode}_{branch.backend}_{run_nonce}"
            sampler = sampler_factory(branch, study_id=study_id)
            bar.set_description(f"smart-sweep {branch.deployment_mode}/{branch.backend}")
            for _ in range(sweep.max_rounds):
                for suggestion in sampler.suggest(per_round):
                    candidate, outcome = _evaluate_suggestion(
                        suggestion,
                        config=config,
                        goal=goal,
                        backend_version=backend_version,
                        evaluator=evaluator,
                        sampler=sampler,
                        load_predictor=load_predictor,
                    )
                    tally[outcome] += 1
                    if candidate is not None:
                        candidates.append(candidate)
                    best = _best()
                    bar.set_postfix(
                        feasible=tally["feasible"],
                        failed=tally["failed"],
                        best=("-" if best is None else f"{best:.4g}"),
                    )
                    bar.update(1)

    if show_progress:
        best = _best()
        evaluated = sum(tally.values())
        summary = (
            f"smart-sweep done: {tally['feasible']}/{evaluated} feasible, "
            f"{tally['infeasible']} gated, {tally['failed']} replay-failed"
        )
        if candidates:
            summary += f"; best {goal.target.value}={best:.4g}"
        else:
            summary += " — NO feasible candidate (check SLA / gpu_budget / replay errors)"
        tqdm.write(summary)
    return rank(candidates)

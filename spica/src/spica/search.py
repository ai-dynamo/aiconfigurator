# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The smart sweep: SearchSpace -> ranked candidates (best-first).

One Vizier study per ``deployment_mode`` branch searches the parallel-config + knob
space (backend is one of the knobs); each suggestion is unrolled, translated to a
deployment, evaluated by replay, scored, and fed back to the optimizer; feasible
candidates are ranked across branches.

Each round is a **barrier**: the study suggests ``per_round`` trials (ask), they are
evaluated **in parallel across worker processes** (``SweepConfig.parallel_evals``;
``<= 1`` runs sequentially), then their scores are fed back (tell). Vizier ask/tell
stay on the main process — workers run only the pure unroll->deploy->replay->score and
never touch the study (the Vizier trial handle never crosses the process boundary). The
load-predictor winner is resolved once and injected into every unroll.

``evaluator`` and ``sampler_factory`` are injectable so the loop is unit-testable
without real replay / Vizier (use ``parallel_evals=1`` to avoid spawning processes).
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import uuid
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
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
from .score import is_feasible, make_candidate, pareto_front, rank
from .search_space import BranchSpace, enumerate_branches


class _Evaluator(Protocol):
    def evaluate(self, plan: Any, *, concurrency_override: int | None = None) -> dict[str, float]: ...


# Result of evaluating one suggestion (no Vizier here): (candidate|None, observe_metrics|None,
# outcome, reason). observe_metrics is the dict fed to sampler.observe — {"objective": score}
# for a single-objective sweep, or {obj_name: raw_value, ...} under a pareto goal. outcome in
# {"feasible","infeasible","failed"}. Both "failed" (replay error) and "infeasible" (over
# gpu_budget) carry a reason and no metrics -> the loop tells the sampler observe_infeasible
# for them (a gated trial is never fed back as a high score). "unsupported" is decided on the
# main process before evaluation and never reaches the worker.
_EvalResult = tuple[Candidate | None, dict[str, float] | None, str, str]


def _evaluate_one(
    selection: dict[str, Any],
    parallel_config: Any,
    *,
    config: SmartSearchConfig,
    goal: OptimizationGoal,
    load_predictor: LoadPredictorResult,
    evaluator: _Evaluator,
) -> _EvalResult:
    """Pure unroll -> deploy -> replay -> score for one (already backend-supported)
    suggestion. No Vizier, no shared mutable state -> safe to run in a worker process."""
    sample = unroll_sample(
        search_space=config.search_space,
        selection=selection,
        parallel_config=parallel_config,
        load_predictor=load_predictor,
    )
    # A pareto concurrency sweep carries the per-trial in-flight cap on the selection;
    # record it on the sample (so each Pareto point knows its concurrency) and drive replay with it.
    concurrency = selection.get("concurrency")
    if concurrency is not None:
        sample["concurrency"] = concurrency
    backend_version = resolve_backend_version(config.search_space.hardware_sku, selection["backend"])
    plan = build_deployment(
        sample,
        backend_version=backend_version,
        optimization_target=goal.target.planner_optimization_target,
        planner_sla=goal.sla,
    )
    try:
        report = evaluator.evaluate(plan, concurrency_override=concurrency)
    except Exception as exc:  # one candidate failing must not abort the sweep
        return None, None, "failed", f"replay failed: {type(exc).__name__}: {exc}"
    if not is_feasible(int(sample["used_gpus"]), config.search_space.gpu_budget):
        # Over gpu_budget: report as infeasible to the optimizer (observe_infeasible, not
        # observe(metrics)) so a high score doesn't steer the sampler into the infeasible
        # region. The trial is gated, not ranked.
        return (
            None,
            None,
            "infeasible",
            f"over gpu_budget: used_gpus={int(sample['used_gpus'])} > gpu_budget={config.search_space.gpu_budget}",
        )
    if goal.is_pareto:
        candidate = make_candidate(sample, report, goal.target, pareto_objectives=goal.resolved_pareto_objectives)
        # Pareto objectives are reported raw (each metric carries its own MAXIMIZE/MINIMIZE goal).
        observe_metrics: dict[str, float] = dict(candidate.objectives or {})
    else:
        candidate = make_candidate(sample, report, goal.target)
        observe_metrics = {"objective": candidate.score}  # single metric, pre-signed higher-is-better
    return candidate, observe_metrics, "feasible", ""


# Worker-process plumbing: the shared read-only state (config/goal/load_predictor/
# evaluator) is sent once via the pool initializer and stashed as a module global, so
# each task only ships the per-suggestion (selection, parallel_config).
_WORKER_CTX: dict[str, Any] = {}


def _init_worker(
    config: SmartSearchConfig, goal: OptimizationGoal, load_predictor: LoadPredictorResult, evaluator: _Evaluator
) -> None:
    _WORKER_CTX.update(config=config, goal=goal, load_predictor=load_predictor, evaluator=evaluator)


def _worker_eval(selection: dict[str, Any], parallel_config: Any) -> _EvalResult:
    return _evaluate_one(selection, parallel_config, **_WORKER_CTX)


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
    the Vizier-backed sampler. Within a round, suggestions are evaluated across
    ``SweepConfig.parallel_evals`` **spawned** worker processes (``<= 1`` -> sequential,
    no pool). With ``parallel_evals > 1`` the caller must guard its entrypoint with
    ``if __name__ == "__main__":`` (spawn re-imports the module) — the ``python -m spica``
    CLI already does; ad-hoc scripts must too, or set ``parallel_evals=1``.
    ``show_progress`` draws a tqdm bar over the
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

    # Thread the configured context_length into KV feasibility so parallel configs that
    # can't fit the requested sequence length are dropped up front (None -> model max).
    branches: list[BranchSpace] = enumerate_branches(config, max_seq_len=config.search_space.context_length)
    load_predictor = sweep_load_predictor(config)
    if evaluator is None:
        evaluator = ReplayEvaluator(config.workload, goal)

    sweep = config.sweep
    per_round = sweep.candidates_per_round or sweep.parallel_evals
    # Upper bound on evaluations (a Vizier round may return fewer than per_round).
    total = len(branches) * sweep.max_rounds * per_round
    candidates: list[Candidate] = []
    tally = {"feasible": 0, "infeasible": 0, "failed": 0, "unsupported": 0}
    # Unique per run: Vizier's datastore persists studies by id, so a fixed id would
    # make a later run inherit a stale study (and its old param space) -> decode crash.
    run_nonce = uuid.uuid4().hex[:8]
    # Multi-objective (pareto) -> one Vizier metric per objective (each with its own
    # direction); single-objective -> the sampler's default single maximized "objective".
    sampler_objectives = [(t.value, t.maximize) for t in goal.resolved_pareto_objectives] if goal.is_pareto else None

    def _best() -> float | None:
        return max((c.score for c in candidates), default=None)

    # Parallel across worker processes when parallel_evals > 1: spawn (not fork —
    # dynamo's tokio runtime isn't fork-safe); shared read-only state goes once via the
    # initializer; one pool for the whole run amortizes the per-worker dynamo import.
    use_pool = sweep.parallel_evals > 1 and per_round > 1
    pool_cm: Any = (
        ProcessPoolExecutor(
            max_workers=min(sweep.parallel_evals, per_round),
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
            initargs=(config, goal, load_predictor, evaluator),
        )
        if use_pool
        else contextlib.nullcontext()
    )

    def _eval_batch(pool, todo: list[Suggestion]):
        """Yield ``(suggestion, _EvalResult)`` for each supported suggestion — across
        worker processes when ``pool`` is set, else sequentially in-process."""
        if pool is None:
            for s in todo:
                yield (
                    s,
                    _evaluate_one(
                        s.selection,
                        s.parallel_config,
                        config=config,
                        goal=goal,
                        load_predictor=load_predictor,
                        evaluator=evaluator,
                    ),
                )
        else:
            try:
                # submit() can also raise BrokenProcessPool (a worker died before/while
                # tasks were queued), so it must be inside the friendly-error wrapper too.
                futures = {pool.submit(_worker_eval, s.selection, s.parallel_config): s for s in todo}
                for fut in as_completed(futures):
                    yield futures[fut], fut.result()
            except BrokenProcessPool as exc:
                raise RuntimeError(
                    "smart-sweep worker pool died (parallel_evals>1 uses spawned processes). The "
                    "usual cause is calling run_smart_search at a script's top level without guarding "
                    "the entrypoint: spawn re-imports the module, so wrap it in `if __name__ == "
                    '"__main__":`. Or set sweep.parallel_evals=1 to evaluate sequentially (no pool)."'
                ) from exc

    with pool_cm as pool, tqdm(total=total, desc="smart-sweep", unit="eval", disable=not show_progress) as bar:

        def _record(outcome: str, candidate: Candidate | None) -> None:
            tally[outcome] += 1
            if candidate is not None:
                candidates.append(candidate)
            best = _best()
            bar.set_postfix(
                feasible=tally["feasible"], failed=tally["failed"], best=("-" if best is None else f"{best:.4g}")
            )
            bar.update(1)

        for branch in branches:
            sampler = sampler_factory(
                branch, study_id=f"spica_{branch.deployment_mode}_{run_nonce}", objectives=sampler_objectives
            )
            bar.set_description(f"smart-sweep {branch.deployment_mode}")
            for _ in range(sweep.max_rounds):
                suggestions = sampler.suggest(per_round)  # ask (main, serial)
                # Split off (backend, config) pairs the backend can't run: told here,
                # never evaluated. The rest fan out.
                todo: list[Suggestion] = []
                for s in suggestions:
                    if s.selection["backend"] in branch.supported_backends.get(s.parallel_config, frozenset()):
                        todo.append(s)
                    else:
                        sampler.observe_infeasible(
                            s, f"backend {s.selection['backend']!r} does not support this parallel config"
                        )
                        _record("unsupported", None)
                for s, (candidate, observe_metrics, outcome, reason) in _eval_batch(pool, todo):
                    # "failed" (replay error) and "infeasible" (over gpu_budget) are both
                    # gated: tell the sampler observe_infeasible so it learns to avoid that
                    # region instead of being steered toward it by a high objective score.
                    if outcome in ("failed", "infeasible"):  # tell (main, serial)
                        sampler.observe_infeasible(s, reason)
                    else:
                        sampler.observe(s, observe_metrics)
                    _record(outcome, candidate)

    # Single-objective -> rank best-first by score; pareto -> the non-dominated front.
    result = pareto_front(candidates, goal.resolved_pareto_objectives) if goal.is_pareto else rank(candidates)
    if show_progress:
        evaluated = sum(tally.values())
        summary = (
            f"smart-sweep done: {tally['feasible']}/{evaluated} feasible, "
            f"{tally['infeasible']} gated, {tally['unsupported']} backend-unsupported, "
            f"{tally['failed']} replay-failed"
        )
        if not candidates:
            summary += " — NO feasible candidate (check backends / SLA / gpu_budget / replay errors)"
        elif goal.is_pareto:
            summary += f"; pareto front: {len(result)} non-dominated candidate(s)"
        else:
            summary += f"; best {goal.target.value}={_best():.4g}"
        tqdm.write(summary)
    return result

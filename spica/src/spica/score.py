# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score a replay ``trace_report`` against the optimization goal.

Three steps, mirroring the existing profiler replay optimizer
(``components/src/dynamo/profiler/utils/replay_optimize``) adapted to spica's
``Candidate`` / ``OptimizationGoal`` and the merged replay report keys:

1. **objective** — map the goal target to a number from the report (the
   ``goodput_per_gpu_hour`` target is ``goodput / gpu_hour``, both from the report).
2. **feasibility** — SLA satisfied (mean latency within bounds) AND within the
   GPU budget. Infeasible candidates are dropped, not scored (per the design).
3. **rank** — feasible candidates best-first by score, ties broken toward fewer GPUs.
"""

from __future__ import annotations

import math

from .config import Candidate, OptimizationGoal, OptimizationTarget, SLATarget

# trace_report keys the report always carries (goodput_* only when an SLA was
# supplied to the replay). Surfaced into Candidate.metrics for inspection.
_METRIC_KEYS = (
    "output_throughput_tok_s",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "mean_e2e_latency_ms",
    "goodput_output_throughput_tok_s",
    "gpu_hours",
)

# SLA field -> the report key its mean is checked against (matches the profiler:
# itl is checked against mean TPOT, the per-request average inter-token latency).
_SLA_REPORT_KEYS = {
    "ttft_ms": "mean_ttft_ms",
    "itl_ms": "mean_tpot_ms",
    "e2e_ms": "mean_e2e_latency_ms",
}


def objective_value(report: dict[str, float], target: OptimizationTarget) -> float:
    """The raw objective metric (NOT yet signed for direction)."""
    if target is OptimizationTarget.THROUGHPUT:
        return float(report.get("output_throughput_tok_s", 0.0))
    if target is OptimizationTarget.E2E_LATENCY:
        return float(report.get("mean_e2e_latency_ms", math.inf))
    if target is OptimizationTarget.GOODPUT:
        return float(report.get("goodput_output_throughput_tok_s", 0.0))
    if target is OptimizationTarget.GOODPUT_PER_GPU_HOUR:
        gpu_hours = float(report.get("gpu_hours", 0.0))
        goodput = float(report.get("goodput_output_throughput_tok_s", 0.0))
        return goodput / gpu_hours if gpu_hours > 0.0 else 0.0
    raise ValueError(f"unknown optimization target: {target!r}")


def score_report(report: dict[str, float], target: OptimizationTarget) -> float:
    """Objective normalized so **higher is better** (minimized targets negated)."""
    value = objective_value(report, target)
    return value if target.maximize else -value


def sla_violation(report: dict[str, float], sla: SLATarget | None) -> float:
    """Total SLA overage: ``sum(max(actual/bound - 1, 0))`` over the set bounds.

    0.0 means every configured bound is met. A bound whose report key is missing
    contributes ``inf`` (fails the gate rather than silently passing).
    """
    if sla is None:
        return 0.0
    penalty = 0.0
    for field, report_key in _SLA_REPORT_KEYS.items():
        bound = getattr(sla, field)
        if bound is None:
            continue
        actual = report.get(report_key)
        if actual is None:
            return math.inf
        penalty += max(float(actual) / float(bound) - 1.0, 0.0)
    return penalty


def is_feasible(report: dict[str, float], used_gpus: int, goal: OptimizationGoal, gpu_budget: int) -> bool:
    """A candidate is feasible iff it meets the SLA and fits the GPU budget."""
    return sla_violation(report, goal.sla) == 0.0 and used_gpus <= gpu_budget


def make_candidate(config: dict, report: dict[str, float], target: OptimizationTarget) -> Candidate:
    """Build a scored :class:`Candidate` from its config + replay report."""
    metrics = {key: float(report[key]) for key in _METRIC_KEYS if key in report}
    return Candidate(
        config=config,
        used_gpus=int(config.get("used_gpus", 0)),
        score=score_report(report, target),
        metrics=metrics,
    )


def rank(candidates: list[Candidate]) -> list[Candidate]:
    """Best-first: highest score, ties broken toward fewer GPUs."""
    return sorted(candidates, key=lambda c: (-c.score, c.used_gpus))

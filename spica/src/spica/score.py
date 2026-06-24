# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score a replay ``trace_report`` against the optimization goal.

Three steps, mirroring the existing profiler replay optimizer
(``components/src/dynamo/profiler/utils/replay_optimize``) adapted to spica's
``Candidate`` / ``OptimizationGoal`` and the merged replay report keys:

1. **objective** — map the goal target to a number from the report. The
   ``goodput_per_gpu`` / ``throughput_per_gpu`` targets divide ``goodput`` / ``throughput``
   (already a tok/s rate) by the **time-averaged provisioned GPU count**
   ``avg_gpu = gpu_hours / e2e_hours`` — units tok/s/gpu, matching a benchmark's
   "throughput per GPU". For a static deployment ``avg_gpu`` is the fixed GPU count
   (``gpu_hours = gpu_count * e2e_hours``); for a planner-scaled run it is the integral
   of provisioned GPUs over the run divided by its duration. (Dividing by ``gpu_hours``
   directly would be wrong — the rate already has time divided out.)
2. **feasibility** — within the GPU budget. SLA is intentionally *not* gated here:
   when the user cares about latency they pick a ``goodput`` / ``goodput_per_gpu``
   target, whose metric already counts only SLA-satisfying requests (the bridge's
   per-request goodput SLA). Over-budget candidates are dropped.
3. **rank** — feasible candidates best-first by score, ties broken toward fewer GPUs.
"""

from __future__ import annotations

import math

from .config import Candidate, OptimizationTarget

# trace_report keys the report always carries (goodput_* only when an SLA was
# supplied to the replay). Surfaced into Candidate.metrics for inspection.
_METRIC_KEYS = (
    "output_throughput_tok_s",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "mean_e2e_latency_ms",
    "goodput_output_throughput_tok_s",
    "gpu_hours",
    "duration_ms",
)


def _avg_gpu(report: dict[str, float]) -> float:
    """Time-averaged provisioned GPU count = ``gpu_hours / e2e_hours`` (the integral of
    provisioned GPUs over the run, divided by its duration). For a static deployment this
    equals the fixed GPU count; for a planner-scaled run it averages over startup + serve +
    drain. Returns 0.0 when gpu_hours / duration are unavailable (guards divide-by-zero)."""
    gpu_hours = float(report.get("gpu_hours", 0.0))
    duration_ms = float(report.get("duration_ms", 0.0))
    if gpu_hours <= 0.0 or duration_ms <= 0.0:
        return 0.0
    return gpu_hours / (duration_ms / 3_600_000.0)


def objective_value(report: dict[str, float], target: OptimizationTarget) -> float:
    """The raw objective metric (NOT yet signed for direction)."""
    if target is OptimizationTarget.THROUGHPUT:
        return float(report.get("output_throughput_tok_s", 0.0))
    if target is OptimizationTarget.E2E_LATENCY:
        return float(report.get("mean_e2e_latency_ms", math.inf))
    if target is OptimizationTarget.GOODPUT:
        return float(report.get("goodput_output_throughput_tok_s", 0.0))
    if target is OptimizationTarget.GOODPUT_PER_GPU:
        avg_gpu = _avg_gpu(report)
        goodput = float(report.get("goodput_output_throughput_tok_s", 0.0))
        return goodput / avg_gpu if avg_gpu > 0.0 else 0.0
    if target is OptimizationTarget.THROUGHPUT_PER_GPU:
        avg_gpu = _avg_gpu(report)
        throughput = float(report.get("output_throughput_tok_s", 0.0))
        return throughput / avg_gpu if avg_gpu > 0.0 else 0.0
    raise ValueError(f"unknown optimization target: {target!r}")


def score_report(report: dict[str, float], target: OptimizationTarget) -> float:
    """Objective normalized so **higher is better** (minimized targets negated)."""
    value = objective_value(report, target)
    return value if target.maximize else -value


def is_feasible(used_gpus: int, gpu_budget: int) -> bool:
    """A candidate is feasible iff it fits the GPU budget.

    SLA is deliberately not a gate: the goodput targets already bake the SLA into
    their metric (the bridge counts only SLA-satisfying requests per-request), so an
    aggregate mean-latency gate here would double-count it and could drop a genuinely
    high-goodput config whose mean is dragged over by the tail.
    """
    return used_gpus <= gpu_budget


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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Queueing (pass-calendar) correction: TTFT/ITL distribution estimates
derived from scheduler semantics — the structural replacement for the
empirical `_ttft_queuing_factor` heuristic.

Two precision tiers of the same model:
    closed_form.operating_point_columns   O(1) arithmetic on the run_agg
                                          operating point (sweep hot path)
    evaluate_closed_loop                  limit-cycle evaluator: the model's
                                          recursion evaluated numerically —
                                          captures cohort effects the
                                          closed form approximates
    closed_form.static_degenerate_columns static-batching mapping
    DatabaseTimingModel                   timing adapter over (model, database)

Scope: stationary workloads (fixed isl/osl/prefix + closed-loop concurrency
or open-loop rate). Timestamped traces are out of scope; the DES oracle in
tools/queueing_oracle is the validation gate for both tiers.
"""

from .calendar import CALENDARS, evaluate_closed_loop
from .closed_form import (
    QUEUEING_COLUMNS,
    open_loop_queue_wait,
    operating_point_columns,
    static_degenerate_columns,
    ttft_steady_mean,
    ttft_transient_mean,
)
from .spec import Distribution, EngineSpec, QueueingReport, TimingModel, WorkloadSpec
from .timing import DatabaseTimingModel

__all__ = [
    "CALENDARS",
    "QUEUEING_COLUMNS",
    "DatabaseTimingModel",
    "Distribution",
    "EngineSpec",
    "QueueingReport",
    "TimingModel",
    "WorkloadSpec",
    "evaluate_closed_loop",
    "open_loop_queue_wait",
    "operating_point_columns",
    "static_degenerate_columns",
    "static_report",
    "ttft_steady_mean",
    "ttft_transient_mean",
]


def static_report(
    context_latency_ms: float,
    gen_step_latency_ms: float,
    osl: int,
    backend: str = "",
    num_requests: int | None = None,
) -> QueueingReport:
    """Static batching degenerate mapping: no queueing, no interference —
    TTFT collapses to the context latency and ITL/TPOT to the generation
    step latency (single mass points, equal to the legacy scalar columns)."""
    ttft = Distribution()
    ttft.add(context_latency_ms)
    transient = Distribution()
    transient.add(context_latency_ms)
    itl = Distribution()
    itl.add(gen_step_latency_ms)
    tpot = Distribution()
    tpot.add(gen_step_latency_ms)
    return QueueingReport(
        ttft_steady=ttft,
        ttft_transient=transient,
        itl=itl,
        tpot=tpot,
        throughput_rps=0.0,
        output_tokens_per_s=0.0,
        backend=backend,
        mode="static",
        num_requests=num_requests,
    )

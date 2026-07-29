# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Queueing (pass-calendar) correction: TTFT/ITL distribution estimates
derived from scheduler semantics — the structural replacement for the
empirical `_ttft_queuing_factor` heuristic.

One model, several entry points along the workload-fidelity contract
(design doc §3.1):
    closed_form.operating_point_columns   O(1) arithmetic on the run_agg
                                          operating point (sweep hot path)
    evaluate_closed_loop                  W0/W2 closed loop (fixed slots,
                                          immediate replacement)
    evaluate_open_loop                    W1-W3 open loop (rate / quantile
                                          streams / exact arrival_trace
                                          replay)
    evaluate_sessions                     W4 session lanes (endogenous
                                          arrivals: turn k+1 dispatches at
                                          completion_k + think gap)
    evaluate_disagg                       fixed-shape prefill/decode tandem
    closed_form.static_degenerate_columns static-batching mapping
    DatabaseTimingModel                   timing adapter over (model, database)
    trace.workload_from_trace             recorded traces -> exact replay +
                                          W3 stream inputs (prefix-cache
                                          oracle included)

Validation methodology and recorded results: docs/design/queueing_model.md §5.
"""

from .calendar import CALENDARS, evaluate_closed_loop, evaluate_open_loop
from .closed_form import (
    QUEUEING_COLUMNS,
    operating_point_columns,
    static_degenerate_columns,
)
from .disagg import DisaggSpec, evaluate_disagg, evaluate_disagg_mixed
from .sessions import SessionTurn, evaluate_sessions
from .spec import (
    Distribution,
    EngineSpec,
    QueueingReport,
    TimingModel,
    WorkloadSpec,
    stratified_quantiles,
    workload_fidelity,
)
from .timing import DatabaseTimingModel
from .trace import (
    TraceRecord,
    TraceWorkload,
    load_cc_sessions_jsonl,
    load_mooncake_jsonl,
    prefix_hits,
    workload_from_trace,
)

__all__ = [
    "CALENDARS",
    "QUEUEING_COLUMNS",
    "DatabaseTimingModel",
    "DisaggSpec",
    "Distribution",
    "EngineSpec",
    "QueueingReport",
    "SessionTurn",
    "TimingModel",
    "TraceRecord",
    "TraceWorkload",
    "WorkloadSpec",
    "evaluate_closed_loop",
    "evaluate_disagg",
    "evaluate_disagg_mixed",
    "evaluate_open_loop",
    "evaluate_sessions",
    "load_cc_sessions_jsonl",
    "load_mooncake_jsonl",
    "operating_point_columns",
    "prefix_hits",
    "static_degenerate_columns",
    "static_report",
    "stratified_quantiles",
    "workload_fidelity",
    "workload_from_trace",
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
        workload_fidelity="W0(static-degenerate)",
    )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Queueing (pass-calendar) model: distribution-level TTFT/ITL/TPOT estimates
derived from scheduler semantics — the structural replacement for the
empirical `_ttft_queuing_factor` heuristic.

Public API:
    estimate_agg(...)     closed-loop agg (limit-cycle evaluator)
    estimate_disagg(...)  P/D tandem fixed point
    static_report(...)    degenerate mapping for static batching
    DatabaseTimingModel   timing adapter over (model, database)

Scope: stationary workloads (fixed isl/osl/prefix + closed-loop concurrency;
open-loop rate via closed_form only). Timestamped traces belong to the DES
oracle in tools/queueing_oracle. Backend calendars: vllm (validated against
the mocker-parity DES), sglang/trtllm (structural, pending oracle validation).
"""

from .calendar import CALENDARS, evaluate_closed_loop
from .closed_form import (
    open_loop_queue_wait,
    ttft_steady_mean,
    ttft_transient_mean,
)
from .disagg import DisaggSpec, evaluate_disagg, kv_handoff_ms
from .spec import (
    Distribution,
    EngineSpec,
    QueueingReport,
    TimingModel,
    WorkloadSpec,
)
from .timing import DatabaseTimingModel

__all__ = [
    "CALENDARS",
    "DatabaseTimingModel",
    "DisaggSpec",
    "Distribution",
    "EngineSpec",
    "QueueingReport",
    "TimingModel",
    "WorkloadSpec",
    "estimate_agg",
    "estimate_disagg",
    "evaluate_closed_loop",
    "evaluate_disagg",
    "kv_handoff_ms",
    "open_loop_queue_wait",
    "static_report",
    "ttft_steady_mean",
    "ttft_transient_mean",
]


def estimate_agg(
    workload: WorkloadSpec, engine: EngineSpec, timing: TimingModel, backend: str = "vllm"
) -> QueueingReport:
    """Full agg estimate. Closed loop runs the limit-cycle evaluator; open
    loop falls back to closed-form means (single-mass distributions)."""
    if workload.concurrency is not None:
        return evaluate_closed_loop(workload, engine, timing, backend=backend)

    wq = open_loop_queue_wait(workload, engine, timing)
    closed_wl = WorkloadSpec(
        isl=workload.isl,
        osl=workload.osl,
        prefix=workload.prefix,
        concurrency=max(1, round((workload.request_rate or 1.0) * workload.osl / 10.0)),
        num_requests=workload.num_requests,
    )
    base = evaluate_closed_loop(closed_wl, engine, timing, backend=backend)
    shifted = Distribution()
    for v, w in zip(base.ttft_steady.values, base.ttft_steady.weights, strict=True):
        shifted.add(v + wq, w)
    base.ttft_steady = shifted
    base.mode = "agg"
    return base


def estimate_disagg(
    workload: WorkloadSpec,
    prefill_engine: EngineSpec,
    decode_engine: EngineSpec,
    timing: TimingModel,
    disagg: DisaggSpec,
    backend: str = "vllm",
) -> QueueingReport:
    return evaluate_disagg(workload, prefill_engine, decode_engine, timing, disagg, backend=backend)


def static_report(
    context_latency_ms: float, gen_step_latency_ms: float, osl: int, backend: str = "", num_requests: int | None = None
) -> QueueingReport:
    """Static batching degenerate mapping: no queueing, no interference —
    TTFT collapses to the context latency and ITL/TPOT to the generation
    step latency (all distributions are single mass points, by construction
    equal to the legacy scalar columns)."""
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

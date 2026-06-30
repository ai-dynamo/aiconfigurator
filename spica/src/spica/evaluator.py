# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay-backed deployment evaluator: a :class:`DeploymentPlan` -> a trace_report.

Three load shapes (``Workload``) x two deployment cases (static / planner) all route to
a single dynamo replay entrypoint that emits the same flat ``trace_report`` dict:

| load | entrypoint |
|---|---|
| **mooncake trace** | ``run_trace_replay(trace_path, ...)`` |
| **synthetic** (rate or concurrency) | ``run_synthetic_trace_replay(isl, osl, request_count, ...)`` |

``planner_config`` selects the case on either entrypoint: ``None`` -> a static
fixed-fleet replay (the mocker's event-driven ``run()`` loop, no scaling); a dict ->
planner-in-the-loop, where the Rust bridge owns the same loop and calls back into the
Python planner adapter once per ``PlannerTick``. Static returns the bare ``trace_report``
dict; planner-in-the-loop returns a ``ReplayPlannerReport`` whose ``.trace_report`` is the
identical flat dict (:func:`_unwrap` normalizes both). By construction
``plan.is_static == (plan.planner_config is None)``.

The closed-loop in-flight cap is ``replay_concurrency`` for a trace and ``concurrency``
for a synthetic workload (``Workload.effective_in_flight_cap``; a pareto sweep overrides
it per-trial via ``concurrency_override``).

The **goodput SLA** (``goal.sla``) is passed as ``sla_*_ms`` on every path (the mocker
classifies SLA-satisfying requests per-request to compute goodput); it is independent of
the planner's own scaling SLA. Under ``kv_router`` the searched router-weight knobs are
built into a real ``KvRouterConfig``; ``round_robin`` passes ``router_config=None``.

dynamo is imported lazily so ``import spica`` stays light and unit tests can stub the
replay entrypoints.
"""

from __future__ import annotations

import json

from .config import OptimizationGoal, Workload
from .deploy import DeploymentPlan


def _build_kv_router_config(payload: dict | None):
    """A dynamo ``KvRouterConfig`` from spica's router-knob dict (its keys map 1:1 to
    KvRouterConfig kwargs), or ``None`` under round_robin (empty/None payload)."""
    if not payload:
        return None
    from dynamo.llm import KvRouterConfig

    return KvRouterConfig(**payload)


def _unwrap(report) -> dict[str, float]:
    """Normalize a replay result to the flat ``trace_report`` dict: a static replay
    returns that dict directly; planner-in-the-loop returns a ``ReplayPlannerReport``
    whose ``.trace_report`` carries the identical shape."""
    return report.trace_report if hasattr(report, "trace_report") else report


class ReplayEvaluator:
    """Evaluate one candidate by replaying the workload (trace or synthetic)."""

    def __init__(
        self,
        workload: Workload,
        goal: OptimizationGoal,
        *,
        trace_block_size: int = 512,
        benchmark_granularity: int = 8,
    ):
        self.workload = workload
        self.goal = goal
        self.trace_block_size = trace_block_size
        self.benchmark_granularity = benchmark_granularity

    def _goodput_sla_kwargs(self) -> dict[str, float | None]:
        sla = self.goal.sla
        if sla is None:
            return {}
        return {"sla_ttft_ms": sla.ttft_ms, "sla_itl_ms": sla.itl_ms, "sla_e2e_ms": sla.e2e_ms}

    def _router_config(self, plan: DeploymentPlan):
        return _build_kv_router_config(plan.router_config)

    def _synthetic_kwargs(self, concurrency_override: int | None = None) -> dict:
        """The synthetic-workload params ``run_synthetic_trace_replay`` takes
        (``arrival_interval_ms`` is ignored in closed-loop mode).

        ``request_count`` is derived from ``num_request_ratio`` and the effective load
        (the per-trial KV-load-derived ``concurrency_override``, else the workload's fixed
        concurrency / request_rate)."""
        wl = self.workload
        return dict(
            input_tokens=wl.isl,
            output_tokens=wl.osl,
            request_count=wl.resolved_request_count(concurrency_override),
            arrival_interval_ms=wl.synthetic_arrival_interval_ms,
            turns_per_session=wl.turns_per_session,
            shared_prefix_ratio=wl.shared_prefix_ratio,
            num_prefix_groups=wl.num_prefix_groups,
            inter_turn_delay_ms=wl.inter_turn_delay_ms,
        )

    def evaluate(self, plan: DeploymentPlan, *, concurrency_override: int | None = None) -> dict[str, float]:
        """Run one replay and return its trace_report dict.

        ``concurrency_override`` is the per-trial in-flight cap derived from KV load; it
        overrides a fixed workload ``concurrency`` for both the closed-loop cap and the
        ``num_request_ratio``-derived request count. Trace workloads ignore it (they cap
        via ``replay_concurrency``)."""
        if self.workload.is_trace_based:
            return self._evaluate_trace(plan)
        return self._evaluate_synthetic(plan, concurrency_override=concurrency_override)

    # -- trace workloads -------------------------------------------------------

    def _evaluate_trace(self, plan: DeploymentPlan) -> dict[str, float]:
        from dynamo.mocker import MockEngineArgs
        from dynamo.replay.api import run_trace_replay

        wl = self.workload
        common = dict(
            trace_files=wl.trace_path,
            router_mode=plan.router_mode,
            router_config=self._router_config(plan),
            arrival_speedup_ratio=wl.arrival_speedup_ratio,
            trace_block_size=self.trace_block_size,
            replay_concurrency=wl.effective_in_flight_cap(),  # None -> arrival timestamps
            planner_config=None if plan.is_static else plan.planner_config,
            benchmark_granularity=self.benchmark_granularity,
            **self._goodput_sla_kwargs(),
        )
        if plan.deployment_mode == "agg":
            extra = MockEngineArgs.from_json(json.dumps(plan.agg_engine_args))
            report = run_trace_replay(extra_engine_args=extra, num_workers=plan.num_workers, **common)
        else:
            prefill = MockEngineArgs.from_json(json.dumps(plan.prefill_engine_args))
            decode = MockEngineArgs.from_json(json.dumps(plan.decode_engine_args))
            report = run_trace_replay(
                prefill_engine_args=prefill,
                decode_engine_args=decode,
                num_prefill_workers=plan.num_prefill_workers,
                num_decode_workers=plan.num_decode_workers,
                **common,
            )
        return _unwrap(report)

    # -- synthetic workloads ---------------------------------------------------

    def _evaluate_synthetic(self, plan: DeploymentPlan, *, concurrency_override: int | None = None) -> dict[str, float]:
        from dynamo.mocker import MockEngineArgs
        from dynamo.replay.api import run_synthetic_trace_replay

        common = dict(
            router_mode=plan.router_mode,
            router_config=self._router_config(plan),
            arrival_speedup_ratio=1.0,
            replay_concurrency=self.workload.effective_in_flight_cap(concurrency_override),
            planner_config=None if plan.is_static else plan.planner_config,
            benchmark_granularity=self.benchmark_granularity,
            **self._goodput_sla_kwargs(),
            **self._synthetic_kwargs(concurrency_override),
        )
        if plan.deployment_mode == "agg":
            extra = MockEngineArgs.from_json(json.dumps(plan.agg_engine_args))
            report = run_synthetic_trace_replay(extra_engine_args=extra, num_workers=plan.num_workers, **common)
        else:
            prefill = MockEngineArgs.from_json(json.dumps(plan.prefill_engine_args))
            decode = MockEngineArgs.from_json(json.dumps(plan.decode_engine_args))
            report = run_synthetic_trace_replay(
                prefill_engine_args=prefill,
                decode_engine_args=decode,
                num_prefill_workers=plan.num_prefill_workers,
                num_decode_workers=plan.num_decode_workers,
                **common,
            )
        return _unwrap(report)

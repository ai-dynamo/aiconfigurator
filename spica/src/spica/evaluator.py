# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay-backed deployment evaluator: a :class:`DeploymentPlan` -> a trace_report.

Routes by the plan's policy (see the disabled==static investigation):

- ``plan.is_static`` -> ``dynamo.replay.api.run_trace_replay`` (plain, fixed worker
  counts; emits gpu_hours + goodput). If ``workload.replay_concurrency`` is set this path
  runs **closed-loop** (cap N requests in flight, ignore trace timestamps) — the blog's
  throughput/latency Pareto sweep; otherwise it replays at arrival timestamps.
- otherwise -> ``dynamo.replay.main._run_planner_replay`` (planner bridge; scaling;
  emits goodput + gpu_hours). The ``PlannerReplayBridge`` binding only drives arrival-time
  replay and exposes no concurrency cap (the offline engine's ``ReplayMode::Concurrency``
  is orthogonal to the planner, but the bridge constructor takes no ``max_in_flight``), so
  a cap is rejected here until the binding grows one.

The **goodput SLA** (``goal.sla``) is passed as ``sla_*_ms`` on BOTH paths (the
mocker classifies SLA-satisfying requests per-request to compute goodput), so a
static/disabled candidate still produces goodput. It is independent of the planner's
own scaling SLA.

Returns the flat ``trace_report`` dict that :mod:`spica.score` consumes. dynamo
is imported lazily so ``import spica`` stays light and unit tests can stub the
replay entrypoints.

Under ``kv_router`` the searched router-weight knobs are built into a real
``KvRouterConfig`` and passed to the replay so they actually shape routing;
``round_robin`` passes ``router_config=None``.
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


class ReplayEvaluator:
    """Evaluate one candidate by replaying the workload trace."""

    def __init__(
        self,
        workload: Workload,
        goal: OptimizationGoal,
        *,
        trace_block_size: int = 512,
        benchmark_granularity: int = 8,
    ):
        if not workload.is_trace_based:
            raise ValueError("ReplayEvaluator requires a trace-based workload (set workload.trace_path)")
        self.workload = workload
        self.goal = goal
        self.trace_block_size = trace_block_size
        self.benchmark_granularity = benchmark_granularity

    def _goodput_sla_kwargs(self) -> dict[str, float | None]:
        sla = self.goal.sla
        if sla is None:
            return {}
        return {"sla_ttft_ms": sla.ttft_ms, "sla_itl_ms": sla.itl_ms, "sla_e2e_ms": sla.e2e_ms}

    def _concurrency_kwargs(self) -> dict:
        """Closed-loop concurrency cap for the static replay path (None -> arrival-time
        replay). Only reached for static candidates (see the guard in ``evaluate``)."""
        conc = self.workload.replay_concurrency
        if conc is None:
            return {}
        return {"replay_concurrency": conc, "replay_mode": "offline"}

    def _common_kwargs(self, plan: DeploymentPlan) -> dict:
        return dict(
            trace_file=self.workload.trace_path,
            router_mode=plan.router_mode,
            router_config=_build_kv_router_config(plan.router_config),  # kv-router weights applied
            arrival_speedup_ratio=self.workload.arrival_speedup_ratio,
            trace_block_size=self.trace_block_size,
        )

    def evaluate(self, plan: DeploymentPlan) -> dict[str, float]:
        """Run one replay and return its trace_report dict."""
        from dynamo.mocker import MockEngineArgs

        if self.workload.replay_concurrency is not None and not plan.is_static:
            raise ValueError(
                "workload.replay_concurrency (a closed-loop in-flight cap) can't be applied to a "
                "planner/scaling candidate: the PlannerReplayBridge binding only drives arrival-time "
                "replay and exposes no concurrency cap (the offline engine supports one, but the bridge "
                "constructor takes no max_in_flight). Use a static deployment "
                "(planner_scaling_policy=['disabled']) for a concurrency-capped experiment."
            )
        common = self._common_kwargs(plan)
        if plan.deployment_mode == "agg":
            extra = MockEngineArgs.from_json(json.dumps(plan.agg_engine_args))
            if plan.is_static:
                from dynamo.replay.api import run_trace_replay

                return run_trace_replay(
                    extra_engine_args=extra,
                    num_workers=plan.num_workers,
                    **self._goodput_sla_kwargs(),
                    **self._concurrency_kwargs(),
                    **common,
                )
            from dynamo.replay.main import _run_planner_replay

            report = _run_planner_replay(
                extra_engine_args=extra,
                prefill_engine_args=None,
                decode_engine_args=None,
                num_workers=plan.num_workers,
                num_prefill_workers=0,
                num_decode_workers=0,
                planner_config_arg=json.dumps(plan.planner_config),
                benchmark_granularity=self.benchmark_granularity,
                **self._goodput_sla_kwargs(),
                **common,
            )
            return report.trace_report

        prefill = MockEngineArgs.from_json(json.dumps(plan.prefill_engine_args))
        decode = MockEngineArgs.from_json(json.dumps(plan.decode_engine_args))
        if plan.is_static:
            from dynamo.replay.api import run_trace_replay

            return run_trace_replay(
                prefill_engine_args=prefill,
                decode_engine_args=decode,
                num_prefill_workers=plan.num_prefill_workers,
                num_decode_workers=plan.num_decode_workers,
                **self._goodput_sla_kwargs(),
                **self._concurrency_kwargs(),
                **common,
            )
        from dynamo.replay.main import _run_planner_replay

        report = _run_planner_replay(
            extra_engine_args=None,
            prefill_engine_args=prefill,
            decode_engine_args=decode,
            num_workers=0,
            num_prefill_workers=plan.num_prefill_workers,
            num_decode_workers=plan.num_decode_workers,
            planner_config_arg=json.dumps(plan.planner_config),
            benchmark_granularity=self.benchmark_granularity,
            **self._goodput_sla_kwargs(),
            **common,
        )
        return report.trace_report

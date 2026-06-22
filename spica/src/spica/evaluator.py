# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay-backed deployment evaluator: a :class:`DeploymentPlan` -> a trace_report.

Routes by the plan's policy (see the disabled==static investigation):

- ``plan.is_static`` -> ``dynamo.replay.api.run_trace_replay`` (plain, fixed worker
  counts; emits gpu_hours + goodput).
- otherwise -> ``dynamo.replay.main._run_planner_replay`` (planner bridge; scaling;
  emits goodput + gpu_hours).

The **goodput SLA** (``goal.sla``) is passed as ``sla_*_ms`` on BOTH paths (the
mocker classifies SLA-satisfying requests per-request to compute goodput), so a
static/disabled candidate still produces goodput. It is independent of the planner's
own scaling SLA.

Returns the flat ``trace_report`` dict that :mod:`spica.score` consumes. dynamo
is imported lazily so ``import spica`` stays light and unit tests can stub the
replay entrypoints.

NOTE (v1 limitation): the kv-router weight knobs are searched and recorded in the
candidate config but not yet applied to the replay router (``router_config`` is
left at the replay default); wiring a ``KvRouterConfig`` from those knobs is a
follow-up.
"""

from __future__ import annotations

import json

from .config import OptimizationGoal, Workload
from .deploy import DeploymentPlan


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

    def _common_kwargs(self, plan: DeploymentPlan) -> dict:
        return dict(
            trace_file=self.workload.trace_path,
            router_mode=plan.router_mode,
            router_config=None,  # v1: kv-router weight knobs not yet applied (see module note)
            arrival_speedup_ratio=self.workload.arrival_speedup_ratio,
            trace_block_size=self.trace_block_size,
        )

    def evaluate(self, plan: DeploymentPlan) -> dict[str, float]:
        """Run one replay and return its trace_report dict."""
        from dynamo.mocker import MockEngineArgs

        common = self._common_kwargs(plan)
        if plan.deployment_mode == "agg":
            extra = MockEngineArgs.from_json(json.dumps(plan.agg_engine_args))
            if plan.is_static:
                from dynamo.replay.api import run_trace_replay

                return run_trace_replay(
                    extra_engine_args=extra,
                    num_workers=plan.num_workers,
                    **self._goodput_sla_kwargs(),
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

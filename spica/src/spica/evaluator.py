# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay-backed deployment evaluator: a :class:`DeploymentPlan` -> a trace_report.

Three load shapes (``Workload``) × two deployment cases (static / planner) all route
to a dynamo replay entrypoint that emits the same flat ``trace_report`` dict:

| load | static (no planner) | planner-in-the-loop |
|---|---|---|
| **mooncake trace** | ``run_trace_replay`` (arrival, or closed-loop if ``replay_concurrency``) | ``_run_planner_replay(trace_file=…)`` |
| **synthetic** (rate or concurrency) | ``PlannerReplayBridge.from_synthetic`` driven to completion with no scaling | ``_run_planner_replay(synthetic=…)`` |

A synthetic *static* run reuses the planner bridge (``from_synthetic``) without ever
calling ``apply_scaling`` — i.e. a fixed-fleet replay — because that constructor threads
the goodput SLA (and emits gpu_hours), whereas the plain ``run_synthetic_trace_replay``
does not yet take an SLA. The closed-loop in-flight cap is ``replay_concurrency`` for a
trace and ``concurrency`` for a synthetic workload (``Workload.in_flight_cap``).

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

# A simulated-time horizon large enough to drain any replay in one advance.
_SIM_FOREVER_MS = 1.0e15


def _build_kv_router_config(payload: dict | None):
    """A dynamo ``KvRouterConfig`` from spica's router-knob dict (its keys map 1:1 to
    KvRouterConfig kwargs), or ``None`` under round_robin (empty/None payload)."""
    if not payload:
        return None
    from dynamo.llm import KvRouterConfig

    return KvRouterConfig(**payload)


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

    def _synthetic_kwargs(self) -> dict:
        """Common synthetic-workload params shared by ``from_synthetic[_disagg]`` and
        ``SyntheticWorkload`` (arrival_interval_ms is ignored in closed-loop mode)."""
        wl = self.workload
        return dict(
            input_tokens=wl.isl,
            output_tokens=wl.osl,
            request_count=wl.request_count,
            arrival_interval_ms=wl.synthetic_arrival_interval_ms,
            turns_per_session=wl.turns_per_session,
            shared_prefix_ratio=wl.shared_prefix_ratio,
            num_prefix_groups=wl.num_prefix_groups,
            inter_turn_delay_ms=wl.inter_turn_delay_ms,
        )

    @staticmethod
    def _drive_static_bridge(bridge) -> dict[str, float]:
        """Run a ``from_synthetic`` bridge to completion with no ``apply_scaling`` —
        a fixed-fleet (static) synthetic replay — and return its trace_report."""
        while not bridge.advance_to(_SIM_FOREVER_MS)["is_done"]:
            pass
        return bridge.finalize()

    def evaluate(self, plan: DeploymentPlan) -> dict[str, float]:
        """Run one replay and return its trace_report dict."""
        if self.workload.is_trace_based:
            return self._evaluate_trace(plan)
        return self._evaluate_synthetic(plan)

    # -- trace workloads -------------------------------------------------------

    def _evaluate_trace(self, plan: DeploymentPlan) -> dict[str, float]:
        from dynamo.mocker import MockEngineArgs

        wl = self.workload
        cap = wl.in_flight_cap  # replay_concurrency (None -> arrival timestamps)
        sla = self._goodput_sla_kwargs()
        common = dict(
            trace_file=wl.trace_path,
            router_mode=plan.router_mode,
            router_config=self._router_config(plan),
            arrival_speedup_ratio=wl.arrival_speedup_ratio,
            trace_block_size=self.trace_block_size,
        )
        if plan.deployment_mode == "agg":
            extra = MockEngineArgs.from_json(json.dumps(plan.agg_engine_args))
            if plan.is_static:
                from dynamo.replay.api import run_trace_replay

                return run_trace_replay(
                    extra_engine_args=extra,
                    num_workers=plan.num_workers,
                    replay_concurrency=cap,
                    **sla,
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
                replay_concurrency=cap,
                **sla,
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
                replay_concurrency=cap,
                **sla,
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
            replay_concurrency=cap,
            **sla,
            **common,
        )
        return report.trace_report

    # -- synthetic workloads ---------------------------------------------------

    def _evaluate_synthetic(self, plan: DeploymentPlan) -> dict[str, float]:
        from dynamo.mocker import MockEngineArgs

        cap = self.workload.in_flight_cap  # int(concurrency), or None for request-rate
        sla = self._goodput_sla_kwargs()
        syn = self._synthetic_kwargs()
        router_config = self._router_config(plan)

        if plan.deployment_mode == "agg":
            extra = MockEngineArgs.from_json(json.dumps(plan.agg_engine_args))
            if plan.is_static:
                from dynamo.mocker import PlannerReplayBridge

                bridge = PlannerReplayBridge.from_synthetic(
                    extra_engine_args=extra,
                    num_workers=plan.num_workers,
                    router_mode=plan.router_mode,
                    router_config=router_config,
                    replay_concurrency=cap,
                    **syn,
                    **sla,
                )
                return self._drive_static_bridge(bridge)
            from dynamo.replay.main import SyntheticWorkload, _run_planner_replay

            report = _run_planner_replay(
                trace_file=None,
                extra_engine_args=extra,
                prefill_engine_args=None,
                decode_engine_args=None,
                router_config=router_config,
                num_workers=plan.num_workers,
                num_prefill_workers=0,
                num_decode_workers=0,
                router_mode=plan.router_mode,
                arrival_speedup_ratio=1.0,
                trace_block_size=self.trace_block_size,
                planner_config_arg=json.dumps(plan.planner_config),
                benchmark_granularity=self.benchmark_granularity,
                replay_concurrency=cap,
                synthetic=SyntheticWorkload(**syn),
                **sla,
            )
            return report.trace_report

        prefill = MockEngineArgs.from_json(json.dumps(plan.prefill_engine_args))
        decode = MockEngineArgs.from_json(json.dumps(plan.decode_engine_args))
        if plan.is_static:
            from dynamo.mocker import PlannerReplayBridge

            bridge = PlannerReplayBridge.from_synthetic_disagg(
                prefill_engine_args=prefill,
                decode_engine_args=decode,
                num_prefill_workers=plan.num_prefill_workers,
                num_decode_workers=plan.num_decode_workers,
                router_mode=plan.router_mode,
                router_config=router_config,
                replay_concurrency=cap,
                **syn,
                **sla,
            )
            return self._drive_static_bridge(bridge)
        from dynamo.replay.main import SyntheticWorkload, _run_planner_replay

        report = _run_planner_replay(
            trace_file=None,
            extra_engine_args=None,
            prefill_engine_args=prefill,
            decode_engine_args=decode,
            router_config=router_config,
            num_workers=0,
            num_prefill_workers=plan.num_prefill_workers,
            num_decode_workers=plan.num_decode_workers,
            router_mode=plan.router_mode,
            arrival_speedup_ratio=1.0,
            trace_block_size=self.trace_block_size,
            planner_config_arg=json.dumps(plan.planner_config),
            benchmark_granularity=self.benchmark_granularity,
            replay_concurrency=cap,
            synthetic=SyntheticWorkload(**syn),
            **sla,
        )
        return report.trace_report

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Scheduler strategy for single-point worker perf prediction.

A ``Scheduler`` defines *how* one config point's performance is predicted.
Today's only implementation is :class:`StaticScheduler`, which wraps
``backend.run_agg`` (analytic IFB simulation, steady-state) and
``backend.run_static`` (single forward step) -- this matches the
pre-Scheduler behavior of the SDK exactly.

Future implementations may swap in different prediction strategies:

- ``DynamicScheduler`` -- models dynamic-traffic effects beyond the
  steady-state assumption baked into the legacy static scheduler.
- ``MockerScheduler`` -- delegates to Dynamo Mocker for request-level
  discrete-event simulation; useful when the user has a real Mooncake-
  style trace and wants per-request metrics rather than aggregate
  latency / throughput numbers.

Callers (``predict.predict_*``, ``sweep.sweep_*``, ``task.Task.run``)
accept an optional ``scheduler`` argument that defaults to
:data:`DEFAULT_SCHEDULER` (a single :class:`StaticScheduler` instance).
The Protocol surface is intentionally minimal -- only the two prediction
entry points the current SDK uses are required.  Future Mocker-style
schedulers that need a different shape (e.g. whole-disagg-system replay
rather than per-phase prediction) can extend the Protocol with
additional methods at the time they are needed; today's two methods are
sufficient for both agg and disagg paths.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

from aiconfigurator.sdk import config
from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.inference_summary import InferenceSummary
from aiconfigurator.sdk.models import BaseModel
from aiconfigurator.sdk.perf_database import PerfDatabase

DisaggRole = Literal["prefill", "decode"]

# Per-phase latency-correction multipliers used by static disagg scheduling
# (calibrated against silicon measurements; preserved verbatim from V1).
DEFAULT_PREFILL_LATENCY_CORRECTION_SCALE: float = 1.1
DEFAULT_DECODE_LATENCY_CORRECTION_SCALE: float = 1.08


@runtime_checkable
class Scheduler(Protocol):
    """Strategy for predicting one worker's perf at one config point.

    Implementations must provide both prediction entry points; agg and
    disagg sweeps each rely on one of them per point evaluated.
    """

    def predict_agg_worker(
        self,
        *,
        model: BaseModel,
        backend: BaseBackend,
        database: PerfDatabase,
        runtime_config: config.RuntimeConfig,
        ctx_tokens: int,
        **backend_kwargs: Any,
    ) -> InferenceSummary:
        """Predict perf for an aggregated IFB worker at a single (parallel, batch, ctx_tokens) point."""
        ...

    def predict_disagg_phase(
        self,
        *,
        model: BaseModel,
        backend: BaseBackend,
        database: PerfDatabase,
        runtime_config: config.RuntimeConfig,
        role: DisaggRole,
        latency_correction: float = 1.0,
        stride: int = 32,
    ) -> InferenceSummary:
        """Predict perf for one phase of a disagg worker (prefill-only or decode-only)."""
        ...


class StaticScheduler:
    """Default scheduler: steady-state analytic predictions.

    ``predict_agg_worker`` wraps ``backend.run_agg`` (which contains the
    embedded num_mix_steps / num_genonly_steps static IFB scheduler today).
    ``predict_disagg_phase`` wraps ``backend.run_static`` with the
    appropriate mode for prefill / decode.

    This is exactly the SDK's pre-Scheduler behavior; using it as the
    default keeps all current callers' results bit-identical.
    """

    _DISAGG_ROLE_TO_MODE: ClassVar[dict[str, str]] = {
        "prefill": "static_ctx",
        "decode": "static_gen",
    }

    def predict_agg_worker(
        self,
        *,
        model: BaseModel,
        backend: BaseBackend,
        database: PerfDatabase,
        runtime_config: config.RuntimeConfig,
        ctx_tokens: int,
        **backend_kwargs: Any,
    ) -> InferenceSummary:
        return backend.run_agg(
            model,
            database,
            runtime_config,
            ctx_tokens=ctx_tokens,
            **backend_kwargs,
        )

    def predict_disagg_phase(
        self,
        *,
        model: BaseModel,
        backend: BaseBackend,
        database: PerfDatabase,
        runtime_config: config.RuntimeConfig,
        role: DisaggRole,
        latency_correction: float = 1.0,
        stride: int = 32,
    ) -> InferenceSummary:
        mode = self._DISAGG_ROLE_TO_MODE[role]
        return backend.run_static(
            model,
            database,
            runtime_config,
            mode,
            stride,
            latency_correction,
        )


#: The default Scheduler instance used when callers don't pass one.
#: Module-level singleton; ``StaticScheduler`` is stateless so sharing is safe.
DEFAULT_SCHEDULER: Scheduler = StaticScheduler()


__all__ = [
    "DEFAULT_DECODE_LATENCY_CORRECTION_SCALE",
    "DEFAULT_PREFILL_LATENCY_CORRECTION_SCALE",
    "DEFAULT_SCHEDULER",
    "DisaggRole",
    "Scheduler",
    "StaticScheduler",
]

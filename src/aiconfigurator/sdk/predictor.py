# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Predictor strategy for single-point worker perf prediction.

A ``Predictor`` defines *how* one config point's performance is predicted.
Today's only implementation is :class:`AnalyticPredictor`, which wraps
``backend.run_agg`` (analytic IFB simulation, steady-state) and
``backend.run_static`` (single forward step) -- this matches the
pre-Predictor behavior of the SDK exactly.

Future implementations may swap in different prediction strategies:

- ``DynamicPredictor`` -- models dynamic-traffic effects (queueing under
  concurrency) beyond the steady-state assumption baked into the analytic
  predictor.
- ``MockerPredictor`` -- a seam for Dynamo Mocker's request-level
  discrete-event simulation, useful for per-request metrics (e.g. from a
  Mooncake-style trace) rather than aggregate latency / throughput.
  Note the integration is inverted relative to the analytic path: Dynamo
  Mocker imports AIC and drives its ``run_static`` / ``run_agg``, so a
  Mocker-backed predictor wraps that path rather than AIC delegating out
  to Mocker.

Callers (``predict.predict_*``, ``sweep.sweep_*``, ``task.Task.run``)
accept an optional ``predictor`` argument that defaults to
:data:`DEFAULT_PREDICTOR` (a single :class:`AnalyticPredictor` instance).
The Protocol surface is intentionally minimal -- only the two prediction
entry points the current SDK uses are required.  Future Mocker-style
predictors that need a different shape (e.g. whole-disagg-system replay
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


@runtime_checkable
class Predictor(Protocol):
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

    def predict_disagg_worker(
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


class AnalyticPredictor:
    """Default predictor: steady-state analytic predictions (zero-queue).

    ``predict_agg_worker`` wraps ``backend.run_agg`` (which contains the
    embedded num_mix_steps / num_genonly_steps analytic IFB scheduler).
    ``predict_disagg_worker`` wraps ``backend.run_static`` with the
    appropriate mode for prefill / decode.

    The output assumes ideal zero-queue arrival -- it does not model
    queueing-under-concurrency effects.  Sweep / picking layers apply
    a separate TTFT correction on top.

    This matches the SDK's pre-Predictor behavior exactly; using it as
    the default keeps all current callers' results bit-identical.
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

    def predict_disagg_worker(
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


#: The default Predictor instance used when callers don't pass one.
#: Module-level singleton; ``AnalyticPredictor`` is stateless so sharing is safe.
DEFAULT_PREDICTOR: Predictor = AnalyticPredictor()


__all__ = ["DEFAULT_PREDICTOR", "AnalyticPredictor", "DisaggRole", "Predictor"]

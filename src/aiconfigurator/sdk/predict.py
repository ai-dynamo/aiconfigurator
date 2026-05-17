# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Worker-level performance prediction primitives.

Two functions, one per worker shape:

- :func:`predict_disagg_phase` — one phase of a disaggregated worker
  (prefill-only or decode-only).  Thin wrapper over
  ``backend.run_static(mode=static_ctx|static_gen)``.

- :func:`predict_agg_worker` — an aggregated IFB worker (steady-state
  analytic).  Thin wrapper over ``backend.run_agg``.  The static scheduler
  that decomposes the run into ``num_mix_steps`` / ``num_genonly_steps``
  currently lives inside ``backend.run_agg``; when a dynamic-traffic
  scheduler is needed, a sibling function ``predict_agg_dynamic_worker``
  will appear next to this one without disturbing the static path.

Both functions are pure wrappers: they hold no state and do no enumeration.
Callers (typically :mod:`aiconfigurator.sdk.sweep`) drive any search over
``parallel``, ``batch_size``, ``ctx_tokens``, etc., and invoke these once
per evaluation point.
"""

from __future__ import annotations

from typing import Any, Literal

from aiconfigurator.sdk import config
from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.inference_summary import InferenceSummary
from aiconfigurator.sdk.models import BaseModel
from aiconfigurator.sdk.perf_database import PerfDatabase

DisaggRole = Literal["prefill", "decode"]

_DISAGG_ROLE_TO_MODE: dict[str, str] = {
    "prefill": "static_ctx",
    "decode": "static_gen",
}


def predict_disagg_phase(
    *,
    model: BaseModel,
    backend: BaseBackend,
    database: PerfDatabase,
    runtime_config: config.RuntimeConfig,
    role: DisaggRole,
    latency_correction: float = 1.0,
    stride: int = 32,
) -> InferenceSummary:
    """Predict perf for one phase of a disaggregated worker.

    Args:
        model: Model handle from ``models.get_model``.
        backend: Backend handle from ``backends.factory.get_backend``.
        database: Perf database for the worker's (system, backend, version).
        runtime_config: Runtime config carrying ``batch_size``, ``isl``,
            ``osl``, etc. for this evaluation point.
        role: ``"prefill"`` — single prefill step (mode=static_ctx).
            ``"decode"`` — single decode step (mode=static_gen).
        latency_correction: Multiplicative correction applied to predicted
            latencies (passes through to backend.run_static).
        stride: Decode-loop sampling stride (passes through unchanged).

    Returns:
        InferenceSummary for the single-phase worker.

    Notes:
        The current implementation models the phase as a single static step.
        When backends grow chunked-prefill support for disagg prefill workers,
        this signature will gain a ``ctx_tokens: int | None = None``
        parameter — when set, simulate IFB-style chunked prefill via a new
        backend method.  No caller will need to change.
    """
    mode = _DISAGG_ROLE_TO_MODE[role]
    return backend.run_static(
        model,
        database,
        runtime_config,
        mode,
        stride,
        latency_correction,
    )


def predict_agg_worker(
    *,
    model: BaseModel,
    backend: BaseBackend,
    database: PerfDatabase,
    runtime_config: config.RuntimeConfig,
    ctx_tokens: int,
    **backend_kwargs: Any,
) -> InferenceSummary:
    """Predict perf for an aggregated IFB worker (steady-state analytic).

    Args:
        model: Model handle from ``models.get_model``.
        backend: Backend handle from ``backends.factory.get_backend``.
        database: Perf database for the worker's (system, backend, version).
        runtime_config: Runtime config carrying ``batch_size``, ``isl``,
            ``osl``, etc. for this evaluation point.
        ctx_tokens: Per-step context-token budget for the IFB scheduler
            inside ``backend.run_agg``.  Callers enumerate candidate values;
            chunked-prefill semantics are implicit in which values are tried
            (chunked off → ``ctx_tokens`` constrained to multiples of isl).
        **backend_kwargs: Other backend-specific kwargs forwarded as-is.

    Returns:
        InferenceSummary for the aggregated worker.

    Notes:
        Today the static scheduler (num_mix_steps / num_genonly_steps logic)
        lives inside ``backend.run_agg``.  When dynamic-traffic modeling is
        added, a sibling ``predict_agg_dynamic_worker`` will be introduced
        without touching this function.
    """
    return backend.run_agg(
        model,
        database,
        runtime_config,
        ctx_tokens=ctx_tokens,
        **backend_kwargs,
    )

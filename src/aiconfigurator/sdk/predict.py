# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Worker-level performance prediction primitives.

Two functions, one per worker shape:

- :func:`predict_disagg_worker` — one phase of a disaggregated worker
  (prefill-only or decode-only).  Delegates to the predictor's
  ``predict_disagg_worker`` (default: wraps
  ``backend.run_static(mode=static_ctx|static_gen)``).

- :func:`predict_agg_worker` — an aggregated IFB worker.  Delegates to
  the predictor's ``predict_agg_worker`` (default: wraps
  ``backend.run_agg``, where the embedded num_mix_steps /
  num_genonly_steps analytic IFB scheduler lives).

Both functions accept an optional ``predictor`` argument.  When omitted,
``DEFAULT_PREDICTOR`` (a singleton :class:`AnalyticPredictor`) is used --
behavior is bit-identical to direct ``backend.run_*`` calls.  Future
work can replace the predictor with a dynamic-traffic / Mocker-based
implementation without changing this surface.

Both functions are pure wrappers: they hold no state and do no
enumeration.  Callers (typically :mod:`aiconfigurator.sdk.sweep`) drive
any search over ``parallel``, ``batch_size``, ``ctx_tokens``, etc., and
invoke these once per evaluation point.
"""

from __future__ import annotations

from typing import Any, Literal

from aiconfigurator.sdk import config
from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.inference_summary import InferenceSummary
from aiconfigurator.sdk.models import BaseModel
from aiconfigurator.sdk.perf_database import PerfDatabase
from aiconfigurator.sdk.predictor import DEFAULT_PREDICTOR, Predictor
from aiconfigurator.sdk.speculative import SpeculativeDecodingProfile

DisaggRole = Literal["prefill", "decode"]


def predict_disagg_worker(
    *,
    model: BaseModel,
    backend: BaseBackend,
    database: PerfDatabase,
    runtime_config: config.RuntimeConfig,
    role: DisaggRole,
    latency_correction: float = 1.0,
    stride: int = 32,
    predictor: Predictor | None = None,
    speculative_profile: SpeculativeDecodingProfile | None = None,
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
        predictor: Optional Predictor strategy.  Defaults to the analytic
            zero-queue predictor (zero behavior change from direct
            backend.run_static calls).  Future Mocker / dynamic
            predictors can be injected here.

    Returns:
        InferenceSummary for the single-phase worker.

    Notes:
        The default predictor models the phase as a single static step.
        When backends grow chunked-prefill support for disagg prefill
        workers, the predictor interface (and the call site) can be
        extended without touching every caller.
    """
    summary = (predictor or DEFAULT_PREDICTOR).predict_disagg_worker(
        model=model,
        backend=backend,
        database=database,
        runtime_config=runtime_config,
        role=role,
        latency_correction=latency_correction,
        stride=stride,
    )
    if speculative_profile is None:
        return summary
    return speculative_profile.project_summary(summary, role=role)


def predict_agg_worker(
    *,
    model: BaseModel,
    backend: BaseBackend,
    database: PerfDatabase,
    runtime_config: config.RuntimeConfig,
    ctx_tokens: int,
    predictor: Predictor | None = None,
    speculative_profile: SpeculativeDecodingProfile | None = None,
    **backend_kwargs: Any,
) -> InferenceSummary:
    """Predict perf for an aggregated IFB worker.

    Args:
        model: Model handle from ``models.get_model``.
        backend: Backend handle from ``backends.factory.get_backend``.
        database: Perf database for the worker's (system, backend, version).
        runtime_config: Runtime config carrying ``batch_size``, ``isl``,
            ``osl``, etc. for this evaluation point.
        ctx_tokens: Per-step context-token budget.  Callers enumerate
            candidate values; chunked-prefill semantics are implicit in
            which values are tried (chunked off → ``ctx_tokens``
            constrained to multiples of isl).
        predictor: Optional Predictor strategy.  Defaults to the analytic
            zero-queue predictor (zero behavior change from direct
            backend.run_agg calls).  Future Mocker / dynamic predictors
            can be injected here.
        **backend_kwargs: Other backend-specific kwargs forwarded
            through the predictor (e.g. max_seq_len,
            free_gpu_memory_fraction).

    Returns:
        InferenceSummary for the aggregated worker.
    """
    if speculative_profile is not None and speculative_profile.expected_accepted_tokens > 0:
        backend_kwargs["decode_tokens_per_iteration"] = speculative_profile.tokens_per_iteration

    summary = (predictor or DEFAULT_PREDICTOR).predict_agg_worker(
        model=model,
        backend=backend,
        database=database,
        runtime_config=runtime_config,
        ctx_tokens=ctx_tokens,
        **backend_kwargs,
    )
    if speculative_profile is None:
        return summary
    return speculative_profile.project_summary(summary, role="agg")

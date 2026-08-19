# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin facade over the compiled Rust engine (``aiconfigurator_core``).

The only supported path is "Python builds, Rust executes":
``sdk.engine.compile_engine``
walks the model once and emits a bincoded ``EngineSpec``; an ``EngineHandle``
wraps the bytes plus a PyO3 ``AicEngine`` and runs the static / per-step
composition pure-Rust. The helpers here map ``RuntimeConfig`` / raw step args
onto that handle and cache one handle per engine identity.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import OrderedDict
from importlib import resources as pkg_resources
from pathlib import Path
from typing import Any

from aiconfigurator_core.sdk.config import RuntimeConfig

logger = logging.getLogger(__name__)
ENGINE_STEP_BACKEND_ENV = "AICONFIGURATOR_ENGINE_STEP_BACKEND"


# Python-step telemetry (#1357): count every remaining Python op.query() use
# by reason and warn once per reason. Post-retirement the reasons are the
# permanent delegations only — synthetic (non-PerfDatabase) databases and the
# AFD orchestration's op-list fallback.
_PYTHON_STEP_FALLBACK_COUNTS: dict[str, int] = {}
_PYTHON_STEP_FALLBACK_WARNED: set[str] = set()
_PYTHON_STEP_FALLBACK_LOCK = threading.Lock()


def note_python_step_fallback(reason: str, detail: str = "") -> None:
    """Record one Python-step use. First occurrence of a ``reason`` logs at
    WARNING; repeats log at DEBUG. Counts are cumulative per process
    (``python_step_fallback_counts()``)."""
    with _PYTHON_STEP_FALLBACK_LOCK:
        _PYTHON_STEP_FALLBACK_COUNTS[reason] = _PYTHON_STEP_FALLBACK_COUNTS.get(reason, 0) + 1
        first = reason not in _PYTHON_STEP_FALLBACK_WARNED
        if first:
            _PYTHON_STEP_FALLBACK_WARNED.add(reason)
    suffix = f": {detail}" if detail else ""
    if first:
        logger.warning(
            "engine step using the python path (%s)%s — further occurrences log at DEBUG; "
            "cumulative counts via rust_engine_step.python_step_fallback_counts().",
            reason,
            suffix,
        )
    else:
        logger.debug("engine step using the python path (%s)%s", reason, suffix)


def python_step_fallback_counts() -> dict[str, int]:
    """Cumulative Python-step uses by reason (freeze-window telemetry)."""
    with _PYTHON_STEP_FALLBACK_LOCK:
        return dict(_PYTHON_STEP_FALLBACK_COUNTS)


def _python_step_fallback_reset() -> None:
    """Test hook: clear telemetry counters and the warn-once memory."""
    global _PYTHON_BACKEND_DEPRECATION_WARNED
    with _PYTHON_STEP_FALLBACK_LOCK:
        _PYTHON_STEP_FALLBACK_COUNTS.clear()
        _PYTHON_STEP_FALLBACK_WARNED.clear()
        _PYTHON_BACKEND_DEPRECATION_WARNED = False


class RustEngineUnsupportedError(RuntimeError):
    """The model's op graph cannot be expressed as a compiled ``EngineSpec``
    (``engine.OpConversionError``). For op-level models this is a hard error —
    the opspec coverage tripwire keeps it unreachable for shipped ops — while
    the AFD op-list evaluation catches it and lets the Python ``op.query()``
    loop own those ops. Distinct from perf-data misses, which stay
    error-symmetric between the engines."""


class RustForwardPassPerfModel:
    """Facade over the compiled Rust forward-pass perf model (PR #1152).

    Built on the PyO3 ``aiconfigurator_core`` extension (the compiled
    ``Engine``). The public class name and method signatures match PR #1152 so
    callers (the Dynamo planner / mocker) are unaffected; FPM inputs are passed
    as Python dictionaries and marshalled to JSON for the Rust boundary.

    This wrapper is forward-pass-level only. It does not model TTFT, ITL, SLA,
    queueing, or engine limits. ``estimate_forward_pass_time_ms()`` takes one
    iteration as a list of FPM dictionaries, one per attention-DP rank. Single
    rank callers may pass either one FPM dictionary or a one-element list.

    The Rust model infers the workload kind from each iteration's scheduled FPM
    fields:

    * prefill: scheduled prefill tokens and no scheduled decode work, using
      ``[sum_prefill_tokens]``
    * decode: scheduled decode work and no scheduled prefill tokens, using
      ``[num_decode_requests, sum_decode_kv_tokens]``
    * mixed/agg: both scheduled prefill and decode work, using
      ``[sum_prefill_tokens, sum_decode_kv_tokens]``
    * empty: no scheduled prefill or decode work, estimates ``0.0`` and is not
      used for tuning

    Queued request fields are accepted for schema compatibility but ignored by
    this AIC forward-pass model. ``estimate_forward_pass_time_ms()`` treats FPM
    as a workload descriptor: scheduled request fields are used, while
    ``wall_time`` is ignored. ``tune_with_fpms()`` treats FPM as observed
    telemetry: scheduled request fields are used as features and positive
    ``wall_time`` is the latency target. For tuning, ``tune_with_fpms()`` accepts
    multiple iterations as ``[[iter0_rank0, iter0_rank1], [iter1_rank0,
    iter1_rank1]]``. Each iteration is merged using max-rank load features and
    max positive ``wall_time`` across ranks.

    Correction grids use fixed constructor-time ranges from ``options``:
    ``max_num_tokens`` bounds ``sum_prefill_tokens`` and defaults to ``8192``,
    ``max_batch_size`` bounds ``num_decode_requests`` and defaults to ``512``,
    and ``max_kv_tokens`` bounds ``sum_decode_kv_tokens`` and defaults to
    ``2000000``. ``min_faster_correction_factor`` places an absolute lower bound
    on corrections below ``1.0`` and must be finite and in ``(0.0, 1.0]``.
    It defaults to ``0.5``, limiting learned speedups to ``2x``.
    ``max_slower_correction_factor`` independently places an absolute upper
    bound on corrections above ``1.0`` and must be finite and at least ``1.0``.
    It defaults to ``2.0``, limiting learned slowdowns to ``2x``. Passing
    ``None`` for either option leaves that direction unbounded. Regression
    fallback ignores both options.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    @classmethod
    def from_native(
        cls,
        config: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> RustForwardPassPerfModel:
        """API: ``RustForwardPassPerfModel.from_native(config, options=None)``.

        Description: create a strict native AIC forward-pass model.

        Crosses into the Rust core, which compiles ``config`` via
        ``aiconfigurator_core.sdk.engine.compile_engine``. Raises if the config is
        unsupported by the native estimator. Use ``best_available()`` when
        unsupported configs should fall back to the learned regression model.
        """
        _configure_default_data_roots()
        import aiconfigurator_core

        inner = aiconfigurator_core.RustForwardPassPerfModel.from_native(
            _json_dumps(config),
            _optional_json_dumps(options),
        )
        return cls(inner)

    @classmethod
    def best_available(
        cls,
        config: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> RustForwardPassPerfModel:
        """API: ``RustForwardPassPerfModel.best_available(config, options=None)``.

        Description: create a native model when possible, otherwise fall back to
        regression. Fallback reason is available from
        ``diagnostics()["last_warning"]``.
        """
        _configure_default_data_roots()
        import aiconfigurator_core

        inner = aiconfigurator_core.RustForwardPassPerfModel.best_available(
            _json_dumps(config),
            _optional_json_dumps(options),
        )
        return cls(inner)

    @classmethod
    def from_regression(
        cls,
        options: dict[str, Any] | None = None,
    ) -> RustForwardPassPerfModel:
        """API: ``RustForwardPassPerfModel.from_regression(options=None)``.

        Description: create a regression-only forward-pass model. Regression
        models return ``None`` for non-empty estimates until enough samples have
        been provided for the inferred workload kind through
        ``tune_with_fpms()``. Correction factor getters return ``None`` in this
        mode.
        """
        _configure_default_data_roots()
        import aiconfigurator_core

        inner = aiconfigurator_core.RustForwardPassPerfModel.from_regression(
            _optional_json_dumps(options),
        )
        return cls(inner)

    def estimate_forward_pass_time_ms(self, metrics: dict[str, Any] | list[dict[str, Any]]) -> float | None:
        """API: ``model.estimate_forward_pass_time_ms(metrics) -> float | None``.

        Description: estimate one forward-pass iteration in milliseconds.

        ``metrics`` represents one iteration. Pass a list of FPM dictionaries
        for attention-DP ranks, or a single FPM dictionary for a single-rank
        convenience form. The inferred workload kind uses only
        ``scheduled_requests``; queued fields and ``wall_time`` are ignored for
        estimation. Regression models return ``None`` until the matching
        inferred workload kind has enough tuned observations. Empty scheduled
        work returns ``0.0``.
        """
        return self._inner.estimate_forward_pass_time_ms(_json_dumps(metrics))

    def tune_with_fpms(self, iterations: dict[str, Any] | list[Any]) -> None:
        """API: ``model.tune_with_fpms(iterations) -> None``.

        Description: tune the model with one or more observed FPM iterations.

        The canonical input is a nested list ``[[iter0_rank0, iter0_rank1],
        [iter1_rank0, iter1_rank1]]``. Each inner list is one iteration's
        per-attention-DP-rank FPMs. For convenience, a single FPM dictionary is
        normalized to ``[[fpm]]``, and a list of FPM dictionaries is normalized
        to one iteration.
        """
        self._inner.tune_with_fpms(_json_dumps(_normalize_tuning_iterations(iterations)))

    def diagnostics(self) -> dict[str, Any]:
        """API: ``model.diagnostics() -> dict[str, Any]``.

        Description: return source, readiness, retained sample count, and
        fallback warning.
        """
        return json.loads(self._inner.diagnostics())

    def get_min_correction_factor(self) -> float | None:
        """API: ``model.get_min_correction_factor() -> float | None``.

        Description: return the smallest ready native correction factor.
        Regression-only models return ``None``; native models return ``None``
        until at least one correction bucket has enough observations.
        """
        return self._inner.min_correction_factor()

    def get_max_correction_factor(self) -> float | None:
        """API: ``model.get_max_correction_factor() -> float | None``.

        Description: return the largest ready native correction factor.
        """
        return self._inner.max_correction_factor()

    def get_avg_correction_factor(self) -> float | None:
        """API: ``model.get_avg_correction_factor() -> float | None``.

        Description: return the average ready native correction factor.
        """
        return self._inner.avg_correction_factor()

    def close(self) -> None:
        # PyO3 objects are reference-counted; dropping the handle is enough.
        self._inner = None


def _json_dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _optional_json_dumps(value: dict[str, Any] | None) -> str | None:
    if value is None:
        return None
    return _json_dumps(value)


def _normalize_tuning_iterations(iterations: dict[str, Any] | list[Any]) -> list[Any]:
    if isinstance(iterations, dict):
        return [[iterations]]
    if not iterations:
        return []
    if all(isinstance(item, dict) for item in iterations):
        return [iterations]
    return iterations


def validate_engine_step_backend(value: Any) -> str | None:
    """Normalize + validate an ``engine_step_backend`` request.

    Returns the lowered token (``"rust"``) or ``None`` when nothing was
    requested; any other value raises. Shared by the step routing gate AND
    the task/config entry points (``Task.__post_init__``) so a programmatic
    caller cannot smuggle the retired ``"python"`` token (or any typo) into
    a path — like the AFD session — that never reaches the routing gate.
    """
    requested = None if value is None else str(value).lower()
    if requested is not None and requested != "rust":
        raise ValueError(
            f"unknown engine_step_backend {requested!r}: the compiled Rust engine is the only "
            "engine-step executor ('rust' is the only accepted value; the deprecated 'python' "
            "no-op was removed after its one-release window)."
        )
    return requested


def should_use_rust_engine_step(runtime_config: RuntimeConfig, database: Any = None) -> bool:
    """Route the engine step to the compiled engine — the only step executor.

    Returns ``False`` only for the one remaining delegation: by default (not
    when ``"rust"`` is explicitly requested), a non-``PerfDatabase`` object —
    the compiled engine re-loads perf data from disk by identity, which a
    synthetic database does not have. Callers own what ``False`` means: the
    AFD orchestration keeps its per-call twin-op loop, while the
    engine-step surfaces in ``base_backend`` raise (there is no Python step
    left to delegate to).

    ``"rust"`` is the only accepted value. The deprecated ``"python"``
    no-op completed its one-release cycle and was dropped with the
    deprecation-cleanup PR; any other value raises — silently computing on
    an engine the caller did not ask for would be worse than failing.
    """
    backend = getattr(runtime_config, "engine_step_backend", None)
    if backend is None:
        backend = os.environ.get(ENGINE_STEP_BACKEND_ENV)
    requested = validate_engine_step_backend(backend)
    if requested is None:
        # Deferred import: perf_database is heavy and this module must stay
        # light to import (engine.py imports it at top level).
        from aiconfigurator_core.sdk.perf_database import PerfDatabase

        if not isinstance(database, PerfDatabase):
            # The compiled engine re-loads perf data from disk by
            # (system, backend, version); a synthetic/duck-typed database has
            # no on-disk identity it could resolve. Only an explicit "rust"
            # request bypasses this (and owns the resulting load error).
            note_python_step_fallback("non_perf_database", type(database).__name__)
            return False
    return True


def _note_rust_provenance(handle: Any) -> None:
    """Forward the compiled engine's per-call empirical provenance tier into
    Python's capture (``util_empirical.note_provenance``).

    ``EngineHandle.last_provenance()`` returns the worst tier fired during the
    engine call just made (``None`` for a pure-silicon answer). Forwarding it
    keeps ``capture_provenance()`` consumers — the support matrix's
    HYBRID_PASS tier labelling — working unchanged when the engine step is
    rust-routed. ``note_provenance`` is a no-op outside an active capture, so
    this costs one getattr-free call per step. Only the worst tier crosses the
    FFI (not the full tag set); ``worst_provenance`` over the captured tags is
    unaffected because max(worst) == worst(all).
    """
    tier = handle.last_provenance()
    if tier is not None and tier != "silicon":
        # Deferred import: keep module import light and cycle-free
        # (sdk.engine imports this module at top level).
        from aiconfigurator_core.sdk.operations import util_empirical

        util_empirical.note_provenance(tier)


def _scale_or_one(value: Any) -> float:
    """Imbalance-scale forwarding: default to ``1.0`` only for ``None``.

    The Python engine path multiplies by the raw scale, so an explicit
    ``0.0`` must pass through unchanged — a truthiness fallback (``or 1.0``)
    would silently clobber it into ``1.0``.
    """
    return 1.0 if value is None else float(value)


# The PyO3 boundary collapses every Rust error into ValueError (py.rs::
# aic_to_py — the uniform-ValueError contract). But the perf-DB miss class
# ("not collected" / out-of-domain / no cell match / interp miss) is
# semantically Python's PerfDataNotAvailableError, and callers above this
# layer branch on that TYPE: sweep.py marks such points unanswerable and
# skips them, where a genuine ValueError aborts the parallel config. All
# `AicError::PerfDatabase` messages carry this display prefix; re-raise them
# as the class the Python route raises for the same conditions, so both
# routes expose ONE error taxonomy to the sweep.
_RUST_PERF_MISS_PREFIX = "perf database error: "


def _reraise_engine_error(exc: ValueError) -> None:
    from aiconfigurator_core.sdk.errors import PerfDataNotAvailableError

    if str(exc).startswith(_RUST_PERF_MISS_PREFIX):
        raise PerfDataNotAvailableError(str(exc)) from exc
    raise exc


def _fold_per_op(
    entries: Any,
    scale: float = 1.0,
) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
    """Fold the compiled engine's per-op tuples into the Python phase dicts.

    ``entries`` is the FFI's ``[(name, latency_ms, energy_wms, source), ...]``
    — already name-folded inside the engine, so this is an idempotent re-fold
    (it also keeps duck-typed handles in tests correct): duplicate names
    accumulate with ``+=`` and sources merge to ``"mixed"`` on mismatch —
    byte-for-byte the accumulation semantics of
    the retired Python phase runners. ``scale`` is the flat
    ``latency_correction_scale`` post-multiply, applied to latency AND energy
    per key exactly like the Python phase runners' downstream scaling. The
    three dicts share one key set (the power-coverage gate pairs latency and
    energy by identical keys).
    """
    latency: dict[str, float] = {}
    energy: dict[str, float] = {}
    source: dict[str, str] = {}
    for name, latency_ms, energy_wms, src in entries:
        latency[name] = latency.get(name, 0.0) + latency_ms * scale
        energy[name] = energy.get(name, 0.0) + energy_wms * scale
        prior = source.get(name)
        if prior is None:
            source[name] = src
        elif prior != src:
            source[name] = "mixed"
    return latency, energy, source


def estimate_static_latency_breakdown_with_rust(
    model: Any,
    database: Any,
    runtime_config: RuntimeConfig,
    mode: str,
    stride: int,
    latency_correction_scale: float,
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, str],
    dict[str, str],
]:
    """Static (context / generation) per-op breakdown via the compiled engine.

    Routes through ``EngineHandle.run_static_per_op`` (the "Python builds,
    Rust executes" path). The engine performs the decode stride quadrature and
    the ``(nextn + 1)`` decode-batch scaling internally (mirroring
    the retired Python ``_run_generation_phase``) and returns every queried op's
    ``(name, latency_ms, energy_wms, source)``; this side folds them into the
    same name-keyed dicts the Python phase runners produce — real op names,
    real energies, real provenance tags. Returns ``(context_latency,
    generation_latency, context_energy_wms, generation_energy_wms,
    context_source, generation_source)``.
    """
    handle = _cached_engine_handle(model, database)
    engine_mode = mode if mode in {"static", "static_ctx", "static_gen"} else "static"
    try:
        context_ops, generation_ops = handle.run_static_per_op(
            batch_size=int(runtime_config.batch_size),
            isl=int(runtime_config.isl),
            osl=int(runtime_config.osl),
            prefix=int(runtime_config.prefix or 0),
            beam_width=int(runtime_config.beam_width or 1),
            seq_imbalance_correction_scale=_scale_or_one(runtime_config.seq_imbalance_correction_scale),
            gen_seq_imbalance_correction_scale=_scale_or_one(runtime_config.gen_seq_imbalance_correction_scale),
            mode=engine_mode,
            stride=int(stride),
        )
    except ValueError as exc:
        _reraise_engine_error(exc)
    _note_rust_provenance(handle)

    context_latency, context_energy, context_source = _fold_per_op(context_ops, latency_correction_scale)
    generation_latency, generation_energy, generation_source = _fold_per_op(generation_ops, latency_correction_scale)
    return (
        context_latency,
        generation_latency,
        context_energy,
        generation_energy,
        context_source,
        generation_source,
    )


def estimate_mixed_step_latency_with_rust(
    model: Any,
    database: Any,
    *,
    ctx_tokens: int,
    gen_tokens: int,
    isl: int,
    osl: int,
    prefix: int,
    seq_imbalance_correction_scale: float = 1.0,
    gen_seq_imbalance_correction_scale: float = 1.0,
) -> float:
    """Estimate one mixed prefill/decode engine step through the compiled engine.

    Delegates to ``EngineHandle.mixed_step_latency``. The Rust
    ``Engine::mixed_step_latency`` is a literal mirror of Python's
    ``_get_mix_step_latency`` three-pass composition (combined non-attention,
    context attention / ceil(isl/ctx), decode attention with the ``(nextn+1)``
    batch), so the raw step args plus the runtime imbalance scales pass
    straight through with no Python-side pre-math.
    """
    handle = _cached_engine_handle(model, database)
    try:
        latency_ms = handle.mixed_step_latency(
            int(ctx_tokens),
            int(gen_tokens),
            int(isl),
            int(osl),
            int(prefix or 0),
            seq_imbalance_correction_scale=_scale_or_one(seq_imbalance_correction_scale),
            gen_seq_imbalance_correction_scale=_scale_or_one(gen_seq_imbalance_correction_scale),
        )
    except ValueError as exc:
        _reraise_engine_error(exc)
    _note_rust_provenance(handle)
    return latency_ms


def estimate_mixed_step_breakdown_with_rust(
    model: Any,
    database: Any,
    *,
    ctx_tokens: int,
    gen_tokens: int,
    isl: int,
    osl: int,
    prefix: int,
    seq_imbalance_correction_scale: float = 1.0,
    gen_seq_imbalance_correction_scale: float = 1.0,
) -> dict[str, Any]:
    """Estimate one mixed step with per-op and per-component values retained.

    Same three-pass composition as ``estimate_mixed_step_latency_with_rust``
    (``latency_ms`` is the identical sum), reported per pass AND per op so
    ``run_mixed`` builds the same ``StepEstimate`` shape as the Python step:
    non-attention ops under their raw names plus the two literal keys
    ``"context_attention (scaled)"`` (pass 2, already divided by
    ``ceil(isl/ctx)``) and ``"generation_attention"`` (pass 3) — mirroring
    ``base_backend.run_mixed``'s Python branch key-for-key, energies included.
    """
    handle = _cached_engine_handle(model, database)
    try:
        shared_ops, ctx_attn_ops, decode_attn_ops = handle.mixed_step_breakdown_per_op(
            int(ctx_tokens),
            int(gen_tokens),
            int(isl),
            int(osl),
            int(prefix or 0),
            seq_imbalance_correction_scale=_scale_or_one(seq_imbalance_correction_scale),
            gen_seq_imbalance_correction_scale=_scale_or_one(gen_seq_imbalance_correction_scale),
        )
    except ValueError as exc:
        _reraise_engine_error(exc)
    _note_rust_provenance(handle)

    shared_latency, shared_energy, shared_source = _fold_per_op(shared_ops)
    ctx_latency, ctx_energy, ctx_source = _fold_per_op(ctx_attn_ops)
    dec_latency, dec_energy, dec_source = _fold_per_op(decode_attn_ops)

    # Pass 2/3 fold to (at most) the single filtered attention key; missing
    # passes report 0.0 under the Python branch's default "silicon" source
    # (mirrors `.get("context_attention", ...)` / `.get(..., "silicon")`).
    ctx_attention_latency = sum(ctx_latency.values())
    ctx_attention_energy = sum(ctx_energy.values())
    dec_attention_latency = sum(dec_latency.values())
    dec_attention_energy = sum(dec_energy.values())
    per_op_latency_ms: dict[str, float] = {
        **shared_latency,
        "context_attention (scaled)": ctx_attention_latency,
        "generation_attention": dec_attention_latency,
    }
    per_op_source: dict[str, str] = {
        **shared_source,
        "context_attention (scaled)": ctx_source.get("context_attention", "silicon"),
        "generation_attention": dec_source.get("generation_attention", "silicon"),
    }
    component_latency_ms = {
        "shared_non_attention": sum(shared_latency.values()),
        "context_attention": ctx_attention_latency,
        "decode_attention": dec_attention_latency,
    }
    component_energy_wms = {
        "shared_non_attention": sum(shared_energy.values()),
        "context_attention": ctx_attention_energy,
        "decode_attention": dec_attention_energy,
    }
    return {
        "latency_ms": sum(component_latency_ms.values()),
        "energy_wms": sum(component_energy_wms.values()),
        "component_latency_ms": component_latency_ms,
        "component_energy_wms": component_energy_wms,
        "per_op_latency_ms": per_op_latency_ms,
        "per_op_source": per_op_source,
    }


def estimate_decode_step_latency_with_rust(
    model: Any,
    database: Any,
    *,
    gen_tokens: int,
    isl: int,
    osl: int,
    gen_seq_imbalance_correction_scale: float = 1.0,
) -> float:
    """Estimate one decode-only engine step through the compiled engine.

    Delegates to ``EngineHandle.decode_step_latency``. The Rust
    ``Engine::decode_step_latency`` mirrors Python's
    ``_get_genonly_step_latency``: one step over the full generation op list
    at ``s = isl + osl//2 + 1`` with the ``(nextn + 1)`` decode-batch scaling
    applied internally, so the raw args pass straight through.
    """
    handle = _cached_engine_handle(model, database)
    try:
        latency_ms = handle.decode_step_latency(
            int(gen_tokens),
            int(isl),
            int(osl),
            gen_seq_imbalance_correction_scale=_scale_or_one(gen_seq_imbalance_correction_scale),
        )
    except ValueError as exc:
        _reraise_engine_error(exc)
    _note_rust_provenance(handle)
    return latency_ms


def estimate_decode_step_breakdown_with_rust(
    model: Any,
    database: Any,
    *,
    gen_tokens: int,
    isl: int,
    osl: int,
    gen_seq_imbalance_correction_scale: float = 1.0,
) -> tuple[float, float, dict[str, float], dict[str, str]]:
    """``estimate_decode_step_latency_with_rust`` with the per-op values kept.

    Returns ``(latency_ms, energy_wms, per_op_latency, per_op_source)`` —
    the exact shape ``base_backend._get_genonly_step_latency`` produces on the
    Python step, with real op names and per-op energies folded from the
    compiled engine's per-op results.
    """
    handle = _cached_engine_handle(model, database)
    entries = handle.decode_step_per_op(
        int(gen_tokens),
        int(isl),
        int(osl),
        gen_seq_imbalance_correction_scale=_scale_or_one(gen_seq_imbalance_correction_scale),
    )
    _note_rust_provenance(handle)
    latency, energy, source = _fold_per_op(entries)
    return sum(latency.values()), sum(energy.values()), latency, source


def evaluate_context_ops_with_rust(
    model: Any,
    database: Any,
    *,
    indices: Any,
    batch_size: int,
    s: int,
    prefix: int = 0,
    seq_imbalance_correction_scale: float = 1.0,
    x: int | None = None,
) -> list[tuple[str, float, float, str]]:
    """Evaluate an index-addressed sublist of the compiled context op list.

    The thin op-list evaluation FFI: Python-side orchestration (AFD A/F
    partitions) passes positions into ``model.context_ops`` (the compiled
    spec preserves that order 1:1) and receives ``(name, latency_ms,
    energy_wms, source)`` tuples, name-folded (repeated names accumulate,
    sources merge to ``"mixed"`` on mismatch) — the orchestration itself
    stays in Python. ``x`` overrides the token count verbatim for callers
    with their own x policy (AFD's uniform ``batch * s``); ``None`` keeps
    the base-phase rule (logits-GEMM exception).
    """
    handle = _cached_engine_handle(model, database)
    result = handle.evaluate_context_ops(
        list(indices),
        batch_size=int(batch_size),
        s=int(s),
        prefix=int(prefix or 0),
        seq_imbalance_correction_scale=_scale_or_one(seq_imbalance_correction_scale),
        x=x,
    )
    _note_rust_provenance(handle)
    return result


def evaluate_generation_ops_with_rust(
    model: Any,
    database: Any,
    *,
    indices: Any,
    batch_size: int,
    s: int,
    gen_seq_imbalance_correction_scale: float = 1.0,
    prefix: int = 0,
    x: int | None = None,
) -> list[tuple[str, float, float, str]]:
    """Evaluate an index-addressed sublist of the compiled generation op list
    at the decode-step shape (see ``evaluate_context_ops_with_rust``). The
    base decode walk carries no prefix; ``prefix`` exists for orchestrations
    that thread it (AFD's ``_sum_latency``)."""
    handle = _cached_engine_handle(model, database)
    result = handle.evaluate_generation_ops(
        list(indices),
        batch_size=int(batch_size),
        s=int(s),
        gen_seq_imbalance_correction_scale=_scale_or_one(gen_seq_imbalance_correction_scale),
        prefix=int(prefix or 0),
        x=x,
    )
    _note_rust_provenance(handle)
    return result


def evaluate_ops_json_with_rust(
    model: Any,
    database: Any,
    *,
    ops_json: str,
    is_context: bool,
    batch_size: int,
    s: int,
    prefix: int = 0,
    imbalance_correction_scale: float = 1.0,
    x: int | None = None,
) -> list[tuple[str, float, float, str]]:
    """Evaluate an ad-hoc op list (JSON array of OpSpec objects) against the
    engine's database — serves op lists deliberately NOT in the compiled spec
    (the VL encoder phase). The caller keeps the shape math and passes the
    resolved ``(batch_size, s)`` — and optionally an explicit ``x`` — per op
    group.
    """
    handle = _cached_engine_handle(model, database)
    result = handle.evaluate_ops_json(
        ops_json,
        is_context=bool(is_context),
        batch_size=int(batch_size),
        s=int(s),
        prefix=int(prefix or 0),
        imbalance_correction_scale=_scale_or_one(imbalance_correction_scale),
        x=x,
    )
    _note_rust_provenance(handle)
    return result


# LRU memo of compiled ``EngineHandle`` objects, keyed by the engine identity
# (model_path + system + backend + version + parallelism + quant + nextn +
# kv_block_size). ``compile_engine`` rebuilds the model and loads the perf DB,
# which is expensive; the engine-step helpers are called many times per sweep,
# so each unique config must compile + load its DB exactly once. The key is
# ``_engine_config_json``, so two runtime points that differ only in
# batch/isl/osl share one handle.
#
# The memo is BOUNDED: every handle pins its own Rust-side perf-DB load, so an
# unbounded dict grows monotonically with the number of engine identities a
# long-lived process touches (a sweep visits one parallel config at a time and
# a webapp compares a handful, so a small LRU never thrashes; eviction only
# costs the ~100ms-scale recompile on a later re-visit). Negative entries
# (``_CachedUnsupported``) live under the same policy.
_ENGINE_HANDLE_CACHE: OrderedDict[str, Any] = OrderedDict()
_ENGINE_HANDLE_CACHE_MAX = 32
# One lock serializes lookup+recency, insertion+eviction, and clearing:
# ``clear_all_op_caches`` may run on a webapp thread while another thread is
# mid-step, and an unserialized get()/move_to_end() pair would KeyError when a
# clear lands between them. Uncontended acquisition is tens of ns against the
# ~20us step budget (perf gate re-run green).
_ENGINE_HANDLE_CACHE_LOCK = threading.Lock()


class _CachedUnsupported:
    """Message-only negative cache entry. Caching the raised
    ``RustEngineUnsupportedError`` instance instead would pin ``model`` /
    ``database`` via ``__cause__``/``__traceback__`` and grow the traceback on
    every cache-hit re-raise; each hit constructs a fresh exception from the
    message instead."""

    __slots__ = ("message",)

    def __init__(self, message: str) -> None:
        self.message = message


def _engine_handle_cache_clear() -> None:
    """Drop every cached ``EngineHandle`` (and negative entry), releasing the
    Rust-side perf DBs they pin. Used by parity harnesses and by
    ``operations.clear_all_op_caches`` (the long-running-webapp eviction
    lever)."""
    with _ENGINE_HANDLE_CACHE_LOCK:
        _ENGINE_HANDLE_CACHE.clear()


def _engine_handle_cache_get(key: str) -> Any:
    """Look up a handle (or negative entry), refreshing its LRU recency."""
    with _ENGINE_HANDLE_CACHE_LOCK:
        entry = _ENGINE_HANDLE_CACHE.get(key)
        if entry is not None:
            _ENGINE_HANDLE_CACHE.move_to_end(key)
        return entry


def _engine_handle_cache_put(key: str, value: Any) -> None:
    """Insert into the handle LRU, evicting least-recently-used overflow."""
    with _ENGINE_HANDLE_CACHE_LOCK:
        _ENGINE_HANDLE_CACHE[key] = value
        _ENGINE_HANDLE_CACHE.move_to_end(key)
        while len(_ENGINE_HANDLE_CACHE) > _ENGINE_HANDLE_CACHE_MAX:
            _ENGINE_HANDLE_CACHE.popitem(last=False)


def _cached_engine_handle(model: Any, database: Any) -> Any:
    """Return a cached ``EngineHandle`` for ``(model, database)``.

    Builds the compiled ``EngineSpec`` from the ALREADY-BUILT ``model`` via
    ``engine.build_engine_spec_json`` (NOT ``compile_engine``, which would
    rebuild the model from flat args and risk quant/parallel-inference drift),
    then wraps the bincode bytes in an ``EngineHandle``. The handle's Rust
    ``AicEngine`` loads its own perf DB; the system yaml is resolved from the
    ``database``'s own ``systems_root`` (the root the Python ``PerfDatabase``
    actually matched under multi-root ``--systems-paths``), falling back to
    ``AICONFIGURATOR_SYSTEMS_PATH`` (set by ``_configure_default_data_roots``)
    for duck-typed databases without a ``systems_root``.
    """
    # The identity JSON is a hot-path cost: the engine-step helpers call this
    # per step and `_engine_config_json` runs ~2-3us of getattr + json.dumps
    # (which the perf regression gate measures against a ~20us step). The
    # identity is immutable for a given (model, database) pair, so memoize the
    # computed key on the model object and only recompute when the database
    # object changes.
    memo = getattr(model, "_aic_engine_identity_memo", None)
    if memo is not None and memo[0] is database:
        key = memo[1]
    else:
        key = _engine_config_json(model, database)
        try:
            model._aic_engine_identity_memo = (database, key)
        except (AttributeError, TypeError):
            pass  # slotted/frozen model objects: recompute per call
    entry = _engine_handle_cache_get(key)
    if isinstance(entry, _CachedUnsupported):
        # Compilation already failed for this engine identity; raise a fresh
        # error from the cached message instead of re-walking the op graph.
        raise RustEngineUnsupportedError(entry.message)
    if entry is not None:
        return entry

    _configure_default_data_roots()
    # Lazy import: ``sdk.engine`` imports from this module at top level
    # (``_quant_to_dtype`` / ``_moe_quant_to_dtype``), so a top-level import
    # here would be a circular import.
    import aiconfigurator_core
    from aiconfigurator_core.sdk.engine import EngineHandle, OpConversionError, build_engine_spec_json

    # Mirror the root the paired database actually resolved from: with
    # multi-root ``--systems-paths`` the Python PerfDatabase searches every
    # root ("first match wins" per system), while the compiled engine resolves
    # the system yaml from exactly one root — pinning the env default (the
    # first existing root) crashes any system that lives in a later root. The
    # env remains the fallback for duck-typed databases under an explicit
    # ``"rust"`` request.
    systems_path = getattr(database, "systems_root", None) or os.environ.get("AICONFIGURATOR_SYSTEMS_PATH")
    nextn = getattr(model, "_nextn", None)
    try:
        spec_json = build_engine_spec_json(
            model,
            model_path=getattr(model, "model_path", getattr(model, "model_name", "")),
            system=database.system,
            backend=_backend_name(database.backend),
            backend_version=getattr(database, "version", None),
            kv_block_size=None,
            systems_path=systems_path,
            nextn=int(nextn) if nextn is not None else 0,
            database=database,
        )
    except OpConversionError as exc:
        _engine_handle_cache_put(key, _CachedUnsupported(str(exc)))
        raise RustEngineUnsupportedError(str(exc)) from exc
    spec_bytes = bytes(aiconfigurator_core.engine_spec_bincode_from_json(spec_json))
    handle = EngineHandle(spec_bytes, systems_path=systems_path)
    _engine_handle_cache_put(key, handle)
    return handle


def _engine_config_json(model: Any, database: Any) -> str:
    model_config = model.config
    # Forward only the MTP draft length. The aic-core layer models iteration compute cost;
    # accepted-token progress belongs to the upper prediction layer.
    nextn = getattr(model, "_nextn", None)
    config = {
        "schema_version": 1,
        "model_name": getattr(model, "model_path", getattr(model, "model_name", "")),
        "model_arch": getattr(model, "architecture", None),
        "system_name": database.system,
        "backend": _backend_name(database.backend),
        "backend_version": getattr(database, "version", None),
        "tp_size": int(model_config.tp_size or 1),
        "pp_size": int(model_config.pp_size or 1),
        "moe_tp_size": _optional_int(getattr(model_config, "moe_tp_size", None)),
        "moe_ep_size": _optional_int(getattr(model_config, "moe_ep_size", None)),
        "attention_dp_size": _optional_int(getattr(model_config, "attention_dp_size", None)),
        # Part of the engine identity so cp variants get distinct cached handles.
        "cp_size": _optional_int(getattr(model_config, "cp_size", None)),
        "weight_dtype": _quant_to_dtype(getattr(model_config, "gemm_quant_mode", None)),
        "moe_dtype": _moe_quant_to_dtype(getattr(model_config, "moe_quant_mode", None)),
        "activation_dtype": _quant_to_dtype(getattr(model_config, "fmha_quant_mode", None)),
        "kv_cache_dtype": _quant_to_dtype(getattr(model_config, "kvcache_quant_mode", None)),
        "kv_block_size": None,
        "nextn": int(nextn) if nextn is not None else None,
        # An op_level and an fpm model with identical parallel/quant configs
        # compile to DIFFERENT engines (granular op list vs one whole-model op
        # per phase); without this key they would share a cached handle and
        # silently answer with the other mode's engine.
        "forward_model": getattr(model, "forward_model", "op_level"),
        # Same identity built against different systems roots reads different
        # perf trees; the root is part of the engine identity.
        "systems_root": str(getattr(database, "systems_root", "") or ""),
        # Mode + transfer policy are part of the engine identity: a HYBRID or
        # EMPIRICAL view of the same model/system must not reuse a SILICON
        # handle (the compiled engine bakes the mode into its query dispatch).
        "database_mode": _database_mode_key(database),
        "transfer_policy": _transfer_policy_key(database),
        # Cache-identity widening. The dtype fields above collapse distinct
        # quant modes onto one wire string (`sq`/`int8_wo` -> "int8",
        # `fp8_ootb` -> "fp8", four 4-bit modes -> "int4", the DSv4 MoE modes
        # -> None), and several ModelConfig fields shape the compiled op list
        # without appearing in the identity at all. Two models differing only
        # in those would otherwise share one cached handle and silently return
        # each other's latencies. `extra` participates in the JSON key, so
        # carrying the RAW enum names + the op-shaping fields here
        # disambiguates the memo without touching the wire schema.
        # Rust `EngineConfig.extra` is `BTreeMap<String, String>`, so the
        # identity payload is one JSON-encoded STRING value (deserializable
        # if this dict ever crosses the wire, and a stable cache key today).
        "extra": {
            "identity": json.dumps(
                {
                    "raw_quant_modes": {
                        "gemm": _raw_quant_name(getattr(model_config, "gemm_quant_mode", None)),
                        "moe": _raw_quant_name(getattr(model_config, "moe_quant_mode", None)),
                        "fmha": _raw_quant_name(getattr(model_config, "fmha_quant_mode", None)),
                        "kvcache": _raw_quant_name(getattr(model_config, "kvcache_quant_mode", None)),
                        "comm": _raw_quant_name(getattr(model_config, "comm_quant_mode", None)),
                    },
                    "model_config": {
                        "cp_style": getattr(model_config, "cp_style", None),
                        "workload_distribution": getattr(model_config, "workload_distribution", None),
                        "overwrite_num_layers": getattr(model_config, "overwrite_num_layers", None),
                        "sms": getattr(model_config, "sms", None),
                        "moe_backend": getattr(model_config, "moe_backend", None),
                        "attention_backend": getattr(model_config, "attention_backend", None),
                        # enable_wideep is gone from the identity: the deprecated
                        # flag is constant False on every Task-built ModelConfig;
                        # moe_comm_backend + num_gpus_per_node below carry the
                        # large-EP regime.
                        "enable_eplb": bool(getattr(model_config, "enable_eplb", False)),
                        "wideep_num_slots": getattr(model_config, "wideep_num_slots", None),
                        # Large EP: the per-phase comm backend selects a whole
                        # different MoE graph (MoEAllToAll/ModeledEPMoE vs the
                        # fused dispatch/MoE pair) and the node width prices its
                        # cross-node all-to-all — two configs differing only in
                        # these must not share one cached handle.
                        "moe_comm_backend": getattr(model_config, "moe_comm_backend", None),
                        "num_gpus_per_node": getattr(model_config, "num_gpus_per_node", None),
                    },
                    # Data-resolution policy. `build_engine_spec_json` bakes
                    # these flags into the compiled handle and the engine
                    # resolves per-op sources from them (schema v13), so two
                    # views of the same on-disk identity that differ only in
                    # shared-layer or strict-provenance policy must not share
                    # a cached handle — a warmed primary-only handle would
                    # otherwise answer (or fail) for the reuse-carrying view
                    # depending on call order.
                    "database_policy": {
                        "enable_shared_layer": bool(getattr(database, "enable_shared_layer", False)),
                        "strict_provenance": bool(getattr(database, "strict_provenance", False)),
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    }
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _database_mode_key(database: Any) -> str:
    mode = getattr(database, "get_default_database_mode", lambda: None)()
    return getattr(mode, "name", str(mode)) if mode is not None else "SILICON"


def _transfer_policy_key(database: Any) -> list[str] | None:
    policy = getattr(database, "transfer_policy", None)
    if policy is None:
        return None
    return sorted(getattr(kind, "value", str(kind)) for kind in policy)


def _raw_quant_name(value: Any) -> str | None:
    """Un-collapsed quant identity for the handle-cache key: the Python enum
    member name (e.g. ``sq``, ``fp8_ootb``, ``w4afp8``) rather than the lossy
    wire ``DataType`` string."""
    if value is None:
        return None
    return getattr(value, "name", str(value))


def _backend_name(value: Any) -> str:
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _quant_to_dtype(value: Any) -> str | None:
    if value is None:
        return None
    name = getattr(value, "name", str(value)).lower()
    value_name = getattr(getattr(value, "value", None), "name", None)
    if value_name:
        name = value_name.lower()
    if name in {"bfloat16", "half", "float16"}:
        return "bfloat16" if name == "bfloat16" else "float16"
    if name in {"fp8", "fp8_ootb"}:
        return "fp8"
    if name == "fp8_static":
        return "fp8_static"
    if name == "fp8_block":
        return "fp8_block"
    if name == "nvfp4":
        return "nvfp4"
    if name in {"int8", "int8_wo", "sq"}:
        return "int8"
    if name in {"int4", "int4_wo", "w4afp8", "w4a16_mxfp4", "w4a8_mxfp4_mxfp8"}:
        return "int4"
    return None


def _moe_quant_to_dtype(value: Any) -> str | None:
    if value is None:
        return None
    name = getattr(value, "name", str(value)).lower()
    value_name = getattr(getattr(value, "value", None), "name", None)
    if value_name:
        name = value_name.lower()
    if name in {
        "w4afp8",
        "w4a16_mxfp4",
        "w4a8_mxfp4_mxfp8",
        "w4a8_mxfp4_mxfp8_trtllm",
        "w4a16_mxfp4_cutlass",
    }:
        return name
    return _quant_to_dtype(value)


def _configure_default_data_roots() -> None:
    if "AICONFIGURATOR_SYSTEMS_PATH" not in os.environ:
        systems_root = _python_sdk_systems_root() or Path(str(pkg_resources.files("aiconfigurator_core") / "systems"))
        if systems_root.exists():
            os.environ["AICONFIGURATOR_SYSTEMS_PATH"] = str(systems_root)
    if "AICONFIGURATOR_MODEL_CONFIGS_PATH" not in os.environ:
        model_configs_root = Path(str(pkg_resources.files("aiconfigurator_core") / "model_configs"))
        if model_configs_root.exists():
            os.environ["AICONFIGURATOR_MODEL_CONFIGS_PATH"] = str(model_configs_root)


def _python_sdk_systems_root() -> Path | None:
    try:
        from aiconfigurator_core.sdk import perf_database
    except Exception:
        return None
    for candidate in perf_database.get_systems_paths():
        path = Path(candidate)
        if path.exists():
            return path
    return None

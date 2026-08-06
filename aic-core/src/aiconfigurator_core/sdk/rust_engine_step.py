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


class RustEngineUnsupportedError(RuntimeError):
    """The model's op graph cannot be expressed as a compiled ``EngineSpec``
    (``engine.OpConversionError``). Python CAN compute these configs, so the
    ``base_backend`` gates catch this and fall back to the Python step
    (parity by delegation) instead of crashing the sweep. Distinct from
    perf-data misses, which must stay error-symmetric on both engines."""


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
    ``2000000``. ``min_faster_correction_factor`` optionally places an absolute
    lower bound on corrections below ``1.0`` and must be finite and in
    ``(0.0, 1.0]``. ``max_slower_correction_factor`` independently places an
    absolute upper bound on corrections above ``1.0`` and must be finite and at
    least ``1.0``. For example, ``0.5`` limits learned speedups to ``2x``, while
    ``2.0`` limits learned slowdowns to ``2x``. Omitting either option leaves
    that direction unbounded. Regression fallback ignores both options.
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


# Database modes the compiled engine answers itself. SILICON plus the
# util-space empirical layer (HYBRID / EMPIRICAL, mirroring
# `sdk/operations/util_empirical.py`); the SOL diagnostic modes stay on the
# Python step.
_RUST_SUPPORTED_DATABASE_MODES = {"SILICON", "HYBRID", "EMPIRICAL"}


def should_use_rust_engine_step(runtime_config: RuntimeConfig, database: Any = None) -> bool:
    """Route to the compiled engine only when it can give the SAME answer.

    The compiled engine is the DEFAULT. The Python step remains reachable
    three ways, all answer-parity delegations rather than capabilities:

    * an explicit ``engine_step_backend="python"`` (config or env) — the
      escape hatch retained for one release cycle;
    * the SOL / SOL_FULL diagnostic modes, which only the Python step
      implements (the compiled engine answers SILICON / HYBRID / EMPIRICAL);
    * by default (not when ``"rust"`` is explicitly requested), a database
      whose perf tables carry measured power columns — energy does not cross
      the FFI yet, so rust-routing an agg sweep would silently zero its
      ``power_w``. Explicit ``"rust"`` keeps its historical force semantics
      (the parity scan tooling relies on it and measures latency only).
    """
    backend = getattr(runtime_config, "engine_step_backend", None) or os.environ.get(ENGINE_STEP_BACKEND_ENV)
    requested = str(backend).lower() if backend else None
    if requested is not None and requested != "rust":
        return False
    if requested is None:
        # Deferred import: perf_database is heavy and this module must stay
        # light to import (engine.py imports it at top level).
        from aiconfigurator_core.sdk.perf_database import PerfDatabase

        if not isinstance(database, PerfDatabase):
            # The compiled engine re-loads perf data from disk by
            # (system, backend, version); a synthetic/duck-typed database has
            # no on-disk identity it could resolve. Only an explicit "rust"
            # request bypasses this (and owns the resulting load error).
            return False
        if _database_has_power_data(database):
            logger.debug(
                "engine-step backend defaulting to the python step: database %s/%s/%s carries "
                "measured power data and energy does not cross the FFI yet "
                "(set %s=rust to force the compiled engine).",
                database.system,
                database.backend,
                database.version,
                ENGINE_STEP_BACKEND_ENV,
            )
            return False
    if database is not None:
        mode = getattr(database, "get_default_database_mode", lambda: None)()
        if mode is not None and getattr(mode, "name", str(mode)) not in _RUST_SUPPORTED_DATABASE_MODES:
            logger.debug(
                "engine-step backend 'rust' requested but database_mode=%s; "
                "using the python step (compiled engine implements SILICON/HYBRID/EMPIRICAL only).",
                getattr(mode, "name", mode),
            )
            return False
    return True


# Power-data probe results keyed by (system, backend, version). The probe is
# a filesystem schema scan, so the answer is immutable for a given identity;
# memoizing here keeps the per-step routing gate free of I/O.
_POWER_DATA_CACHE: dict[tuple[str, str, str], bool] = {}


def _database_has_power_data(database: Any) -> bool:
    """True when the database's perf-data tree carries measured power columns.

    Detection is a parquet *schema* scan (no row reads) over the database's
    ``<data_dir>/<family>/<backend>/<version>/*.parquet`` tree, plus the
    deprecated ``<data_dir>/<backend>/<version>`` layout. All sibling version
    dirs of the backend are scanned, not just ``database.version``: the loader
    may fill gaps from sibling-version channels, and over-matching only keeps
    that database on the (status quo) Python step. Collectors write power
    columns only when power was actually measured, so column presence is the
    signal — no row values are inspected.

    Any probe failure (mock database objects in tests, missing tree, no
    pyarrow) means "no power data": those databases cannot produce energy on
    the Python step either, so rust-routing them changes nothing.
    """
    system = getattr(database, "system", None)
    backend = getattr(database, "backend", None)
    if not system or not backend:
        return False
    key = (str(system), _backend_name(backend), str(getattr(database, "version", "")))
    cached = _POWER_DATA_CACHE.get(key)
    if cached is not None:
        return cached
    result = _scan_for_power_columns(database)
    _POWER_DATA_CACHE[key] = result
    return result


def _scan_for_power_columns(database: Any) -> bool:
    try:
        import pyarrow.parquet as pq

        systems_root = getattr(database, "systems_root", None)
        spec = getattr(database, "system_spec", None)
        data_dir_rel = spec.get("data_dir") if isinstance(spec, dict) else None
        if not systems_root or not data_dir_rel:
            return False
        data_dir = Path(systems_root) / data_dir_rel
        backend = _backend_name(database.backend)
        candidates = list(data_dir.glob(f"*/{backend}/*/*.parquet")) + list(data_dir.glob(f"{backend}/*/*.parquet"))
        for parquet_path in candidates:
            try:
                names = pq.read_schema(parquet_path).names
            except Exception:  # one unreadable file must not poison the probe
                continue
            if any("power" in name for name in names):
                return True
        return False
    except Exception:  # probe failures mean "no power data", see docstring
        return False


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


def estimate_static_latency_breakdown_with_rust(
    model: Any,
    database: Any,
    runtime_config: RuntimeConfig,
    mode: str,
    stride: int,
    latency_correction_scale: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, str], dict[str, str]]:
    """Static (context / generation) latency breakdown via the compiled engine.

    Routes through ``EngineHandle.run_static`` (the "Python builds, Rust
    executes" path). ``run_static`` performs the decode stride quadrature and
    the ``(nextn + 1)`` decode-batch scaling internally (mirroring
    ``base_backend._run_generation_phase``), so the Python side here only maps
    ``mode`` -> the engine ``mode`` string, applies ``latency_correction_scale``
    after the call, and collapses the scalar phase totals into the synthetic
    single-key breakdown dicts the caller sums.
    """
    handle = _cached_engine_handle(model, database)
    engine_mode = mode if mode in {"static", "static_ctx", "static_gen"} else "static"
    context_latency_ms, generation_latency_ms, _ = handle.run_static(
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
    _note_rust_provenance(handle)

    if latency_correction_scale != 1.0:
        context_latency_ms *= latency_correction_scale
        generation_latency_ms *= latency_correction_scale

    context_latency = {"rust_engine_step_context": context_latency_ms} if context_latency_ms > 0.0 else {}
    generation_latency = {"rust_engine_step_generation": generation_latency_ms} if generation_latency_ms > 0.0 else {}
    context_source = dict.fromkeys(context_latency, "rust")
    generation_source = dict.fromkeys(generation_latency, "rust")
    return context_latency, generation_latency, context_source, generation_source


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
    latency_ms = handle.mixed_step_latency(
        int(ctx_tokens),
        int(gen_tokens),
        int(isl),
        int(osl),
        int(prefix or 0),
        seq_imbalance_correction_scale=_scale_or_one(seq_imbalance_correction_scale),
        gen_seq_imbalance_correction_scale=_scale_or_one(gen_seq_imbalance_correction_scale),
    )
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
) -> dict[str, float]:
    """Estimate one mixed step and retain its three execution components.

    Same three-pass composition as ``estimate_mixed_step_latency_with_rust``
    (``total`` is the identical sum), reported per pass so the agg speculative
    scheduler can consume the shared/context-attention/decode-attention split.
    """
    handle = _cached_engine_handle(model, database)
    total, shared_non_attention, context_attention, decode_attention = handle.mixed_step_breakdown(
        int(ctx_tokens),
        int(gen_tokens),
        int(isl),
        int(osl),
        int(prefix or 0),
        seq_imbalance_correction_scale=_scale_or_one(seq_imbalance_correction_scale),
        gen_seq_imbalance_correction_scale=_scale_or_one(gen_seq_imbalance_correction_scale),
    )
    _note_rust_provenance(handle)
    return {
        "total": float(total),
        "shared_non_attention": float(shared_non_attention),
        "context_attention": float(context_attention),
        "decode_attention": float(decode_attention),
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
    latency_ms = handle.decode_step_latency(
        int(gen_tokens),
        int(isl),
        int(osl),
        gen_seq_imbalance_correction_scale=_scale_or_one(gen_seq_imbalance_correction_scale),
    )
    _note_rust_provenance(handle)
    return latency_ms


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
    # The power probe is filesystem-derived but keyed without the systems
    # root; like the handles above it must not survive a ``set_systems_paths``
    # switch, or a stale ``False`` would rust-route a now-power-carrying
    # identity and silently zero agg ``power_w``.
    _POWER_DATA_CACHE.clear()


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
                        "enable_wideep": bool(getattr(model_config, "enable_wideep", False)),
                        "enable_eplb": bool(getattr(model_config, "enable_eplb", False)),
                        "wideep_num_slots": getattr(model_config, "wideep_num_slots", None),
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

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
from importlib import resources as pkg_resources
from pathlib import Path
from typing import Any

from aiconfigurator_core.sdk.config import RuntimeConfig

logger = logging.getLogger(__name__)
ENGINE_STEP_BACKEND_ENV = "AICONFIGURATOR_ENGINE_STEP_BACKEND"


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
    ``2000000``.
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


def should_use_rust_engine_step(runtime_config: RuntimeConfig, database: Any = None) -> bool:
    """Route to the compiled engine only when it can give the SAME answer.

    The compiled engine implements the SILICON path only (no util_empirical
    layer), so HYBRID/EMPIRICAL databases must stay on the Python step:
    wherever silicon data misses, Python fills in empirically while the
    compiled engine would fail the config -- delegating keeps the two
    backends answer-identical instead of capability-divergent (parity by
    delegation; the empirical-layer port is tracked in issue #1333).
    """
    backend = getattr(runtime_config, "engine_step_backend", None) or os.environ.get(ENGINE_STEP_BACKEND_ENV)
    if str(backend or "python").lower() != "rust":
        return False
    if database is not None:
        mode = getattr(database, "get_default_database_mode", lambda: None)()
        if mode is not None and getattr(mode, "name", str(mode)) != "SILICON":
            logger.debug(
                "engine-step backend 'rust' requested but database_mode=%s; "
                "using the python step (compiled engine is SILICON-only).",
                getattr(mode, "name", mode),
            )
            return False
    return True


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
        seq_imbalance_correction_scale=float(runtime_config.seq_imbalance_correction_scale or 1.0),
        gen_seq_imbalance_correction_scale=float(runtime_config.gen_seq_imbalance_correction_scale or 1.0),
        mode=engine_mode,
        stride=int(stride),
    )

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
) -> float:
    """Estimate one mixed prefill/decode engine step through the compiled engine.

    Delegates to ``EngineHandle.mixed_step_latency``. The Rust
    ``Engine::mixed_step_latency`` (``engine/runtime.rs:280``) reproduces the
    full FPM packing the old ctypes bridge did inline — the
    ``ceil(ctx_tokens / isl)`` prefill-request count, the cached-prefix
    subtraction, the ``(nextn + 1)`` decode multiplier, and the kv-token
    packing — so the raw step args pass straight through with no Python-side
    pre-math.
    """
    handle = _cached_engine_handle(model, database)
    return handle.mixed_step_latency(
        int(ctx_tokens),
        int(gen_tokens),
        int(isl),
        int(osl),
        int(prefix or 0),
    )


def estimate_mixed_step_breakdown_with_rust(
    model: Any,
    database: Any,
    *,
    ctx_tokens: int,
    gen_tokens: int,
    isl: int,
    osl: int,
    prefix: int,
) -> dict[str, float]:
    """Estimate one mixed step and retain its three execution components."""
    handle = _cached_engine_handle(model, database)
    total, shared_non_attention, context_attention, decode_attention = handle.mixed_step_breakdown(
        int(ctx_tokens),
        int(gen_tokens),
        int(isl),
        int(osl),
        int(prefix or 0),
    )
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
) -> float:
    """Estimate one decode-only engine step through the compiled engine.

    Delegates to ``EngineHandle.decode_step_latency``. The Rust
    ``Engine::decode_step_latency`` (``engine/runtime.rs:342``) applies the
    ``(nextn + 1)`` decode-batch scaling and the ``s = isl + osl/2`` sequence
    length internally, so the raw args pass straight through.
    """
    handle = _cached_engine_handle(model, database)
    return handle.decode_step_latency(int(gen_tokens), int(isl), int(osl))


# Memo of compiled ``EngineHandle`` objects, keyed by the engine identity
# (model_path + system + backend + version + parallelism + quant + nextn +
# kv_block_size). ``compile_engine`` rebuilds the model and loads the perf DB,
# which is expensive; the engine-step helpers are called many times per sweep,
# so each unique config must compile + load its DB exactly once. The key is
# ``_engine_config_json``, so two runtime points that differ only in
# batch/isl/osl share one handle.
_ENGINE_HANDLE_CACHE: dict[str, Any] = {}


def _engine_handle_cache_clear() -> None:
    """Reset the compiled-engine handle memo (used by parity harnesses)."""
    _ENGINE_HANDLE_CACHE.clear()


def _cached_engine_handle(model: Any, database: Any) -> Any:
    """Return a cached ``EngineHandle`` for ``(model, database)``.

    Builds the compiled ``EngineSpec`` from the ALREADY-BUILT ``model`` via
    ``engine.build_engine_spec_json`` (NOT ``compile_engine``, which would
    rebuild the model from flat args and risk quant/parallel-inference drift),
    then wraps the bincode bytes in an ``EngineHandle``. The handle's Rust
    ``AicEngine`` loads its own perf DB; ``_configure_default_data_roots`` sets
    ``AICONFIGURATOR_SYSTEMS_PATH`` so it resolves to the same systems tree the
    Python ``database`` came from.
    """
    key = _engine_config_json(model, database)
    handle = _ENGINE_HANDLE_CACHE.get(key)
    if handle is not None:
        return handle

    _configure_default_data_roots()
    # Lazy import: ``sdk.engine`` imports from this module at top level
    # (``_quant_to_dtype`` / ``_moe_quant_to_dtype``), so a top-level import
    # here would be a circular import.
    import aiconfigurator_core
    from aiconfigurator_core.sdk.engine import EngineHandle, build_engine_spec_json

    systems_path = os.environ.get("AICONFIGURATOR_SYSTEMS_PATH")
    nextn = getattr(model, "_nextn", None)
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
    spec_bytes = bytes(aiconfigurator_core.engine_spec_bincode_from_json(spec_json))
    handle = EngineHandle(spec_bytes, systems_path=systems_path)
    _ENGINE_HANDLE_CACHE[key] = handle
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
        "extra": {},
    }
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


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
    if name in {"w4afp8", "w4a16_mxfp4", "w4a8_mxfp4_mxfp8"}:
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

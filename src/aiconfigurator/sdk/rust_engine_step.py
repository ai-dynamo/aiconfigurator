# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin facade over the compiled Rust engine (``aiconfigurator_core``).

Phase 1.5 (E7): the legacy ctypes JSON FFI path (``RustEngineStepEstimator`` /
``RustForwardPassPerfModel`` over ``libaiconfigurator_core``) is gone. The only
supported path is "Python builds, Rust executes": ``sdk.engine.compile_engine``
walks the model once and emits a bincoded ``EngineSpec``; an ``EngineHandle``
wraps the bytes plus a PyO3 ``AicEngine`` and runs the static / per-step
composition pure-Rust. The helpers here map ``RuntimeConfig`` / raw step args
onto that handle and cache one handle per engine identity.
"""

from __future__ import annotations

import json
import os
from importlib import resources as pkg_resources
from pathlib import Path
from typing import Any

from aiconfigurator.sdk.config import RuntimeConfig

ENGINE_STEP_BACKEND_ENV = "AICONFIGURATOR_ENGINE_STEP_BACKEND"


def should_use_rust_engine_step(runtime_config: RuntimeConfig) -> bool:
    backend = getattr(runtime_config, "engine_step_backend", None) or os.environ.get(ENGINE_STEP_BACKEND_ENV)
    return str(backend or "python").lower() == "rust"


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
    from aiconfigurator.sdk.engine import EngineHandle, build_engine_spec_json

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
        nextn_accept_rates=getattr(model, "_nextn_accept_rates", None),
        database=database,
    )
    spec_bytes = bytes(aiconfigurator_core.engine_spec_bincode_from_json(spec_json))
    handle = EngineHandle(spec_bytes, systems_path=systems_path)
    _ENGINE_HANDLE_CACHE[key] = handle
    return handle


def _engine_config_json(model: Any, database: Any) -> str:
    model_config = model.config
    # Forward MTP speculative-decoding params so the Rust DeepSeek-family +
    # Qwen3.5 model builders can compute the same `_mtp_scale_factor` Python
    # applies (`sdk/models/base.py:105-110`). Python sets `nextn=1` for
    # DeepSeek/DSv32/DSv4/Kimi-K2.5/Qwen3.5 by default (`sdk/task.py:448-449`)
    # and stores it on the model object via `BaseModel._nextn`. Rust treats
    # `nextn=None` or `nextn=0` as MTP-disabled (scale=1.0).
    nextn = getattr(model, "_nextn", None)
    nextn_accept_rates = getattr(model, "_nextn_accept_rates", None)
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
        "weight_dtype": _quant_to_dtype(getattr(model_config, "gemm_quant_mode", None)),
        "moe_dtype": _moe_quant_to_dtype(getattr(model_config, "moe_quant_mode", None)),
        "activation_dtype": _quant_to_dtype(getattr(model_config, "fmha_quant_mode", None)),
        "kv_cache_dtype": _quant_to_dtype(getattr(model_config, "kvcache_quant_mode", None)),
        "kv_block_size": None,
        "nextn": int(nextn) if nextn is not None else None,
        "nextn_accept_rates": ([float(r) for r in nextn_accept_rates] if nextn_accept_rates is not None else None),
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
        systems_root = _python_sdk_systems_root() or Path(str(pkg_resources.files("aiconfigurator") / "systems"))
        if systems_root.exists():
            os.environ["AICONFIGURATOR_SYSTEMS_PATH"] = str(systems_root)
    if "AICONFIGURATOR_MODEL_CONFIGS_PATH" not in os.environ:
        model_configs_root = Path(str(pkg_resources.files("aiconfigurator") / "model_configs"))
        if model_configs_root.exists():
            os.environ["AICONFIGURATOR_MODEL_CONFIGS_PATH"] = str(model_configs_root)


def _python_sdk_systems_root() -> Path | None:
    try:
        from aiconfigurator.sdk import perf_database
    except Exception:
        return None
    for candidate in perf_database.get_systems_paths():
        path = Path(candidate)
        if path.exists():
            return path
    return None

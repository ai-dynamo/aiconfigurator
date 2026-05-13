# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import ctypes
import json
import math
import os
import platform
from functools import cache
from importlib import resources as pkg_resources
from pathlib import Path
from typing import Any

from aiconfigurator.sdk.config import RuntimeConfig

ENGINE_STEP_BACKEND_ENV = "AICONFIGURATOR_ENGINE_STEP_BACKEND"
RUST_CORE_LIB_ENV = "AICONFIGURATOR_RUST_CORE_LIB"
_NATIVE_PACKAGE = "aiconfigurator._native"


class RustCoreUnavailableError(RuntimeError):
    """Raised when the Rust core shared library is not available."""


class RustCoreError(RuntimeError):
    """Raised when the Rust core returns an estimator error."""


class RustEngineStepEstimator:
    """ctypes wrapper over the Rust `aiconfigurator-core` FPM estimator."""

    def __init__(self, config: dict[str, Any]) -> None:
        _configure_default_data_roots()
        self._lib = _load_library()
        self._handle = ctypes.c_void_p()
        config_json = _json_bytes(config)
        err = self._lib.aic_engine_step_estimator_new(config_json, ctypes.byref(self._handle))
        _raise_for_error(self._lib, err)

    def forward_pass_time_ms(self, metrics: dict[str, Any] | list[dict[str, Any]]) -> float:
        out_ms = ctypes.c_double()
        metrics_json = _json_bytes(metrics)
        err = self._lib.aic_engine_step_forward_pass_time_ms(
            self._handle,
            metrics_json,
            ctypes.byref(out_ms),
        )
        _raise_for_error(self._lib, err)
        return float(out_ms.value)

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is None or not handle.value:
            return
        self._lib.aic_engine_step_estimator_free(handle)
        self._handle = ctypes.c_void_p()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


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
    estimator = _cached_estimator(_engine_config_json(model, database))
    context_latency_ms = 0.0
    generation_latency_ms = 0.0

    if mode in {"static", "static_ctx"}:
        context_latency_ms = estimator.forward_pass_time_ms(
            _metrics_by_attention_dp_rank(
                model,
                _prefill_metrics(
                    batch_size=int(runtime_config.batch_size),
                    isl=int(runtime_config.isl),
                    prefix=int(runtime_config.prefix or 0),
                ),
            )
        )

    if mode in {"static", "static_gen"}:
        decode_batch_size = int(runtime_config.batch_size) * (int(getattr(model, "_nextn", 0)) + 1)
        beam_width = int(runtime_config.beam_width or 1)
        for i in range(0, max(int(runtime_config.osl) - 1, 0), stride):
            step_latency_ms = estimator.forward_pass_time_ms(
                _metrics_by_attention_dp_rank(
                    model,
                    _decode_metrics(
                        batch_size=decode_batch_size * beam_width,
                        context_length=int(runtime_config.isl) + i,
                    ),
                )
            )
            repeat_count = min(stride, int(runtime_config.osl) - 1 - i)
            generation_latency_ms += step_latency_ms * repeat_count

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
    """Estimate one mixed prefill/decode engine step through the Rust FPM API."""
    estimator = _cached_estimator(_engine_config_json(model, database))
    ctx_tokens = max(int(ctx_tokens), 0)
    gen_tokens = max(int(gen_tokens), 0)
    isl = max(int(isl), 1)
    osl = max(int(osl), 1)
    prefix = max(int(prefix or 0), 0)

    scheduled_requests: dict[str, Any] = {}
    if ctx_tokens > 0:
        num_prefill_requests = max(math.ceil(ctx_tokens / isl), 1)
        scheduled_requests.update(
            {
                "num_prefill_requests": num_prefill_requests,
                "sum_prefill_tokens": ctx_tokens,
                "sum_prefill_kv_tokens": prefix * num_prefill_requests,
            }
        )
    if gen_tokens > 0:
        scheduled_requests.update(
            {
                "num_decode_requests": gen_tokens,
                "sum_decode_kv_tokens": gen_tokens * (isl + osl // 2),
            }
        )

    if not scheduled_requests:
        return 0.0
    return estimator.forward_pass_time_ms(
        _metrics_by_attention_dp_rank(model, {"version": 1, "scheduled_requests": scheduled_requests})
    )


def estimate_decode_step_latency_with_rust(
    model: Any,
    database: Any,
    *,
    gen_tokens: int,
    isl: int,
    osl: int,
) -> float:
    """Estimate one decode-only engine step through the Rust FPM API."""
    estimator = _cached_estimator(_engine_config_json(model, database))
    gen_tokens = max(int(gen_tokens), 0)
    if gen_tokens == 0:
        return 0.0
    context_length = max(int(isl), 1) + max(int(osl), 1) // 2
    return estimator.forward_pass_time_ms(
        _metrics_by_attention_dp_rank(model, _decode_metrics(batch_size=gen_tokens, context_length=context_length))
    )


def is_rust_core_available() -> bool:
    try:
        _load_library()
    except RustCoreUnavailableError:
        return False
    return True


@cache
def _cached_estimator(config_json: str) -> RustEngineStepEstimator:
    return RustEngineStepEstimator(json.loads(config_json))


def _metrics_by_attention_dp_rank(model: Any, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rank_count = max(int(getattr(model.config, "attention_dp_size", 1) or 1), 1)
    return [copy.deepcopy(metrics) for _ in range(rank_count)]


def _engine_config_json(model: Any, database: Any) -> str:
    model_config = model.config
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
        "activation_dtype": _quant_to_dtype(getattr(model_config, "fmha_quant_mode", None)),
        "kv_cache_dtype": _quant_to_dtype(getattr(model_config, "kvcache_quant_mode", None)),
        "kv_block_size": None,
        "extra": {},
    }
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _prefill_metrics(*, batch_size: int, isl: int, prefix: int) -> dict[str, Any]:
    effective_isl = max(isl - prefix, 0)
    return {
        "version": 1,
        "scheduled_requests": {
            "num_prefill_requests": batch_size,
            "sum_prefill_tokens": batch_size * effective_isl,
            "sum_prefill_kv_tokens": batch_size * prefix,
        },
    }


def _decode_metrics(*, batch_size: int, context_length: int) -> dict[str, Any]:
    return {
        "version": 1,
        "scheduled_requests": {
            "num_decode_requests": batch_size,
            "sum_decode_kv_tokens": batch_size * context_length,
        },
    }


@cache
def _load_library() -> ctypes.CDLL:
    library_path = _find_library()
    if library_path is None:
        raise RustCoreUnavailableError(
            "Rust core shared library not found. Install the project with "
            "`pip install -e .` (which builds the cdylib via setuptools-rust), or "
            f"set {RUST_CORE_LIB_ENV} to a prebuilt library."
        )

    lib = ctypes.CDLL(str(library_path))
    lib.aic_engine_step_estimator_new.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.aic_engine_step_estimator_new.restype = ctypes.c_void_p
    lib.aic_engine_step_forward_pass_time_ms.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.aic_engine_step_forward_pass_time_ms.restype = ctypes.c_void_p
    lib.aic_engine_step_estimator_free.argtypes = [ctypes.c_void_p]
    lib.aic_engine_step_estimator_free.restype = None
    lib.aic_engine_step_string_free.argtypes = [ctypes.c_void_p]
    lib.aic_engine_step_string_free.restype = None
    return lib


def _find_library() -> Path | None:
    explicit = os.environ.get(RUST_CORE_LIB_ENV)
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path
        raise RustCoreUnavailableError(f"{RUST_CORE_LIB_ENV} points to a missing file: {path}")

    packaged = _find_packaged_library()
    if packaged is not None:
        return packaged

    crate_root = _crate_root()
    if crate_root is None:
        return None
    lib_name = _library_name()
    candidates = [
        crate_root / "target" / "release" / lib_name,
        crate_root / "target" / "debug" / lib_name,
    ]
    return next((path for path in candidates if path.is_file()), None)


_PACKAGED_LIBRARY_EXTENSIONS: tuple[str, ...] = (".so", ".dylib", ".pyd", ".dll")
_PACKAGED_LIBRARY_STEM = "aiconfigurator_core"


def _find_packaged_library() -> Path | None:
    """Resolve the cdylib that setuptools-rust drops into ``aiconfigurator._native``.

    setuptools-rust names the artifact with Python's ``EXT_SUFFIX``
    (e.g. ``aiconfigurator_core.cpython-312-x86_64-linux-gnu.so``) rather than
    the bare ``libaiconfigurator_core.so`` that ``cargo build`` produces. We
    glob for any filename starting with ``aiconfigurator_core`` and ending in
    a known shared-library extension so both naming conventions resolve.
    """
    try:
        package_root = pkg_resources.files(_NATIVE_PACKAGE)
    except (ModuleNotFoundError, FileNotFoundError):
        return None
    try:
        entries = list(package_root.iterdir())
    except (AttributeError, OSError, NotADirectoryError):
        return None

    matches: list[Path] = []
    for resource in entries:
        name = getattr(resource, "name", "")
        if not name.startswith(_PACKAGED_LIBRARY_STEM):
            continue
        if not name.endswith(_PACKAGED_LIBRARY_EXTENSIONS):
            continue
        try:
            path = Path(str(resource))
        except TypeError:
            continue
        if path.is_file():
            matches.append(path)

    if not matches:
        return None
    # Prefer the longest filename — setuptools-rust's EXT_SUFFIX form is more
    # specific than a bare `aiconfigurator_core.so`, and if both exist we want
    # the one matching the running interpreter ABI.
    matches.sort(key=lambda p: len(p.name), reverse=True)
    return matches[0]


def _crate_root() -> Path | None:
    search_starts = [Path(__file__).resolve().parent, Path.cwd().resolve()]
    searched: set[Path] = set()
    for start in search_starts:
        for parent in (start, *start.parents):
            if parent in searched:
                continue
            searched.add(parent)
            candidate = parent / "rust" / "aiconfigurator-core"
            if (candidate / "Cargo.toml").is_file():
                return candidate
    return None


def _library_name() -> str:
    system = platform.system()
    if system == "Darwin":
        return "libaiconfigurator_core.dylib"
    if system == "Windows":
        return "aiconfigurator_core.dll"
    return "libaiconfigurator_core.so"


def _raise_for_error(lib: ctypes.CDLL, error_ptr: int | None) -> None:
    if not error_ptr:
        return
    try:
        message = ctypes.cast(error_ptr, ctypes.c_char_p).value
        raise RustCoreError((message or b"unknown Rust core error").decode("utf-8", errors="replace"))
    finally:
        lib.aic_engine_step_string_free(error_ptr)


def _json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


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

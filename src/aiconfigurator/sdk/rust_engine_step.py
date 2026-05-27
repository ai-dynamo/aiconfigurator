# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import ctypes
import json
import math
import os
import platform
import shutil
import subprocess
from functools import cache
from importlib import resources as pkg_resources
from pathlib import Path
from typing import Any

from aiconfigurator.sdk.config import RuntimeConfig

ENGINE_STEP_BACKEND_ENV = "AICONFIGURATOR_ENGINE_STEP_BACKEND"
RUST_CORE_LIB_ENV = "AICONFIGURATOR_RUST_CORE_LIB"
RUST_CORE_AUTOBUILD_ENV = "AICONFIGURATOR_RUST_CORE_AUTOBUILD"


class RustCoreUnavailableError(RuntimeError):
    """Raised when the Rust core shared library is not available."""


class RustCoreError(RuntimeError):
    """Raised when the Rust core returns an estimator error."""


class RustEngineStepEstimator:
    """ctypes wrapper over the Rust `aiconfigurator-core` FPM estimator."""

    def __init__(self, config: dict[str, Any], *, autobuild: bool | None = None) -> None:
        _configure_default_data_roots()
        self._lib = _load_library(bool(autobuild) or _truthy(os.environ.get(RUST_CORE_AUTOBUILD_ENV)))
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
        """API: `model.close() -> None`.

        Description: release the underlying Rust forward-pass perf model handle.
        """
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


class RustForwardPassPerfModel:
    """ctypes wrapper over the tuned/fallback Rust forward-pass perf model.

    This wrapper is forward-pass-level only. It does not model TTFT, ITL, SLA,
    queueing, or engine limits. `estimate_forward_pass_time_ms()` takes one
    iteration as a list of FPM dictionaries, one per attention-DP rank. Single
    rank callers may pass either one FPM dictionary or a one-element list.

    The Rust model infers the tuning shape from each iteration's scheduled FPM
    fields:

    * prefill: scheduled prefill tokens and no scheduled decode work, using
      `[sum_prefill_tokens]`
    * decode: scheduled decode work and no scheduled prefill tokens, using
      `[num_decode_requests, sum_decode_kv_tokens]`
    * mixed/agg: both scheduled prefill and decode work, using
      `[sum_prefill_tokens, sum_decode_kv_tokens]`
    * empty: no scheduled prefill or decode work, estimates `0.0` and is not
      used for tuning

    Queued request fields are accepted for schema compatibility but ignored by
    this AIC forward-pass model. For tuning, `tune_with_fpms()` accepts multiple
    iterations as `[[iter0_rank0, iter0_rank1], [iter1_rank0, iter1_rank1]]`.
    Each iteration is merged using max-rank load features and max positive
    `wall_time` across ranks.
    """

    def __init__(self, handle: ctypes.c_void_p, lib: ctypes.CDLL) -> None:
        self._handle = handle
        self._lib = lib

    @classmethod
    def from_native(
        cls,
        config: dict[str, Any],
        options: dict[str, Any] | None = None,
        *,
        autobuild: bool | None = None,
    ) -> RustForwardPassPerfModel:
        """API: `RustForwardPassPerfModel.from_native(config, options=None, *, autobuild=None)`.

        Description: create a strict native AIC forward-pass model.

        This constructor raises `RustCoreError` if the config is unsupported by
        the native estimator. Use `auto()` when unsupported configs should fall
        back to the learned regression model.
        """
        return cls._create(
            "aic_forward_pass_perf_model_from_native",
            config=config,
            options=options,
            autobuild=autobuild,
        )

    @classmethod
    def auto(
        cls,
        config: dict[str, Any],
        options: dict[str, Any] | None = None,
        *,
        autobuild: bool | None = None,
    ) -> RustForwardPassPerfModel:
        """API: `RustForwardPassPerfModel.auto(config, options=None, *, autobuild=None)`.

        Description: create a native model when possible, otherwise fall back to
        regression.

        Fallback reason is available from `diagnostics()["last_warning"]`.
        """
        return cls._create(
            "aic_forward_pass_perf_model_auto",
            config=config,
            options=options,
            autobuild=autobuild,
        )

    @classmethod
    def from_regression(
        cls,
        options: dict[str, Any] | None = None,
        *,
        autobuild: bool | None = None,
    ) -> RustForwardPassPerfModel:
        """API: `RustForwardPassPerfModel.from_regression(options=None, *, autobuild=None)`.

        Description: create a regression-only forward-pass model.

        Regression models return `None` for non-empty estimates until enough
        samples have been provided for the inferred shape through
        `tune_with_fpms()`. Correction factor getters return `None` in this
        mode.
        """
        return cls._create(
            "aic_forward_pass_perf_model_from_regression",
            options=options,
            autobuild=autobuild,
        )

    @classmethod
    def _create(
        cls,
        function_name: str,
        *,
        config: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        autobuild: bool | None = None,
    ) -> RustForwardPassPerfModel:
        _configure_default_data_roots()
        lib = _load_library(bool(autobuild) or _truthy(os.environ.get(RUST_CORE_AUTOBUILD_ENV)))
        _bind_forward_pass_perf_model_api(lib)
        handle = ctypes.c_void_p()
        constructor = getattr(lib, function_name)
        if config is None:
            err = constructor(_optional_json_bytes(options), ctypes.byref(handle))
        else:
            err = constructor(_json_bytes(config), _optional_json_bytes(options), ctypes.byref(handle))
        _raise_for_error(lib, err)
        return cls(handle, lib)

    def estimate_forward_pass_time_ms(self, metrics: dict[str, Any] | list[dict[str, Any]]) -> float | None:
        """API: `model.estimate_forward_pass_time_ms(metrics) -> float | None`.

        Description: estimate one forward-pass iteration in milliseconds.

        `metrics` represents one iteration. Pass a list of FPM dictionaries for
        attention-DP ranks, or a single FPM dictionary for a single-rank
        convenience form. The inferred shape uses only `scheduled_requests`;
        queued fields and `wall_time` are ignored for estimation.

        Native models return an estimate immediately, optionally corrected by a
        ready tuning bucket. Regression models return `None` until the matching
        inferred shape has enough tuned observations. Empty scheduled work
        returns `0.0`.
        """
        out_ms = ctypes.c_double()
        out_has_value = ctypes.c_bool()
        err = self._lib.aic_forward_pass_perf_model_estimate_forward_pass_time_ms(
            self._handle,
            _json_bytes(metrics),
            ctypes.byref(out_ms),
            ctypes.byref(out_has_value),
        )
        _raise_for_error(self._lib, err)
        return float(out_ms.value) if out_has_value.value else None

    def tune_with_fpms(self, iterations: dict[str, Any] | list[Any]) -> None:
        """API: `model.tune_with_fpms(iterations) -> None`.

        Description: tune the model with one or more observed FPM iterations.

        The canonical input is a nested list:
        `[[iter0_rank0, iter0_rank1], [iter1_rank0, iter1_rank1]]`.
        Each inner list is one iteration's per-attention-DP-rank FPMs. For
        convenience, a single FPM dictionary is normalized to `[[fpm]]`, and a
        list of FPM dictionaries is normalized to one iteration.

        Tuning infers the shape from scheduled request fields. It ignores empty
        iterations and iterations with no positive finite `wall_time`. For
        multi-rank input, one observation is recorded using max-rank load
        features and max positive `wall_time` across ranks.
        """
        err = self._lib.aic_forward_pass_perf_model_tune_with_fpms(
            self._handle,
            _json_bytes(_normalize_tuning_iterations(iterations)),
        )
        _raise_for_error(self._lib, err)

    def diagnostics(self) -> dict[str, Any]:
        """API: `model.diagnostics() -> dict[str, Any]`.

        Description: return source, readiness, retained sample count, and
        fallback warning.
        """
        out_json = ctypes.c_void_p()
        err = self._lib.aic_forward_pass_perf_model_diagnostics_json(
            self._handle,
            ctypes.byref(out_json),
        )
        _raise_for_error(self._lib, err)
        try:
            message = ctypes.cast(out_json, ctypes.c_char_p).value
            return json.loads((message or b"{}").decode("utf-8", errors="replace"))
        finally:
            self._lib.aic_engine_step_string_free(out_json)

    def get_min_correction_factor(self) -> float | None:
        """API: `model.get_min_correction_factor() -> float | None`.

        Description: return the smallest ready native correction factor.

        Regression-only models return `None`. Native models return `None` until
        at least one correction bucket has enough observations.
        """
        return self._get_correction_factor("aic_forward_pass_perf_model_min_correction_factor")

    def get_max_correction_factor(self) -> float | None:
        """API: `model.get_max_correction_factor() -> float | None`.

        Description: return the largest ready native correction factor.

        Regression-only models return `None`. Native models return `None` until
        at least one correction bucket has enough observations.
        """
        return self._get_correction_factor("aic_forward_pass_perf_model_max_correction_factor")

    def get_avg_correction_factor(self) -> float | None:
        """API: `model.get_avg_correction_factor() -> float | None`.

        Description: return the average ready native correction factor.

        Regression-only models return `None`. Native models return `None` until
        at least one correction bucket has enough observations.
        """
        return self._get_correction_factor("aic_forward_pass_perf_model_avg_correction_factor")

    def _get_correction_factor(self, function_name: str) -> float | None:
        out_value = ctypes.c_double()
        out_has_value = ctypes.c_bool()
        err = getattr(self._lib, function_name)(
            self._handle,
            ctypes.byref(out_value),
            ctypes.byref(out_has_value),
        )
        _raise_for_error(self._lib, err)
        return float(out_value.value) if out_has_value.value else None

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is None or not handle.value:
            return
        self._lib.aic_forward_pass_perf_model_free(handle)
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


def is_rust_core_available(*, autobuild: bool = False) -> bool:
    try:
        _load_library(autobuild)
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
        "moe_dtype": _moe_quant_to_dtype(getattr(model_config, "moe_quant_mode", None)),
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
def _load_library(autobuild: bool) -> ctypes.CDLL:
    library_path = _find_library(include_debug=not autobuild)
    if library_path is None and autobuild:
        library_path = _build_rust_core()
    if library_path is None:
        raise RustCoreUnavailableError(
            "Rust core shared library not found. Build it with "
            "`cargo build --release --manifest-path rust/aiconfigurator-core/Cargo.toml`, "
            f"set {RUST_CORE_LIB_ENV}, or set {RUST_CORE_AUTOBUILD_ENV}=1."
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


def _bind_forward_pass_perf_model_api(lib: ctypes.CDLL) -> None:
    if lib.__dict__.get("_aic_forward_pass_perf_model_api_bound", False):
        return

    try:
        _bind_forward_pass_perf_model_symbols(lib)
    except AttributeError as exc:
        raise RustCoreUnavailableError(
            "Rust core shared library does not expose the ForwardPassPerfModel API. "
            "Build a newer aiconfigurator-core shared library with "
            "`cargo build --release --manifest-path rust/aiconfigurator-core/Cargo.toml`, "
            f"set {RUST_CORE_LIB_ENV} to that library, or set {RUST_CORE_AUTOBUILD_ENV}=1."
        ) from exc

    lib.__dict__["_aic_forward_pass_perf_model_api_bound"] = True


def _bind_forward_pass_perf_model_symbols(lib: ctypes.CDLL) -> None:
    lib.aic_forward_pass_perf_model_from_native.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.aic_forward_pass_perf_model_from_native.restype = ctypes.c_void_p
    lib.aic_forward_pass_perf_model_auto.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.aic_forward_pass_perf_model_auto.restype = ctypes.c_void_p
    lib.aic_forward_pass_perf_model_from_regression.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.aic_forward_pass_perf_model_from_regression.restype = ctypes.c_void_p
    lib.aic_forward_pass_perf_model_estimate_forward_pass_time_ms.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_bool),
    ]
    lib.aic_forward_pass_perf_model_estimate_forward_pass_time_ms.restype = ctypes.c_void_p
    lib.aic_forward_pass_perf_model_tune_with_fpms.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    lib.aic_forward_pass_perf_model_tune_with_fpms.restype = ctypes.c_void_p
    lib.aic_forward_pass_perf_model_diagnostics_json.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.aic_forward_pass_perf_model_diagnostics_json.restype = ctypes.c_void_p
    for name in [
        "aic_forward_pass_perf_model_min_correction_factor",
        "aic_forward_pass_perf_model_max_correction_factor",
        "aic_forward_pass_perf_model_avg_correction_factor",
    ]:
        function = getattr(lib, name)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_bool),
        ]
        function.restype = ctypes.c_void_p
    lib.aic_forward_pass_perf_model_free.argtypes = [ctypes.c_void_p]
    lib.aic_forward_pass_perf_model_free.restype = None


def _find_library(*, include_debug: bool = True) -> Path | None:
    explicit = os.environ.get(RUST_CORE_LIB_ENV)
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path
        raise RustCoreUnavailableError(f"{RUST_CORE_LIB_ENV} points to a missing file: {path}")

    crate_root = _crate_root()
    if crate_root is None:
        return None
    lib_name = _library_name()
    candidates = [crate_root / "target" / "release" / lib_name]
    if include_debug:
        candidates.append(crate_root / "target" / "debug" / lib_name)
    return next((path for path in candidates if path.is_file()), None)


def _build_rust_core() -> Path:
    crate_root = _crate_root()
    if crate_root is None:
        raise RustCoreUnavailableError("could not locate rust/aiconfigurator-core/Cargo.toml")
    if shutil.which("cargo") is None:
        raise RustCoreUnavailableError("cargo is not available on PATH")

    subprocess.run(
        ["cargo", "build", "--release", "--manifest-path", str(crate_root / "Cargo.toml")],
        check=True,
    )
    library_path = crate_root / "target" / "release" / _library_name()
    if not library_path.is_file():
        raise RustCoreUnavailableError(f"cargo build completed but did not produce {library_path}")
    return library_path


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


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _optional_json_bytes(value: dict[str, Any] | None) -> bytes | None:
    if value is None:
        return None
    return _json_bytes(value)


def _normalize_tuning_iterations(iterations: dict[str, Any] | list[Any]) -> list[Any]:
    if isinstance(iterations, dict):
        return [[iterations]]
    if not iterations:
        return []
    if all(isinstance(item, dict) for item in iterations):
        return [iterations]
    return iterations


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


def _truthy(value: str | None) -> bool:
    return str(value or "").lower() in {"1", "true", "yes", "on"}

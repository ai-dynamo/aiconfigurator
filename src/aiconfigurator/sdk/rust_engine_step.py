# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
import uuid
from functools import cache
from importlib import resources as pkg_resources
from pathlib import Path
from typing import Any

from aiconfigurator.sdk.config import RuntimeConfig

ENGINE_STEP_BACKEND_ENV = "AICONFIGURATOR_ENGINE_STEP_BACKEND"
RUST_SYSTEMS_SOURCE_ENV = "AICONFIGURATOR_RUST_SYSTEMS_SOURCE_PATH"

_RUST_VERSION_PERF_FILES = (
    "gemm_perf.txt",
    "context_attention_perf.txt",
    "generation_attention_perf.txt",
    "custom_allreduce_perf.txt",
    "moe_perf.txt",
    "context_mla_perf.txt",
    "generation_mla_perf.txt",
)


class RustCoreUnavailableError(RuntimeError):
    """Raised when the Rust core PyO3 extension is not importable."""


class RustCoreError(RuntimeError):
    """Raised when the Rust core returns an estimator error."""


class RustEngineStepEstimator:
    """PyO3 wrapper over the Rust ``aiconfigurator-core`` FPM estimator."""

    def __init__(self, config: dict[str, Any]) -> None:
        _configure_default_data_roots(config)
        try:
            self._inner = _new_estimator(_json_str(config))
        except RustCoreUnavailableError:
            raise
        except Exception as err:
            raise RustCoreError(str(err)) from err

    def forward_pass_time_ms(self, metrics: dict[str, Any] | list[dict[str, Any]]) -> float:
        if isinstance(metrics, dict):
            metrics = [metrics]
        try:
            return float(self._inner.forward_pass_time_ms(_json_str(metrics)))
        except Exception as err:
            raise RustCoreError(str(err)) from err

    # TODO(remove-after-rust-migration): parity check/benchmark-only cache reset.
    def clear_runtime_caches(self) -> None:
        try:
            self._inner.clear_runtime_caches()
        except Exception as err:
            raise RustCoreError(str(err)) from err

    def close(self) -> None:
        self._inner = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class RustForwardPassPerfModel:
    """PyO3 wrapper over the tuned/fallback Rust forward-pass perf model.

    This wrapper is forward-pass-level only. It does not model TTFT, ITL, SLA,
    queueing, or engine limits. `estimate_forward_pass_time_ms()` takes one
    iteration as a list of FPM dictionaries, one per attention-DP rank. Single
    rank callers may pass either one FPM dictionary or a one-element list.

    The Rust model infers the workload kind from each iteration's scheduled FPM
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
    this AIC forward-pass model. `estimate_forward_pass_time_ms()` treats FPM as
    a workload descriptor: scheduled request fields are used, while `wall_time`
    is ignored. `tune_with_fpms()` treats FPM as observed telemetry: scheduled
    request fields are used as features and positive `wall_time` is the latency
    target. For tuning, `tune_with_fpms()` accepts multiple iterations as
    `[[iter0_rank0, iter0_rank1], [iter1_rank0, iter1_rank1]]`. Each iteration
    is merged using max-rank load features and max positive `wall_time` across
    ranks.

    Correction grids use fixed constructor-time ranges from `options`:
    `max_num_tokens` bounds `sum_prefill_tokens` and defaults to `8192`,
    `max_batch_size` bounds `num_decode_requests` and defaults to `512`, and
    `max_kv_tokens` bounds `sum_decode_kv_tokens` and defaults to `2000000`.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    @classmethod
    def from_native(
        cls,
        config: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> RustForwardPassPerfModel:
        """API: `RustForwardPassPerfModel.from_native(config, options=None)`.

        Description: create a strict native AIC forward-pass model.

        This constructor raises `RustCoreError` if the config is unsupported by
        the native estimator. Use `best_available()` when unsupported configs
        should fall back to the learned regression model.
        """
        return cls._create("from_native", config=config, options=options)

    @classmethod
    def best_available(
        cls,
        config: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> RustForwardPassPerfModel:
        """API: `RustForwardPassPerfModel.best_available(config, options=None)`.

        Description: create a native model when possible, otherwise fall back to
        regression.

        Fallback reason is available from `diagnostics()["last_warning"]`.
        """
        return cls._create("best_available", config=config, options=options)

    @classmethod
    def from_regression(
        cls,
        options: dict[str, Any] | None = None,
    ) -> RustForwardPassPerfModel:
        """API: `RustForwardPassPerfModel.from_regression(options=None)`.

        Description: create a regression-only forward-pass model.

        Regression models return `None` for non-empty estimates until enough
        samples have been provided for the inferred workload kind through
        `tune_with_fpms()`. Correction factor getters return `None` in this
        mode.
        """
        return cls._create("from_regression", options=options)

    @classmethod
    def _create(
        cls,
        method_name: str,
        *,
        config: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> RustForwardPassPerfModel:
        _configure_default_data_roots(config)
        factory = getattr(_perf_model_class(), method_name)
        options_json = _json_str(options) if options is not None else None
        try:
            if config is None:
                inner = factory(options_json)
            else:
                inner = factory(_json_str(config), options_json)
        except Exception as err:
            raise RustCoreError(str(err)) from err
        return cls(inner)

    def estimate_forward_pass_time_ms(self, metrics: dict[str, Any] | list[dict[str, Any]]) -> float | None:
        """API: `model.estimate_forward_pass_time_ms(metrics) -> float | None`.

        Description: estimate one forward-pass iteration in milliseconds.

        `metrics` represents one iteration. Pass a list of FPM dictionaries for
        attention-DP ranks, or a single FPM dictionary for a single-rank
        convenience form. The inferred workload kind uses only `scheduled_requests`;
        queued fields and `wall_time` are ignored for estimation.

        Native models return an estimate immediately, multiplied by the
        correction factor for the matching workload region. Inferred workload
        kinds with fewer than `min_observations` total samples and empty regions
        use the default factor `1.0`. Queries outside the configured correction
        bounds in `options` also use factor `1.0`. Regression models return
        `None` until the matching inferred workload kind has enough tuned
        observations. Empty scheduled work returns `0.0`.
        """
        try:
            value = self._inner.estimate_forward_pass_time_ms(_json_str(metrics))
        except Exception as err:
            raise RustCoreError(str(err)) from err
        return float(value) if value is not None else None

    def tune_with_fpms(self, iterations: dict[str, Any] | list[Any]) -> None:
        """API: `model.tune_with_fpms(iterations) -> None`.

        Description: tune the model with one or more observed FPM iterations.

        The canonical input is a nested list:
        `[[iter0_rank0, iter0_rank1], [iter1_rank0, iter1_rank1]]`.
        Each inner list is one iteration's per-attention-DP-rank FPMs. For
        convenience, a single FPM dictionary is normalized to `[[fpm]]`, and a
        list of FPM dictionaries is normalized to one iteration.

        Tuning infers the workload kind from scheduled request fields. It
        ignores empty iterations and iterations with no positive finite
        `wall_time`. For native models, each observation updates only its
        matching correction region; those regions are used after the inferred
        workload kind has enough total samples. Native correction ignores
        observations outside the configured correction bounds. For multi-rank
        input, one observation is recorded using max-rank load features and max
        positive `wall_time` across ranks.
        """
        try:
            self._inner.tune_with_fpms(_json_str(iterations))
        except Exception as err:
            raise RustCoreError(str(err)) from err

    def diagnostics(self) -> dict[str, Any]:
        """API: `model.diagnostics() -> dict[str, Any]`.

        Description: return source, readiness, retained sample count, and
        fallback warning.
        """
        try:
            return json.loads(self._inner.diagnostics_json())
        except Exception as err:
            raise RustCoreError(str(err)) from err

    def get_min_correction_factor(self) -> float | None:
        """API: `model.get_min_correction_factor() -> float | None`.

        Description: return the smallest ready native correction factor.

        Regression-only models return `None`. Native models return `None` until
        at least one correction bucket has enough observations.
        """
        return self._inner.min_correction_factor()

    def get_max_correction_factor(self) -> float | None:
        """API: `model.get_max_correction_factor() -> float | None`.

        Description: return the largest ready native correction factor.

        Regression-only models return `None`. Native models return `None` until
        at least one correction bucket has enough observations.
        """
        return self._inner.max_correction_factor()

    def get_avg_correction_factor(self) -> float | None:
        """API: `model.get_avg_correction_factor() -> float | None`.

        Description: return the average ready native correction factor.

        Regression-only models return `None`. Native models return `None` until
        at least one correction bucket has enough observations.
        """
        return self._inner.avg_correction_factor()

    def close(self) -> None:
        self._inner = None

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
        _import_rust_core()
    except RustCoreUnavailableError:
        return False
    return True


@cache
def _cached_estimator(config_json: str) -> RustEngineStepEstimator:
    return RustEngineStepEstimator(json.loads(config_json))


@cache
def _import_rust_core():
    """Import the PyO3 extension module once and cache it.

    The Rust accelerator ships as a separate distribution
    (``aiconfigurator-core``) so the bare ``aiconfigurator`` wheel stays
    pure-Python and platform-independent. Users opt into the rust path with
    the ``[rust]`` extra; on platforms where no precompiled extension wheel
    exists we fall back to the Python latency path.
    """
    try:
        import aiconfigurator_core  # type: ignore[import-not-found]
    except ImportError as err:
        raise RustCoreUnavailableError(
            "Rust core not installed. Install the optional extension with:\n"
            "  pip install aiconfigurator[rust]\n"
            "or directly:\n"
            "  pip install aiconfigurator-core\n"
            "Bare `pip install aiconfigurator` ships only the pure-Python path."
        ) from err
    return aiconfigurator_core


def _new_estimator(config_json: str):
    return _import_rust_core().PyEngineStepEstimator(config_json)


def _perf_model_class():
    return _import_rust_core().PyForwardPassPerfModel


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


def _json_str(value: dict[str, Any] | list[dict[str, Any]]) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


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


def _configure_default_data_roots(config: dict[str, Any] | None = None) -> None:
    systems_root = _configured_systems_source_root()
    if systems_root and systems_root.exists():
        rust_systems_root = _ensure_rust_csv_systems_root(systems_root, config)
        os.environ["AICONFIGURATOR_SYSTEMS_PATH"] = str(rust_systems_root)
        if rust_systems_root != systems_root:
            os.environ[RUST_SYSTEMS_SOURCE_ENV] = str(systems_root)
    if "AICONFIGURATOR_MODEL_CONFIGS_PATH" not in os.environ:
        model_configs_root = Path(str(pkg_resources.files("aiconfigurator") / "model_configs"))
        if model_configs_root.exists():
            os.environ["AICONFIGURATOR_MODEL_CONFIGS_PATH"] = str(model_configs_root)


def _configured_systems_source_root() -> Path | None:
    explicit = os.environ.get("AICONFIGURATOR_SYSTEMS_PATH")
    if explicit:
        explicit_path = Path(explicit)
        source_override = os.environ.get(RUST_SYSTEMS_SOURCE_ENV)
        if source_override and _is_rust_overlay_root(explicit_path):
            return Path(source_override)
        return explicit_path
    source_override = os.environ.get(RUST_SYSTEMS_SOURCE_ENV)
    if source_override:
        return Path(source_override)
    return _python_sdk_systems_root() or Path(str(pkg_resources.files("aiconfigurator") / "systems"))


def _ensure_rust_csv_systems_root(systems_root: Path, config: dict[str, Any] | None) -> Path:
    required_paths = _rust_perf_paths_for_config(systems_root, config)
    if not required_paths:
        return systems_root

    needs_overlay = any(
        not (systems_root / path).is_file() and (systems_root / path).with_suffix(".parquet").is_file()
        for path in required_paths
    )
    if not needs_overlay:
        return systems_root

    source_key = hashlib.sha256(str(systems_root.resolve()).encode("utf-8")).hexdigest()[:16]
    overlay_root = _rust_overlay_base() / source_key
    _copy_system_file_for_rust_overlay(systems_root, overlay_root, config)
    for path in required_paths:
        _materialize_rust_perf_csv(systems_root, overlay_root, path)
    return overlay_root


def _rust_overlay_base() -> Path:
    return Path(tempfile.gettempdir()) / "aiconfigurator-rust-perf-csv"


def _is_rust_overlay_root(path: Path) -> bool:
    try:
        path.resolve().relative_to(_rust_overlay_base().resolve())
        return True
    except ValueError:
        return False


def _rust_perf_paths_for_config(systems_root: Path, config: dict[str, Any] | None) -> list[Path]:
    if not config:
        return []
    system_name = str(config.get("system_name") or "").strip()
    backend = str(config.get("backend") or "").strip()
    if not system_name or not backend:
        return []

    system_file = systems_root / f"{system_name}.yaml"
    if not system_file.is_file():
        return []

    try:
        import yaml

        system_spec = yaml.safe_load(system_file.read_text()) or {}
    except Exception:
        return []

    data_dir = system_spec.get("data_dir")
    if not data_dir:
        return []

    backend_root = systems_root / data_dir / backend
    backend_version = config.get("backend_version")
    if backend_version:
        version_dirs = [backend_root / str(backend_version)]
    elif backend_root.is_dir():
        version_dirs = sorted(path for path in backend_root.iterdir() if path.is_dir())
    else:
        version_dirs = []

    paths: list[Path] = []
    for version_dir in version_dirs:
        version_rel = version_dir.relative_to(systems_root)
        paths.extend(version_rel / name for name in _RUST_VERSION_PERF_FILES)

    nccl_version = (system_spec.get("misc") or {}).get("nccl_version")
    if nccl_version:
        paths.append(Path(data_dir) / "nccl" / str(nccl_version) / "nccl_perf.txt")
    return paths


def _copy_system_file_for_rust_overlay(systems_root: Path, overlay_root: Path, config: dict[str, Any] | None) -> None:
    if not config:
        return
    system_name = str(config.get("system_name") or "").strip()
    if not system_name:
        return
    source = systems_root / f"{system_name}.yaml"
    target = overlay_root / f"{system_name}.yaml"
    if source.is_file():
        _copy_if_stale(source, target)


def _materialize_rust_perf_csv(systems_root: Path, overlay_root: Path, relative_path: Path) -> None:
    source_txt = systems_root / relative_path
    source_parquet = source_txt.with_suffix(".parquet")
    target_txt = overlay_root / relative_path
    if source_txt.is_file():
        _copy_if_stale(source_txt, target_txt)
        return
    if not source_parquet.is_file():
        target_txt.parent.mkdir(parents=True, exist_ok=True)
        return
    if target_txt.is_file() and target_txt.stat().st_mtime_ns >= source_parquet.stat().st_mtime_ns:
        return

    try:
        import pyarrow.csv as pc
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RustCoreUnavailableError(
            "Rust engine-step estimator currently consumes CSV perf files. "
            "Materializing parquet perf data for Rust requires pyarrow."
        ) from exc

    target_txt.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_txt.with_name(f".{target_txt.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        pc.write_csv(pq.read_table(source_parquet), tmp_path)
        os.replace(tmp_path, target_txt)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _copy_if_stale(source: Path, target: Path) -> None:
    if target.is_file() and target.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


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

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from typing import TYPE_CHECKING, ClassVar

from aiconfigurator.sdk import interpolation
from aiconfigurator.sdk.operations.base import Operation, _read_filtered_rows
from aiconfigurator.sdk.performance_result import PerformanceResult

if TYPE_CHECKING:
    from aiconfigurator.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)
_MODE_INDEX_KEY = "__mode_index__"
_MAX_NUM_BATCHED_INDEX_KEY = "__max_num_batched_tokens_index__"
_MAX_NUM_BATCHED_MODE_INDEX_KEY = "__max_num_batched_tokens_mode_index__"
_PARALLEL_INDEX_KEY = "__parallel_index__"
_PARALLEL_MODE_INDEX_KEY = "__parallel_mode_index__"
_MAX_NUM_BATCHED_PARALLEL_INDEX_KEY = "__max_num_batched_tokens_parallel_index__"
_MAX_NUM_BATCHED_PARALLEL_MODE_INDEX_KEY = "__max_num_batched_tokens_parallel_mode_index__"
_ALLOW_PHYSICAL_GPUS_ENV = "AIC_LAYERWISE_ALLOW_PHYSICAL_GPUS"


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _parse_optional_float(value) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _validate_layerwise_physical_gpus(value: float | None, layerwise_file) -> None:
    """Reject multi-GPU physical rows from canonical layerwise data."""

    if value is None or value <= 1:
        return
    if os.getenv(_ALLOW_PHYSICAL_GPUS_ENV):
        return
    raise ValueError(
        f"Layerwise rows must be collected with one physical GPU per worker; "
        f"found physical_gpus={value:g} in {layerwise_file}. "
        f"Move physical TP/EP diagnostics out of canonical data or set "
        f"{_ALLOW_PHYSICAL_GPUS_ENV}=1 for diagnostic-only analysis."
    )


def _parse_int(value, default: int = 0) -> int:
    """Parse integer CSV fields that may have been round-tripped as floats."""

    if value in (None, ""):
        return default
    return int(float(value))


def _entry_scale(entry: dict) -> float:
    raw_multiplier = float(entry.get("layer_multiplier", 0.0) or 0.0)
    if raw_multiplier <= 0.0:
        return 1.0
    measured = max(float(entry.get("measured_layer_count", 1.0) or 1.0), 1.0)
    return raw_multiplier / measured


def _entry_component(entry: dict) -> dict:
    """Return the unscaled component row used to build a merged entry."""

    component = {key: value for key, value in entry.items() if key != "components"}
    component.setdefault("includes_moe", False)
    return component


def _entry_components(entry: dict) -> list[dict]:
    """Return raw component rows for an exact layerwise entry."""

    components = entry.get("components")
    if isinstance(components, list):
        return [dict(component) for component in components if isinstance(component, dict)]
    return [_entry_component(entry)]


def _is_scheduler_envelope_entry(entry: dict | None) -> bool:
    """Return whether a row measures a scheduler/worker iteration envelope."""

    if not isinstance(entry, dict):
        return False
    return str(entry.get("latency_source") or "") in {"schedule_to_update", "worker_wall", "fpm_wall"}


def _entry_mode(entry: dict | None) -> str:
    """Return the MoE weight mode label for a layerwise entry."""

    if not isinstance(entry, dict):
        return ""
    value = entry.get("moe_weight_mode")
    if value in (None, ""):
        return ""
    return str(value)


def _entry_max_num_batched_tokens(entry: dict | None) -> float | None:
    """Return the context scheduler-token budget represented by an entry."""

    if not isinstance(entry, dict):
        return None
    value = entry.get("max_num_batched_tokens")
    if value in (None, ""):
        return None
    return float(value)


def _store_layerwise_entry(index: dict, keys: tuple, entry: dict) -> None:
    """Merge one layerwise entry into a nested index at ``keys``."""

    cursor = index
    for key in keys[:-1]:
        cursor = cursor[key]
    final_key = keys[-1]
    cursor[final_key] = _merge_layerwise_entries(cursor.get(final_key), entry)


def _preferred_default_entry(existing: dict, entry: dict) -> dict:
    """Pick the default row when alternative MoE row modes share one shape."""

    priority = {"dummy": 30, "real_router": 25, "full": 20, "": 10, "noop": 0}
    existing_score = priority.get(_entry_mode(existing), 10)
    entry_score = priority.get(_entry_mode(entry), 10)
    selected = entry if entry_score > existing_score else existing
    result = dict(selected)
    result.setdefault("components", [_entry_component(result)])
    return result


def _merge_layerwise_entries(existing: dict | None, entry: dict) -> dict:
    """Merge representative rows for the same public layerwise query shape."""

    if existing is None:
        result = dict(entry)
        result["components"] = [_entry_component(entry)]
        return result
    existing_chunk = _entry_max_num_batched_tokens(existing)
    entry_chunk = _entry_max_num_batched_tokens(entry)
    if existing_chunk != entry_chunk:
        # Different vLLM scheduler budgets are alternate measurements of the
        # same public shape, not additive layer components. Explicit budget
        # queries use the chunk index built during CSV load; legacy/default
        # queries prefer the row without an explicit budget when available.
        selected = existing if existing_chunk is None else entry if entry_chunk is None else existing
        result = dict(selected)
        result.setdefault("components", [_entry_component(result)])
        return result
    if _entry_mode(existing) and _entry_mode(entry) and _entry_mode(existing) != _entry_mode(entry):
        return _preferred_default_entry(existing, entry)
    if _is_scheduler_envelope_entry(existing) != _is_scheduler_envelope_entry(entry):
        selected = dict(existing if _is_scheduler_envelope_entry(existing) else entry)
        selected.setdefault("components", [_entry_component(selected)])
        return selected

    def _scaled(value: dict, metric: str) -> float:
        return float(value.get(metric, 0.0) or 0.0) * _entry_scale(value)

    def _copy_uniform_numeric_metrics(result: dict, components: list[dict], metrics: tuple[str, ...]) -> None:
        for metric in metrics:
            values = {
                float(component[metric])
                for component in components
                if component.get(metric) not in (None, "")
            }
            if len(values) == 1:
                result[metric] = values.pop()

    components = _entry_components(existing) + [_entry_component(entry)]
    if _is_scheduler_envelope_entry(existing) and _is_scheduler_envelope_entry(entry):
        # Scheduler/worker wall rows already include the common vLLM step
        # envelope. Hybrid representative rows are alternate slices of that
        # envelope, so summing them double-counts scheduler overhead.
        result = {
            "latency": max(_scaled(component, "latency") for component in components),
            "energy": max(_scaled(component, "energy") for component in components),
            "rms_latency": max(_scaled(component, "rms_latency") for component in components),
            "rms_kernel_count": max(int(component.get("rms_kernel_count", 0) or 0) for component in components),
            "includes_moe": any(bool(component.get("includes_moe", False)) for component in components),
            "layer_type": "combined",
            "layer_index": 0.0,
            "measured_layer_count": 1.0,
            "layer_multiplier": 1.0,
            "components": components,
        }
        chunk_sizes = {
            float(component["max_num_batched_tokens"])
            for component in components
            if component.get("max_num_batched_tokens") not in (None, "")
        }
        if len(chunk_sizes) == 1:
            result["max_num_batched_tokens"] = chunk_sizes.pop()
        sources = {
            str(component.get("latency_source") or "")
            for component in components
            if component.get("latency_source") not in (None, "")
        }
        if len(sources) == 1:
            result["latency_source"] = sources.pop()
        _copy_uniform_numeric_metrics(result, components, ("seq_len_q", "seq_len_kv_cache"))
        return result

    result = {
        "latency": sum(_scaled(component, "latency") for component in components),
        "energy": sum(_scaled(component, "energy") for component in components),
        "rms_latency": sum(_scaled(component, "rms_latency") for component in components),
        "rms_kernel_count": sum(int(component.get("rms_kernel_count", 0) or 0) for component in components),
        "includes_moe": any(bool(component.get("includes_moe", False)) for component in components),
        "layer_type": "combined",
        "layer_index": 0.0,
        "measured_layer_count": 1.0,
        "layer_multiplier": 1.0,
        "components": components,
    }
    chunk_sizes = {
        float(component["max_num_batched_tokens"])
        for component in components
        if component.get("max_num_batched_tokens") not in (None, "")
    }
    if len(chunk_sizes) == 1:
        result["max_num_batched_tokens"] = chunk_sizes.pop()
    _copy_uniform_numeric_metrics(result, components, ("seq_len_q", "seq_len_kv_cache"))
    return result


def _cache_key(database: PerfDatabase) -> tuple:
    return (
        database.systems_root,
        database.system,
        database.backend,
        database.version,
        database.enable_shared_layer,
    )


def load_layerwise_data(layerwise_file):
    rows = _read_filtered_rows(layerwise_file)
    if rows is None:
        logger.debug("Layerwise data file %s not found.", layerwise_file)
        return None

    scheduler_ctx_shapes = {
        (
            str(row["model"]).lower(),
            _parse_int(row.get("tp_size") or row.get("attn_tp") or row.get("moe_tp"), 1),
            _parse_int(row.get("seq_len_q") or row.get("new_tokens"), 1),
        )
        for row in rows
        if str(row["phase"]).upper() == "CTX"
        and _parse_int(row.get("seq_len_kv_cache") or row.get("past_kv"), 0) == 0
        and _is_scheduler_envelope_entry(row)
    }
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))))
    mode_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))))
    max_batched_data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))))
    )
    max_batched_mode_data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))))
        )
    )
    parallel_data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
            )
        )
    )
    parallel_mode_data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
                )
            )
        )
    )
    max_batched_parallel_data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
                )
            )
        )
    )
    max_batched_parallel_mode_data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
                    )
                )
            )
        )
    )
    for row in rows:
        model = str(row["model"]).lower()
        phase = str(row["phase"]).upper()
        tp_size = _parse_int(row.get("tp_size") or row.get("attn_tp") or row.get("moe_tp"), 1)
        moe_tp_size = _parse_int(row.get("moe_tp") or row.get("moe_tp_size"), 1)
        ep_size = _parse_int(row.get("ep") or row.get("moe_ep_size"), 1)
        batch_size = _parse_int(row["batch_size"])
        seq_len_q = _parse_int(row.get("seq_len_q") or row.get("new_tokens"), 1)
        seq_len_kv_cache = _parse_int(row.get("seq_len_kv_cache") or row.get("past_kv"), 0)
        if (
            phase == "CTX"
            and seq_len_kv_cache == 0
            and (model, tp_size, seq_len_q) in scheduler_ctx_shapes
            and not _is_scheduler_envelope_entry(row)
        ):
            continue
        if row.get("latency_ms") not in (None, ""):
            latency_ms = float(row["latency_ms"])
        else:
            latency_ms = float(row["total_time_us"]) / 1000.0

        entry = {"latency": latency_ms, "energy": 0.0}
        if row.get("rms_latency_ms") not in (None, ""):
            entry["rms_latency"] = float(row["rms_latency_ms"])
        if row.get("rms_kernel_count") not in (None, ""):
            entry["rms_kernel_count"] = int(float(row["rms_kernel_count"]))
        if row.get("layer_type") not in (None, ""):
            entry["layer_type"] = str(row["layer_type"])
        entry["seq_len_q"] = float(seq_len_q)
        entry["seq_len_kv_cache"] = float(seq_len_kv_cache)
        for metric in ("layer_index", "measured_layer_count", "layer_multiplier"):
            value = _parse_optional_float(row.get(metric))
            if value is not None:
                entry[metric] = value
        value = _parse_optional_float(row.get("physical_gpus"))
        _validate_layerwise_physical_gpus(value, layerwise_file)
        if value is not None:
            entry["physical_gpus"] = value
        value = _parse_optional_float(row.get("max_num_batched_tokens"))
        if value is not None:
            entry["max_num_batched_tokens"] = value
        entry["moe_tp_size"] = float(moe_tp_size)
        entry["moe_ep_size"] = float(ep_size)
        entry["includes_moe"] = _parse_bool(row.get("includes_moe"))
        if row.get("moe_weight_mode") not in (None, ""):
            entry["moe_weight_mode"] = str(row["moe_weight_mode"])
        for metric in ("latency_source", "measurement_mode", "attribution_target", "vllm_config_hash"):
            if row.get(metric) not in (None, ""):
                entry[metric] = str(row[metric])
        if phase == "CTX":
            existing = data[model][phase][tp_size][seq_len_q].get(seq_len_kv_cache)
            data[model][phase][tp_size][seq_len_q][seq_len_kv_cache] = _merge_layerwise_entries(existing, entry)
            _store_layerwise_entry(
                parallel_data,
                (model, phase, tp_size, moe_tp_size, ep_size, seq_len_q, seq_len_kv_cache),
                entry,
            )
            max_num_batched_tokens = entry.get("max_num_batched_tokens")
            if max_num_batched_tokens not in (None, ""):
                max_key = int(float(max_num_batched_tokens))
                existing = max_batched_data[model][phase][tp_size][max_key][seq_len_q].get(seq_len_kv_cache)
                max_batched_data[model][phase][tp_size][max_key][seq_len_q][seq_len_kv_cache] = (
                    _merge_layerwise_entries(existing, entry)
                )
                _store_layerwise_entry(
                    max_batched_parallel_data,
                    (model, phase, tp_size, max_key, moe_tp_size, ep_size, seq_len_q, seq_len_kv_cache),
                    entry,
                )
            if entry.get("moe_weight_mode") not in (None, ""):
                mode = str(entry["moe_weight_mode"])
                existing = mode_data[model][phase][tp_size][mode][seq_len_q].get(seq_len_kv_cache)
                mode_data[model][phase][tp_size][mode][seq_len_q][seq_len_kv_cache] = _merge_layerwise_entries(
                    existing,
                    entry,
                )
                _store_layerwise_entry(
                    parallel_mode_data,
                    (model, phase, tp_size, mode, moe_tp_size, ep_size, seq_len_q, seq_len_kv_cache),
                    entry,
                )
                if max_num_batched_tokens not in (None, ""):
                    max_key = int(float(max_num_batched_tokens))
                    existing = max_batched_mode_data[model][phase][tp_size][mode][max_key][seq_len_q].get(
                        seq_len_kv_cache
                    )
                    max_batched_mode_data[model][phase][tp_size][mode][max_key][seq_len_q][seq_len_kv_cache] = (
                        _merge_layerwise_entries(existing, entry)
                    )
                    _store_layerwise_entry(
                        max_batched_parallel_mode_data,
                        (model, phase, tp_size, mode, max_key, moe_tp_size, ep_size, seq_len_q, seq_len_kv_cache),
                        entry,
                    )
        else:
            existing = data[model][phase][tp_size][batch_size].get(seq_len_kv_cache)
            data[model][phase][tp_size][batch_size][seq_len_kv_cache] = _merge_layerwise_entries(existing, entry)
            _store_layerwise_entry(
                parallel_data,
                (model, phase, tp_size, moe_tp_size, ep_size, batch_size, seq_len_kv_cache),
                entry,
            )
            if entry.get("moe_weight_mode") not in (None, ""):
                mode = str(entry["moe_weight_mode"])
                existing = mode_data[model][phase][tp_size][mode][batch_size].get(seq_len_kv_cache)
                mode_data[model][phase][tp_size][mode][batch_size][seq_len_kv_cache] = _merge_layerwise_entries(
                    existing,
                    entry,
                )
                _store_layerwise_entry(
                    parallel_mode_data,
                    (model, phase, tp_size, mode, moe_tp_size, ep_size, batch_size, seq_len_kv_cache),
                    entry,
                )

    if mode_data:
        data[_MODE_INDEX_KEY] = mode_data
    if max_batched_data:
        data[_MAX_NUM_BATCHED_INDEX_KEY] = max_batched_data
    if max_batched_mode_data:
        data[_MAX_NUM_BATCHED_MODE_INDEX_KEY] = max_batched_mode_data
    if parallel_data:
        data[_PARALLEL_INDEX_KEY] = parallel_data
    if parallel_mode_data:
        data[_PARALLEL_MODE_INDEX_KEY] = parallel_mode_data
    if max_batched_parallel_data:
        data[_MAX_NUM_BATCHED_PARALLEL_INDEX_KEY] = max_batched_parallel_data
    if max_batched_parallel_mode_data:
        data[_MAX_NUM_BATCHED_PARALLEL_MODE_INDEX_KEY] = max_batched_parallel_mode_data
    return data


def _interpolate_metric_2d(x: int, y: int, data: dict, metric: str, extracted_metrics_cache: dict) -> float:
    metric_data = {}
    has_metric = False
    for x_key, y_data in data.items():
        metric_data[x_key] = {}
        for y_key, value in y_data.items():
            if isinstance(value, dict) and metric in value:
                has_metric = True
                metric_data[x_key][y_key] = float(value[metric])
            else:
                metric_data[x_key][y_key] = 0.0
    if not has_metric:
        return 0.0
    return float(interpolation.interp_2d_linear(x, y, metric_data, extracted_metrics_cache)["latency"])


def _uniform_bool_metric(data: dict, metric: str, default: bool = False) -> bool:
    values: set[bool] = set()

    def _walk(value) -> None:
        if isinstance(value, dict) and ("latency" in value or "power" in value):
            values.add(bool(value.get(metric, default)))
            return
        if isinstance(value, dict):
            for child in value.values():
                _walk(child)

    _walk(data)
    if len(values) == 1:
        return values.pop()
    return default


def _uniform_float_metric(data: dict, metric: str, default: float = 0.0) -> float:
    values: set[float] = set()

    def _walk(value) -> None:
        if isinstance(value, dict) and ("latency" in value or "power" in value):
            if metric in value:
                values.add(float(value[metric]))
            return
        if isinstance(value, dict):
            for child in value.values():
                _walk(child)

    _walk(data)
    if len(values) == 1:
        return values.pop()
    return default


def _interpolated_layer_scale_metadata(data: dict) -> tuple[float, float] | None:
    """Return measured/multiplier metadata for an interpolated layerwise row.

    Interpolation returns a latency from the public layerwise table. Uniform
    one-layer surfaces still need their normal representative-layer scale.
    Mixed tables that contain already-merged public rows must not fall back to
    the model layer count, or the latency is scaled twice.
    """

    scales: set[float] = set()
    saw_scale_metadata = False
    saw_missing_scale_metadata = False

    def _walk(value) -> None:
        nonlocal saw_missing_scale_metadata, saw_scale_metadata
        if isinstance(value, dict) and ("latency" in value or "power" in value):
            has_measured_count = value.get("measured_layer_count") not in (None, "")
            has_multiplier = value.get("layer_multiplier") not in (None, "")
            if has_measured_count or has_multiplier:
                saw_scale_metadata = True
                scales.add(float(_entry_scale(value)))
            else:
                saw_missing_scale_metadata = True
            return
        if isinstance(value, dict):
            for child in value.values():
                _walk(child)

    _walk(data)
    if not saw_scale_metadata or saw_missing_scale_metadata:
        return None
    if len(scales) == 1:
        return 1.0, scales.pop()
    return 1.0, 1.0


def _uniform_str_metric(data: dict, metric: str, default: str = "") -> str:
    values: set[str] = set()

    def _walk(value) -> None:
        if isinstance(value, dict) and ("latency" in value or "power" in value):
            if metric in value:
                values.add(str(value[metric]))
            return
        if isinstance(value, dict):
            for child in value.values():
                _walk(child)

    _walk(data)
    if len(values) == 1:
        return values.pop()
    return default


def _representative_components(data: dict) -> list[dict]:
    """Return component metadata from the first leaf entry in a layerwise grid."""

    def _walk(value) -> list[dict] | None:
        if isinstance(value, dict) and ("latency" in value or "power" in value):
            return _entry_components(value)
        if isinstance(value, dict):
            for child in value.values():
                found = _walk(child)
                if found is not None:
                    return found
        return None

    return _walk(data) or []


class Layerwise(Operation):
    _data_cache: ClassVar[dict] = {}

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        from aiconfigurator.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            data_dir = os.path.join(system_data_root, database.backend, database.version)
            primary_path = os.path.join(data_dir, PerfDataFilename.layerwise.value)
            sources = database._build_op_sources(PerfDataFilename.layerwise, primary_path, system_data_root)
            cls._data_cache[key] = LoadedOpData(load_layerwise_data(sources), PerfDataFilename.layerwise, primary_path)
            cls._record_load()

        if "_layerwise_data" not in database.__dict__:
            database._layerwise_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    @classmethod
    def _query_layerwise_detail_table(
        cls,
        database: PerfDatabase,
        model: str,
        phase: str,
        tp_size: int,
        batch_size: int,
        seq_len: int,
        seq_len_kv_cache: int = 0,
        moe_weight_mode: str | None = None,
        max_num_batched_tokens: int | None = None,
        moe_tp_size: int | None = None,
        moe_ep_size: int | None = None,
    ) -> dict[str, float]:
        from aiconfigurator.sdk.perf_database import PerfDataNotAvailableError

        model = model.lower()
        phase = phase.upper()
        cls.load_data(database)
        data_wrapper = database._layerwise_data
        if data_wrapper is None or not data_wrapper.loaded:
            raise PerfDataNotAvailableError("Layerwise data not available for this system/backend/version")

        data = data_wrapper.data
        mode = str(moe_weight_mode or "")
        parallel_requested = moe_tp_size is not None or moe_ep_size is not None
        parallel_fallback_ep_size: int | None = None
        if parallel_requested:
            query_moe_tp = int(moe_tp_size or 1)
            query_ep = int(moe_ep_size or 1)

            def _select_parallel_family(family: dict | None) -> tuple[dict | None, int | None]:
                if not family:
                    return None, None
                if query_ep in family and family[query_ep]:
                    return family[query_ep], None
                candidates = [
                    (int(candidate_ep), candidate_data)
                    for candidate_ep, candidate_data in family.items()
                    if int(candidate_ep) != query_ep and candidate_data
                ]
                if not candidates:
                    return None, None
                candidates.sort(
                    key=lambda item: (
                        abs(math.log2(max(float(item[0]), 1.0) / max(float(query_ep), 1.0))),
                        item[0],
                    )
                )
                return candidates[0][1], candidates[0][0]

            try:
                if phase == "CTX" and max_num_batched_tokens is not None and mode:
                    max_key = int(max_num_batched_tokens)
                    parallel_family = data[_MAX_NUM_BATCHED_PARALLEL_MODE_INDEX_KEY][model][phase][tp_size][mode][
                        max_key
                    ][query_moe_tp]
                elif phase == "CTX" and max_num_batched_tokens is not None:
                    max_key = int(max_num_batched_tokens)
                    parallel_family = data[_MAX_NUM_BATCHED_PARALLEL_INDEX_KEY][model][phase][tp_size][max_key][
                        query_moe_tp
                    ]
                elif mode:
                    parallel_family = data[_PARALLEL_MODE_INDEX_KEY][model][phase][tp_size][mode][query_moe_tp]
                else:
                    parallel_family = data[_PARALLEL_INDEX_KEY][model][phase][tp_size][query_moe_tp]
                tp_data, parallel_fallback_ep_size = _select_parallel_family(parallel_family)
            except KeyError:
                tp_data = None
        else:
            tp_data = None
        if tp_data is not None:
            pass
        elif phase == "CTX" and max_num_batched_tokens is not None:
            max_key = int(max_num_batched_tokens)
            if mode:
                max_mode_index = data.get(_MAX_NUM_BATCHED_MODE_INDEX_KEY, {})
                try:
                    data = max_mode_index[model][phase][tp_size][mode][max_key]
                    tp_data = data
                except KeyError as exc:
                    raise PerfDataNotAvailableError(
                        f"Layerwise data for moe_weight_mode={mode!r}, max_num_batched_tokens={max_key} "
                        f"not found for {model}/{phase}/tp{tp_size}"
                    ) from exc
            else:
                max_index = data.get(_MAX_NUM_BATCHED_INDEX_KEY, {})
                try:
                    data = max_index[model][phase][tp_size][max_key]
                    tp_data = data
                except KeyError as exc:
                    raise PerfDataNotAvailableError(
                        f"Layerwise data for max_num_batched_tokens={max_key} not found for "
                        f"{model}/{phase}/tp{tp_size}"
                    ) from exc
        elif mode:
            mode_index = data.get(_MODE_INDEX_KEY, {})
            try:
                data = mode_index[model][phase][tp_size][mode]
                tp_data = data
            except KeyError as exc:
                raise PerfDataNotAvailableError(
                    f"Layerwise data for moe_weight_mode={mode!r} not found for {model}/{phase}/tp{tp_size}"
                ) from exc
        else:
            if model not in data:
                raise PerfDataNotAvailableError(f"Model {model!r} not found in layerwise data")
            if phase not in data[model]:
                raise PerfDataNotAvailableError(f"Phase {phase!r} not found in layerwise data for {model}")
            if tp_size not in data[model][phase]:
                raise PerfDataNotAvailableError(f"tp_size={tp_size} not found in layerwise data for {model}/{phase}")

            tp_data = data[model][phase][tp_size]
        if phase == "CTX":
            if seq_len in tp_data and seq_len_kv_cache in tp_data[seq_len]:
                result = tp_data[seq_len][seq_len_kv_cache]
            elif len(tp_data) < 2:
                raise PerfDataNotAvailableError(f"Not enough CTX layerwise data points for tp_size={tp_size}")
            else:
                result = interpolation.interp_2d_linear(
                    seq_len,
                    seq_len_kv_cache,
                    tp_data,
                    database._extracted_metrics_cache,
                )
                result["rms_latency"] = _interpolate_metric_2d(
                    seq_len,
                    seq_len_kv_cache,
                    tp_data,
                    "rms_latency",
                    database._extracted_metrics_cache,
                )
                result["includes_moe"] = _uniform_bool_metric(tp_data, "includes_moe")
                result["layer_type"] = _uniform_str_metric(tp_data, "layer_type")
                result["layer_index"] = _uniform_float_metric(tp_data, "layer_index")
                scale_metadata = _interpolated_layer_scale_metadata(tp_data)
                if scale_metadata is not None:
                    result["measured_layer_count"], result["layer_multiplier"] = scale_metadata
                elif seq_len_kv_cache == 0:
                    result["measured_layer_count"] = _uniform_float_metric(tp_data, "measured_layer_count", 1.0)
                    result["layer_multiplier"] = _uniform_float_metric(tp_data, "layer_multiplier")
                result["max_num_batched_tokens"] = _uniform_float_metric(tp_data, "max_num_batched_tokens")
                result["physical_gpus"] = _uniform_float_metric(tp_data, "physical_gpus")
                result["latency_source"] = _uniform_str_metric(tp_data, "latency_source")
                result["components"] = _representative_components(tp_data)
        elif batch_size in tp_data and seq_len in tp_data[batch_size]:
            result = tp_data[batch_size][seq_len]
        else:
            result = interpolation.interp_2d_linear(batch_size, seq_len, tp_data, database._extracted_metrics_cache)
            result["rms_latency"] = _interpolate_metric_2d(
                batch_size,
                seq_len,
                tp_data,
                "rms_latency",
                database._extracted_metrics_cache,
            )
            result["includes_moe"] = _uniform_bool_metric(tp_data, "includes_moe")
            result["layer_type"] = _uniform_str_metric(tp_data, "layer_type")
            result["layer_index"] = _uniform_float_metric(tp_data, "layer_index")
            scale_metadata = _interpolated_layer_scale_metadata(tp_data)
            if scale_metadata is not None:
                result["measured_layer_count"], result["layer_multiplier"] = scale_metadata
            else:
                result["measured_layer_count"] = _uniform_float_metric(tp_data, "measured_layer_count", 1.0)
                result["layer_multiplier"] = _uniform_float_metric(tp_data, "layer_multiplier")
            result["max_num_batched_tokens"] = _uniform_float_metric(tp_data, "max_num_batched_tokens")
            result["physical_gpus"] = _uniform_float_metric(tp_data, "physical_gpus")
            result["latency_source"] = _uniform_str_metric(tp_data, "latency_source")
            result["components"] = _representative_components(tp_data)

        if not isinstance(result, dict):
            result = {"latency": float(result), "energy": 0.0}
        out = {
            "latency": float(result["latency"]),
            "energy": float(result.get("energy", 0.0)),
            "rms_latency": float(result.get("rms_latency", 0.0)),
            "rms_kernel_count": float(result.get("rms_kernel_count", 0.0)),
            "includes_moe": bool(result.get("includes_moe", False)),
            "query_seq_len_q": float(seq_len),
            "query_seq_len_kv_cache": float(seq_len_kv_cache),
        }
        if result.get("layer_type") not in (None, ""):
            out["layer_type"] = str(result["layer_type"])
        for metric in (
            "layer_index",
            "measured_layer_count",
            "layer_multiplier",
            "max_num_batched_tokens",
            "seq_len_q",
            "seq_len_kv_cache",
            "query_seq_len_q",
            "query_seq_len_kv_cache",
        ):
            if result.get(metric) not in (None, ""):
                out[metric] = float(result[metric])
        if result.get("physical_gpus") not in (None, ""):
            out["physical_gpus"] = float(result["physical_gpus"])
        for metric in ("latency_source", "measurement_mode", "attribution_target", "vllm_config_hash"):
            if result.get(metric) not in (None, ""):
                out[metric] = str(result[metric])
        if result.get("moe_weight_mode") not in (None, ""):
            out["moe_weight_mode"] = str(result["moe_weight_mode"])
        if parallel_fallback_ep_size is not None:
            out["parallel_fallback_moe_ep_size"] = float(parallel_fallback_ep_size)
            out["requested_moe_ep_size"] = float(moe_ep_size or 1)
        if isinstance(result.get("components"), list):
            out["components"] = [dict(component) for component in result["components"] if isinstance(component, dict)]
        return out

    @classmethod
    def _query_layerwise_table(
        cls,
        database: PerfDatabase,
        model: str,
        phase: str,
        tp_size: int,
        batch_size: int,
        seq_len: int,
        seq_len_kv_cache: int = 0,
    ) -> PerformanceResult:
        result = cls._query_layerwise_detail_table(
            database,
            model,
            phase,
            tp_size,
            batch_size,
            seq_len,
            seq_len_kv_cache,
            max_num_batched_tokens=None,
        )

        return PerformanceResult(result["latency"], energy=result["energy"], source="silicon")

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        return self._query_layerwise_table(
            database,
            kwargs["model"],
            kwargs["phase"],
            kwargs["tp_size"],
            kwargs["batch_size"],
            kwargs["seq_len"],
            kwargs.get("seq_len_kv_cache", 0),
        )

    def get_weights(self, **kwargs):
        return 0.0

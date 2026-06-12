# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
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
    if _entry_mode(existing) and _entry_mode(entry) and _entry_mode(existing) != _entry_mode(entry):
        return _preferred_default_entry(existing, entry)
    if _is_scheduler_envelope_entry(existing) != _is_scheduler_envelope_entry(entry):
        selected = dict(existing if _is_scheduler_envelope_entry(existing) else entry)
        selected.setdefault("components", [_entry_component(selected)])
        return selected

    def _scaled(value: dict, metric: str) -> float:
        return float(value.get(metric, 0.0) or 0.0) * _entry_scale(value)

    components = _entry_components(existing) + [_entry_component(entry)]

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
    for row in rows:
        model = str(row["model"]).lower()
        phase = str(row["phase"]).upper()
        tp_size = _parse_int(row.get("tp_size") or row.get("attn_tp") or row.get("moe_tp"), 1)
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
        for metric in ("layer_index", "measured_layer_count", "layer_multiplier"):
            value = _parse_optional_float(row.get(metric))
            if value is not None:
                entry[metric] = value
        value = _parse_optional_float(row.get("physical_gpus"))
        if value is not None:
            entry["physical_gpus"] = value
        value = _parse_optional_float(row.get("max_num_batched_tokens"))
        if value is not None:
            entry["max_num_batched_tokens"] = value
        entry["includes_moe"] = _parse_bool(row.get("includes_moe"))
        if row.get("moe_weight_mode") not in (None, ""):
            entry["moe_weight_mode"] = str(row["moe_weight_mode"])
        for metric in ("latency_source", "measurement_mode", "attribution_target", "vllm_config_hash"):
            if row.get(metric) not in (None, ""):
                entry[metric] = str(row[metric])
        if phase == "CTX":
            existing = data[model][phase][tp_size][seq_len_q].get(seq_len_kv_cache)
            data[model][phase][tp_size][seq_len_q][seq_len_kv_cache] = _merge_layerwise_entries(existing, entry)
            if entry.get("moe_weight_mode") not in (None, ""):
                mode = str(entry["moe_weight_mode"])
                existing = mode_data[model][phase][tp_size][mode][seq_len_q].get(seq_len_kv_cache)
                mode_data[model][phase][tp_size][mode][seq_len_q][seq_len_kv_cache] = _merge_layerwise_entries(
                    existing,
                    entry,
                )
        else:
            existing = data[model][phase][tp_size][batch_size].get(seq_len_kv_cache)
            data[model][phase][tp_size][batch_size][seq_len_kv_cache] = _merge_layerwise_entries(existing, entry)
            if entry.get("moe_weight_mode") not in (None, ""):
                mode = str(entry["moe_weight_mode"])
                existing = mode_data[model][phase][tp_size][mode][batch_size].get(seq_len_kv_cache)
                mode_data[model][phase][tp_size][mode][batch_size][seq_len_kv_cache] = _merge_layerwise_entries(
                    existing,
                    entry,
                )

    if mode_data:
        data[_MODE_INDEX_KEY] = mode_data
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
        if mode:
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
        }
        if result.get("layer_type") not in (None, ""):
            out["layer_type"] = str(result["layer_type"])
        for metric in ("layer_index", "measured_layer_count", "layer_multiplier", "max_num_batched_tokens"):
            if result.get(metric) not in (None, ""):
                out[metric] = float(result[metric])
        if result.get("physical_gpus") not in (None, ""):
            out["physical_gpus"] = float(result["physical_gpus"])
        for metric in ("latency_source", "measurement_mode", "attribution_target", "vllm_config_hash"):
            if result.get(metric) not in (None, ""):
                out[metric] = str(result[metric])
        if result.get("moe_weight_mode") not in (None, ""):
            out["moe_weight_mode"] = str(result["moe_weight_mode"])
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

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare run_static results from SILICON and EMPIRICAL databases.

The runner deliberately creates a separate model, database lookup, and inference
session for each database mode.  A failure is an observation, not a skipped
sample: every configured (case, point, phase, mode) appears in the output.

See ``empirical_fidelity_matrix.json`` for the matrix schema and an example.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any, Protocol

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_MATRIX = Path(__file__).with_name("empirical_fidelity_matrix.json")
MODES = ("SILICON", "EMPIRICAL")
DEFAULT_PHASES = ("prefill", "decode")
TRANSFER_POLICY_CHOICES = ("off", "conservative", "balanced", "aggressive")
TRANSFER_PROVENANCE_TAGS = frozenset({"xshape", "xquant", "xprofile", "xop"})
PROVENANCE_ORDER = ("silicon", "empirical", "xshape", "xquant", "xprofile", "xop")
PHASE_TO_STATIC_MODE = {"prefill": "static_ctx", "decode": "static_gen", "encoder": "static_ctx"}
POINT_FIELDS = (
    "batch_size",
    "isl",
    "osl",
    "prefix",
    "beam_width",
    "image_height",
    "image_width",
    "num_images_per_request",
    "num_image_tokens",
)

OBSERVATION_FIELDS = (
    "case_id",
    "family",
    "quant",
    "model",
    "system",
    "backend",
    "version",
    "transfer_policy",
    "model_config_json",
    "resolved_model_config_json",
    "point_id",
    "sample_kind",
    *POINT_FIELDS,
    "phase",
    "mode",
    "status",
    "oom",
    "memory_gib",
    "value_ms",
    "error_type",
    "error_message",
    "elapsed_ms",
    "op_latencies_json",
    "op_sources_json",
    "provenance_tags_json",
    "worst_provenance",
)
PAIR_FIELDS = (
    "case_id",
    "family",
    "quant",
    "model",
    "system",
    "backend",
    "version",
    "transfer_policy",
    "model_config_json",
    "resolved_model_config_json",
    "point_id",
    "sample_kind",
    *POINT_FIELDS,
    "phase",
    "pair_status",
    "silicon_status",
    "empirical_status",
    "silicon_oom",
    "empirical_oom",
    "silicon_ms",
    "empirical_ms",
    "signed_error_ms",
    "absolute_error_ms",
    "signed_error_pct",
    "ape_pct",
    "silicon_error",
    "empirical_error",
    "silicon_provenance_tags_json",
    "empirical_provenance_tags_json",
    "silicon_worst_provenance",
    "empirical_worst_provenance",
    "transfer_tagged",
    "transfer_tags_json",
)
SUMMARY_FIELDS = (
    "scope",
    "case_id",
    "family",
    "quant",
    "phase",
    "sample_kind",
    "transfer_policy",
    "attempted",
    "silicon_ooms",
    "empirical_ooms",
    "silicon_successes",
    "empirical_successes",
    "both_successes",
    "comparable",
    "eligible",
    "silicon_coverage_pct",
    "empirical_coverage_pct",
    "empirical_given_silicon_coverage_pct",
    "paired_coverage_pct",
    "eligible_coverage_pct",
    "mean_ape_pct",
    "median_ape_pct",
    "p90_ape_pct",
    "max_ape_pct",
    "wape_pct",
    "signed_bias_pct",
    "mean_signed_error_pct",
    "transfer_tagged_pairs",
    "transfer_excluded_pairs",
)
ATTRIBUTION_FIELDS = (
    "case_id",
    "family",
    "quant",
    "phase",
    "transfer_policy",
    "model_config_json",
    "resolved_model_config_json",
    "worst_rank",
    "point_id",
    "sample_kind",
    *POINT_FIELDS,
    "full_silicon_ms",
    "full_empirical_ms",
    "full_ape_pct",
    "op",
    "silicon_ms",
    "empirical_ms",
    "delta_ms",
    "absolute_delta_ms",
    "absolute_delta_share_pct",
    "op_ape_pct",
    "silicon_latency_share_pct",
    "signed_contribution_pct",
    "cancellation_ratio",
    "silicon_source",
    "empirical_source",
)
OP_SUMMARY_FIELDS = (
    "scope",
    "case_id",
    "family",
    "quant",
    "phase",
    "op",
    "transfer_policy",
    "count",
    "silicon_present_count",
    "empirical_present_count",
    "both_present_count",
    "silicon_missing_count",
    "empirical_missing_count",
    "zero_silicon_count",
    "op_ape_count",
    "mean_op_ape_pct",
    "median_op_ape_pct",
    "p90_op_ape_pct",
    "max_op_ape_pct",
    "wape_pct",
    "signed_bias_pct",
    "mean_silicon_latency_share_pct",
)


@dataclass(frozen=True)
class Measurement:
    """A single successful run_static phase measurement."""

    value_ms: float
    op_latencies: Mapping[str, float]
    op_sources: Mapping[str, str] = dataclass_field(default_factory=dict)
    oom: bool = False
    memory_gib: float | None = None
    resolved_model_config: Mapping[str, Any] = dataclass_field(default_factory=dict)
    provenance_tags: frozenset[str] = dataclass_field(default_factory=frozenset)
    worst_provenance: str = ""


@dataclass
class FidelityReport:
    observations: list[dict[str, Any]]
    pairs: list[dict[str, Any]]
    summary: list[dict[str, Any]]
    op_summary: list[dict[str, Any]]
    attribution: list[dict[str, Any]]


class ModeRunner(Protocol):
    def __call__(self, point: Mapping[str, Any], phase: str) -> Measurement: ...


ModeRunnerFactory = Callable[[Mapping[str, Any], str, str], ModeRunner]
ProgressCallback = Callable[[str], None]


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _canonical_phases(phases: Any) -> list[str]:
    if phases is None:
        phases = list(DEFAULT_PHASES)
    if isinstance(phases, str):
        phases = [phases]
    result = []
    for phase in phases:
        normalized = str(phase).lower()
        aliases = {"static_ctx": "prefill", "static_gen": "decode"}
        normalized = aliases.get(normalized, normalized)
        if normalized not in PHASE_TO_STATIC_MODE:
            raise ValueError(f"Unsupported phase {phase!r}; expected prefill, decode, and/or encoder")
        if normalized not in result:
            result.append(normalized)
    if not result:
        raise ValueError("At least one phase is required")
    return result


def _validate_point(point: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    required = ("batch_size", "isl", "osl")
    missing = [name for name in required if name not in point]
    if missing:
        raise ValueError(f"{source} is missing point fields: {', '.join(missing)}")
    normalized = {
        "batch_size": int(point["batch_size"]),
        "isl": int(point["isl"]),
        "osl": int(point["osl"]),
        "prefix": int(point.get("prefix", 0)),
        "beam_width": int(point.get("beam_width", 1)),
        "image_height": int(point.get("image_height", 0)),
        "image_width": int(point.get("image_width", 0)),
        "num_images_per_request": int(point.get("num_images_per_request", 1)),
        "num_image_tokens": int(point.get("num_image_tokens", 0)),
    }
    if normalized["batch_size"] <= 0 or normalized["isl"] <= 0 or normalized["osl"] <= 0:
        raise ValueError(f"{source} batch_size, isl, and osl must be positive")
    if normalized["prefix"] < 0 or normalized["prefix"] > normalized["isl"]:
        raise ValueError(f"{source} prefix must be between zero and isl")
    if normalized["beam_width"] <= 0:
        raise ValueError(f"{source} beam_width must be positive")
    if normalized["image_height"] < 0 or normalized["image_width"] < 0 or normalized["num_image_tokens"] < 0:
        raise ValueError(f"{source} image dimensions and num_image_tokens must be non-negative")
    if normalized["num_images_per_request"] <= 0:
        raise ValueError(f"{source} num_images_per_request must be positive")
    normalized["phases"] = _canonical_phases(point.get("phases")) if "phases" in point else None
    return normalized


def _sample_integer(rng: random.Random, bounds: Sequence[Any], distribution: str) -> int:
    if len(bounds) != 2:
        raise ValueError(f"Point range must contain [min, max], got {bounds!r}")
    low, high = int(bounds[0]), int(bounds[1])
    if low <= 0 or high < low:
        raise ValueError(f"Invalid positive integer range [{low}, {high}]")
    if low == high:
        return low
    if distribution == "uniform":
        return rng.randint(low, high)
    if distribution == "log_uniform":
        return max(low, min(high, round(math.exp(rng.uniform(math.log(low), math.log(high))))))
    raise ValueError(f"Unsupported point distribution {distribution!r}")


def generate_points(spec: Mapping[str, Any], default_phases: Sequence[str]) -> list[dict[str, Any]]:
    """Generate reproducible integer points while avoiding configured grid anchors."""

    count = int(spec.get("count", 10))
    if count <= 0:
        raise ValueError("point_generation.count must be positive")
    seed = int(spec.get("seed", 0))
    distribution = str(spec.get("distribution", "log_uniform"))
    ranges = spec.get("ranges", {})
    for field in ("batch_size", "isl", "osl"):
        if field not in ranges:
            raise ValueError(f"point_generation.ranges.{field} is required")
    avoid = {field: {int(value) for value in values} for field, values in spec.get("avoid", {}).items()}
    constants = {"prefix": 0, "beam_width": 1, **spec.get("constants", {})}
    max_batch_tokens = int(spec.get("max_batch_tokens", 0))
    max_sequence = int(spec.get("max_sequence", 0))
    if max_batch_tokens < 0 or max_sequence < 0:
        raise ValueError("point_generation limits must be non-negative")
    phases = _canonical_phases(spec.get("phases", default_phases))
    rng = random.Random(seed)
    points: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    max_attempts = max(1000, count * 100)
    for _ in range(max_attempts):
        sampled = {field: _sample_integer(rng, ranges[field], distribution) for field in ("batch_size", "isl", "osl")}
        sampled.update(constants)
        point = _validate_point(sampled, source="generated point")
        if max_batch_tokens and point["batch_size"] * (point["isl"] + point["osl"]) > max_batch_tokens:
            continue
        if max_sequence and point["isl"] + point["osl"] > max_sequence:
            continue
        if any(point.get(field) in values for field, values in avoid.items()):
            continue
        key = tuple(point[field] for field in POINT_FIELDS)
        if key in seen:
            continue
        seen.add(key)
        point["point_id"] = f"g{len(points):03d}"
        point["sample_kind"] = str(spec.get("sample_kind", "offgrid"))
        point["phases"] = phases
        points.append(point)
        if len(points) == count:
            return points
    raise ValueError(
        f"Could only generate {len(points)} of {count} unique off-grid points; "
        "widen ranges or relax point_generation.avoid"
    )


def _canonical_transfer_policy(spec: Any) -> str:
    """Validate a global policy and return a stable report label."""
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from aiconfigurator.sdk import common

    resolved = common.resolve_transfer_policy(spec)
    for name in TRANSFER_POLICY_CHOICES:
        if resolved == common.resolve_transfer_policy(name):
            return name
    return ",".join(sorted(kind.value for kind in resolved))


def normalize_matrix(matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Merge defaults, validate cases, and expand their explicit/generated points."""

    defaults = matrix.get("defaults", {})
    if "transfer_policy" in defaults:
        raise ValueError("transfer_policy is a global run option and cannot be set in matrix defaults")
    raw_cases = matrix.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Matrix must contain a non-empty cases list")
    result = []
    case_ids: set[str] = set()
    for case_index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, Mapping):
            raise TypeError(f"cases[{case_index}] must be an object")
        if "transfer_policy" in raw_case:
            raise ValueError("transfer_policy is a global run option and cannot vary by case")
        case = _deep_merge(defaults, raw_case)
        case_id = str(case.get("id", f"case-{case_index:03d}"))
        if case_id in case_ids:
            raise ValueError(f"Duplicate case id {case_id!r}")
        case_ids.add(case_id)
        for field in ("model", "system", "backend", "version"):
            if not case.get(field):
                raise ValueError(f"Case {case_id!r} is missing {field}")
        phases = _canonical_phases(case.get("phases"))
        raw_phase_sample_kinds = case.get("phase_sample_kinds", {})
        if not isinstance(raw_phase_sample_kinds, Mapping):
            raise TypeError(f"Case {case_id!r} phase_sample_kinds must be an object")
        case["phase_sample_kinds"] = {
            _canonical_phases([phase])[0]: str(sample_kind) for phase, sample_kind in raw_phase_sample_kinds.items()
        }
        explicit_points = case.get("points", [])
        if explicit_points and not isinstance(explicit_points, list):
            raise ValueError(f"Case {case_id!r} points must be a list")
        points: list[dict[str, Any]] = []
        point_ids: set[str] = set()
        for point_index, raw_point in enumerate(explicit_points):
            point = _validate_point(raw_point, source=f"case {case_id!r} point {point_index}")
            point_id = str(raw_point.get("id", f"p{point_index:03d}"))
            if point_id in point_ids:
                raise ValueError(f"Case {case_id!r} has duplicate point id {point_id!r}")
            point_ids.add(point_id)
            point["point_id"] = point_id
            point["sample_kind"] = str(raw_point.get("sample_kind", "explicit"))
            point["phases"] = point["phases"] or phases
            points.append(point)
        if case.get("point_generation"):
            generated = generate_points(case["point_generation"], phases)
            for point in generated:
                point_id = point["point_id"]
                while point_id in point_ids:
                    point_id = f"generated-{point_id}"
                point["point_id"] = point_id
                point_ids.add(point_id)
                points.append(point)
        if not points:
            raise ValueError(f"Case {case_id!r} must provide points and/or point_generation")
        case["id"] = case_id
        case["phases"] = phases
        case["points"] = points
        result.append(case)
    return result


def load_matrix(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        matrix = json.load(stream)
    if not isinstance(matrix, dict):
        raise TypeError("Matrix root must be a JSON object")
    return matrix


def _enum_value(enum_type: Any, value: Any) -> Any:
    if value is None or not isinstance(value, str):
        return value
    try:
        return enum_type[value]
    except KeyError as error:
        choices = ", ".join(enum_type.__members__)
        raise ValueError(f"Unknown {enum_type.__name__} {value!r}; expected one of {choices}") from error


def _summary_ops(summary: Any, phase: str, osl: int) -> dict[str, float]:
    if phase == "prefill":
        groups = (("context", summary.get_context_latency_dict()),)
        divisor = 1
    elif phase == "encoder":
        groups = (("encoder", summary.get_encoder_latency_dict()),)
        divisor = 1
    else:
        groups = (("generation", summary.get_generation_latency_dict()),)
        divisor = max(osl - 1, 1)
    result: dict[str, float] = {}
    for group_name, values in groups:
        for op, value in values.items():
            key = str(op)
            if key in result:
                key = f"{group_name}:{key}"
            result[key] = float(value) / divisor
    return result


def _summary_sources(summary: Any, phase: str) -> dict[str, str]:
    if phase == "prefill":
        groups = (("context", summary.get_context_source_dict()),)
    elif phase == "encoder":
        groups = (("encoder", summary.get_encoder_source_dict()),)
    else:
        groups = (("generation", summary.get_generation_source_dict()),)
    result: dict[str, str] = {}
    for group_name, values in groups:
        for op, value in values.items():
            key = str(op)
            if key in result:
                key = f"{group_name}:{key}"
            result[key] = str(value)
    return result


def _summary_frame_row(summary: Any) -> Any:
    frame = summary.get_summary_df()
    if hasattr(frame, "iloc"):
        return frame.iloc[0]
    elif isinstance(frame, Sequence) and not isinstance(frame, str | bytes):
        return frame[0]
    return frame


def _summary_value(summary: Any, phase: str) -> float:
    row = _summary_frame_row(summary)
    # Match the validation contract directly: context_latency for static_ctx,
    # TPOT for static_gen. Per-op sums remain attribution data and must not
    # silently redefine the full-model metric (notably for VL encoders).
    field = {"prefill": "context_latency", "decode": "tpot", "encoder": "encoder_latency"}[phase]
    return float(row[field])


class _SdkModeRunner:
    def __init__(self, case: Mapping[str, Any], mode: str, transfer_policy: str) -> None:
        if str(SRC_ROOT) not in sys.path:
            sys.path.insert(0, str(SRC_ROOT))
        from aiconfigurator.sdk import common, config
        from aiconfigurator.sdk.backends.factory import get_backend
        from aiconfigurator.sdk.inference_session import InferenceSession
        from aiconfigurator.sdk.models import get_model
        from aiconfigurator.sdk.operations import util_empirical
        from aiconfigurator.sdk.perf_database import get_database

        enum_fields = {
            "gemm_quant_mode": common.GEMMQuantMode,
            "moe_quant_mode": common.MoEQuantMode,
            "kvcache_quant_mode": common.KVCacheQuantMode,
            "fmha_quant_mode": common.FMHAQuantMode,
            "comm_quant_mode": common.CommQuantMode,
        }
        model_config_values = dict(case.get("model_config", {}))
        for field, enum_type in enum_fields.items():
            if field in model_config_values:
                model_config_values[field] = _enum_value(enum_type, model_config_values[field])
        database = get_database(
            system=case["system"],
            backend=case["backend"],
            version=case["version"],
            systems_paths=case.get("systems_paths"),
            allow_missing_data=bool(case.get("allow_missing_data", False)),
            database_mode=mode,
        )
        if database is None:
            raise RuntimeError(f"No database for {case['system']}/{case['backend']}/{case['version']} in {mode} mode")
        # Policy must be fixed before get_model/session execution triggers any
        # operation's lazy data load or process-global utilization-grid build.
        database.set_transfer_policy(transfer_policy)
        database.set_default_database_mode(common.DatabaseMode[mode])
        self._case = case
        self._config = config
        self._get_model = get_model
        self._model_config_values = model_config_values
        self._database = database
        self._backend = get_backend(case["backend"])
        self._inference_session = InferenceSession
        self._util_empirical = util_empirical
        self._stride = int(case.get("stride", 32))

    def __call__(self, point: Mapping[str, Any], phase: str) -> Measurement:
        # FallbackOp records a permanent "primary unavailable" bit after a
        # failed lookup. A fresh model/session per observation keeps one random
        # point from changing every point that follows it.
        model_config = self._config.ModelConfig(**self._model_config_values)
        model = self._get_model(self._case["model"], model_config, self._case["backend"])
        session = self._inference_session(model, self._database, self._backend)
        resolved_fields = (
            "tp_size",
            "pp_size",
            "attention_dp_size",
            "attention_cp_size",
            "moe_tp_size",
            "moe_ep_size",
            "gemm_quant_mode",
            "moe_quant_mode",
            "kvcache_quant_mode",
            "fmha_quant_mode",
            "comm_quant_mode",
        )
        resolved_model_config = {}
        for name in resolved_fields:
            value = getattr(model_config, name)
            resolved_model_config[name] = getattr(value, "name", value)
        runtime_config = self._config.RuntimeConfig(
            batch_size=point["batch_size"],
            beam_width=point["beam_width"],
            isl=point["isl"],
            osl=point["osl"],
            prefix=point["prefix"],
            image_height=point["image_height"],
            image_width=point["image_width"],
            num_images_per_request=point["num_images_per_request"],
            num_image_tokens=point["num_image_tokens"],
        )
        with self._util_empirical.capture_provenance() as provenance_tags:
            summary = session.run_static(
                runtime_config=runtime_config,
                mode=PHASE_TO_STATIC_MODE[phase],
                stride=self._stride,
            )
        op_latencies = _summary_ops(summary, phase, point["osl"])
        return Measurement(
            value_ms=_summary_value(summary, phase),
            op_latencies=op_latencies,
            op_sources=_summary_sources(summary, phase),
            oom=bool(summary.check_oom()),
            memory_gib=float(_summary_frame_row(summary).get("memory", 0.0)),
            resolved_model_config=resolved_model_config,
            provenance_tags=frozenset(provenance_tags),
            worst_provenance=self._util_empirical.worst_provenance(provenance_tags),
        )


def default_mode_runner_factory(case: Mapping[str, Any], mode: str, transfer_policy: str) -> ModeRunner:
    return _SdkModeRunner(case, mode, transfer_policy)


def _metadata(case: Mapping[str, Any], point: Mapping[str, Any], phase: str, transfer_policy: str) -> dict[str, Any]:
    sample_kind = case.get("phase_sample_kinds", {}).get(phase, point["sample_kind"])
    return {
        "case_id": case["id"],
        "family": case.get("family", ""),
        "quant": case.get("quant", ""),
        "model": case["model"],
        "system": case["system"],
        "backend": case["backend"],
        "version": case["version"],
        "transfer_policy": transfer_policy,
        "model_config_json": json.dumps(case.get("model_config", {}), sort_keys=True, separators=(",", ":")),
        "resolved_model_config_json": "{}",
        "point_id": point["point_id"],
        "sample_kind": str(sample_kind),
        **{field: point[field] for field in POINT_FIELDS},
        "phase": phase,
    }


def _error_text(observation: Mapping[str, Any]) -> str:
    if observation["status"] == "success":
        return ""
    if observation["status"] == "oom":
        return "OOM"
    return f"{observation['error_type']}: {observation['error_message']}".strip()


def _worst_provenance(tags: Sequence[str]) -> str:
    ranks = {tag: rank for rank, tag in enumerate(PROVENANCE_ORDER)}
    return max(tags, key=lambda tag: ranks.get(tag, 0), default="silicon")


def _observation(
    case: Mapping[str, Any],
    point: Mapping[str, Any],
    phase: str,
    mode: str,
    transfer_policy: str,
    runner: ModeRunner | None,
    build_error: Exception | None,
) -> dict[str, Any]:
    row = {
        **_metadata(case, point, phase, transfer_policy),
        "mode": mode,
        "status": "error",
        "oom": False,
        "memory_gib": None,
        "value_ms": None,
        "error_type": "",
        "error_message": "",
        "elapsed_ms": 0.0,
        "op_latencies_json": "{}",
        "op_sources_json": "{}",
        "provenance_tags_json": "[]",
        "worst_provenance": "silicon",
    }
    if build_error is not None:
        row["error_type"] = type(build_error).__name__
        row["error_message"] = str(build_error)
        return row
    start = time.perf_counter()
    try:
        if runner is None:
            raise RuntimeError("Mode runner was not initialized")
        measurement = runner(point, phase)
        value = float(measurement.value_ms)
        op_latencies = {str(key): float(item) for key, item in measurement.op_latencies.items()}
        op_sources = {str(key): str(item) for key, item in measurement.op_sources.items()}
        provenance_tags = sorted({str(tag) for tag in measurement.provenance_tags})
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Non-finite or negative latency: {value!r}")
        if any(not math.isfinite(item) or item < 0 for item in op_latencies.values()):
            raise ValueError("Per-op latency contains a non-finite or negative value")
        row.update(
            status="oom" if measurement.oom else "success",
            oom=bool(measurement.oom),
            memory_gib=measurement.memory_gib,
            value_ms=value,
            op_latencies_json=json.dumps(op_latencies, sort_keys=True, separators=(",", ":")),
            op_sources_json=json.dumps(op_sources, sort_keys=True, separators=(",", ":")),
            provenance_tags_json=json.dumps(provenance_tags, separators=(",", ":")),
            worst_provenance=measurement.worst_provenance or _worst_provenance(provenance_tags),
            resolved_model_config_json=json.dumps(
                measurement.resolved_model_config,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    except Exception as error:  # Each failure must be retained in the report.
        row["error_type"] = type(error).__name__
        row["error_message"] = str(error)
    finally:
        row["elapsed_ms"] = (time.perf_counter() - start) * 1000
    return row


def build_pairs(observations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index: dict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = {}
    for row in observations:
        key = (row["case_id"], row["point_id"], row["phase"])
        index.setdefault(key, {})[row["mode"]] = row
    pairs = []
    for mode_rows in index.values():
        silicon = mode_rows.get("SILICON")
        empirical = mode_rows.get("EMPIRICAL")
        template = next(
            (row for row in (silicon, empirical) if row is not None and row["status"] in {"success", "oom"}),
            silicon or empirical,
        )
        if template is None:
            continue
        silicon_tags_json = silicon.get("provenance_tags_json", "[]") if silicon else "[]"
        empirical_tags_json = empirical.get("provenance_tags_json", "[]") if empirical else "[]"
        transfer_tags = sorted(
            (set(json.loads(silicon_tags_json)) | set(json.loads(empirical_tags_json))) & TRANSFER_PROVENANCE_TAGS
        )
        pair = {
            **{field: template[field] for field in PAIR_FIELDS if field in template},
            "pair_status": "mode_missing",
            "silicon_status": silicon["status"] if silicon else "missing",
            "empirical_status": empirical["status"] if empirical else "missing",
            "silicon_oom": bool(silicon and silicon.get("oom")),
            "empirical_oom": bool(empirical and empirical.get("oom")),
            "silicon_ms": silicon["value_ms"] if silicon else None,
            "empirical_ms": empirical["value_ms"] if empirical else None,
            "signed_error_ms": None,
            "absolute_error_ms": None,
            "signed_error_pct": None,
            "ape_pct": None,
            "silicon_error": _error_text(silicon) if silicon else "missing observation",
            "empirical_error": _error_text(empirical) if empirical else "missing observation",
            "silicon_provenance_tags_json": silicon_tags_json,
            "empirical_provenance_tags_json": empirical_tags_json,
            "silicon_worst_provenance": silicon.get("worst_provenance", "silicon") if silicon else "missing",
            "empirical_worst_provenance": empirical.get("worst_provenance", "silicon") if empirical else "missing",
            "transfer_tagged": bool(transfer_tags),
            "transfer_tags_json": json.dumps(transfer_tags, separators=(",", ":")),
        }
        if silicon and empirical:
            if silicon["status"] == "oom" or empirical["status"] == "oom":
                pair["pair_status"] = "oom"
            elif silicon["status"] != "success":
                pair["pair_status"] = "silicon_error"
            elif empirical["status"] != "success":
                pair["pair_status"] = (
                    "empirical_gap"
                    if empirical.get("error_type") == "EmpiricalNotImplementedError"
                    else "empirical_error"
                )
            elif silicon["value_ms"] <= 0:
                pair["pair_status"] = "invalid_reference"
            else:
                delta = empirical["value_ms"] - silicon["value_ms"]
                signed_pct = delta / silicon["value_ms"] * 100
                pair.update(
                    pair_status="comparable",
                    signed_error_ms=delta,
                    absolute_error_ms=abs(delta),
                    signed_error_pct=signed_pct,
                    ape_pct=abs(signed_pct),
                )
                if pair["transfer_policy"] == "off" and transfer_tags:
                    pair["pair_status"] = "transfer_excluded"
        pairs.append(pair)
    return pairs


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    # Nearest-rank is deterministic and makes a tail guardrail refer to an
    # actually observed point instead of an invented value between two runs.
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _summary_row(scope: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    attempted = len(rows)
    comparable = [row for row in rows if row["pair_status"] == "comparable"]
    both_successes = sum(row["silicon_status"] == "success" and row["empirical_status"] == "success" for row in rows)
    apes = [float(row["ape_pct"]) for row in comparable]
    refs = [float(row["silicon_ms"]) for row in comparable]
    deltas = [float(row["signed_error_ms"]) for row in comparable]
    ref_sum = sum(refs)
    first = rows[0] if rows else {}
    silicon_successes = sum(row["silicon_status"] == "success" for row in rows)
    empirical_successes = sum(row["empirical_status"] == "success" for row in rows)
    eligible = silicon_successes

    def coverage(count: int) -> float:
        return count / attempted * 100 if attempted else 0.0

    return {
        "scope": scope,
        "case_id": first.get("case_id", "") if "case" in scope else "",
        "family": first.get("family", "") if "case" in scope else "",
        "quant": first.get("quant", "") if "case" in scope else "",
        "phase": first.get("phase", "") if "phase" in scope else "",
        "sample_kind": first.get("sample_kind", "") if "sample_kind" in scope else "",
        "transfer_policy": first.get("transfer_policy", ""),
        "attempted": attempted,
        "silicon_ooms": sum(bool(row.get("silicon_oom")) for row in rows),
        "empirical_ooms": sum(bool(row.get("empirical_oom")) for row in rows),
        "silicon_successes": silicon_successes,
        "empirical_successes": empirical_successes,
        "both_successes": both_successes,
        "comparable": len(comparable),
        "eligible": eligible,
        "silicon_coverage_pct": coverage(silicon_successes),
        "empirical_coverage_pct": coverage(empirical_successes),
        "empirical_given_silicon_coverage_pct": (
            both_successes / silicon_successes * 100 if silicon_successes else 0.0
        ),
        "paired_coverage_pct": coverage(both_successes),
        "eligible_coverage_pct": len(comparable) / eligible * 100 if eligible else 0.0,
        "mean_ape_pct": statistics.fmean(apes) if apes else None,
        "median_ape_pct": statistics.median(apes) if apes else None,
        "p90_ape_pct": _percentile(apes, 0.9),
        "max_ape_pct": max(apes) if apes else None,
        "wape_pct": sum(abs(delta) for delta in deltas) / ref_sum * 100 if ref_sum > 0 else None,
        "signed_bias_pct": sum(deltas) / ref_sum * 100 if ref_sum > 0 else None,
        "mean_signed_error_pct": (
            statistics.fmean(float(row["signed_error_pct"]) for row in comparable) if comparable else None
        ),
        "transfer_tagged_pairs": sum(bool(row.get("transfer_tagged")) for row in rows),
        "transfer_excluded_pairs": sum(row["pair_status"] == "transfer_excluded" for row in rows),
    }


def build_summary(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: list[tuple[str, list[Mapping[str, Any]]]] = [("all", list(pairs))]
    for phase in PHASE_TO_STATIC_MODE:
        rows = [row for row in pairs if row["phase"] == phase]
        if rows:
            groups.append(("phase", rows))
    sample_kinds = list(dict.fromkeys(row["sample_kind"] for row in pairs))
    for sample_kind in sample_kinds:
        kind_rows = [row for row in pairs if row["sample_kind"] == sample_kind]
        groups.append(("sample_kind", kind_rows))
        for phase in PHASE_TO_STATIC_MODE:
            rows = [row for row in kind_rows if row["phase"] == phase]
            if rows:
                groups.append(("phase_sample_kind", rows))
    case_ids = list(dict.fromkeys(row["case_id"] for row in pairs))
    for case_id in case_ids:
        case_rows = [row for row in pairs if row["case_id"] == case_id]
        groups.append(("case", case_rows))
        for phase in PHASE_TO_STATIC_MODE:
            rows = [row for row in case_rows if row["phase"] == phase]
            if rows:
                groups.append(("case_phase", rows))
                for sample_kind in sample_kinds:
                    kind_rows = [row for row in rows if row["sample_kind"] == sample_kind]
                    if kind_rows:
                        groups.append(("case_phase_sample_kind", kind_rows))
    return [_summary_row(scope, rows) for scope, rows in groups]


def _all_comparable_op_rows(
    pairs: Sequence[Mapping[str, Any]], observations: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Expand every comparable full-model pair to the union of its op names.

    Missing ops are retained with an explicit presence flag.  Their latency is
    treated as zero only for additive deltas, so a disappeared empirical op
    contributes a 100% APE when the silicon reference is positive rather than
    vanishing from the aggregate.
    """
    observation_index = {(row["case_id"], row["point_id"], row["phase"], row["mode"]): row for row in observations}
    rows = []
    for pair in pairs:
        if pair["pair_status"] != "comparable":
            continue
        key = (pair["case_id"], pair["point_id"], pair["phase"])
        silicon_observation = observation_index[(*key, "SILICON")]
        empirical_observation = observation_index[(*key, "EMPIRICAL")]
        silicon_ops = json.loads(silicon_observation["op_latencies_json"])
        empirical_ops = json.loads(empirical_observation["op_latencies_json"])
        full_silicon_ms = float(pair["silicon_ms"])
        for op in sorted(set(silicon_ops) | set(empirical_ops)):
            silicon_present = op in silicon_ops
            empirical_present = op in empirical_ops
            silicon_ms = float(silicon_ops.get(op, 0.0))
            empirical_ms = float(empirical_ops.get(op, 0.0))
            delta_ms = empirical_ms - silicon_ms
            rows.append(
                {
                    "case_id": pair["case_id"],
                    "family": pair["family"],
                    "quant": pair["quant"],
                    "phase": pair["phase"],
                    "op": op,
                    "transfer_policy": pair["transfer_policy"],
                    "silicon_present": silicon_present,
                    "empirical_present": empirical_present,
                    "silicon_ms": silicon_ms,
                    "empirical_ms": empirical_ms,
                    "delta_ms": delta_ms,
                    "op_ape_pct": abs(delta_ms) / silicon_ms * 100 if silicon_present and silicon_ms > 0 else None,
                    "silicon_latency_share_pct": (silicon_ms / full_silicon_ms * 100 if full_silicon_ms > 0 else None),
                }
            )
    return rows


def _op_summary_row(scope: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    apes = [float(row["op_ape_pct"]) for row in rows if row["op_ape_pct"] is not None]
    deltas = [float(row["delta_ms"]) for row in rows]
    refs = [float(row["silicon_ms"]) for row in rows]
    shares = [float(row["silicon_latency_share_pct"]) for row in rows if row["silicon_latency_share_pct"] is not None]
    ref_sum = sum(refs)
    return {
        "scope": scope,
        "case_id": first["case_id"] if scope == "case_phase_op" else "",
        "family": first["family"] if scope == "case_phase_op" else "",
        "quant": first["quant"] if scope == "case_phase_op" else "",
        "phase": first["phase"],
        "op": first["op"],
        "transfer_policy": first["transfer_policy"],
        "count": len(rows),
        "silicon_present_count": sum(bool(row["silicon_present"]) for row in rows),
        "empirical_present_count": sum(bool(row["empirical_present"]) for row in rows),
        "both_present_count": sum(bool(row["silicon_present"]) and bool(row["empirical_present"]) for row in rows),
        "silicon_missing_count": sum(not row["silicon_present"] for row in rows),
        "empirical_missing_count": sum(not row["empirical_present"] for row in rows),
        "zero_silicon_count": sum(row["silicon_present"] and float(row["silicon_ms"]) <= 0 for row in rows),
        "op_ape_count": len(apes),
        "mean_op_ape_pct": statistics.fmean(apes) if apes else None,
        "median_op_ape_pct": statistics.median(apes) if apes else None,
        "p90_op_ape_pct": _percentile(apes, 0.9),
        "max_op_ape_pct": max(apes) if apes else None,
        "wape_pct": sum(abs(delta) for delta in deltas) / ref_sum * 100 if ref_sum > 0 else None,
        "signed_bias_pct": sum(deltas) / ref_sum * 100 if ref_sum > 0 else None,
        "mean_silicon_latency_share_pct": statistics.fmean(shares) if shares else None,
    }


def build_op_summary(
    pairs: Sequence[Mapping[str, Any]], observations: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Aggregate all comparable pairs by phase/op and case/phase/op."""
    op_rows = _all_comparable_op_rows(pairs, observations)
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for row in op_rows:
        groups.setdefault(("phase_op", row["phase"], row["op"]), []).append(row)
    for row in op_rows:
        groups.setdefault(("case_phase_op", row["case_id"], row["phase"], row["op"]), []).append(row)
    return [_op_summary_row(key[0], rows) for key, rows in groups.items()]


def build_attribution(
    pairs: Sequence[Mapping[str, Any]], observations: Sequence[Mapping[str, Any]], worst_n: int
) -> list[dict[str, Any]]:
    if worst_n <= 0:
        return []
    observation_index = {(row["case_id"], row["point_id"], row["phase"], row["mode"]): row for row in observations}
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for pair in pairs:
        if pair["pair_status"] == "comparable":
            groups.setdefault((pair["case_id"], pair["phase"]), []).append(pair)
    rows = []
    for group in groups.values():
        worst = sorted(group, key=lambda pair: float(pair["ape_pct"]), reverse=True)[:worst_n]
        for rank, pair in enumerate(worst, start=1):
            key = (pair["case_id"], pair["point_id"], pair["phase"])
            silicon_ops = json.loads(observation_index[(*key, "SILICON")]["op_latencies_json"])
            empirical_ops = json.loads(observation_index[(*key, "EMPIRICAL")]["op_latencies_json"])
            silicon_sources = json.loads(observation_index[(*key, "SILICON")]["op_sources_json"])
            empirical_sources = json.loads(observation_index[(*key, "EMPIRICAL")]["op_sources_json"])
            deltas = {
                op: float(empirical_ops.get(op, 0.0)) - float(silicon_ops.get(op, 0.0))
                for op in set(silicon_ops) | set(empirical_ops)
            }
            abs_total = sum(abs(delta) for delta in deltas.values())
            cancellation_ratio = abs(sum(deltas.values())) / abs_total if abs_total else 0.0
            for op, delta in sorted(deltas.items(), key=lambda item: abs(item[1]), reverse=True):
                silicon_ms = float(silicon_ops.get(op, 0.0))
                empirical_ms = float(empirical_ops.get(op, 0.0))
                full_silicon_ms = float(pair["silicon_ms"])
                rows.append(
                    {
                        **{field: pair[field] for field in ATTRIBUTION_FIELDS if field in pair},
                        "worst_rank": rank,
                        "full_silicon_ms": pair["silicon_ms"],
                        "full_empirical_ms": pair["empirical_ms"],
                        "full_ape_pct": pair["ape_pct"],
                        "op": op,
                        "silicon_ms": silicon_ms,
                        "empirical_ms": empirical_ms,
                        "delta_ms": delta,
                        "absolute_delta_ms": abs(delta),
                        "absolute_delta_share_pct": abs(delta) / abs_total * 100 if abs_total else 0.0,
                        "op_ape_pct": abs(delta) / silicon_ms * 100 if silicon_ms > 0 else None,
                        "silicon_latency_share_pct": (
                            silicon_ms / full_silicon_ms * 100 if full_silicon_ms > 0 else None
                        ),
                        "signed_contribution_pct": delta / full_silicon_ms * 100 if full_silicon_ms > 0 else None,
                        "cancellation_ratio": cancellation_ratio,
                        "silicon_source": silicon_sources.get(op, ""),
                        "empirical_source": empirical_sources.get(op, ""),
                    }
                )
    return rows


def run_matrix(
    matrix: Mapping[str, Any],
    *,
    mode_runner_factory: ModeRunnerFactory = default_mode_runner_factory,
    transfer_policy: Any = "off",
    worst_n: int = 0,
    progress: ProgressCallback | None = None,
) -> FidelityReport:
    transfer_policy_label = _canonical_transfer_policy(transfer_policy)
    cases = normalize_matrix(matrix)
    if progress:
        progress(f"transfer policy: {transfer_policy_label}")
    observations = []
    for case in cases:
        if mode_runner_factory is default_mode_runner_factory:
            # Util grids are process-global and most op cache keys intentionally
            # omit systems_paths. Keep custom roots and case-local transfer data
            # from leaking into a later case with the same stack names.
            if str(SRC_ROOT) not in sys.path:
                sys.path.insert(0, str(SRC_ROOT))
            from aiconfigurator.sdk.operations import util_empirical

            util_empirical.clear_grid_cache()
        runners: dict[str, ModeRunner | None] = {}
        build_errors: dict[str, Exception | None] = {}
        for mode in MODES:
            if progress:
                progress(f"building {case['id']} {mode}")
            try:
                runners[mode] = mode_runner_factory(case, mode, transfer_policy_label)
                build_errors[mode] = None
            except Exception as error:
                runners[mode] = None
                build_errors[mode] = error
        for point in case["points"]:
            for phase in point["phases"]:
                if progress:
                    progress(f"running {case['id']} {point['point_id']} {phase}")
                for mode in MODES:
                    observations.append(
                        _observation(
                            case,
                            point,
                            phase,
                            mode,
                            transfer_policy_label,
                            runners[mode],
                            build_errors[mode],
                        )
                    )
    pairs = build_pairs(observations)
    return FidelityReport(
        observations=observations,
        pairs=pairs,
        summary=build_summary(pairs),
        op_summary=build_op_summary(pairs, observations),
        attribution=build_attribution(pairs, observations, worst_n),
    )


def _json_ready(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _write_dataset(output_dir: Path, name: str, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with (output_dir / f"{name}.json").open("w", encoding="utf-8") as stream:
        json.dump(_json_ready(rows), stream, indent=2, sort_keys=False, allow_nan=False)
        stream.write("\n")
    with (output_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(report: FidelityReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = (
        ("observations", report.observations, OBSERVATION_FIELDS),
        ("pairs", report.pairs, PAIR_FIELDS),
        ("summary", report.summary, SUMMARY_FIELDS),
        ("op_summary", report.op_summary, OP_SUMMARY_FIELDS),
        ("attribution", report.attribution, ATTRIBUTION_FIELDS),
    )
    for name, rows, fields in datasets:
        _write_dataset(output_dir, name, rows, fields)
    with (output_dir / "report.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {name: _json_ready(rows) for name, rows, _ in datasets},
            stream,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")


def unexpected_errors(report: FidelityReport) -> list[Mapping[str, Any]]:
    """Return failures that are not an expected fidelity coverage outcome."""
    expected = {
        "SILICON": {"PerfDataNotAvailableError"},
        "EMPIRICAL": {"EmpiricalNotImplementedError"},
    }
    return [
        row
        for row in report.observations
        if row["status"] == "error" and row["error_type"] not in expected.get(row["mode"], set())
    ]


def threshold_failures(
    summary: Mapping[str, Any],
    *,
    min_eligible_coverage: float | None = None,
    min_eligible_count: int | None = None,
    min_silicon_coverage: float | None = None,
    max_mean_ape: float | None = None,
) -> list[str]:
    failures = []
    if min_eligible_coverage is not None and summary["eligible_coverage_pct"] < min_eligible_coverage:
        failures.append(f"eligible coverage {summary['eligible_coverage_pct']:.3f}% < {min_eligible_coverage:.3f}%")
    if min_eligible_count is not None and summary["eligible"] < min_eligible_count:
        failures.append(f"eligible count {summary['eligible']} < {min_eligible_count}")
    if min_silicon_coverage is not None and summary["silicon_coverage_pct"] < min_silicon_coverage:
        failures.append(f"silicon coverage {summary['silicon_coverage_pct']:.3f}% < {min_silicon_coverage:.3f}%")
    if max_mean_ape is not None:
        if summary["mean_ape_pct"] is None:
            failures.append("mean APE is unavailable because there are no comparable pairs")
        elif summary["mean_ape_pct"] > max_mean_ape:
            failures.append(f"mean APE {summary['mean_ape_pct']:.3f}% > {max_mean_ape:.3f}%")
    return failures


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", nargs="?", type=Path, default=DEFAULT_MATRIX, help="Input JSON case matrix")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("empirical_fidelity_results"),
        help="Directory for CSV and JSON report datasets",
    )
    parser.add_argument(
        "--worst-n",
        type=int,
        default=0,
        help="Emit per-op attribution for the N worst comparable points per case and phase",
    )
    parser.add_argument(
        "--transfer-policy",
        choices=TRANSFER_POLICY_CHOICES,
        default="off",
        help="Global empirical transfer policy; primary fidelity defaults to strict off",
    )
    parser.add_argument("--quiet", action="store_true", help="Do not print per-case progress")
    parser.add_argument(
        "--verbose-library-logs",
        action="store_true",
        help="Show interpolation/database logs that are suppressed by default",
    )
    parser.add_argument(
        "--allow-unexpected-errors",
        action="store_true",
        help="Return success even when a run fails for reasons other than an expected data-coverage miss",
    )
    parser.add_argument(
        "--min-eligible-coverage",
        type=float,
        help="Fail when comparable/silicon-success coverage is below this percentage",
    )
    parser.add_argument(
        "--min-eligible-count",
        type=int,
        help="Fail when fewer than this many non-OOM silicon references are available",
    )
    parser.add_argument(
        "--min-silicon-coverage",
        type=float,
        help="Fail when non-OOM silicon successes are below this percentage of attempts",
    )
    parser.add_argument(
        "--max-mean-ape",
        type=float,
        help="Fail when the overall comparable-pair mean APE exceeds this percentage",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.verbose_library_logs:
        logging.disable(logging.ERROR)
    matrix = load_matrix(args.matrix)
    progress = None if args.quiet else lambda message: print(message, file=sys.stderr, flush=True)
    report = run_matrix(
        matrix,
        transfer_policy=args.transfer_policy,
        worst_n=args.worst_n,
        progress=progress,
    )
    write_report(report, args.output_dir)
    all_summary = report.summary[0]
    print(
        f"wrote {len(report.observations)} observations and {len(report.pairs)} pairs to "
        f"{args.output_dir} (transfer policy={all_summary['transfer_policy']}, "
        f"eligible coverage={all_summary['eligible_coverage_pct']:.1f}%, "
        f"mean APE={all_summary['mean_ape_pct'] if all_summary['mean_ape_pct'] is not None else 'n/a'})"
    )
    failures = []
    errors = unexpected_errors(report)
    if errors and not args.allow_unexpected_errors:
        failures.append(f"{len(errors)} unexpected run error(s)")
        for row in errors[:5]:
            print(
                f"unexpected error: {row['case_id']}/{row['point_id']}/{row['phase']}/{row['mode']}: "
                f"{row['error_type']}: {row['error_message']}",
                file=sys.stderr,
            )
    failures.extend(
        threshold_failures(
            all_summary,
            min_eligible_coverage=args.min_eligible_coverage,
            min_eligible_count=args.min_eligible_count,
            min_silicon_coverage=args.min_silicon_coverage,
            max_mean_ape=args.max_mean_ape,
        )
    )
    if failures:
        print("validation failed: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

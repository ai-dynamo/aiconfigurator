# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict single-sample aggregation and formal FPM database publication."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from collector.framework_manifest import get_collector_runtime

from .planner import FPMCell, FPMCollectionPlan

_ROW_KEY = (
    "cell_id",
    "model_path",
    "system",
    "backend",
    "backend_version",
    "weight_quantization",
    "gemm_quant_mode",
    "moe_quant_mode",
    "fmha_quant_mode",
    "comm_quant_mode",
    "kv_cache_dtype",
    "tp",
    "pp",
    "dp",
    "moe_tp",
    "moe_ep",
    "cp",
    "backend_axis",
    "backend_policy",
    "workload_kind",
    "batch_size",
    "suffix_length",
    "prefix_length",
)


def _point_key(point: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(point["workload_kind"]),
        int(point["batch_size"]),
        int(point["suffix_length"]),
        int(point["prefix_length"]),
    )


def _dotted_get(payload: object, path: str) -> object:
    value = payload
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(path)
        value = value[part]
    return value


def _validate_backend_markers(cell: FPMCell, cell_dir: Path) -> None:
    expected = cell.backend_policy.expected_markers
    if not expected:
        return
    paths = sorted((cell_dir / "raw").glob("**/resolved-config*.json"))
    if not paths:
        raise ValueError(f"backend policy {cell.backend_policy.policy_id} requires resolved-config evidence")
    for path in paths:
        payload = json.loads(path.read_text())
        mismatches = {}
        for marker_path, marker_value in expected.items():
            try:
                actual = _dotted_get(payload, marker_path)
            except KeyError:
                actual = "<missing>"
            if actual != marker_value:
                mismatches[marker_path] = {"actual": actual, "expected": marker_value}
        if mismatches:
            raise ValueError(f"backend marker mismatch in {path}: {mismatches}")


def _validate_fpm(point: tuple[str, int, int, int], fpm: dict[str, Any]) -> tuple[int, float]:
    phase, batch, suffix, prefix = point
    rank = int(fpm["dp_rank"])
    latency = float(fpm["wall_time"])
    if rank < 0 or not math.isfinite(latency) or latency <= 0:
        raise ValueError(f"invalid FPM rank/latency: rank={rank}, wall_time={latency}")
    scheduled = fpm.get("scheduled_requests")
    if not isinstance(scheduled, dict):
        raise TypeError("FPM scheduled_requests must be a mapping")
    expected = (
        {
            "num_prefill_requests": batch,
            "sum_prefill_tokens": batch * suffix,
            "sum_prefill_kv_tokens": batch * prefix,
            "num_decode_requests": 0,
            "sum_decode_kv_tokens": 0,
        }
        if phase == "prefill"
        else {
            "num_prefill_requests": 0,
            "sum_prefill_tokens": 0,
            "sum_prefill_kv_tokens": 0,
            "num_decode_requests": batch,
            "sum_decode_kv_tokens": batch * prefix,
        }
    )
    mismatches = {key: (scheduled.get(key), value) for key, value in expected.items() if scheduled.get(key) != value}
    if mismatches:
        raise ValueError(f"FPM scheduled workload mismatch: {mismatches}")
    return rank, latency


def aggregate_cell(plan: FPMCollectionPlan, cell: FPMCell, cell_dir: Path) -> list[dict[str, Any]]:
    """Validate all rank files and take max-rank latency for the one repeat."""

    _validate_backend_markers(cell, cell_dir)
    paths = sorted((cell_dir / "raw").glob("**/benchmark*.json"))
    if not paths:
        raise ValueError(f"no benchmark result files found for {cell.cell_id}")
    by_point: dict[tuple[str, int, int, int], dict[int, float]] = defaultdict(dict)
    for path in paths:
        payload = json.loads(path.read_text())
        if (
            payload.get("schema_version") != 1
            or payload.get("status") != "complete"
            or payload.get("valid") is not True
        ):
            raise ValueError(f"invalid terminal result envelope: {path}")
        collector = payload.get("collector")
        if not isinstance(collector, dict):
            raise TypeError(f"missing collector envelope: {path}")
        if collector.get("plan_sha256") != plan.sha256 or collector.get("cell_id") != cell.cell_id:
            raise ValueError(f"result identity mismatch: {path}")
        if collector.get("warmup_repeats") != 0 or collector.get("measured_repeats") != 1:
            raise ValueError(f"V1 requires no warmup and exactly one measurement: {path}")
        rows = payload.get("campaign_results")
        if not isinstance(rows, list):
            raise TypeError(f"campaign_results must be a list: {path}")
        for row in rows:
            point = _point_key(row["point"])
            if point[0] != cell.workload_kind:
                raise ValueError(f"mixed workload kind in {path}: {point[0]}")
            warmups = row.get("warmup_fpms")
            measurements = row.get("fpms")
            if not isinstance(warmups, list) or warmups:
                raise ValueError(f"expected no warmup FPMs for {point} in {path}")
            if not isinstance(measurements, list) or len(measurements) != 1:
                raise ValueError(f"expected one measured FPM for {point} in {path}")
            rank, latency = _validate_fpm(point, measurements[0])
            if rank in by_point[point]:
                raise ValueError(f"duplicate dp_rank={rank} for point={point}")
            by_point[point][rank] = latency

    target_count = cell.execution_profile.selected_point_count
    if len(by_point) != target_count:
        raise ValueError(f"measured point count {len(by_point)} != selected count {target_count} for {cell.cell_id}")
    ordered_population = {point.key for point in cell.execution_profile.ordered_points}
    unexpected = sorted(set(by_point) - ordered_population)
    if unexpected:
        raise ValueError(f"measured points are outside the frozen ordered population for {cell.cell_id}: {unexpected}")
    expected_ranks = set(range(cell.topology.dp))
    backend_version = get_collector_runtime(plan.backend).version
    capability = getattr(plan, "capability", None)
    rows = []
    for point, rank_latencies in sorted(by_point.items()):
        if set(rank_latencies) != expected_ranks:
            raise ValueError(
                f"DP rank set mismatch for {point}: actual={sorted(rank_latencies)} expected={sorted(expected_ranks)}"
            )
        phase, batch, suffix, prefix = point
        rows.append(
            {
                "cell_id": cell.cell_id,
                "model_path": plan.model_path,
                "system": plan.system,
                "backend": plan.backend,
                "backend_version": backend_version,
                "weight_quantization": cell.weight_quantization,
                "gemm_quant_mode": cell.gemm_quant_mode or cell.weight_quantization,
                "moe_quant_mode": cell.moe_quant_mode,
                "fmha_quant_mode": cell.fmha_quant_mode,
                "comm_quant_mode": cell.comm_quant_mode,
                "kv_cache_dtype": cell.kv_cache_dtype,
                "parallel_strategy": cell.parallel_strategy,
                "tp": cell.topology.tp,
                "pp": cell.topology.pp,
                "dp": cell.topology.dp,
                "moe_tp": cell.topology.moe_tp,
                "moe_ep": cell.topology.moe_ep,
                "cp": cell.topology.cp,
                "backend_axis": cell.backend_policy.axis,
                "backend_policy": cell.backend_policy.policy_id,
                "workload_kind": phase,
                "batch_size": batch,
                "suffix_length": suffix,
                "prefix_length": prefix,
                "latency_ms": max(rank_latencies.values()) * 1000.0,
                "warmup_repeats": 0,
                "measurement_repeats": 1,
                "measurement_policy": "single_sample_v1",
                "model_support_level": getattr(capability, "support_level", "unknown"),
                "model_template_id": getattr(capability, "template_id", "unknown"),
                "model_template_version": getattr(capability, "template_version", 0),
                "aic_database_version": getattr(capability, "aic_database_version", "unknown"),
                "source_plan_sha256": plan.sha256,
            }
        )
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_formal_database(
    plan: FPMCollectionPlan,
    rows: list[dict[str, Any]],
    *,
    systems_root: Path | None = None,
) -> tuple[Path, Path]:
    """Atomically merge conflict-free rows into the normal AIC data tree."""

    if not rows:
        raise ValueError("refusing to write an empty FPM database")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("writing fpm_forward_perf.parquet requires pyarrow") from error

    version = get_collector_runtime(plan.backend).version
    if systems_root is None:
        systems_root = Path(__file__).resolve().parents[2] / "src" / "aiconfigurator" / "systems" / "data"
    destination = systems_root / plan.system / plan.backend / version
    destination.mkdir(parents=True, exist_ok=True)
    parquet_path = destination / "fpm_forward_perf.parquet"
    metadata_path = destination / "fpm_forward_perf.metadata.json"

    merged = []
    if parquet_path.exists():
        merged.extend(pq.read_table(parquet_path).to_pylist())
    index = {tuple(row[key] for key in _ROW_KEY): row for row in merged}
    if len(index) != len(merged):
        raise ValueError(f"existing FPM database contains duplicate physical keys: {parquet_path}")
    for row in rows:
        key = tuple(row[name] for name in _ROW_KEY)
        existing = index.get(key)
        if existing is not None:
            if existing != row:
                raise ValueError(f"conflicting FPM database row for key={key}")
            continue
        index[key] = row
        merged.append(row)
    merged.sort(key=lambda row: tuple(row[name] for name in _ROW_KEY))

    temporary = parquet_path.with_name(f".{parquet_path.name}.tmp")
    pq.write_table(pa.Table.from_pylist(merged), temporary, compression="zstd")
    os.replace(temporary, parquet_path)
    metadata = {
        "schema_name": "aic_fpm_forward_perf",
        "schema_version": 3,
        "measurement_policy": "single_sample_v1",
        "warmup_repeats": 0,
        "measurement_repeats": 1,
        "row_count": len(merged),
        "parquet_sha256": _sha256(parquet_path),
        "source_plan_sha256": sorted({str(row["source_plan_sha256"]) for row in merged}),
        "aic_revision": plan.aic_revision,
        "model_paths": sorted({str(row["model_path"]) for row in merged}),
        "system": plan.system,
        "backend": plan.backend,
        "backend_version": version,
    }
    temporary_metadata = metadata_path.with_name(f".{metadata_path.name}.tmp")
    temporary_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_metadata, metadata_path)
    return parquet_path, metadata_path

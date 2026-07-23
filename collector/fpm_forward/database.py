# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate Dynamo-native iteration totals and publish the formal FPM database."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .native_artifact import validate_native_collection
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
    "total_prefill_tokens",
    "total_kv_read_tokens",
    "partition_policy",
)
_RUN_IDENTITY_FIELDS = (
    "source_plan_sha256",
    "collector_attempt_id",
    "runtime_run_id",
    "runtime_grid_digest",
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


def aggregate_cell(
    plan: FPMCollectionPlan,
    cell: FPMCell,
    cell_dir: Path,
    *,
    expected_attempt_id: str,
) -> list[dict[str, Any]]:
    """Validate native rank artifacts and take max-rank latency per grid point."""

    if not expected_attempt_id:
        raise ValueError(f"cannot aggregate {cell.cell_id} without an expected Collector attempt identity")
    _validate_backend_markers(cell, cell_dir)
    collection = validate_native_collection(
        cell,
        cell_dir / "raw",
        expected_plan_sha256=plan.sha256,
        expected_attempt_id=expected_attempt_id,
    )
    backend_version = collection.backend_version
    capability = plan.capability
    rows = []
    for measurement in collection.points:
        point = measurement.point
        phase = str(point["point_type"])
        batch = int(point["batch_size"])
        total_prefill = int(point["total_prefill_tokens"])
        total_kv = int(point["total_kv_read_tokens"])
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
                "fmha_resolution": cell.fmha_resolution,
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
                "total_prefill_tokens": total_prefill,
                "total_kv_read_tokens": total_kv,
                "partition_policy": "balanced_v1",
                "latency_ms": max(latency for _rank, latency in measurement.rank_wall_times) * 1000.0,
                "global_warmup_iterations": plan.options.warmup_iterations,
                "warmup_repeats": 0,
                "measurement_repeats": 1,
                "measurement_policy": "dynamo_native_single_sample_v1",
                "model_support_level": capability.support_level,
                "model_template_id": capability.template_id,
                "model_template_version": capability.template_version,
                "aic_database_version": capability.aic_database_version,
                "source_plan_sha256": plan.sha256,
                "collector_attempt_id": collection.collector_attempt_id,
                "runtime_run_id": collection.runtime_run_id,
                "runtime_grid_digest": collection.runtime_grid_digest,
            }
        )
    return rows


def _run_identities_by_cell(rows: list[dict[str, Any]], *, source: str) -> dict[str, tuple[str, ...]]:
    identities: dict[str, tuple[str, ...]] = {}
    for row in rows:
        cell_id = row.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError(f"{source} FPM row has no cell identity")
        values = tuple(row.get(field) for field in _RUN_IDENTITY_FIELDS)
        if not all(isinstance(value, str) and value for value in values):
            invalid = [
                field
                for field, value in zip(_RUN_IDENTITY_FIELDS, values, strict=True)
                if not isinstance(value, str) or not value
            ]
            raise ValueError(f"{source} FPM row for {cell_id} has invalid run identity fields: {invalid}")
        typed_values = tuple(str(value) for value in values)
        existing = identities.setdefault(cell_id, typed_values)
        if existing != typed_values:
            raise ValueError(f"{source} FPM rows mix run identities for cell_id={cell_id!r}")
    return identities


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _curated_systems_root() -> Path:
    return Path(__file__).resolve().parents[2] / "src" / "aiconfigurator" / "systems" / "data"


@contextmanager
def _publication_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _temporary_path(destination: Path) -> Path:
    descriptor, raw_path = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    return Path(raw_path)


def write_formal_database(
    plan: FPMCollectionPlan,
    rows: list[dict[str, Any]],
    *,
    systems_root: Path | None = None,
) -> tuple[Path, Path]:
    """Atomically merge conflict-free native-grid rows into the AIC data tree."""

    if not rows:
        raise ValueError("refusing to write an empty FPM database")
    incoming_identities = _run_identities_by_cell(rows, source="incoming")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("writing fpm_forward_perf.parquet requires pyarrow") from error

    versions = {str(row.get("backend_version") or "") for row in rows}
    if len(versions) != 1 or not next(iter(versions)):
        raise ValueError(f"FPM rows must contain one non-empty runtime backend_version, got {sorted(versions)!r}")
    version = next(iter(versions))
    curated_root = systems_root is None
    if systems_root is None:
        systems_root = _curated_systems_root()
    destination = systems_root / plan.system / plan.backend / version
    if curated_root and not destination.is_dir():
        # The SDK's version discovery treats ANY populated directory under the
        # curated tree as a declared database version, so materializing a new
        # directory that holds only FPM files would make
        # get_latest_database_version return a dataless version and poison
        # every later default-version resolution for this system.
        raise ValueError(
            f"pod-reported backend_version {version!r} has no curated AIC database directory at "
            f"{destination}; publish against a curated version or pass --fpm-database-root to "
            "write into an explicit tree"
        )
    destination.mkdir(parents=True, exist_ok=True)
    parquet_path = destination / "fpm_forward_perf.parquet"
    metadata_path = destination / "fpm_forward_perf.metadata.json"

    lock_path = destination / ".fpm_forward_perf.lock"
    with _publication_lock(lock_path):
        merged = []
        if parquet_path.exists():
            table = pq.read_table(parquet_path)
            required = {
                "total_prefill_tokens",
                "total_kv_read_tokens",
                "partition_policy",
                *_RUN_IDENTITY_FIELDS,
            }
            if not required.issubset(table.column_names):
                raise ValueError(
                    "existing FPM database predates the attempt-bound schema-v5 contract; "
                    f"publish to a clean destination: {parquet_path}"
                )
            merged.extend(table.to_pylist())
        existing_versions = {str(row.get("backend_version") or "") for row in merged}
        if existing_versions and existing_versions != {version}:
            raise ValueError(
                f"existing FPM database runtime version mismatch: actual={sorted(existing_versions)!r} "
                f"expected={version!r}"
            )
        existing_identities = _run_identities_by_cell(merged, source="existing")
        for cell_id, incoming_identity in incoming_identities.items():
            existing_identity = existing_identities.get(cell_id)
            if existing_identity is not None and existing_identity != incoming_identity:
                raise ValueError(
                    f"refusing to mix FPM run identities for cell_id={cell_id!r}: "
                    f"existing={existing_identity!r}, incoming={incoming_identity!r}"
                )
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

        temporary = _temporary_path(parquet_path)
        temporary_metadata = _temporary_path(metadata_path)
        try:
            pq.write_table(pa.Table.from_pylist(merged), temporary, compression="zstd")
            metadata = {
                "schema_name": "aic_fpm_forward_perf",
                "schema_version": 5,
                "coordinate_system": "iteration_totals_balanced_v1",
                "measurement_policy": "dynamo_native_single_sample_v1",
                "warmup_repeats": 0,
                "measurement_repeats": 1,
                "row_count": len(merged),
                "parquet_sha256": _sha256(temporary),
                "source_plan_sha256": sorted({str(row["source_plan_sha256"]) for row in merged}),
                "collector_attempt_ids": sorted({str(row["collector_attempt_id"]) for row in merged}),
                "runtime_run_ids": sorted({str(row["runtime_run_id"]) for row in merged}),
                "runtime_grid_digests": sorted({str(row["runtime_grid_digest"]) for row in merged}),
                "aic_revision": plan.aic_revision,
                "model_paths": sorted({str(row["model_path"]) for row in merged}),
                "system": plan.system,
                "backend": plan.backend,
                "backend_version": version,
            }
            temporary_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, parquet_path)
            # Metadata is the commit record: readers must validate its parquet
            # digest and ignore an unmatched pair after an interrupted writer.
            os.replace(temporary_metadata, metadata_path)
        finally:
            temporary.unlink(missing_ok=True)
            temporary_metadata.unlink(missing_ok=True)
    return parquet_path, metadata_path

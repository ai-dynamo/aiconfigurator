# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict reader for Dynamo PR11509 native self-benchmark rank artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiconfigurator.fpm_contract import FPM_NATIVE_BENCHMARK_RESULT_SCHEMA_VERSION

from .planner import FPMCell

COLLECTOR_PROVENANCE_FILENAME = "collector-provenance.json"


@dataclass(frozen=True, slots=True)
class NativePointMeasurement:
    point: dict[str, Any]
    rank_wall_times: tuple[tuple[int, float], ...]


@dataclass(frozen=True, slots=True)
class NativeCollection:
    points: tuple[NativePointMeasurement, ...]
    rank_timings: tuple[tuple[int, float, float], ...]
    backend_version: str
    collector_attempt_id: str
    runtime_run_id: str
    runtime_grid_digest: str


def _validate_collector_provenance(
    cell: FPMCell,
    raw_root: Path,
    rank_payloads: list[tuple[Path, dict[str, Any]]],
    *,
    expected_plan_sha256: str | None,
    expected_attempt_id: str | None,
) -> tuple[str, str]:
    pod_names = set()
    for path, _payload in rank_payloads:
        relative = path.relative_to(raw_root)
        if len(relative.parts) < 2:
            raise ValueError(f"native rank artifact is not scoped to a collected pod: {path}")
        pod_names.add(relative.parts[0])

    canonical: dict[str, Any] | None = None
    for pod_name in sorted(pod_names):
        path = raw_root / pod_name / COLLECTOR_PROVENANCE_FILENAME
        if not path.is_file():
            raise ValueError(f"native result is missing Collector provenance: {path}")
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise TypeError(f"Collector provenance must be a mapping: {path}")
        if payload.get("schema_name") != "aic_fpm_collector_provenance" or payload.get("schema_version") != 1:
            raise ValueError(f"unsupported Collector provenance schema: {path}")
        if payload.get("cell_id") != cell.cell_id:
            raise ValueError(
                f"Collector provenance cell mismatch: actual={payload.get('cell_id')!r} expected={cell.cell_id!r}"
            )
        plan_sha256 = payload.get("plan_sha256")
        attempt_id = payload.get("attempt_id")
        if not isinstance(plan_sha256, str) or not plan_sha256:
            raise ValueError(f"Collector provenance has no plan identity: {path}")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError(f"Collector provenance has no attempt identity: {path}")
        if expected_plan_sha256 is not None and plan_sha256 != expected_plan_sha256:
            raise ValueError(
                f"Collector provenance plan mismatch: actual={plan_sha256!r} expected={expected_plan_sha256!r}"
            )
        if expected_attempt_id is not None and attempt_id != expected_attempt_id:
            raise ValueError(
                f"Collector provenance attempt mismatch: actual={attempt_id!r} expected={expected_attempt_id!r}"
            )
        runtime = payload.get("runtime")
        if (
            not isinstance(runtime, dict)
            or runtime.get("backend") != "vllm"
            or not isinstance(runtime.get("backend_version"), str)
            or not runtime["backend_version"]
        ):
            raise ValueError(f"Collector provenance has invalid runtime identity: {path}")
        if canonical is None:
            canonical = payload
        elif payload != canonical:
            raise ValueError(f"Collector provenance differs across pods: {path}")

    assert canonical is not None
    return str(canonical["runtime"]["backend_version"]), str(canonical["attempt_id"])


def _expected_scheduled(point: dict[str, Any]) -> dict[str, int]:
    phase = str(point["point_type"])
    batch = int(point["batch_size"])
    total_prefill = int(point["total_prefill_tokens"])
    total_kv = int(point["total_kv_read_tokens"])
    if batch < 1 or min(total_prefill, total_kv) < 0:
        raise ValueError(f"native point has invalid totals: {point}")
    if phase == "prefill":
        if total_prefill < 1:
            raise ValueError(f"native prefill point has no scheduled tokens: {point}")
        return {
            "num_prefill_requests": batch,
            "sum_prefill_tokens": total_prefill,
            "sum_prefill_kv_tokens": total_kv,
            "num_decode_requests": 0,
            "sum_decode_kv_tokens": 0,
        }
    if phase == "decode":
        if total_prefill != 0:
            raise ValueError(f"native decode point carries prefill tokens: {point}")
        return {
            "num_prefill_requests": 0,
            "sum_prefill_tokens": 0,
            "sum_prefill_kv_tokens": 0,
            "num_decode_requests": batch,
            "sum_decode_kv_tokens": total_kv,
        }
    raise ValueError(f"unknown native point type: {phase!r}")


def _validate_fpm(point: dict[str, Any], fpm: object, *, rank: int) -> float:
    if not isinstance(fpm, dict):
        raise TypeError("native FPM sample must be a mapping")
    benchmark_id = int(point["benchmark_id"])
    if int(fpm.get("counter_id", -1)) != benchmark_id:
        raise ValueError(
            f"native FPM counter mismatch: counter_id={fpm.get('counter_id')!r}, benchmark_id={benchmark_id}"
        )
    if int(fpm.get("dp_rank", -1)) != rank:
        raise ValueError(f"native FPM rank mismatch: fpm={fpm.get('dp_rank')!r}, artifact={rank}")
    wall_time = float(fpm.get("wall_time", 0.0))
    if not math.isfinite(wall_time) or wall_time <= 0:
        raise ValueError(f"native FPM has invalid wall_time={wall_time}")
    scheduled = fpm.get("scheduled_requests")
    if not isinstance(scheduled, dict):
        raise TypeError("native FPM scheduled_requests must be a mapping")
    expected = _expected_scheduled(point)
    mismatches = {key: (scheduled.get(key), value) for key, value in expected.items() if scheduled.get(key) != value}
    if mismatches:
        raise ValueError(f"native FPM workload mismatch: {mismatches}")
    return wall_time


def _rank_artifacts(raw_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    artifacts = []
    for path in sorted(raw_root.glob("**/benchmark*.json")):
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise TypeError(f"native benchmark artifact is not a mapping: {path}")
        if payload.get("artifact_type") == "merged" or path.stem.endswith("_merged"):
            continue
        artifacts.append((path, payload))
    return artifacts


def validate_native_collection(
    cell: FPMCell,
    raw_root: Path,
    *,
    expected_plan_sha256: str | None = None,
    expected_attempt_id: str | None = None,
) -> NativeCollection:
    """Validate a complete native rank set and return synchronized measurements."""

    rank_payloads = _rank_artifacts(raw_root)
    if not rank_payloads:
        raise ValueError(f"integrated runtime emitted no benchmark results for {cell.cell_id}")
    backend_version, collector_attempt_id = _validate_collector_provenance(
        cell,
        raw_root,
        rank_payloads,
        expected_plan_sha256=expected_plan_sha256,
        expected_attempt_id=expected_attempt_id,
    )
    expected_ranks = list(range(cell.topology.dp))
    seen_ranks: set[int] = set()
    canonical_points: list[dict[str, Any]] | None = None
    canonical_groups: list[dict[str, Any]] | None = None
    run_identity: tuple[str, str] | None = None
    local_fpms: dict[tuple[int, int], dict[str, Any]] = {}
    rank_timings: list[tuple[int, float, float]] = []

    for path, payload in rank_payloads:
        if (
            payload.get("schema_version") != FPM_NATIVE_BENCHMARK_RESULT_SCHEMA_VERSION
            or payload.get("artifact_type") != "rank"
        ):
            raise ValueError(f"result is not a PR11509 native rank artifact: {path}")
        if (
            payload.get("status") != "complete"
            or payload.get("valid") is not True
            or payload.get("usable") is not True
            or payload.get("timing_valid") is not True
            or payload.get("stop_reason") is not None
            or payload.get("error") is not None
        ):
            raise ValueError(f"invalid native terminal result envelope: {path}")
        if payload.get("skipped_points") != [] or payload.get("missing_phases") != []:
            raise ValueError(f"native result skipped work: {path}")

        config = payload.get("config")
        if not isinstance(config, dict) or config.get("mode") != cell.workload_kind:
            raise ValueError(f"native benchmark mode mismatch: {path}")
        coverage = payload.get("coverage")
        rows = payload.get("results")
        groups = payload.get("iteration_groups")
        if not isinstance(coverage, dict) or not isinstance(rows, list) or not isinstance(groups, list):
            raise TypeError(f"native result is missing coverage/results/iteration_groups: {path}")
        expected = coverage.get("expected_points")
        if (
            not isinstance(expected, int)
            or expected < 1
            or coverage.get("completed_points") != expected
            or coverage.get("skipped_points") != 0
            or len(rows) != expected
            or len(groups) != expected
        ):
            raise ValueError(f"native benchmark coverage is incomplete: {path}")

        dp = payload.get("dp")
        if not isinstance(dp, dict) or dp.get("size") != cell.topology.dp or not isinstance(dp.get("rank"), int):
            raise ValueError(f"native DP metadata mismatch: {path}")
        rank = int(dp["rank"])
        if rank in seen_ranks:
            raise ValueError(f"duplicate native dp_rank={rank}: {path}")
        seen_ranks.add(rank)

        identity = (payload.get("run_id"), payload.get("grid_digest"))
        if not all(isinstance(value, str) and value for value in identity):
            raise ValueError(f"native run identity is missing: {path}")
        typed_identity = (str(identity[0]), str(identity[1]))
        if run_identity is None:
            run_identity = typed_identity
        elif typed_identity != run_identity:
            raise ValueError(f"native DP ranks have different run identities: {path}")

        timing = payload.get("timing")
        if not isinstance(timing, dict):
            raise TypeError(f"native result has no timing object: {path}")
        elapsed = float(timing.get("benchmark_elapsed_seconds", -1))
        measured = float(timing.get("measured_iteration_seconds", -1))
        if (
            not math.isfinite(elapsed)
            or not math.isfinite(measured)
            or min(elapsed, measured) < 0
            or measured > elapsed + 1e-12
        ):
            raise ValueError(f"native result has invalid timing: {path}")
        rank_timings.append((rank, elapsed, measured))

        points = []
        for benchmark_id, row in enumerate(rows, start=1):
            if not isinstance(row, dict) or not isinstance(row.get("point"), dict):
                raise TypeError(f"native result entry is malformed: {path}")
            point = row["point"]
            required = {
                "point_type",
                "benchmark_id",
                "total_prefill_tokens",
                "total_kv_read_tokens",
                "batch_size",
            }
            if not required.issubset(point) or point["point_type"] != cell.workload_kind:
                raise ValueError(f"native point contract mismatch: {point}")
            if int(point["benchmark_id"]) != benchmark_id:
                raise ValueError(f"native benchmark IDs are not contiguous: {path}")
            _expected_scheduled(point)
            fpms = row.get("fpms")
            if not isinstance(fpms, list) or len(fpms) != 1:
                raise ValueError(f"native point must contain exactly one local FPM: {path}")
            _validate_fpm(point, fpms[0], rank=rank)
            local_fpms[(benchmark_id, rank)] = fpms[0]
            points.append(point)
        if canonical_points is None:
            canonical_points = points
        elif points != canonical_points:
            raise ValueError(f"native DP ranks generated different grids: {path}")
        if canonical_groups is None:
            canonical_groups = groups
        elif groups != canonical_groups:
            raise ValueError(f"native DP ranks have different synchronized iteration groups: {path}")

    if seen_ranks != set(expected_ranks):
        raise ValueError(f"native DP rank set mismatch: actual={sorted(seen_ranks)} expected={expected_ranks}")
    assert canonical_points is not None and canonical_groups is not None and run_identity is not None

    measurements = []
    measured_iteration_seconds = 0.0
    for benchmark_id, (point, group) in enumerate(zip(canonical_points, canonical_groups, strict=True), start=1):
        if not isinstance(group, dict):
            raise TypeError(f"native iteration group {benchmark_id} is not a mapping")
        if (
            group.get("benchmark_id") != benchmark_id
            or group.get("point") != point
            or group.get("expected_dp_ranks") != expected_ranks
            or group.get("complete") is not True
        ):
            raise ValueError(f"native iteration group contract mismatch for benchmark_id={benchmark_id}")
        rank_results = group.get("rank_results")
        if (
            not isinstance(rank_results, list)
            or not all(isinstance(item, dict) for item in rank_results)
            or [item.get("dp_rank") for item in rank_results] != expected_ranks
        ):
            raise ValueError(f"native iteration group rank mismatch for benchmark_id={benchmark_id}")
        wall_times = []
        for rank_result in rank_results:
            rank = int(rank_result["dp_rank"])
            fpms = rank_result.get("fpms")
            if not isinstance(fpms, list) or len(fpms) != 1:
                raise ValueError(f"native iteration group has invalid FPM count for rank={rank}")
            wall_time = _validate_fpm(point, fpms[0], rank=rank)
            if fpms[0] != local_fpms[(benchmark_id, rank)]:
                raise ValueError(f"native local result differs from synchronized group for rank={rank}")
            wall_times.append((rank, wall_time))
        group_wall_time = float(group.get("wall_time", -1))
        expected_wall_time = max(value for _, value in wall_times)
        if not math.isfinite(group_wall_time) or not math.isclose(
            group_wall_time, expected_wall_time, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError(f"native iteration wall_time mismatch for benchmark_id={benchmark_id}")
        measured_iteration_seconds += group_wall_time
        measurements.append(NativePointMeasurement(point=dict(point), rank_wall_times=tuple(wall_times)))

    for rank, _elapsed, measured in rank_timings:
        if not math.isclose(measured, measured_iteration_seconds, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"native measured timing mismatch for rank={rank}")
    return NativeCollection(
        points=tuple(measurements),
        rank_timings=tuple(sorted(rank_timings)),
        backend_version=backend_version,
        collector_attempt_id=collector_attempt_id,
        runtime_run_id=run_identity[0],
        runtime_grid_digest=run_identity[1],
    )

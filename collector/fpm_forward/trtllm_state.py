# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable manifest and resume state for TRT-LLM FPM measurements."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TRTLLM_COORDINATE_SYSTEM = "iteration_totals_balanced_v1"
TRTLLM_MEASUREMENT_POLICY = "trtllm_gpu_forward_rank_max_single_sample_v1"
_TRTLLM_TIMING_SOURCE = "trtllm_iteration_stats"

_MANIFEST_SCHEMA_NAME = "aic_trtllm_fpm_coordinate_manifest"
_MANIFEST_SCHEMA_VERSION = 1
_ACCEPTED_SCHEMA_NAME = "aic_trtllm_fpm_accepted_measurement"
_ACCEPTED_SCHEMA_VERSION = 1

_REQUIRED_IBS_FIELDS = (
    "numContextRequests",
    "numCtxTokens",
    "numCtxKvTokens",
    "numGenRequests",
    "numGenKvTokens",
    "numQueuedContextRequests",
    "numQueuedCtxTokens",
    "numQueuedGenRequests",
    "numQueuedGenKvTokens",
    "numPausedRequests",
    "numPausedKvTokens",
)
_RUNTIME_LIMIT_KEYS = frozenset(
    {
        "max_seq_len",
        "max_num_requests",
        "max_batch_size",
        "max_num_tokens",
        "kv_cache_max_num_blocks",
        "kv_cache_tokens_per_block",
        "decode_cuda_graph_batch_sizes",
    }
)
_COORDINATE_KEYS = frozenset(
    {"workload_kind", "batch_size", "total_prefill_tokens", "total_kv_read_tokens", "point_id"}
)
_MANIFEST_KEYS = frozenset(
    {
        "schema_name",
        "schema_version",
        "campaign_id",
        "timing_rank_count",
        "coordinate_system",
        "measurement_policy",
        "runtime_limits",
        "coordinates",
        "sha256",
    }
)
_ACCEPTED_KEYS = frozenset(
    {
        "schema_name",
        "schema_version",
        "manifest_sha256",
        "campaign_id",
        "point_id",
        "attempt_id",
        "iteration_id",
        "coordinate",
        "inflight_batching_stats",
        "rank_times_ms",
        "latency_ms",
        "timing_source",
        "measurement_policy",
    }
)
_RANK_TIME_KEYS = frozenset({"rank", "gpu_forward_time_ms"})


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _require_strict_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, got {value!r}")
    return value


def _require_exact_keys(payload: dict[str, object], expected: frozenset[str], field_name: str) -> None:
    actual = set(payload)
    if actual != expected:
        raise ValueError(
            f"{field_name} keys differ: missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(_canonical_json(payload) + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _balanced_partition(
    total: int,
    count: int,
    *,
    unit: int = 1,
    minimum_units: int = 0,
) -> tuple[int, ...]:
    total_units = total // unit
    quotient, remainder = divmod(total_units - count * minimum_units, count)
    return tuple((minimum_units + quotient + int(index < remainder)) * unit for index in range(count))


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _validate_runtime_feasibility(
    coordinate: TrtllmCoordinate,
    runtime_limits: TrtllmRuntimeLimits,
) -> None:
    batch_limit = min(runtime_limits.max_num_requests, runtime_limits.max_batch_size)
    if coordinate.batch_size > batch_limit:
        raise ValueError(f"coordinate exceeds the direct runtime limit for batch size: {coordinate.point_id}")
    scheduled_tokens = (
        coordinate.total_prefill_tokens if coordinate.workload_kind == "prefill" else coordinate.batch_size
    )
    if scheduled_tokens > runtime_limits.max_num_tokens:
        raise ValueError(f"coordinate exceeds the direct runtime limit for scheduled tokens: {coordinate.point_id}")
    block_size = runtime_limits.kv_cache_tokens_per_block
    if coordinate.workload_kind == "decode":
        context_lengths = _balanced_partition(
            coordinate.total_kv_read_tokens,
            coordinate.batch_size,
            minimum_units=1,
        )
        if any(context_length + 2 > runtime_limits.max_seq_len for context_length in context_lengths):
            raise ValueError(
                f"coordinate exceeds the runtime limit for decode request sequence length: {coordinate.point_id}"
            )
        required_blocks = sum(_ceil_div(context_length + 1, block_size) for context_length in context_lengths)
        if required_blocks > runtime_limits.kv_cache_max_num_blocks:
            raise ValueError(f"coordinate exceeds the runtime limit for KV-cache blocks: {coordinate.point_id}")
        return

    if coordinate.total_kv_read_tokens % block_size != 0:
        raise ValueError(
            "coordinate violates the runtime limit: prefill KV-read total must contain whole KV-cache blocks: "
            f"{coordinate.point_id}"
        )
    if 0 < coordinate.total_kv_read_tokens < coordinate.batch_size * block_size:
        raise ValueError(
            "coordinate violates the runtime limit: prefill must provide at least one KV-cache block per request: "
            f"{coordinate.point_id}"
        )

    new_token_lengths = _balanced_partition(
        coordinate.total_prefill_tokens,
        coordinate.batch_size,
        minimum_units=1,
    )
    if coordinate.total_kv_read_tokens == 0:
        kv_read_lengths = (0,) * coordinate.batch_size
    else:
        kv_read_lengths = _balanced_partition(
            coordinate.total_kv_read_tokens,
            coordinate.batch_size,
            unit=block_size,
            minimum_units=1,
        )
    prompt_lengths = tuple(
        kv_read_tokens + new_tokens
        for kv_read_tokens, new_tokens in zip(kv_read_lengths, new_token_lengths, strict=True)
    )
    if any(prompt_length + 1 > runtime_limits.max_seq_len for prompt_length in prompt_lengths):
        raise ValueError(
            f"coordinate exceeds the runtime limit for prefill request sequence length: {coordinate.point_id}"
        )
    required_blocks = sum(_ceil_div(length, block_size) for length in prompt_lengths)
    if required_blocks > runtime_limits.kv_cache_max_num_blocks:
        raise ValueError(f"coordinate exceeds the runtime limit for KV-cache blocks: {coordinate.point_id}")


def _expected_ibs_coordinate(coordinate: TrtllmCoordinate) -> dict[str, int]:
    expected = dict.fromkeys(_REQUIRED_IBS_FIELDS, 0)
    if coordinate.workload_kind == "prefill":
        expected.update(
            numContextRequests=coordinate.batch_size,
            numCtxTokens=coordinate.total_prefill_tokens,
            numCtxKvTokens=coordinate.total_kv_read_tokens,
        )
    else:
        expected.update(
            numGenRequests=coordinate.batch_size,
            numGenKvTokens=coordinate.total_kv_read_tokens,
        )
    return expected


def _validate_rank_stats(
    coordinate: TrtllmCoordinate,
    rank_stats: Sequence[Mapping[str, object]],
    timing_rank_count: int,
) -> tuple[int, dict[str, object], tuple[tuple[int, float], ...]]:
    if not isinstance(rank_stats, Sequence) or isinstance(rank_stats, (str, bytes, bytearray)):
        raise TypeError("rank_stats must be a sequence of mappings")
    rows = tuple(rank_stats)
    if len(rows) != timing_rank_count:
        raise ValueError(f"rank_stats must contain exactly {timing_rank_count} rows, got {len(rows)}")

    expected_ibs = _expected_ibs_coordinate(coordinate)
    iterations: list[int] = []
    ibs_documents: list[tuple[str, dict[str, object]]] = []
    rank_times: list[tuple[int, float]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"rank_stats[{index}] must be a mapping")
        rank = _require_strict_int(row.get("rank"), f"rank_stats[{index}].rank")
        iteration = _require_strict_int(row.get("iter"), f"rank_stats[{index}].iter")
        if iteration < 0:
            raise ValueError(f"rank_stats[{index}].iter must be non-negative")
        gpu_time = row.get("gpuForwardTimeMS")
        if not isinstance(gpu_time, (int, float)) or isinstance(gpu_time, bool):
            raise TypeError(f"rank_stats[{index}].gpuForwardTimeMS must be numeric")
        gpu_time_ms = float(gpu_time)
        if not math.isfinite(gpu_time_ms) or gpu_time_ms <= 0:
            raise ValueError(f"rank_stats[{index}].gpuForwardTimeMS must be finite and positive")

        ibs = row.get("inflightBatchingStats")
        if not isinstance(ibs, Mapping):
            raise TypeError(f"rank_stats[{index}].inflightBatchingStats must be a mapping")
        if any(not isinstance(key, str) for key in ibs):
            raise TypeError(f"rank_stats[{index}].inflightBatchingStats keys must be strings")
        for field_name in _REQUIRED_IBS_FIELDS:
            value = _require_strict_int(
                ibs.get(field_name),
                f"rank_stats[{index}].inflightBatchingStats.{field_name}",
            )
            if value < 0:
                raise ValueError(f"rank_stats[{index}].inflightBatchingStats.{field_name} must be non-negative")
            if value != expected_ibs[field_name]:
                raise ValueError(
                    f"rank_stats[{index}] does not match requested coordinate: "
                    f"{field_name}={value}, expected={expected_ibs[field_name]}"
                )
        if "numScheduledRequests" in ibs:
            scheduled_requests = _require_strict_int(
                ibs["numScheduledRequests"],
                f"rank_stats[{index}].inflightBatchingStats.numScheduledRequests",
            )
            if scheduled_requests != coordinate.batch_size:
                raise ValueError(f"rank_stats[{index}] does not match requested coordinate: numScheduledRequests")

        ibs_json = _canonical_json(dict(ibs))
        ibs_documents.append((ibs_json, json.loads(ibs_json)))
        iterations.append(iteration)
        rank_times.append((rank, gpu_time_ms))

    ranks = sorted(rank for rank, _time in rank_times)
    if ranks != list(range(timing_rank_count)):
        raise ValueError(f"rank_stats must contain exactly ranks 0..{timing_rank_count - 1}, got {ranks}")
    if len(set(iterations)) != 1:
        raise ValueError(f"rank_stats iteration IDs differ: {sorted(set(iterations))}")
    if len({document for document, _ibs in ibs_documents}) != 1:
        raise ValueError("rank_stats inflightBatchingStats mappings differ across ranks")

    sorted_rank_times = tuple(sorted(rank_times))
    return iterations[0], ibs_documents[0][1], sorted_rank_times


@dataclass(frozen=True, slots=True)
class TrtllmRuntimeLimits:
    max_seq_len: int
    max_num_requests: int
    max_batch_size: int
    max_num_tokens: int
    kv_cache_max_num_blocks: int
    kv_cache_tokens_per_block: int
    decode_cuda_graph_batch_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "max_seq_len",
            "max_num_requests",
            "max_batch_size",
            "max_num_tokens",
            "kv_cache_max_num_blocks",
            "kv_cache_tokens_per_block",
        ):
            value = _require_strict_int(getattr(self, field_name), f"TrtllmRuntimeLimits.{field_name}")
            if value <= 0:
                raise ValueError(f"TrtllmRuntimeLimits.{field_name} must be positive")
        if not isinstance(self.decode_cuda_graph_batch_sizes, tuple):
            raise TypeError("TrtllmRuntimeLimits.decode_cuda_graph_batch_sizes must be a tuple")
        for value in self.decode_cuda_graph_batch_sizes:
            _require_strict_int(value, "TrtllmRuntimeLimits.decode_cuda_graph_batch_sizes[]")
        if not self.decode_cuda_graph_batch_sizes:
            raise ValueError("TrtllmRuntimeLimits.decode_cuda_graph_batch_sizes must not be empty")
        if any(value <= 0 for value in self.decode_cuda_graph_batch_sizes):
            raise ValueError("TrtllmRuntimeLimits.decode_cuda_graph_batch_sizes must be positive")
        if any(
            left >= right
            for left, right in zip(
                self.decode_cuda_graph_batch_sizes,
                self.decode_cuda_graph_batch_sizes[1:],
                strict=False,
            )
        ):
            raise ValueError("TrtllmRuntimeLimits.decode_cuda_graph_batch_sizes must be strictly increasing")
        graph_batch_limit = min(self.max_num_requests, self.max_batch_size, self.max_num_tokens)
        if self.decode_cuda_graph_batch_sizes[-1] > graph_batch_limit:
            raise ValueError(
                "TrtllmRuntimeLimits.decode_cuda_graph_batch_sizes exceed the direct scheduler batch limit"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "max_seq_len": self.max_seq_len,
            "max_num_requests": self.max_num_requests,
            "max_batch_size": self.max_batch_size,
            "max_num_tokens": self.max_num_tokens,
            "kv_cache_max_num_blocks": self.kv_cache_max_num_blocks,
            "kv_cache_tokens_per_block": self.kv_cache_tokens_per_block,
            "decode_cuda_graph_batch_sizes": list(self.decode_cuda_graph_batch_sizes),
        }


@dataclass(frozen=True, slots=True)
class TrtllmCoordinate:
    workload_kind: Literal["prefill", "decode"]
    batch_size: int
    total_prefill_tokens: int
    total_kv_read_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.workload_kind, str):
            raise TypeError(f"TrtllmCoordinate.workload_kind must be a string, got {self.workload_kind!r}")
        if self.workload_kind not in ("prefill", "decode"):
            raise ValueError(f"unsupported TrtllmCoordinate.workload_kind: {self.workload_kind!r}")
        for field_name in ("batch_size", "total_prefill_tokens", "total_kv_read_tokens"):
            _require_strict_int(getattr(self, field_name), f"TrtllmCoordinate.{field_name}")
        if self.batch_size <= 0:
            raise ValueError("TrtllmCoordinate.batch_size must be positive")
        if self.total_prefill_tokens < 0:
            raise ValueError("TrtllmCoordinate.total_prefill_tokens must be non-negative")
        if self.total_kv_read_tokens < 0:
            raise ValueError("TrtllmCoordinate.total_kv_read_tokens must be non-negative")
        if self.workload_kind == "prefill":
            if self.total_prefill_tokens < self.batch_size:
                raise ValueError("prefill total_prefill_tokens must be at least batch_size")
            if 0 < self.total_kv_read_tokens < self.batch_size:
                raise ValueError("prefill total_kv_read_tokens must be zero or at least batch_size")
        elif self.total_prefill_tokens != 0:
            raise ValueError("decode coordinates require zero total_prefill_tokens")
        elif self.total_kv_read_tokens < self.batch_size:
            raise ValueError("decode total_kv_read_tokens must be at least batch_size")

    @property
    def point_id(self) -> str:
        return _sha256(self._physical_dict())

    def _physical_dict(self) -> dict[str, object]:
        return {
            "workload_kind": self.workload_kind,
            "batch_size": self.batch_size,
            "total_prefill_tokens": self.total_prefill_tokens,
            "total_kv_read_tokens": self.total_kv_read_tokens,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._physical_dict(), "point_id": self.point_id}


@dataclass(frozen=True, slots=True, init=False)
class TrtllmManifest:
    campaign_id: str
    timing_rank_count: int
    runtime_limits: TrtllmRuntimeLimits
    coordinates: tuple[TrtllmCoordinate, ...]
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        campaign_id: str,
        timing_rank_count: int,
        runtime_limits: TrtllmRuntimeLimits,
        coordinates: Iterable[TrtllmCoordinate],
    ) -> TrtllmManifest:
        if not isinstance(campaign_id, str):
            raise TypeError(f"campaign_id must be a string, got {campaign_id!r}")
        if not campaign_id.strip():
            raise ValueError("campaign_id must not be empty")
        rank_count = _require_strict_int(timing_rank_count, "timing_rank_count")
        if rank_count <= 0:
            raise ValueError("timing_rank_count must be positive")
        if not isinstance(runtime_limits, TrtllmRuntimeLimits):
            raise TypeError("runtime_limits must be a TrtllmRuntimeLimits record")
        frozen_coordinates = tuple(coordinates)
        if not frozen_coordinates:
            raise ValueError("coordinates must not be empty")
        if any(not isinstance(coordinate, TrtllmCoordinate) for coordinate in frozen_coordinates):
            raise TypeError("coordinates must contain only TrtllmCoordinate records")
        point_ids = [coordinate.point_id for coordinate in frozen_coordinates]
        if len(set(point_ids)) != len(point_ids):
            raise ValueError("coordinates contain a duplicate physical coordinate")
        for coordinate in frozen_coordinates:
            _validate_runtime_feasibility(coordinate, runtime_limits)
        payload = cls._identity_dict(
            campaign_id=campaign_id,
            timing_rank_count=timing_rank_count,
            runtime_limits=runtime_limits,
            coordinates=frozen_coordinates,
        )
        manifest = object.__new__(cls)
        object.__setattr__(manifest, "campaign_id", campaign_id)
        object.__setattr__(manifest, "timing_rank_count", timing_rank_count)
        object.__setattr__(manifest, "runtime_limits", runtime_limits)
        object.__setattr__(manifest, "coordinates", frozen_coordinates)
        object.__setattr__(manifest, "sha256", _sha256(payload))
        return manifest

    @classmethod
    def load(cls, path: str | Path) -> TrtllmManifest:
        manifest_path = Path(path)
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot load TRT-LLM manifest {manifest_path}: {error}") from error
        if not isinstance(payload, dict):
            raise TypeError(f"TRT-LLM manifest must be a mapping: {manifest_path}")
        _require_exact_keys(payload, _MANIFEST_KEYS, "manifest")

        if payload["schema_name"] != _MANIFEST_SCHEMA_NAME:
            raise ValueError(f"unsupported TRT-LLM manifest schema_name: {payload['schema_name']!r}")
        schema_version = _require_strict_int(payload["schema_version"], "manifest.schema_version")
        if schema_version != _MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported TRT-LLM manifest schema_version: {schema_version}")
        if payload["coordinate_system"] != TRTLLM_COORDINATE_SYSTEM:
            raise ValueError(f"unsupported TRT-LLM coordinate_system: {payload['coordinate_system']!r}")
        if payload["measurement_policy"] != TRTLLM_MEASUREMENT_POLICY:
            raise ValueError(f"unsupported TRT-LLM measurement_policy: {payload['measurement_policy']!r}")

        runtime_payload = payload["runtime_limits"]
        if not isinstance(runtime_payload, dict):
            raise TypeError("manifest.runtime_limits must be a mapping")
        _require_exact_keys(runtime_payload, _RUNTIME_LIMIT_KEYS, "manifest.runtime_limits")
        graph_sizes = runtime_payload["decode_cuda_graph_batch_sizes"]
        if not isinstance(graph_sizes, list):
            raise TypeError("manifest.runtime_limits.decode_cuda_graph_batch_sizes must be a list")
        runtime_limits = TrtllmRuntimeLimits(
            max_seq_len=runtime_payload["max_seq_len"],
            max_num_requests=runtime_payload["max_num_requests"],
            max_batch_size=runtime_payload["max_batch_size"],
            max_num_tokens=runtime_payload["max_num_tokens"],
            kv_cache_max_num_blocks=runtime_payload["kv_cache_max_num_blocks"],
            kv_cache_tokens_per_block=runtime_payload["kv_cache_tokens_per_block"],
            decode_cuda_graph_batch_sizes=tuple(graph_sizes),
        )

        coordinates_payload = payload["coordinates"]
        if not isinstance(coordinates_payload, list):
            raise TypeError("manifest.coordinates must be a list")
        coordinates = []
        for index, coordinate_payload in enumerate(coordinates_payload):
            if not isinstance(coordinate_payload, dict):
                raise TypeError(f"manifest.coordinates[{index}] must be a mapping")
            _require_exact_keys(coordinate_payload, _COORDINATE_KEYS, f"manifest.coordinates[{index}]")
            coordinate = TrtllmCoordinate(
                workload_kind=coordinate_payload["workload_kind"],
                batch_size=coordinate_payload["batch_size"],
                total_prefill_tokens=coordinate_payload["total_prefill_tokens"],
                total_kv_read_tokens=coordinate_payload["total_kv_read_tokens"],
            )
            if coordinate_payload["point_id"] != coordinate.point_id:
                raise ValueError(f"manifest coordinate point_id mismatch at index {index}")
            coordinates.append(coordinate)

        manifest = cls.build(
            campaign_id=payload["campaign_id"],
            timing_rank_count=payload["timing_rank_count"],
            runtime_limits=runtime_limits,
            coordinates=coordinates,
        )
        supplied_sha256 = payload["sha256"]
        if not isinstance(supplied_sha256, str) or supplied_sha256 != manifest.sha256:
            raise ValueError(
                f"TRT-LLM manifest sha256 mismatch: supplied={supplied_sha256!r}, computed={manifest.sha256}"
            )
        return manifest

    @staticmethod
    def _identity_dict(
        *,
        campaign_id: str,
        timing_rank_count: int,
        runtime_limits: TrtllmRuntimeLimits,
        coordinates: tuple[TrtllmCoordinate, ...],
    ) -> dict[str, object]:
        return {
            "schema_name": _MANIFEST_SCHEMA_NAME,
            "schema_version": _MANIFEST_SCHEMA_VERSION,
            "campaign_id": campaign_id,
            "timing_rank_count": timing_rank_count,
            "coordinate_system": TRTLLM_COORDINATE_SYSTEM,
            "measurement_policy": TRTLLM_MEASUREMENT_POLICY,
            "runtime_limits": runtime_limits.to_dict(),
            "coordinates": [coordinate.to_dict() for coordinate in coordinates],
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._identity_dict(
                campaign_id=self.campaign_id,
                timing_rank_count=self.timing_rank_count,
                runtime_limits=self.runtime_limits,
                coordinates=self.coordinates,
            ),
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True, init=False)
class TrtllmAcceptedMeasurement:
    manifest_sha256: str
    campaign_id: str
    point_id: str
    attempt_id: str
    iteration_id: int
    coordinate: TrtllmCoordinate
    rank_times_ms: tuple[tuple[int, float], ...]
    latency_ms: float
    timing_source: str
    measurement_policy: str
    _inflight_batching_stats_json: str

    @classmethod
    def _create(
        cls,
        *,
        manifest_sha256: str,
        campaign_id: str,
        point_id: str,
        attempt_id: str,
        iteration_id: int,
        coordinate: TrtllmCoordinate,
        inflight_batching_stats: dict[str, object],
        rank_times_ms: tuple[tuple[int, float], ...],
        latency_ms: float,
    ) -> TrtllmAcceptedMeasurement:
        measurement = object.__new__(cls)
        object.__setattr__(measurement, "manifest_sha256", manifest_sha256)
        object.__setattr__(measurement, "campaign_id", campaign_id)
        object.__setattr__(measurement, "point_id", point_id)
        object.__setattr__(measurement, "attempt_id", attempt_id)
        object.__setattr__(measurement, "iteration_id", iteration_id)
        object.__setattr__(measurement, "coordinate", coordinate)
        object.__setattr__(measurement, "rank_times_ms", rank_times_ms)
        object.__setattr__(measurement, "latency_ms", latency_ms)
        object.__setattr__(measurement, "timing_source", _TRTLLM_TIMING_SOURCE)
        object.__setattr__(measurement, "measurement_policy", TRTLLM_MEASUREMENT_POLICY)
        object.__setattr__(
            measurement,
            "_inflight_batching_stats_json",
            _canonical_json(inflight_batching_stats),
        )
        return measurement

    @classmethod
    def _load(
        cls,
        path: Path,
        *,
        manifest: TrtllmManifest,
        expected_coordinate: TrtllmCoordinate,
    ) -> TrtllmAcceptedMeasurement:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot load accepted TRT-LLM measurement {path}: {error}") from error
        if not isinstance(payload, dict):
            raise TypeError(f"accepted TRT-LLM measurement must be a mapping: {path}")
        _require_exact_keys(payload, _ACCEPTED_KEYS, "accepted measurement")
        if payload["schema_name"] != _ACCEPTED_SCHEMA_NAME:
            raise ValueError(f"unsupported accepted measurement schema_name: {payload['schema_name']!r}")
        schema_version = _require_strict_int(
            payload["schema_version"],
            "accepted measurement.schema_version",
        )
        if schema_version != _ACCEPTED_SCHEMA_VERSION:
            raise ValueError(f"unsupported accepted measurement schema_version: {schema_version}")
        if payload["manifest_sha256"] != manifest.sha256:
            raise ValueError(f"accepted measurement has the wrong manifest identity: {path}")
        if payload["campaign_id"] != manifest.campaign_id:
            raise ValueError(f"accepted measurement has the wrong campaign identity: {path}")
        if payload["point_id"] != expected_coordinate.point_id:
            raise ValueError(f"accepted measurement has the wrong point identity: {path}")
        attempt_id = payload["attempt_id"]
        if not isinstance(attempt_id, str):
            raise TypeError(f"accepted measurement attempt_id must be a string: {path}")
        if not attempt_id.strip():
            raise ValueError(f"accepted measurement attempt_id must not be empty: {path}")
        iteration_id = _require_strict_int(payload["iteration_id"], "accepted measurement.iteration_id")
        if iteration_id < 0:
            raise ValueError(f"accepted measurement iteration_id must be non-negative: {path}")

        coordinate_payload = payload["coordinate"]
        if not isinstance(coordinate_payload, dict):
            raise TypeError(f"accepted measurement coordinate must be a mapping: {path}")
        _require_exact_keys(coordinate_payload, _COORDINATE_KEYS, "accepted measurement.coordinate")
        coordinate = TrtllmCoordinate(
            workload_kind=coordinate_payload["workload_kind"],
            batch_size=coordinate_payload["batch_size"],
            total_prefill_tokens=coordinate_payload["total_prefill_tokens"],
            total_kv_read_tokens=coordinate_payload["total_kv_read_tokens"],
        )
        if coordinate_payload["point_id"] != coordinate.point_id or coordinate != expected_coordinate:
            raise ValueError(f"accepted measurement coordinate does not match its frozen point: {path}")

        inflight_batching_stats = payload["inflight_batching_stats"]
        if not isinstance(inflight_batching_stats, dict):
            raise TypeError(f"accepted measurement inflight_batching_stats must be a mapping: {path}")
        rank_times_payload = payload["rank_times_ms"]
        if not isinstance(rank_times_payload, list):
            raise TypeError(f"accepted measurement rank_times_ms must be a list: {path}")
        serialized_rank_times = []
        reconstructed_rows = []
        for index, rank_time_payload in enumerate(rank_times_payload):
            if not isinstance(rank_time_payload, dict):
                raise TypeError(f"accepted measurement rank_times_ms[{index}] must be a mapping: {path}")
            _require_exact_keys(
                rank_time_payload,
                _RANK_TIME_KEYS,
                f"accepted measurement.rank_times_ms[{index}]",
            )
            rank = rank_time_payload["rank"]
            gpu_time = rank_time_payload["gpu_forward_time_ms"]
            serialized_rank_times.append((rank, gpu_time))
            reconstructed_rows.append(
                {
                    "rank": rank,
                    "iter": iteration_id,
                    "gpuForwardTimeMS": gpu_time,
                    "inflightBatchingStats": inflight_batching_stats,
                }
            )
        validated_iteration, validated_ibs, rank_times_ms = _validate_rank_stats(
            coordinate,
            reconstructed_rows,
            manifest.timing_rank_count,
        )
        if validated_iteration != iteration_id or tuple(serialized_rank_times) != rank_times_ms:
            raise ValueError(f"accepted measurement rank times are not sorted by rank: {path}")

        latency = payload["latency_ms"]
        if not isinstance(latency, (int, float)) or isinstance(latency, bool):
            raise TypeError(f"accepted measurement latency_ms must be numeric: {path}")
        latency_ms = float(latency)
        expected_latency_ms = max(time for _rank, time in rank_times_ms)
        if not math.isfinite(latency_ms) or latency_ms != expected_latency_ms:
            raise ValueError(f"accepted measurement latency_ms is not the rank maximum: {path}")
        if payload["timing_source"] != _TRTLLM_TIMING_SOURCE:
            raise ValueError(f"accepted measurement has the wrong timing_source: {path}")
        if payload["measurement_policy"] != TRTLLM_MEASUREMENT_POLICY:
            raise ValueError(f"accepted measurement has the wrong measurement_policy: {path}")

        return cls._create(
            manifest_sha256=manifest.sha256,
            campaign_id=manifest.campaign_id,
            point_id=coordinate.point_id,
            attempt_id=attempt_id,
            iteration_id=iteration_id,
            coordinate=coordinate,
            inflight_batching_stats=validated_ibs,
            rank_times_ms=rank_times_ms,
            latency_ms=latency_ms,
        )

    @property
    def inflight_batching_stats(self) -> dict[str, object]:
        return json.loads(self._inflight_batching_stats_json)

    def _to_dict(self) -> dict[str, object]:
        return {
            "schema_name": _ACCEPTED_SCHEMA_NAME,
            "schema_version": _ACCEPTED_SCHEMA_VERSION,
            "manifest_sha256": self.manifest_sha256,
            "campaign_id": self.campaign_id,
            "point_id": self.point_id,
            "attempt_id": self.attempt_id,
            "iteration_id": self.iteration_id,
            "coordinate": self.coordinate.to_dict(),
            "inflight_batching_stats": self.inflight_batching_stats,
            "rank_times_ms": [{"rank": rank, "gpu_forward_time_ms": time} for rank, time in self.rank_times_ms],
            "latency_ms": self.latency_ms,
            "timing_source": self.timing_source,
            "measurement_policy": self.measurement_policy,
        }


class TrtllmLedger:
    """Manifest-bound accepted-coordinate state derived from atomic files."""

    def __init__(self, root: Path, manifest: TrtllmManifest) -> None:
        self._root = root
        self._manifest = manifest
        self._accepted: dict[str, TrtllmAcceptedMeasurement] = {}

    @classmethod
    def open(
        cls,
        root: str | Path,
        *,
        manifest: TrtllmManifest,
        current_runtime_limits: TrtllmRuntimeLimits,
    ) -> TrtllmLedger:
        if not isinstance(manifest, TrtllmManifest):
            raise TypeError("manifest must be a TrtllmManifest record")
        if not isinstance(current_runtime_limits, TrtllmRuntimeLimits):
            raise TypeError("current_runtime_limits must be a TrtllmRuntimeLimits record")
        if current_runtime_limits != manifest.runtime_limits:
            raise ValueError("current TRT-LLM runtime limits do not match the frozen manifest")

        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        manifest_path = root_path / "manifest.json"
        if manifest_path.exists():
            persisted_manifest = TrtllmManifest.load(manifest_path)
            if persisted_manifest != manifest:
                raise ValueError("ledger manifest does not match the requested manifest")
        else:
            _atomic_write_json(manifest_path, manifest.to_dict())
        accepted_dir = root_path / "accepted"
        accepted_dir.mkdir(exist_ok=True)

        ledger = cls(root_path, manifest)
        coordinates_by_id = {coordinate.point_id: coordinate for coordinate in manifest.coordinates}
        for entry in sorted(accepted_dir.iterdir()):
            if entry.name.startswith(".") and entry.name.endswith(".tmp"):
                continue
            if not entry.is_file() or entry.suffix != ".json":
                raise ValueError(f"unknown entry in TRT-LLM accepted ledger: {entry}")
            coordinate = coordinates_by_id.get(entry.stem)
            if coordinate is None:
                raise ValueError(f"accepted ledger contains unknown point_id={entry.stem}")
            ledger._accepted[coordinate.point_id] = TrtllmAcceptedMeasurement._load(
                entry,
                manifest=manifest,
                expected_coordinate=coordinate,
            )
        return ledger

    @property
    def manifest(self) -> TrtllmManifest:
        return self._manifest

    @property
    def pending_coordinates(self) -> tuple[TrtllmCoordinate, ...]:
        return tuple(
            coordinate for coordinate in self._manifest.coordinates if coordinate.point_id not in self._accepted
        )

    @property
    def accepted_measurements(self) -> tuple[TrtllmAcceptedMeasurement, ...]:
        return tuple(
            self._accepted[coordinate.point_id]
            for coordinate in self._manifest.coordinates
            if coordinate.point_id in self._accepted
        )

    def accept(
        self,
        coordinate: TrtllmCoordinate,
        *,
        attempt_id: str,
        rank_stats: Sequence[Mapping[str, object]],
    ) -> TrtllmAcceptedMeasurement:
        if not isinstance(coordinate, TrtllmCoordinate):
            raise TypeError("coordinate must be a TrtllmCoordinate record")
        manifest_coordinates = {item.point_id: item for item in self._manifest.coordinates}
        frozen_coordinate = manifest_coordinates.get(coordinate.point_id)
        if frozen_coordinate is None or frozen_coordinate != coordinate:
            raise ValueError(f"coordinate is not present in the frozen manifest: {coordinate.point_id}")
        if not isinstance(attempt_id, str):
            raise TypeError(f"attempt_id must be a string, got {attempt_id!r}")
        if not attempt_id.strip():
            raise ValueError("attempt_id must not be empty")

        iteration_id, inflight_batching_stats, rank_times_ms = _validate_rank_stats(
            frozen_coordinate,
            rank_stats,
            self._manifest.timing_rank_count,
        )
        measurement = TrtllmAcceptedMeasurement._create(
            manifest_sha256=self._manifest.sha256,
            campaign_id=self._manifest.campaign_id,
            point_id=frozen_coordinate.point_id,
            attempt_id=attempt_id,
            iteration_id=iteration_id,
            coordinate=frozen_coordinate,
            inflight_batching_stats=inflight_batching_stats,
            rank_times_ms=rank_times_ms,
            latency_ms=max(time for _rank, time in rank_times_ms),
        )

        existing = self._accepted.get(frozen_coordinate.point_id)
        if existing is not None:
            if existing == measurement:
                return existing
            raise ValueError(f"conflicting accepted measurement for point_id={frozen_coordinate.point_id}")

        accepted_path = self._root / "accepted" / f"{frozen_coordinate.point_id}.json"
        if accepted_path.exists():
            raise ValueError(f"accepted measurement file appeared after ledger open: {accepted_path}")
        _atomic_write_json(accepted_path, measurement._to_dict())
        self._accepted[frozen_coordinate.point_id] = measurement
        return measurement

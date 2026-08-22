# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic TRT-LLM whole-forward FPM coordinate grid."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from collector.fpm_forward.trtllm_state import (
    TRTLLM_DECODE_MAX_NEW_TOKENS,
    TRTLLM_PREFILL_MAX_NEW_TOKENS,
    TrtllmCoordinate,
    TrtllmRuntimeLimits,
)

TrtllmWorkloadKind = Literal["prefill", "decode"]

_GRID_BUILD_SCHEMA_NAME = "aic_trtllm_fpm_runtime_grid"
_GRID_BUILD_SCHEMA_VERSION = 1

logger = logging.getLogger(__name__)


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class TrtllmGridProfile:
    """Versioned declaration of TRT-LLM sweep density, anchors, and order."""

    profile_id: str
    prefill_total_tokens_geometric_base: int
    prefill_batch_size_geometric_base: int
    prefill_kv_read_blocks_geometric_base: int
    prefill_prefix_blocks_per_request_anchors: tuple[int, ...]
    decode_graph_batch_offsets: tuple[int, ...]
    decode_eager_batch_geometric_base: int
    decode_kv_read_tokens_geometric_base: int
    include_direct_limit_endpoints: bool
    include_kv_capacity_endpoints: bool
    prefill_max_new_tokens: int
    decode_max_new_tokens: int
    phase_order: tuple[TrtllmWorkloadKind, ...]
    descending_phases: tuple[TrtllmWorkloadKind, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str):
            raise TypeError("TrtllmGridProfile.profile_id must be a string")
        if not self.profile_id.strip():
            raise ValueError("TrtllmGridProfile.profile_id must be a non-blank string")
        for field_name in (
            "prefill_total_tokens_geometric_base",
            "prefill_batch_size_geometric_base",
            "prefill_kv_read_blocks_geometric_base",
            "decode_eager_batch_geometric_base",
            "decode_kv_read_tokens_geometric_base",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"TrtllmGridProfile.{field_name} must be an integer")
            if value < 2:
                raise ValueError(f"TrtllmGridProfile.{field_name} must be at least 2")
        for field_name in ("prefill_max_new_tokens", "decode_max_new_tokens"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"TrtllmGridProfile.{field_name} must be an integer")
            if value <= 0:
                raise ValueError(f"TrtllmGridProfile.{field_name} must be positive")
        for field_name in ("include_direct_limit_endpoints", "include_kv_capacity_endpoints"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"TrtllmGridProfile.{field_name} must be a boolean")
        for field_name in (
            "prefill_prefix_blocks_per_request_anchors",
            "decode_graph_batch_offsets",
        ):
            values = getattr(self, field_name)
            if not isinstance(values, tuple):
                raise TypeError(f"TrtllmGridProfile.{field_name} must be a tuple")
            if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
                raise TypeError(f"TrtllmGridProfile.{field_name} values must be integers")
            if any(value < 0 for value in values):
                raise ValueError(f"TrtllmGridProfile.{field_name} values must be non-negative")
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"TrtllmGridProfile.{field_name} must be sorted and unique")
        for field_name in ("phase_order", "descending_phases"):
            values = getattr(self, field_name)
            if not isinstance(values, tuple):
                raise TypeError(f"TrtllmGridProfile.{field_name} must be a tuple")
            if any(not isinstance(value, str) for value in values):
                raise TypeError(f"TrtllmGridProfile.{field_name} values must be strings")
        if set(self.phase_order) != {"prefill", "decode"} or len(self.phase_order) != 2:
            raise ValueError("TrtllmGridProfile.phase_order must contain prefill and decode exactly once")
        descending_phases = set(self.descending_phases)
        if len(descending_phases) != len(self.descending_phases) or not descending_phases.issubset(self.phase_order):
            raise ValueError("TrtllmGridProfile.descending_phases must be a unique subset of phase_order")
        if self.descending_phases != tuple(phase for phase in self.phase_order if phase in descending_phases):
            raise ValueError("TrtllmGridProfile.descending_phases must follow phase_order")

    def to_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "prefill_total_tokens_geometric_base": self.prefill_total_tokens_geometric_base,
            "prefill_batch_size_geometric_base": self.prefill_batch_size_geometric_base,
            "prefill_kv_read_blocks_geometric_base": self.prefill_kv_read_blocks_geometric_base,
            "prefill_prefix_blocks_per_request_anchors": list(self.prefill_prefix_blocks_per_request_anchors),
            "decode_graph_batch_offsets": list(self.decode_graph_batch_offsets),
            "decode_eager_batch_geometric_base": self.decode_eager_batch_geometric_base,
            "decode_kv_read_tokens_geometric_base": self.decode_kv_read_tokens_geometric_base,
            "include_direct_limit_endpoints": self.include_direct_limit_endpoints,
            "include_kv_capacity_endpoints": self.include_kv_capacity_endpoints,
            "prefill_max_new_tokens": self.prefill_max_new_tokens,
            "decode_max_new_tokens": self.decode_max_new_tokens,
            "phase_order": list(self.phase_order),
            "descending_phases": list(self.descending_phases),
        }


TRTLLM_FULL_GRID_V1 = TrtllmGridProfile(
    profile_id="trtllm_full_grid_v1",
    prefill_total_tokens_geometric_base=2,
    prefill_batch_size_geometric_base=2,
    prefill_kv_read_blocks_geometric_base=2,
    prefill_prefix_blocks_per_request_anchors=(0, 1),
    decode_graph_batch_offsets=(0, 1),
    decode_eager_batch_geometric_base=2,
    decode_kv_read_tokens_geometric_base=2,
    include_direct_limit_endpoints=True,
    include_kv_capacity_endpoints=True,
    prefill_max_new_tokens=TRTLLM_PREFILL_MAX_NEW_TOKENS,
    decode_max_new_tokens=TRTLLM_DECODE_MAX_NEW_TOKENS,
    phase_order=("prefill", "decode"),
    descending_phases=("prefill",),
)


@dataclass(frozen=True, slots=True)
class TrtllmGridAdmissionSummary:
    candidate_count: int
    admitted_count: int
    dropped_by_reason: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.dropped_by_reason, tuple):
            raise TypeError("TrtllmGridAdmissionSummary.dropped_by_reason must be a tuple")
        if any(not isinstance(entry, tuple) or len(entry) != 2 for entry in self.dropped_by_reason):
            raise TypeError("each dropped_by_reason entry must be a 2-tuple")
        for field_name in ("candidate_count", "admitted_count"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"TrtllmGridAdmissionSummary.{field_name} must be an integer")
        if self.candidate_count < 0 or self.admitted_count < 0:
            raise ValueError("TRT-LLM grid admission counts must be non-negative")
        if self.admitted_count > self.candidate_count:
            raise ValueError("TRT-LLM admitted count cannot exceed candidate count")
        reasons = [reason for reason, _count in self.dropped_by_reason]
        if any(not isinstance(reason, str) for reason in reasons):
            raise TypeError("TRT-LLM grid drop reasons must be strings")
        if any(not reason.strip() for reason in reasons):
            raise ValueError("TRT-LLM grid drop reasons must be non-blank strings")
        if tuple(reasons) != tuple(sorted(set(reasons))):
            raise ValueError("TRT-LLM grid drop reasons must be sorted and unique")
        counts = [count for _reason, count in self.dropped_by_reason]
        if any(not isinstance(count, int) or isinstance(count, bool) for count in counts):
            raise TypeError("TRT-LLM grid drop counts must be integers")
        if any(count <= 0 for count in counts):
            raise ValueError("TRT-LLM grid drop counts must be positive")
        if sum(counts) != self.dropped_count:
            raise ValueError("TRT-LLM grid drop reasons must account for every dropped candidate")

    @property
    def dropped_count(self) -> int:
        return self.candidate_count - self.admitted_count

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_count": self.candidate_count,
            "admitted_count": self.admitted_count,
            "dropped_count": self.dropped_count,
            "dropped_by_reason": dict(self.dropped_by_reason),
        }


@dataclass(frozen=True, slots=True)
class TrtllmGridBuild:
    profile: TrtllmGridProfile
    runtime_limits: TrtllmRuntimeLimits
    coordinates: tuple[TrtllmCoordinate, ...]
    admission_summary: TrtllmGridAdmissionSummary
    runtime_grid_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.profile, TrtllmGridProfile):
            raise TypeError("TrtllmGridBuild.profile must be a TrtllmGridProfile record")
        if not isinstance(self.runtime_limits, TrtllmRuntimeLimits):
            raise TypeError("TrtllmGridBuild.runtime_limits must be a TrtllmRuntimeLimits record")
        if not isinstance(self.admission_summary, TrtllmGridAdmissionSummary):
            raise TypeError("TrtllmGridBuild.admission_summary must be a TrtllmGridAdmissionSummary record")
        if not isinstance(self.coordinates, tuple):
            raise TypeError("TrtllmGridBuild.coordinates must be a tuple")
        if any(not isinstance(coordinate, TrtllmCoordinate) for coordinate in self.coordinates):
            raise TypeError("TrtllmGridBuild.coordinates must contain only TrtllmCoordinate records")
        point_ids = [coordinate.point_id for coordinate in self.coordinates]
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("TrtllmGridBuild.coordinates contain a duplicate physical coordinate")
        if self.admission_summary.admitted_count != len(self.coordinates):
            raise ValueError("TrtllmGridBuild.admission_summary.admitted_count must equal the coordinate count")
        object.__setattr__(
            self,
            "runtime_grid_digest",
            hashlib.sha256(_canonical_json(self._identity_dict()).encode()).hexdigest(),
        )

    def _identity_dict(self) -> dict[str, object]:
        return {
            "schema_name": _GRID_BUILD_SCHEMA_NAME,
            "schema_version": _GRID_BUILD_SCHEMA_VERSION,
            "profile": self.profile.to_dict(),
            "runtime_limits": self.runtime_limits.to_dict(),
            "coordinates": [coordinate.to_dict() for coordinate in self.coordinates],
            "admission_summary": self.admission_summary.to_dict(),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._identity_dict(),
            "runtime_grid_digest": self.runtime_grid_digest,
        }


def _geometric_values(
    limit: int,
    *,
    base: int,
    include_endpoint: bool,
) -> tuple[int, ...]:
    values: list[int] = []
    value = 1
    while value <= limit:
        values.append(value)
        value *= base
    if include_endpoint and values[-1] != limit:
        values.append(limit)
    return tuple(values)


def _balanced_partition(total: int, count: int, *, minimum: int) -> tuple[int, ...]:
    quotient, remainder = divmod(total - count * minimum, count)
    return tuple(minimum + quotient + int(index < remainder) for index in range(count))


def _prefill_sequence_is_feasible(
    *,
    total_new_tokens: int,
    batch_size: int,
    total_prefix_blocks: int,
    runtime_limits: TrtllmRuntimeLimits,
    profile: TrtllmGridProfile,
) -> bool:
    new_lengths = _balanced_partition(total_new_tokens, batch_size, minimum=1)
    if total_prefix_blocks == 0:
        prefix_blocks = (0,) * batch_size
    elif total_prefix_blocks < batch_size:
        return False
    else:
        prefix_blocks = _balanced_partition(total_prefix_blocks, batch_size, minimum=1)

    block_size = runtime_limits.kv_cache_tokens_per_block
    return all(
        prefix_block_count * block_size + new_length + profile.prefill_max_new_tokens <= runtime_limits.max_seq_len
        for prefix_block_count, new_length in zip(prefix_blocks, new_lengths, strict=True)
    )


def _prefill_required_blocks(
    *,
    total_new_tokens: int,
    batch_size: int,
    total_prefix_blocks: int,
    runtime_limits: TrtllmRuntimeLimits,
    profile: TrtllmGridProfile,
) -> int:
    new_lengths = _balanced_partition(total_new_tokens, batch_size, minimum=1)
    prefix_blocks = (
        (0,) * batch_size
        if total_prefix_blocks == 0
        else _balanced_partition(total_prefix_blocks, batch_size, minimum=1)
    )
    block_size = runtime_limits.kv_cache_tokens_per_block
    return sum(
        prefix_block_count + (new_length + profile.prefill_max_new_tokens + block_size - 1) // block_size
        for prefix_block_count, new_length in zip(prefix_blocks, new_lengths, strict=True)
    )


def _maximum_prefill_prefix_blocks_by_sequence(
    *,
    total_new_tokens: int,
    batch_size: int,
    runtime_limits: TrtllmRuntimeLimits,
    profile: TrtllmGridProfile,
) -> int | None:
    if not _prefill_sequence_is_feasible(
        total_new_tokens=total_new_tokens,
        batch_size=batch_size,
        total_prefix_blocks=0,
        runtime_limits=runtime_limits,
        profile=profile,
    ):
        return None

    if not _prefill_sequence_is_feasible(
        total_new_tokens=total_new_tokens,
        batch_size=batch_size,
        total_prefix_blocks=batch_size,
        runtime_limits=runtime_limits,
        profile=profile,
    ):
        return 0

    lower = batch_size
    upper = batch_size * (
        (runtime_limits.max_seq_len - 1 - profile.prefill_max_new_tokens) // runtime_limits.kv_cache_tokens_per_block
    )
    while lower < upper:
        candidate = (lower + upper + 1) // 2
        if _prefill_sequence_is_feasible(
            total_new_tokens=total_new_tokens,
            batch_size=batch_size,
            total_prefix_blocks=candidate,
            runtime_limits=runtime_limits,
            profile=profile,
        ):
            lower = candidate
        else:
            upper = candidate - 1
    return lower


def _maximum_prefill_prefix_blocks_by_capacity(
    *,
    total_new_tokens: int,
    batch_size: int,
    sequence_maximum: int,
    runtime_limits: TrtllmRuntimeLimits,
    profile: TrtllmGridProfile,
) -> int | None:
    capacity = runtime_limits.kv_cache_max_num_blocks
    if (
        _prefill_required_blocks(
            total_new_tokens=total_new_tokens,
            batch_size=batch_size,
            total_prefix_blocks=0,
            runtime_limits=runtime_limits,
            profile=profile,
        )
        > capacity
    ):
        return None
    if sequence_maximum < batch_size or (
        _prefill_required_blocks(
            total_new_tokens=total_new_tokens,
            batch_size=batch_size,
            total_prefix_blocks=batch_size,
            runtime_limits=runtime_limits,
            profile=profile,
        )
        > capacity
    ):
        return 0

    lower = batch_size
    upper = sequence_maximum
    while lower < upper:
        candidate = (lower + upper + 1) // 2
        if (
            _prefill_required_blocks(
                total_new_tokens=total_new_tokens,
                batch_size=batch_size,
                total_prefix_blocks=candidate,
                runtime_limits=runtime_limits,
                profile=profile,
            )
            <= capacity
        ):
            lower = candidate
        else:
            upper = candidate - 1
    return lower


def _prefill_candidates(
    runtime_limits: TrtllmRuntimeLimits,
    profile: TrtllmGridProfile,
) -> tuple[TrtllmCoordinate, ...]:
    coordinates: list[TrtllmCoordinate] = []
    for total_new_tokens in _geometric_values(
        runtime_limits.max_num_tokens,
        base=profile.prefill_total_tokens_geometric_base,
        include_endpoint=profile.include_direct_limit_endpoints,
    ):
        batch_limit = min(
            total_new_tokens,
            runtime_limits.max_num_requests,
            runtime_limits.max_batch_size,
        )
        for batch_size in _geometric_values(
            batch_limit,
            base=profile.prefill_batch_size_geometric_base,
            include_endpoint=profile.include_direct_limit_endpoints,
        ):
            sequence_maximum = _maximum_prefill_prefix_blocks_by_sequence(
                total_new_tokens=total_new_tokens,
                batch_size=batch_size,
                runtime_limits=runtime_limits,
                profile=profile,
            )
            if sequence_maximum is None:
                continue

            prefix_block_totals = {
                blocks_per_request * batch_size
                for blocks_per_request in profile.prefill_prefix_blocks_per_request_anchors
                if blocks_per_request * batch_size <= sequence_maximum
            }
            if profile.include_direct_limit_endpoints:
                prefix_block_totals.add(sequence_maximum)
            if profile.include_kv_capacity_endpoints:
                capacity_maximum = _maximum_prefill_prefix_blocks_by_capacity(
                    total_new_tokens=total_new_tokens,
                    batch_size=batch_size,
                    sequence_maximum=sequence_maximum,
                    runtime_limits=runtime_limits,
                    profile=profile,
                )
                if capacity_maximum is not None:
                    prefix_block_totals.add(capacity_maximum)
            if sequence_maximum >= batch_size:
                prefix_block_totals.update(
                    value
                    for value in _geometric_values(
                        sequence_maximum,
                        base=profile.prefill_kv_read_blocks_geometric_base,
                        include_endpoint=False,
                    )
                    if value >= batch_size
                )
            for total_prefix_blocks in sorted(prefix_block_totals):
                coordinates.append(
                    TrtllmCoordinate(
                        workload_kind="prefill",
                        batch_size=batch_size,
                        total_prefill_tokens=total_new_tokens,
                        total_kv_read_tokens=(total_prefix_blocks * runtime_limits.kv_cache_tokens_per_block),
                    )
                )
    ordered = tuple(coordinates)
    return tuple(reversed(ordered)) if "prefill" in profile.descending_phases else ordered


def _decode_batch_sizes(
    runtime_limits: TrtllmRuntimeLimits,
    direct_maximum: int,
    profile: TrtllmGridProfile,
) -> tuple[int, ...]:
    capture_sizes = runtime_limits.decode_cuda_graph_batch_sizes
    batch_sizes: set[int] = set()
    for capture_size in capture_sizes:
        for offset in profile.decode_graph_batch_offsets:
            candidate = capture_size + offset
            if candidate <= direct_maximum:
                batch_sizes.add(candidate)

    if capture_sizes[-1] <= direct_maximum:
        value = capture_sizes[-1] * profile.decode_eager_batch_geometric_base
        while value < direct_maximum:
            batch_sizes.add(value)
            value *= profile.decode_eager_batch_geometric_base
    if profile.include_direct_limit_endpoints:
        batch_sizes.add(direct_maximum)
    if profile.include_kv_capacity_endpoints:
        blocks_per_minimum_request = (
            1 + profile.decode_max_new_tokens + runtime_limits.kv_cache_tokens_per_block - 1
        ) // runtime_limits.kv_cache_tokens_per_block
        capacity_maximum = min(
            direct_maximum,
            runtime_limits.kv_cache_max_num_blocks // blocks_per_minimum_request,
        )
        if capacity_maximum > 0:
            batch_sizes.add(capacity_maximum)
    return tuple(sorted(batch_sizes))


def _decode_required_blocks(
    *,
    total_context_tokens: int,
    batch_size: int,
    runtime_limits: TrtllmRuntimeLimits,
    profile: TrtllmGridProfile,
) -> int:
    context_lengths = _balanced_partition(total_context_tokens, batch_size, minimum=1)
    return sum(
        (context_length + profile.decode_max_new_tokens + runtime_limits.kv_cache_tokens_per_block - 1)
        // runtime_limits.kv_cache_tokens_per_block
        for context_length in context_lengths
    )


def _maximum_decode_context_tokens_by_capacity(
    *,
    batch_size: int,
    sequence_maximum: int,
    runtime_limits: TrtllmRuntimeLimits,
    profile: TrtllmGridProfile,
) -> int | None:
    capacity = runtime_limits.kv_cache_max_num_blocks
    if (
        _decode_required_blocks(
            total_context_tokens=batch_size,
            batch_size=batch_size,
            runtime_limits=runtime_limits,
            profile=profile,
        )
        > capacity
    ):
        return None

    lower = batch_size
    upper = sequence_maximum
    while lower < upper:
        candidate = (lower + upper + 1) // 2
        if (
            _decode_required_blocks(
                total_context_tokens=candidate,
                batch_size=batch_size,
                runtime_limits=runtime_limits,
                profile=profile,
            )
            <= capacity
        ):
            lower = candidate
        else:
            upper = candidate - 1
    return lower


def _decode_candidates(
    runtime_limits: TrtllmRuntimeLimits,
    profile: TrtllmGridProfile,
) -> tuple[TrtllmCoordinate, ...]:
    if runtime_limits.max_seq_len < 1 + profile.decode_max_new_tokens:
        return ()

    direct_batch_limit = min(
        runtime_limits.max_num_requests,
        runtime_limits.max_batch_size,
        runtime_limits.max_num_tokens,
    )
    coordinates: list[TrtllmCoordinate] = []
    for batch_size in _decode_batch_sizes(runtime_limits, direct_batch_limit, profile):
        sequence_maximum = batch_size * (runtime_limits.max_seq_len - profile.decode_max_new_tokens)

        context_token_totals: set[int] = set()
        if profile.include_direct_limit_endpoints:
            context_token_totals.update((batch_size, sequence_maximum))
        if profile.include_kv_capacity_endpoints:
            capacity_maximum = _maximum_decode_context_tokens_by_capacity(
                batch_size=batch_size,
                sequence_maximum=sequence_maximum,
                runtime_limits=runtime_limits,
                profile=profile,
            )
            if capacity_maximum is not None:
                context_token_totals.add(capacity_maximum)
        context_token_totals.update(
            value
            for value in _geometric_values(
                sequence_maximum,
                base=profile.decode_kv_read_tokens_geometric_base,
                include_endpoint=False,
            )
            if value >= batch_size
        )
        for total_context_tokens in sorted(context_token_totals):
            coordinates.append(
                TrtllmCoordinate(
                    workload_kind="decode",
                    batch_size=batch_size,
                    total_prefill_tokens=0,
                    total_kv_read_tokens=total_context_tokens,
                )
            )
    ordered = tuple(coordinates)
    return tuple(reversed(ordered)) if "decode" in profile.descending_phases else ordered


def _required_kv_cache_blocks(
    coordinate: TrtllmCoordinate,
    runtime_limits: TrtllmRuntimeLimits,
    profile: TrtllmGridProfile,
) -> int:
    if coordinate.workload_kind == "decode":
        return _decode_required_blocks(
            total_context_tokens=coordinate.total_kv_read_tokens,
            batch_size=coordinate.batch_size,
            runtime_limits=runtime_limits,
            profile=profile,
        )
    return _prefill_required_blocks(
        total_new_tokens=coordinate.total_prefill_tokens,
        batch_size=coordinate.batch_size,
        total_prefix_blocks=(coordinate.total_kv_read_tokens // runtime_limits.kv_cache_tokens_per_block),
        runtime_limits=runtime_limits,
        profile=profile,
    )


def _admit_candidates(
    candidates: tuple[TrtllmCoordinate, ...],
    runtime_limits: TrtllmRuntimeLimits,
    profile: TrtllmGridProfile,
) -> tuple[tuple[TrtllmCoordinate, ...], TrtllmGridAdmissionSummary]:
    admitted: list[TrtllmCoordinate] = []
    dropped: Counter[str] = Counter()
    for coordinate in candidates:
        if _required_kv_cache_blocks(coordinate, runtime_limits, profile) > runtime_limits.kv_cache_max_num_blocks:
            dropped["kv_cache_capacity"] += 1
        else:
            admitted.append(coordinate)
    summary = TrtllmGridAdmissionSummary(
        candidate_count=len(candidates),
        admitted_count=len(admitted),
        dropped_by_reason=tuple(sorted(dropped.items())),
    )
    return tuple(admitted), summary


def build_trtllm_grid(
    runtime_limits: TrtllmRuntimeLimits,
    *,
    profile: TrtllmGridProfile,
) -> TrtllmGridBuild:
    """Expand one declared grid profile under the supplied runtime limits."""

    candidate_phases = {
        "prefill": _prefill_candidates(runtime_limits, profile),
        "decode": _decode_candidates(runtime_limits, profile),
    }
    candidates = tuple(coordinate for phase in profile.phase_order for coordinate in candidate_phases[phase])
    coordinates, admission_summary = _admit_candidates(candidates, runtime_limits, profile)
    return TrtllmGridBuild(
        profile=profile,
        runtime_limits=runtime_limits,
        coordinates=coordinates,
        admission_summary=admission_summary,
    )


def build_trtllm_full_grid(
    runtime_limits: TrtllmRuntimeLimits,
) -> tuple[TrtllmCoordinate, ...]:
    """Build the declared full grid admitted by TRT-LLM limits."""

    build = build_trtllm_grid(runtime_limits, profile=TRTLLM_FULL_GRID_V1)
    if build.admission_summary.dropped_count:
        logger.info(
            "trtllm_fpm_forward: dropped %d/%d coordinates "
            "(KV-cache capacity, device=%d blocks x %d tokens/block=%d tokens)",
            build.admission_summary.dropped_count,
            build.admission_summary.candidate_count,
            runtime_limits.kv_cache_max_num_blocks,
            runtime_limits.kv_cache_tokens_per_block,
            runtime_limits.kv_cache_max_num_blocks * runtime_limits.kv_cache_tokens_per_block,
        )
    return build.coordinates

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic latency-blind Anchor -> Halton -> Maximin sampling."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass

from .types import FPMPoint

_BUDGET_DIVISORS = {
    "one_eighth": 8,
    "one_quarter": 4,
    "one_half": 2,
    "full": 1,
}


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _normalize_rows(rows: dict[FPMPoint, tuple[float, ...]]) -> dict[FPMPoint, tuple[float, ...]]:
    dimensions = len(next(iter(rows.values())))
    bounds = [
        (
            min(value[index] for value in rows.values()),
            max(value[index] for value in rows.values()),
        )
        for index in range(dimensions)
    ]

    def norm(value: float, low: float, high: float) -> float:
        return 0.0 if low == high else (value - low) / (high - low)

    return {
        point: tuple(norm(value[index], *bounds[index]) for index in range(dimensions)) for point, value in rows.items()
    }


def _coordinates(points: list[FPMPoint], block_size: int) -> dict[FPMPoint, tuple[float, ...]]:
    phase = points[0].workload_kind
    if phase == "prefill":
        raw = {
            point: (
                math.log2(point.batch_size),
                math.log2(point.suffix_length),
                math.log2(point.prefix_length + block_size),
                math.log2(point.batch_size * point.suffix_length),
                math.log2(point.batch_size * (point.prefix_length + point.suffix_length)),
                math.log2(point.batch_size * point.suffix_length * (point.prefix_length + point.suffix_length)),
            )
            for point in points
        }
        normalized = _normalize_rows(raw)
        return {point: (value[0], math.sqrt(value[1]), *value[2:]) for point, value in normalized.items()}

    raw = {
        point: (
            math.log2(point.batch_size),
            math.log2(point.prefix_length),
            math.log2(point.batch_size * point.prefix_length),
        )
        for point in points
    }
    normalized = _normalize_rows(raw)
    return {point: (value[0], math.sqrt(value[1]), value[2]) for point, value in normalized.items()}


def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def _prefill_anchors(points: list[FPMPoint]) -> tuple[list[FPMPoint], dict[FPMPoint, set[str]]]:
    by_prefix: dict[int, list[FPMPoint]] = defaultdict(list)
    for point in points:
        by_prefix[point.prefix_length].append(point)
    roles: dict[FPMPoint, set[str]] = defaultdict(set)

    def add(point: FPMPoint, role: str) -> None:
        roles[point].add(role)

    for prefix, slice_points in sorted(by_prefix.items()):
        batches = sorted({point.batch_size for point in slice_points})
        for batch, batch_role in ((batches[0], "min_batch"), (batches[-1], "max_batch")):
            batch_points = sorted(
                (point for point in slice_points if point.batch_size == batch),
                key=lambda point: point.suffix_length,
            )
            add(batch_points[0], f"p{prefix}_{batch_role}_min_suffix")
            add(batch_points[-1], f"p{prefix}_{batch_role}_max_suffix")

    prefix_faces = [min(by_prefix), max(by_prefix)]
    for prefix in dict.fromkeys(prefix_faces):
        slice_points = by_prefix[prefix]
        by_batch: dict[int, list[FPMPoint]] = defaultdict(list)
        for point in slice_points:
            by_batch[point.batch_size].append(point)
        batches = sorted(by_batch)
        min_batch = batches[0]
        for point in sorted(by_batch[min_batch], key=lambda point: point.suffix_length):
            if point.suffix_length == min(p.suffix_length for p in by_batch[min_batch]) or _is_power_of_two(
                point.suffix_length
            ):
                add(point, f"p{prefix}_min_batch_suffix_landmark")
        boundary_indices = sorted(set(range(0, len(batches), 2)) | {len(batches) - 1})
        for index in boundary_indices:
            batch_points = sorted(by_batch[batches[index]], key=lambda point: point.suffix_length)
            add(batch_points[0], f"p{prefix}_lower_suffix_boundary")
            add(batch_points[-1], f"p{prefix}_upper_suffix_boundary")
    return sorted(roles), roles


def _decode_anchors(points: list[FPMPoint]) -> tuple[list[FPMPoint], dict[FPMPoint, set[str]]]:
    by_batch: dict[int, list[FPMPoint]] = defaultdict(list)
    for point in points:
        by_batch[point.batch_size].append(point)
    roles: dict[FPMPoint, set[str]] = defaultdict(set)

    def add(point: FPMPoint, role: str) -> None:
        roles[point].add(role)

    batches = sorted(by_batch)
    for batch, batch_role in ((batches[0], "min_batch"), (batches[-1], "max_batch")):
        batch_points = sorted(by_batch[batch], key=lambda point: point.prefix_length)
        add(batch_points[0], f"{batch_role}_min_length")
        add(batch_points[-1], f"{batch_role}_max_length")
    for point in sorted(by_batch[batches[0]], key=lambda point: point.prefix_length):
        if _is_power_of_two(point.prefix_length):
            add(point, "min_batch_length_landmark")
    boundary_indices = sorted(set(range(0, len(batches), 2)) | {len(batches) - 1})
    for index in boundary_indices:
        batch_points = sorted(by_batch[batches[index]], key=lambda point: point.prefix_length)
        add(batch_points[0], "lower_length_boundary")
        add(batch_points[-1], "upper_length_boundary")
    return sorted(roles), roles


def _halton(index: int, base: int) -> float:
    value = 0.0
    fraction = 1.0 / base
    while index:
        value += fraction * (index % base)
        index //= base
        fraction /= base
    return value


def _halton_order(
    remaining: set[FPMPoint],
    coordinates: dict[FPMPoint, tuple[float, ...]],
    *,
    phase: str,
    count: int,
) -> list[FPMPoint]:
    bases = (2, 3, 5) if phase == "prefill" else (2, 3)
    selected = []
    index = 1
    while remaining and len(selected) < count:
        target = tuple(_halton(index, base) for base in bases)
        point = min(
            remaining,
            key=lambda item: (
                _distance(coordinates[item][: len(bases)], target),
                item.key,
            ),
        )
        remaining.remove(point)
        selected.append(point)
        index += 1
    return selected


def _maximin_order(
    remaining: set[FPMPoint],
    selected: list[FPMPoint],
    coordinates: dict[FPMPoint, tuple[float, ...]],
) -> list[FPMPoint]:
    order = []
    nearest_distance = {
        point: min(_distance(coordinates[point], coordinates[chosen]) for chosen in selected) for point in remaining
    }
    while nearest_distance:
        point = max(
            nearest_distance,
            key=lambda item: (
                nearest_distance[item],
                tuple(-value if isinstance(value, int) else value for value in item.key[1:]),
            ),
        )
        remaining.remove(point)
        nearest_distance.pop(point)
        selected.append(point)
        order.append(point)
        for candidate in nearest_distance:
            nearest_distance[candidate] = min(
                nearest_distance[candidate],
                _distance(coordinates[candidate], coordinates[point]),
            )
    return order


def _sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class SamplingDesign:
    phase: str
    population: tuple[FPMPoint, ...]
    ordered_points: tuple[FPMPoint, ...]
    roles: dict[FPMPoint, tuple[str, ...]]
    budget_counts: dict[str, int]
    active_budget: str
    sha256: str

    @property
    def selected(self) -> tuple[FPMPoint, ...]:
        return self.ordered_points[: self.budget_counts[self.active_budget]]

    @property
    def reserve(self) -> tuple[FPMPoint, ...]:
        return self.ordered_points[self.budget_counts[self.active_budget] :]

    def to_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "method": "anchors_then_halton_then_workload_maximin",
            "latency_labels_used": False,
            "population_count": len(self.population),
            "active_budget": self.active_budget,
            "budget_counts": self.budget_counts,
            "ordered_points": [
                {
                    **point.to_dict(),
                    "roles": list(self.roles.get(point, ())),
                }
                for point in self.ordered_points
            ],
            "selected_count": len(self.selected),
            "reserve_count": len(self.reserve),
            "sha256": self.sha256,
        }


def build_sampling_design(
    points: tuple[FPMPoint, ...] | list[FPMPoint],
    *,
    block_size: int,
    active_budget: str,
) -> SamplingDesign:
    """Build one phase's nested sparse design without accepting latency input."""

    unique = sorted(set(points))
    if not unique:
        raise ValueError("sampling population must not be empty")
    phases = {point.workload_kind for point in unique}
    if len(phases) != 1:
        raise ValueError(f"sampling population must contain one phase, got {sorted(phases)}")
    if active_budget not in _BUDGET_DIVISORS:
        raise ValueError(f"unknown sampling budget: {active_budget}")

    phase = unique[0].workload_kind
    anchors, raw_roles = _prefill_anchors(unique) if phase == "prefill" else _decode_anchors(unique)
    coordinates = _coordinates(unique, block_size)
    remaining = set(unique) - set(anchors)
    one_eighth_target = max(len(anchors), math.ceil(len(unique) / 8))
    halton_count = min(math.ceil(one_eighth_target / 4), len(remaining))
    halton_points = _halton_order(
        remaining,
        coordinates,
        phase=phase,
        count=halton_count,
    )
    roles = {point: set(point_roles) for point, point_roles in raw_roles.items()}
    for point in halton_points:
        roles.setdefault(point, set()).add("halton")
    ordered = [*anchors, *halton_points]
    maximin_points = _maximin_order(remaining, ordered.copy(), coordinates)
    for point in maximin_points:
        roles.setdefault(point, set()).add("maximin")
    ordered.extend(maximin_points)

    budget_counts = {
        name: max(len(anchors), math.ceil(len(unique) / divisor)) for name, divisor in _BUDGET_DIVISORS.items()
    }
    budget_counts["full"] = len(unique)
    for name in ("one_eighth", "one_quarter", "one_half", "full"):
        budget_counts[name] = min(budget_counts[name], len(unique))

    canonical = {
        "phase": phase,
        "population": [point.to_dict() for point in unique],
        "ordered": [point.to_dict() for point in ordered],
        "roles": {str(point.key): sorted(point_roles) for point, point_roles in roles.items()},
        "budget_counts": budget_counts,
        "active_budget": active_budget,
        "block_size": block_size,
    }
    return SamplingDesign(
        phase=phase,
        population=tuple(unique),
        ordered_points=tuple(ordered),
        roles={point: tuple(sorted(point_roles)) for point, point_roles in roles.items()},
        budget_counts=budget_counts,
        active_budget=active_budget,
        sha256=_sha256(canonical),
    )

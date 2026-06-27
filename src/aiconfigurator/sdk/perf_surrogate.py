# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiled curve/simplex interpolation for sparse performance tables."""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial import Delaunay, QhullError

from aiconfigurator.sdk.interpolation import InterpolationDataNotAvailableError

Direction = Literal["never", "lower", "upper", "both"]
Response = Literal["raw", "sqrt", "baseline_ratio"]
Baseline = Callable[[Mapping[str, float]], float]


@dataclass(frozen=True)
class Axis:
    """A numeric axis and its authorized exterior direction."""

    name: str
    log: bool = False
    extrapolate: Direction = "never"

    def encode(self, value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"axis {self.name!r} must be finite, got {value}")
        if self.log:
            if value <= 0:
                raise ValueError(f"axis {self.name!r} uses log2 coordinates and requires a positive value")
            return math.log2(value)
        return value

    def allows(self, query: float, boundary: float) -> bool:
        if math.isclose(query, boundary, rel_tol=0.0, abs_tol=1e-10):
            return True
        direction = "upper" if query > boundary else "lower"
        return self.extrapolate in (direction, "both")


@dataclass(frozen=True)
class _Curve:
    fixed_key: tuple[float, ...]
    sample_indices: tuple[int, ...]


@dataclass(frozen=True)
class _Support:
    indices: tuple[int, ...]
    weights: tuple[float, ...]
    exterior: bool = False


def _flatten(table: Mapping, axes: tuple[Axis, ...]) -> tuple[tuple, np.ndarray, np.ndarray]:
    rows: list[tuple[tuple[float, ...], float, float]] = []

    def visit(node, depth: int, point: tuple[float, ...]) -> None:
        if depth < len(axes):
            if not isinstance(node, Mapping):
                raise TypeError(f"expected mapping at axis {axes[depth].name!r}")
            for key, child in node.items():
                value = float(key)
                axes[depth].encode(value)
                visit(child, depth + 1, (*point, value))
            return

        if isinstance(node, Mapping):
            if "latency" not in node:
                raise ValueError(f"metric leaf at {point} has no latency")
            latency = float(node["latency"])
            energy = float(node["energy"]) if "energy" in node else float(node.get("power", 0.0)) * latency
        else:
            latency = float(node)
            energy = 0.0
        if not math.isfinite(latency) or not math.isfinite(energy) or latency < 0 or energy < 0:
            raise ValueError(f"invalid metric leaf at {point}: latency={latency}, energy={energy}")
        rows.append((point, latency, energy))

    visit(table, 0, ())
    if not rows:
        raise InterpolationDataNotAvailableError("sparse table contains no samples")
    rows.sort(key=lambda row: row[0])
    points = tuple(row[0] for row in rows)
    return points, np.asarray([row[1] for row in rows]), np.asarray([row[2] for row in rows])


class _FixedMesh:
    """Simplex support over curve keys; production tables need at most 2-D."""

    def __init__(self, axes: tuple[Axis, ...], points: tuple[tuple[float, ...], ...]) -> None:
        if len(axes) > 2:
            raise ValueError("sparse surrogate supports at most two fixed axes")
        self.axes = axes
        self.encoded = np.asarray(
            [[axis.encode(value) for axis, value in zip(axes, point, strict=True)] for point in points],
            dtype=float,
        )
        varying = np.ptp(self.encoded, axis=0) > 1e-12 if axes else np.empty(0, dtype=bool)
        self.active = np.flatnonzero(varying)
        self.inactive = np.flatnonzero(~varying)
        active_values = self.encoded[:, self.active]
        self.mean = active_values.mean(axis=0) if len(self.active) else np.empty(0)
        self.scale = active_values.std(axis=0) if len(self.active) else np.empty(0)
        self.scale[self.scale < 1e-12] = 1.0
        self.points = (active_values - self.mean) / self.scale if len(self.active) else np.empty((len(points), 0))
        self.line_order: np.ndarray | None = None
        self.tri: Delaunay | None = None

        if len(self.active) == 1:
            self.line_order = np.argsort(self.points[:, 0])
        elif len(self.active) == 2 and len(points) >= 3 and np.linalg.matrix_rank(self.points - self.points[0]) == 2:
            try:
                self.tri = Delaunay(self.points)
            except QhullError as exc:
                raise InterpolationDataNotAvailableError(f"cannot compile fixed curve mesh: {exc}") from exc

    def _query(self, values: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
        encoded = np.asarray([axis.encode(value) for axis, value in zip(self.axes, values, strict=True)], dtype=float)
        normalized = (encoded[self.active] - self.mean) / self.scale if len(self.active) else np.empty(0)
        return encoded, normalized

    def _inactive_allowed(self, encoded: np.ndarray) -> bool:
        return all(self.axes[int(index)].allows(encoded[index], self.encoded[0, index]) for index in self.inactive)

    def locate(self, values: tuple[float, ...]) -> _Support | None:
        encoded, query = self._query(values)
        if any(not math.isclose(encoded[i], self.encoded[0, i], abs_tol=1e-10) for i in self.inactive):
            return None
        dimensions = len(self.active)
        if dimensions == 0:
            return _Support((0,), (1.0,))
        if dimensions == 1:
            assert self.line_order is not None
            line = self.points[self.line_order, 0]
            position = bisect.bisect_left(line.tolist(), query[0])
            if position < len(line) and math.isclose(line[position], query[0], abs_tol=1e-10):
                return _Support((int(self.line_order[position]),), (1.0,))
            if position == 0 or position == len(line):
                return None
            left, right = position - 1, position
            weight = float((query[0] - line[left]) / (line[right] - line[left]))
            return _Support((int(self.line_order[left]), int(self.line_order[right])), (1.0 - weight, weight))

        if self.tri is None:
            return None
        simplex = int(self.tri.find_simplex(query, tol=1e-9))
        if simplex < 0:
            return None
        transform = self.tri.transform[simplex]
        barycentric = transform[:2].dot(query - transform[2])
        weights = np.append(barycentric, 1.0 - barycentric.sum())
        weights = np.clip(weights, 0.0, None)
        weights /= weights.sum()
        return _Support(
            tuple(int(index) for index in self.tri.simplices[simplex]),
            tuple(float(weight) for weight in weights),
        )

    def project(self, values: tuple[float, ...]) -> _Support | None:
        encoded, query = self._query(values)
        if not self._inactive_allowed(encoded):
            return None
        dimensions = len(self.active)
        if dimensions == 0:
            return _Support((0,), (1.0,), exterior=True)
        if dimensions == 1:
            assert self.line_order is not None
            line = self.points[self.line_order, 0]
            position = bisect.bisect_left(line.tolist(), query[0])
            if position < len(line) and math.isclose(line[position], query[0], abs_tol=1e-10):
                return _Support((int(self.line_order[position]),), (1.0,), exterior=True)
            if 0 < position < len(line):
                left, right = position - 1, position
                weight = float((query[0] - line[left]) / (line[right] - line[left]))
                return _Support(
                    (int(self.line_order[left]), int(self.line_order[right])),
                    (1.0 - weight, weight),
                    exterior=True,
                )
            endpoint = 0 if position == 0 else -1
            index = int(self.line_order[endpoint])
            axis_index = int(self.active[0])
            if not self.axes[axis_index].allows(encoded[axis_index], self.encoded[index, axis_index]):
                return None
            return _Support((index,), (1.0,), exterior=True)

        if self.tri is None:
            return None
        active_axes = tuple(self.axes[int(index)] for index in self.active)
        best: tuple[float, int, int, float] | None = None
        for left_index, right_index in self.tri.convex_hull:
            left_index = int(left_index)
            right_index = int(right_index)
            left = self.points[left_index]
            right = self.points[right_index]
            delta = right - left
            denominator = float(delta.dot(delta))
            candidates = [0.0, 1.0]
            if denominator > 0:
                candidates.append(float(np.clip((query - left).dot(delta) / denominator, 0.0, 1.0)))
            for axis_index, axis in enumerate(active_axes):
                if axis.extrapolate == "never" and abs(delta[axis_index]) > 1e-12:
                    candidates.append(float((query[axis_index] - left[axis_index]) / delta[axis_index]))
            for weight_right in candidates:
                if weight_right < -1e-10 or weight_right > 1.0 + 1e-10:
                    continue
                weight_right = float(np.clip(weight_right, 0.0, 1.0))
                candidate = left + weight_right * delta
                candidate_encoded = candidate * self.scale + self.mean
                if not all(
                    axis.allows(encoded[int(self.active[i])], candidate_encoded[i])
                    for i, axis in enumerate(active_axes)
                ):
                    continue
                distance = float(np.linalg.norm(query - candidate))
                if best is None or distance < best[0]:
                    best = (distance, left_index, right_index, weight_right)
        if best is None:
            return None
        _, left_index, right_index, weight_right = best
        if weight_right <= 1e-10:
            return _Support((left_index,), (1.0,), exterior=True)
        if weight_right >= 1.0 - 1e-10:
            return _Support((right_index,), (1.0,), exterior=True)
        return _Support((left_index, right_index), (1.0 - weight_right, weight_right), exterior=True)


class _SparseModel:
    def __init__(self, table: Mapping, axes: tuple[Axis, ...], varying: str) -> None:
        if not axes or len({axis.name for axis in axes}) != len(axes):
            raise ValueError("sparse surrogate requires unique named axes")
        names = tuple(axis.name for axis in axes)
        if varying not in names:
            raise ValueError(f"varying axis {varying!r} is not in {names}")
        self.axes = axes
        self.names = names
        self.varying_index = names.index(varying)
        self.fixed_indices = tuple(index for index in range(len(axes)) if index != self.varying_index)
        self.points, self.latencies, self.energies = _flatten(table, axes)
        self.exact = {point: index for index, point in enumerate(self.points)}
        self.powers = np.divide(
            self.energies, self.latencies, out=np.zeros_like(self.energies), where=self.latencies > 0
        )
        grouped: dict[tuple[float, ...], list[int]] = {}
        for index, point in enumerate(self.points):
            key = tuple(point[i] for i in self.fixed_indices)
            grouped.setdefault(key, []).append(index)
        self.curves = tuple(
            _Curve(
                key,
                tuple(
                    sorted(
                        indices,
                        key=lambda index: axes[self.varying_index].encode(self.points[index][self.varying_index]),
                    )
                ),
            )
            for key, indices in sorted(grouped.items())
        )
        self.curve_by_key = {curve.fixed_key: curve for curve in self.curves}
        self.mesh = _FixedMesh(
            tuple(axes[i] for i in self.fixed_indices), tuple(curve.fixed_key for curve in self.curves)
        )

    def _baseline(self, baseline: Baseline | None, point: tuple[float, ...]) -> float:
        if baseline is None:
            raise ValueError("baseline_ratio response requires a baseline")
        value = float(baseline(dict(zip(self.names, point, strict=True))))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"baseline must return a finite positive value, got {value}")
        return value

    def _blend(
        self,
        query: tuple[float, ...],
        points: tuple[tuple[float, ...], ...],
        latencies: np.ndarray,
        powers: np.ndarray,
        weights: tuple[float, ...],
        response: Response,
        baseline: Baseline | None,
    ) -> tuple[float, float]:
        weight_array = np.asarray(weights)
        if response == "raw":
            latency = float(weight_array.dot(latencies))
        elif response == "sqrt":
            latency = float(weight_array.dot(np.sqrt(latencies)) ** 2)
        elif response == "baseline_ratio":
            ratios = np.asarray(
                [latency / self._baseline(baseline, point) for latency, point in zip(latencies, points, strict=True)]
            )
            latency = float(weight_array.dot(ratios) * self._baseline(baseline, query))
        else:
            raise ValueError(f"unknown response {response!r}")
        energy = float(weight_array.dot(powers) * latency)
        if not math.isfinite(latency) or not math.isfinite(energy) or latency < 0 or energy < 0:
            raise InterpolationDataNotAvailableError("sparse surrogate produced invalid metrics")
        return latency, energy

    def _curve_estimate(
        self,
        curve: _Curve,
        query: tuple[float, ...],
        response: Response,
        exterior: Response,
        baseline: Baseline | None,
    ) -> tuple[float, float, bool]:
        axis = self.axes[self.varying_index]
        indices = curve.sample_indices
        encoded = [axis.encode(self.points[index][self.varying_index]) for index in indices]
        query_encoded = axis.encode(query[self.varying_index])
        position = bisect.bisect_left(encoded, query_encoded)
        is_exterior = (position == 0 and query_encoded < encoded[0]) or position == len(encoded)
        if is_exterior:
            endpoint = 0 if query_encoded < encoded[0] else -1
            sample_index = indices[endpoint]
            boundary = self.points[sample_index][self.varying_index]
            if not axis.allows(query[self.varying_index], boundary):
                raise InterpolationDataNotAvailableError(
                    f"axis {axis.name!r} cannot extrapolate from {boundary} to {query[self.varying_index]}"
                )
            sample_indices = (sample_index,)
            weights = (1.0,)
            response = exterior
        elif position < len(encoded) and math.isclose(encoded[position], query_encoded, abs_tol=1e-12):
            sample_indices = (indices[position],)
            weights = (1.0,)
        else:
            left, right = position - 1, position
            weight_right = (query_encoded - encoded[left]) / (encoded[right] - encoded[left])
            sample_indices = (indices[left], indices[right])
            weights = (1.0 - weight_right, weight_right)
        sample_points = tuple(self.points[index] for index in sample_indices)
        latency, energy = self._blend(
            query,
            sample_points,
            self.latencies[list(sample_indices)],
            self.powers[list(sample_indices)],
            weights,
            response,
            baseline,
        )
        return latency, energy, is_exterior

    def estimate(
        self,
        query: Mapping[str, float],
        *,
        curve: Response,
        mesh: Response,
        exterior: Response,
        baseline: Baseline | None,
    ) -> tuple[float, float]:
        if set(query) != set(self.names):
            raise ValueError(f"query axes must be exactly {self.names}")
        point = tuple(float(query[name]) for name in self.names)
        for axis, value in zip(self.axes, point, strict=True):
            axis.encode(value)
        exact_index = self.exact.get(point)
        if exact_index is not None:
            return float(self.latencies[exact_index]), float(self.energies[exact_index])
        fixed_key = tuple(point[index] for index in self.fixed_indices)
        exact_curve = self.curve_by_key.get(fixed_key)
        if exact_curve is not None:
            latency, energy, _ = self._curve_estimate(exact_curve, point, curve, exterior, baseline)
            return latency, energy

        support = self.mesh.locate(fixed_key)
        if support is None:
            support = self.mesh.project(fixed_key)
        if support is None:
            raise InterpolationDataNotAvailableError(
                f"fixed-axis point {fixed_key} lies outside authorized sparse support"
            )

        component_points: list[tuple[float, ...]] = []
        component_latency: list[float] = []
        component_power: list[float] = []
        component_exterior = False
        for curve_index in support.indices:
            selected_curve = self.curves[curve_index]
            component_query = list(point)
            for axis_index, value in zip(self.fixed_indices, selected_curve.fixed_key, strict=True):
                component_query[axis_index] = value
            component_point = tuple(component_query)
            latency, energy, curve_exterior = self._curve_estimate(
                selected_curve, component_point, curve, exterior, baseline
            )
            component_points.append(component_point)
            component_latency.append(latency)
            component_power.append(0.0 if latency == 0 else energy / latency)
            component_exterior |= curve_exterior
        response = exterior if support.exterior or component_exterior else mesh
        return self._blend(
            point,
            tuple(component_points),
            np.asarray(component_latency),
            np.asarray(component_power),
            support.weights,
            response,
            baseline,
        )


def estimate_sparse(
    database: object,
    key: Hashable,
    table: Mapping,
    query: Mapping[str, float],
    *,
    axes: tuple[Axis, ...],
    varying: str,
    curve: Response = "raw",
    mesh: Response = "raw",
    exterior: Response = "baseline_ratio",
    baseline: Baseline | None = None,
) -> tuple[float, float]:
    """Estimate ``(latency, energy)``; cache geometry by key and table identity."""

    cache = getattr(database, "_sparse_surrogate_cache", None)
    if cache is None:
        cache = {}
        database._sparse_surrogate_cache = cache
    cache_key = (key, axes, varying)
    cached = cache.get(cache_key)
    if cached is None or cached[0] is not table:
        cached = (table, _SparseModel(table, axes, varying))
        cache[cache_key] = cached
    return cached[1].estimate(query, curve=curve, mesh=mesh, exterior=exterior, baseline=baseline)

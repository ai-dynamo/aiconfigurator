# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``TableQuery`` — reusable interp/extrap orchestrator for an op's perf table.

One interface for every 3-axis op (GEMM ``m,n,k``; attention ``n,s,b``; …); the op
hands over its raw nested table ``data[x][y][z]`` (leaf = float or
``{"latency","power","energy"}``) plus two per-op knobs:

- **interpolation space** — ``value_transform``: ``"raw"`` (interpolate latency
  directly; GEMM ``m`` is near-linear) or ``"sqrt"`` (interpolate ``sqrt(latency)``;
  linearises attention's ``seq_len ~ s²`` so even large collection gaps interpolate
  well — regime-agnostic, unlike a SOL-based transform).
- **extrapolation** — ``sol_fn`` + ``extrap_axes``: past the collected range on a
  declared axis, **util-hold** — snap those axes to the nearest collected point and
  let the analytic SOL carry the growth: ``latency = anchor · SOL(query)/SOL(anchor)``.
  A miss on a NON-declared axis stays a genuine miss (interior raises).

Numeric interpolation is delegated to ``sdk.interpolation``; this module is the
policy/orchestration layer.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable


@runtime_checkable
class PerfSurrogate(Protocol):
    """The query seam ops depend on: approximate one op's metric at a shape.

    ``TableQuery`` is today's interpolation-backed implementation. The point of
    routing every op through this single interface (rather than scattered
    ``interp_3d`` calls) is that a different engine — a better numerical method
    or a learned model — can drop in behind the same ``query`` without touching
    the ops. A leaf is a float or a ``{"latency", "power", "energy"}`` dict.
    """

    def query(self, x, y, z): ...


# id-keyed cache (identity-checked) for the sqrt view, so per-query TableQuery
# construction stays cheap.
_SQRT_VIEW_CACHE: dict[int, tuple] = {}


def _sqrt_leaf(leaf):
    if isinstance(leaf, dict):
        return {k: (math.sqrt(v) if isinstance(v, (int, float)) and v > 0 else 0.0) for k, v in leaf.items()}
    return math.sqrt(leaf) if leaf > 0 else 0.0


def _square_leaf(leaf):
    if isinstance(leaf, dict):
        return {k: (v * v if isinstance(v, (int, float)) else v) for k, v in leaf.items()}
    return leaf * leaf


def _sqrt_view(data):
    cached = _SQRT_VIEW_CACHE.get(id(data))
    if cached is not None and cached[0] is data:
        return cached[1]
    view = {x: {y: {z: _sqrt_leaf(leaf) for z, leaf in yd.items()} for y, yd in xd.items()} for x, xd in data.items()}
    _SQRT_VIEW_CACHE[id(data)] = (data, view)
    return view


class TableQuery:
    """Reusable interp/extrap over ``data[x][y][z] -> leaf`` (float or metrics dict)."""

    def __init__(
        self,
        data,
        *,
        method="cubic",
        value_transform="raw",
        sol_fn=None,
        extrap_axes=(0,),
        allow_singleton_axes=False,
        extracted_metrics_cache=None,
    ):
        self.data = data
        self.method = method
        self.value_transform = value_transform
        self.sol_fn = sol_fn
        # Axis indices util-hold may extrapolate (match what the op collected /
        # used to pre-expand): GEMM -> (0,)=m; attention -> (0,1,2)=heads,seq,batch.
        # An out-of-range axis NOT listed here stays a genuine miss.
        self.extrap_axes = tuple(extrap_axes)
        self.allow_singleton_axes = allow_singleton_axes
        self.cache = extracted_metrics_cache if extracted_metrics_cache is not None else {}

    def _interior(self, x, y, z):
        """Interior interpolation in the configured value space (raises if out of hull)."""
        from aiconfigurator.sdk import interpolation

        if self.value_transform == "sqrt":
            sp = _sqrt_view(self.data)
            if x in sp and y in sp[x] and z in sp[x][y]:
                leaf = sp[x][y][z]
            else:
                leaf = interpolation.interp_3d(
                    x, y, z, sp, self.method, {}, allow_singleton_axes=self.allow_singleton_axes
                )
            return _square_leaf(leaf)

        data = self.data
        if x in data and y in data[x] and z in data[x][y]:
            return data[x][y][z]
        return interpolation.interp_3d(
            x, y, z, data, self.method, self.cache, allow_singleton_axes=self.allow_singleton_axes
        )

    def query(self, x, y, z):
        data = self.data
        if x in data and y in data[x] and z in data[x][y]:
            return data[x][y][z]
        try:
            return self._interior(x, y, z)
        except ValueError:
            if self.sol_fn is None:
                raise
            return self._util_hold_extrap(x, y, z)

    def _snap_anchor(self, coords):
        """Snap the declared extrap axes to the nearest collected key along the
        nested path (so the anchor sits on real data, per-slice). Non-extrap axes
        keep the query coord (interpolated by ``_interior``)."""
        anchor = list(coords)
        node = self.data
        for axis in range(3):
            keys = list(node.keys())
            c = coords[axis]
            if axis in self.extrap_axes:
                nearest = min(keys, key=lambda k, c=c: abs(k - c))
                anchor[axis] = nearest
                descend = nearest
            else:
                descend = c if c in node else min(keys, key=lambda k, c=c: abs(k - c))
            node = node[descend]
        return tuple(anchor)

    def _util_hold_extrap(self, x, y, z):
        """Extrapolate the declared extrap axes via util-hold; re-raise if a
        non-extrap axis is the genuine miss."""
        from aiconfigurator.sdk import interpolation

        anchor = self._snap_anchor((x, y, z))
        base = self._interior(*anchor)  # interpolates non-extrap axes; raises on a genuine miss
        base_lat = interpolation.get_value(base, "latency")
        if base_lat <= 0:
            return base
        sol_anchor = self.sol_fn(*anchor)
        if sol_anchor <= 0:
            return base
        latency = base_lat * self.sol_fn(x, y, z) / sol_anchor
        return {**base, "latency": latency} if isinstance(base, dict) else latency


if __name__ == "__main__":
    # Tiny demo over a 3-D grid: latency ~ x·y²·z (so the y axis is quadratic).
    data = {
        x: {y: {z: {"latency": 1e-9 * x * y * y * z, "energy": 0.0} for z in (1, 2)} for y in (256, 512, 1024)}
        for x in (8, 16, 32)
    }

    def sol(x, y, z):
        return 1e-9 * x * y * y * z

    raw_tq = TableQuery(data, sol_fn=sol, extrap_axes=(1,))
    sqrt_tq = TableQuery(data, value_transform="sqrt", sol_fn=sol, extrap_axes=(1,))
    truth768, truth2048 = 1e-9 * 16 * 768 * 768, 1e-9 * 16 * 2048 * 2048
    print("exact  (16,512, 1):     ", raw_tq.query(16, 512, 1)["latency"])
    print("interp (16,768, 1) raw: ", round(raw_tq.query(16, 768, 1)["latency"], 6), " truth", round(truth768, 6))
    print("interp (16,768, 1) sqrt:", round(sqrt_tq.query(16, 768, 1)["latency"], 6), " truth", round(truth768, 6))
    print("extrap (16,2048,1) hold:", round(sqrt_tq.query(16, 2048, 1)["latency"], 6), " truth", round(truth2048, 6))

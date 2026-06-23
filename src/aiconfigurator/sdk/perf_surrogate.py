# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``TableQuery`` — reusable interp/extrap orchestrator for an op's perf table.

An op hands ``TableQuery`` its raw nested table ``data[x][y][z]`` (leaf = float or
``{"latency", "power", "energy"}``) plus a few knobs (``sol_fn``, interior
``method``). ``TableQuery`` owns the source-precedence resolution and returns the
metric leaf/dict:

1. exact hit
2. 1-D along the OUTER axis at an exact ``(y, z)`` slice — interpolate raw, but
   EXTRAPOLATE via util-hold (``latency = SOL(x,y,z) / util_boundary``) when a
   ``sol_fn`` is given (else raw linear extrapolation, the legacy behaviour).
3. full 3-D interior interpolation (delegated to ``interpolation.interp_3d``).

The same call shape works for any 3-axis op (GEMM ``m,n,k``; attention ``n,s,b``;
…) — pass that op's table + its SOL formula + interior method. SOL is the op's
single ``max(compute, mem)`` value; util is used for extrapolation only (the SOL
cancels in interpolation, so interpolation stays raw).

Numeric interpolation is delegated to ``sdk.interpolation`` primitives; this
module is just the policy/orchestration layer.
"""

from __future__ import annotations


class TableQuery:
    """Reusable interp/extrap over ``data[x][y][z] -> leaf`` (float or metrics dict).

    ``sol_fn(x, y, z) -> float`` enables util-hold extrapolation along the outer
    axis; without it, that path falls back to raw linear extrapolation. ``method``
    is the interior 3-D method (kept per op, e.g. ``"cubic"``).
    """

    def __init__(self, data, *, sol_fn=None, method="cubic", extracted_metrics_cache=None):
        self.data = data
        self.sol_fn = sol_fn
        self.method = method
        self.cache = extracted_metrics_cache if extracted_metrics_cache is not None else {}

    def query(self, x, y, z):
        from aiconfigurator.sdk import interpolation

        data = self.data
        if x in data and y in data[x] and z in data[x][y]:
            return data[x][y][z]

        x_values = sorted(xv for xv in data if y in data[xv] and z in data[xv][y])
        if len(x_values) >= 2:
            x_left, x_right = interpolation.nearest_1d_point_helper(x, x_values, inner_only=False)
            left, right = data[x_left][y][z], data[x_right][y][z]
            result = interpolation.interp_1d([x_left, x_right], [left, right], x)
            extrapolating = x < x_values[0] or x > x_values[-1]
            if self.sol_fn is not None and extrapolating:
                # util-hold: hold the boundary efficiency, let SOL carry the growth.
                x_bnd = x_values[0] if x < x_values[0] else x_values[-1]
                lat_bnd = interpolation.get_value(data[x_bnd][y][z], "latency")
                if lat_bnd > 0:
                    util_bnd = self.sol_fn(x_bnd, y, z) / lat_bnd
                    if util_bnd > 0:
                        latency = self.sol_fn(x, y, z) / util_bnd
                        result = {**result, "latency": latency} if isinstance(result, dict) else latency
            return result

        return interpolation.interp_3d(x, y, z, data, self.method, self.cache)


if __name__ == "__main__":
    # Tiny self-contained demo: latency ~ x*y*z, SOL ~ x*y*z (so util ~ const).
    data = {m: {8: {16: {"latency": 1e-9 * m * 8 * 16, "energy": 0.0}}} for m in (256, 512, 1024)}

    def sol(x, y, z):
        return 1e-9 * x * y * z

    tq = TableQuery(data, sol_fn=sol)
    print("exact  (512, 8,16):", tq.query(512, 8, 16))  # exact hit
    print("interp (384, 8,16):", tq.query(384, 8, 16))  # 1-D interp along x
    print("extrap (2048,8,16):", tq.query(2048, 8, 16))  # util-hold extrapolation

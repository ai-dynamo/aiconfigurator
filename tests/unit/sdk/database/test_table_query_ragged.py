# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression: TableQuery tolerates corner-truncated (ragged) perf grids.

Real perf tables are corner-truncated, not complete Cartesian grids: large
seq_len x large batch is omitted (OOM / cost-bound collection), so a per-sub-key
``(heads, seq, batch)`` grid is ~75% filled with a staircase frontier. A query
past that frontier makes the legacy ``interp_3d`` raise; TableQuery's util-hold
extrapolation (``extrap_axes`` over every axis + ``sol_fn``) rescues it with a
finite, SOL-anchored value. These tests lock in that ragged tolerance — verified
on real h100_sxm/sglang data (1705 ragged queries, 0 crashes) and reproduced here
on a deterministic synthetic staircase.
"""

import math

import pytest

from aiconfigurator.sdk import interpolation
from aiconfigurator.sdk.perf_surrogate import PerfSurrogate, TableQuery

pytestmark = pytest.mark.unit


def test_table_query_conforms_to_perf_surrogate_protocol():
    # The swap seam: TableQuery is one PerfSurrogate; a learned engine could
    # drop in behind the same interface.
    assert isinstance(TableQuery({}, sol_fn=lambda x, y, z: 1.0), PerfSurrogate)


# Corner-truncated staircase: the larger the seq_len, the fewer batches collected.
_PRESENT = {512: [1, 2, 4, 8], 1024: [1, 2, 4, 8], 2048: [1, 2, 4], 4096: [1, 2]}
_HEADS = [8, 16]


def _lat(n, s, b):
    return 1e-6 * n * b * s * s  # attention-like: ~ heads * batch * seq^2


def _grid():
    return {
        n: {s: {b: {"latency": _lat(n, s, b), "energy": 0.0} for b in bs} for s, bs in _PRESENT.items()} for n in _HEADS
    }


# Axis order is (num_heads, full_s, batch); the SOL matches the latency model so
# util-hold is exact when the boundary efficiency carries.
def _sol(x, y, z):
    return 1e-6 * x * z * y * y


def _surrogate():
    return TableQuery(_grid(), method="linear", value_transform="sqrt", sol_fn=_sol, extrap_axes=(0, 1, 2))


def test_legacy_interp_3d_raises_in_corner_hole():
    # batch=8 is past the collected frontier (batch<=2) at seq=4096.
    with pytest.raises(ValueError):
        interpolation.interp_3d(8, 4096, 8, _grid(), "linear", {})


def test_table_query_rescues_corner_hole():
    lat = interpolation.get_value(_surrogate().query(8, 4096, 8), "latency")
    assert math.isfinite(lat) and lat > 0


def test_table_query_no_crash_across_all_holes():
    tq = _surrogate()
    holes = [(n, s, b) for n in _HEADS for s in _PRESENT for b in (1, 2, 4, 8) if b not in _PRESENT[s]]
    assert holes, "fixture should contain ragged holes"
    for n, s, b in holes:
        lat = interpolation.get_value(tq.query(n, s, b), "latency")
        assert math.isfinite(lat) and lat > 0, f"corner hole ({n},{s},{b}) -> {lat}"

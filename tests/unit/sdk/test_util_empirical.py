# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the util-space empirical core (per-component SOL split).

Locks the contract that makes the compute/mem SOL split safe to roll out:
- a scalar SOL reduces EXACTLY to the legacy ``SOL / util`` (no regression);
- a ``(sol_compute, sol_mem)`` tuple reconstructs ``max_i(sol_i / util_i)`` so the
  binding roofline wins -- the point of the split is correctness across the
  binding regime;
- a missing grid raises instead of fabricating a value.
"""

import math

import pytest

from aiconfigurator.sdk.errors import EmpiricalNotImplementedError
from aiconfigurator.sdk.operations import util_empirical as ue


def _grid_from(samples):
    return ue.UtilGrid([ue.UtilSample(tuple(float(c) for c in coords), tuple(utils)) for coords, utils in samples])


def test_as_components_scalar_and_tuple():
    assert ue._as_components(2.0) == (2.0,)
    assert ue._as_components((3.0, 4.0)) == (3.0, 4.0)


def test_estimate_scalar_reduces_to_legacy():
    # one collected point: coord (16,), util 0.5 -> latency = sol/util
    grid = _grid_from([((16.0,), (0.5,))])
    lat, util = ue.estimate(2.0, (16.0,), grid)
    assert math.isclose(lat, 2.0 / 0.5)
    assert util == (0.5,)


def test_estimate_two_component_takes_binding_max():
    # util_compute=0.5, util_mem=0.25; query SOL (compute=1.0, mem=1.0)
    # -> compute cand 1/0.5=2.0, mem cand 1/0.25=4.0 -> max picks mem (binding)
    grid = _grid_from([((16.0,), (0.5, 0.25))])
    lat, util = ue.estimate((1.0, 1.0), (16.0,), grid)
    assert math.isclose(lat, 4.0)
    assert util == (0.5, 0.25)


def test_estimate_two_component_zero_sol_skips_that_bound():
    # mem bound has zero SOL for this query (no mem traffic) -> only compute counts
    grid = _grid_from([((16.0,), (0.5, 0.25))])
    lat, _ = ue.estimate((1.0, 0.0), (16.0,), grid)
    assert math.isclose(lat, 1.0 / 0.5)


def test_estimate_util_scale_divides():
    grid = _grid_from([((16.0,), (0.5,))])
    lat, _ = ue.estimate(2.0, (16.0,), grid, util_scale=2.0)
    assert math.isclose(lat, 2.0 / (0.5 * 2.0))


def test_estimate_raises_without_grid():
    with pytest.raises(EmpiricalNotImplementedError):
        ue.estimate(2.0, (16.0,), None)
    with pytest.raises(EmpiricalNotImplementedError):
        ue.estimate(2.0, (16.0,), ue.UtilGrid([]))


def test_build_samples_round_trip_recovers_measured():
    # node: coords (n)->{lat}; sol_fn returns (sol_compute, sol_mem). At a collected
    # point, estimate must recover the measured latency exactly (both components).
    node = {16: {"latency": 4.0}, 32: {"latency": 8.0}}

    def sol_fn(c):  # c=(n,); compute-bound here (sol_compute > sol_mem)
        return (2.0 * c[0] / 16.0, 0.5 * c[0] / 16.0)

    grid = ue.UtilGrid(ue.build_samples(node, depth=1, sol_fn=sol_fn))
    for n, lat_true in [(16, 4.0), (32, 8.0)]:
        lat, _ = ue.estimate(sol_fn((n,)), (float(n),), grid)
        assert math.isclose(lat, lat_true, rel_tol=1e-9)

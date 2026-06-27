# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from aiconfigurator.sdk.interpolation import InterpolationDataNotAvailableError
from aiconfigurator.sdk.perf_surrogate import Axis, estimate_sparse


def _m(latency: float, power: float = 2.0) -> dict[str, float]:
    return {"latency": latency, "power": power, "energy": latency * power}


def _estimate(table, query, axes, varying, **kwargs):
    database = kwargs.pop("database", SimpleNamespace())
    key = kwargs.pop("key", "test")
    return estimate_sparse(database, key, table, query, axes=axes, varying=varying, **kwargs)


def test_exact_and_power_only_leaves_preserve_energy() -> None:
    axes = (Axis("batch"), Axis("tokens"))
    assert _estimate({1: {8: _m(3, 4)}}, {"batch": 1, "tokens": 8}, axes, "tokens") == (3, 12)
    assert _estimate({1: {"latency": 2.5, "power": 6}}, {"tokens": 1}, (Axis("tokens"),), "tokens") == (2.5, 15)


@pytest.mark.parametrize(
    ("table", "axis", "response", "expected"),
    [
        ({1: _m(2, 4), 3: _m(6, 8)}, Axis("x"), "raw", (4, 24)),
        ({1: _m(1), 3: _m(9)}, Axis("x"), "sqrt", (4, 8)),
        ({1: _m(2), 4: _m(8)}, Axis("x", log=True), "raw", (5, 10)),
    ],
)
def test_curve_coordinate_and_response(table, axis, response, expected) -> None:
    assert _estimate(table, {"x": 2}, (axis,), "x", curve=response) == pytest.approx(expected)


def test_curve_exterior_uses_baseline_ratio_and_direction() -> None:
    table = {10: _m(4), 20: _m(7)}
    axis = (Axis("tokens", extrapolate="upper"),)
    baseline = lambda point: point["tokens"]
    assert _estimate(table, {"tokens": 40}, axis, "tokens", baseline=baseline) == pytest.approx((14, 28))
    with pytest.raises(InterpolationDataNotAvailableError, match="cannot extrapolate"):
        _estimate(table, {"tokens": 5}, axis, "tokens", baseline=baseline)
    assert _estimate(table, {"tokens": 40}, axis, "tokens", exterior="raw") == (7, 14)


def test_baseline_ratio_supports_zero_latency() -> None:
    latency, energy = _estimate(
        {1: _m(0, 0), 3: _m(4)},
        {"tokens": 2},
        (Axis("tokens"),),
        "tokens",
        curve="baseline_ratio",
        baseline=lambda point: point["tokens"],
    )
    assert (latency, energy) == pytest.approx((4 / 3, 4 / 3))


def test_fixed_line_mesh_interpolates_real_curves() -> None:
    table = {1: {10: _m(2), 20: _m(4)}, 3: {10: _m(6), 20: _m(8)}}
    result = _estimate(table, {"batch": 2, "tokens": 15}, (Axis("batch"), Axis("tokens")), "tokens")
    assert result == pytest.approx((5, 10))


def _triangle_table():
    return {
        0: {0: {10: _m(2), 20: _m(3)}, 2: {10: _m(6), 40: _m(9)}},
        2: {0: {10: _m(4), 30: _m(6)}},
    }


def test_ragged_triangle_fills_interior_without_cartesian_product() -> None:
    result = _estimate(
        _triangle_table(),
        {"x": 0.5, "y": 0.5, "tokens": 15},
        (Axis("x"), Axis("y"), Axis("tokens")),
        "tokens",
    )
    assert result == pytest.approx((4, 8))


def test_triangle_bounding_box_requires_authorized_extrapolation() -> None:
    query = {"x": 1.5, "y": 1.5, "tokens": 15}
    forbidden = (Axis("x"), Axis("y"), Axis("tokens"))
    with pytest.raises(InterpolationDataNotAvailableError, match="outside authorized"):
        _estimate(_triangle_table(), query, forbidden, "tokens")

    allowed = (Axis("x", extrapolate="upper"), Axis("y", extrapolate="upper"), Axis("tokens"))
    latency, _ = _estimate(_triangle_table(), query, allowed, "tokens", exterior="raw")
    assert latency == pytest.approx(5.5)
    with pytest.raises(InterpolationDataNotAvailableError):
        _estimate(
            _triangle_table(),
            {"x": -1, "y": -1, "tokens": 15},
            allowed,
            "tokens",
            exterior="raw",
        )


def test_hull_projection_can_keep_never_axis_fixed() -> None:
    axes = (Axis("x"), Axis("y", extrapolate="upper"), Axis("tokens"))
    latency, _ = _estimate(_triangle_table(), {"x": 1, "y": 2, "tokens": 15}, axes, "tokens", exterior="raw")
    assert latency == pytest.approx(5.5)


def test_mesh_baseline_ratio_uses_component_baselines() -> None:
    table = {1: {10: _m(4), 20: _m(8)}, 3: {10: _m(8), 20: _m(16)}}
    latency, _ = _estimate(
        table,
        {"batch": 2, "tokens": 15},
        (Axis("batch"), Axis("tokens")),
        "tokens",
        mesh="baseline_ratio",
        baseline=lambda point: point["batch"] * point["tokens"],
    )
    assert latency == pytest.approx(((4 / 10 + 8 / 30) / 2) * 30)


def test_constant_fixed_axis_only_moves_when_authorized() -> None:
    table = {2: {10: _m(2), 20: _m(4)}}
    query = {"batch": 3, "tokens": 15}
    with pytest.raises(InterpolationDataNotAvailableError):
        _estimate(table, query, (Axis("batch"), Axis("tokens")), "tokens", exterior="raw")
    result = _estimate(table, query, (Axis("batch", extrapolate="upper"), Axis("tokens")), "tokens", exterior="raw")
    assert result == pytest.approx((3, 6))


def test_constant_fixed_axis_can_move_while_active_axis_interpolates() -> None:
    table = {0: {1: {10: _m(10)}, 4: {10: _m(40)}}}
    result = _estimate(
        table,
        {"prefix": 8, "batch": 2, "tokens": 10},
        (Axis("prefix", extrapolate="both"), Axis("batch"), Axis("tokens")),
        "tokens",
        exterior="raw",
    )
    assert result == pytest.approx((20, 40))


def test_table_identity_invalidates_cached_geometry() -> None:
    database = SimpleNamespace()
    axes = (Axis("tokens"),)
    first = _estimate({1: _m(1), 3: _m(3)}, {"tokens": 2}, axes, "tokens", database=database)
    second = _estimate({1: _m(10), 3: _m(30)}, {"tokens": 2}, axes, "tokens", database=database)
    assert first == (2, 4)
    assert second == (20, 40)
    assert len(database._sparse_surrogate_cache) == 1


def test_rank_deficient_mesh_still_allows_exact_curve() -> None:
    table = {0: {0: {1: _m(1)}}, 1: {1: {1: _m(2)}}, 2: {2: {1: _m(3)}}}
    axes = (Axis("x"), Axis("y"), Axis("tokens"))
    assert _estimate(table, {"x": 1, "y": 1, "tokens": 1}, axes, "tokens") == (2, 4)
    with pytest.raises(InterpolationDataNotAvailableError):
        _estimate(table, {"x": 1, "y": 1.5, "tokens": 1}, axes, "tokens")


@pytest.mark.parametrize(
    ("table", "query", "axes", "varying", "message"),
    [
        ({1: _m(1)}, {"wrong": 1}, (Axis("tokens"),), "tokens", "query axes"),
        ({1: _m(1)}, {"tokens": 1}, (Axis("tokens"),), "batch", "varying axis"),
        (
            {1: {1: {1: {1: _m(1)}}}},
            {"a": 1, "b": 1, "c": 1, "t": 1},
            (Axis("a"), Axis("b"), Axis("c"), Axis("t")),
            "t",
            "at most two fixed axes",
        ),
    ],
)
def test_configuration_errors_fail_loudly(table, query, axes, varying, message) -> None:
    with pytest.raises(ValueError, match=message):
        _estimate(table, query, axes, varying)


def test_invalid_log_baseline_and_table_fail_loudly() -> None:
    with pytest.raises(ValueError, match="positive"):
        _estimate({1: _m(1)}, {"x": 0}, (Axis("x", log=True),), "x")
    with pytest.raises(ValueError, match="requires a baseline"):
        _estimate({1: _m(1), 2: _m(2)}, {"x": 3}, (Axis("x", extrapolate="upper"),), "x")
    with pytest.raises(InterpolationDataNotAvailableError, match="no samples"):
        _estimate({}, {"x": 1}, (Axis("x"),), "x")
    with pytest.raises(ValueError, match="no latency"):
        _estimate({1: {"power": 2}}, {"x": 1}, (Axis("x"),), "x")

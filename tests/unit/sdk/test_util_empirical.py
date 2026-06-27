# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from aiconfigurator.sdk.operations.util_empirical import (
    UtilGrid,
    UtilSample,
    bracketed_2d_util,
    estimate_bracketed_2d,
)

pytestmark = pytest.mark.unit


def test_empty_grid_has_no_util():
    assert UtilGrid([]).util((1.0,)) is None


def test_1d_exact_hit_preserves_measured_util_with_unsorted_samples():
    grid = UtilGrid(
        [
            UtilSample((16.0,), 0.8),
            UtilSample((8.0,), 0.2),
            UtilSample((9.0,), 0.4),
        ]
    )

    assert grid.util((9.0,)) == pytest.approx(0.4)


def test_1d_uses_bracketing_samples_with_log_space_idw():
    # The two globally nearest samples to 11 are 9 and 8 in log space. The
    # interpolation contract must instead bracket the query with 9 and 16.
    grid = UtilGrid(
        [
            UtilSample((16.0,), 0.8),
            UtilSample((8.0,), 0.2),
            UtilSample((9.0,), 0.4),
        ]
    )
    alpha = (math.log(11.0) - math.log(9.0)) / (math.log(16.0) - math.log(9.0))
    expected = 0.4 + alpha * (0.8 - 0.4)

    assert grid.util((11.0,)) == pytest.approx(expected)


def test_1d_extrapolation_freezes_boundary_util():
    grid = UtilGrid(
        [
            UtilSample((16.0,), 0.8),
            UtilSample((8.0,), 0.2),
        ]
    )

    assert grid.util((1.0,)) == pytest.approx(0.2)
    assert grid.util((128.0,)) == pytest.approx(0.8)


def test_1d_singleton_and_constant_log_coordinate_return_only_util():
    singleton = UtilGrid([UtilSample((0.0,), 0.3)])
    repeated_coordinate = UtilGrid(
        [
            UtilSample((4.0,), 0.6),
            UtilSample((4.0,), 0.7),
            UtilSample((16.0,), 0.9),
        ]
    )
    alpha = (math.log(8.0) - math.log(4.0)) / (math.log(16.0) - math.log(4.0))

    assert singleton.util((100.0,)) == pytest.approx(0.3)
    assert repeated_coordinate.util((4.0,)) == pytest.approx(0.6)
    assert repeated_coordinate.util((8.0,)) == pytest.approx(0.6 + alpha * (0.9 - 0.6))


def test_multidimensional_grid_retains_single_nearest_neighbour():
    grid = UtilGrid(
        [
            UtilSample((1.0, 100.0), 0.1),
            UtilSample((10.0, 10.0), 0.5),
            UtilSample((100.0, 1.0), 0.9),
        ]
    )

    assert grid.util((9.0, 9.0)) == pytest.approx(0.5)


def test_bracketed_2d_interpolates_each_ragged_curve_then_first_axis():
    # The two batch curves intentionally have different sequence anchors: no
    # Cartesian rectangle exists at the query's neighbouring coordinates.
    samples = [
        UtilSample((2.0, 100.0), 0.2),
        UtilSample((2.0, 200.0), 0.4),
        UtilSample((4.0, 80.0), 0.4),
        UtilSample((4.0, 240.0), 0.8),
    ]

    # Backward-compatible default: physical-coordinate interpolation.
    assert bracketed_2d_util(samples, (3.0, 150.0)) == pytest.approx(0.4375)


def test_bracketed_2d_can_use_log_coordinates_explicitly():
    samples = [
        UtilSample((2.0, 100.0), 0.2),
        UtilSample((2.0, 200.0), 0.4),
        UtilSample((4.0, 80.0), 0.4),
        UtilSample((4.0, 240.0), 0.8),
    ]

    seq_alpha_2 = math.log(150.0 / 100.0) / math.log(200.0 / 100.0)
    seq_alpha_4 = math.log(150.0 / 80.0) / math.log(240.0 / 80.0)
    batch_alpha = math.log(3.0 / 2.0) / math.log(4.0 / 2.0)
    batch_2_util = 0.2 + seq_alpha_2 * (0.4 - 0.2)
    batch_4_util = 0.4 + seq_alpha_4 * (0.8 - 0.4)
    expected = batch_2_util + batch_alpha * (batch_4_util - batch_2_util)

    assert bracketed_2d_util(samples, (3.0, 150.0), log_space=True) == pytest.approx(expected)


def test_bracketed_2d_default_freezes_each_axis_boundary_util():
    samples = [
        UtilSample((2.0, 100.0), 0.2),
        UtilSample((2.0, 200.0), 0.4),
        UtilSample((4.0, 80.0), 0.5),
        UtilSample((4.0, 240.0), 0.9),
    ]

    assert bracketed_2d_util(samples, (1.0, 1000.0)) == pytest.approx(0.4)
    assert bracketed_2d_util(samples, (10.0, 1.0)) == pytest.approx(0.5)


def test_bracketed_2d_excludes_curves_that_do_not_cover_sequence():
    samples = [
        UtilSample((1.0, 100.0), 0.1),
        UtilSample((1.0, 200.0), 0.3),
        UtilSample((2.0, 100.0), 0.2),
        UtilSample((2.0, 200.0), 0.4),
        # This nominal upper curve stopped before sequence=150. It must not
        # bracket batch=3; the largest eligible lower curve (batch=2) wins.
        UtilSample((4.0, 80.0), 0.5),
        UtilSample((4.0, 120.0), 0.9),
    ]
    seq_alpha = math.log(150.0 / 100.0) / math.log(200.0 / 100.0)
    expected = 0.2 + seq_alpha * (0.4 - 0.2)

    assert bracketed_2d_util(
        samples,
        (3.0, 150.0),
        log_space=True,
        require_y_coverage=True,
    ) == pytest.approx(expected)


def test_bracketed_2d_returns_none_when_no_curve_covers_sequence():
    samples = [
        UtilSample((2.0, 100.0), 0.2),
        UtilSample((2.0, 200.0), 0.4),
        UtilSample((4.0, 80.0), 0.5),
        UtilSample((4.0, 240.0), 0.9),
    ]

    assert bracketed_2d_util(samples, (3.0, 1000.0), require_y_coverage=True) is None


def test_bracketed_2d_miss_returns_none_without_changing_default_lookup():
    one_dimensional = UtilGrid([UtilSample((2.0,), 0.25)])

    assert bracketed_2d_util(one_dimensional.samples, (2.0, 100.0)) is None
    assert estimate_bracketed_2d(1.0, (2.0, 100.0), one_dimensional) is None
    assert one_dimensional.util((2.0,)) == pytest.approx(0.25)


def test_estimate_bracketed_2d_records_provenance_only_on_success():
    from aiconfigurator.sdk.operations import util_empirical

    grid = UtilGrid(
        [
            UtilSample((2.0, 100.0), 0.2),
            UtilSample((2.0, 200.0), 0.4),
            UtilSample((4.0, 100.0), 0.4),
            UtilSample((4.0, 200.0), 0.8),
        ]
    )
    with util_empirical.capture_provenance() as tags:
        result = estimate_bracketed_2d(0.9, (3.0, 150.0), grid)

    assert result == pytest.approx((2.0, 0.45))
    assert tags == {"empirical"}

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from aiconfigurator.sdk.errors import PerfDataNotAvailableError
from aiconfigurator.sdk.interpolation import InterpolationDataNotAvailableError
from aiconfigurator.sdk.operations.util_empirical import (
    ReferenceCandidate,
    UtilGrid,
    UtilSample,
    capture_provenance,
    clear_grid_cache,
    estimate,
    grid_for,
    grid_from_reference,
    require_data_slice,
    worst_provenance,
)

pytestmark = pytest.mark.unit


def test_exact_singleton_duplicate_and_empty_grid_contracts():
    exact = UtilGrid(
        [
            UtilSample((16.0,), 0.8),
            UtilSample((8.0,), 0.2),
            UtilSample((9.0,), 0.4),
        ]
    )
    duplicate = UtilGrid([UtilSample((4.0,), 0.6), UtilSample((4.0,), 0.7)])

    assert exact.util((9.0,)) == pytest.approx(0.4)
    assert UtilGrid([UtilSample((0.0,), 0.3)]).util((100.0,)) == pytest.approx(0.3)
    assert duplicate.util((4.0,)) == pytest.approx(0.6)
    assert UtilGrid([]).util((1.0, 2.0)) is None


def test_1d_k2_idw_uses_nearest_samples_in_normalized_log_space():
    grid = UtilGrid(
        [
            UtilSample((16.0,), 0.8),
            UtilSample((8.0,), 0.2),
            UtilSample((9.0,), 0.4),
        ]
    )
    distance_9 = math.log(11.0) - math.log(9.0)
    distance_8 = math.log(11.0) - math.log(8.0)
    expected = (0.4 / distance_9 + 0.2 / distance_8) / (1.0 / distance_9 + 1.0 / distance_8)

    assert grid.util((11.0,)) == pytest.approx(expected)


def test_multidimensional_k2_idw_uses_nearest_samples():
    grid = UtilGrid(
        [
            UtilSample((1.0, 1.0), 0.2),
            UtilSample((100.0, 1.0), 0.6),
            UtilSample((100.0, 100.0), 1.0),
        ]
    )

    # (10, 1) is equidistant from the first two normalized-log samples.
    assert grid.util((10.0, 1.0)) == pytest.approx(0.4)


def test_1d_extrapolation_clamps_to_measured_bounds():
    grid = UtilGrid([UtilSample((8.0,), 0.2), UtilSample((16.0,), 0.8)])

    assert grid.util((1.0,)) == pytest.approx(0.2)
    assert grid.util((128.0,)) == pytest.approx(0.8)


def test_multidimensional_extrapolation_clamps_each_axis():
    grid = UtilGrid(
        [
            UtilSample((1.0, 1.0), 0.2),
            UtilSample((1.0, 10.0), 0.4),
            UtilSample((10.0, 1.0), 0.8),
        ]
    )

    # Clamping (0.1, 100) produces the exact measured boundary (1, 10).
    assert grid.util((0.1, 100.0)) == pytest.approx(0.4)


@pytest.mark.parametrize(
    "coverage_error",
    [
        PerfDataNotAvailableError("table is not loaded"),
        InterpolationDataNotAvailableError("slice cannot be interpolated"),
    ],
)
def test_grid_for_returns_none_for_typed_coverage_failures(coverage_error):
    clear_grid_cache()

    def unavailable_slice():
        raise coverage_error

    assert grid_for(("typed-coverage", type(coverage_error)), unavailable_slice, lambda _: 1.0, depth=1) is None


def test_grid_cache_isolated_by_loaded_slice_identity():
    clear_grid_cache()
    first_data = {1: 2.0}
    second_data = {1: 4.0}

    first = grid_for(("same-logical-view",), lambda: first_data, lambda _: 1.0, depth=1)
    second = grid_for(("same-logical-view",), lambda: second_data, lambda _: 1.0, depth=1)
    first_again = grid_for(("same-logical-view",), lambda: first_data, lambda _: 1.0, depth=1)

    assert first.util((1.0,)) == pytest.approx(0.5)
    assert second.util((1.0,)) == pytest.approx(0.25)
    assert first_again is first


def test_reference_grid_cache_isolated_by_selected_candidate_identity():
    clear_grid_cache()
    first_data = {1: 2.0}
    second_data = {1: 4.0}

    def candidate(node):
        return [ReferenceCandidate(features=(1.0,), node=node, sol_fn=lambda _: 1.0)]

    first = grid_from_reference(("same-reference-view",), (1.0,), lambda: candidate(first_data), depth=1)
    second = grid_from_reference(("same-reference-view",), (1.0,), lambda: candidate(second_data), depth=1)
    first_again = grid_from_reference(("same-reference-view",), (1.0,), lambda: candidate(first_data), depth=1)

    assert first.util((1.0,)) == pytest.approx(0.5)
    assert second.util((1.0,)) == pytest.approx(0.25)
    assert first_again is first


def test_reference_selection_is_policy_isolated_and_preserves_provenance():
    clear_grid_cache()
    shape_data = {1: 2.0}
    quant_data = {1: 4.0}

    shape_candidate = ReferenceCandidate(
        features=(1.0,),
        node=shape_data,
        sol_fn=lambda _: 1.0,
        provenance="xshape",
    )

    def aggressive_candidates():
        return [
            shape_candidate,
            ReferenceCandidate(
                features=(100.0,),
                node=quant_data,
                sol_fn=lambda _: 1.0,
                provenance="xquant",
            ),
        ]

    conservative = grid_from_reference(
        ("nearest-reference",),
        (90.0,),
        lambda: [shape_candidate],
        depth=1,
        selection_key=(id(shape_data), "conservative"),
    )
    aggressive = grid_from_reference(
        ("nearest-reference",),
        (90.0,),
        aggressive_candidates,
        depth=1,
        selection_key=(id(quant_data), "aggressive"),
    )

    with capture_provenance() as tags:
        conservative_result = estimate(1.0, (1.0,), conservative, provenance=conservative.reference_provenance)
        aggressive_result = estimate(1.0, (1.0,), aggressive, provenance=aggressive.reference_provenance)

    assert conservative_result == pytest.approx((2.0, 0.5))
    assert aggressive_result == pytest.approx((4.0, 0.25))
    assert tags == {"xshape", "xquant"}
    assert worst_provenance(tags) == "xquant"


def test_require_data_slice_types_only_explicit_missing_keys():
    assert require_data_slice({"quant": {"shape": {1: 2.0}}}, "quant", "shape") == {1: 2.0}

    with pytest.raises(PerfDataNotAvailableError, match="not loaded"):
        require_data_slice(None, "quant")

    with pytest.raises(PerfDataNotAvailableError, match="requested slice"):
        require_data_slice({"quant": {}}, "quant", "shape")

    with pytest.raises(PerfDataNotAvailableError, match="requested slice"):
        require_data_slice({"quant": {}}, "quant")

    with pytest.raises(TypeError, match="Malformed performance data"):
        require_data_slice({"quant": []}, "quant", "shape")


@pytest.mark.parametrize(
    "programming_error",
    [
        KeyError("unexpected schema key"),
        IndexError("unexpected schema position"),
        ValueError("invalid latency value"),
        RuntimeError("grid builder bug"),
    ],
)
def test_grid_for_propagates_programming_and_schema_errors(programming_error):
    clear_grid_cache()

    def broken_slice():
        raise programming_error

    with pytest.raises(type(programming_error)) as exc_info:
        grid_for(("programming-error", type(programming_error)), broken_slice, lambda _: 1.0, depth=1)

    assert exc_info.value is programming_error


def test_grid_from_reference_returns_none_for_typed_coverage_failure():
    clear_grid_cache()

    def unavailable_candidates():
        raise PerfDataNotAvailableError("reference slice is unavailable")

    assert grid_from_reference(("typed-reference-miss",), (1.0,), unavailable_candidates, depth=1) is None


def test_grid_from_reference_propagates_builder_errors():
    clear_grid_cache()

    def broken_candidates():
        raise TypeError("malformed reference table")

    with pytest.raises(TypeError, match="malformed reference table"):
        grid_from_reference(("broken-reference",), (1.0,), broken_candidates, depth=1)

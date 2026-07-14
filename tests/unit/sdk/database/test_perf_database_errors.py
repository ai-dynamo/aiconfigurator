# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.errors import InterpolationDataNotAvailableError
from aiconfigurator.sdk.perf_database import (
    PerfDatabase,
    PerfDataNotAvailableError,
    has_perf_data_not_available_cause,
)
from aiconfigurator.sdk.performance_result import PerformanceResult

pytestmark = pytest.mark.unit


def test_perf_data_cause_detection_accepts_direct_error() -> None:
    error = PerfDataNotAvailableError("missing perf data")

    assert has_perf_data_not_available_cause(error)


def test_perf_data_cause_detection_accepts_explicit_cause() -> None:
    error = RuntimeError("outer")
    error.__cause__ = PerfDataNotAvailableError("missing perf data")

    assert has_perf_data_not_available_cause(error)


def test_perf_data_cause_detection_accepts_implicit_context() -> None:
    error = RuntimeError("outer")
    error.__context__ = PerfDataNotAvailableError("missing perf data")

    assert has_perf_data_not_available_cause(error)


def test_perf_data_cause_detection_prefers_explicit_cause() -> None:
    error = RuntimeError("outer")
    error.__cause__ = RuntimeError("explicit cause")
    error.__context__ = PerfDataNotAvailableError("missing perf data")

    assert not has_perf_data_not_available_cause(error)


def test_perf_data_cause_detection_ignores_suppressed_context() -> None:
    try:
        try:
            raise PerfDataNotAvailableError("missing perf data")
        except PerfDataNotAvailableError:
            raise KeyError("real bug while handling perf-data miss") from None
    except KeyError as error:
        assert error.__suppress_context__
        assert not has_perf_data_not_available_cause(error)


def test_perf_data_cause_detection_avoids_cycles() -> None:
    error = RuntimeError("outer")
    nested = RuntimeError("nested")
    error.__cause__ = nested
    nested.__context__ = error

    assert not has_perf_data_not_available_cause(error)


@pytest.mark.parametrize(
    "coverage_error",
    [
        PerfDataNotAvailableError("table is not loaded"),
        InterpolationDataNotAvailableError("axis has only 1 value"),
    ],
)
def test_hybrid_falls_back_for_typed_silicon_coverage_failures(coverage_error) -> None:
    database = object.__new__(PerfDatabase)

    def unavailable_silicon():
        raise coverage_error

    result = database._query_silicon_or_hybrid(
        get_silicon=unavailable_silicon,
        get_empirical=lambda: 4.25,
        database_mode=common.DatabaseMode.HYBRID,
        error_msg="missing test data",
    )

    assert isinstance(result, PerformanceResult)
    assert float(result) == pytest.approx(4.25)
    assert result.source == "empirical"


@pytest.mark.parametrize(
    "programming_error",
    [
        KeyError("unexpected schema key"),
        IndexError("unexpected schema position"),
        ValueError("invalid latency value"),
        RuntimeError("silicon query bug"),
    ],
)
def test_hybrid_propagates_programming_and_schema_errors(programming_error) -> None:
    database = object.__new__(PerfDatabase)
    empirical_called = False

    def broken_silicon():
        raise programming_error

    def empirical():
        nonlocal empirical_called
        empirical_called = True
        return 4.25

    with pytest.raises(type(programming_error)) as exc_info:
        database._query_silicon_or_hybrid(
            get_silicon=broken_silicon,
            get_empirical=empirical,
            database_mode=common.DatabaseMode.HYBRID,
            error_msg="broken test query",
        )

    assert exc_info.value is programming_error
    assert not empirical_called


@pytest.mark.parametrize(
    "coverage_error",
    [
        PerfDataNotAvailableError("table is not loaded"),
        InterpolationDataNotAvailableError("axis has only 1 value"),
    ],
)
def test_silicon_propagates_typed_miss_without_logging_warning(coverage_error, caplog) -> None:
    database = object.__new__(PerfDatabase)

    def unavailable_silicon():
        raise coverage_error

    caplog.set_level(logging.WARNING, logger="aiconfigurator.sdk.perf_database")
    with pytest.raises(PerfDataNotAvailableError) as exc_info:
        database._query_silicon_or_hybrid(
            get_silicon=unavailable_silicon,
            get_empirical=lambda: 4.25,
            database_mode=common.DatabaseMode.SILICON,
            error_msg="missing test data",
        )

    if isinstance(coverage_error, PerfDataNotAvailableError):
        assert exc_info.value is coverage_error
    else:
        assert exc_info.value.__cause__ is coverage_error
    assert "Consider using HYBRID mode" in str(exc_info.value)
    assert not [
        record
        for record in caplog.records
        if record.name == "aiconfigurator.sdk.perf_database" and record.levelno >= logging.WARNING
    ]

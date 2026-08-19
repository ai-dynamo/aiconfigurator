# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiconfigurator.sdk.perf_database import (
    PerfDataNotAvailableError,
    has_perf_data_not_available_cause,
)

pytestmark = pytest.mark.unit

# The silicon/hybrid closure dispatcher these tests used to pin
# (``PerfDatabase._query_silicon_or_hybrid``: typed coverage misses fall back
# in HYBRID, programming errors propagate, SILICON re-raises with the HYBRID
# hint) retired to the compiled engine with #1357 PR-5; its behaviour is
# anchored by the frozen parity goldens and
# the frozen parity goldens.


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

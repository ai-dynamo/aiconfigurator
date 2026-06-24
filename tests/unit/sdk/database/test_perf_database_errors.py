# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiconfigurator.sdk.interpolation import InterpolationDataNotAvailableError
from aiconfigurator.sdk.perf_database import (
    _MISSING_SILICON_DATA_EXCEPTIONS,
    PerfDataNotAvailableError,
    has_perf_data_not_available_cause,
)

pytestmark = pytest.mark.unit


def test_perf_data_cause_detection_traverses_cause_and_context() -> None:
    error = RuntimeError("outer")
    error.__cause__ = RuntimeError("explicit cause")
    error.__context__ = PerfDataNotAvailableError("missing perf data")

    assert has_perf_data_not_available_cause(error)


def test_perf_data_cause_detection_avoids_cycles() -> None:
    error = RuntimeError("outer")
    nested = RuntimeError("nested")
    error.__cause__ = nested
    nested.__context__ = error

    assert not has_perf_data_not_available_cause(error)


def test_missing_silicon_data_exceptions_narrowed_to_interpolation_class() -> None:
    """A genuine programming bug raising plain ``ValueError`` must not be misclassified as missing data.

    The interpolation-data signal is ``InterpolationDataNotAvailableError`` (a
    ``ValueError`` subclass); the perf-DB layer should only treat that class —
    plus ``KeyError``/``IndexError`` — as "missing silicon data".
    """
    assert InterpolationDataNotAvailableError in _MISSING_SILICON_DATA_EXCEPTIONS
    assert ValueError not in _MISSING_SILICON_DATA_EXCEPTIONS

    bug = ValueError("a deeper programming bug, not missing data")
    assert not isinstance(bug, _MISSING_SILICON_DATA_EXCEPTIONS)

    missing = InterpolationDataNotAvailableError("axis has only 1 value")
    assert isinstance(missing, _MISSING_SILICON_DATA_EXCEPTIONS)

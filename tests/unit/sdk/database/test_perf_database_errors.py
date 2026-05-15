# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiconfigurator.sdk.perf_database import (
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

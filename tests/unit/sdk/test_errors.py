# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NoResultsError taxonomy and its chain-walking recognizer.

The CLI experiment handler (cli/main.py) and any downstream sweep driver rely on
two contracts verified here:

1. Every *expected* "no results" exception is-a ``NoResultsError`` (so a single
   ``isinstance`` check classifies it) AND is-a ``RuntimeError`` (so existing
   ``except RuntimeError`` / ``except Exception`` catches keep working).
2. ``is_expected_no_result_cause`` recognizes the family through the exception
   chain, so a generic wrapper raised ``from`` a family member is treated as a
   clean outcome -- while a wrapper around a genuine bug is NOT, preserving its
   traceback.
"""

import pytest

from aiconfigurator.sdk.errors import (
    InsufficientMemoryError,
    KVCacheCapacityError,
    NoFeasibleConfigError,
    NoResultsError,
    is_expected_no_result_cause,
)
from aiconfigurator.sdk.perf_database import PerfDataNotAvailableError

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "exc_type",
    [NoFeasibleConfigError, InsufficientMemoryError, KVCacheCapacityError],
)
def test_expected_no_result_types_are_noresults_and_runtimeerror(exc_type) -> None:
    # is-a NoResultsError -> a single isinstance classifies it cleanly.
    assert issubclass(exc_type, NoResultsError)
    # is-a RuntimeError -> backward-compatible with existing broad catches.
    assert issubclass(exc_type, RuntimeError)


def test_perf_data_miss_is_not_a_no_results_error() -> None:
    # A per-op data miss can be skipped while other configs still produce
    # results, so it is deliberately NOT in the NoResultsError family (it keeps
    # its own recognizer, has_perf_data_not_available_cause).
    assert not issubclass(PerfDataNotAvailableError, NoResultsError)
    assert issubclass(PerfDataNotAvailableError, RuntimeError)
    assert not is_expected_no_result_cause(PerfDataNotAvailableError("miss"))


def test_recognizer_accepts_direct_family_member() -> None:
    assert is_expected_no_result_cause(InsufficientMemoryError("oom"))


def test_recognizer_accepts_generic_wrapper_chained_from_family_member() -> None:
    # Mirrors sweep.py: ``raise RuntimeError(...) from exceptions[-1]`` where the
    # last per-config failure was a NoResultsError-family member.
    try:
        raise RuntimeError("no results for any config") from InsufficientMemoryError("oom")
    except RuntimeError as exc:
        assert is_expected_no_result_cause(exc)


def test_recognizer_rejects_wrapper_around_real_bug() -> None:
    # A genuine bug surfacing as the last exception must keep its traceback.
    try:
        raise RuntimeError("no results for any config") from KeyError("unexpected")
    except RuntimeError as exc:
        assert not is_expected_no_result_cause(exc)


def test_recognizer_rejects_plain_runtimeerror() -> None:
    assert not is_expected_no_result_cause(RuntimeError("something else"))


def test_recognizer_avoids_cycles() -> None:
    error = RuntimeError("outer")
    nested = RuntimeError("nested")
    error.__cause__ = nested
    nested.__context__ = error

    assert not is_expected_no_result_cause(error)

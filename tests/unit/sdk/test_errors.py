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
    EmpiricalNotImplementedError,
    InsufficientMemoryError,
    KVCacheCapacityError,
    NoFeasibleConfigError,
    NoResultsError,
    is_expected_cli_error,
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


def test_recognizer_accepts_implicit_expected_context() -> None:
    inner = InsufficientMemoryError("expected no-result")
    outer = RuntimeError("wrapper")
    outer.__context__ = inner

    assert is_expected_no_result_cause(outer)


def test_recognizer_prefers_explicit_cause_over_expected_context() -> None:
    try:
        try:
            raise InsufficientMemoryError("expected no-result")
        except InsufficientMemoryError:
            raise RuntimeError("wrapper") from KeyError("real bug")
    except RuntimeError as exc:
        assert not is_expected_no_result_cause(exc)


def test_recognizer_ignores_suppressed_expected_context() -> None:
    try:
        try:
            raise InsufficientMemoryError("expected no-result")
        except InsufficientMemoryError:
            raise KeyError("real bug while handling expected result") from None
    except KeyError as exc:
        assert exc.__suppress_context__
        assert not is_expected_no_result_cause(exc)


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


# ---------------------------------------------------------------------------
# is_expected_cli_error — the shared CLI classifier (NVBug 6401889)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        NoFeasibleConfigError("sla"),
        InsufficientMemoryError("oom"),
        KVCacheCapacityError("kv"),
        PerfDataNotAvailableError("missing silicon data"),
        EmpiricalNotImplementedError("no basis"),
        ValueError("Unsupported gemm quant mode 'fp8_static'"),
    ],
)
def test_cli_classifier_accepts_expected_user_errors(exc) -> None:
    """SLA/OOM/KV, perf-data misses, and config/compatibility ValueErrors are concise."""
    assert is_expected_cli_error(exc)


@pytest.mark.parametrize(
    "exc",
    [
        KeyError("bug"),
        AttributeError("bug"),
        TypeError("bug"),
        RuntimeError("OOM: model does not fit"),  # plain RuntimeError (e.g. OOM) keeps traceback
        IndexError("bug"),
    ],
)
def test_cli_classifier_rejects_programming_errors(exc) -> None:
    """Genuine defects (and OOM RuntimeError) are NOT concise — they keep tracebacks."""
    assert not is_expected_cli_error(exc)


def test_cli_classifier_accepts_wrapper_chained_from_expected_cause() -> None:
    # A ValueError (unsupported quant) wrapped by a generic RuntimeError is still expected.
    try:
        raise RuntimeError("task failed") from ValueError("Unsupported quant mode")
    except RuntimeError as exc:
        assert is_expected_cli_error(exc)
    # Same for a perf-data miss surfaced through a wrapper.
    try:
        raise RuntimeError("task failed") from PerfDataNotAvailableError("miss")
    except RuntimeError as exc:
        assert is_expected_cli_error(exc)


def test_cli_classifier_rejects_wrapper_around_real_bug() -> None:
    try:
        raise RuntimeError("task failed") from KeyError("unexpected")
    except RuntimeError as exc:
        assert not is_expected_cli_error(exc)

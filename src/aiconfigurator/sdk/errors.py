# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared SDK exception types."""


class NoResultsError(RuntimeError):
    """Base class for *expected* "the sweep produced no results" outcomes.

    These are not bugs: they mean a sweep ran to completion but every parallel
    configuration was ruled out for an understood reason (SLA infeasible, OOM,
    KV-cache capacity). They carry an actionable message and should be reported
    cleanly (no Python traceback), unlike a genuine crash.

    A per-op data miss (``PerfDataNotAvailableError``) is deliberately NOT in
    this family -- it can be skipped on one config while others still produce
    results -- and has its own recognizer ``has_perf_data_not_available_cause``.

    Subclasses ``RuntimeError`` so existing ``except RuntimeError`` / ``except
    Exception`` callers (e.g. the per-config sweep catch, support-matrix) keep
    catching them unchanged. Recognize the whole family via
    :func:`is_expected_no_result_cause`, which walks the exception chain so a
    generic wrapper raised ``from`` one of these is still classified correctly.
    """


class NoFeasibleConfigError(NoResultsError):
    """Raised when no configuration satisfies user-provided SLA constraints."""


class InsufficientMemoryError(NoResultsError):
    """Raised when the model does not fit in GPU memory for any parallel config."""


class KVCacheCapacityError(NoResultsError):
    """Raised when the requested batch size exceeds KV-cache capacity for all configs."""


class UnsupportedWideepConfigError(ValueError):
    """Raised when a requested WideEP configuration is not in the perf database.

    Subclasses ``ValueError`` so callers that ``except ValueError`` still catch it.
    """


def is_expected_no_result_cause(error: BaseException) -> bool:
    """Return True when ``error`` or its effective chain has a NoResultsError.

    Follows an explicit ``__cause__`` or an unsuppressed ``__context__`` so a
    generic wrapper such as
    ``RuntimeError(...) from InsufficientMemoryError(...)`` is recognized as an
    expected no-result outcome, while a wrapper around a genuine bug (e.g. an
    unexpected ``KeyError``) returns False and keeps its traceback.
    """
    seen: set[int] = set()
    stack: list[BaseException] = [error]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        if isinstance(current, NoResultsError):
            return True
        seen.add(id(current))
        if current.__cause__ is not None:
            stack.append(current.__cause__)
        elif not current.__suppress_context__ and current.__context__ is not None:
            stack.append(current.__context__)
    return False

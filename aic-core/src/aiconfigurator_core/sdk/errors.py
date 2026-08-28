# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared SDK exception types and per-experiment outcome classification."""

from __future__ import annotations

import dataclasses


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


class MissingSystemFlopsError(ValueError):
    """Raised when a quant mode's compute dtype has no ``*_tc_flops`` entry in the system YAML.

    A missing entry means either the platform has no hardware support for that
    dtype (the quant mode must not be modeled on it) or the system YAML is
    incomplete and must be filled in. We raise instead of extrapolating
    ``bfloat16_tc_flops * compute_factor`` so a fictional throughput is never
    silently produced (b300/gb300 already break the fixed-ratio assumption).

    Subclasses ``ValueError`` per the SDK convention for unsupported quant /
    hardware rejections, so it is reported as an expected CLI error.
    """


class InterpolationDataNotAvailableError(ValueError):
    """Raised when interpolation cannot produce a real value from available data.

    Subclasses ``ValueError`` so existing callers that catch ``ValueError``
    keep working. The perf-DB layer catches this specific class to classify
    the failure as "missing silicon data" without swallowing genuine
    programming bugs that raise plain ``ValueError`` deeper in the stack.
    """


class PerfDataNotAvailableError(RuntimeError):
    """Raised when required performance data is missing or unsupported for a requested mode.

    This is a *per-op* data-miss raised deep inside a single config's evaluation
    (and in non-sweep paths like validate / single-point estimate), so it is
    deliberately NOT a :class:`NoResultsError`: a miss on one config can be
    skipped while other configs still produce results. Recognize it via
    ``has_perf_data_not_available_cause`` in ``perf_database``.
    """


class EmpiricalNotImplementedError(RuntimeError):
    """Raised when the empirical (SOL/util) path has no basis to estimate an op.

    Distinct from ``PerfDataNotAvailableError`` (SILICON has no exact bracket but
    HYBRID may still estimate): this is the terminal signal that even the
    empirical fallback found nothing to calibrate from — no own-shape util, no
    cross-shape/sibling transfer reference. We raise instead of returning a
    placeholder ``SOL / constant`` so missing coverage surfaces honestly rather
    than as a fabricated number. Genuinely table-less ops (mem / p2p /
    element-wise) keep their own analytic formulas and never reach here.
    """


class SolNotImplementedError(RuntimeError):
    """Raised when the analytic SOL path cannot model a required operator.

    This is a modeling-coverage gap, distinct from invalid user input and from
    missing silicon data. Callers may omit an optional SOL comparison while
    preserving the primary estimate, but an explicitly requested SOL estimate
    still fails with this typed error.
    """


def _chain_has(error: BaseException, types: tuple[type[BaseException], ...]) -> bool:
    """Return True when ``error`` or its effective chain contains one of ``types``.

    Follows an explicit ``__cause__`` or an unsuppressed ``__context__`` so a
    generic wrapper such as ``RuntimeError(...) from InsufficientMemoryError(...)``
    is classified by the underlying cause, while a wrapper around a genuine bug
    (e.g. an unexpected ``KeyError``) is not matched and keeps its traceback.
    """
    seen: set[int] = set()
    stack: list[BaseException] = [error]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        if isinstance(current, types):
            return True
        seen.add(id(current))
        if current.__cause__ is not None:
            stack.append(current.__cause__)
        elif not current.__suppress_context__ and current.__context__ is not None:
            stack.append(current.__context__)
    return False


def is_expected_no_result_cause(error: BaseException) -> bool:
    """Return True when ``error`` or its effective chain has a NoResultsError."""
    return _chain_has(error, (NoResultsError,))


def is_expected_cli_error(error: BaseException) -> bool:
    """Return True when ``error`` is an *expected*, user-actionable CLI failure.

    Expected = the user's inputs/environment can't be served and the message
    already says why: SLA-infeasible / OOM / KV-cache (``NoResultsError``), a
    perf-data or modeling coverage gap (``PerfDataNotAvailableError`` /
    ``EmpiricalNotImplementedError`` / ``SolNotImplementedError``), or a
    configuration / compatibility rejection (``ValueError`` — the whole SDK
    raises this by convention for unsupported quant modes, invalid parallelism,
    hardware requirements, etc.).

    Such errors should be reported as a concise ``Error: <message>`` line, not a
    Python traceback. Genuine programming defects (``KeyError``,
    ``AttributeError``, ``TypeError``, ``RuntimeError`` such as OOM, …) are NOT
    matched and keep their traceback so real bugs stay visible. Callers should
    still emit the full traceback at DEBUG level (``exc_info=True``) so
    ``--log-level DEBUG`` can recover it when diagnosing.

    Walks the exception chain, so a wrapper raised ``from`` an expected cause is
    still recognized.
    """
    return _chain_has(
        error,
        (
            NoResultsError,
            PerfDataNotAvailableError,
            EmpiricalNotImplementedError,
            SolNotImplementedError,
            ValueError,
        ),
    )


def is_gpu_retriable(error: BaseException) -> bool:
    """Return True when ``error`` might be resolved by adding more GPUs.

    ``InsufficientMemoryError`` and ``KVCacheCapacityError`` are retriable:
    more GPUs means more aggregate memory and KV-cache capacity.  All other
    expected errors (SLA-infeasible, missing perf data, unsupported config)
    are structural and will not be helped by more hardware.

    Walks the exception chain via :func:`_chain_has`.
    """
    return _chain_has(error, (InsufficientMemoryError, KVCacheCapacityError))


@dataclasses.dataclass(frozen=True, slots=True)
class ExperimentOutcome:
    """Per-experiment verdict from :func:`_execute_tasks`.

    ``error is None`` means the experiment succeeded (results are in the
    parallel ``best_configs`` / ``pareto_fronts`` dicts).  A non-None
    ``error`` carries the original exception for caller classification
    (e.g. :func:`is_gpu_retriable`).
    """

    experiment: str
    error: BaseException | None = None

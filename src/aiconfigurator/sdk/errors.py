# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared SDK exception types."""


class NoFeasibleConfigError(RuntimeError):
    """Raised when no configuration satisfies user-provided SLA constraints."""


class UnsupportedWideepConfigError(ValueError):
    """Raised when a requested WideEP configuration is not in the perf database.

    Note: V1 ``sdk.task`` defines a separately-named class with the same purpose.
    Both subclass ``ValueError``; callers that ``except ValueError`` work for both.
    Future code paths use this one (from ``sdk.errors``).
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

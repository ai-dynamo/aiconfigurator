# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composite operations (ISSUE-14).

Two op classes migrated from ``_legacy.py``:

- ``FallbackOp`` — try a primary op, fall back to a sequence of ops on
  ``PerfDataNotAvailableError``. In HYBRID mode the primary runs against a
  reusable SILICON query view, so HYBRID does not silently swallow a miss with
  an empirical estimate and the caller's shared view is never mutated.
- ``OverlapOp`` — model two op groups that execute in parallel (TRT-LLM
  ``maybe_execute_in_parallel`` behavior on different CUDA streams during
  generation with CUDA Graph enabled). ``latency = max(sum_a, sum_b)``,
  ``energy = sum_a + sum_b``.

Neither op owns any CSV data — they delegate to inner ``Operation``
instances and their ``query()`` methods. No ``_data_cache``, no
``load_data``, no ``clear_cache``; the ``Operation`` base class provides
empty defaults that suffice.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations.base import Operation
from aiconfigurator.sdk.performance_result import PerformanceResult

if TYPE_CHECKING:
    from aiconfigurator.sdk.perf_database import PerfDatabase


logger = logging.getLogger(__name__)


def _silicon_query_view(database: PerfDatabase) -> PerfDatabase:
    """Return a SILICON child view without mutating ``database``.

    Production ``PerfDatabase`` instances cache ``query_view`` by mode/policy,
    preserving their query LRUs across repeated fallback attempts. Lightweight
    legacy/fake databases may not implement that API; deep-copying them keeps
    nested runtime caches isolated while the scalar default mode is overridden.
    """
    query_view = getattr(database, "query_view", None)
    if callable(getattr(type(database), "query_view", None)) and callable(query_view):
        return query_view(common.DatabaseMode.SILICON, getattr(database, "transfer_policy", None))

    view = copy.deepcopy(database)
    view._default_database_mode = common.DatabaseMode.SILICON
    return view


class FallbackOp(Operation):
    """
    Try a primary operation first; if it raises PerfDataNotAvailableError,
    fall back to a sequence of fallback operations (summed).

    This supports transitional periods where some systems have module-level
    profiling data (single op) while others still have granular per-kernel data
    (multiple ops). The fallback is symmetric: either group can be primary.

    In HYBRID mode, the primary is queried in SILICON mode so that HYBRID does
    not silently swallow a miss with an empirical estimate — the fallback ops
    (which have real data) should be preferred over an empirical guess. In
    explicit EMPIRICAL/SOL modes, the primary respects the requested mode.

    Once the primary reports ``PerfDataNotAvailableError``, it is skipped on
    subsequent calls to avoid redundant work. Raw schema/programming errors
    propagate and never activate fallback.

    Latency = primary.query()  OR  sum(fallback[i].query())
    Energy  = same source as whichever succeeds
    Weights = sum of whichever group is used (primary or fallback)
    """

    def __init__(self, name: str, primary: Operation, fallback: list[Operation]) -> None:
        """
        Args:
            name: Operation name for latency breakdown reporting.
            primary: Single operation to try first.
            fallback: List of operations to sum if primary fails.
        """
        super().__init__(name, 1.0)  # scale_factor handled by inner ops
        self._primary = primary
        self._fallback = fallback
        self._primary_unavailable = False

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        from aiconfigurator.sdk.perf_database import PerfDataNotAvailableError

        if not self._primary_unavailable:
            primary_database = (
                _silicon_query_view(database)
                if database._default_database_mode == common.DatabaseMode.HYBRID
                else database
            )

            try:
                return self._primary.query(primary_database, **kwargs)
            except PerfDataNotAvailableError as e:
                self._primary_unavailable = True
                logger.debug(
                    "FallbackOp '%s': primary op '%s' failed (%s: %s), using fallback ops",
                    self._name,
                    self._primary._name,
                    type(e).__name__,
                    e,
                )

        total = PerformanceResult(0.0, energy=0.0, source="empirical")
        for op in self._fallback:
            total += op.query(database, **kwargs)
        return total

    def get_weights(self, **kwargs):
        # Use primary weights if available, otherwise sum fallback weights.
        # In practice both should be equivalent since they model the same block.
        if not self._primary_unavailable:
            primary_w = self._primary.get_weights(**kwargs)
            if primary_w > 0:
                return primary_w
        return sum(op.get_weights(**kwargs) for op in self._fallback)


class OverlapOp(Operation):
    """
    Two groups of operations that execute in parallel (overlap).

    This models the TRT-LLM `maybe_execute_in_parallel` behavior where two
    operation groups run concurrently on different CUDA streams during
    generation phase (CUDA Graph enabled).

    Latency = max(sum(group_a latencies), sum(group_b latencies))
    Energy  = sum(all ops in both groups)  # both groups consume power
    Weights = sum(all ops in both groups)
    """

    def __init__(self, name: str, group_a: list, group_b: list) -> None:
        """
        Args:
            name: Operation name for latency breakdown reporting.
            group_a: List of Operation objects for the first parallel group
                     (e.g., routed expert path on main stream).
            group_b: List of Operation objects for the second parallel group
                     (e.g., shared expert path on aux stream).
        """
        super().__init__(name, 1.0)  # scale_factor handled by inner ops
        self._group_a = group_a
        self._group_b = group_b

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """
        Query overlap operation latency.

        Returns:
            PerformanceResult with latency = max(group_a, group_b)
            and energy = sum of all ops.
        """
        total_a = PerformanceResult(0.0, energy=0.0, source="empirical")
        for op in self._group_a:
            total_a += op.query(database, **kwargs)

        total_b = PerformanceResult(0.0, energy=0.0, source="empirical")
        for op in self._group_b:
            total_b += op.query(database, **kwargs)

        merged = total_a + total_b
        return PerformanceResult(
            latency=max(float(total_a), float(total_b)),
            energy=total_a.energy + total_b.energy,
            source=merged.source,
        )

    def get_weights(self, **kwargs):
        weights = 0.0
        for op in self._group_a + self._group_b:
            weights += op.get_weights(**kwargs)
        return weights

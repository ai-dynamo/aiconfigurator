# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composite operations (ISSUE-14).

Two op classes migrated from ``_legacy.py``:

- ``FallbackOp`` — try a primary op, fall back to a sequence of ops on
  ``PerfDataNotAvailableError``. In HYBRID mode the primary runs against a
  SILICON-configured copy, so HYBRID does not silently swallow a miss with an
  empirical estimate and the caller's database is never mutated.
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

import logging
from typing import TYPE_CHECKING, ClassVar

from aiconfigurator_core.sdk.operations.base import Operation

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


def _infer_phase(op) -> bool | None:
    """First phase marker found in a composite subtree: ``_is_context`` /
    ``_phase`` instance fields, a phase-declaring ``_ENGINE_QUERY_SHAPE``, or
    recursion into Overlap/Fallback child groups."""
    is_context = getattr(op, "_is_context", None)
    if is_context is not None:
        return bool(is_context)
    # context/generation: mamba/gdn kernels; prefill/decode: FPMForwardOp.
    phase = getattr(op, "_phase", None)
    if phase in ("context", "prefill"):
        return True
    if phase in ("generation", "decode"):
        return False
    shape = getattr(type(op), "_ENGINE_QUERY_SHAPE", None)
    if shape in ("context", "generation"):
        return shape == "context"
    for group in ("_group_a", "_group_b", "_fallback"):
        for child in getattr(op, group, ()) or ():
            found = _infer_phase(child)
            if found is not None:
                return found
    primary = getattr(op, "_primary", None)
    if primary is not None:
        return _infer_phase(primary)
    return None


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

    A data miss applies only to the current query. Later shapes try the primary
    again because a missing interpolation bracket does not imply that the whole
    module table is unavailable. Raw schema/programming errors propagate.

    Latency = primary.query()  OR  sum(fallback[i].query())
    Energy  = same source as whichever succeeds
    Weights = primary weights when defined, otherwise the fallback sum
    """

    _CP_AWARE: ClassVar[bool] = True  # wrapper: inner ops carry their own seq_split

    def __init__(self, name: str, primary: Operation, fallback: list[Operation], *, seq_split: int = 1) -> None:
        """
        Args:
            name: Operation name for latency breakdown reporting.
            primary: Single operation to try first.
            fallback: List of operations to sum if primary fails.
            seq_split: Carried for API uniformity. The wrapper delegates to
                inner ops which carry their own ``seq_split``; this one is
                stored on the base class for completeness but not used here.
        """
        super().__init__(name, 1.0, seq_split=seq_split)  # scale_factor handled by inner ops
        self._primary = primary
        self._fallback = fallback

    _ENGINE_QUERY_SHAPE = "module"

    def _engine_query_is_context(self, kwargs: dict) -> bool:
        """Composites carry no phase of their own: use the explicit
        ``is_context=`` kwarg when given, else infer it from the first
        phase-marked descendant (mixed-phase groups do not occur — the model
        builders assemble composites per phase list)."""
        hint = kwargs.get("is_context")
        if hint is not None:
            return bool(hint)
        inferred = _infer_phase(self)
        if inferred is None:
            raise ValueError(
                f"{type(self).__name__}.query cannot infer the evaluation phase from its children; "
                "pass is_context=True/False."
            )
        return inferred

    def get_weights(self, **kwargs):
        # Use primary weights if available, otherwise sum fallback weights.
        # In practice both should be equivalent since they model the same block.
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

    _CP_AWARE: ClassVar[bool] = True  # wrapper: inner ops carry their own seq_split

    def __init__(self, name: str, group_a: list, group_b: list, *, seq_split: int = 1) -> None:
        """
        Args:
            name: Operation name for latency breakdown reporting.
            group_a: List of Operation objects for the first parallel group
                     (e.g., routed expert path on main stream).
            group_b: List of Operation objects for the second parallel group
                     (e.g., shared expert path on aux stream).
            seq_split: Carried for API uniformity. Inner ops carry their own.
        """
        super().__init__(name, 1.0, seq_split=seq_split)  # scale_factor handled by inner ops
        self._group_a = group_a
        self._group_b = group_b

    _ENGINE_QUERY_SHAPE = "module"

    def _engine_query_is_context(self, kwargs: dict) -> bool:
        """Composites carry no phase of their own: use the explicit
        ``is_context=`` kwarg when given, else infer it from the first
        phase-marked descendant (mixed-phase groups do not occur — the model
        builders assemble composites per phase list)."""
        hint = kwargs.get("is_context")
        if hint is not None:
            return bool(hint)
        inferred = _infer_phase(self)
        if inferred is None:
            raise ValueError(
                f"{type(self).__name__}.query cannot infer the evaluation phase from its children; "
                "pass is_context=True/False."
            )
        return inferred

    def get_weights(self, **kwargs):
        weights = 0.0
        for op in self._group_a + self._group_b:
            weights += op.get_weights(**kwargs)
        return weights

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
from typing import TYPE_CHECKING

import aiconfigurator_core._aiconfigurator_core as _core
from aiconfigurator_core.sdk.operations.base import OpShellKit

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
    if phase in ("generation", "decode", "verify"):
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


def _has_leaves(op) -> bool:
    """True if the composite subtree reaches at least one LEAF op (a node
    that is not itself an Overlap/Fallback composite)."""
    composite = False
    for group in ("_group_a", "_group_b", "_fallback"):
        children = getattr(op, group, None)
        if children is not None:
            composite = True
            if any(_has_leaves(child) for child in children):
                return True
    primary = getattr(op, "_primary", None)
    if primary is not None:
        composite = True
        if _has_leaves(primary):
            return True
    return not composite


class FallbackOp(_core.FallbackOp, OpShellKit):
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

    @property
    def _seq_split(self) -> int:
        """CP shard factor. Wrappers store it for API uniformity
        only (inner ops carry their own). The Rust op carries no
        ``seq_split`` field (nothing crosses the wire), so the value lives in
        the shell instance ``__dict__`` — written by the models' CP wiring
        (``apply_cp_to_context_ops``) and read by the shim x-division.
        Survives pickle via the default object state (``__dict__``)."""
        return self.__dict__.get("_py_seq_split", 1)

    @_seq_split.setter
    def _seq_split(self, value: int) -> None:
        self.__dict__["_py_seq_split"] = int(value)

    def _engine_query_plan(self, kwargs: dict):
        """Composites carry no phase of their own. A phase-marked descendant
        (or an explicit ``is_context=`` kwarg) selects the batch-major plan;
        a subtree with NO marker anywhere <=> every leaf is token-shaped
        (batch-major leaves always carry a marker via class shape or instance
        fields), so the plan is TOKEN-shaped — preserving the legacy
        ``query(db, x=...)``-only call shape, which never required
        ``batch_size``/``s`` for token children. A truly EMPTY composite (no
        leaf anywhere) keeps the legacy bare ``query(db)`` shape: zero work
        needs no token count, so ``x`` defaults."""
        if kwargs.get("is_context") is None and _infer_phase(self) is None:
            x = kwargs.get("x")
            if x is None:
                if _has_leaves(self):
                    raise ValueError(f"{type(self).__name__}.query requires 'x' (num tokens) for token-only groups.")
                x = 1
            return self, {"is_context": True, "batch_size": 1, "s": 1, "x": int(x)}
        return super()._engine_query_plan(kwargs)

    def _engine_query_is_context(self, kwargs: dict) -> bool:
        hint = kwargs.get("is_context")
        if hint is not None:
            return bool(hint)
        inferred = _infer_phase(self)
        # Unreachable when inferred is None (the plan override takes the
        # token-shaped path first); kept as a safe default.
        return True if inferred is None else inferred


class OverlapOp(_core.OverlapOp, OpShellKit):
    """
    Two groups of operations that execute in parallel (overlap).

    This models the TRT-LLM `maybe_execute_in_parallel` behavior where two
    operation groups run concurrently on different CUDA streams during
    generation phase (CUDA Graph enabled).

    Latency = max(sum(group_a latencies), sum(group_b latencies))
    Energy  = sum(all ops in both groups)  # both groups consume power
    Weights = sum(all ops in both groups)
    """

    @property
    def _seq_split(self) -> int:
        """CP shard factor. Wrappers store it for API uniformity
        only (inner ops carry their own). The Rust op carries no
        ``seq_split`` field (nothing crosses the wire), so the value lives in
        the shell instance ``__dict__`` — written by the models' CP wiring
        (``apply_cp_to_context_ops``) and read by the shim x-division.
        Survives pickle via the default object state (``__dict__``)."""
        return self.__dict__.get("_py_seq_split", 1)

    @_seq_split.setter
    def _seq_split(self, value: int) -> None:
        self.__dict__["_py_seq_split"] = int(value)

    def _engine_query_plan(self, kwargs: dict):
        """Composites carry no phase of their own. A phase-marked descendant
        (or an explicit ``is_context=`` kwarg) selects the batch-major plan;
        a subtree with NO marker anywhere <=> every leaf is token-shaped
        (batch-major leaves always carry a marker via class shape or instance
        fields), so the plan is TOKEN-shaped — preserving the legacy
        ``query(db, x=...)``-only call shape, which never required
        ``batch_size``/``s`` for token children. A truly EMPTY composite (no
        leaf anywhere) keeps the legacy bare ``query(db)`` shape: zero work
        needs no token count, so ``x`` defaults."""
        if kwargs.get("is_context") is None and _infer_phase(self) is None:
            x = kwargs.get("x")
            if x is None:
                if _has_leaves(self):
                    raise ValueError(f"{type(self).__name__}.query requires 'x' (num tokens) for token-only groups.")
                x = 1
            return self, {"is_context": True, "batch_size": 1, "s": 1, "x": int(x)}
        return super()._engine_query_plan(kwargs)

    def _engine_query_is_context(self, kwargs: dict) -> bool:
        hint = kwargs.get("is_context")
        if hint is not None:
            return bool(hint)
        inferred = _infer_phase(self)
        # Unreachable when inferred is None (the plan override takes the
        # token-shaped path first); kept as a safe default.
        return True if inferred is None else inferred

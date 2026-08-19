# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Empirical-provenance pipeline (and quant-profile metadata).

The empirical-utilization MATH that used to live here — the ``UtilGrid``
two-neighbour estimator, the cross-shape/quant/profile/op transfer ladder,
``grid_from_reference`` — was retired with the Python per-call query stack
(#1357 PR-5). Its single oracle is the compiled engine
(``aic-core/rust/aiconfigurator-core/src/operators/util_empirical.rs``); the
transfer-ladder semantics live there and are guarded by the frozen parity
goldens.

What remains here, by design:

* the PROVENANCE pipeline — the compiled engine reports which empirical tier
  fired back through :func:`note_provenance` (see
  ``rust_engine_step``/``EngineHandle.last_provenance``), and
  :func:`capture_provenance` / :func:`worst_provenance` summarise a run's
  effective data source (support-matrix HYBRID labelling);
* :func:`quant_profile` — the (memory, compute) classification key of the
  per-op util-LEVEL admission tables (``task_v2``'s validate gate consults
  those via ``xprofile_util_level_known``); metadata, not estimation math.

The single-oracle contract test
(``tests/cross_package/test_single_oracle_contract.py``) freezes this
module's public surface — re-growing estimation math here is a deliberate,
review-visible act.
"""

from __future__ import annotations

import contextlib
import contextvars

# ---------------------------------------------------------------------------
# Provenance capture: record which empirical path produced a value, so a run's
# overall data source can be summarised (e.g. the support matrix labelling a
# config SILICON vs HYBRID and at which transfer tier). Recording is a no-op
# unless a capture is active. Tags ordered by DECREASING confidence (later =
# relies on more aggressive transfer).
# ---------------------------------------------------------------------------
PROVENANCE_ORDER: tuple[str, ...] = (
    "silicon",  # pure silicon table data (never recorded here; the default when nothing fired)
    "empirical",  # own-shape util (no transfer)
    "xshape",  # cross-shape, same quant
    "xquant",  # cross-quant, same profile
    "xprofile",  # cross-quant, cross profile
    "xop",  # cross-op (borrowed a different op's util)
)
_PROVENANCE_RANK = {tag: i for i, tag in enumerate(PROVENANCE_ORDER)}
_PROVENANCE: contextvars.ContextVar = contextvars.ContextVar("aic_provenance", default=None)


def note_provenance(tag: str) -> None:
    """Record that an empirical path of kind ``tag`` fired (no-op outside a capture)."""
    sink = _PROVENANCE.get()
    if sink is not None:
        sink.add(tag)


@contextlib.contextmanager
def capture_provenance():
    """Collect the set of empirical-path tags fired within the block."""
    sink: set[str] = set()
    token = _PROVENANCE.set(sink)
    try:
        yield sink
    finally:
        _PROVENANCE.reset(token)


def worst_provenance(tags) -> str:
    """The least-confident tag in ``tags`` (the run's effective data source);
    ``"silicon"`` when empty (no empirical path fired)."""
    return max(tags, key=lambda t: _PROVENANCE_RANK.get(t, 0), default="silicon")


def clear_grid_cache() -> None:
    """Compat no-op: the util-grid cache retired with the estimation math
    (#1357 PR-5). Kept so the ``clear_all_op_caches`` eviction contract and
    long-standing callers stay valid through the deprecation window."""
    return None


def quant_profile(quant) -> tuple[float, float]:
    """The (memory, compute) profile of a quant enum member — the key of the
    per-op util-LEVEL admission tables (``xprofile_util_level_known``)."""
    return (quant.value.memory, quant.value.compute)

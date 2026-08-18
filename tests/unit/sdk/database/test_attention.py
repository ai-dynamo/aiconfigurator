# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Database query-mode management contracts.

The attention/MLA per-call query behaviour this file used to pin retired to
the compiled engine with #1357 PR-5 and is anchored by the frozen parity
goldens (the per-call ``query_*`` shims and their one-release baseline were
removed by the deprecation-cleanup PR). What stays Python-owned —
default-mode entry/rotation and the SOL_FULL mode-entry refusals — is tested
here on a real shipped database through the single-op plumbing and the
SOL-decomposition FFI.
"""

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import get_database

pytestmark = pytest.mark.unit


def _context_attention_value(db) -> float:
    """One ContextAttention twin evaluated under the database's LIVE mode
    through the single-op plumbing (the b200 sglang 0.5.14 case the retired
    shim test probed)."""
    from aiconfigurator_core.sdk.engine import _evaluate_single_op
    from aiconfigurator_core.sdk.operations.attention import ContextAttention

    op = ContextAttention(
        "context_attention_query",
        1.0,
        8,
        4,
        common.KVCacheQuantMode.bfloat16,
        common.FMHAQuantMode.bfloat16,
    )
    return float(_evaluate_single_op(db, op, is_context=True, batch_size=1, s=32, prefix=0))


def test_default_database_mode():
    """Setting the default mode changes what live-mode evaluations return:
    the single-op plumbing probes the database's CURRENT view, so a mode
    rotation must be visible without any per-call mode argument."""
    db = get_database("b200_sxm", "sglang", "0.5.14")
    assert db.get_default_database_mode() == common.DatabaseMode.SILICON
    try:
        non_sol_result = _context_attention_value(db)

        db.set_default_database_mode(common.DatabaseMode.SOL)
        assert db.get_default_database_mode() == common.DatabaseMode.SOL
        sol_result = _context_attention_value(db)
        assert sol_result != non_sol_result
    finally:
        db.set_default_database_mode(common.DatabaseMode.SILICON)


def test_sol_full_is_per_call_diagnostic_never_default_mode(mutable_comprehensive_perf_db):
    """DatabaseMode.SOL_FULL is a per-call diagnostic (the sanity notebook
    unpacks its raw 3-tuple) but can never become the active mode: every
    mode-entry choke point raises."""
    from aiconfigurator_core.sdk.perf_database import _normalize_database_mode

    db = mutable_comprehensive_perf_db
    with pytest.raises(ValueError, match="cannot be a database's default mode"):
        db.set_default_database_mode(common.DatabaseMode.SOL_FULL)
    # The refused mode must not stick.
    assert db.get_default_database_mode() == common.DatabaseMode.SILICON

    # The get_database / get_database_view string+enum normalizer refuses too.
    with pytest.raises(ValueError, match="cannot be a database's default mode"):
        _normalize_database_mode("SOL_FULL")
    with pytest.raises(ValueError, match="cannot be a database's default mode"):
        _normalize_database_mode(common.DatabaseMode.SOL_FULL)

    # The per-call diagnostic contract lives on the op-list FFI now: a raw
    # (sol, sol_math, sol_mem) decomposition per op, unpackable exactly as
    # tools/sanity_check/validate_database.ipynb consumes it (via
    # EngineReference). The value rides the engine's SOL-decomposition FFI,
    # so it needs a real database the probe engine can load from disk.
    from aiconfigurator_core.sdk import engine
    from aiconfigurator_core.sdk.operations.elementwise import ElementWise

    real_db = get_database("b200_sxm", "sglang", "0.5.14")
    mem_twin = ElementWise("mem_op_query", 1.0, -(-(1 << 20) // 2), 0)
    ops_json = engine.build_ops_json(
        [mem_twin], model=engine._PROBE_MODEL_STUB, backend=real_db.backend, database=real_db
    )
    key, systems_path = engine._probe_spec_key(real_db, common.DatabaseMode.SOL.name)
    handle = engine._probe_handle_from_key(key, systems_path)
    (_, sol_time, sol_math, sol_mem) = handle.evaluate_ops_sol_json(ops_json, is_context=True, batch_size=1, s=1, x=1)[0]
    assert sol_time == pytest.approx(max(sol_math, sol_mem))

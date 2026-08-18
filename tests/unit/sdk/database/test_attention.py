# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Database query-mode management contracts.

The attention/MLA per-call query behaviour this file used to pin on the
synthetic comprehensive fixture (silicon interpolation, SOL formulas, the
head-size reference-grid transfer, window-slice empirical fallbacks, typed
coverage misses) retired to the compiled engine with #1357 PR-5; it is
anchored by tests/cross_package/test_query_shim_baseline.py and the frozen
parity goldens. What stays Python-owned — default-mode entry/rotation, the
lru-cache eviction contract, and the per-call SOL_FULL diagnostic surface —
is tested here on a real shipped database (the engine-routed shims load
their tables from disk).
"""

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import get_database

pytestmark = pytest.mark.unit


def test_default_database_mode():
    """Setting the default mode changes what unqualified queries return, and
    rotating the mode clears the per-facade lru caches."""
    db = get_database("b200_sxm", "sglang", "0.5.14")
    assert db.get_default_database_mode() == common.DatabaseMode.SILICON
    try:
        non_sol_result = db.query_context_attention(
            1, 32, 0, 8, 4, common.KVCacheQuantMode.bfloat16, common.FMHAQuantMode.bfloat16
        )
        assert db.query_context_attention.cache_info().currsize >= 1

        db.set_default_database_mode(common.DatabaseMode.SOL)
        assert db.get_default_database_mode() == common.DatabaseMode.SOL
        # Cache should be cleared on mode rotation.
        assert db.query_context_attention.cache_info().currsize == 0

        # Query should use default mode when not specified.
        sol_result = db.query_context_attention(
            1, 32, 0, 8, 4, common.KVCacheQuantMode.bfloat16, common.FMHAQuantMode.bfloat16
        )
        cache_info = db.query_context_attention.cache_info()
        assert cache_info.misses == 1
        assert cache_info.hits == 0
        assert cache_info.currsize == 1
        assert float(sol_result) != float(non_sol_result)
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

    # The per-call diagnostic contract stays: a raw (sol, sol_math, sol_mem)
    # tuple, unpackable exactly as tools/sanity_check/validate_database.ipynb
    # consumes it. The value rides the engine's SOL-decomposition FFI, so it
    # needs a real database the probe engine can load from disk.
    real_db = get_database("b200_sxm", "sglang", "0.5.14")
    sol_time, sol_math, sol_mem = real_db.query_mem_op(1 << 20, database_mode=common.DatabaseMode.SOL_FULL)
    assert sol_time == pytest.approx(max(sol_math, sol_mem))
    result = real_db.query_context_attention(
        b=1,
        s=128,
        n=16,
        n_kv=16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        database_mode=common.DatabaseMode.SOL_FULL,
        prefix=0,
    )
    assert isinstance(result, tuple) and len(result) == 3

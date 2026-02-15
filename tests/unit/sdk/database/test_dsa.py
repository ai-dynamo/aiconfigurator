# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from aiconfigurator.sdk import common

pytestmark = pytest.mark.unit


def _default_context_query(
    db,
    *,
    s: int,
    prefix: int,
    index_topk: int = 2048,
    index_n_heads: int = 64,
    index_head_dim: int = 128,
    database_mode: common.DatabaseMode = common.DatabaseMode.SOL,
):
    return db.query_context_dsa(
        b=2,
        s=s,
        prefix=prefix,
        num_heads=32,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        kvcache_quant_mode=common.KVCacheQuantMode.float16,
        fmha_quant_mode=common.FMHAQuantMode.float16,
        database_mode=database_mode,
    )


def test_query_context_dsa_sol_respects_prefix_and_parameterization(comprehensive_perf_db):
    no_prefix = _default_context_query(comprehensive_perf_db, s=256, prefix=0, index_topk=2048)
    with_prefix = _default_context_query(comprehensive_perf_db, s=256, prefix=256, index_topk=2048)
    smaller_topk = _default_context_query(comprehensive_perf_db, s=256, prefix=256, index_topk=128)
    changed_index_heads = _default_context_query(comprehensive_perf_db, s=256, prefix=256, index_n_heads=96)

    no_prefix_sol = _default_context_query(
        comprehensive_perf_db,
        s=256,
        prefix=0,
        index_topk=2048,
        database_mode=common.DatabaseMode.SOL_FULL,
    )
    with_prefix_sol = _default_context_query(
        comprehensive_perf_db,
        s=256,
        prefix=256,
        index_topk=2048,
        database_mode=common.DatabaseMode.SOL_FULL,
    )

    assert with_prefix > no_prefix
    assert smaller_topk < with_prefix
    assert changed_index_heads != with_prefix
    assert math.isclose(with_prefix_sol[0], max(with_prefix_sol[1], with_prefix_sol[2]), rel_tol=1e-6)
    assert math.isclose(no_prefix_sol[0], max(no_prefix_sol[1], no_prefix_sol[2]), rel_tol=1e-6)


def test_query_context_dsa_silicon_applies_prefix_correction(comprehensive_perf_db):
    comprehensive_perf_db.query_context_dsa.cache_clear()

    fmha = common.FMHAQuantMode.float16
    kv = common.KVCacheQuantMode.float16
    num_heads = 32
    b = 2
    full_s = 512

    context_data = {}
    for heads in (16, 32):
        context_data.setdefault(heads, {})
        for seq in (256, 512):
            context_data[heads].setdefault(seq, {})
            for batch in (1, 2):
                context_data[heads][seq][batch] = {"latency": 10.0, "energy": 2.0}
    comprehensive_perf_db._context_dsa_data = {fmha: {kv: context_data}}

    result = comprehensive_perf_db.query_context_dsa(
        b=b,
        s=256,
        prefix=256,
        num_heads=num_heads,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=2048,
        kvcache_quant_mode=kv,
        fmha_quant_mode=fmha,
        database_mode=common.DatabaseMode.SILICON,
    )

    base_sol = comprehensive_perf_db.query_context_dsa(
        b=b,
        s=full_s,
        prefix=0,
        num_heads=num_heads,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=2048,
        kvcache_quant_mode=kv,
        fmha_quant_mode=fmha,
        database_mode=common.DatabaseMode.SOL,
    )
    target_sol = comprehensive_perf_db.query_context_dsa(
        b=b,
        s=256,
        prefix=256,
        num_heads=num_heads,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=2048,
        kvcache_quant_mode=kv,
        fmha_quant_mode=fmha,
        database_mode=common.DatabaseMode.SOL,
    )

    correction = float(target_sol) / float(base_sol)
    assert math.isclose(result, 10.0 * correction, rel_tol=1e-6)
    assert math.isclose(result.energy, 2.0 * correction, rel_tol=1e-6)


def test_query_generation_dsa_sol_respects_parameterization(comprehensive_perf_db):
    default_result = comprehensive_perf_db.query_generation_dsa(
        b=4,
        s=1024,
        num_heads=32,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=2048,
        kv_cache_dtype=common.KVCacheQuantMode.float16,
        database_mode=common.DatabaseMode.SOL,
    )
    smaller_topk = comprehensive_perf_db.query_generation_dsa(
        b=4,
        s=1024,
        num_heads=32,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
        kv_cache_dtype=common.KVCacheQuantMode.float16,
        database_mode=common.DatabaseMode.SOL,
    )
    changed_index_heads = comprehensive_perf_db.query_generation_dsa(
        b=4,
        s=1024,
        num_heads=32,
        index_n_heads=96,
        index_head_dim=128,
        index_topk=2048,
        kv_cache_dtype=common.KVCacheQuantMode.float16,
        database_mode=common.DatabaseMode.SOL,
    )

    assert smaller_topk < default_result
    assert changed_index_heads != default_result

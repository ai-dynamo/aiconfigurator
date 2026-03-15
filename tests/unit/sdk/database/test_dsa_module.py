# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from aiconfigurator.sdk import common

pytestmark = pytest.mark.unit


# ═══════════════════════════════════════════════════════════════════════
# Context DSA Module
# ═══════════════════════════════════════════════════════════════════════


class TestContextDSAModule:
    """Tests for query_context_dsa_module."""

    def test_sol_returns_positive(self, comprehensive_perf_db):
        result = comprehensive_perf_db.query_context_dsa_module(
            b=2,
            s=256,
            prefix=0,
            num_heads=32,
            kvcache_quant_mode=common.KVCacheQuantMode.float16,
            fmha_quant_mode=common.FMHAQuantMode.float16,
            gemm_quant_mode=common.GEMMQuantMode.float16,
            database_mode=common.DatabaseMode.SOL,
        )
        assert float(result) > 0

    def test_sol_full_returns_three_tuple(self, comprehensive_perf_db):
        result = comprehensive_perf_db.query_context_dsa_module(
            b=2,
            s=256,
            prefix=0,
            num_heads=32,
            kvcache_quant_mode=common.KVCacheQuantMode.float16,
            fmha_quant_mode=common.FMHAQuantMode.float16,
            gemm_quant_mode=common.GEMMQuantMode.float16,
            database_mode=common.DatabaseMode.SOL_FULL,
        )
        assert len(result) == 3
        sol_time, sol_math, sol_mem = result
        assert math.isclose(sol_time, max(sol_math, sol_mem), rel_tol=1e-6)

    def test_sol_increases_with_seq_len(self, comprehensive_perf_db):
        r1 = comprehensive_perf_db.query_context_dsa_module(
            b=4,
            s=128,
            prefix=0,
            num_heads=32,
            kvcache_quant_mode=common.KVCacheQuantMode.float16,
            fmha_quant_mode=common.FMHAQuantMode.float16,
            gemm_quant_mode=common.GEMMQuantMode.float16,
            database_mode=common.DatabaseMode.SOL,
        )
        r2 = comprehensive_perf_db.query_context_dsa_module(
            b=4,
            s=1024,
            prefix=0,
            num_heads=32,
            kvcache_quant_mode=common.KVCacheQuantMode.float16,
            fmha_quant_mode=common.FMHAQuantMode.float16,
            gemm_quant_mode=common.GEMMQuantMode.float16,
            database_mode=common.DatabaseMode.SOL,
        )
        assert r2 > r1

    def test_prefix_correction_increases_latency(self, comprehensive_perf_db):
        """With prefix > 0, the full_s is larger so SOL should increase."""
        no_prefix = comprehensive_perf_db.query_context_dsa_module(
            b=2,
            s=256,
            prefix=0,
            num_heads=32,
            kvcache_quant_mode=common.KVCacheQuantMode.float16,
            fmha_quant_mode=common.FMHAQuantMode.float16,
            gemm_quant_mode=common.GEMMQuantMode.float16,
            database_mode=common.DatabaseMode.SOL,
        )
        with_prefix = comprehensive_perf_db.query_context_dsa_module(
            b=2,
            s=256,
            prefix=256,
            num_heads=32,
            kvcache_quant_mode=common.KVCacheQuantMode.float16,
            fmha_quant_mode=common.FMHAQuantMode.float16,
            gemm_quant_mode=common.GEMMQuantMode.float16,
            database_mode=common.DatabaseMode.SOL,
        )
        assert with_prefix > no_prefix

    def test_empirical_returns_positive(self, comprehensive_perf_db):
        result = comprehensive_perf_db.query_context_dsa_module(
            b=2,
            s=256,
            prefix=0,
            num_heads=32,
            kvcache_quant_mode=common.KVCacheQuantMode.float16,
            fmha_quant_mode=common.FMHAQuantMode.float16,
            gemm_quant_mode=common.GEMMQuantMode.float16,
            database_mode=common.DatabaseMode.EMPIRICAL,
        )
        assert float(result) > 0

    def test_hybrid_falls_back_to_empirical_when_no_data(self, comprehensive_perf_db):
        """HYBRID mode should fallback to empirical when no silicon data loaded."""
        result = comprehensive_perf_db.query_context_dsa_module(
            b=2,
            s=256,
            prefix=0,
            num_heads=32,
            kvcache_quant_mode=common.KVCacheQuantMode.float16,
            fmha_quant_mode=common.FMHAQuantMode.float16,
            gemm_quant_mode=common.GEMMQuantMode.float16,
            database_mode=common.DatabaseMode.HYBRID,
        )
        assert float(result) > 0

    def test_different_index_params_change_sol(self, comprehensive_perf_db):
        """Different index_topk should yield different SOL estimates."""
        r1 = comprehensive_perf_db.query_context_dsa_module(
            b=4,
            s=4096,
            prefix=0,
            num_heads=32,
            index_topk=2048,
            kvcache_quant_mode=common.KVCacheQuantMode.float16,
            fmha_quant_mode=common.FMHAQuantMode.float16,
            gemm_quant_mode=common.GEMMQuantMode.float16,
            database_mode=common.DatabaseMode.SOL,
        )
        r2 = comprehensive_perf_db.query_context_dsa_module(
            b=4,
            s=4096,
            prefix=0,
            num_heads=32,
            index_topk=512,
            kvcache_quant_mode=common.KVCacheQuantMode.float16,
            fmha_quant_mode=common.FMHAQuantMode.float16,
            gemm_quant_mode=common.GEMMQuantMode.float16,
            database_mode=common.DatabaseMode.SOL,
        )
        assert r1 != r2


# ═══════════════════════════════════════════════════════════════════════
# Generation DSA Module
# ═══════════════════════════════════════════════════════════════════════


class TestGenerationDSAModule:
    """Tests for query_generation_dsa_module."""

    def test_sol_returns_positive(self, comprehensive_perf_db):
        result = comprehensive_perf_db.query_generation_dsa_module(
            b=4,
            s=1024,
            num_heads=32,
            kv_cache_dtype=common.KVCacheQuantMode.float16,
            gemm_quant_mode=common.GEMMQuantMode.float16,
            database_mode=common.DatabaseMode.SOL,
        )
        assert float(result) > 0

    def test_sol_full_returns_three_tuple(self, comprehensive_perf_db):
        result = comprehensive_perf_db.query_generation_dsa_module(
            b=4,
            s=1024,
            num_heads=32,
            kv_cache_dtype=common.KVCacheQuantMode.float16,
            gemm_quant_mode=common.GEMMQuantMode.float16,
            database_mode=common.DatabaseMode.SOL_FULL,
        )
        assert len(result) == 3
        sol_time, sol_math, sol_mem = result
        assert math.isclose(sol_time, max(sol_math, sol_mem), rel_tol=1e-6)

    def test_sol_increases_with_batch_size(self, comprehensive_perf_db):
        r1 = comprehensive_perf_db.query_generation_dsa_module(
            b=1,
            s=1024,
            num_heads=32,
            kv_cache_dtype=common.KVCacheQuantMode.float16,
            gemm_quant_mode=common.GEMMQuantMode.float16,
            database_mode=common.DatabaseMode.SOL,
        )
        r2 = comprehensive_perf_db.query_generation_dsa_module(
            b=64,
            s=1024,
            num_heads=32,
            kv_cache_dtype=common.KVCacheQuantMode.float16,
            gemm_quant_mode=common.GEMMQuantMode.float16,
            database_mode=common.DatabaseMode.SOL,
        )
        assert r2 > r1

    def test_different_index_topk_changes_sol(self, comprehensive_perf_db):
        r1 = comprehensive_perf_db.query_generation_dsa_module(
            b=4,
            s=4096,
            num_heads=32,
            index_topk=2048,
            kv_cache_dtype=common.KVCacheQuantMode.float16,
            gemm_quant_mode=common.GEMMQuantMode.float16,
            database_mode=common.DatabaseMode.SOL,
        )
        r2 = comprehensive_perf_db.query_generation_dsa_module(
            b=4,
            s=4096,
            num_heads=32,
            index_topk=512,
            kv_cache_dtype=common.KVCacheQuantMode.float16,
            gemm_quant_mode=common.GEMMQuantMode.float16,
            database_mode=common.DatabaseMode.SOL,
        )
        assert r2 < r1

    def test_empirical_returns_positive(self, comprehensive_perf_db):
        result = comprehensive_perf_db.query_generation_dsa_module(
            b=4,
            s=1024,
            num_heads=32,
            kv_cache_dtype=common.KVCacheQuantMode.float16,
            gemm_quant_mode=common.GEMMQuantMode.float16,
            database_mode=common.DatabaseMode.EMPIRICAL,
        )
        assert float(result) > 0

    def test_hybrid_falls_back_when_no_data(self, comprehensive_perf_db):
        result = comprehensive_perf_db.query_generation_dsa_module(
            b=4,
            s=1024,
            num_heads=32,
            kv_cache_dtype=common.KVCacheQuantMode.float16,
            gemm_quant_mode=common.GEMMQuantMode.float16,
            database_mode=common.DatabaseMode.HYBRID,
        )
        assert float(result) > 0

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from aiconfigurator.sdk import common


class TestContextAttention:
    """Test cases for query_context_attention method."""

    def test_query_context_attention_database_mode(self, comprehensive_perf_db):
        """Test SOL mode calculation for context attention."""
        b, full_s, prefix, n, n_kv = 2, 64, 0, 16, 8
        s = full_s - prefix
        kv_cache_quant_mode = common.KVCacheQuantMode.float16
        fmha_quant_mode = common.FMHAQuantMode.float16

        result = comprehensive_perf_db.query_context_attention(
            b, s, prefix, n, n_kv, kv_cache_quant_mode, fmha_quant_mode, database_mode=common.DatabaseMode.SOL
        )

        # Calculate expected SOL result
        ops = (
            2 * b * (full_s * full_s - prefix * prefix) * n * 128 * 2 / 2
        )  # 2 for fma, 2 for q*k^t+*v, 2 for causality
        mem_bytes = 2 * b * (n * s * 128 + 2 * n_kv * full_s * 128 + n * s * 128)

        sol_math = (
            ops / comprehensive_perf_db.system_spec["gpu"]["float16_tc_flops"] * 1000 / fmha_quant_mode.value.compute
        )
        sol_mem = mem_bytes / comprehensive_perf_db.system_spec["gpu"]["mem_bw"] * 1000
        expected = max(sol_math, sol_mem)

        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_query_context_attention_sol_full_mode(self, comprehensive_perf_db):
        """Test SOL_FULL mode returns (sol_time, sol_math, sol_mem)."""
        b, full_s, prefix, n, n_kv = 1, 32, 0, 8, 4
        s = full_s - prefix
        kv_cache_quant_mode = common.KVCacheQuantMode.float16
        fmha_quant_mode = common.FMHAQuantMode.float16

        sol_time, sol_math, sol_mem = comprehensive_perf_db.query_context_attention(
            b, s, prefix, n, n_kv, kv_cache_quant_mode, fmha_quant_mode, database_mode=common.DatabaseMode.SOL_FULL
        )

        sol_only = comprehensive_perf_db.query_context_attention(
            b, s, prefix, n, n_kv, kv_cache_quant_mode, fmha_quant_mode, database_mode=common.DatabaseMode.SOL
        )
        assert sol_time > 0
        assert math.isclose(sol_time, float(sol_only), rel_tol=1e-6)
        assert math.isclose(sol_time, max(sol_math, sol_mem), rel_tol=1e-6)

    def test_query_context_attention_non_database_mode_mha(self, comprehensive_perf_db):
        """Test SILICON mode with MHA (n_kv == n)."""
        b, full_s, prefix, n = 2, 32, 0, 16
        s = full_s - prefix
        n_kv = n  # MHA case
        kv_cache_quant_mode = common.KVCacheQuantMode.float16
        fmha_quant_mode = common.FMHAQuantMode.float16

        result = comprehensive_perf_db.query_context_attention(
            b, s, prefix, n, n_kv, kv_cache_quant_mode, fmha_quant_mode, database_mode=common.DatabaseMode.SILICON
        )

        # Should use data from attention_dict[0] for MHA
        expected = comprehensive_perf_db._context_attention_data[fmha_quant_mode][kv_cache_quant_mode][0][128][0][n][s][
            b
        ]
        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_query_context_attention_non_database_mode_xqa(self, comprehensive_perf_db):
        """Test SILICON mode with XQA (n_kv < n)."""
        b, full_s, prefix, n, n_kv = 2, 32, 0, 16, 4
        s = full_s - prefix
        kv_cache_quant_mode = common.KVCacheQuantMode.float16
        fmha_quant_mode = common.FMHAQuantMode.float16

        result = comprehensive_perf_db.query_context_attention(
            b, s, prefix, n, n_kv, kv_cache_quant_mode, fmha_quant_mode, database_mode=common.DatabaseMode.SILICON
        )

        # Should use data from attention_dict[n_kv] for XQA
        expected = comprehensive_perf_db._context_attention_data[fmha_quant_mode][kv_cache_quant_mode][n_kv][128][0][n][
            s
        ][b]
        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_query_context_attention_non_sol_mode_small_s(self, comprehensive_perf_db):
        """
        Test that query context attention works even when s is smaller than what exists
        in the collected data.
        """
        # Testing s = 1, but in comprehensive_perf_db, smallest s is 16.
        b, s, prefix, n, n_kv = 2, 1, 0, 16, 4
        kv_cache_quant_mode = common.KVCacheQuantMode.float16
        fmha_quant_mode = common.FMHAQuantMode.float16

        result = comprehensive_perf_db.query_context_attention(
            b, s, prefix, n, n_kv, kv_cache_quant_mode, fmha_quant_mode, database_mode=common.DatabaseMode.SILICON
        )
        assert result > 0

    def test_query_context_attention_assertion_error(self, comprehensive_perf_db):
        """Test that n_kv > n raises assertion error."""
        with pytest.raises(AssertionError):
            comprehensive_perf_db.query_context_attention(
                1,
                32,
                0,
                8,
                16,  # n_kv=16 > n=8
                common.KVCacheQuantMode.float16,
                common.FMHAQuantMode.float16,
            )


class TestGenerationAttention:
    """Test cases for query_generation_attention method."""

    def test_query_generation_attention_database_mode(self, comprehensive_perf_db):
        """Test SOL mode calculation for generation attention."""
        b, s, n, n_kv = 4, 128, 32, 8
        kv_cache_quant_mode = common.KVCacheQuantMode.float16

        result = comprehensive_perf_db.query_generation_attention(
            b, s, n, n_kv, kv_cache_quant_mode, database_mode=common.DatabaseMode.SOL
        )
        kv_len = s - 1
        # Calculate expected SOL result
        ops = 2 * b * n * 128 * 2 * (kv_len)  # 2 for fma, 2 for q*k^t+*v
        mem_bytes = b * (n * 128 * 2 + 2 * n_kv * kv_len * 128 * kv_cache_quant_mode.value.memory + n * 128 * 2)
        sol_math = ops / comprehensive_perf_db.system_spec["gpu"]["float16_tc_flops"] * 1000
        sol_mem = mem_bytes / comprehensive_perf_db.system_spec["gpu"]["mem_bw"] * 1000
        expected = max(sol_math, sol_mem)

        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_query_generation_attention_sol_full_mode(self, comprehensive_perf_db):
        """Test SOL_FULL mode returns (sol_time, sol_math, sol_mem)."""
        b, s, n, n_kv = 2, 64, 16, 4
        kv_cache_quant_mode = common.KVCacheQuantMode.fp8

        sol_time, sol_math, sol_mem = comprehensive_perf_db.query_generation_attention(
            b, s, n, n_kv, kv_cache_quant_mode, database_mode=common.DatabaseMode.SOL_FULL
        )

        sol_only = comprehensive_perf_db.query_generation_attention(
            b, s, n, n_kv, kv_cache_quant_mode, database_mode=common.DatabaseMode.SOL
        )
        assert sol_time > 0
        assert math.isclose(sol_time, float(sol_only), rel_tol=1e-6)
        assert math.isclose(sol_time, max(sol_math, sol_mem), rel_tol=1e-6)

    def test_query_generation_attention_non_database_mode(self, comprehensive_perf_db):
        """Test SILICON mode with interpolation."""
        b, s, n, n_kv = 2, 64, 16, 8
        kv_cache_quant_mode = common.KVCacheQuantMode.float16

        result = comprehensive_perf_db.query_generation_attention(
            b, s, n, n_kv, kv_cache_quant_mode, database_mode=common.DatabaseMode.SILICON
        )

        # Should use interpolation from generation_attention_data
        assert isinstance(result, float)
        assert result > 0

    def test_query_generation_attention_non_database_mode_mha(self, comprehensive_perf_db):
        """Test SILICON mode with MHA (n_kv == n)."""
        b, s, n = 2, 64, 16
        n_kv = n  # MHA case
        kv_cache_quant_mode = common.KVCacheQuantMode.float16

        result = comprehensive_perf_db.query_generation_attention(
            b, s, n, n_kv, kv_cache_quant_mode, database_mode=common.DatabaseMode.SILICON
        )

        # Should use n_kv=0 for MHA
        expected = comprehensive_perf_db._generation_attention_data[kv_cache_quant_mode][0][128][0][n][b][s]

        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_query_generation_attention_edge_cases(self, comprehensive_perf_db):
        """Test edge cases like s=1."""
        # When s=1, there's no KV cache to load from previous steps
        result = comprehensive_perf_db.query_generation_attention(
            1, 1, 8, 4, common.KVCacheQuantMode.float16, database_mode=common.DatabaseMode.SOL
        )
        assert result > 0


class TestContextMLA:
    """Test cases for query_context_mla method."""

    def test_query_context_mla_database_mode(self, comprehensive_perf_db):
        """Test SOL mode calculation for context MLA."""
        b, s, prefix, num_heads = 2, 64, 0, 32
        kv_cache_quant_mode = common.KVCacheQuantMode.float16
        fmha_quant_mode = common.FMHAQuantMode.float16

        result = comprehensive_perf_db.query_context_mla(
            b, s, prefix, num_heads, kv_cache_quant_mode, fmha_quant_mode, database_mode=common.DatabaseMode.SOL
        )

        # Calculate expected SOL result
        ops = (
            b * num_heads * 2 / 2 * (192 + 128) * (s * s - prefix * prefix)
        )  # 2 for fma, 2 for causality. num_heads, for local heads
        # s * 192 for q read, full_s * 192 for k read, full_s * 128 for v read, s * 192 for write.
        mem_bytes = b * num_heads * 2 * (s * (192 + 128) + (s - prefix) * (192 + 128))  # 2 for fp16, TODO
        sol_math = (
            ops / comprehensive_perf_db.system_spec["gpu"]["float16_tc_flops"] * 1000 / fmha_quant_mode.value.compute
        )
        sol_mem = mem_bytes / comprehensive_perf_db.system_spec["gpu"]["mem_bw"] * 1000
        expected = max(sol_math, sol_mem)

        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_query_context_mla_non_database_mode(self, comprehensive_perf_db):
        """Test SILICON mode with interpolation."""
        b, s, prefix, num_heads = 4, 32, 0, 32
        kv_cache_quant_mode = common.KVCacheQuantMode.float16
        fmha_quant_mode = common.FMHAQuantMode.float16

        result = comprehensive_perf_db.query_context_mla(
            b, s, prefix, num_heads, kv_cache_quant_mode, fmha_quant_mode, database_mode=common.DatabaseMode.SILICON
        )

        # Should use data from context_mla_data
        expected = comprehensive_perf_db._context_mla_data[fmha_quant_mode][kv_cache_quant_mode][num_heads][s][b]
        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_query_context_mla_different_tp_sizes(self, comprehensive_perf_db):
        """Test MLA with different tensor parallelism sizes."""
        b, s = 2, 64
        kv_cache_quant_mode = common.KVCacheQuantMode.float16
        fmha_quant_mode = common.FMHAQuantMode.float16

        results = []
        for num_heads in [16, 32, 64, 128]:
            result = comprehensive_perf_db.query_context_mla(
                b,
                s,
                0,
                num_heads,
                kv_cache_quant_mode,
                fmha_quant_mode,
                database_mode=common.DatabaseMode.SILICON,
            )
            results.append(result)

        # Generally, larger TP should result in lower latency per GPU
        assert all(r > 0 for r in results)


class TestGenerationMLA:
    """Test cases for query_generation_mla method."""

    def test_query_generation_mla_database_mode(self, comprehensive_perf_db):
        """Test SOL mode calculation for generation MLA."""
        b, s, num_heads = 4, 128, 32
        kv_cache_quant_mode = common.KVCacheQuantMode.float16

        result = comprehensive_perf_db.query_generation_mla(
            b, s, num_heads, kv_cache_quant_mode, database_mode=common.DatabaseMode.SOL
        )

        # Calculate expected SOL result
        n = num_heads
        ops = 2 * b * n * 1088 * s  # 2 for fma
        mem_bytes = b * (n * 1088 * 2 + (s - 1) * 1088 * kv_cache_quant_mode.value.memory)
        sol_math = ops / comprehensive_perf_db.system_spec["gpu"]["float16_tc_flops"] * 1000
        sol_mem = mem_bytes / comprehensive_perf_db.system_spec["gpu"]["mem_bw"] * 1000
        expected = max(sol_math, sol_mem)

        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_query_generation_mla_non_database_mode(self, comprehensive_perf_db):
        """Test SILICON mode with interpolation."""
        b, s, num_heads = 2, 64, 32
        kv_cache_quant_mode = common.KVCacheQuantMode.float16

        result = comprehensive_perf_db.query_generation_mla(
            b, s, num_heads, kv_cache_quant_mode, database_mode=common.DatabaseMode.SILICON
        )

        # Should use data from generation_mla_data
        expected = comprehensive_perf_db._generation_mla_data[kv_cache_quant_mode][num_heads][b][s]
        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_query_generation_mla_sol_full_mode(self, comprehensive_perf_db):
        """Test SOL_FULL mode returns (sol_time, sol_math, sol_mem)."""
        sol_time, sol_math, sol_mem = comprehensive_perf_db.query_generation_mla(
            1, 32, 32, common.KVCacheQuantMode.float16, database_mode=common.DatabaseMode.SOL_FULL
        )

        sol_only = comprehensive_perf_db.query_generation_mla(
            1, 32, 32, common.KVCacheQuantMode.float16, database_mode=common.DatabaseMode.SOL
        )
        assert sol_time > 0
        assert math.isclose(sol_time, float(sol_only), rel_tol=1e-6)
        assert math.isclose(sol_time, max(sol_math, sol_mem), rel_tol=1e-6)


def test_default_database_mode(comprehensive_perf_db):
    """Test setting and getting default database mode, and that query cache is cleared when default mode is changed."""
    # Initially should be SILICON
    assert comprehensive_perf_db.get_default_database_mode() == common.DatabaseMode.SILICON

    non_sol_result = comprehensive_perf_db.query_context_attention(
        1, 32, 0, 8, 4, common.KVCacheQuantMode.float16, common.FMHAQuantMode.float16
    )
    assert comprehensive_perf_db.query_context_attention.cache_info().currsize >= 1

    # Set to SOL mode
    comprehensive_perf_db.set_default_database_mode(common.DatabaseMode.SOL)
    assert comprehensive_perf_db.get_default_database_mode() == common.DatabaseMode.SOL
    # Cache should be cleared
    assert comprehensive_perf_db.query_context_attention.cache_info().currsize == 0

    # Query should use default mode when not specified
    sol_result = comprehensive_perf_db.query_context_attention(
        1, 32, 0, 8, 4, common.KVCacheQuantMode.float16, common.FMHAQuantMode.float16
    )

    cache_info = comprehensive_perf_db.query_context_attention.cache_info()
    assert cache_info.misses == 1
    assert cache_info.hits == 0
    assert cache_info.currsize == 1
    assert isinstance(sol_result, float)
    assert sol_result != non_sol_result

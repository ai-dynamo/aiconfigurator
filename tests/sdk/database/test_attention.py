# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import math
from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import PerfDatabase

class TestContextAttention:
    """Test cases for query_context_attention method."""
    
    def test_query_context_attention_sol_mode(self, comprehensive_perf_db):
        """Test SOL mode calculation for context attention."""
        b, s, n, n_kv = 2, 64, 16, 8
        kv_cache_quant_mode = common.KVCacheQuantMode.float16
        fmha_quant_mode = common.FMHAQuantMode.float16
        
        result = comprehensive_perf_db.query_context_attention(
            b, s, n, n_kv, kv_cache_quant_mode, fmha_quant_mode, 
            sol_mode=common.SOLMode.SOL
        )
        
        # Calculate expected SOL result
        ops = 2 * b * s * s * n * 128 * 2 / 2  # 2 for fma, 2 for q*k^t+*v, 2 for causality
        mem_bytes = 2 * b * (n*s*128 + 2*n_kv*s*128 + n*s*128)
        sol_math = ops / comprehensive_perf_db.system_spec['gpu']['float16_tc_flops'] * 1000 / fmha_quant_mode.value.compute
        sol_mem = mem_bytes / comprehensive_perf_db.system_spec['gpu']['mem_bw'] * 1000
        expected = max(sol_math, sol_mem)
        
        assert math.isclose(result, expected, rel_tol=1e-6)
    
    def test_query_context_attention_sol_full_mode(self, comprehensive_perf_db):
        """Test SOL_FULL mode returns tuple with math and memory components."""
        b, s, n, n_kv = 1, 32, 8, 4
        kv_cache_quant_mode = common.KVCacheQuantMode.float16
        fmha_quant_mode = common.FMHAQuantMode.float16
        
        result = comprehensive_perf_db.query_context_attention(
            b, s, n, n_kv, kv_cache_quant_mode, fmha_quant_mode,
            sol_mode=common.SOLMode.SOL_FULL
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert result[0] == max(result[1], result[2])  # sol_time = max(sol_math, sol_mem)
    
    def test_query_context_attention_non_sol_mode_mha(self, comprehensive_perf_db):
        """Test non-SOL mode with MHA (n_kv == n)."""
        b, s, n = 2, 32, 16
        n_kv = n  # MHA case
        kv_cache_quant_mode = common.KVCacheQuantMode.float16
        fmha_quant_mode = common.FMHAQuantMode.float16
        
        result = comprehensive_perf_db.query_context_attention(
            b, s, n, n_kv, kv_cache_quant_mode, fmha_quant_mode,
            sol_mode=common.SOLMode.NON_SOL
        )
        
        # Should use data from attention_dict[0] for MHA
        expected = comprehensive_perf_db._context_attention_data[fmha_quant_mode][kv_cache_quant_mode][0][n][s][b]
        assert math.isclose(result, expected, rel_tol=1e-6)
    
    def test_query_context_attention_non_sol_mode_xqa(self, comprehensive_perf_db):
        """Test non-SOL mode with XQA (n_kv < n)."""
        b, s, n, n_kv = 2, 32, 16, 4
        kv_cache_quant_mode = common.KVCacheQuantMode.float16
        fmha_quant_mode = common.FMHAQuantMode.float16
        
        result = comprehensive_perf_db.query_context_attention(
            b, s, n, n_kv, kv_cache_quant_mode, fmha_quant_mode,
            sol_mode=common.SOLMode.NON_SOL
        )
        
        # Should use data from attention_dict[n_kv] for XQA
        expected = comprehensive_perf_db._context_attention_data[fmha_quant_mode][kv_cache_quant_mode][n_kv][n][s][b]
        assert math.isclose(result, expected, rel_tol=1e-6)
    
    def test_query_context_attention_assertion_error(self, comprehensive_perf_db):
        """Test that n_kv > n raises assertion error."""
        with pytest.raises(AssertionError):
            comprehensive_perf_db.query_context_attention(
                1, 32, 8, 16,  # n_kv=16 > n=8
                common.KVCacheQuantMode.float16,
                common.FMHAQuantMode.float16
            )


class TestGenerationAttention:
    """Test cases for query_generation_attention method."""
    
    def test_query_generation_attention_sol_mode(self, comprehensive_perf_db):
        """Test SOL mode calculation for generation attention."""
        b, s, n, n_kv = 4, 128, 32, 8
        kv_cache_quant_mode = common.KVCacheQuantMode.float16
        
        result = comprehensive_perf_db.query_generation_attention(
            b, s, n, n_kv, kv_cache_quant_mode,
            sol_mode=common.SOLMode.SOL
        )
        
        # Calculate expected SOL result
        ops = 2 * b * n * 128 * 2 * s  # 2 for fma, 2 for q*k^t+*v
        mem_bytes = b * (n*128*2 + 2*n_kv*(s-1)*128*kv_cache_quant_mode.value.memory + n*128*2)
        sol_math = ops / comprehensive_perf_db.system_spec['gpu']['float16_tc_flops'] * 1000
        sol_mem = mem_bytes / comprehensive_perf_db.system_spec['gpu']['mem_bw'] * 1000
        expected = max(sol_math, sol_mem)
        
        assert math.isclose(result, expected, rel_tol=1e-6)
    
    def test_query_generation_attention_sol_full_mode(self, comprehensive_perf_db):
        """Test SOL_FULL mode returns tuple."""
        b, s, n, n_kv = 2, 64, 16, 4
        kv_cache_quant_mode = common.KVCacheQuantMode.fp8
        
        result = comprehensive_perf_db.query_generation_attention(
            b, s, n, n_kv, kv_cache_quant_mode,
            sol_mode=common.SOLMode.SOL_FULL
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(x, (int, float)) for x in result)
    
    def test_query_generation_attention_non_sol_mode(self, comprehensive_perf_db):
        """Test non-SOL mode with interpolation."""
        b, s, n, n_kv = 2, 64, 16, 8
        kv_cache_quant_mode = common.KVCacheQuantMode.float16
        
        result = comprehensive_perf_db.query_generation_attention(
            b, s, n, n_kv, kv_cache_quant_mode,
            sol_mode=common.SOLMode.NON_SOL
        )
        
        # Should use interpolation from generation_attention_data
        assert isinstance(result, float)
        assert result > 0
    
    def test_query_generation_attention_edge_cases(self, comprehensive_perf_db):
        """Test edge cases like s=1."""
        # When s=1, there's no KV cache to load from previous steps
        result = comprehensive_perf_db.query_generation_attention(
            1, 1, 8, 4, common.KVCacheQuantMode.float16,
            sol_mode=common.SOLMode.SOL
        )
        assert result > 0


class TestContextMLA:
    """Test cases for query_context_mla method."""
    
    def test_query_context_mla_sol_mode(self, comprehensive_perf_db):
        """Test SOL mode calculation for context MLA."""
        b, s, num_heads = 2, 64, 32
        kv_cache_quant_mode = common.KVCacheQuantMode.float16
        fmha_quant_mode = common.FMHAQuantMode.float16
        
        result = comprehensive_perf_db.query_context_mla(
            b, s, num_heads, kv_cache_quant_mode, fmha_quant_mode,
            sol_mode=common.SOLMode.SOL
        )
        
        # Calculate expected SOL result
        ops = b * num_heads * 2 / 2 * (s * s * 192 + s * s * 128)
        mem_bytes = b * num_heads * 2 * (2*s*192 + 2*s*128)
        sol_math = ops / comprehensive_perf_db.system_spec['gpu']['float16_tc_flops'] * 1000 / fmha_quant_mode.value.compute
        sol_mem = mem_bytes / comprehensive_perf_db.system_spec['gpu']['mem_bw'] * 1000
        expected = max(sol_math, sol_mem)
        
        assert math.isclose(result, expected, rel_tol=1e-6)
    
    def test_query_context_mla_non_sol_mode(self, comprehensive_perf_db):
        """Test non-SOL mode with interpolation."""
        b, s, num_heads = 4, 32, 32
        kv_cache_quant_mode = common.KVCacheQuantMode.float16
        fmha_quant_mode = common.FMHAQuantMode.float16
        
        result = comprehensive_perf_db.query_context_mla(
            b, s, num_heads, kv_cache_quant_mode, fmha_quant_mode,
            sol_mode=common.SOLMode.NON_SOL
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
                b, s, num_heads, kv_cache_quant_mode, fmha_quant_mode,
                sol_mode=common.SOLMode.NON_SOL
            )
            results.append(result)
        
        # Generally, larger TP should result in lower latency per GPU
        assert all(r > 0 for r in results)


class TestGenerationMLA:
    """Test cases for query_generation_mla method."""
    
    def test_query_generation_mla_sol_mode(self, comprehensive_perf_db):
        """Test SOL mode calculation for generation MLA."""
        b, s, num_heads = 4, 128, 32
        kv_cache_quant_mode = common.KVCacheQuantMode.float16
        
        result = comprehensive_perf_db.query_generation_mla(
            b, s, num_heads, kv_cache_quant_mode,
            sol_mode=common.SOLMode.SOL
        )
        
        # Calculate expected SOL result
        n = num_heads
        ops = 2 * b * n * 1088 * s  # 2 for fma
        mem_bytes = b * (n * 1088 * 2 + (s-1)*1088 * kv_cache_quant_mode.value.memory)
        sol_math = ops / comprehensive_perf_db.system_spec['gpu']['float16_tc_flops'] * 1000
        sol_mem = mem_bytes / comprehensive_perf_db.system_spec['gpu']['mem_bw'] * 1000
        expected = max(sol_math, sol_mem)
        
        assert math.isclose(result, expected, rel_tol=1e-6)
    
    def test_query_generation_mla_non_sol_mode(self, comprehensive_perf_db):
        """Test non-SOL mode with interpolation."""
        b, s, num_heads = 2, 64, 32
        kv_cache_quant_mode = common.KVCacheQuantMode.float16
        
        result = comprehensive_perf_db.query_generation_mla(
            b, s, num_heads, kv_cache_quant_mode,
            sol_mode=common.SOLMode.NON_SOL
        )
        
        # Should use data from generation_mla_data
        expected = comprehensive_perf_db._generation_mla_data[kv_cache_quant_mode][num_heads][b][s]
        assert math.isclose(result, expected, rel_tol=1e-6)
    
    def test_query_generation_mla_sol_full_mode(self, comprehensive_perf_db):
        """Test SOL_FULL mode returns complete tuple."""
        result = comprehensive_perf_db.query_generation_mla(
            1, 32, 32, common.KVCacheQuantMode.float16,
            sol_mode=common.SOLMode.SOL_FULL
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert result[0] == max(result[1], result[2])


def test_default_sol_mode(comprehensive_perf_db):
    """Test setting and getting default SOL mode."""
    # Initially should be NON_SOL
    assert comprehensive_perf_db.get_default_sol_mode() == common.SOLMode.NON_SOL
    
    # Set to SOL mode
    comprehensive_perf_db.set_default_sol_mode(common.SOLMode.SOL)
    assert comprehensive_perf_db.get_default_sol_mode() == common.SOLMode.SOL
    
    # Query should use default mode when not specified
    result = comprehensive_perf_db.query_context_attention(
        1, 32, 8, 4,
        common.KVCacheQuantMode.float16,
        common.FMHAQuantMode.float16
    )
    assert isinstance(result, float) 
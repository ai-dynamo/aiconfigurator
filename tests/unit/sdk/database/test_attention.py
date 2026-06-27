# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import math

import pytest

from aiconfigurator.sdk import common, interpolation

pytestmark = pytest.mark.unit


class TestContextAttention:
    """Test cases for query_context_attention method."""

    def test_query_context_attention_database_mode(self, comprehensive_perf_db):
        """Test SOL mode calculation for context attention."""
        b, full_s, prefix, n, n_kv = 2, 64, 0, 16, 8
        s = full_s - prefix
        kv_cache_quant_mode = common.KVCacheQuantMode.bfloat16
        fmha_quant_mode = common.FMHAQuantMode.bfloat16

        result = comprehensive_perf_db.query_context_attention(
            b, s, prefix, n, n_kv, kv_cache_quant_mode, fmha_quant_mode, database_mode=common.DatabaseMode.SOL
        )

        # Calculate expected SOL result
        ops = (
            2 * b * (full_s * full_s - prefix * prefix) * n * 128 * 2 / 2
        )  # 2 for fma, 2 for q*k^t+*v, 2 for causality
        mem_bytes = 2 * b * (n * s * 128 + 2 * n_kv * full_s * 128 + n * s * 128)

        sol_math = (
            ops / comprehensive_perf_db.system_spec["gpu"]["bfloat16_tc_flops"] * 1000 / fmha_quant_mode.value.compute
        )
        sol_mem = mem_bytes / comprehensive_perf_db.system_spec["gpu"]["mem_bw"] * 1000
        expected = max(sol_math, sol_mem)

        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_windowed_context_attention_not_above_full(self, comprehensive_perf_db):
        """SWA (windowed) context attention must never exceed full attention -- it does
        strictly less work. Regression for corrupt per-model windowed silicon data
        (hs192/win128 fp8 recorded ~83000x the full-attention latency, blowing up TTFT
        for hybrid models). Windowed latency is now derived from the window=0 measurement
        scaled by the window-aware SOL ratio, so the invariant holds by construction."""
        args = (2, 256, 0, 16, 8, common.KVCacheQuantMode.bfloat16, common.FMHAQuantMode.bfloat16)
        kw = dict(database_mode=common.DatabaseMode.HYBRID, head_size=128)
        full = float(comprehensive_perf_db.query_context_attention(*args, window_size=0, **kw))
        windowed = float(comprehensive_perf_db.query_context_attention(*args, window_size=128, **kw))
        assert windowed > 0
        assert windowed <= full * 1.0001  # s(256) > window(128): windowed work <= full

    def test_provenance_capture_records_transfer_tier(self, comprehensive_perf_db):
        """capture_provenance records which empirical tier a query relied on: a cross-head
        transfer (uncollected head_size) tags 'xshape'; a silicon-covered query tags nothing
        (worst_provenance -> 'silicon')."""
        from aiconfigurator.sdk.operations import util_empirical as ue

        base = (8, 256, 0, 16, 8, common.KVCacheQuantMode.bfloat16, common.FMHAQuantMode.bfloat16)
        with ue.capture_provenance() as tags:
            comprehensive_perf_db.query_context_attention(
                *base, database_mode=common.DatabaseMode.EMPIRICAL, head_size=256, window_size=0
            )  # head_size 256 not collected (stub has 64/128) -> cross-head transfer
        assert ue.worst_provenance(tags) == "xshape"
        with ue.capture_provenance() as tags2:
            comprehensive_perf_db.query_context_attention(
                *base, database_mode=common.DatabaseMode.SILICON, head_size=128, window_size=0
            )  # collected -> pure silicon, no empirical path
        assert ue.worst_provenance(tags2) == "silicon"

    def test_cross_head_transfer_gated_by_xshape_policy(self, comprehensive_perf_db):
        """Cross-head_size transfer is an XSHAPE transfer. A head_size with no own data
        (stub collects 64/128) borrows from the nearest collected head_size when XSHAPE
        is permitted, and raises when the transfer policy excludes XSHAPE."""
        from aiconfigurator.sdk.errors import EmpiricalNotImplementedError

        args = (8, 256, 0, 16, 8, common.KVCacheQuantMode.bfloat16, common.FMHAQuantMode.bfloat16)
        kw = dict(database_mode=common.DatabaseMode.EMPIRICAL, head_size=256, window_size=0)
        try:
            comprehensive_perf_db.set_transfer_policy("aggressive")  # XSHAPE on
            assert float(comprehensive_perf_db.query_context_attention(*args, **kw)) > 0
            comprehensive_perf_db.set_transfer_policy(["xquant"])  # no XSHAPE
            with pytest.raises(EmpiricalNotImplementedError):
                comprehensive_perf_db.query_context_attention(*args, **kw)
        finally:
            comprehensive_perf_db.set_transfer_policy(None)

    def test_query_context_attention_sol_full_mode(self, comprehensive_perf_db):
        """Test SOL_FULL mode returns (sol_time, sol_math, sol_mem)."""
        b, full_s, prefix, n, n_kv = 1, 32, 0, 8, 4
        s = full_s - prefix
        kv_cache_quant_mode = common.KVCacheQuantMode.bfloat16
        fmha_quant_mode = common.FMHAQuantMode.bfloat16

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
        kv_cache_quant_mode = common.KVCacheQuantMode.bfloat16
        fmha_quant_mode = common.FMHAQuantMode.bfloat16

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
        kv_cache_quant_mode = common.KVCacheQuantMode.bfloat16
        fmha_quant_mode = common.FMHAQuantMode.bfloat16

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
        kv_cache_quant_mode = common.KVCacheQuantMode.bfloat16
        fmha_quant_mode = common.FMHAQuantMode.bfloat16

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
                common.KVCacheQuantMode.bfloat16,
                common.FMHAQuantMode.bfloat16,
            )


class TestGenerationAttention:
    """Test cases for query_generation_attention method."""

    def test_query_generation_attention_database_mode(self, comprehensive_perf_db):
        """Test SOL mode calculation for generation attention."""
        b, s, n, n_kv = 4, 128, 32, 8
        kv_cache_quant_mode = common.KVCacheQuantMode.bfloat16

        result = comprehensive_perf_db.query_generation_attention(
            b, s, n, n_kv, kv_cache_quant_mode, database_mode=common.DatabaseMode.SOL
        )
        kv_len = s - 1
        # Calculate expected SOL result
        ops = 2 * b * n * 128 * 2 * (kv_len)  # 2 for fma, 2 for q*k^t+*v
        mem_bytes = b * (n * 128 * 2 + 2 * n_kv * kv_len * 128 * kv_cache_quant_mode.value.memory + n * 128 * 2)
        sol_math = ops / comprehensive_perf_db.system_spec["gpu"]["bfloat16_tc_flops"] * 1000
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
        kv_cache_quant_mode = common.KVCacheQuantMode.bfloat16

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
        kv_cache_quant_mode = common.KVCacheQuantMode.bfloat16

        result = comprehensive_perf_db.query_generation_attention(
            b, s, n, n_kv, kv_cache_quant_mode, database_mode=common.DatabaseMode.SILICON
        )

        # Should use n_kv=0 for MHA
        attention_dict = comprehensive_perf_db._generation_attention_data[kv_cache_quant_mode][0][128][0]
        s_min = max(1, int(s * 0.9))
        s_max = max(s_min, int(s * 1.1))
        sample_cnt = 5
        s_samples = [s_min + (s_max - s_min) * i // (sample_cnt - 1) for i in range(sample_cnt)]
        expected = (
            sum(
                interpolation.interp_3d(
                    n, b, s_i, attention_dict, "bilinear", comprehensive_perf_db._extracted_metrics_cache
                )["latency"]
                for s_i in s_samples
            )
            / sample_cnt
        )

        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_query_generation_attention_edge_cases(self, comprehensive_perf_db):
        """Test edge cases like s=1."""
        # When s=1, there's no KV cache to load from previous steps
        result = comprehensive_perf_db.query_generation_attention(
            1, 1, 8, 4, common.KVCacheQuantMode.bfloat16, database_mode=common.DatabaseMode.SOL
        )
        assert result > 0

    def test_raw_cache_is_preserved_before_working_grid_extrapolation(self, comprehensive_perf_db):
        from aiconfigurator.sdk.operations.attention import GenerationAttention

        GenerationAttention.load_data(comprehensive_perf_db)
        key = GenerationAttention._cache_key(comprehensive_perf_db)
        raw = GenerationAttention._raw_data_cache[key]
        working = GenerationAttention._data_cache[key]
        path = (common.KVCacheQuantMode.bfloat16, 1, 128, 0, 8)

        raw_curve = raw[path[0]][path[1]][path[2]][path[3]][path[4]]
        working_curve = working[path[0]][path[1]][path[2]][path[3]][path[4]]
        assert 8192 not in raw_curve
        assert 8192 in working_curve

    def test_exact_head_empirical_uses_ragged_bracketed_util(self, comprehensive_perf_db, monkeypatch):
        """An interior b=3/s~=15k query must not inherit the b=4 utilization.

        These are the real B200/SGLang FP8 neighbours behind the expanded
        fidelity matrix's former 33.34% generation-attention error.
        """
        from aiconfigurator.sdk.operations import util_empirical
        from aiconfigurator.sdk.perf_database import LoadedOpData

        quant = common.KVCacheQuantMode.fp8

        def leaf(latency):
            return {"latency": latency, "power": 0.0, "energy": 0.0}

        exact_head = {
            2: {
                8192: leaf(0.012430399656295776),
                16384: leaf(0.016531200706958772),
                32768: leaf(0.01656000018119812),
            },
            4: {
                8192: leaf(0.012478400021791458),
                16384: leaf(0.014590400457382201),
                32768: leaf(0.016420799493789672),
            },
        }
        # SILICON's generic 3-D interpolation validates that all axes vary,
        # even though n=8 is an exact hit.  A second head slice supplies that
        # variation but is deliberately not consulted by the exact-head path.
        table = {quant: {1: {128: {0: {8: exact_head, 16: copy.deepcopy(exact_head)}}}}}
        working = LoadedOpData(copy.deepcopy(table), common.PerfDataFilename.generation_attention, "unused")
        raw = LoadedOpData(copy.deepcopy(table), common.PerfDataFilename.generation_attention, "unused")

        monkeypatch.setattr(comprehensive_perf_db, "_generation_attention_data", working)
        monkeypatch.setattr(comprehensive_perf_db, "_raw_generation_attention_data", raw, raising=False)
        util_empirical.clear_grid_cache()
        comprehensive_perf_db.query_generation_attention.cache_clear()
        try:
            args = (3, 14996, 8, 1, quant)
            silicon = float(
                comprehensive_perf_db.query_generation_attention(
                    *args, database_mode=common.DatabaseMode.SILICON, head_size=128
                )
            )
            empirical = float(
                comprehensive_perf_db.query_generation_attention(
                    *args, database_mode=common.DatabaseMode.EMPIRICAL, head_size=128
                )
            )

            sol_query = float(
                comprehensive_perf_db.query_generation_attention(
                    *args, database_mode=common.DatabaseMode.SOL, head_size=128
                )
            )
            sol_nearest = float(
                comprehensive_perf_db.query_generation_attention(
                    4, 16384, 8, 1, quant, database_mode=common.DatabaseMode.SOL, head_size=128
                )
            )
            old_nearest = sol_query / (sol_nearest / exact_head[4][16384]["latency"])

            assert abs(old_nearest / silicon - 1.0) > 0.30
            assert abs(empirical / silicon - 1.0) < 0.04
        finally:
            util_empirical.clear_grid_cache()
            comprehensive_perf_db.query_generation_attention.cache_clear()

    def test_exact_head_util_cache_tracks_raw_data_identity(self, comprehensive_perf_db, monkeypatch):
        from aiconfigurator.sdk.operations import util_empirical
        from aiconfigurator.sdk.perf_database import LoadedOpData

        quant = common.KVCacheQuantMode.bfloat16

        def wrapper(latency):
            table = {
                quant: {
                    1: {
                        128: {
                            0: {
                                8: {
                                    2: {
                                        8192: {"latency": latency, "power": 0.0, "energy": 0.0},
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return LoadedOpData(table, common.PerfDataFilename.generation_attention, "unused")

        first_raw = wrapper(0.02)
        replacement_raw = wrapper(0.04)
        util_empirical.clear_grid_cache()
        try:
            monkeypatch.setattr(
                comprehensive_perf_db,
                "_raw_generation_attention_data",
                first_raw,
                raising=False,
            )
            comprehensive_perf_db.query_generation_attention.cache_clear()
            first = float(
                comprehensive_perf_db.query_generation_attention(
                    2,
                    8192,
                    8,
                    1,
                    quant,
                    database_mode=common.DatabaseMode.EMPIRICAL,
                    head_size=128,
                )
            )

            # Do not clear the process-global util cache: the raw data identity
            # in the cache key must select a newly calibrated grid.
            monkeypatch.setattr(comprehensive_perf_db, "_raw_generation_attention_data", replacement_raw)
            comprehensive_perf_db.query_generation_attention.cache_clear()
            replacement = float(
                comprehensive_perf_db.query_generation_attention(
                    2,
                    8192,
                    8,
                    1,
                    quant,
                    database_mode=common.DatabaseMode.EMPIRICAL,
                    head_size=128,
                )
            )

            assert first == pytest.approx(0.02)
            assert replacement == pytest.approx(0.04)
        finally:
            util_empirical.clear_grid_cache()
            comprehensive_perf_db.query_generation_attention.cache_clear()


class TestContextMLA:
    """Test cases for query_context_mla method."""

    def test_query_context_mla_database_mode(self, comprehensive_perf_db):
        """Test SOL mode calculation for context MLA."""
        b, s, prefix, num_heads = 2, 64, 0, 32
        kv_cache_quant_mode = common.KVCacheQuantMode.bfloat16
        fmha_quant_mode = common.FMHAQuantMode.bfloat16

        result = comprehensive_perf_db.query_context_mla(
            b, s, prefix, num_heads, kv_cache_quant_mode, fmha_quant_mode, database_mode=common.DatabaseMode.SOL
        )

        # Calculate expected SOL result
        ops = (
            b * num_heads * 2 / 2 * (192 + 128) * (s * s - prefix * prefix)
        )  # 2 for fma, 2 for causality. num_heads, for local heads
        # s * 192 for q read, full_s * 192 for k read, full_s * 128 for v read, s * 192 for write.
        mem_bytes = b * num_heads * 2 * (s * (192 + 128) + (s - prefix) * (192 + 128))  # 2 for bfloat16, TODO
        sol_math = (
            ops / comprehensive_perf_db.system_spec["gpu"]["bfloat16_tc_flops"] * 1000 / fmha_quant_mode.value.compute
        )
        sol_mem = mem_bytes / comprehensive_perf_db.system_spec["gpu"]["mem_bw"] * 1000
        expected = max(sol_math, sol_mem)

        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_query_context_mla_non_database_mode(self, comprehensive_perf_db):
        """Test SILICON mode with interpolation."""
        b, s, prefix, num_heads = 4, 32, 0, 32
        kv_cache_quant_mode = common.KVCacheQuantMode.bfloat16
        fmha_quant_mode = common.FMHAQuantMode.bfloat16

        result = comprehensive_perf_db.query_context_mla(
            b, s, prefix, num_heads, kv_cache_quant_mode, fmha_quant_mode, database_mode=common.DatabaseMode.SILICON
        )

        # Should use data from context_mla_data
        expected = comprehensive_perf_db._context_mla_data[fmha_quant_mode][kv_cache_quant_mode][num_heads][s][b]
        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_query_context_mla_different_tp_sizes(self, comprehensive_perf_db):
        """Test MLA with different tensor parallelism sizes."""
        b, s = 2, 64
        kv_cache_quant_mode = common.KVCacheQuantMode.bfloat16
        fmha_quant_mode = common.FMHAQuantMode.bfloat16

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
        kv_cache_quant_mode = common.KVCacheQuantMode.bfloat16

        result = comprehensive_perf_db.query_generation_mla(
            b, s, num_heads, kv_cache_quant_mode, database_mode=common.DatabaseMode.SOL
        )

        # Calculate expected SOL result
        n = num_heads
        ops = 2 * b * n * 1088 * s  # 2 for fma
        mem_bytes = b * (n * 1088 * 2 + (s - 1) * 1088 * kv_cache_quant_mode.value.memory)
        sol_math = ops / comprehensive_perf_db.system_spec["gpu"]["bfloat16_tc_flops"] * 1000
        sol_mem = mem_bytes / comprehensive_perf_db.system_spec["gpu"]["mem_bw"] * 1000
        expected = max(sol_math, sol_mem)

        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_query_generation_mla_non_database_mode(self, comprehensive_perf_db):
        """Test SILICON mode with interpolation."""
        b, s, num_heads = 2, 64, 32
        kv_cache_quant_mode = common.KVCacheQuantMode.bfloat16

        result = comprehensive_perf_db.query_generation_mla(
            b, s, num_heads, kv_cache_quant_mode, database_mode=common.DatabaseMode.SILICON
        )

        # Should use data from generation_mla_data
        expected = comprehensive_perf_db._generation_mla_data[kv_cache_quant_mode][num_heads][b][s]
        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_query_generation_mla_sol_full_mode(self, comprehensive_perf_db):
        """Test SOL_FULL mode returns (sol_time, sol_math, sol_mem)."""
        sol_time, sol_math, sol_mem = comprehensive_perf_db.query_generation_mla(
            1, 32, 32, common.KVCacheQuantMode.bfloat16, database_mode=common.DatabaseMode.SOL_FULL
        )

        sol_only = comprehensive_perf_db.query_generation_mla(
            1, 32, 32, common.KVCacheQuantMode.bfloat16, database_mode=common.DatabaseMode.SOL
        )
        assert sol_time > 0
        assert math.isclose(sol_time, float(sol_only), rel_tol=1e-6)
        assert math.isclose(sol_time, max(sol_math, sol_mem), rel_tol=1e-6)


def test_default_database_mode(mutable_comprehensive_perf_db):
    """Test setting and getting default database mode, and that query cache is cleared when default mode is changed."""
    db = mutable_comprehensive_perf_db
    # Initially should be SILICON
    assert db.get_default_database_mode() == common.DatabaseMode.SILICON

    non_sol_result = db.query_context_attention(
        1, 32, 0, 8, 4, common.KVCacheQuantMode.bfloat16, common.FMHAQuantMode.bfloat16
    )
    assert db.query_context_attention.cache_info().currsize >= 1

    # Set to SOL mode
    db.set_default_database_mode(common.DatabaseMode.SOL)
    assert db.get_default_database_mode() == common.DatabaseMode.SOL
    # Cache should be cleared
    assert db.query_context_attention.cache_info().currsize == 0

    # Query should use default mode when not specified
    sol_result = db.query_context_attention(
        1, 32, 0, 8, 4, common.KVCacheQuantMode.bfloat16, common.FMHAQuantMode.bfloat16
    )

    cache_info = db.query_context_attention.cache_info()
    assert cache_info.misses == 1
    assert cache_info.hits == 0
    assert cache_info.currsize == 1
    assert isinstance(sol_result, float)
    assert sol_result != non_sol_result

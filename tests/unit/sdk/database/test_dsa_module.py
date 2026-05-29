# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import DEFAULT_DSA_ARCHITECTURE, LoadedOpData, PerfDataNotAvailableError

pytestmark = pytest.mark.unit


def _dsa_value(latency: float) -> dict[str, float]:
    return {"latency": latency, "power": 10.0, "energy": latency * 10.0}


def _context_dsa_data(dsa_dict: dict) -> dict:
    return {
        common.FMHAQuantMode.bfloat16: {
            common.KVCacheQuantMode.bfloat16: {
                common.GEMMQuantMode.bfloat16: {
                    DEFAULT_DSA_ARCHITECTURE: dsa_dict,
                },
            },
        },
    }


def _generation_dsa_data(dsa_dict: dict) -> dict:
    return {
        common.KVCacheQuantMode.bfloat16: {
            common.GEMMQuantMode.bfloat16: {
                DEFAULT_DSA_ARCHITECTURE: dsa_dict,
            },
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# Context DSA Module
# ═══════════════════════════════════════════════════════════════════════


class TestContextDSAModule:
    """Tests for query_context_dsa_module."""

    def test_missing_architecture_raises_perf_data_not_available(self, stub_perf_db):
        dsa_dict = {32: {256: {1: _dsa_value(10.0)}}}
        stub_perf_db._context_dsa_module_data = LoadedOpData(
            _context_dsa_data(dsa_dict), common.PerfDataFilename.dsa_context_module, "extrapolated"
        )

        with pytest.raises(PerfDataNotAvailableError, match="Context DSA module data not available"):
            stub_perf_db.query_context_dsa_module(
                b=1,
                s=256,
                prefix=0,
                num_heads=32,
                kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
                fmha_quant_mode=common.FMHAQuantMode.bfloat16,
                gemm_quant_mode=common.GEMMQuantMode.bfloat16,
                database_mode=common.DatabaseMode.SILICON,
                architecture="GlmMoeDsaForCausalLM",
            )

    def test_topk_piecewise_from_raw_handles_both_boundary_sides(self, stub_perf_db):
        raw_dsa_dict = {
            32: {
                1024: {1: _dsa_value(10.0)},
                2048: {1: _dsa_value(20.0)},
                3072: {1: _dsa_value(80.0)},
                4096: {1: _dsa_value(100.0)},
            }
        }

        below = stub_perf_db._interp_dsa_context_topk_piecewise_from_raw(32, 2047, 1, raw_dsa_dict, 2048)
        above = stub_perf_db._interp_dsa_context_topk_piecewise_from_raw(32, 2049, 1, raw_dsa_dict, 2048)

        assert below is not None
        assert above is not None
        assert below["latency"] == pytest.approx(10.0 + (20.0 - 10.0) / 1024.0 * (2047 - 1024))
        assert above["latency"] == pytest.approx(80.0 + (100.0 - 80.0) / 1024.0 * (2049 - 3072))
        assert above["latency"] > raw_dsa_dict[32][2048][1]["latency"]

    def test_topk_plus_one_uses_raw_piecewise_instead_of_cubic_fallback(self, stub_perf_db, monkeypatch):
        raw_dsa_dict = {
            32: {
                2048: {1: _dsa_value(20.0)},
                3072: {1: _dsa_value(80.0)},
                4096: {1: _dsa_value(100.0)},
            }
        }
        extrapolated_dsa_dict = {
            32: {
                2048: {1: _dsa_value(20.0)},
                2049: {1: _dsa_value(21.0)},
                3072: {1: _dsa_value(80.0)},
                4096: {1: _dsa_value(100.0)},
            }
        }
        stub_perf_db._raw_context_dsa_module_data = LoadedOpData(
            _context_dsa_data(raw_dsa_dict), common.PerfDataFilename.dsa_context_module, "raw"
        )
        stub_perf_db._context_dsa_module_data = LoadedOpData(
            _context_dsa_data(extrapolated_dsa_dict), common.PerfDataFilename.dsa_context_module, "extrapolated"
        )

        def fail_interp_3d(*args, **kwargs):
            raise AssertionError("_interp_3d should not be used for topk + 1 when raw right-regime anchors exist")

        monkeypatch.setattr(stub_perf_db, "_interp_3d", fail_interp_3d)

        result = stub_perf_db.query_context_dsa_module(
            b=1,
            s=2049,
            prefix=0,
            num_heads=32,
            index_topk=2048,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            database_mode=common.DatabaseMode.SILICON,
        )

        assert float(result) == pytest.approx(80.0 + (100.0 - 80.0) / 1024.0 * (2049 - 3072))
        assert result.energy == pytest.approx((80.0 + (100.0 - 80.0) / 1024.0 * (2049 - 3072)) * 10.0)

    def test_topk_piecewise_falls_back_when_raw_same_regime_anchors_are_unavailable(self, stub_perf_db, monkeypatch):
        raw_dsa_dict = {
            32: {
                2048: {1: _dsa_value(20.0)},
                4096: {1: _dsa_value(100.0)},
            }
        }
        stub_perf_db._raw_context_dsa_module_data = LoadedOpData(
            _context_dsa_data(raw_dsa_dict), common.PerfDataFilename.dsa_context_module, "raw"
        )
        stub_perf_db._context_dsa_module_data = LoadedOpData(
            _context_dsa_data(raw_dsa_dict), common.PerfDataFilename.dsa_context_module, "extrapolated"
        )
        cubic_calls = []

        def fake_interp_3d(*args, **kwargs):
            cubic_calls.append((args, kwargs))
            return {"latency": 123.0, "power": 0.0, "energy": 456.0}

        monkeypatch.setattr(stub_perf_db, "_interp_3d", fake_interp_3d)

        result = stub_perf_db.query_context_dsa_module(
            b=1,
            s=2049,
            prefix=0,
            num_heads=32,
            index_topk=2048,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            database_mode=common.DatabaseMode.SILICON,
        )

        assert float(result) == pytest.approx(123.0)
        assert result.energy == pytest.approx(456.0)
        assert len(cubic_calls) == 1

    def test_unsupported_silicon_candidate_logs_warning_without_traceback(self, stub_perf_db, caplog):
        dsa_dict = {64: {4000: {1: _dsa_value(10.0)}}}
        stub_perf_db._context_dsa_module_data = LoadedOpData(
            _context_dsa_data(dsa_dict), common.PerfDataFilename.dsa_context_module, "single-head"
        )

        caplog.set_level(logging.WARNING, logger="aiconfigurator.sdk.perf_database")
        with pytest.raises(PerfDataNotAvailableError, match="Context DSA module data not available"):
            stub_perf_db.query_context_dsa_module(
                b=1,
                s=4000,
                prefix=0,
                num_heads=32,
                index_topk=2048,
                kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
                fmha_quant_mode=common.FMHAQuantMode.bfloat16,
                gemm_quant_mode=common.GEMMQuantMode.bfloat16,
                database_mode=common.DatabaseMode.SILICON,
            )

        assert any("Context DSA module data not available" in record.getMessage() for record in caplog.records)
        assert all(record.exc_info is None for record in caplog.records)

    def test_sol_returns_positive(self, comprehensive_perf_db):
        result = comprehensive_perf_db.query_context_dsa_module(
            b=2,
            s=256,
            prefix=0,
            num_heads=32,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            database_mode=common.DatabaseMode.SOL,
        )
        assert float(result) > 0

    def test_sol_full_returns_three_tuple(self, comprehensive_perf_db):
        result = comprehensive_perf_db.query_context_dsa_module(
            b=2,
            s=256,
            prefix=0,
            num_heads=32,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
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
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            database_mode=common.DatabaseMode.SOL,
        )
        r2 = comprehensive_perf_db.query_context_dsa_module(
            b=4,
            s=1024,
            prefix=0,
            num_heads=32,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
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
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            database_mode=common.DatabaseMode.SOL,
        )
        with_prefix = comprehensive_perf_db.query_context_dsa_module(
            b=2,
            s=256,
            prefix=256,
            num_heads=32,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            database_mode=common.DatabaseMode.SOL,
        )
        assert with_prefix > no_prefix

    def test_empirical_returns_positive(self, comprehensive_perf_db):
        result = comprehensive_perf_db.query_context_dsa_module(
            b=2,
            s=256,
            prefix=0,
            num_heads=32,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
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
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
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
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            database_mode=common.DatabaseMode.SOL,
        )
        r2 = comprehensive_perf_db.query_context_dsa_module(
            b=4,
            s=4096,
            prefix=0,
            num_heads=32,
            index_topk=512,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            database_mode=common.DatabaseMode.SOL,
        )
        assert r1 != r2


# ═══════════════════════════════════════════════════════════════════════
# Generation DSA Module
# ═══════════════════════════════════════════════════════════════════════


class TestGenerationDSAModule:
    """Tests for query_generation_dsa_module."""

    def test_missing_architecture_raises_perf_data_not_available(self, stub_perf_db):
        dsa_dict = {32: {1: {256: _dsa_value(10.0)}}}
        stub_perf_db._generation_dsa_module_data = LoadedOpData(
            _generation_dsa_data(dsa_dict), common.PerfDataFilename.dsa_generation_module, "extrapolated"
        )

        with pytest.raises(PerfDataNotAvailableError, match="Generation DSA module data not available"):
            stub_perf_db.query_generation_dsa_module(
                b=1,
                s=256,
                num_heads=32,
                kv_cache_dtype=common.KVCacheQuantMode.bfloat16,
                gemm_quant_mode=common.GEMMQuantMode.bfloat16,
                database_mode=common.DatabaseMode.SILICON,
                architecture="GlmMoeDsaForCausalLM",
            )

    def test_unsupported_silicon_candidate_logs_warning_without_traceback(self, stub_perf_db, caplog):
        dsa_dict = {64: {1: {4000: _dsa_value(10.0)}}}
        stub_perf_db._generation_dsa_module_data = LoadedOpData(
            _generation_dsa_data(dsa_dict), common.PerfDataFilename.dsa_generation_module, "single-head"
        )

        caplog.set_level(logging.WARNING, logger="aiconfigurator.sdk.perf_database")
        with pytest.raises(PerfDataNotAvailableError, match="Generation DSA module data not available"):
            stub_perf_db.query_generation_dsa_module(
                b=1,
                s=4000,
                num_heads=32,
                index_topk=2048,
                kv_cache_dtype=common.KVCacheQuantMode.bfloat16,
                gemm_quant_mode=common.GEMMQuantMode.bfloat16,
                database_mode=common.DatabaseMode.SILICON,
            )

        assert any("Generation DSA module data not available" in record.getMessage() for record in caplog.records)
        assert all(record.exc_info is None for record in caplog.records)

    def test_past_kv_sequence_normalization_applies_to_interpolation(self, stub_perf_db):
        dsa_dict = {
            32: {
                1: {
                    99: _dsa_value(10.0),
                    199: _dsa_value(30.0),
                },
                2: {
                    149: _dsa_value(20.0),
                },
            }
        }
        stub_perf_db._generation_dsa_module_data = LoadedOpData(
            _generation_dsa_data(dsa_dict), common.PerfDataFilename.dsa_generation_module, "past-kv"
        )

        result = stub_perf_db.query_generation_dsa_module(
            b=1,
            s=150,
            num_heads=32,
            kv_cache_dtype=common.KVCacheQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            database_mode=common.DatabaseMode.SILICON,
        )

        assert float(result) == pytest.approx(20.0)

    def test_sol_returns_positive(self, comprehensive_perf_db):
        result = comprehensive_perf_db.query_generation_dsa_module(
            b=4,
            s=1024,
            num_heads=32,
            kv_cache_dtype=common.KVCacheQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            database_mode=common.DatabaseMode.SOL,
        )
        assert float(result) > 0

    def test_sol_full_returns_three_tuple(self, comprehensive_perf_db):
        result = comprehensive_perf_db.query_generation_dsa_module(
            b=4,
            s=1024,
            num_heads=32,
            kv_cache_dtype=common.KVCacheQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
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
            kv_cache_dtype=common.KVCacheQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            database_mode=common.DatabaseMode.SOL,
        )
        r2 = comprehensive_perf_db.query_generation_dsa_module(
            b=64,
            s=1024,
            num_heads=32,
            kv_cache_dtype=common.KVCacheQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            database_mode=common.DatabaseMode.SOL,
        )
        assert r2 > r1

    def test_different_index_topk_changes_sol(self, comprehensive_perf_db):
        r1 = comprehensive_perf_db.query_generation_dsa_module(
            b=4,
            s=4096,
            num_heads=32,
            index_topk=2048,
            kv_cache_dtype=common.KVCacheQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            database_mode=common.DatabaseMode.SOL,
        )
        r2 = comprehensive_perf_db.query_generation_dsa_module(
            b=4,
            s=4096,
            num_heads=32,
            index_topk=512,
            kv_cache_dtype=common.KVCacheQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            database_mode=common.DatabaseMode.SOL,
        )
        assert r2 < r1

    def test_empirical_returns_positive(self, comprehensive_perf_db):
        result = comprehensive_perf_db.query_generation_dsa_module(
            b=4,
            s=1024,
            num_heads=32,
            kv_cache_dtype=common.KVCacheQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            database_mode=common.DatabaseMode.EMPIRICAL,
        )
        assert float(result) > 0

    def test_hybrid_falls_back_when_no_data(self, comprehensive_perf_db):
        result = comprehensive_perf_db.query_generation_dsa_module(
            b=4,
            s=1024,
            num_heads=32,
            kv_cache_dtype=common.KVCacheQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            database_mode=common.DatabaseMode.HYBRID,
        )
        assert float(result) > 0

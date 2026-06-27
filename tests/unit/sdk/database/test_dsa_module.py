# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aiconfigurator.sdk import common, interpolation
from aiconfigurator.sdk.operations.dsa import (
    DEFAULT_DSA_ARCHITECTURE,
    load_context_dsa_module_data,
    load_generation_dsa_module_data,
)
from aiconfigurator.sdk.perf_database import LoadedOpData, PerfDataNotAvailableError

pytestmark = pytest.mark.unit

GLM5_ARCHITECTURE = "GlmMoeDsaForCausalLM"


def _dsa_value(latency: float) -> dict[str, float]:
    return {"latency": latency, "power": 10.0, "energy": latency * 10.0}


def _context_dsa_data(dsa_dict: dict, architecture: str = DEFAULT_DSA_ARCHITECTURE) -> dict:
    return {
        common.FMHAQuantMode.bfloat16: {
            common.KVCacheQuantMode.bfloat16: {
                common.GEMMQuantMode.bfloat16: {
                    architecture: dsa_dict,
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


def test_dsa_loaders_keep_primary_row_on_shared_source_conflict(tmp_path):
    header = "kernel_source,architecture,gemm_type,mla_dtype,kv_cache_dtype,num_heads,batch_size,isl,step,latency\n"
    row = "default,DeepseekV32ForCausalLM,bfloat16,bfloat16,bfloat16,32,1,256,0,{latency}\n"
    primary = tmp_path / "primary.txt"
    sibling = tmp_path / "sibling.txt"
    primary.write_text(header + row.format(latency=10.0))
    sibling.write_text(header + row.format(latency=99.0))
    sources = [(str(primary), None), (str(sibling), {"default"})]

    context = load_context_dsa_module_data(sources)
    context_leaf = context[common.FMHAQuantMode.bfloat16][common.KVCacheQuantMode.bfloat16][
        common.GEMMQuantMode.bfloat16
    ][DEFAULT_DSA_ARCHITECTURE]["flashmla_kv"][32][0][256][1]
    assert context_leaf["latency"] == pytest.approx(10.0)

    generation = load_generation_dsa_module_data(sources)
    generation_leaf = generation[common.KVCacheQuantMode.bfloat16][common.GEMMQuantMode.bfloat16][
        DEFAULT_DSA_ARCHITECTURE
    ]["flashmla_kv"][32][1][256]
    assert generation_leaf["latency"] == pytest.approx(10.0)


def test_dsa_loaders_keep_legacy_default_rows_separate_by_framework(tmp_path):
    data_path = tmp_path / "dsa_module_perf.txt"
    data_path.write_text(
        "framework,kernel_source,architecture,gemm_type,mla_dtype,kv_cache_dtype,"
        "num_heads,batch_size,isl,step,latency\n"
        "TRTLLM,default,DeepseekV32ForCausalLM,bfloat16,bfloat16,bfloat16,32,1,256,0,10.0\n"
        "VLLM,default,DeepseekV32ForCausalLM,bfloat16,bfloat16,bfloat16,32,1,256,0,20.0\n"
    )

    context = load_context_dsa_module_data(str(data_path))
    architecture_data = context[common.FMHAQuantMode.bfloat16][common.KVCacheQuantMode.bfloat16][
        common.GEMMQuantMode.bfloat16
    ][DEFAULT_DSA_ARCHITECTURE]
    assert architecture_data["trtllm"][32][0][256][1]["latency"] == pytest.approx(10.0)
    assert architecture_data["flashmla_kv"][32][0][256][1]["latency"] == pytest.approx(20.0)

    generation = load_generation_dsa_module_data(str(data_path))
    architecture_data = generation[common.KVCacheQuantMode.bfloat16][common.GEMMQuantMode.bfloat16][
        DEFAULT_DSA_ARCHITECTURE
    ]
    assert architecture_data["trtllm"][32][1][256]["latency"] == pytest.approx(10.0)
    assert architecture_data["flashmla_kv"][32][1][256]["latency"] == pytest.approx(20.0)


def test_dsa_queries_default_to_active_backend(stub_perf_db):
    stub_perf_db.backend = common.BackendName.vllm.value
    stub_perf_db._context_dsa_module_data = LoadedOpData(
        _context_dsa_data(
            {
                "trtllm": {32: {0: {256: {1: _dsa_value(10.0)}}}},
                "flashmla_kv": {32: {0: {256: {1: _dsa_value(20.0)}}}},
            }
        ),
        common.PerfDataFilename.dsa_context_module,
        "backends",
    )
    stub_perf_db._generation_dsa_module_data = LoadedOpData(
        _generation_dsa_data(
            {
                "trtllm": {32: {1: {256: _dsa_value(11.0)}}},
                "flashmla_kv": {32: {1: {256: _dsa_value(21.0)}}},
            }
        ),
        common.PerfDataFilename.dsa_generation_module,
        "backends",
    )

    context = stub_perf_db.query_context_dsa_module(
        b=1,
        s=256,
        prefix=0,
        num_heads=32,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        database_mode=common.DatabaseMode.SILICON,
    )
    generation = stub_perf_db.query_generation_dsa_module(
        b=1,
        s=256,
        num_heads=32,
        kv_cache_dtype=common.KVCacheQuantMode.bfloat16,
        database_mode=common.DatabaseMode.SILICON,
    )

    assert float(context) == pytest.approx(20.0)
    assert float(generation) == pytest.approx(21.0)


def test_dsa_queries_do_not_fallback_to_another_backend(stub_perf_db):
    stub_perf_db._context_dsa_module_data = LoadedOpData(
        _context_dsa_data({"flashmla_kv": {32: {0: {256: {1: _dsa_value(20.0)}}}}}),
        common.PerfDataFilename.dsa_context_module,
        "flash-only",
    )
    stub_perf_db._generation_dsa_module_data = LoadedOpData(
        _generation_dsa_data({"flashmla_kv": {32: {1: {256: _dsa_value(21.0)}}}}),
        common.PerfDataFilename.dsa_generation_module,
        "flash-only",
    )

    with pytest.raises(PerfDataNotAvailableError, match="Context DSA module data not available"):
        stub_perf_db.query_context_dsa_module(
            b=1,
            s=256,
            prefix=0,
            num_heads=32,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            dsa_backend="trtllm",
            database_mode=common.DatabaseMode.SILICON,
        )
    with pytest.raises(PerfDataNotAvailableError, match="Generation DSA module data not available"):
        stub_perf_db.query_generation_dsa_module(
            b=1,
            s=256,
            num_heads=32,
            kv_cache_dtype=common.KVCacheQuantMode.bfloat16,
            dsa_backend="trtllm",
            database_mode=common.DatabaseMode.SILICON,
        )
    hybrid = stub_perf_db.query_context_dsa_module(
        b=1,
        s=256,
        prefix=0,
        num_heads=32,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        dsa_backend="trtllm",
        database_mode=common.DatabaseMode.HYBRID,
    )
    assert hybrid.source == "empirical"


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

    def test_glm5_context_loader_requires_step_column(self, tmp_path):
        data_path = tmp_path / "dsa_context_module_perf.txt"
        data_path.write_text(
            "architecture,gemm_type,mla_dtype,kv_cache_dtype,num_heads,batch_size,isl,latency\n"
            f"{GLM5_ARCHITECTURE},bfloat16,bfloat16,bfloat16,32,1,256,10.0\n"
        )

        with pytest.raises(ValueError, match="requires a non-empty step column"):
            load_context_dsa_module_data(str(data_path))

    def test_glm5_context_loader_accepts_numeric_zero_step(self, tmp_path):
        data_path = tmp_path / "dsa_context_module_perf.parquet"
        table = pa.table(
            {
                "architecture": [GLM5_ARCHITECTURE],
                "gemm_type": ["bfloat16"],
                "mla_dtype": ["bfloat16"],
                "kv_cache_dtype": ["bfloat16"],
                "num_heads": [32],
                "batch_size": [1],
                "isl": [256],
                "step": [0],
                "latency": [10.0],
            }
        )
        pq.write_table(table, data_path)

        data = load_context_dsa_module_data(str(data_path))

        value = data[common.FMHAQuantMode.bfloat16][common.KVCacheQuantMode.bfloat16][common.GEMMQuantMode.bfloat16][
            GLM5_ARCHITECTURE
        ]["flashmla_kv"][32][0][256][1]
        assert value["latency"] == pytest.approx(10.0)

    def test_default_context_loader_treats_whitespace_step_as_missing(self, tmp_path):
        data_path = tmp_path / "dsa_context_module_perf.txt"
        data_path.write_text(
            "architecture,gemm_type,mla_dtype,kv_cache_dtype,num_heads,batch_size,isl,step,latency\n"
            f"{DEFAULT_DSA_ARCHITECTURE},bfloat16,bfloat16,bfloat16,32,1,256,  ,10.0\n"
        )

        data = load_context_dsa_module_data(str(data_path))

        value = data[common.FMHAQuantMode.bfloat16][common.KVCacheQuantMode.bfloat16][common.GEMMQuantMode.bfloat16][
            DEFAULT_DSA_ARCHITECTURE
        ]["flashmla_kv"][32][0][256][1]
        assert value["latency"] == pytest.approx(10.0)

    def test_glm5_context_rejects_legacy_shape_without_prefix_axis(self, stub_perf_db):
        legacy_dsa_dict = {32: {256: {1: _dsa_value(10.0)}}}
        stub_perf_db._context_dsa_module_data = LoadedOpData(
            _context_dsa_data(legacy_dsa_dict, GLM5_ARCHITECTURE),
            common.PerfDataFilename.dsa_context_module,
            "legacy-glm5",
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
                architecture=GLM5_ARCHITECTURE,
            )

    def test_glm5_context_accepts_prefix_axis(self, stub_perf_db):
        dsa_dict = {32: {0: {256: {1: _dsa_value(10.0)}}}}
        stub_perf_db._context_dsa_module_data = LoadedOpData(
            _context_dsa_data(dsa_dict, GLM5_ARCHITECTURE), common.PerfDataFilename.dsa_context_module, "glm5-prefix"
        )

        result = stub_perf_db.query_context_dsa_module(
            b=1,
            s=256,
            prefix=0,
            num_heads=32,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            database_mode=common.DatabaseMode.SILICON,
            architecture=GLM5_ARCHITECTURE,
        )

        assert float(result) == pytest.approx(10.0)

    def test_glm5_context_prefix_axis_sparse_batch_falls_back_to_smaller_batch(self, stub_perf_db):
        dsa_dict = {16: {0: {16384: {1: _dsa_value(6.2052)}}}}
        stub_perf_db._context_dsa_module_data = LoadedOpData(
            _context_dsa_data(dsa_dict, GLM5_ARCHITECTURE), common.PerfDataFilename.dsa_context_module, "glm5-prefix"
        )

        result = stub_perf_db.query_context_dsa_module(
            b=2,
            s=16384,
            prefix=0,
            num_heads=16,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            database_mode=common.DatabaseMode.SILICON,
            architecture=GLM5_ARCHITECTURE,
        )

        assert float(result) == pytest.approx(12.4104)
        assert result.energy == pytest.approx(124.104)

    def test_topk_piecewise_from_raw_handles_both_boundary_sides(self, stub_perf_db):
        raw_dsa_dict = {
            32: {
                1024: {1: _dsa_value(10.0)},
                2048: {1: _dsa_value(20.0)},
                3072: {1: _dsa_value(80.0)},
                4096: {1: _dsa_value(100.0)},
            }
        }

        below = interpolation.interp_dsa_context_topk_piecewise_from_raw(32, 2047, 1, raw_dsa_dict, 2048)
        above = interpolation.interp_dsa_context_topk_piecewise_from_raw(32, 2049, 1, raw_dsa_dict, 2048)

        assert below is not None
        assert above is not None
        assert below["latency"] == pytest.approx(10.0 + (20.0 - 10.0) / 1024.0 * (2047 - 1024))
        assert above["latency"] == pytest.approx(80.0 + (100.0 - 80.0) / 1024.0 * (2049 - 3072))
        assert above["latency"] > raw_dsa_dict[32][2048][1]["latency"]

    def test_topk_plus_one_uses_only_post_topk_sparse_samples(self, stub_perf_db):
        dsa_dict = {
            32: {
                2048: {1: _dsa_value(20.0)},
                3072: {1: _dsa_value(80.0)},
                4096: {1: _dsa_value(100.0)},
            }
        }
        stub_perf_db._context_dsa_module_data = LoadedOpData(
            _context_dsa_data(dsa_dict), common.PerfDataFilename.dsa_context_module, "sparse"
        )

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
        query_sol = stub_perf_db.query_context_dsa_module(
            b=1,
            s=2049,
            prefix=0,
            num_heads=32,
            index_topk=2048,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            database_mode=common.DatabaseMode.SOL,
        )
        anchor_sol = stub_perf_db.query_context_dsa_module(
            b=1,
            s=3072,
            prefix=0,
            num_heads=32,
            index_topk=2048,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            database_mode=common.DatabaseMode.SOL,
        )
        expected = 80.0 * float(query_sol) / float(anchor_sol)
        assert float(result) == pytest.approx(expected)
        assert result.energy == pytest.approx(expected * 10.0)

    def test_prefix_axis_interpolates_measured_prefix_slices(self, stub_perf_db):
        dsa_dict = {
            32: {
                0: {256: {1: _dsa_value(10.0)}},
                1024: {256: {1: _dsa_value(50.0)}},
            }
        }
        stub_perf_db._context_dsa_module_data = LoadedOpData(
            _context_dsa_data(dsa_dict), common.PerfDataFilename.dsa_context_module, "prefix"
        )
        stub_perf_db._raw_context_dsa_module_data = LoadedOpData(
            _context_dsa_data(dsa_dict), common.PerfDataFilename.dsa_context_module, "raw-prefix"
        )

        result = stub_perf_db.query_context_dsa_module(
            b=1,
            s=256,
            prefix=512,
            num_heads=32,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            database_mode=common.DatabaseMode.SILICON,
        )

        assert float(result) == pytest.approx(30.0)
        assert result.energy == pytest.approx(300.0)

    def test_prefix_axis_uses_exact_prefix_slice_when_available(self, stub_perf_db):
        dsa_dict = {
            32: {
                0: {256: {1: _dsa_value(10.0)}},
                512: {256: {1: _dsa_value(33.0)}},
                1024: {256: {1: _dsa_value(50.0)}},
            }
        }
        stub_perf_db._context_dsa_module_data = LoadedOpData(
            _context_dsa_data(dsa_dict), common.PerfDataFilename.dsa_context_module, "prefix"
        )

        result = stub_perf_db.query_context_dsa_module(
            b=1,
            s=256,
            prefix=512,
            num_heads=32,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            database_mode=common.DatabaseMode.SILICON,
        )

        assert float(result) == pytest.approx(33.0)

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

    def test_sparse_ragged_mesh_stays_inside_topk_regime(self, stub_perf_db):
        table = {
            16: {
                1: {1: _dsa_value(1.0), 5: _dsa_value(5.0), 6: _dsa_value(106.0), 10: _dsa_value(110.0)},
                3: {1: _dsa_value(3.0), 5: _dsa_value(15.0), 6: _dsa_value(118.0), 10: _dsa_value(130.0)},
            }
        }
        stub_perf_db._generation_dsa_module_data = LoadedOpData(
            _generation_dsa_data(table), common.PerfDataFilename.dsa_generation_module, "sparse"
        )

        pre = stub_perf_db.query_generation_dsa_module(
            2, 3, 16, common.KVCacheQuantMode.bfloat16, index_topk=5, database_mode=common.DatabaseMode.SILICON
        )
        post = stub_perf_db.query_generation_dsa_module(
            2, 8, 16, common.KVCacheQuantMode.bfloat16, index_topk=5, database_mode=common.DatabaseMode.SILICON
        )

        assert float(pre) == pytest.approx(6.0)
        assert float(post) == pytest.approx(116.0)
        assert post.energy == pytest.approx(1160.0)

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

    def test_generation_loader_indexes_total_decode_length(self, tmp_path):
        data_path = tmp_path / "dsa_generation_module_perf.parquet"
        table = pa.table(
            {
                "architecture": [DEFAULT_DSA_ARCHITECTURE],
                "gemm_type": ["bfloat16"],
                "mla_dtype": ["bfloat16"],
                "kv_cache_dtype": ["bfloat16"],
                "num_heads": [32],
                "batch_size": [1],
                "isl": [1],
                "tp_size": [1],
                "step": [149],
                "latency": [20.0],
                "power": [10.0],
            }
        )
        pq.write_table(table, data_path)

        data = load_generation_dsa_module_data(str(data_path))

        assert data[common.KVCacheQuantMode.bfloat16][common.GEMMQuantMode.bfloat16][DEFAULT_DSA_ARCHITECTURE][
            "flashmla_kv"
        ][32][1][150] == _dsa_value(20.0)

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

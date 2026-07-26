# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aiconfigurator.sdk import common
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


def _latency(result) -> float:
    return result.latency if hasattr(result, "latency") else (result[0] if isinstance(result, tuple) else result)


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


# The real loader nests an extra dsa_backend axis: ...[architecture][dsa_backend][num_heads]...
# (see load_context/generation_dsa_module_data). The empirical path must descend past it
# exactly like silicon's _select_dsa_backend, or the grid never resolves.
def _context_dsa_data_with_backend(
    dsa_dict: dict, architecture: str = DEFAULT_DSA_ARCHITECTURE, dsa_backend: str = "flashmla_kv"
) -> dict:
    return _context_dsa_data({dsa_backend: dsa_dict}, architecture)


def _generation_dsa_data_with_backend(dsa_dict: dict, dsa_backend: str = "flashmla_kv") -> dict:
    return _generation_dsa_data({dsa_backend: dsa_dict})


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

    def test_context_loader_keeps_first_source_on_coordinate_conflict(self, tmp_path):
        header = (
            "architecture,kernel_source,gemm_type,mla_dtype,kv_cache_dtype,"
            "num_heads,batch_size,isl,step,latency,power\n"
        )
        active = tmp_path / "active_context.txt"
        fallback = tmp_path / "fallback_context.txt"
        active.write_text(
            header
            + f"{DEFAULT_DSA_ARCHITECTURE},default,bfloat16,bfloat16,bfloat16,32,1,256,0,7.0,10.0\n"
            + f"{DEFAULT_DSA_ARCHITECTURE},default,bfloat16,bfloat16,bfloat16,32,1,256,0,10.0,10.0\n"
        )
        fallback.write_text(
            header
            + f"{DEFAULT_DSA_ARCHITECTURE},default,bfloat16,bfloat16,bfloat16,32,1,256,0,99.0,10.0\n"
            + f"{DEFAULT_DSA_ARCHITECTURE},default,bfloat16,bfloat16,bfloat16,32,2,512,0,20.0,10.0\n"
        )

        data = load_context_dsa_module_data([(str(active), None), (str(fallback), {"default"})])
        head_data = data[common.FMHAQuantMode.bfloat16][common.KVCacheQuantMode.bfloat16][
            common.GEMMQuantMode.bfloat16
        ][DEFAULT_DSA_ARCHITECTURE]["flashmla_kv"][32][0]

        assert head_data[256][1] == _dsa_value(10.0)
        assert head_data[512][2] == _dsa_value(20.0)

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

    def test_skip_indexer_routes_to_skip_table(self, stub_perf_db):
        # skip_indexer=True must read _context_dsa_module_skip_data, not the full
        # table. Distinct latencies (full 10.0 vs skip 4.0) make a wrong wiring
        # (a skip query reading the full table) fail instead of silently passing.
        full = {32: {0: {256: {1: _dsa_value(10.0)}}}}
        skip = {32: {0: {256: {1: _dsa_value(4.0)}}}}
        stub_perf_db._context_dsa_module_data = LoadedOpData(
            _context_dsa_data(full, GLM5_ARCHITECTURE), common.PerfDataFilename.dsa_context_module, "glm5-full"
        )
        stub_perf_db._context_dsa_module_skip_data = LoadedOpData(
            _context_dsa_data(skip, GLM5_ARCHITECTURE), common.PerfDataFilename.dsa_context_module, "glm5-skip"
        )

        def _q(skip_indexer):
            return stub_perf_db.query_context_dsa_module(
                b=1,
                s=256,
                prefix=0,
                num_heads=32,
                kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
                fmha_quant_mode=common.FMHAQuantMode.bfloat16,
                gemm_quant_mode=common.GEMMQuantMode.bfloat16,
                database_mode=common.DatabaseMode.SILICON,
                architecture=GLM5_ARCHITECTURE,
                skip_indexer=skip_indexer,
            )

        assert float(_q(False)) == pytest.approx(10.0)
        assert float(_q(True)) == pytest.approx(4.0)

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

    def test_topk_plus_one_plain_crossing_on_raw_table(self, stub_perf_db):
        """s = topk+1 interpolates PLAINLY between the collected 2048 and 3072
        rows. The old topk-piecewise (same-regime-only anchors) was removed:
        leave-one-out on every real config showed plain crossing ties or beats
        it, incl. just above the boundary (the knee itself is collected)."""
        raw_dsa_dict = {
            32: {
                2048: {1: _dsa_value(20.0)},
                3072: {1: _dsa_value(80.0)},
                4096: {1: _dsa_value(100.0)},
            }
        }
        stub_perf_db._raw_context_dsa_module_data = LoadedOpData(
            _context_dsa_data(raw_dsa_dict), common.PerfDataFilename.dsa_context_module, "raw"
        )
        stub_perf_db._context_dsa_module_data = stub_perf_db._raw_context_dsa_module_data

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

        expected = 20.0 + (80.0 - 20.0) / (3072 - 2048) * (2049 - 2048)
        assert float(result) == pytest.approx(expected)
        assert result.energy == pytest.approx(expected * 10.0)

    def test_topk_plus_one_plain_crossing_with_sparse_anchors(self, stub_perf_db):
        """With only 2048 and 4096 collected, s=2049 plainly brackets them —
        no cubic fallback, no piecewise; the engine resolves the raw grid."""
        raw_dsa_dict = {
            32: {
                2048: {1: _dsa_value(20.0)},
                4096: {1: _dsa_value(100.0)},
            }
        }
        stub_perf_db._raw_context_dsa_module_data = LoadedOpData(
            _context_dsa_data(raw_dsa_dict), common.PerfDataFilename.dsa_context_module, "raw"
        )
        stub_perf_db._context_dsa_module_data = stub_perf_db._raw_context_dsa_module_data

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

        expected = 20.0 + (100.0 - 20.0) / (4096 - 2048) * (2049 - 2048)
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

    def test_empirical_raises_without_data(self, comprehensive_perf_db):
        from aiconfigurator.sdk.errors import EmpiricalNotImplementedError

        with pytest.raises(EmpiricalNotImplementedError):
            comprehensive_perf_db.query_context_dsa_module(
                b=2,
                s=256,
                prefix=0,
                num_heads=32,
                kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
                fmha_quant_mode=common.FMHAQuantMode.bfloat16,
                gemm_quant_mode=common.GEMMQuantMode.bfloat16,
                database_mode=common.DatabaseMode.EMPIRICAL,
            )

    def test_hybrid_does_not_hide_malformed_context_schema(self, mutable_comprehensive_perf_db):
        db = mutable_comprehensive_perf_db
        db._context_dsa_module_data = LoadedOpData(
            {common.FMHAQuantMode.bfloat16: []},
            common.PerfDataFilename.dsa_context_module,
            "malformed",
        )

        with pytest.raises(TypeError, match="Malformed performance data"):
            db.query_context_dsa_module(
                b=2,
                s=256,
                prefix=0,
                num_heads=32,
                kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
                fmha_quant_mode=common.FMHAQuantMode.bfloat16,
                gemm_quant_mode=common.GEMMQuantMode.bfloat16,
                database_mode=common.DatabaseMode.HYBRID,
            )

    @pytest.mark.parametrize(
        ("raw_heads", "expected_path"),
        [
            pytest.param(32, "exact-head", id="exact-head"),
            pytest.param(64, "cross-head-fallback", id="cross-head-fallback"),
        ],
    )
    def test_empirical_uses_raw_head_grid(self, mutable_comprehensive_perf_db, raw_heads, expected_path):
        db = mutable_comprehensive_perf_db
        backend = "unit_ctx_exact_raw"

        def sol(batch, sequence, num_heads):
            return float(
                db.query_context_dsa_module(
                    b=batch,
                    s=sequence,
                    prefix=0,
                    num_heads=num_heads,
                    kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
                    fmha_quant_mode=common.FMHAQuantMode.bfloat16,
                    gemm_quant_mode=common.GEMMQuantMode.bfloat16,
                    database_mode=common.DatabaseMode.SOL,
                    dsa_backend=backend,
                )
            )

        # Exact-head and cross-head fallback both use raw coordinates
        # (num_heads, prefix, sequence, batch). Sequence 200 is the normalized-
        # log midpoint of 100 and 400, so generic k=2 IDW yields util 0.4.
        raw = {
            raw_heads: {
                0: {
                    100: {3: _dsa_value(sol(3, 100, raw_heads) / 0.2)},
                    400: {3: _dsa_value(sol(3, 400, raw_heads) / 0.6)},
                }
            }
        }
        if expected_path == "exact-head":
            # A different TP/head slice has an exact query point but must not
            # beat the requested head's two measured neighbors.
            raw[64] = {0: {200: {3: _dsa_value(999.0)}}}
        working = {32: {0: {200: {3: _dsa_value(777.0)}}}}
        db._raw_context_dsa_module_data = LoadedOpData(
            _context_dsa_data_with_backend(raw, dsa_backend=backend),
            common.PerfDataFilename.dsa_context_module,
            "raw",
        )
        db._context_dsa_module_data = LoadedOpData(
            _context_dsa_data_with_backend(working, dsa_backend=backend),
            common.PerfDataFilename.dsa_context_module,
            "working",
        )
        db.clear_runtime_caches()

        result = db.query_context_dsa_module(
            b=3,
            s=200,
            prefix=0,
            num_heads=32,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            database_mode=common.DatabaseMode.EMPIRICAL,
            dsa_backend=backend,
        )

        assert float(result) == pytest.approx(sol(3, 200, 32) / 0.4)
        assert float(result) != pytest.approx(777.0)
        assert float(result) != pytest.approx(999.0)
        assert result.source == "empirical"

    def test_empirical_explicit_prefix_axis_is_used_under_backend_data(self, mutable_comprehensive_perf_db):
        """The explicit-prefix shape nests ...[dsa_backend][num_heads][prefix][s][b]. After
        descending past dsa_backend, the prefix axis must be DETECTED and USED, so two queries
        that differ only in prefix must resolve to different measured slices (a collapse here
        would mean the prefix axis was folded away). Data gives distinct per-prefix latencies."""

        db = mutable_comprehensive_perf_db
        # num_heads -> prefix -> s -> b, with a large gap between the two prefix slices.
        nh = {32: {0: {256: {1: _dsa_value(5.0)}}, 256: {256: {1: _dsa_value(50.0)}}}}
        wrapper = LoadedOpData(
            _context_dsa_data_with_backend(nh, architecture=GLM5_ARCHITECTURE),
            common.PerfDataFilename.dsa_context_module,
            "measured",
        )
        db._raw_context_dsa_module_data = wrapper
        db._context_dsa_module_data = wrapper

        def query(prefix):
            return _latency(
                db.query_context_dsa_module(
                    b=1,
                    s=256,
                    prefix=prefix,
                    num_heads=32,
                    kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
                    fmha_quant_mode=common.FMHAQuantMode.bfloat16,
                    gemm_quant_mode=common.GEMMQuantMode.bfloat16,
                    database_mode=common.DatabaseMode.EMPIRICAL,
                    architecture=GLM5_ARCHITECTURE,
                )
            )

        lo, hi = query(0), query(256)
        assert math.isfinite(lo) and math.isfinite(hi)
        assert hi > lo  # prefix axis genuinely used; not collapsed to one slice

    def test_empirical_prefix_cache_is_faster_at_fixed_isl_on_prefix0_only_data(self, mutable_comprehensive_perf_db):
        """When the prefix axis is degenerate (only prefix=0 collected), prefix>0 must anchor
        util at the prefix=0 slice at full_s=s+prefix (regime-matched) and carry the prefix
        effect via SOL -- NOT borrow util at the query's own small-s point (which sits below
        the indexer-on boundary and on the launch-overhead floor, inflating the estimate). The
        physical check: at a fixed total length (ISL), reusing a cached prefix (fewer new tokens
        to compute) must be FASTER than computing all tokens fresh."""

        db = mutable_comprehensive_perf_db
        # 4D shape but ONLY prefix=0 collected (mirrors DeepseekV32 on trtllm). Latencies are
        # launch-overhead-floored at small s (so the old same-s util borrow would misfire).
        s_grid = {1024: 2.4, 2048: 2.8, 3072: 3.4, 4096: 4.0, 6144: 5.6, 8192: 7.5}
        nh = {128: {0: {s: {1: _dsa_value(lat)} for s, lat in s_grid.items()}}}
        wrapper = LoadedOpData(
            _context_dsa_data_with_backend(nh, architecture="DeepseekV32ForCausalLM"),
            common.PerfDataFilename.dsa_context_module,
            "measured",
        )
        db._raw_context_dsa_module_data = wrapper
        db._context_dsa_module_data = wrapper

        def query(s, prefix):
            return _latency(
                db.query_context_dsa_module(
                    b=1,
                    s=s,
                    prefix=prefix,
                    num_heads=128,
                    kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
                    fmha_quant_mode=common.FMHAQuantMode.bfloat16,
                    gemm_quant_mode=common.GEMMQuantMode.bfloat16,
                    database_mode=common.DatabaseMode.EMPIRICAL,
                    architecture="DeepseekV32ForCausalLM",
                )
            )

        all_fresh = query(s=4096, prefix=0)  # ISL=4096, nothing cached
        cached = query(s=2048, prefix=2048)  # ISL=4096, half cached -> fewer new tokens
        assert math.isfinite(all_fresh) and math.isfinite(cached)
        assert cached < all_fresh  # prefix-cache win; full_s anchor avoids the inflation artifact

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

    def test_generation_loader_keeps_first_source_on_total_sequence_conflict(self, tmp_path):
        header = (
            "architecture,kernel_source,gemm_type,mla_dtype,kv_cache_dtype,"
            "num_heads,batch_size,isl,step,latency,power\n"
        )
        active = tmp_path / "active_generation.txt"
        fallback = tmp_path / "fallback_generation.txt"
        active.write_text(
            header
            + f"{DEFAULT_DSA_ARCHITECTURE},default,bfloat16,bfloat16,bfloat16,32,1,1,149,7.0,10.0\n"
            + f"{DEFAULT_DSA_ARCHITECTURE},default,bfloat16,bfloat16,bfloat16,32,1,2,148,10.0,10.0\n"
        )
        fallback.write_text(
            header
            # Different isl/step decomposition, same indexed total sequence.
            + f"{DEFAULT_DSA_ARCHITECTURE},default,bfloat16,bfloat16,bfloat16,32,1,2,148,99.0,10.0\n"
            + f"{DEFAULT_DSA_ARCHITECTURE},default,bfloat16,bfloat16,bfloat16,32,2,1,150,20.0,10.0\n"
        )

        data = load_generation_dsa_module_data([(str(active), None), (str(fallback), {"default"})])
        head_data = data[common.KVCacheQuantMode.bfloat16][common.GEMMQuantMode.bfloat16][DEFAULT_DSA_ARCHITECTURE][
            "flashmla_kv"
        ][32]

        assert head_data[1][150] == _dsa_value(10.0)
        assert head_data[2][151] == _dsa_value(20.0)

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

    def test_empirical_raises_without_data(self, comprehensive_perf_db):
        from aiconfigurator.sdk.errors import EmpiricalNotImplementedError

        with pytest.raises(EmpiricalNotImplementedError):
            comprehensive_perf_db.query_generation_dsa_module(
                b=4,
                s=1024,
                num_heads=32,
                kv_cache_dtype=common.KVCacheQuantMode.bfloat16,
                gemm_quant_mode=common.GEMMQuantMode.bfloat16,
                database_mode=common.DatabaseMode.EMPIRICAL,
            )

    def test_hybrid_does_not_hide_malformed_generation_schema(self, mutable_comprehensive_perf_db):
        db = mutable_comprehensive_perf_db
        db._generation_dsa_module_data = LoadedOpData(
            {common.KVCacheQuantMode.bfloat16: []},
            common.PerfDataFilename.dsa_generation_module,
            "malformed",
        )

        with pytest.raises(TypeError, match="Malformed performance data"):
            db.query_generation_dsa_module(
                b=4,
                s=1024,
                num_heads=32,
                kv_cache_dtype=common.KVCacheQuantMode.bfloat16,
                gemm_quant_mode=common.GEMMQuantMode.bfloat16,
                database_mode=common.DatabaseMode.HYBRID,
            )

    def test_silicon_sequence_overflow_uses_raw_boundary_util(self, mutable_comprehensive_perf_db):
        db = mutable_comprehensive_perf_db
        # No pre-expansion in v2: the working table IS the raw measurements
        # (bogus synthesized points can no longer exist to supersede them).
        raw = {32: {4: {65: _dsa_value(10.0), 129: _dsa_value(11.0)}}}
        wrapper = LoadedOpData(
            _generation_dsa_data_with_backend(raw), common.PerfDataFilename.dsa_generation_module, "raw"
        )
        db._raw_generation_dsa_module_data = wrapper
        db._generation_dsa_module_data = wrapper
        db.clear_runtime_caches()

        query = {
            "b": 4,
            "num_heads": 32,
            "kv_cache_dtype": common.KVCacheQuantMode.bfloat16,
            "gemm_quant_mode": common.GEMMQuantMode.bfloat16,
        }
        sol_boundary = float(db.query_generation_dsa_module(s=129, database_mode=common.DatabaseMode.SOL, **query))
        sol_query = float(db.query_generation_dsa_module(s=11000, database_mode=common.DatabaseMode.SOL, **query))
        result = db.query_generation_dsa_module(s=11000, database_mode=common.DatabaseMode.SILICON, **query)
        expected = 11.0 * sol_query / sol_boundary

        assert float(result) == pytest.approx(expected)
        assert result.energy == pytest.approx(expected * 10.0)
        assert result.power == pytest.approx(10.0)
        assert result.source == "silicon"

    def test_silicon_in_grid_sequence_keeps_latency_interpolation(self, mutable_comprehensive_perf_db):
        db = mutable_comprehensive_perf_db
        measured = {32: {4: {64: _dsa_value(10.0), 128: _dsa_value(14.0)}}}
        wrapper = LoadedOpData(
            _generation_dsa_data_with_backend(measured), common.PerfDataFilename.dsa_generation_module, "measured"
        )
        db._raw_generation_dsa_module_data = wrapper
        db._generation_dsa_module_data = wrapper
        db.clear_runtime_caches()

        result = db.query_generation_dsa_module(
            b=4,
            s=96,
            num_heads=32,
            kv_cache_dtype=common.KVCacheQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            database_mode=common.DatabaseMode.SILICON,
        )

        assert float(result) == pytest.approx(12.0)
        assert result.energy == pytest.approx(120.0)
        assert result.power == pytest.approx(10.0)

    def test_silicon_sequence_overflow_snaps_batch_and_holds_boundary_util(self, mutable_comprehensive_perf_db):
        """Query (b=3, s=1024) with both axes off-grid and s beyond every
        collected sweep: the engine snaps outer axes to the nearest collected
        path (batch 3 -> 2), holds that curve's boundary util (s=128), and
        lets SOL(query) carry the growth."""
        db = mutable_comprehensive_perf_db
        raw = {
            32: {
                2: {64: _dsa_value(9.0), 128: _dsa_value(10.0)},
                4: {64: _dsa_value(17.0), 128: _dsa_value(18.0)},
            }
        }
        wrapper = LoadedOpData(
            _generation_dsa_data_with_backend(raw), common.PerfDataFilename.dsa_generation_module, "raw"
        )
        db._raw_generation_dsa_module_data = wrapper
        db._generation_dsa_module_data = wrapper
        db.clear_runtime_caches()

        def sol(batch: int, sequence: int) -> float:
            return float(
                db.query_generation_dsa_module(
                    b=batch,
                    s=sequence,
                    num_heads=32,
                    kv_cache_dtype=common.KVCacheQuantMode.bfloat16,
                    gemm_quant_mode=common.GEMMQuantMode.bfloat16,
                    database_mode=common.DatabaseMode.SOL,
                )
            )

        result = db.query_generation_dsa_module(
            b=3,
            s=1024,
            num_heads=32,
            kv_cache_dtype=common.KVCacheQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            database_mode=common.DatabaseMode.SILICON,
        )

        expected = 10.0 * sol(3, 1024) / sol(2, 128)  # anchor: batch snapped to 2, boundary s=128
        assert float(result) == pytest.approx(expected)
        assert result.power == pytest.approx(10.0)
        assert result.energy == pytest.approx(expected * 10.0)

    def test_empirical_prefers_exact_head_slice_over_longer_other_tp(self, mutable_comprehensive_perf_db):
        db = mutable_comprehensive_perf_db
        measured = {
            16: {4: {65: _dsa_value(10.0), 129: _dsa_value(11.0)}},
            # This exact-sequence sample would win the old global 3-D nearest
            # neighbour even though heads=64 represents a different TP shape.
            64: {4: {11000: _dsa_value(999.0)}},
        }
        wrapper = LoadedOpData(
            _generation_dsa_data_with_backend(measured), common.PerfDataFilename.dsa_generation_module, "measured"
        )
        db._raw_generation_dsa_module_data = wrapper
        db._generation_dsa_module_data = wrapper
        db.clear_runtime_caches()

        query = {
            "b": 4,
            "num_heads": 16,
            "kv_cache_dtype": common.KVCacheQuantMode.bfloat16,
            "gemm_quant_mode": common.GEMMQuantMode.bfloat16,
            "dsa_backend": "unit_exact_head",
        }
        sol_boundary = float(db.query_generation_dsa_module(s=129, database_mode=common.DatabaseMode.SOL, **query))
        sol_query = float(db.query_generation_dsa_module(s=11000, database_mode=common.DatabaseMode.SOL, **query))
        result = db.query_generation_dsa_module(s=11000, database_mode=common.DatabaseMode.EMPIRICAL, **query)

        assert float(result) == pytest.approx(11.0 * sol_query / sol_boundary)
        assert result.source == "empirical"


class TestLoadGlm5SparseFamilyAnchor:
    """_load_glm5_sparse resolves the sparse version DIR via a representative
    filename. The three sparse tables (mqa / topk / dsa_attn) are collected as
    independent ops, so the anchor must be whichever sibling exists first — a
    family dir holding only topk/dsa_attn (no mqa) must still be found instead
    of silently falling back to the legacy dir."""

    @staticmethod
    def _write_sparse_parquet(path, latency: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "batch_size": pa.array([1], pa.int64()),
                "isl": pa.array([4096], pa.int64()),
                "step": pa.array([0], pa.int64()),
                "latency": pa.array([latency], pa.float64()),
                "num_heads": pa.array([64], pa.int64()),
            }
        )
        pq.write_table(table, path)

    @staticmethod
    def _fake_database(tmp_path):
        from types import SimpleNamespace

        return SimpleNamespace(
            systems_root=str(tmp_path),
            system="h200_sxm",
            backend="sglang",
            version="0.5.14",
            enable_shared_layer=False,
            system_spec={"data_dir": "data/h200_sxm"},
        )

    def test_family_dir_without_mqa_still_resolves_siblings(self, tmp_path):
        from aiconfigurator.sdk.operations.dsa import ContextDSAModule

        vdir = tmp_path / "data" / "h200_sxm" / "sparse_attention" / "sglang" / "0.5.14"
        self._write_sparse_parquet(vdir / "glm5_topk_module_perf.parquet", 12.5)

        ContextDSAModule._glm5_sparse_cache.clear()
        try:
            tables = ContextDSAModule._load_glm5_sparse(self._fake_database(tmp_path), GLM5_ARCHITECTURE, 64)
        finally:
            ContextDSAModule._glm5_sparse_cache.clear()

        assert tables["_2d"]["topk_last"] == {1: {(4096, 0): 12.5}}

    def test_mqa_anchor_still_preferred_when_present(self, tmp_path):
        from aiconfigurator.sdk.operations.dsa import ContextDSAModule

        vdir = tmp_path / "data" / "h200_sxm" / "sparse_attention" / "sglang" / "0.5.14"
        self._write_sparse_parquet(vdir / "glm5_mqa_logits_module_perf.parquet", 3.0)
        self._write_sparse_parquet(vdir / "glm5_topk_module_perf.parquet", 12.5)

        ContextDSAModule._glm5_sparse_cache.clear()
        try:
            tables = ContextDSAModule._load_glm5_sparse(self._fake_database(tmp_path), GLM5_ARCHITECTURE, 64)
        finally:
            ContextDSAModule._glm5_sparse_cache.clear()

        assert tables["_2d"]["mqa"] == {1: {(4096, 0): 3.0}}
        assert tables["_2d"]["topk_last"] == {1: {(4096, 0): 12.5}}

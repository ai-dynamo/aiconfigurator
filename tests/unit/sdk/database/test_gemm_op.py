# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct tests for ``GEMM`` ownership of CSV-backed perf data.

These tests cover the Stage-2 migration of ISSUE-05: GEMM now owns its
three CSV tables (gemm / compute_scale / scale_matrix), SOL correction,
and grid extrapolation. ``PerfDatabase.query_gemm`` etc. are one-line
delegations to ``GEMM._query_*_table``.
"""

from __future__ import annotations

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.errors import MissingSystemFlopsError, PerfDataNotAvailableError
from aiconfigurator.sdk.operations.gemm import GEMM

pytestmark = pytest.mark.unit


class TestGEMMCacheStructure:
    """The three caches must exist as class-level dicts."""

    def test_class_level_caches_exist(self):
        assert isinstance(GEMM._data_cache, dict)
        assert isinstance(GEMM._compute_scale_cache, dict)
        assert isinstance(GEMM._scale_matrix_cache, dict)

    def test_cache_key_includes_systems_root_and_shared_layer(self, stub_perf_db):
        """Cache key components must include systems_root + enable_shared_layer
        so test fixtures with separate tmp_paths and HYBRID/SILICON loads
        get distinct entries."""
        key = GEMM._cache_key(stub_perf_db)
        # Order: (systems_root, system, backend, version, enable_shared_layer)
        assert len(key) == 5
        assert key[0] == stub_perf_db.systems_root
        assert key[1] == stub_perf_db.system
        assert key[2] == stub_perf_db.backend
        assert key[3] == stub_perf_db.version
        assert key[4] == stub_perf_db.enable_shared_layer

    def test_clear_cache_clears_compute_scale_delta_lookup(self, monkeypatch):
        # Patch every class cache to keep this test isolated from the warmed
        # singleton caches used by the rest of the database suite.
        monkeypatch.setattr(GEMM, "_data_cache", {"sentinel": object()})
        monkeypatch.setattr(GEMM, "_compute_scale_cache", {"sentinel": object()})
        monkeypatch.setattr(GEMM, "_scale_matrix_cache", {"sentinel": object()})
        monkeypatch.setattr(GEMM, "_compute_scale_delta_lookup_cache", {1: object()})

        GEMM.clear_cache()

        assert GEMM._data_cache == {}
        assert GEMM._compute_scale_cache == {}
        assert GEMM._scale_matrix_cache == {}
        assert GEMM._compute_scale_delta_lookup_cache == {}


class TestStaticHelpers:
    """``common.get_quant_tc_flops`` + ``_normalize_gemm_quant_mode_for_table``."""

    def test_normalize_fp8_static_maps_to_fp8(self):
        result = GEMM._normalize_gemm_quant_mode_for_table(common.GEMMQuantMode.fp8_static)
        assert result == common.GEMMQuantMode.fp8

    def test_normalize_passes_through_other_modes(self):
        for qm in [
            common.GEMMQuantMode.bfloat16,
            common.GEMMQuantMode.fp8,
            common.GEMMQuantMode.nvfp4,
            common.GEMMQuantMode.w4a16_nvfp4,
        ]:
            assert GEMM._normalize_gemm_quant_mode_for_table(qm) == qm

    def test_empirical_grid_key_keeps_w4a16_nvfp4_table_identity(self):
        database = type("Database", (), {"system": "b200", "backend": "vllm", "version": "1.0"})()

        int4_key = GEMM._empirical_grid_key(database, common.GEMMQuantMode.int4_wo)
        nvfp4_key = GEMM._empirical_grid_key(database, common.GEMMQuantMode.w4a16_nvfp4)

        assert int4_key[-2] == "int4_wo"
        assert nvfp4_key[-2] == "w4a16_nvfp4"
        assert int4_key[-1] == "int4_wo"
        assert nvfp4_key[-1] == "w4a16_nvfp4"
        assert int4_key != nvfp4_key

    def test_get_quant_tc_flops_uses_specific_key_when_present(self):
        system_spec = {"gpu": {"bfloat16_tc_flops": 1000.0, "fp8_tc_flops": 2000.0, "fp4_tc_flops": 4000.0}}
        assert common.get_quant_tc_flops(system_spec, common.GEMMQuantMode.bfloat16) == 1000.0
        assert common.get_quant_tc_flops(system_spec, common.GEMMQuantMode.fp8) == 2000.0

    def test_get_quant_tc_flops_resolves_by_compute_dtype(self):
        """The mapping is per compute dtype, not per speedup factor: `sq`
        (int8 pipeline) reads int8_tc_flops even though its compute factor
        matches fp8's, and weight-only modes read bfloat16_tc_flops."""
        system_spec = {"gpu": {"bfloat16_tc_flops": 1000.0, "int8_tc_flops": 30.0, "fp8_tc_flops": 2000.0}}
        assert common.get_quant_tc_flops(system_spec, common.GEMMQuantMode.sq) == 30.0
        assert common.get_quant_tc_flops(system_spec, common.GEMMQuantMode.int8_wo) == 1000.0
        assert common.get_quant_tc_flops(system_spec, common.GEMMQuantMode.int4_wo) == 1000.0

    def test_get_quant_tc_flops_missing_key_raises(self):
        """No bf16-scaled extrapolation: a missing ``*_tc_flops`` entry means
        the platform lacks that dtype (or the YAML is incomplete) and must
        raise instead of fabricating a throughput."""
        system_spec = {"gpu": {"bfloat16_tc_flops": 1000.0}}
        with pytest.raises(MissingSystemFlopsError, match="fp8_tc_flops"):
            common.get_quant_tc_flops(system_spec, common.GEMMQuantMode.fp8)
        with pytest.raises(MissingSystemFlopsError, match="fp4_tc_flops"):
            common.get_quant_tc_flops(system_spec, common.GEMMQuantMode.nvfp4)

    def test_get_quant_tc_flops_non_positive_entry_raises(self):
        """A zero/negative entry is a placeholder or a typo: letting it
        through would turn SOL into inf and load-time clamps into silent
        data corruption, so it is rejected like a missing entry."""
        system_spec = {"gpu": {"bfloat16_tc_flops": 1000.0, "fp8_tc_flops": 0}}
        with pytest.raises(MissingSystemFlopsError, match="fp8_tc_flops"):
            common.get_quant_tc_flops(system_spec, common.GEMMQuantMode.fp8)
        system_spec = {"gpu": {"bfloat16_tc_flops": 1000.0, "fp8_tc_flops": float("nan")}}
        with pytest.raises(MissingSystemFlopsError, match="fp8_tc_flops"):
            common.get_quant_tc_flops(system_spec, common.GEMMQuantMode.fp8)
        # +inf (PyYAML parses `.inf`) would zero sol_math and silently collapse
        # compute-bound SOL onto the memory roof.
        system_spec = {"gpu": {"bfloat16_tc_flops": 1000.0, "fp8_tc_flops": float("inf")}}
        with pytest.raises(MissingSystemFlopsError, match="fp8_tc_flops"):
            common.get_quant_tc_flops(system_spec, common.GEMMQuantMode.fp8)

    def test_get_quant_tc_flops_memory_only_mode_raises(self):
        system_spec = {"gpu": {"bfloat16_tc_flops": 1000.0}}
        with pytest.raises(MissingSystemFlopsError, match="memory-only"):
            common.get_quant_tc_flops(system_spec, common.KVCacheQuantMode.fp8)

    def test_get_quant_tc_flops_b300_fp4_uses_real_entry(self):
        """b300 breaks the fixed 4x ratio (fp4 = 14 PFLOPS, bf16*4 = 9 PFLOPS):
        the YAML entry must win over any compute-factor scaling (issue #1398)."""
        system_spec = {"gpu": {"bfloat16_tc_flops": 2.25e15, "fp8_tc_flops": 4.5e15, "fp4_tc_flops": 1.4e16}}
        assert common.get_quant_tc_flops(system_spec, common.MoEQuantMode.nvfp4) == 1.4e16
        assert common.get_quant_tc_flops(system_spec, common.GEMMQuantMode.nvfp4) == 1.4e16


class TestLoadData:
    """``GEMM.load_data`` is idempotent and binds instance attrs."""

    def test_load_data_binds_instance_attrs(self, stub_perf_db):
        # stub_perf_db's __init__ already triggered GEMM.load_data eagerly,
        # so the instance attrs are bound from the start.
        assert hasattr(stub_perf_db, "_gemm_data")
        assert hasattr(stub_perf_db, "_compute_scale_data")
        assert hasattr(stub_perf_db, "_scale_matrix_data")

    def test_load_data_is_idempotent(self, stub_perf_db):
        """Calling ``GEMM.load_data`` repeatedly must not increment the
        load counter. Does NOT clear ``GEMM._data_cache`` — that would
        invalidate the comprehensive_perf_db singleton used by sibling
        tests and force a real-disk re-load with no loader patches active."""
        from aiconfigurator.sdk.operations.base import Operation

        initial_count = Operation._load_data_call_count.get(GEMM, 0)
        for _ in range(5):
            GEMM.load_data(stub_perf_db)
        assert Operation._load_data_call_count.get(GEMM, 0) == initial_count, (
            "repeated load_data calls must not re-load"
        )

    def test_load_data_respects_test_overrides(self, mutable_comprehensive_perf_db):
        """If a test overwrites ``_gemm_data`` after construction, a later
        ``load_data`` call must not clobber it."""
        db = mutable_comprehensive_perf_db
        sentinel = object()
        db._gemm_data = sentinel

        GEMM.load_data(db)
        assert db._gemm_data is sentinel, "load_data must not override test-set _gemm_data"


class TestQueryDelegation:
    """``PerfDatabase.query_gemm`` etc. delegate to GEMM classmethods."""

    def test_query_gemm_via_database_matches_direct_classmethod(self, comprehensive_perf_db):
        db = comprehensive_perf_db
        m, n, k = 4, 256, 256
        quant_mode = common.GEMMQuantMode.bfloat16

        via_db = db.query_gemm(m, n, k, quant_mode)
        direct = GEMM._query_gemm_table(db, m, n, k, quant_mode)

        assert float(via_db) == float(direct)
        assert via_db.energy == direct.energy

    def test_query_gemm_sol_mode_does_not_require_data(self, stub_perf_db):
        """SOL mode is pure formula — must work even with no real data."""
        result = stub_perf_db.query_gemm(
            128, 256, 256, common.GEMMQuantMode.bfloat16, database_mode=common.DatabaseMode.SOL
        )
        assert float(result) > 0

    def test_w4a16_nvfp4_uses_transfer_policy_instead_of_table_alias(self, comprehensive_perf_db):
        db = comprehensive_perf_db
        with pytest.raises(PerfDataNotAvailableError):
            db.query_gemm(
                4,
                256,
                256,
                common.GEMMQuantMode.w4a16_nvfp4,
                database_mode=common.DatabaseMode.SILICON,
            )

        result = db.query_gemm(
            4,
            256,
            256,
            common.GEMMQuantMode.w4a16_nvfp4,
            database_mode=common.DatabaseMode.HYBRID,
        )
        assert result.source == "empirical"
        assert float(result) > 0

    def test_query_compute_scale_sol_mode(self, stub_perf_db):
        result = stub_perf_db.query_compute_scale(
            128, 256, common.GEMMQuantMode.fp8, database_mode=common.DatabaseMode.SOL
        )
        assert float(result) > 0

    def test_query_scale_matrix_sol_mode(self, stub_perf_db):
        result = stub_perf_db.query_scale_matrix(
            128, 256, common.GEMMQuantMode.fp8, database_mode=common.DatabaseMode.SOL
        )
        assert float(result) > 0


class TestSolCorrection:
    """``GEMM._correct_sol`` clamps mutated GEMM data back to >= SOL."""

    def test_correct_sol_clamps_low_gemm_latency(self, mutable_comprehensive_perf_db):
        db = mutable_comprehensive_perf_db
        quant_mode = common.GEMMQuantMode.bfloat16
        m, n, k = 64, 128, 256

        sol_value = float(db.query_gemm(m, n, k, quant_mode, database_mode=common.DatabaseMode.SOL))

        # Set an artificially low value (lower than SOL)
        db._gemm_data[quant_mode][m][n][k] = {"latency": sol_value * 0.5, "power": 0.0, "energy": 0.0}

        GEMM._correct_sol(db)

        clamped = db._gemm_data[quant_mode][m][n][k]
        clamped_latency = clamped["latency"] if isinstance(clamped, dict) else clamped
        assert clamped_latency >= sol_value


@pytest.mark.parametrize("quant_mode", [common.GEMMQuantMode.bfloat16, common.GEMMQuantMode.fp8])
def test_query_returns_silicon_source_for_loaded_table(comprehensive_perf_db, quant_mode):
    """Per-op silicon/empirical attribution (PR #956) must survive the
    GEMM migration."""
    result = comprehensive_perf_db.query_gemm(4, 256, 256, quant_mode)
    assert getattr(result, "source", None) == "silicon"


def test_below_grid_shape_degrades_to_sol_only_with_flag(comprehensive_perf_db):
    """`below_grid_sol` opt-in: a SILICON shape miss degrades to SOL; without
    the flag it raises, and a quant-mode miss stays strict either way."""
    db = comprehensive_perf_db
    quant = common.GEMMQuantMode.bfloat16

    # n=1 is octaves below the fixture grid (n >= 128).
    with pytest.raises(PerfDataNotAvailableError):
        db.query_gemm(4, 1, 256, quant)

    result = db.query_gemm(4, 1, 256, quant, below_grid_sol=True)
    assert result.source == "sol"
    assert float(result) == float(db.query_gemm(4, 1, 256, quant, database_mode=common.DatabaseMode.SOL))

    # int8_wo has system flops but no collected table: strict despite the flag.
    with pytest.raises(PerfDataNotAvailableError):
        db.query_gemm(4, 1, 256, common.GEMMQuantMode.int8_wo, below_grid_sol=True)


def test_below_grid_flag_threads_from_op_and_leaves_hybrid_unchanged(comprehensive_perf_db):
    db = comprehensive_perf_db
    quant = common.GEMMQuantMode.bfloat16

    # Op-level: the constructor kwarg must reach the table query.
    assert GEMM("gate", 1, 1, 256, quant, below_grid_sol=True).query(db, x=4).source == "sol"
    with pytest.raises(PerfDataNotAvailableError):
        GEMM("gate", 1, 1, 256, quant).query(db, x=4)

    # HYBRID keeps its empirical fallback regardless of the flag.
    with_flag = db.query_gemm(4, 1, 256, quant, database_mode=common.DatabaseMode.HYBRID, below_grid_sol=True)
    without = db.query_gemm(4, 1, 256, quant, database_mode=common.DatabaseMode.HYBRID)
    assert with_flag.source == without.source == "empirical"
    assert float(with_flag) == float(without)

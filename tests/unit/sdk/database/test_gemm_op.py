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
from aiconfigurator.sdk.errors import MissingSystemFlopsError
from aiconfigurator.sdk.operations.gemm import GEMM


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


class TestStaticHelpers:
    """``common.get_quant_tc_flops`` per-dtype resolution."""

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


# ---------------------------------------------------------------------------
# Retired with #1357 PR-5 (single oracle = the compiled engine):
# - the ``PerfDatabase.query_gemm`` -> ``GEMM._query_gemm_table`` delegation
#   and the SOL-mode formula smokes (the classmethods are gone; the public
#   facades are engine-routed deprecation shims);
# - ``GEMM._correct_sol`` load-time clamping (the engine clamps its own load);
# - per-op silicon-source attribution and the ``below_grid_sol`` degradation
#   flag threading (below-grid handling is engine-internal now).
# Anchored by tests/cross_package/test_query_shim_baseline.py and the frozen
# parity goldens.
# ---------------------------------------------------------------------------

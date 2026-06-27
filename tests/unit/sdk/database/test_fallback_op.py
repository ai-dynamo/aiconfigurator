# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FallbackOp and MLAModule operations."""

from unittest.mock import MagicMock

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import FallbackOp, MLAModule, PerformanceResult
from aiconfigurator.sdk.perf_database import PerfDataNotAvailableError


def _make_mock_db():
    """Create a mock database with _default_database_mode."""
    db = MagicMock()
    db._default_database_mode = common.DatabaseMode.SILICON
    return db


pytestmark = pytest.mark.unit


def _make_mock_op(latency: float, energy: float, weights: float = 0.0):
    """Create a mock operation that returns the given latency/energy/weights."""
    op = MagicMock()
    op._name = "mock_op"
    op.query.return_value = PerformanceResult(latency, energy=energy)
    op.get_weights.return_value = weights
    return op


def _make_failing_op(error_cls=PerfDataNotAvailableError, msg="data not available"):
    """Create a mock operation that raises on query."""
    op = MagicMock()
    op._name = "failing_op"
    op.query.side_effect = error_cls(msg)
    op.get_weights.return_value = 0.0
    return op


class TestFallbackOp:
    """Test cases for FallbackOp class."""

    def test_primary_succeeds(self):
        """When primary succeeds, fallback ops are never called."""
        mock_db = _make_mock_db()
        primary = _make_mock_op(10.0, 100.0)
        fallback_1 = _make_mock_op(5.0, 50.0)
        fallback_2 = _make_mock_op(3.0, 30.0)

        op = FallbackOp("test", primary=primary, fallback=[fallback_1, fallback_2])
        result = op.query(mock_db, batch_size=4)

        assert float(result) == 10.0
        assert result.energy == 100.0
        primary.query.assert_called_once()
        fallback_1.query.assert_not_called()
        fallback_2.query.assert_not_called()

    def test_primary_fails_fallback_succeeds(self):
        """When primary raises PerfDataNotAvailableError, fallback ops are summed."""
        mock_db = _make_mock_db()
        primary = _make_failing_op(PerfDataNotAvailableError)
        fallback_1 = _make_mock_op(5.0, 50.0)
        fallback_2 = _make_mock_op(3.0, 30.0)

        op = FallbackOp("test", primary=primary, fallback=[fallback_1, fallback_2])
        result = op.query(mock_db, batch_size=4)

        assert float(result) == 8.0  # 5 + 3
        assert result.energy == 80.0  # 50 + 30

    @pytest.mark.parametrize(
        ("error_cls", "message"),
        ((KeyError, "fp8_block"), (AssertionError, "values is None or empty")),
    )
    def test_raw_primary_errors_propagate(self, error_cls, message):
        """Untyped schema/programming errors must never be converted to fallback."""
        mock_db = _make_mock_db()
        primary = _make_failing_op(error_cls, message)
        fallback_1 = _make_mock_op(5.0, 50.0)

        op = FallbackOp("test", primary=primary, fallback=[fallback_1])
        for batch_size in (4, 8):
            with pytest.raises(error_cls):
                op.query(mock_db, batch_size=batch_size)

        assert primary.query.call_count == 2
        fallback_1.query.assert_not_called()

    def test_both_fail_raises(self):
        """When primary fails and fallback also fails, the fallback error propagates."""
        mock_db = _make_mock_db()
        primary = _make_failing_op(PerfDataNotAvailableError, "no module data")
        fallback_1 = _make_failing_op(PerfDataNotAvailableError, "no granular data")

        op = FallbackOp("test", primary=primary, fallback=[fallback_1])
        with pytest.raises(PerfDataNotAvailableError, match="no granular data"):
            op.query(mock_db, batch_size=4)

    def test_unexpected_error_not_caught(self):
        """Errors other than the expected types are not caught."""
        mock_db = _make_mock_db()
        primary = _make_failing_op(ValueError, "unexpected")
        fallback_1 = _make_mock_op(5.0, 50.0)

        op = FallbackOp("test", primary=primary, fallback=[fallback_1])
        with pytest.raises(ValueError, match="unexpected"):
            op.query(mock_db, batch_size=4)

    def test_primary_skipped_after_perf_data_not_available(self):
        """Once primary fails with PerfDataNotAvailableError, it is skipped on subsequent calls."""
        mock_db = _make_mock_db()
        primary = _make_failing_op(PerfDataNotAvailableError)
        fallback_1 = _make_mock_op(5.0, 50.0)

        op = FallbackOp("test", primary=primary, fallback=[fallback_1])
        op.query(mock_db, batch_size=4)
        op.query(mock_db, batch_size=8)
        op.query(mock_db, batch_size=16)

        # Primary should only be called once (the first attempt)
        assert primary.query.call_count == 1
        # Fallback should be called for all three queries
        assert fallback_1.query.call_count == 3

    def test_primary_forces_silicon_mode(self):
        """A legacy child is deep-copied: SILICON mode and caches stay isolated."""

        class LegacyDatabase:
            def __init__(self):
                self._default_database_mode = common.DatabaseMode.HYBRID
                self._extracted_metrics_cache = {"table": {"hits": []}}

        mock_db = LegacyDatabase()

        primary = _make_mock_op(10.0, 100.0)
        seen_modes = []
        original_modes = []
        seen_databases = []
        seen_caches = []

        def _query(database, **kwargs):
            seen_modes.append(database._default_database_mode)
            original_modes.append(mock_db._default_database_mode)
            seen_databases.append(database)
            seen_caches.append(database._extracted_metrics_cache)
            database._extracted_metrics_cache["table"]["hits"].append("primary")
            return PerformanceResult(10.0, energy=100.0)

        primary.query.side_effect = _query
        fallback_1 = _make_mock_op(5.0, 50.0)

        op = FallbackOp("test", primary=primary, fallback=[fallback_1])
        op.query(mock_db, batch_size=4)

        assert seen_modes == [common.DatabaseMode.SILICON]
        assert original_modes == [common.DatabaseMode.HYBRID]
        assert seen_databases[0] is not mock_db
        assert seen_caches[0] is not mock_db._extracted_metrics_cache
        assert seen_caches[0]["table"] is not mock_db._extracted_metrics_cache["table"]
        assert mock_db._extracted_metrics_cache == {"table": {"hits": []}}
        assert mock_db._default_database_mode == common.DatabaseMode.HYBRID

    def test_primary_reuses_cached_silicon_query_view(self):
        """Repeated HYBRID fallback attempts reuse one immutable SILICON child."""

        class FakeDatabase:
            def __init__(self, mode):
                self._default_database_mode = mode
                self.transfer_policy = common.ALL_TRANSFERS
                self.views = {}

            def query_view(self, mode, transfer_policy=None):
                key = (mode, common.resolve_transfer_policy(transfer_policy))
                if key not in self.views:
                    child = FakeDatabase(mode)
                    child.transfer_policy = key[1]
                    self.views[key] = child
                return self.views[key]

        hybrid_db = FakeDatabase(common.DatabaseMode.HYBRID)
        primary = _make_mock_op(10.0, 100.0)
        seen = []

        def _query(database, **kwargs):
            seen.append((database, database._default_database_mode, hybrid_db._default_database_mode))
            return PerformanceResult(10.0, energy=100.0)

        primary.query.side_effect = _query
        op = FallbackOp("test", primary=primary, fallback=[_make_mock_op(5.0, 50.0)])

        op.query(hybrid_db, batch_size=4)
        op.query(hybrid_db, batch_size=8)

        assert len({id(database) for database, _, _ in seen}) == 1
        assert all(database is not hybrid_db for database, _, _ in seen)
        assert all(mode is common.DatabaseMode.SILICON for _, mode, _ in seen)
        assert all(original_mode is common.DatabaseMode.HYBRID for _, _, original_mode in seen)
        assert hybrid_db._default_database_mode is common.DatabaseMode.HYBRID

    def test_primary_respects_explicit_sol_mode(self):
        """Primary uses SOL mode directly when the caller explicitly requests SOL."""
        mock_db = _make_mock_db()
        mock_db._default_database_mode = common.DatabaseMode.SOL

        primary = _make_mock_op(10.0, 100.0)
        seen_modes = []

        def _query(database, **kwargs):
            seen_modes.append(database._default_database_mode)
            return PerformanceResult(4.0, energy=0.0, source="sol")

        primary.query.side_effect = _query
        fallback_1 = _make_mock_op(5.0, 50.0)

        op = FallbackOp("test", primary=primary, fallback=[fallback_1])
        result = op.query(mock_db, batch_size=4)

        assert seen_modes == [common.DatabaseMode.SOL]
        assert float(result) == 4.0
        assert result.source == "sol"
        assert mock_db._default_database_mode == common.DatabaseMode.SOL

    def test_database_mode_restored_after_primary_failure(self):
        """Database mode is restored to original even when primary fails."""
        mock_db = _make_mock_db()
        mock_db._default_database_mode = common.DatabaseMode.HYBRID

        primary = _make_failing_op(PerfDataNotAvailableError)
        fallback_1 = _make_mock_op(5.0, 50.0)

        op = FallbackOp("test", primary=primary, fallback=[fallback_1])
        op.query(mock_db, batch_size=4)

        assert mock_db._default_database_mode == common.DatabaseMode.HYBRID

    def test_get_weights_from_primary(self):
        """get_weights uses primary when it has nonzero weights."""
        primary = _make_mock_op(10.0, 100.0, weights=500.0)
        fallback_1 = _make_mock_op(5.0, 50.0, weights=200.0)
        fallback_2 = _make_mock_op(3.0, 30.0, weights=100.0)

        op = FallbackOp("test", primary=primary, fallback=[fallback_1, fallback_2])
        assert op.get_weights() == 500.0

    def test_get_weights_from_fallback(self):
        """get_weights sums fallback ops when primary has zero weights."""
        primary = _make_mock_op(10.0, 100.0, weights=0.0)
        fallback_1 = _make_mock_op(5.0, 50.0, weights=200.0)
        fallback_2 = _make_mock_op(3.0, 30.0, weights=100.0)

        op = FallbackOp("test", primary=primary, fallback=[fallback_1, fallback_2])
        assert op.get_weights() == 300.0


class TestMLAModule:
    """Test cases for MLAModule class."""

    def test_context_calls_context_query(self):
        """Context MLAModule calls query_context_mla_module."""
        from aiconfigurator.sdk import common

        mock_db = MagicMock()
        mock_db.query_context_mla_module.return_value = PerformanceResult(10.0, energy=100.0)

        op = MLAModule(
            "test_ctx",
            1.0,
            True,
            16,
            common.KVCacheQuantMode.fp8,
            common.FMHAQuantMode.bfloat16,
            common.GEMMQuantMode.fp8_block,
        )
        result = op.query(mock_db, batch_size=4, s=4000, prefix=0)

        mock_db.query_context_mla_module.assert_called_once_with(
            b=4,
            s=4000,
            prefix=0,
            num_heads=16,
            kvcache_quant_mode=common.KVCacheQuantMode.fp8,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        )
        mock_db.query_generation_mla_module.assert_not_called()
        assert float(result) == 10.0

    def test_generation_calls_generation_query(self):
        """Generation MLAModule calls query_generation_mla_module."""
        from aiconfigurator.sdk import common

        mock_db = MagicMock()
        mock_db.query_generation_mla_module.return_value = PerformanceResult(5.0, energy=50.0)

        op = MLAModule(
            "test_gen",
            1.0,
            False,
            16,
            common.KVCacheQuantMode.fp8,
            common.FMHAQuantMode.bfloat16,
            common.GEMMQuantMode.fp8_block,
        )
        result = op.query(mock_db, batch_size=4, s=4000, beam_width=1)

        mock_db.query_generation_mla_module.assert_called_once_with(
            b=4,
            s=4000,
            num_heads=16,
            kv_cache_dtype=common.KVCacheQuantMode.fp8,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        )
        mock_db.query_context_mla_module.assert_not_called()
        assert float(result) == 5.0

    def test_generation_rejects_beam_width_not_1(self):
        """Generation MLAModule raises ValueError for beam_width != 1."""
        from aiconfigurator.sdk import common

        mock_db = MagicMock()
        op = MLAModule(
            "test_gen",
            1.0,
            False,
            16,
            common.KVCacheQuantMode.fp8,
            common.FMHAQuantMode.bfloat16,
            common.GEMMQuantMode.fp8_block,
        )
        with pytest.raises(ValueError, match="beam_width=1"):
            op.query(mock_db, batch_size=4, s=4000, beam_width=2)

    def test_scale_factor_applied(self):
        """Scale factor is applied to both latency and energy."""
        from aiconfigurator.sdk import common

        mock_db = MagicMock()
        mock_db.query_context_mla_module.return_value = PerformanceResult(10.0, energy=100.0)

        op = MLAModule(
            "test",
            0.5,
            True,
            16,
            common.KVCacheQuantMode.fp8,
            common.FMHAQuantMode.bfloat16,
            common.GEMMQuantMode.fp8_block,
        )
        result = op.query(mock_db, batch_size=1, s=1000, prefix=0)

        assert float(result) == pytest.approx(5.0)
        assert result.energy == pytest.approx(50.0)

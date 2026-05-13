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

    def test_primary_fails_with_key_error(self):
        """FallbackOp catches KeyError from missing quant mode combinations."""
        mock_db = _make_mock_db()
        primary = _make_failing_op(KeyError, "fp8_block")
        fallback_1 = _make_mock_op(7.0, 70.0)

        op = FallbackOp("test", primary=primary, fallback=[fallback_1])
        result = op.query(mock_db, batch_size=4)

        assert float(result) == 7.0
        assert result.energy == 70.0

    def test_primary_fails_with_assertion_error(self):
        """FallbackOp catches AssertionError from empty interpolation data."""
        mock_db = _make_mock_db()
        primary = _make_failing_op(AssertionError, "values is None or empty")
        fallback_1 = _make_mock_op(7.0, 70.0)

        op = FallbackOp("test", primary=primary, fallback=[fallback_1])
        result = op.query(mock_db, batch_size=4)

        assert float(result) == 7.0

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

    def test_primary_retried_after_key_error(self):
        """KeyError doesn't permanently disable primary — it may work for other params."""
        mock_db = _make_mock_db()
        primary = _make_failing_op(KeyError, "fp8_block")
        fallback_1 = _make_mock_op(5.0, 50.0)

        op = FallbackOp("test", primary=primary, fallback=[fallback_1])
        op.query(mock_db, batch_size=4)
        op.query(mock_db, batch_size=8)

        # Primary should be retried on each call
        assert primary.query.call_count == 2

    def test_primary_forces_silicon_mode(self):
        """Primary is queried with SILICON mode even when database uses HYBRID."""
        mock_db = _make_mock_db()
        mock_db._default_database_mode = common.DatabaseMode.HYBRID

        primary = _make_mock_op(10.0, 100.0)
        fallback_1 = _make_mock_op(5.0, 50.0)

        op = FallbackOp("test", primary=primary, fallback=[fallback_1])
        op.query(mock_db, batch_size=4)

        # During primary query, database mode should have been SILICON
        # After query, it should be restored to HYBRID
        assert mock_db._default_database_mode == common.DatabaseMode.HYBRID

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

    def test_primary_error_logs_not_leaked(self):
        """The perf_database logger level is restored after primary failure."""
        import logging

        mock_db = _make_mock_db()
        primary = _make_failing_op(PerfDataNotAvailableError)
        fallback_1 = _make_mock_op(5.0, 50.0)

        perf_logger = logging.getLogger("aiconfigurator.sdk.perf_database")
        original_level = perf_logger.level

        op = FallbackOp("test", primary=primary, fallback=[fallback_1])
        op.query(mock_db, batch_size=4)

        assert perf_logger.level == original_level

    def test_fallback_result_tagged_with_source_fallback(self):
        """Result returned via fallback path is tagged source='fallback' (AIC-1059)."""
        mock_db = _make_mock_db()
        primary = _make_failing_op(PerfDataNotAvailableError)
        fallback_1 = _make_mock_op(5.0, 50.0)

        op = FallbackOp("test", primary=primary, fallback=[fallback_1])
        result = op.query(mock_db, batch_size=4)

        assert result.source == "fallback"

    def test_primary_success_preserves_source(self):
        """When primary succeeds, its source tag is preserved (not overwritten)."""
        mock_db = _make_mock_db()
        primary = MagicMock()
        primary._name = "primary"
        primary.query.return_value = PerformanceResult(10.0, energy=100.0, source="silicon")
        fallback_1 = _make_mock_op(5.0, 50.0)

        op = FallbackOp("test", primary=primary, fallback=[fallback_1])
        result = op.query(mock_db, batch_size=4)

        assert result.source == "silicon"

    def test_warning_logged_on_first_fallback(self, caplog):
        """First time primary fails, FallbackOp emits a WARNING (AIC-1059)."""
        import logging

        mock_db = _make_mock_db()
        primary = _make_failing_op(PerfDataNotAvailableError, "missing module data")
        primary._name = "context_mla_module"
        fallback_1 = _make_mock_op(5.0, 50.0)
        fallback_1._name = "context_attention"

        op = FallbackOp("context_mla_block", primary=primary, fallback=[fallback_1])

        with caplog.at_level(logging.WARNING, logger="aiconfigurator.sdk.operations"):
            op.query(mock_db, batch_size=4)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        msg = warnings[0].getMessage()
        assert "context_mla_block" in msg
        assert "context_mla_module" in msg
        assert "PerfDataNotAvailableError" in msg
        assert "context_attention" in msg  # fallback op listed
        assert "missing module data" in msg

    def test_warning_deduplicated_per_exception_type(self, caplog):
        """Repeated failures with the same exception type warn ONCE, then debug-log.

        FallbackOp is queried many times during a sweep. Spamming WARNING on
        every iteration would drown out other diagnostics. We warn once per
        (instance, exception type), then drop to DEBUG.
        """
        import logging

        mock_db = _make_mock_db()
        # Use KeyError so the primary is retried on every call (not latched off).
        primary = _make_failing_op(KeyError, "fp8_block")
        fallback_1 = _make_mock_op(5.0, 50.0)

        op = FallbackOp("test", primary=primary, fallback=[fallback_1])

        with caplog.at_level(logging.WARNING, logger="aiconfigurator.sdk.operations"):
            for _ in range(10):
                op.query(mock_db, batch_size=4)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, f"expected 1 warning, got {len(warnings)}"

    def test_warning_emitted_per_exception_type(self, caplog):
        """Different exception types each get their own warning."""
        import logging

        mock_db = _make_mock_db()
        # Cycle through different exception types across calls.
        primary = MagicMock()
        primary._name = "primary"
        primary.query.side_effect = [
            KeyError("missing key"),
            AssertionError("empty data"),
            KeyError("missing key again"),
        ]
        fallback_1 = _make_mock_op(5.0, 50.0)

        op = FallbackOp("test", primary=primary, fallback=[fallback_1])

        with caplog.at_level(logging.WARNING, logger="aiconfigurator.sdk.operations"):
            for _ in range(3):
                op.query(mock_db, batch_size=4)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        # First KeyError + first AssertionError -> 2 warnings; second KeyError dedup'd.
        assert len(warnings) == 2

    def test_no_warning_when_primary_succeeds(self, caplog):
        """No warning is emitted on the happy path."""
        import logging

        mock_db = _make_mock_db()
        primary = _make_mock_op(10.0, 100.0)
        fallback_1 = _make_mock_op(5.0, 50.0)

        op = FallbackOp("test", primary=primary, fallback=[fallback_1])

        with caplog.at_level(logging.WARNING, logger="aiconfigurator.sdk.operations"):
            op.query(mock_db, batch_size=4)

        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


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

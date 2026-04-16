# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FallbackOp and MLAModule operations."""

from unittest.mock import MagicMock

import pytest

from aiconfigurator.sdk.operations import FallbackOp, MLAModule, PerformanceResult
from aiconfigurator.sdk.perf_database import PerfDataNotAvailableError

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
        mock_db = MagicMock()
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
        mock_db = MagicMock()
        primary = _make_failing_op(PerfDataNotAvailableError)
        fallback_1 = _make_mock_op(5.0, 50.0)
        fallback_2 = _make_mock_op(3.0, 30.0)

        op = FallbackOp("test", primary=primary, fallback=[fallback_1, fallback_2])
        result = op.query(mock_db, batch_size=4)

        assert float(result) == 8.0  # 5 + 3
        assert result.energy == 80.0  # 50 + 30

    def test_primary_fails_with_key_error(self):
        """FallbackOp catches KeyError from missing quant mode combinations."""
        mock_db = MagicMock()
        primary = _make_failing_op(KeyError, "fp8_block")
        fallback_1 = _make_mock_op(7.0, 70.0)

        op = FallbackOp("test", primary=primary, fallback=[fallback_1])
        result = op.query(mock_db, batch_size=4)

        assert float(result) == 7.0
        assert result.energy == 70.0

    def test_primary_fails_with_assertion_error(self):
        """FallbackOp catches AssertionError from empty interpolation data."""
        mock_db = MagicMock()
        primary = _make_failing_op(AssertionError, "values is None or empty")
        fallback_1 = _make_mock_op(7.0, 70.0)

        op = FallbackOp("test", primary=primary, fallback=[fallback_1])
        result = op.query(mock_db, batch_size=4)

        assert float(result) == 7.0

    def test_both_fail_raises(self):
        """When primary fails and fallback also fails, the fallback error propagates."""
        mock_db = MagicMock()
        primary = _make_failing_op(PerfDataNotAvailableError, "no module data")
        fallback_1 = _make_failing_op(PerfDataNotAvailableError, "no granular data")

        op = FallbackOp("test", primary=primary, fallback=[fallback_1])
        with pytest.raises(PerfDataNotAvailableError, match="no granular data"):
            op.query(mock_db, batch_size=4)

    def test_unexpected_error_not_caught(self):
        """Errors other than the expected types are not caught."""
        mock_db = MagicMock()
        primary = _make_failing_op(ValueError, "unexpected")
        fallback_1 = _make_mock_op(5.0, 50.0)

        op = FallbackOp("test", primary=primary, fallback=[fallback_1])
        with pytest.raises(ValueError, match="unexpected"):
            op.query(mock_db, batch_size=4)

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

        mock_db = MagicMock()
        primary = _make_failing_op(PerfDataNotAvailableError)
        fallback_1 = _make_mock_op(5.0, 50.0)

        perf_logger = logging.getLogger("aiconfigurator.sdk.perf_database")
        original_level = perf_logger.level

        op = FallbackOp("test", primary=primary, fallback=[fallback_1])
        op.query(mock_db, batch_size=4)

        assert perf_logger.level == original_level


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
            common.FMHAQuantMode.float16,
            common.GEMMQuantMode.fp8_block,
        )
        result = op.query(mock_db, batch_size=4, s=4000, prefix=0)

        mock_db.query_context_mla_module.assert_called_once_with(
            b=4,
            s=4000,
            prefix=0,
            num_heads=16,
            kvcache_quant_mode=common.KVCacheQuantMode.fp8,
            fmha_quant_mode=common.FMHAQuantMode.float16,
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
            common.FMHAQuantMode.float16,
            common.GEMMQuantMode.fp8_block,
        )
        result = op.query(mock_db, batch_size=4, s=4000, beam_width=1)

        mock_db.query_generation_mla_module.assert_called_once_with(
            b=4,
            s=4000,
            num_heads=16,
            kv_cache_dtype=common.KVCacheQuantMode.fp8,
            fmha_quant_mode=common.FMHAQuantMode.float16,
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
            common.FMHAQuantMode.float16,
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
            common.FMHAQuantMode.float16,
            common.GEMMQuantMode.fp8_block,
        )
        result = op.query(mock_db, batch_size=1, s=1000, prefix=0)

        assert float(result) == pytest.approx(5.0)
        assert result.energy == pytest.approx(50.0)

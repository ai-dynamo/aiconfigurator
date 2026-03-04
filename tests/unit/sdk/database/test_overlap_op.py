# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OverlapOp operation."""

from unittest.mock import MagicMock

import pytest

from aiconfigurator.sdk.operations import OverlapOp, PerformanceResult

pytestmark = pytest.mark.unit


def _make_mock_op(latency: float, energy: float, weights: float = 0.0):
    """Create a mock operation that returns the given latency/energy/weights."""
    op = MagicMock()
    op.query.return_value = PerformanceResult(latency, energy=energy)
    op.get_weights.return_value = weights
    return op


class TestOverlapOp:
    """Test cases for OverlapOp class."""

    def test_initialization(self):
        """Test that group_a and group_b are stored correctly."""
        op_a = _make_mock_op(10.0, 1.0)
        op_b = _make_mock_op(5.0, 2.0)

        overlap = OverlapOp("test_overlap", group_a=[op_a], group_b=[op_b])

        assert overlap._name == "test_overlap"
        assert overlap._group_a == [op_a]
        assert overlap._group_b == [op_b]

    def test_query_group_a_slower(self):
        """Latency should equal group_a when group_a is slower."""
        mock_db = MagicMock()
        op_a1 = _make_mock_op(10.0, 100.0)
        op_a2 = _make_mock_op(8.0, 80.0)
        op_b1 = _make_mock_op(5.0, 50.0)

        overlap = OverlapOp("test", group_a=[op_a1, op_a2], group_b=[op_b1])
        result = overlap.query(mock_db, x=16)

        assert isinstance(result, PerformanceResult)
        assert float(result) == 18.0  # max(10+8, 5) = 18
        assert result.energy == 230.0  # 100+80+50

    def test_query_group_b_slower(self):
        """Latency should equal group_b when group_b is slower."""
        mock_db = MagicMock()
        op_a1 = _make_mock_op(3.0, 30.0)
        op_b1 = _make_mock_op(7.0, 70.0)
        op_b2 = _make_mock_op(6.0, 60.0)

        overlap = OverlapOp("test", group_a=[op_a1], group_b=[op_b1, op_b2])
        result = overlap.query(mock_db, x=16)

        assert float(result) == 13.0  # max(3, 7+6) = 13
        assert result.energy == 160.0  # 30+70+60

    def test_query_equal_latency(self):
        """Latency should be correct when both groups have identical latency."""
        mock_db = MagicMock()
        op_a = _make_mock_op(10.0, 100.0)
        op_b = _make_mock_op(10.0, 200.0)

        overlap = OverlapOp("test", group_a=[op_a], group_b=[op_b])
        result = overlap.query(mock_db, x=16)

        assert float(result) == 10.0
        assert result.energy == 300.0

    def test_query_empty_group_b(self):
        """When group_b is empty, latency equals group_a total."""
        mock_db = MagicMock()
        op_a = _make_mock_op(10.0, 100.0)

        overlap = OverlapOp("test", group_a=[op_a], group_b=[])
        result = overlap.query(mock_db, x=16)

        assert float(result) == 10.0  # max(10, 0)
        assert result.energy == 100.0

    def test_query_empty_group_a(self):
        """When group_a is empty, latency equals group_b total."""
        mock_db = MagicMock()
        op_b = _make_mock_op(7.0, 70.0)

        overlap = OverlapOp("test", group_a=[], group_b=[op_b])
        result = overlap.query(mock_db, x=16)

        assert float(result) == 7.0  # max(0, 7)
        assert result.energy == 70.0

    def test_query_passes_kwargs_to_inner_ops(self):
        """Verify that kwargs (e.g. x) are forwarded to inner ops."""
        mock_db = MagicMock()
        op_a = _make_mock_op(1.0, 1.0)
        op_b = _make_mock_op(1.0, 1.0)

        overlap = OverlapOp("test", group_a=[op_a], group_b=[op_b])
        overlap.query(mock_db, x=32)

        op_a.query.assert_called_once_with(mock_db, x=32)
        op_b.query.assert_called_once_with(mock_db, x=32)

    def test_get_weights_sums_all_ops(self):
        """get_weights should return sum of weights from both groups."""
        op_a1 = _make_mock_op(1.0, 1.0, weights=100.0)
        op_a2 = _make_mock_op(1.0, 1.0, weights=200.0)
        op_b1 = _make_mock_op(1.0, 1.0, weights=50.0)

        overlap = OverlapOp("test", group_a=[op_a1, op_a2], group_b=[op_b1])

        assert overlap.get_weights() == 350.0  # 100+200+50

    def test_get_weights_empty_groups(self):
        """get_weights should return 0 when both groups are empty."""
        overlap = OverlapOp("test", group_a=[], group_b=[])
        assert overlap.get_weights() == 0.0

    def test_query_multiple_ops_per_group(self):
        """Test with multiple ops in both groups to verify summation logic."""
        mock_db = MagicMock()
        group_a = [_make_mock_op(3.0, 30.0), _make_mock_op(4.0, 40.0), _make_mock_op(5.0, 50.0)]
        group_b = [_make_mock_op(6.0, 60.0), _make_mock_op(7.0, 70.0)]

        overlap = OverlapOp("test", group_a=group_a, group_b=group_b)
        result = overlap.query(mock_db, x=16)

        assert float(result) == 13.0  # max(3+4+5=12, 6+7=13) = 13
        assert result.energy == 250.0  # 30+40+50+60+70


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

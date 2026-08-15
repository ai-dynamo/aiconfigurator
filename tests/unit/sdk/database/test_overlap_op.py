# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OverlapOp operation.

The latency composition (``max(sum_a, sum_b)``, summed energy, kwarg
forwarding to inner ops) retired to the compiled engine with #1357 PR-5:
``OverlapOp.query`` is now a deprecation shim that converts the whole
composite (children included) and evaluates it in Rust, so there is no
Python seam where per-child queries happen. That behaviour is anchored by
the frozen parity goldens and tests/cross_package/test_query_shim_baseline.py.
What stays Python-owned — construction and weight accounting — is tested here.
"""

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OverlapOp operation.

The latency composition (``max(sum_a, sum_b)``, summed energy, kwarg
forwarding to inner ops) retired to the compiled engine with #1357 PR-5:
``OverlapOp.query`` is now a deprecation shim that converts the whole
composite (children included) and evaluates it in Rust, so there is no
Python seam where per-child queries happen. That behaviour is anchored by
the frozen parity goldens.
What stays Python-owned — construction and the aggregation shape of weight
accounting (per-child weights themselves come from the engine since PR-6) —
is tested here.
"""

import pytest

from aiconfigurator.sdk.operations import OverlapOp

pytestmark = pytest.mark.unit


class TestOverlapOp:
    """Test cases for OverlapOp class."""

    def test_initialization(self):
        """Groups are captured by the Rust constructor; the getters re-wrap
        children as Rust base classes with the same names and wire state
        (Mock children are refused — composites require engine-backed ops)."""
        from aiconfigurator.sdk import common
        from aiconfigurator.sdk.operations import GEMM

        op_a = GEMM("a", 1.0, 10, 5, common.GEMMQuantMode.bfloat16)
        op_b = GEMM("b", 1.0, 5, 5, common.GEMMQuantMode.bfloat16)

        overlap = OverlapOp("test_overlap", group_a=[op_a], group_b=[op_b])

        assert overlap._name == "test_overlap"
        assert [c._name for c in overlap._group_a] == ["a"]
        assert [c._name for c in overlap._group_b] == ["b"]
        assert overlap._group_a[0]._spec_json() == op_a._spec_json()

        with pytest.raises(TypeError, match="engine-backed"):
            OverlapOp("bad", group_a=[object()], group_b=[])

    def test_get_weights_sums_all_ops(self):
        """get_weights should return sum of weights from both groups.

        Weights route through the engine (PR-6), so the children must be
        real spec-expressible ops: bf16 GEMM weighs n*k*2 bytes."""
        from aiconfigurator.sdk import common
        from aiconfigurator.sdk.operations import GEMM

        op_a1 = GEMM("a1", 1.0, 10, 5, common.GEMMQuantMode.bfloat16)  # 100 B
        op_a2 = GEMM("a2", 1.0, 20, 5, common.GEMMQuantMode.bfloat16)  # 200 B
        op_b1 = GEMM("b1", 1.0, 5, 5, common.GEMMQuantMode.bfloat16)  # 50 B

        overlap = OverlapOp("test", group_a=[op_a1, op_a2], group_b=[op_b1])

        assert overlap.get_weights() == 350.0  # 100+200+50

    def test_get_weights_empty_groups(self):
        """get_weights should return 0 when both groups are empty."""
        overlap = OverlapOp("test", group_a=[], group_b=[])
        assert overlap.get_weights() == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

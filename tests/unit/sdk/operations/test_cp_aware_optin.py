# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression for the per-op ``_CP_AWARE`` opt-in gate.

The ``Operation`` base class raises ``NotImplementedError`` at construction
when ``seq_split > 1`` is passed to a subclass that has not explicitly opted
in (``_CP_AWARE = True``). This protects against silently mis-modeling CP for
new ops: adding a token-major op without auditing how it should respond to
``seq_split`` fails loudly rather than producing wrong perf numbers.
"""

from __future__ import annotations

import pytest

import aiconfigurator.sdk.common as common
import aiconfigurator.sdk.operations as ops
from aiconfigurator.sdk.operations.base import Operation

pytestmark = pytest.mark.unit


class _UnauditedOp(Operation):
    """Subclass without ``_CP_AWARE`` opt-in (default False)."""


class _AuditedOp(Operation):
    _CP_AWARE = True


def test_unaudited_op_raises_when_seq_split_gt_one():
    with pytest.raises(NotImplementedError, match="not been audited for context parallelism"):
        _UnauditedOp("unaudited", 1.0, seq_split=4)


def test_unaudited_op_constructs_when_seq_split_is_one():
    assert _UnauditedOp("unaudited", 1.0, seq_split=1)._seq_split == 1


def test_unaudited_op_constructs_when_seq_split_omitted():
    assert _UnauditedOp("unaudited", 1.0)._seq_split == 1


def test_audited_op_accepts_seq_split_gt_one():
    assert _AuditedOp("audited", 1.0, seq_split=8)._seq_split == 8


# --- Real ops: token-major comm/gemm are audited; Mamba SSM is deliberately not.


def test_gemm_is_cp_aware():
    """GEMM divides its M-axis by seq_split, so it must accept seq_split>1."""
    op = ops.GEMM("g", 1, 4096, 4096, common.GEMMQuantMode.bfloat16, seq_split=8)
    assert op._seq_split == 8


def test_nccl_is_cp_aware():
    op = ops.NCCL(
        "ag",
        1,
        "all_gather",
        num_elements_per_token=128,
        num_gpus=8,
        comm_quant_mode=common.CommQuantMode.half,
        seq_split=8,
    )
    assert op._seq_split == 8


def test_mamba_ssm_is_not_cp_aware():
    """Mamba SSM scan is order-dependent -> cannot shard tokens; fail loud."""
    with pytest.raises(NotImplementedError, match="not been audited for context parallelism"):
        ops.Mamba2Kernel("m", 1, "trtllm", "context", 4096, 8, 128, 128, 4, 1, 256, seq_split=2)


def test_mamba_ssm_constructs_at_seq_split_one():
    op = ops.Mamba2Kernel("m", 1, "trtllm", "context", 4096, 8, 128, 128, 4, 1, 256, seq_split=1)
    assert op._seq_split == 1

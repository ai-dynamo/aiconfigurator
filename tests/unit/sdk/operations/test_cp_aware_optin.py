# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression for the per-op ``_CP_AWARE`` opt-in gate.

The ``Operation`` base class raises ``NotImplementedError`` at
construction when ``seq_split > 1`` is passed to a subclass that has not
explicitly opted in. The goal is to protect against silently mis-modeling
CP for new ops -- adding a token-major op without auditing how it should
respond to ``seq_split`` fails loudly rather than producing wrong perf
numbers.
"""

from __future__ import annotations

import pytest

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
    op = _UnauditedOp("unaudited", 1.0, seq_split=1)
    assert op._seq_split == 1


def test_unaudited_op_constructs_when_seq_split_omitted():
    op = _UnauditedOp("unaudited", 1.0)
    assert op._seq_split == 1


def test_audited_op_accepts_seq_split_gt_one():
    op = _AuditedOp("audited", 1.0, seq_split=8)
    assert op._seq_split == 8

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the DeepSeek-V4 CP prefill composition (``_query_cp``).

Mirrors ``test_cp_dsa_modeling.py``: the components (base module, sparse
kernel, topk calib, nccl) are stubbed with known values so the tests lock the
COMPOSITION arithmetic — per-card monolithic base + full/cp swap of the
super-linear sub-kernels + CP all-gathers — independent of table data.
The communication assertions mirror the SGLang 0.5.16 DSV4 tensor shapes
and dtypes, including fp32 compressor-score gathers.
"""

from unittest.mock import MagicMock

import pytest

from aiconfigurator.sdk.errors import PerfDataNotAvailableError
from aiconfigurator.sdk.operations.dsv4 import ContextDeepSeekV4AttentionModule

pytestmark = pytest.mark.unit


def _make_module(cp, ratio, window=None):
    m = ContextDeepSeekV4AttentionModule.__new__(ContextDeepSeekV4AttentionModule)
    m._cp_size = cp
    m._compress_ratio = ratio
    m._head_dim = 512
    m._index_head_dim = 128
    m._native_heads = 64
    m._tp_size = 1
    m._num_heads = 64
    m._scale_factor = 1.0
    m._window_size = window
    return m


def test_query_cp_csa_composition(monkeypatch):
    cp, isl, b = 8, 16384, 1
    per_card = -(-isl // cp)  # 2048
    m = _make_module(cp, ratio=4)

    monkeypatch.setattr(
        ContextDeepSeekV4AttentionModule,
        "_module_base",
        lambda self, db, bb, s, prefix: 4300.0 if s == per_card else pytest.fail("base must query per_card"),
    )
    # Chunked decomposition (chunk = 8192): mqa_full = mqa(8192, past=0)
    # + mqa(8192, past=8192); mqa_perc = mqa(2048, past=0). Keyed by
    # (isl, past) so the test locks the in-grid chunk walk, not a single
    # full-isl extrapolated lookup (review pt.1: pair count is additive).
    sparse = {(8192, 0): 900.0, (8192, 8192): 700.0, (per_card, 0): 25.0}
    monkeypatch.setattr(
        ContextDeepSeekV4AttentionModule,
        "_lookup_sparse_kernel",
        classmethod(lambda cls, db, kernel, bs, isl_q, past, tp_size, native_heads: sparse[(isl_q, past)]),
    )
    topk = {isl: 800.0, per_card: 100.0}
    monkeypatch.setattr(
        ContextDeepSeekV4AttentionModule,
        "_csa_topk_top_last",
        classmethod(lambda cls, db, isl_q, step, nh, bb: topk[isl_q]),
    )
    db = MagicMock()
    db.query_nccl.return_value = 50.0

    res = m._query_cp(db, b=b, isl=isl, prefix=0)

    # mqa_full = 900 + 700 = 1600 (two in-grid chunks); mqa_perc = 25
    # delta_mqa  = 1600/8 - 25  = 175
    # delta_topk = 800/8  - 100 = 0
    # latency = base 4300 + 175 + 0 + three DSV4 attention AGs * 50 = 4625
    assert float(res) == pytest.approx(4625.0)
    assert res.source == "estimated"
    # Global payloads, expressed as half elements:
    #   KV bf16 [isl,512]
    #   indexer kv_score fp32 [isl,4*128] -> x2 half elements
    #   attention kv_score fp32 [isl,4*512] -> x2 half elements
    ag_sizes = sorted(call.args[3] for call in db.query_nccl.call_args_list)
    assert ag_sizes == sorted(
        [
            b * isl * 512,
            b * isl * 4 * 128 * 2,
            b * isl * 4 * 512 * 2,
        ]
    )


def test_query_cp_hca_composition(monkeypatch):
    cp, isl, b, window = 4, 8192, 2, 2048
    per_card = -(-isl // cp)
    m = _make_module(cp, ratio=128, window=window)

    monkeypatch.setattr(
        ContextDeepSeekV4AttentionModule,
        "_module_base",
        lambda self, db, bb, s, prefix: 1000.0 if s == per_card else pytest.fail("base must query per_card"),
    )
    db = MagicMock()
    db.query_nccl.return_value = 30.0

    res = m._query_cp(db, b=b, isl=isl, prefix=0)

    # HCA: base + bf16 KV AG + fp32 attention-compressor kv_score AG.
    assert float(res) == pytest.approx(1000.0 + 30.0 + 30.0)
    ag_sizes = sorted(call.args[3] for call in db.query_nccl.call_args_list)
    assert ag_sizes == sorted(
        [
            b * isl * 512,
            b * isl * 2 * 512 * 2,
        ]
    )


def test_query_cp_fails_loud_without_sparse_tables(monkeypatch):
    m = _make_module(cp=2, ratio=4)
    monkeypatch.setattr(ContextDeepSeekV4AttentionModule, "_module_base", lambda self, db, bb, s, prefix: 100.0)
    monkeypatch.setattr(
        ContextDeepSeekV4AttentionModule,
        "_lookup_sparse_kernel",
        classmethod(lambda cls, db, kernel, bs, isl_q, past, tp_size, native_heads: None),
    )
    monkeypatch.setattr(
        ContextDeepSeekV4AttentionModule,
        "_csa_topk_top_last",
        classmethod(lambda cls, db, isl_q, step, nh, bb: None),
    )
    with pytest.raises(PerfDataNotAvailableError, match="sparse tables"):
        m._query_cp(MagicMock(), b=1, isl=4096, prefix=0)

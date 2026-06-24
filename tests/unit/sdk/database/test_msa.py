# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax Sparse Attention (MSA) op: SOL and the cross-op (XOP) transfer from DSA.

MSA has no own silicon data, so its empirical path is a cross-op transfer gated by the
XOP transfer kind. These tests cover the SOL path and the XOP gate (the headline new op
otherwise had no coverage)."""

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.errors import EmpiricalNotImplementedError

pytestmark = pytest.mark.unit


def _ctx_msa():
    from aiconfigurator.sdk.operations.msa import ContextMSAModule

    # M3-like per-GPU shape: 8 q / 1 kv heads, head_dim 128, v 128, top-16 blocks * 128.
    return ContextMSAModule(
        "msa",
        1.0,
        num_heads=8,
        num_kv_heads=1,
        hidden_size=4096,
        head_dim=128,
        v_head_dim=128,
        index_n_heads=4,
        index_head_dim=128,
        index_topk=2048,
        block_size=128,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.fp8,
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
    )


def test_msa_sol_positive(comprehensive_perf_db):
    """SOL mode computes the three-group MSA SOL (gemm + fp8 indexer + sparse attn)."""
    comprehensive_perf_db.set_default_database_mode(common.DatabaseMode.SOL)
    try:
        v = float(_ctx_msa().query(comprehensive_perf_db, batch_size=8, s=2048, prefix=0))
        assert v > 0
    finally:
        comprehensive_perf_db.set_default_database_mode(common.DatabaseMode.SILICON)


def test_msa_xop_gating(comprehensive_perf_db):
    """The DSA->MSA borrow is the XOP transfer kind. With XOP excluded, MSA (no own data)
    raises at the gate; with XOP enabled the gate is passed (it then either transfers or
    raises 'no DSA util' — not the policy gate)."""
    comprehensive_perf_db.set_default_database_mode(common.DatabaseMode.HYBRID)
    kw = dict(batch_size=8, s=2048, prefix=0)
    try:
        comprehensive_perf_db.set_transfer_policy(["xshape", "xquant"])  # no XOP
        with pytest.raises(EmpiricalNotImplementedError) as exc:
            _ctx_msa().query(comprehensive_perf_db, **kw)
        assert "xop" in str(exc.value).lower()  # gated at the policy, not a data miss

        comprehensive_perf_db.set_transfer_policy(None)  # XOP allowed
        try:
            assert float(_ctx_msa().query(comprehensive_perf_db, **kw)) > 0
        except EmpiricalNotImplementedError as exc2:
            assert "xop" not in str(exc2).lower()  # got past the gate (DSA data simply absent)
    finally:
        comprehensive_perf_db.set_transfer_policy(None)
        comprehensive_perf_db.set_default_database_mode(common.DatabaseMode.SILICON)

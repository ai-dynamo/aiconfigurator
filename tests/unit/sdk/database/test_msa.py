# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax Sparse Attention (MSA) op: per-op SOL coverage through the query shim.

The cross-op (XOP) DSA-to-MSA utilization transfer this file also used to pin
(policy gating + xop provenance tagging via the ``_dsa_context_util`` seam)
retired to the compiled engine with #1357 PR-5; transfer-ladder behaviour is
anchored by the frozen parity goldens and
the frozen parity goldens."""

import pytest

from aiconfigurator.sdk import common

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


def test_msa_sol_scales_with_workload():
    """SOL mode computes the three-group MSA SOL (gemm + fp8 indexer + sparse attn). Assert it
    RESPONDS to the workload rather than returning a constant: more new tokens (s) add work, and
    a longer cached prefix adds indexer/attention work (full_s > index_topk). Runs on a real
    shipped database: ``op._engine_query`` is the permanent internal single-op plumbing routed
    through the compiled engine's probe, which loads its tables from disk (the synthetic
    fixture is invisible to it)."""
    from aiconfigurator.sdk.perf_database import get_database_view

    db = get_database_view("b200_sxm", "sglang", "0.5.14", database_mode="SOL")
    assert db is not None, "b200_sxm/sglang/0.5.14 data missing"
    op = _ctx_msa()
    small = float(op._engine_query(db, batch_size=8, s=512, prefix=0))
    large = float(op._engine_query(db, batch_size=8, s=2048, prefix=0))
    with_prefix = float(op._engine_query(db, batch_size=8, s=2048, prefix=2048))
    assert 0 < small < large  # scales with new-token count
    assert with_prefix > large  # cached prefix adds indexer work beyond index_topk


def test_rtx_trtllm_rc23_loads_and_m3_is_explicitly_rejected():
    """Dynamo 1.3 pins trtllm 1.3.0rc23; rtx ships rc20-reuse markers so the
    exact-version gate passes (review 4969690316 Spec-3) and every reused
    family serves, while M3 MSA — no rtx table, no DSA xop donor on trtllm —
    fails with a typed empirical error (an explicitly rejected cell), never
    a silent fallback or a version-gate exit."""
    from aiconfigurator.sdk.perf_database import get_database_view

    db = get_database_view("rtx_pro_6000_server", "trtllm", "1.3.0rc23", database_mode="HYBRID")
    assert db is not None, "rc23 reuse markers must make the version root loadable"
    op = _ctx_msa()
    with pytest.raises(Exception, match=r"(?i)empirical|no DSA util"):
        op._engine_query(db, batch_size=2, s=512, prefix=0)


def test_nvfp4_checkpoint_lane_resolution_per_backend():
    """End-to-end lookup for the NVFP4 checkpoint's MSA lane (review
    4969690316 Spec-2): the SDK prices its MXFP8 projections as
    gemm=fp8_block. trtllm/vllm b200 tables carry that gemm tier — SILICON
    must resolve; the sglang tables are bf16-gemm-only by declaration (the
    checkpoint's quantized flow is unsupported in SGLang serving), so
    SILICON must miss and HYBRID takes the documented empirical transfer."""
    from aiconfigurator.sdk.operations.msa import ContextMSAModule
    from aiconfigurator.sdk.perf_database import get_database_view

    def op():
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
            index_topk=16,
            block_size=128,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        )

    for backend, version in (("trtllm", "1.3.0rc23"), ("vllm", "0.24.0")):
        db = get_database_view("b200_sxm", backend, version, database_mode="SILICON")
        assert db is not None, f"b200 {backend} data missing"
        latency = float(op()._engine_query(db, batch_size=2, s=512, prefix=0))
        assert latency > 0, f"{backend} fp8_block gemm lane must resolve in SILICON"

    sg_silicon = get_database_view("b200_sxm", "sglang", "0.5.16", database_mode="SILICON")
    assert sg_silicon is not None
    with pytest.raises(Exception, match=r"(?i)silicon|missing|not supported"):
        op()._engine_query(sg_silicon, batch_size=2, s=512, prefix=0)

    sg_hybrid = get_database_view("b200_sxm", "sglang", "0.5.16", database_mode="HYBRID")
    latency = float(op()._engine_query(sg_hybrid, batch_size=2, s=512, prefix=0))
    assert latency > 0, "sglang HYBRID must fall back to the empirical transfer for the fp8_block lane"

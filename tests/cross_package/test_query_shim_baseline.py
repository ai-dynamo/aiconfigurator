# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned pre-retirement baseline for the deprecated per-call query shims.

PR-5 of #1357 deletes the Python per-call query math and re-routes the public
surface (``PerfDatabase.query_*`` and ``Operation.query``) through the
compiled engine as one-release deprecation shims. The values in
``query_shim_baseline.json`` were captured FROM THE PYTHON MATH on this PR's
exact base commit (``--regen`` below, run on a checkout of the merge base);
this test replays every case through the live surface and requires bit-equal
results, so a shim whose op construction or kwargs mapping drifts from the
legacy semantics fails loudly. Coverage is enforced against the FROZEN
legacy-surface manifest at the bottom of this file (every retired facade,
every op class with retired query semantics, and the named semantic
dimensions — per-call overrides, verify phase, explicit zero scale,
composites, tombstones, typed misses), not against the case list itself.

Two deliberate exceptions, pinned against the ENGINE (not the legacy math)
because PR-4 moved their public semantics to the op level with review:

- ``query_context_attention``: engine values include the fused rope/kv-write
  extras charged on top of the attention table;
- ``query_gemm`` with ``fp8_static``: engine charts the op model
  (dynamic-fp8 row minus overhead tables); the legacy facade's fp8_static
  lookup was normalized to the fp8 table.

Byte-granular comm shims (``query_p2p`` / ``query_mem_op`` and the AFD comm
ops) express arbitrary byte counts as ``ceil(bytes/2)`` bfloat16 elements —
at most 1 byte of rounding on multi-MB messages — so those cases carry a
1e-6 relative tolerance instead of the default 1e-9.

``source`` tags are recorded for information but not asserted (provenance
parity has its own coverage); latency and energy are asserted.

Regenerate (ONLY meaningful on a checkout that still has the Python math):

    uv run --no-sync python tests/cross_package/test_query_shim_baseline.py --regen
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import pytest

from aiconfigurator_core.sdk import common

pytestmark = pytest.mark.unit

BASELINE_PATH = Path(__file__).with_name("query_shim_baseline.json")

_DEFAULT_REL_TOL = 1e-9
_DEFAULT_ABS_TOL = 1e-12

# ----------------------------------------------------------------------------- #
# Case table. db aliases resolve through _DBS; versions are resolved at --regen
# time and PINNED into the JSON (the test skips a case whose exact pinned
# database is not shippable in this checkout).
# ----------------------------------------------------------------------------- #

_DBS = {
    "h200": ("h200_sxm", "trtllm"),
    "b200": ("b200_sxm", "trtllm"),
    "gb200": ("gb200", "trtllm"),
    "h200-sglang": ("h200_sxm", "sglang"),
    "b200-sglang": ("b200_sxm", "sglang"),
}

_MODES = (None, "SOL", "SOL_FULL")


def facade_case(case_id, db, method, tol=None, expected_from="legacy", version=None, expect_error=None, **kwargs):
    """Facade case: ``database.<method>(**kwargs)``. ``expect_error`` marks a
    retired entry (tombstone / conversion refusal): the live surface must
    raise that exception type; the captured legacy value stays in the JSON as
    the record of what the entry used to return."""
    return {
        "id": case_id,
        "db": db,
        "version": version,
        "surface": "facade",
        "method": method,
        "kwargs": kwargs,
        "tol": tol,
        "expected_from": expected_from,
        "expect_error": expect_error,
    }


def op_case(case_id, db, op, ctor_args, query, ctor_kwargs=None, tol=None, version=None, expect_error=None):
    """Op case: ``<op>(*ctor_args, **ctor_kwargs).query(database, **query)``.
    ``expect_error`` as in :func:`facade_case`."""
    return {
        "id": case_id,
        "db": db,
        "version": version,
        "surface": "op",
        "op": op,
        "ctor_args": ctor_args,
        "ctor_kwargs": ctor_kwargs or {},
        "query": query,
        "tol": tol,
        "expect_error": expect_error,
    }


def _facade_mode_sweep(base_id, db, method, tol=None, expected_from="legacy", **kwargs):
    return [
        facade_case(
            f"{base_id}-{mode or 'live'}",
            db,
            method,
            tol=tol,
            expected_from=expected_from,
            database_mode=mode,
            **kwargs,
        )
        for mode in _MODES
    ]


_GEN_QUERY = {"batch_size": 64, "beam_width": 1, "s": 4096, "prefix": 0, "x": 64}
_CTX_QUERY = {"batch_size": 1, "beam_width": 1, "s": 2048, "prefix": 0, "x": 2048}

CASES = [
    # ---- gemm ---------------------------------------------------------------
    *_facade_mode_sweep("gemm-bf16", "h200", "query_gemm", m=256, n=4096, k=4096, quant_mode="GEMMQuantMode.bfloat16"),
    facade_case("gemm-fp8-silicon", "h200", "query_gemm", m=7, n=4096, k=4096, quant_mode="GEMMQuantMode.fp8"),
    facade_case(
        "gemm-bf16-below-grid",
        "h200",
        "query_gemm",
        m=1,
        n=1024,
        k=1024,
        quant_mode="GEMMQuantMode.bfloat16",
        below_grid_sol=True,
    ),
    facade_case(
        "gemm-fp8static-silicon",
        "b200",
        "query_gemm",
        expected_from="engine",
        m=256,
        n=4096,
        k=4096,
        quant_mode="GEMMQuantMode.fp8_static",
    ),
    facade_case(
        "gemm-fp8static-solfull",
        "b200",
        "query_gemm",
        m=256,
        n=4096,
        k=4096,
        quant_mode="GEMMQuantMode.fp8_static",
        database_mode="SOL_FULL",
    ),
    op_case(
        "gemm-op-scaled",
        "h200",
        "gemm.GEMM",
        ["gemm", 2.0, 4096, 4096, "GEMMQuantMode.bfloat16"],
        {"x": 256},
    ),
    # ---- attention ----------------------------------------------------------
    *_facade_mode_sweep(
        "ctx-attn",
        "h200",
        "query_context_attention",
        expected_from="engine",
        b=1,
        s=2048,
        prefix=0,
        n=32,
        n_kv=8,
        kvcache_quant_mode="KVCacheQuantMode.bfloat16",
        fmha_quant_mode="FMHAQuantMode.bfloat16",
    ),
    facade_case(
        "ctx-attn-prefix",
        "h200",
        "query_context_attention",
        expected_from="engine",
        b=1,
        s=1434,
        prefix=614,
        n=32,
        n_kv=8,
        kvcache_quant_mode="KVCacheQuantMode.bfloat16",
        fmha_quant_mode="FMHAQuantMode.bfloat16",
    ),
    *_facade_mode_sweep(
        "gen-attn",
        "h200",
        "query_generation_attention",
        b=64,
        s=4096,
        n=32,
        n_kv=4,
        kvcache_quant_mode="KVCacheQuantMode.bfloat16",
    ),
    op_case(
        "ctx-attn-op",
        "h200",
        "attention.ContextAttention",
        ["context_attention", 1.0, 32, 8, "KVCacheQuantMode.bfloat16", "FMHAQuantMode.bfloat16"],
        {"batch_size": 1, "s": 1434, "prefix": 614},
    ),
    op_case(
        "gen-attn-op",
        "h200",
        "attention.GenerationAttention",
        ["generation_attention", 1.0, 32, 4, "KVCacheQuantMode.bfloat16"],
        dict(_GEN_QUERY),
    ),
    # ---- mla ----------------------------------------------------------------
    *_facade_mode_sweep(
        "ctx-mla",
        "h200",
        "query_context_mla",
        b=8,
        s=1434,
        prefix=614,
        num_heads=16,
        kvcache_quant_mode="KVCacheQuantMode.bfloat16",
        fmha_quant_mode="FMHAQuantMode.bfloat16",
    ),
    *_facade_mode_sweep(
        "gen-mla",
        "h200",
        "query_generation_mla",
        b=64,
        s=4096,
        num_heads=16,
        kvcache_quant_mode="KVCacheQuantMode.bfloat16",
    ),
    # ---- moe ----------------------------------------------------------------
    *_facade_mode_sweep(
        "moe-fp8block",
        "h200",
        "query_moe",
        num_tokens=512,
        hidden_size=7168,
        inter_size=2048,
        topk=8,
        num_experts=256,
        moe_tp_size=1,
        moe_ep_size=8,
        quant_mode="MoEQuantMode.fp8_block",
        workload_distribution="power_law_1.01",
    ),
    op_case(
        "moe-op-quant-override-missing",
        "h200",
        "moe.MoE",
        [
            "moe",
            1.0,
            7168,
            2048,
            8,
            256,
            1,
            8,
            "MoEQuantMode.fp8_block",
            "power_law_1.01",
        ],
        {"x": 512, "quant_mode": "MoEQuantMode.w4afp8"},
        ctor_kwargs={"attention_dp_size": 2},
    ),
    op_case(
        "moe-op-quant-override",
        "b200",
        "moe.MoE",
        [
            "moe",
            1.0,
            7168,
            2048,
            8,
            256,
            1,
            8,
            "MoEQuantMode.fp8_block",
            "power_law_1.01",
        ],
        {"x": 512, "quant_mode": "MoEQuantMode.nvfp4"},
        ctor_kwargs={"attention_dp_size": 2},
    ),
    # ---- comm ---------------------------------------------------------------
    *_facade_mode_sweep(
        "custom-ar",
        "h200",
        "query_custom_allreduce",
        quant_mode="CommQuantMode.half",
        tp_size=8,
        size=2**24,
    ),
    facade_case(
        "custom-ar-u32",
        "h200",
        "query_custom_allreduce",
        quant_mode="CommQuantMode.half",
        tp_size=8,
        size=2**32,
    ),
    *_facade_mode_sweep(
        "nccl-ag",
        "h200",
        "query_nccl",
        dtype="CommQuantMode.half",
        num_gpus=16,
        operation="all_gather",
        message_size=2**20,
    ),
    facade_case(
        "nccl-alltoall-odd",
        "h200",
        "query_nccl",
        dtype="CommQuantMode.half",
        num_gpus=8,
        operation="alltoall",
        message_size=1234567,
    ),
    *_facade_mode_sweep("p2p", "h200", "query_p2p", tol=1e-6, message_bytes=8388608),
    facade_case("p2p-odd", "h200", "query_p2p", tol=1e-6, message_bytes=1234567),
    *_facade_mode_sweep("mem-op", "h200", "query_mem_op", tol=1e-6, mem_bytes=4194304),
    facade_case("mem-op-odd", "h200", "query_mem_op", tol=1e-6, mem_bytes=1234567),
    op_case(
        "nccl-op",
        "h200",
        "communication.NCCL",
        ["nccl", 1.0, "all_reduce", 4096, 8, "CommQuantMode.half"],
        {"x": 64},
    ),
    op_case("p2p-op", "h200", "communication.P2P", ["p2p", 1.0, 4096, 2], {"x": 64}, tol=1e-6),
    op_case(
        "custom-ar-op",
        "h200",
        "communication.CustomAllReduce",
        ["custom_allreduce", 1.0, 4096, 8],
        {"x": 64},
    ),
    op_case("elementwise-op", "h200", "elementwise.ElementWise", ["norm", 1.0, 4096, 4096], {"x": 256}, tol=1e-6),
    op_case("embedding-op", "h200", "embedding.Embedding", ["embed", 1.0, 129280, 7168], {"x": 256}, tol=1e-6),
    # ---- dsa ----------------------------------------------------------------
    facade_case(
        "ctx-dsa-silicon",
        "b200",
        "query_context_dsa_module",
        b=1,
        s=4096,
        prefix=0,
        num_heads=128,
        kvcache_quant_mode="KVCacheQuantMode.bfloat16",
        fmha_quant_mode="FMHAQuantMode.bfloat16",
    ),
    facade_case(
        "gen-dsa-silicon",
        "b200",
        "query_generation_dsa_module",
        b=1,
        s=8192,
        num_heads=128,
        kv_cache_dtype="KVCacheQuantMode.bfloat16",
    ),
    # ---- composites ---------------------------------------------------------
    op_case(
        "afd-transfer-a2f",
        "h200",
        "afd_transfer.AFDTransfer",
        ["afd_a2f", 1.0],
        {"x": 512},
        ctor_kwargs={
            "direction": "a2f",
            "hidden_size": 7168,
            "n_a_workers": 4,
            "n_f_workers": 8,
            "gpus_per_node": 8,
            "num_experts": 256,
            "topk": 8,
        },
        tol=1e-6,
    ),
    op_case(
        "afd-f-allgather",
        "h200",
        "afd_transfer.AFDFAllGather",
        ["afd_ag", 1.0],
        {"x": 512},
        ctor_kwargs={
            "hidden_size": 7168,
            "n_a_workers": 4,
            "n_f_workers": 8,
            "gpus_per_node": 8,
            "num_experts": 256,
            "topk": 8,
        },
        tol=1e-6,
    ),
    op_case(
        "afd-f-reducescatter",
        "h200",
        "afd_transfer.AFDFReduceScatter",
        ["afd_rs", 1.0],
        {"x": 512},
        ctor_kwargs={
            "hidden_size": 7168,
            "n_a_workers": 4,
            "n_f_workers": 8,
            "gpus_per_node": 8,
            "num_experts": 256,
            "topk": 8,
        },
        tol=1e-6,
    ),
    op_case(
        "afd-combine",
        "h200",
        "afd_transfer.AFDCombine",
        ["afd_combine", 1.0],
        {"x": 512},
        ctor_kwargs={"hidden_size": 7168, "tp_a": 2, "f_moe_ep_size": 8},
        tol=1e-6,
    ),
    # ---- per-call mode overrides ---------------------------------------------
    facade_case(
        "gemm-bf16-explicit-silicon",
        "h200",
        "query_gemm",
        m=256,
        n=4096,
        k=4096,
        quant_mode="GEMMQuantMode.bfloat16",
        database_mode="SILICON",
    ),
    facade_case(
        "gemm-bf16-empirical",
        "h200",
        "query_gemm",
        m=300,
        n=4096,
        k=4096,
        quant_mode="GEMMQuantMode.bfloat16",
        database_mode="EMPIRICAL",
    ),
    facade_case(
        "gen-attn-hybrid-offgrid",
        "h200",
        "query_generation_attention",
        b=64,
        s=5000,
        n=32,
        n_kv=4,
        kvcache_quant_mode="KVCacheQuantMode.bfloat16",
        database_mode="HYBRID",
    ),
    # ---- large-EP / wideep (mined from unit tests by family) -----------------
    op_case(
        "moe-a2a-op",
        "h200-sglang",
        "moe_comm.MoEAllToAll",
        ["tp_noop_dispatch", 1.0],
        {"x": 1},
        ctor_kwargs={
            "phase": "dispatch",
            "comm_backend": "deepep_ht",
            "hidden_size": 7168,
            "topk": 8,
            "num_experts": 256,
            "moe_ep_size": 8,
            "node_num": 1,
            "sms": 20,
        },
        version="0.5.6.post2",
    ),
    op_case(
        "moe-expert-compute-op",
        "h200-sglang",
        "moe_comm.MoEExpertCompute",
        ["review_probe", 1.0],
        {"x": 32},
        ctor_kwargs={
            "hidden_size": 7168,
            "inter_size": 2048,
            "topk": 8,
            "num_experts": 256,
            "moe_ep_size": 2,
            "quant_mode": "MoEQuantMode.fp8_block",
            "workload_distribution": "uniform",
            "attention_dp_size": 1,
            "inference_phase": "context",
        },
        version="0.5.6.post2",
    ),
    op_case(
        "wideep-moe-op",
        "gb200",
        "moe.TrtLLMWideEPMoE",
        ["test_wideep_moe", 1.0, 7168, 2048, 8, 256, 1, 2, "MoEQuantMode.nvfp4", "power_law_1.01_eplb", 1],
        {"x": 1},
        ctor_kwargs={"num_slots": 256},
        version="1.3.0rc10",
    ),
    op_case(
        "wideep-dispatch-op",
        "gb200",
        "moe.TrtLLMWideEPMoEDispatch",
        ["test_dispatch", 1.0, 7168, 8, 256, 1, 4, 1, True, "MoEQuantMode.bfloat16"],
        {"x": 1},
        ctor_kwargs={"node_num": 1},
        version="1.3.0rc10",
        # Tombstone: the legacy op folded raw per-phase alltoall rows.
        expect_error="NotImplementedError",
    ),
    op_case(
        "moe-dispatch-op",
        "h200-sglang",
        "moe.MoEDispatch",
        ["test_dispatch", 1, 7168, 8, 256, 1, 32, 32, True],
        {"x": 96},
        ctor_kwargs={
            "quant_mode": "MoEQuantMode.fp8_block",
            "moe_backend": "deepep_moe",
            "is_context": True,
            "sms": 20,
        },
        version="0.5.6.post2",
        # deepep_moe dispatch has no native variant (AIC-1601): conversion
        # refuses LOUDLY (the pre-PR-5 converter silently serialized it as
        # CustomAllReduce — the algorithm swap its own docstring forbids).
        expect_error="OpConversionError",
    ),
    facade_case(
        "moe-a2a-facade",
        "h200-sglang",
        "query_moe_a2a",
        version="0.5.6.post2",
        comm_backend="deepep_ht",
        phase="dispatch",
        comm_dtype="default",
        ep_size=8,
        node_num=1,
        hidden_size=7168,
        topk=8,
        num_experts=256,
        num_tokens=1,
        sms=20,
    ),
    facade_case(
        "moe-expert-compute-facade",
        "h200-sglang",
        "query_moe_expert_compute",
        version="0.5.6.post2",
        kernel_source="deepep_moe",
        quant_mode="MoEQuantMode.fp8_block",
        workload_distribution="uniform",
        inference_phase="context",
        topk=8,
        num_experts=256,
        num_slots=256,
        hidden_size=7168,
        inter_size=2048,
        moe_tp_size=1,
        moe_ep_size=2,
        num_tokens=32,
    ),
    facade_case(
        "wideep-deepep-ll-facade",
        "h200-sglang",
        "query_wideep_deepep_ll",
        version="0.5.6.post2",
        expect_error="NotImplementedError",
        node_num=4,
        num_tokens=8,
        num_experts=256,
        topk=8,
        hidden_size=7168,
    ),
    facade_case(
        "wideep-deepep-normal-facade",
        "h200-sglang",
        "query_wideep_deepep_normal",
        version="0.5.6.post2",
        expect_error="NotImplementedError",
        node_num=4,
        num_tokens=96,
        num_experts=256,
        topk=8,
        hidden_size=7168,
        sms=20,
    ),
    facade_case(
        "wideep-moe-compute-facade",
        "gb200",
        "query_wideep_moe_compute",
        version="1.3.0rc10",
        num_tokens=1,
        hidden_size=7168,
        inter_size=2048,
        topk=8,
        num_experts=256,
        num_slots=256,
        moe_tp_size=1,
        moe_ep_size=2,
        quant_mode="MoEQuantMode.nvfp4",
        workload_distribution="power_law_1.01_eplb",
    ),
    facade_case(
        "trtllm-alltoall-facade",
        "gb200",
        "query_trtllm_alltoall",
        version="1.3.0rc10",
        expect_error="NotImplementedError",
        op_name="alltoall_dispatch",
        num_tokens=1,
        hidden_size=7168,
        topk=8,
        num_experts=256,
        moe_ep_size=4,
        quant_mode="MoEQuantMode.bfloat16",
        node_num=1,
        moe_backend="wideep",
    ),
    # ---- mla modules / bmm / wideep mla ---------------------------------------
    op_case(
        "mla-module-ctx-op",
        "h200",
        "mla.MLAModule",
        ["test_ctx", 1.0, True, 16, "KVCacheQuantMode.fp8", "FMHAQuantMode.bfloat16", "GEMMQuantMode.fp8_block"],
        {"batch_size": 4, "s": 4000, "prefix": 0},
        version="1.3.0rc10",
    ),
    op_case(
        "mla-module-gen-op",
        "h200",
        "mla.MLAModule",
        ["test_gen", 1.0, False, 16, "KVCacheQuantMode.fp8", "FMHAQuantMode.bfloat16", "GEMMQuantMode.fp8_block"],
        {"batch_size": 64, "s": 4096, "beam_width": 1},
        version="1.3.0rc10",
    ),
    op_case(
        "mla-bmm-op",
        "h200",
        "mla.MLABmm",
        ["mla_bmm_pre", 1.0, 128, "GEMMQuantMode.bfloat16", True],
        {"batch_size": 32, "beam_width": 1},
        version="1.3.0rc10",
    ),
    op_case(
        "wideep-ctx-mla-op",
        "h200-sglang",
        "mla.WideEPContextMLA",
        ["context_attention", 1.0, 1, "KVCacheQuantMode.fp8", "FMHAQuantMode.fp8_block"],
        {"batch_size": 1, "s": 1024, "prefix": 0},
        ctor_kwargs={"attn_backend": "flashinfer"},
        version="0.5.6.post2",
    ),
    op_case(
        "wideep-gen-mla-op",
        "h200-sglang",
        "mla.WideEPGenerationMLA",
        ["generation_attention", 1.0, 1, "KVCacheQuantMode.fp8", "FMHAQuantMode.fp8_block"],
        {"batch_size": 1, "s": 1024},
        ctor_kwargs={"attn_backend": "flashinfer"},
        version="0.5.6.post2",
    ),
    facade_case(
        "ctx-mla-module-facade",
        "h200",
        "query_context_mla_module",
        version="1.3.0rc10",
        b=4,
        s=4000,
        prefix=0,
        num_heads=16,
        kvcache_quant_mode="KVCacheQuantMode.fp8",
        fmha_quant_mode="FMHAQuantMode.bfloat16",
        gemm_quant_mode="GEMMQuantMode.fp8_block",
    ),
    facade_case(
        "gen-mla-module-facade",
        "h200",
        "query_generation_mla_module",
        version="1.3.0rc10",
        b=64,
        s=4096,
        num_heads=16,
        kv_cache_dtype="KVCacheQuantMode.fp8",
        gemm_quant_mode="GEMMQuantMode.fp8_block",
    ),
    facade_case(
        "mla-bmm-facade",
        "h200",
        "query_mla_bmm",
        version="1.3.0rc10",
        num_tokens=32,
        num_heads=128,
        quant_mode="GEMMQuantMode.bfloat16",
        if_pre=True,
    ),
    facade_case(
        "wideep-ctx-mla-facade",
        "h200-sglang",
        "query_wideep_context_mla",
        version="0.5.6.post2",
        b=1,
        s=1024,
        prefix=0,
        tp_size=1,
        kvcache_quant_mode="KVCacheQuantMode.fp8",
        fmha_quant_mode="FMHAQuantMode.fp8_block",
        attention_backend="flashinfer",
    ),
    facade_case(
        "wideep-gen-mla-facade",
        "h200-sglang",
        "query_wideep_generation_mla",
        version="0.5.6.post2",
        b=1,
        s=1024,
        tp_size=1,
        kvcache_quant_mode="KVCacheQuantMode.fp8",
        fmha_quant_mode="FMHAQuantMode.fp8_block",
        attention_backend="flashinfer",
    ),
    # ---- encoder attention -----------------------------------------------------
    op_case(
        "encoder-attn-op",
        "h200",
        "attention.EncoderAttention",
        ["encoder_attention"],
        {"batch_size": 2, "s": 64},
        ctor_kwargs={
            "scale_factor": 1.0,
            "num_heads": 16,
            "head_size": 72,
            "fmha_quant_mode": "FMHAQuantMode.bfloat16",
            "partial_rotary_factor": 0.0,
        },
        version="1.3.0rc10",
    ),
    facade_case(
        "encoder-attn-facade",
        "h200",
        "query_encoder_attention",
        version="1.3.0rc10",
        b=2,
        s=64,
        n=16,
        head_size=72,
        fmha_quant_mode="FMHAQuantMode.bfloat16",
    ),
    # ---- msa (no silicon table: live SILICON pins the typed error) -------------
    op_case(
        "msa-ctx-op",
        "b200-sglang",
        "msa.ContextMSAModule",
        ["msa", 1.0],
        {"batch_size": 8, "s": 512, "prefix": 0},
        ctor_kwargs={
            "num_heads": 8,
            "num_kv_heads": 1,
            "hidden_size": 4096,
            "head_dim": 128,
            "v_head_dim": 128,
            "index_n_heads": 4,
            "index_head_dim": 128,
            "index_topk": 2048,
            "block_size": 128,
            "kvcache_quant_mode": "KVCacheQuantMode.fp8",
            "fmha_quant_mode": "FMHAQuantMode.fp8",
            "gemm_quant_mode": "GEMMQuantMode.fp8_block",
        },
        version="0.5.14",
    ),
    op_case(
        "msa-gen-op",
        "b200-sglang",
        "msa.GenerationMSAModule",
        ["msa", 1.0],
        {"batch_size": 8, "s": 2048},
        ctor_kwargs={
            "num_heads": 8,
            "num_kv_heads": 1,
            "hidden_size": 4096,
            "head_dim": 128,
            "v_head_dim": 128,
            "index_n_heads": 4,
            "index_head_dim": 128,
            "index_topk": 2048,
            "block_size": 128,
            "kvcache_quant_mode": "KVCacheQuantMode.fp8",
            "fmha_quant_mode": "FMHAQuantMode.fp8",
            "gemm_quant_mode": "GEMMQuantMode.fp8_block",
        },
        version="0.5.14",
    ),
    # ---- mamba / gdn / kda ------------------------------------------------------
    op_case(
        "mamba2-kernel-op",
        "h200",
        "mamba.Mamba2Kernel",
        ["context_mamba_conv1d", 1.0, "causal_conv1d_fn", "context"],
        {"batch_size": 1, "s": 4096},
        ctor_kwargs={
            "hidden_size": 4096,
            "nheads": 128,
            "head_dim": 64,
            "d_state": 128,
            "d_conv": 4,
            "n_groups": 8,
            "chunk_size": 128,
        },
        version="1.3.0rc10",
    ),
    op_case(
        "gdn-kernel-op",
        "b200-sglang",
        "mamba.GDNKernel",
        ["context_gdn_conv1d", 1.0, "causal_conv1d_fn", "context", 2048, 16, 128, 32, 128, 4],
        {"batch_size": 32, "s": 4096},
        version="0.5.14",
    ),
    op_case(
        "kda-kernel-op",
        "h200-sglang",
        "mamba.KDAKernel",
        ["context_kda_scan", 1.0, "chunk_kda", "context", 7168, 24, 128, 24, 128, 4],
        {"batch_size": 1, "s": 4096},
        version="0.5.16",
    ),
    op_case(
        "mamba2-composite-op",
        "h200",
        "mamba.Mamba2",
        ["mamba2", 1.0],
        {"x": 512},
        ctor_kwargs={
            "hidden_size": 4096,
            "nheads": 128,
            "head_dim": 64,
            "d_state": 128,
            "d_conv": 4,
            "n_groups": 8,
            "chunk_size": 128,
            "tp_size": 1,
            "quant_mode": "GEMMQuantMode.bfloat16",
        },
        tol=1e-6,
        version="1.3.0rc10",
    ),
    facade_case(
        "mamba2-facade",
        "h200",
        "query_mamba2",
        version="1.3.0rc10",
        phase="context",
        kernel_source="causal_conv1d_fn",
        batch_size=1,
        seq_len=4096,
        d_model=4096,
        d_state=128,
        d_conv=4,
        nheads=128,
        head_dim=64,
        n_groups=8,
        chunk_size=128,
    ),
    facade_case(
        "gdn-facade",
        "b200-sglang",
        "query_gdn",
        version="0.5.14",
        phase="context",
        kernel_source="causal_conv1d_fn",
        batch_size=32,
        seq_len=4096,
        d_model=2048,
        num_k_heads=16,
        head_k_dim=128,
        num_v_heads=32,
        head_v_dim=128,
        d_conv=4,
    ),
    # ---- dsv4 / mhc -------------------------------------------------------------
    op_case(
        "mhc-op",
        "b200-sglang",
        "dsv4.DeepSeekV4MHCModule",
        ["mhc", 1, "pre", 7168, 4, 20, "GEMMQuantMode.bfloat16"],
        {"x": 512},
        version="0.5.14",
    ),
    op_case(
        "dsv4-ctx-op",
        "b200-sglang",
        "dsv4.ContextDeepSeekV4AttentionModule",
        [
            "context_attention",
            1.0,
            16,
            128,
            8,
            7168,
            1536,
            1024,
            512,
            64,
            64,
            128,
            1024,
            128,
            4,
            2,
            "KVCacheQuantMode.fp8",
            "FMHAQuantMode.bfloat16",
            "GEMMQuantMode.fp8_block",
        ],
        {"batch_size": 2, "s": 256, "prefix": 0},
        version="0.5.14",
    ),
    op_case(
        "dsv4-gen-op",
        "b200-sglang",
        "dsv4.GenerationDeepSeekV4AttentionModule",
        [
            "generation_attention",
            1.0,
            16,
            128,
            8,
            7168,
            1536,
            1024,
            512,
            64,
            64,
            128,
            1024,
            128,
            4,
            2,
            "KVCacheQuantMode.fp8",
            "FMHAQuantMode.bfloat16",
            "GEMMQuantMode.fp8_block",
        ],
        {"batch_size": 2, "s": 4096, "beam_width": 1},
        version="0.5.14",
    ),
    op_case(
        "dsv4-megamoe-op",
        "b200-sglang",
        "dsv4.DeepSeekV4MegaMoEModule",
        ["context_megamoe", 1.0, 3584, 3072, 16, 896, 1, 8, "MoEQuantMode.w4a8_mxfp4_mxfp8", "power_law_1.01"],
        {"x": 1024},
        ctor_kwargs={"is_context": True, "num_fused_shared_experts": 0},
        version="0.5.16",
    ),
    facade_case(
        "mhc-facade",
        "b200-sglang",
        "query_mhc_module",
        version="0.5.14",
        num_tokens=512,
        hidden_size=7168,
        hc_mult=4,
        sinkhorn_iters=20,
        op="pre",
        quant_mode="GEMMQuantMode.bfloat16",
    ),
    facade_case(
        "dsv4-ctx-facade",
        "b200-sglang",
        "query_context_deepseek_v4_attention_module",
        version="0.5.14",
        b=2,
        s=256,
        num_heads=16,
        native_heads=128,
        tp_size=8,
        hidden_size=7168,
        q_lora_rank=1536,
        o_lora_rank=1024,
        head_dim=512,
        rope_head_dim=64,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=1024,
        window_size=128,
        compress_ratio=4,
        o_groups=2,
        kvcache_quant_mode="KVCacheQuantMode.fp8",
        fmha_quant_mode="FMHAQuantMode.bfloat16",
        gemm_quant_mode="GEMMQuantMode.fp8_block",
        prefix=0,
    ),
    facade_case(
        "dsv4-gen-facade",
        "b200-sglang",
        "query_generation_deepseek_v4_attention_module",
        version="0.5.14",
        b=2,
        s=4096,
        num_heads=16,
        native_heads=128,
        tp_size=8,
        hidden_size=7168,
        q_lora_rank=1536,
        o_lora_rank=1024,
        head_dim=512,
        rope_head_dim=64,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=1024,
        window_size=128,
        compress_ratio=4,
        o_groups=2,
        kvcache_quant_mode="KVCacheQuantMode.fp8",
        fmha_quant_mode="FMHAQuantMode.bfloat16",
        gemm_quant_mode="GEMMQuantMode.fp8_block",
    ),
    facade_case(
        "dsv4-megamoe-facade",
        "b200-sglang",
        "query_dsv4_megamoe_module",
        version="0.5.16",
        num_tokens=1024,
        hidden_size=3584,
        inter_size=3072,
        topk=16,
        num_experts=896,
        moe_tp_size=1,
        moe_ep_size=8,
        quant_mode="MoEQuantMode.w4a8_mxfp4_mxfp8",
        workload_distribution="power_law_1.01",
        is_context=True,
        num_fused_shared_experts=0,
    ),  # ---- review round 2 (#1552 Arsene12358): coverage-gap dispositions ---------
    op_case(
        "overlap-token-only-op",
        "h200",
        "overlap.OverlapOp",
        [
            "ov",
            [{"__op__": "gemm.GEMM", "ctor_args": ["g1", 1.0, 4096, 4096, "GEMMQuantMode.bfloat16"]}],
            [{"__op__": "gemm.GEMM", "ctor_args": ["g2", 1.0, 4096, 4096, "GEMMQuantMode.bfloat16"]}],
        ],
        {"x": 64, "batch_size": 64, "s": 4096, "beam_width": 1},
        version="1.3.0rc20",
    ),
    op_case(
        "fallback-token-only-op",
        "h200",
        "overlap.FallbackOp",
        [
            "fb",
            {"__op__": "gemm.GEMM", "ctor_args": ["g1", 1.0, 4096, 4096, "GEMMQuantMode.bfloat16"]},
            [{"__op__": "gemm.GEMM", "ctor_args": ["g2", 1.0, 4096, 4096, "GEMMQuantMode.bfloat16"]}],
        ],
        {"x": 64, "batch_size": 64, "s": 4096, "beam_width": 1},
        version="1.3.0rc20",
    ),
    op_case(
        "overlap-mixed-op",
        "h200",
        "overlap.OverlapOp",
        [
            "ov_mixed",
            [{"__op__": "gemm.GEMM", "ctor_args": ["g1", 1.0, 4096, 4096, "GEMMQuantMode.bfloat16"]}],
            [
                {
                    "__op__": "attention.GenerationAttention",
                    "ctor_args": ["ga", 1.0, 32, 4, "KVCacheQuantMode.bfloat16"],
                }
            ],
        ],
        {"x": 64, "batch_size": 64, "s": 4096, "beam_width": 1},
        version="1.3.0rc20",
    ),
    op_case(
        "gemm-op-quant-override",
        "h200",
        "gemm.GEMM",
        ["g", 1.0, 4096, 4096, "GEMMQuantMode.bfloat16"],
        {"x": 64, "quant_mode": "GEMMQuantMode.fp8"},
        version="1.3.0rc20",
    ),
    op_case(
        "megamoe-op-quant-override-miss",
        "b200-sglang",
        "dsv4.DeepSeekV4MegaMoEModule",
        ["m", 1.0, 3584, 3072, 16, 896, 1, 8, "MoEQuantMode.w4a8_mxfp4_mxfp8", "power_law_1.01"],
        {"x": 1024, "quant_mode": "MoEQuantMode.w4a16_mxfp4"},
        ctor_kwargs={"is_context": True, "num_fused_shared_experts": 0},
        version="0.5.16",
    ),
    op_case(
        "ctx-mla-op",
        "h200",
        "mla.ContextMLA",
        ["context_mla", 1.0, 16, "KVCacheQuantMode.bfloat16", "FMHAQuantMode.bfloat16"],
        {"batch_size": 8, "s": 1434, "prefix": 614},
        version="1.3.0rc20",
    ),
    op_case(
        "gen-mla-op",
        "h200",
        "mla.GenerationMLA",
        ["generation_mla", 1.0, 16, "KVCacheQuantMode.bfloat16"],
        {"batch_size": 64, "s": 4096, "beam_width": 1},
        version="1.3.0rc20",
    ),
    op_case(
        "ctx-dsa-op",
        "b200",
        "dsa.ContextDSAModule",
        [
            "context_dsa",
            1.0,
            128,
            "KVCacheQuantMode.bfloat16",
            "FMHAQuantMode.bfloat16",
            "GEMMQuantMode.bfloat16",
        ],
        {"batch_size": 1, "s": 4096, "prefix": 0},
    ),
    op_case(
        "gen-dsa-op",
        "b200",
        "dsa.GenerationDSAModule",
        ["generation_dsa", 1.0, 128, "KVCacheQuantMode.bfloat16", "GEMMQuantMode.bfloat16"],
        {"batch_size": 1, "s": 8192, "beam_width": 1},
    ),
    op_case(
        "kda-verify-op",
        "h200-sglang",
        "mamba.KDAKernel",
        ["context_kda_verify", 1.0, "chunk_kda", "verify", 7168, 24, 128, 24, 128, 4],
        {"batch_size": 1, "s": 4096},
        ctor_kwargs={"draft_tokens": 3},
        version="0.5.16",
    ),
    op_case(
        "gen-attn-zero-scale-op",
        "h200",
        "attention.GenerationAttention",
        ["generation_attention", 1.0, 32, 4, "KVCacheQuantMode.bfloat16"],
        {"batch_size": 64, "s": 4096, "beam_width": 1, "gen_seq_imbalance_correction_scale": 0.0},
        version="1.3.0rc20",
    ),
    facade_case(
        "compute-scale-tombstone",
        "h200",
        "query_compute_scale",
        expect_error="NotImplementedError",
        m=256,
        k=4096,
        quant_mode="GEMMQuantMode.fp8_static",
    ),
    facade_case(
        "scale-matrix-tombstone",
        "h200",
        "query_scale_matrix",
        expect_error="NotImplementedError",
        m=256,
        k=4096,
        quant_mode="GEMMQuantMode.fp8_static",
    ),
]


# ----------------------------------------------------------------------------- #
# Case execution.
# ----------------------------------------------------------------------------- #


def _resolve_value(value):
    """Decode ``"<EnumClass>.<member>"`` strings to enum values; leave the rest."""
    if isinstance(value, str) and "." in value:
        cls_name, _, member = value.partition(".")
        cls = getattr(common, cls_name, None)
        if cls is not None and hasattr(cls, member):
            return getattr(cls, member)
    if isinstance(value, str) and value in ("SOL", "SOL_FULL", "SILICON", "HYBRID", "EMPIRICAL"):
        return getattr(common.DatabaseMode, value)
    return value


def _resolve_kwargs(kwargs):
    return {key: _resolve_value(value) for key, value in kwargs.items()}


def _op_class(dotted):
    import importlib

    module_name, _, class_name = dotted.rpartition(".")
    module = importlib.import_module(f"aiconfigurator_core.sdk.operations.{module_name}")
    return getattr(module, class_name)


def _resolve_ctor_value(value):
    """Ctor args may nest child op specs (composites): {"__op__": "gemm.GEMM",
    "ctor_args": [...]} or lists thereof; everything else goes through the
    enum decoder."""
    if isinstance(value, dict) and "__op__" in value:
        cls = _op_class(value["__op__"])
        args = [_resolve_ctor_value(a) for a in value.get("ctor_args", [])]
        kwargs = {k: _resolve_ctor_value(v) for k, v in value.get("ctor_kwargs", {}).items()}
        return cls(*args, **kwargs)
    if isinstance(value, list):
        return [_resolve_ctor_value(v) for v in value]
    return _resolve_value(value)


def _build_op(case):
    cls = _op_class(case["op"])
    args = [_resolve_ctor_value(a) for a in case["ctor_args"]]
    kwargs = {k: _resolve_ctor_value(v) for k, v in case["ctor_kwargs"].items()}
    return cls(*args, **kwargs)


def _encode_result(result):
    if isinstance(result, tuple):
        return {"triple": [float(v) for v in result]}
    return {
        "latency": float(result),
        "energy": float(getattr(result, "energy", 0.0)),
        "source": getattr(result, "source", None),
    }


def _run_case(case, database):
    """Execute one case against the live surface, encoding value or error."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            if case["surface"] == "facade":
                result = getattr(database, case["method"])(**_resolve_kwargs(case["kwargs"]))
            else:
                op = _build_op(case)
                result = op.query(database, **_resolve_kwargs(case["query"]))
    except Exception as exc:
        return {"error": type(exc).__name__}
    return _encode_result(result)


def _load_database(system, backend, version):
    from aiconfigurator_core.sdk.perf_database import get_database

    try:
        return get_database(system=system, backend=backend, version=version)
    except Exception:
        return None


def _load_baseline():
    if not BASELINE_PATH.exists():
        pytest.skip("query_shim_baseline.json not generated")
    return json.loads(BASELINE_PATH.read_text())


def _assert_close(expected, got, rel_tol):
    assert math.isclose(expected, got, rel_tol=rel_tol, abs_tol=_DEFAULT_ABS_TOL), (expected, got)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_query_shim_matches_pre_retirement_baseline(case):
    baseline = _load_baseline()
    pinned = baseline["cases"].get(case["id"])
    assert pinned is not None, f"case {case['id']} missing from baseline JSON — regenerate before deleting math"
    system, backend, version = pinned["db"]
    database = _load_database(system, backend, version)
    if database is None:
        pytest.skip(f"pinned database {system}/{backend}/{version} unavailable")
    got = _run_case(case, database)
    if case.get("expect_error"):
        # Retired entry: the pinned JSON keeps the legacy value for the
        # record; the live surface must raise the declared type instead.
        assert got.get("error") == case["expect_error"], (case["expect_error"], got)
        return
    expected = pinned["value"]
    if "error" in expected:
        assert got.get("error") == expected["error"], (expected, got)
        return
    assert "error" not in got, (expected, got)
    rel_tol = case.get("tol") or _DEFAULT_REL_TOL
    if "triple" in expected:
        assert "triple" in got, (expected, got)
        for exp_v, got_v in zip(expected["triple"], got["triple"], strict=True):
            _assert_close(exp_v, got_v, rel_tol)
        return
    _assert_close(expected["latency"], got["latency"], rel_tol)
    if expected.get("energy") is not None:
        _assert_close(expected["energy"], got["energy"], rel_tol)


def test_baseline_covers_every_case():
    baseline = _load_baseline()
    missing = {case["id"] for case in CASES} - set(baseline["cases"])
    assert not missing, f"cases missing from baseline JSON (regen required): {sorted(missing)}"


# ----------------------------------------------------------------------------- #
# --regen: capture the baseline from the LIVE surface. Run this only on a
# checkout whose Python per-call math is still present (plus the compiled
# engine for the two expected_from="engine" families).
# ----------------------------------------------------------------------------- #


def _engine_expected(case, database):
    """Expected value for expected_from="engine" cases, sourced from the
    compiled engine via the sanity-check reference (PR-4's alignment surface).
    ``EngineReference`` exposes latency/SOL-triples only, so these entries pin
    energy as ``None`` (not asserted); latency/triple are the semantic anchor."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "sanity_check"))
    from engine_reference import EngineReference

    reference = EngineReference(database)
    kwargs = _resolve_kwargs(case["kwargs"])
    mode = kwargs.pop("database_mode", None) or common.DatabaseMode.SILICON
    method = getattr(reference, case["method"])
    try:
        if mode == common.DatabaseMode.SOL:
            triple = method(**kwargs, database_mode=common.DatabaseMode.SOL_FULL)
            return {"latency": float(triple[0]), "energy": None, "source": "sol"}
        result = method(**kwargs, database_mode=mode)
    except Exception as exc:
        return {"error": type(exc).__name__}
    if isinstance(result, tuple):
        return {"triple": [float(v) for v in result]}
    return {"latency": float(result), "energy": None, "source": "engine"}


def _regen():
    from aiconfigurator_core.sdk.perf_database import get_latest_database_version

    entries = {}
    for case in CASES:
        system, backend = _DBS[case["db"]]
        version = case.get("version") or get_latest_database_version(system, backend)
        database = _load_database(system, backend, version)
        if database is None:
            raise RuntimeError(f"regen requires database {system}/{backend} (case {case['id']})")
        if case.get("expected_from") == "engine":
            value = _engine_expected(case, database)
        else:
            value = _run_case(case, database)
        entries[case["id"]] = {"db": [system, backend, version], "value": value}
        print(f"{case['id']}: {value}")
    payload = {
        "_comment": (
            "Pre-retirement per-call query baseline (#1357 PR-5). Captured from the Python "
            "per-call math (expected_from=engine cases: from the compiled engine) immediately "
            "before that math was deleted; see test_query_shim_baseline.py."
        ),
        "cases": entries,
    }
    BASELINE_PATH.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"\nwrote {len(entries)} cases -> {BASELINE_PATH}")


if __name__ == "__main__":
    import sys

    if "--regen" in sys.argv:
        _regen()
    else:
        print(__doc__)


# ----------------------------------------------------------------------------- #
# Frozen legacy-surface manifest (review #1552: coverage must be derived from
# an independent inventory, not from the hand-written CASES themselves).
# ----------------------------------------------------------------------------- #

# Every PerfDatabase.query_* method the retired stack exposed (== the shim set
# frozen by test_single_oracle_contract.PERF_DATABASE_QUERY_SHIMS).
LEGACY_FACADE_SURFACE = frozenset(
    {
        "query_gemm",
        "query_compute_scale",
        "query_scale_matrix",
        "query_context_attention",
        "query_encoder_attention",
        "query_generation_attention",
        "query_context_mla",
        "query_generation_mla",
        "query_context_mla_module",
        "query_generation_mla_module",
        "query_wideep_generation_mla",
        "query_wideep_context_mla",
        "query_custom_allreduce",
        "query_nccl",
        "query_moe",
        "query_mla_bmm",
        "query_mem_op",
        "query_mamba2",
        "query_gdn",
        "query_p2p",
        "query_wideep_deepep_ll",
        "query_wideep_deepep_normal",
        "query_wideep_moe_compute",
        "query_trtllm_alltoall",
        "query_moe_a2a",
        "query_moe_expert_compute",
        "query_context_dsa_module",
        "query_generation_dsa_module",
        "query_mhc_module",
        "query_context_deepseek_v4_attention_module",
        "query_generation_deepseek_v4_attention_module",
        "query_dsv4_megamoe_module",
    }
)

# Every Operation subclass whose retired query() had semantics of its own.
# FPMForwardOp is deliberately absent: it never defined query() pre-retirement
# (base raised NotImplementedError), so there is no legacy behavior to pin —
# its phase mapping is covered by the base-plan tests in test_base_queries.py.
LEGACY_OP_QUERY_SURFACE = frozenset(
    {
        "gemm.GEMM",
        "embedding.Embedding",
        "elementwise.ElementWise",
        "attention.ContextAttention",
        "attention.GenerationAttention",
        "attention.EncoderAttention",
        "mla.ContextMLA",
        "mla.GenerationMLA",
        "mla.MLAModule",
        "mla.MLABmm",
        "mla.WideEPContextMLA",
        "mla.WideEPGenerationMLA",
        "dsa.ContextDSAModule",
        "dsa.GenerationDSAModule",
        "msa.ContextMSAModule",
        "msa.GenerationMSAModule",
        "dsv4.DeepSeekV4MHCModule",
        "dsv4.ContextDeepSeekV4AttentionModule",
        "dsv4.GenerationDeepSeekV4AttentionModule",
        "dsv4.DeepSeekV4MegaMoEModule",
        "mamba.Mamba2Kernel",
        "mamba.GDNKernel",
        "mamba.KDAKernel",
        "mamba.Mamba2",
        "moe.MoE",
        "moe.MoEDispatch",
        "moe.TrtLLMWideEPMoE",
        "moe.TrtLLMWideEPMoEDispatch",
        "moe_comm.MoEAllToAll",
        "moe_comm.MoEExpertCompute",
        "communication.CustomAllReduce",
        "communication.NCCL",
        "communication.P2P",
        "overlap.OverlapOp",
        "overlap.FallbackOp",
        "afd_transfer.AFDTransfer",
        "afd_transfer.AFDFAllGather",
        "afd_transfer.AFDFReduceScatter",
        "afd_transfer.AFDCombine",
    }
)

# Semantic dimensions the sweep must exercise, each naming its case ids.
SEMANTIC_DIMENSION_CASES = {
    "per-call quant override honored": {"gemm-op-quant-override", "moe-op-quant-override"},
    "per-call quant override missing-data miss": {"moe-op-quant-override-missing", "megamoe-op-quant-override-miss"},
    "verify phase (speculative KDA)": {"kda-verify-op"},
    "explicit zero imbalance scale": {"gen-attn-zero-scale-op"},
    "token-only composites": {"overlap-token-only-op", "fallback-token-only-op"},
    "mixed-phase composite": {"overlap-mixed-op"},
    "tombstones assert their error": {
        "compute-scale-tombstone",
        "scale-matrix-tombstone",
        "wideep-deepep-ll-facade",
        "wideep-deepep-normal-facade",
        "trtllm-alltoall-facade",
        "wideep-dispatch-op",
    },
    "typed data-miss errors": {"msa-ctx-op", "msa-gen-op"},
    "per-call mode overrides": {"gemm-bf16-explicit-silicon", "gemm-bf16-empirical", "gen-attn-hybrid-offgrid"},
}


def test_cases_cover_frozen_legacy_surface():
    """Coverage is judged against the FROZEN inventories above — not against
    the hand-written CASES — so dropping a facade/op disposition from the
    sweep fails here instead of silently shrinking coverage."""
    facade_methods = {case["method"] for case in CASES if case["surface"] == "facade"}
    missing_facades = LEGACY_FACADE_SURFACE - facade_methods
    assert not missing_facades, f"facades without a pinned disposition: {sorted(missing_facades)}"

    op_classes = {case["op"] for case in CASES if case["surface"] == "op"}
    missing_ops = LEGACY_OP_QUERY_SURFACE - op_classes
    assert not missing_ops, f"op classes without a pinned disposition: {sorted(missing_ops)}"

    case_ids = {case["id"] for case in CASES}
    for dimension, ids in SEMANTIC_DIMENSION_CASES.items():
        gone = ids - case_ids
        assert not gone, f"semantic dimension {dimension!r} lost its cases: {sorted(gone)}"


def test_manifest_matches_contract_shim_set():
    """The facade manifest and the single-oracle contract's frozen shim set
    must be the SAME surface — a drift means one of them forgot an entry."""
    from test_single_oracle_contract import PERF_DATABASE_QUERY_SHIMS

    assert frozenset(PERF_DATABASE_QUERY_SHIMS) == LEGACY_FACADE_SURFACE

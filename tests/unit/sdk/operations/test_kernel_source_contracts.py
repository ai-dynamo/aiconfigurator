# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the producer/consumer kernel_source contracts.

The SGLang 0.5.14 collector records EXECUTED kernel names; these tests pin
the consumer-side classification/aliasing rules that review #1342 found
drifting twice (Python fixed without Rust, and vice versa). Each test mirrors
one adjudicated finding: B1 (DSA buckets), B2 (DSV4 arch remap), D1 (topk
calib v1/v2 phase split), D2 (GDN decode-recurrence aliases).
"""

import pytest

from aiconfigurator.sdk import common

pytestmark = pytest.mark.unit


# --- B1: DSA kernel_source -> configured-backend bucket(s) ------------------


def _dsa_bucket_view(tmp_path, rows):
    """Serve the DSA context view over synthetic rows (the Python bucket
    helper retired with the parsers; the rule is observed through which
    backend buckets a row's isl lands under — same fixture pattern as
    test_table_view_dsa_dsv4_shapes.py)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml

    from aiconfigurator.sdk.perf_database import PerfDatabase
    from aiconfigurator_core.sdk.engine_table_view import fetch_table_view

    root = tmp_path / "systems"
    root.mkdir(exist_ok=True)
    (root / "h100_sxm.yaml").write_text(
        yaml.safe_dump(
            {
                "data_dir": "data/h100_sxm",
                "gpu": {
                    "sm_version": 90,
                    "mem_bw": 4_800_000_000_000.0,
                    "mem_bw_empirical_scaling_factor": 0.8,
                    "mem_empirical_constant_latency": 0.000003,
                    "bfloat16_tc_flops": 989_000_000_000_000.0,
                    "fp8_tc_flops": 1_978_000_000_000_000.0,
                },
                "node": {
                    "num_gpus_per_node": 8,
                    "inter_node_bw": 50_000_000_000.0,
                    "intra_node_bw": 450_000_000_000.0,
                    "p2p_latency": 0.00001,
                },
                "misc": {"nccl_version": "2.26.2"},
            }
        ),
        encoding="utf-8",
    )
    path = root / "data/h100_sxm/sparse_attention/sglang/1.0.0/dsa_context_module_perf.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({k: [r[k] for r in rows] for k in rows[0]}), path)
    # Synthetic bucket-classification rows intentionally omit Collector V3 sidecars.
    db = PerfDatabase("h100_sxm", "sglang", "1.0.0", str(root), database_mode="HYBRID", strict_provenance=False)
    return fetch_table_view(db, "_context_dsa_module_data")


def _dsa_bucket_row(ks: str, kv: str, isl: int) -> dict:
    return {
        "architecture": "DeepseekV32ForCausalLM",
        "kernel_source": ks,
        "gemm_type": "bfloat16",
        "mla_dtype": "bfloat16",
        "kv_cache_dtype": kv,
        "num_heads": 32,
        "batch_size": 1,
        "isl": isl,
        "step": 0,
        "latency": 1.0,
        "power": 1.0,
    }


def _buckets_containing(view, kv_mode, isl: int) -> tuple[str, ...]:
    arch_table = view[common.FMHAQuantMode.bfloat16][kv_mode][common.GEMMQuantMode.bfloat16]["DeepseekV32ForCausalLM"]
    return tuple(
        bucket for bucket in ("trtllm", "flashmla_kv") if isl in arch_table.get(bucket, {}).get(32, {}).get(0, {})
    )


def test_dsa_bf16_rows_back_both_backend_buckets(tmp_path):
    sources = (
        "sglang_dsa_indexer_trtllm",
        "sglang_dsa_indexer_flashmla_sparse",
        "sglang_dsa_dense_mha_trtllm_ragged",
        "legacy_whatever",
    )
    rows = [_dsa_bucket_row(ks, "bfloat16", 100 + i) for i, ks in enumerate(sources)]
    view = _dsa_bucket_view(tmp_path, rows)
    for i, _ in enumerate(sources):
        assert _buckets_containing(view, common.KVCacheQuantMode.bfloat16, 100 + i) == (
            "trtllm",
            "flashmla_kv",
        )


def test_dsa_fp8_rows_bucket_by_executed_kernel_name(tmp_path):
    cases = [
        ("sglang_dsa_indexer_trtllm", ("trtllm",)),
        ("sglang_dsa_indexer_flashmla_sparse", ("flashmla_kv",)),
        # Dense ragged prefill is selected by SHAPE under either configured
        # backend, so its rows back both buckets.
        ("sglang_dsa_dense_mha_trtllm_ragged", ("trtllm", "flashmla_kv")),
        # Legacy (pre-0.5.14) names keep the old substring rule.
        ("trtllm_gen", ("trtllm",)),
        ("default", ("flashmla_kv",)),
    ]
    rows = [_dsa_bucket_row(ks, "fp8", 100 + i) for i, (ks, _) in enumerate(cases)]
    view = _dsa_bucket_view(tmp_path, rows)
    for i, (_, expected) in enumerate(cases):
        assert _buckets_containing(view, common.KVCacheQuantMode.fp8, 100 + i) == expected


# --- B2: native DSV4 checkpoints remap to arch-specific MoE quant modes -----


def test_dsv4_native_checkpoints_remap_by_system_family():
    from aiconfigurator.sdk.models.helpers import resolve_dsv4_moe_arch_mode

    for path in ("deepseek-ai/DeepSeek-V4-Pro", "deepseek-ai/DeepSeek-V4-Flash"):
        assert resolve_dsv4_moe_arch_mode(path, "b200_sxm", "sglang") is common.MoEQuantMode.w4a8_mxfp4_mxfp8_trtllm
        assert resolve_dsv4_moe_arch_mode(path, "h200_sxm", "sglang") is common.MoEQuantMode.w4a16_mxfp4_cutlass
    # Requant artifacts, other backends, and megamoe stay untouched.
    assert resolve_dsv4_moe_arch_mode("sgl-project/DeepSeek-V4-Pro-FP8", "b200_sxm", "sglang") is None
    assert resolve_dsv4_moe_arch_mode("deepseek-ai/DeepSeek-V4-Pro", "b200_sxm", "trtllm") is None
    assert (
        resolve_dsv4_moe_arch_mode("deepseek-ai/DeepSeek-V4-Pro", "b200_sxm", "sglang", moe_backend="megamoe") is None
    )


def test_kimi_k3_moe_remaps_to_w4a8_on_blackwell_only():
    from aiconfigurator.sdk.models.helpers import resolve_kimi_k3_moe_arch_mode

    # Blackwell serving quantizes activations to mxfp8 (kimi-k3 branch
    # Mxfp4MoEMethod default precision); Hopper keeps the checkpoint's plain
    # W4A16 marlin lane, so the resolver stays silent there.
    for system in ("b200_sxm", "b300_sxm", "gb200", "gb300"):
        mode = resolve_kimi_k3_moe_arch_mode("moonshotai/Kimi-K3", system, "sglang")
        assert mode is common.MoEQuantMode.w4a8_mxfp4_mxfp8
    for system in ("h200_sxm", "h100_sxm"):
        assert resolve_kimi_k3_moe_arch_mode("moonshotai/Kimi-K3", system, "sglang") is None
    assert resolve_kimi_k3_moe_arch_mode("moonshotai/Kimi-K3", "b300_sxm", "vllm") is None
    assert resolve_kimi_k3_moe_arch_mode("moonshotai/Kimi-K2.5", "b300_sxm", "sglang") is None


def test_dsv4_arch_remap_never_overrides_explicit_mode():
    from aiconfigurator.sdk.models.helpers import resolve_dsv4_moe_arch

    class _Cfg:
        moe_quant_mode = common.MoEQuantMode.fp8_block

    cfg = _Cfg()
    resolve_dsv4_moe_arch(cfg, "deepseek-ai/DeepSeek-V4-Pro", system_name="b200_sxm", backend_name="sglang")
    assert cfg.moe_quant_mode is common.MoEQuantMode.fp8_block

    class _Auto:
        moe_quant_mode = None

    auto = _Auto()
    resolve_dsv4_moe_arch(auto, "deepseek-ai/DeepSeek-V4-Pro", system_name="b200_sxm", backend_name="sglang")
    assert auto.moe_quant_mode is common.MoEQuantMode.w4a8_mxfp4_mxfp8_trtllm


# --- D2: GDN decode-recurrence kernel names alias to one modeling identity --


# test_gdn_decode_recurrence_names_alias_to_canonical_key retired with the
# Python GDN loader (PR-6): the decode-recurrence kernel-name aliasing now
# lives in the engine's view fold (table_view.rs::gdn_kernel_alias) and is
# pinned by the data-plane baseline digests (gdn tables, all pins).

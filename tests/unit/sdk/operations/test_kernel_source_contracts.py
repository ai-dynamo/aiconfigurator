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


def test_dsa_bf16_rows_back_both_backend_buckets():
    from aiconfigurator.sdk.operations.dsa import _dsa_kernel_source_buckets

    for ks in (
        "sglang_dsa_indexer_trtllm",
        "sglang_dsa_indexer_flashmla_sparse",
        "sglang_dsa_dense_mha_trtllm_ragged",
        "legacy_whatever",
    ):
        assert _dsa_kernel_source_buckets(ks, common.KVCacheQuantMode.bfloat16) == (
            "trtllm",
            "flashmla_kv",
        )


def test_dsa_fp8_rows_bucket_by_executed_kernel_name():
    from aiconfigurator.sdk.operations.dsa import _dsa_kernel_source_buckets

    fp8 = common.KVCacheQuantMode.fp8
    assert _dsa_kernel_source_buckets("sglang_dsa_indexer_trtllm", fp8) == ("trtllm",)
    assert _dsa_kernel_source_buckets("sglang_dsa_skip_indexer_trtllm", fp8) == ("trtllm",)
    assert _dsa_kernel_source_buckets("sglang_dsa_indexer_flashmla_sparse", fp8) == ("flashmla_kv",)
    assert _dsa_kernel_source_buckets("sglang_dsa_skip_indexer_flashmla_sparse", fp8) == ("flashmla_kv",)
    # Dense ragged prefill is selected by SHAPE under either configured
    # backend, so its rows back both buckets.
    assert _dsa_kernel_source_buckets("sglang_dsa_dense_mha_trtllm_ragged", fp8) == (
        "trtllm",
        "flashmla_kv",
    )
    # Legacy (pre-0.5.14) names keep the old substring rule.
    assert _dsa_kernel_source_buckets("trtllm_gen", fp8) == ("trtllm",)
    assert _dsa_kernel_source_buckets("default", fp8) == ("flashmla_kv",)


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

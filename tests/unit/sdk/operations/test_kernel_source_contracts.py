# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the producer/consumer kernel_source contracts.

The SGLang 0.5.14 collector records EXECUTED kernel names; these tests pin
the consumer-side classification/aliasing rules that review #1342 found
drifting twice (Python fixed without Rust, and vice versa). Each test mirrors
one adjudicated finding: B1 (DSA buckets), B2 (DSV4 arch remap), D1 (topk
calib v1/v2 phase split), D2 (GDN decode-recurrence aliases).
"""

import csv
import os
import tempfile

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


def test_gdn_decode_recurrence_names_alias_to_canonical_key():
    from aiconfigurator.sdk.operations.mamba import load_gdn_data

    header = [
        "framework",
        "version",
        "device",
        "op_name",
        "kernel_source",
        "phase",
        "batch_size",
        "seq_len",
        "num_tokens",
        "d_model",
        "d_conv",
        "num_k_heads",
        "head_k_dim",
        "num_v_heads",
        "head_v_dim",
        "model_name",
        "latency",
    ]

    def row(kernel_source, batch, latency):
        return {
            "framework": "SGLang",
            "version": "0.5.14",
            "device": "B200",
            "op_name": "gdn",
            "kernel_source": kernel_source,
            "phase": "generation",
            "batch_size": batch,
            "seq_len": 1,
            "num_tokens": batch,
            "d_model": 2048,
            "d_conv": 4,
            "num_k_heads": 16,
            "head_k_dim": 128,
            "num_v_heads": 32,
            "head_v_dim": 128,
            "model_name": "Qwen/Qwen3.5-27B",
            "latency": latency,
        }

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "gdn_perf.txt")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            # 0.5.10-era and 0.5.14 executed-kernel names for the same
            # decode recurrence; both must land under the canonical key.
            writer.writerow(row("fused_recurrent_gated_delta_rule", 1, 0.5))
            writer.writerow(row("fused_recurrent_gated_delta_rule_packed_decode", 2, 0.7))
            writer.writerow(row("fused_sigmoid_gating_delta_rule_update", 4, 0.9))
        data = load_gdn_data(path)

    assert set(data.keys()) == {"fused_sigmoid_gating_delta_rule_update"}
    leaves = data["fused_sigmoid_gating_delta_rule_update"]["generation"][(2048, 16, 128, 32, 128, 4)]
    assert {b: leaves[b]["latency"] for b in sorted(leaves)} == {1: 0.5, 2: 0.7, 4: 0.9}


# --- D1: topk calib pairs v1/v2 phase variants separately --------------------


def test_topk_calib_builder_splits_v1_and_v2_variants():
    from aiconfigurator.sdk.operations.dsv4 import _build_topk_calib_from_rows

    by_mode = {
        0: {
            4096: {
                8: {
                    "v1_flat": {"latency": 1.5},
                    "v1_top_last": {"latency": 1.0},
                    "v2_flat": {"latency": 0.9},
                    "v2_top_last": {"latency": 0.7},
                }
            }
        }
    }
    calib = _build_topk_calib_from_rows(by_mode)
    assert calib["v1"]["exact"][(0, 4096, 8)] == pytest.approx(0.5)
    assert calib["v2"]["exact"][(0, 4096, 8)] == pytest.approx(0.2)

    # A file carrying only one variant leaves the other None rather than
    # silently borrowing across phases.
    only_v1 = {0: {4096: {8: {"v1_flat": {"latency": 1.5}, "v1_top_last": {"latency": 1.0}}}}}
    calib = _build_topk_calib_from_rows(only_v1)
    assert calib["v1"] is not None
    assert calib["v2"] is None

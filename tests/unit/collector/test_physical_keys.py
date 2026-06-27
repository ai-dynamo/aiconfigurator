# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from collector.planner.physical_keys import (
    PHYSICAL_KEY_REGISTRY,
    PHYSICAL_KEY_SCHEMA_VERSION,
    PhysicalKeyError,
    physical_row_key,
)


def test_unknown_table_is_explicitly_legacy_and_not_guessed():
    row = {"gemm_dtype": "bfloat16", "m": 1, "n": 2, "k": 3}

    assert physical_row_key("unknown_perf.parquet", row) is None


def test_txt_and_parquet_names_share_one_versioned_table_scope():
    row = {"gemm_dtype": "bf16", "m": "1", "n": 2, "k": 3}

    txt_key = physical_row_key("/tmp/results/gemm_perf.txt", row)
    parquet_key = physical_row_key("gemm_perf.parquet", row)

    assert txt_key == parquet_key
    assert txt_key is not None
    assert txt_key.schema_version == PHYSICAL_KEY_SCHEMA_VERSION
    assert txt_key.table == "gemm_perf.parquet"
    assert txt_key.values == ("bfloat16", 1, 2, 3)


def test_model_alias_and_measurement_fields_do_not_change_mla_module_key():
    base = {
        "model": "deepseek-ai/DeepSeek-V3",
        "architecture": "DeepseekV3ForCausalLM",
        "mla_dtype": "bfloat16",
        "kv_cache_dtype": "fp8_e4m3",
        "gemm_type": "fp8_block",
        "num_heads": 16,
        "batch_size": 8,
        "isl": 1024,
        "tp_size": 8,
        "step": 0,
        "latency": 1.0,
        "power": 400,
        "framework": "TRTLLM",
        "version": "1.3.0",
        "device": "H100",
    }
    alias = {
        **base,
        "model": "nvidia/DeepSeek-V3-NVFP4",
        "architecture": "AnIgnoredAliasArchitecture",
        "tp_size": 64,
        "latency": 99.0,
        "power": 700,
        "framework": "VLLM",
        "version": "different",
        "device": "different",
    }

    assert physical_row_key("mla_context_module_perf.parquet", base) == physical_row_key(
        "mla_context_module_perf.parquet", alias
    )


def test_table_scope_keeps_equal_payloads_from_different_tables_distinct():
    row = {"quant_dtype": "fp8", "m": 32, "k": 4096}

    assert physical_row_key("computescale_perf.parquet", row) != physical_row_key("scale_matrix_perf.parquet", row)


def test_context_attention_normalizes_mha_kv_heads_to_zero():
    mha = {
        "attn_dtype": "bf16",
        "kv_cache_dtype": "bfloat16",
        "num_heads": 32,
        "num_key_value_heads": 32,
        "head_dim": 128,
        "batch_size": 4,
        "isl": 2048,
    }
    already_normalized = {**mha, "num_key_value_heads": 0, "window_size": 0}

    mha_key = physical_row_key("context_attention_perf.txt", mha)
    normalized_key = physical_row_key("context_attention_perf.parquet", already_normalized)

    assert mha_key == normalized_key
    assert mha_key is not None
    assert mha_key.values[2] == 0


def test_context_attention_keeps_real_gqa_kv_heads():
    row = {
        "attn_dtype": "bfloat16",
        "kv_cache_dtype": "fp8",
        "num_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "window_size": 4096,
        "batch_size": 2,
        "isl": 8192,
    }

    key = physical_row_key("context_attention_perf.parquet", row)

    assert key is not None
    assert key.values[2] == 8


def test_generation_attention_uses_total_sequence_and_ignores_attn_dtype():
    row = {
        "attn_dtype": "bfloat16",
        "kv_cache_dtype": "fp8",
        "num_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "window_size": 0,
        "batch_size": 16,
        "isl": 1,
        "step": 4095,
    }

    key = physical_row_key("generation_attention_perf.parquet", row)
    changed_measurement_only = physical_row_key(
        "generation_attention_perf.parquet",
        {**row, "attn_dtype": "fp8", "latency": 123},
    )

    assert key == changed_measurement_only
    assert key is not None
    assert key.values[-1] == 4096


def test_gdn_legacy_recurrence_alias_normalizes_to_runtime_kernel():
    base = {
        "kernel_source": "fused_recurrent_gated_delta_rule",
        "phase": "generation",
        "batch_size": 64,
        "seq_len": 1,
        "d_model": 2048,
        "d_conv": 4,
        "num_k_heads": 16,
        "head_k_dim": 128,
        "num_v_heads": 32,
        "head_v_dim": 64,
    }
    legacy = {**base, "kernel_source": "fused_sigmoid_gating_delta_rule_update"}

    assert physical_row_key("gdn_perf.parquet", base) == physical_row_key("gdn_perf.parquet", legacy)


def test_dsv4_context_key_uses_profile_tp_local_heads_and_prefix():
    row = {
        "model": "deepseek-ai/DeepSeek-V4-Pro",
        "mla_dtype": "bfloat16",
        "kv_cache_dtype": "fp8_e4m3",
        "gemm_type": "fp8_block",
        # New rows can log native heads; the loader derives the physical shard.
        "num_heads": 128,
        "tp_size": 4,
        "compress_ratio": 4,
        "step": 8192,
        "isl": 1024,
        "batch_size": 2,
    }
    alias = {**row, "model": "sgl-project/DeepSeek-V4-Pro-FP8", "num_heads": 32, "latency": 2.0}

    key = physical_row_key("dsv4_csa_context_module_perf.parquet", row)
    alias_key = physical_row_key("dsv4_csa_context_module_perf.parquet", alias)

    assert key == alias_key
    assert key is not None
    assert key.values[3:6] == ("pro", 4, 32)
    assert key.values[7] == 8192


def test_dsv4_profiles_and_prefixes_remain_distinct():
    base = {
        "model": "deepseek-ai/DeepSeek-V4-Pro",
        "mla_dtype": "bfloat16",
        "kv_cache_dtype": "fp8",
        "gemm_type": "fp8_block",
        "num_heads": 128,
        "tp_size": 2,
        "compress_ratio": 4,
        "step": 0,
        "isl": 1024,
        "batch_size": 1,
    }

    pro = physical_row_key("dsv4_csa_context_module_perf.parquet", base)
    flash = physical_row_key(
        "dsv4_csa_context_module_perf.parquet",
        {**base, "model": "deepseek-ai/DeepSeek-V4-Flash", "num_heads": 64},
    )
    prefixed = physical_row_key("dsv4_csa_context_module_perf.parquet", {**base, "step": 4096})

    assert pro != flash
    assert pro != prefixed


def test_dsv4_sparse_ignores_model_alias_but_topk_uses_profile_and_score_mode():
    sparse = {
        "model": "deepseek-ai/DeepSeek-V4-Pro",
        "num_heads": 128,
        "tp_size": 1,
        "step": 4096,
        "isl": 512,
        "batch_size": 4,
        "latency": 1.0,
    }
    sparse_alias = {**sparse, "model": "sgl-project/DeepSeek-V4-Pro-FP8", "latency": 9.0}

    assert physical_row_key("dsv4_paged_mqa_logits_module_perf.parquet", sparse) == physical_row_key(
        "dsv4_paged_mqa_logits_module_perf.parquet", sparse_alias
    )

    topk_flat = physical_row_key(
        "dsv4_csa_topk_calib_perf.parquet",
        {**sparse, "score_mode": "flat"},
    )
    topk_last = physical_row_key(
        "dsv4_csa_topk_calib_perf.parquet",
        {**sparse_alias, "score_mode": "top_last"},
    )
    assert topk_flat != topk_last
    assert topk_flat is not None
    assert topk_flat.values[0] == "pro"


def test_shared_mla_bmm_table_includes_op_name():
    common = {
        "bmm_dtype": "fp8",
        "num_heads": 16,
        "num_tokens": 128,
        "latency": 0.1,
    }

    pre = physical_row_key("mla_bmm_perf.parquet", {**common, "op_name": "mla_gen_pre"})
    post = physical_row_key("mla_bmm_perf.parquet", {**common, "op_name": "mla_gen_post"})
    registry_alias = physical_row_key("mla_bmm_perf.parquet", {**common, "op_name": "mla_bmm_gen_pre"})

    assert pre != post
    assert pre == registry_alias


def test_known_table_missing_consumer_field_fails_instead_of_guessing():
    with pytest.raises(PhysicalKeyError, match="num_key_value_heads"):
        physical_row_key(
            "context_attention_perf.parquet",
            {
                "attn_dtype": "bfloat16",
                "kv_cache_dtype": "bfloat16",
                "num_heads": 32,
                "head_dim": 128,
                "batch_size": 1,
                "isl": 128,
            },
        )


def test_registry_explicitly_covers_required_schema_families():
    required = {
        "gemm_perf.parquet",
        "computescale_perf.parquet",
        "scale_matrix_perf.parquet",
        "context_attention_perf.parquet",
        "generation_attention_perf.parquet",
        "encoder_attention_perf.parquet",
        "context_mla_perf.parquet",
        "generation_mla_perf.parquet",
        "mla_bmm_perf.parquet",
        "mamba2_perf.parquet",
        "gdn_perf.parquet",
        "mla_context_module_perf.parquet",
        "mla_generation_module_perf.parquet",
        "dsa_context_module_perf.parquet",
        "dsa_generation_module_perf.parquet",
        "wideep_context_mla_perf.parquet",
        "wideep_generation_mla_perf.parquet",
        "dsv4_csa_context_module_perf.parquet",
        "dsv4_hca_generation_module_perf.parquet",
        "dsv4_paged_mqa_logits_module_perf.parquet",
        "dsv4_hca_attn_module_perf.parquet",
        "dsv4_csa_topk_calib_perf.parquet",
        "mhc_module_perf.parquet",
    }

    assert required <= PHYSICAL_KEY_REGISTRY.keys()

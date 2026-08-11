# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy

import pytest

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.afd_partition import build_afd_ops_partition
from aiconfigurator.sdk.models import LagunaModel, check_is_moe, get_model, get_model_family
from aiconfigurator.sdk.utils import get_model_config_from_model_path

pytestmark = pytest.mark.unit

LAGUNA_MODEL_PATH = "poolside/Laguna-S-2.1-FP8"


@pytest.fixture
def laguna_model_path(tmp_path):
    config_json = deepcopy(get_model_config_from_model_path(LAGUNA_MODEL_PATH)["raw_config"])
    model_dir = tmp_path / "Laguna-S-v2.1"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(config_json))
    return str(model_dir)


def _model_config(tp_size=8, moe_ep_size=8):
    return config.ModelConfig(
        tp_size=tp_size,
        pp_size=1,
        moe_tp_size=1,
        moe_ep_size=moe_ep_size,
        attention_dp_size=1,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        moe_quant_mode=common.MoEQuantMode.bfloat16,
    )


def _model_config_with_quant_defaults(tp_size=8, moe_ep_size=8):
    return config.ModelConfig(
        tp_size=tp_size,
        pp_size=1,
        moe_tp_size=1,
        moe_ep_size=moe_ep_size,
        attention_dp_size=1,
    )


def _op_names(model, phase="context"):
    ops = model.context_ops if phase == "context" else model.generation_ops
    return {op._name for op in ops}


def _laguna_model(model_path, tp_size=8, moe_ep_size=8):
    return get_model(model_path, _model_config(tp_size=tp_size, moe_ep_size=moe_ep_size), backend_name="vllm")


def test_laguna_local_path_resolves_family_and_moe(laguna_model_path):
    assert get_model_family(laguna_model_path) == "LAGUNA"
    assert check_is_moe(laguna_model_path) is True

    model = _laguna_model(laguna_model_path)

    assert isinstance(model, LagunaModel)
    assert model.model_family == "LAGUNA"
    assert model.architecture == "LagunaForCausalLM"


def test_laguna_s2_fp8_resolves_vllm_quant_defaults():
    model_config = _model_config_with_quant_defaults(tp_size=4, moe_ep_size=4)

    model = get_model(LAGUNA_MODEL_PATH, model_config, backend_name="vllm")

    assert isinstance(model, LagunaModel)
    assert model._count_layer_types() == {
        "global_dense": 1,
        "global_moe": 11,
        "swa_moe": 36,
        "swa_dense": 0,
    }
    assert model_config.gemm_quant_mode == common.GEMMQuantMode.bfloat16
    assert model_config.moe_quant_mode == common.MoEQuantMode.fp8_block
    assert model_config.kvcache_quant_mode == common.KVCacheQuantMode.fp8
    assert model_config.fmha_quant_mode == common.FMHAQuantMode.bfloat16
    shared_expert_ops = [
        op
        for op in [*model.context_ops, *model.generation_ops]
        if "_shared_" in op._name and hasattr(op, "_quant_mode")
    ]
    assert shared_expert_ops
    assert all(op._quant_mode == common.GEMMQuantMode.bfloat16 for op in shared_expert_ops)


def test_laguna_counts_expected_layer_buckets(laguna_model_path):
    model = _laguna_model(laguna_model_path)

    assert model._count_layer_types() == {
        "global_dense": 1,
        "global_moe": 11,
        "swa_moe": 36,
        "swa_dense": 0,
    }


def test_laguna_context_and_generation_ops_include_expected_recipes(laguna_model_path):
    model = _laguna_model(laguna_model_path)

    context_names = _op_names(model, "context")
    generation_names = _op_names(model, "generation")

    # Check all the expected ops are present
    for names, phase in ((context_names, "context"), (generation_names, "generation")):
        assert f"{phase}_global_dense_qkv_gemm" in names
        assert f"{phase}_global_qkv_gemm" in names
        assert f"{phase}_swa_qkv_gemm" in names
        assert f"{phase}_global_dense_attention_gate_gemm" in names
        assert f"{phase}_global_attention_gate_gemm" in names
        assert f"{phase}_swa_attention_gate_gemm" in names
        assert f"{phase}_global_router_gemm" in names
        assert f"{phase}_swa_router_gemm" in names
        assert f"{phase}_global_moe" in names
        assert f"{phase}_swa_moe" in names
        assert f"{phase}_global_shared_gate_up_gemm" in names
        assert f"{phase}_swa_shared_gate_up_gemm" in names
        assert f"{phase}_global_dense_dense_gate_up_gemm" in names
        assert f"{phase}_attention" in names
        assert f"{phase}_logits_gemm" in names

    # Check no shared expert ops are in the dense layers
    assert "context_global_dense_shared_gate_up_gemm" not in context_names
    assert "generation_global_dense_shared_gate_up_gemm" not in generation_names


def test_laguna_qkv_width_uses_per_layer_attention_heads(laguna_model_path):
    model = _laguna_model(laguna_model_path, tp_size=8, moe_ep_size=8)

    # Get the QKV GEMM ops for the global and SWA layers
    global_qkv = next(op for op in model.context_ops if op._name == "context_global_qkv_gemm")
    swa_qkv = next(op for op in model.context_ops if op._name == "context_swa_qkv_gemm")

    # Check the QKV GEMM output width is correct
    # Formula: (n_q_heads / tp) * head_dim + (n_kv_heads / tp) * head_dim * 2
    # global: (48 / 8) * 128 + (8 / 8) * 128 * 2  →  6 * 128 + 1 * 256
    # swa:    (72 / 8) * 128 + (8 / 8) * 128 * 2  →  9 * 128 + 1 * 256
    assert global_qkv._n == 6 * 128 + 1 * 128 * 2
    assert swa_qkv._n == 9 * 128 + 1 * 128 * 2


def test_laguna_kvcache_bytes_window_caps_swa_layers(laguna_model_path):
    model = _laguna_model(laguna_model_path, tp_size=8, moe_ep_size=8)
    seq_len = 8000
    bytes_per_elem = common.KVCacheQuantMode.bfloat16.value.memory

    # Formula: n_layers * kv_heads_per_gpu * head_dim * 2 * bytes_per_elem * effective_seq_len
    # SWA:    36 layers * (8 kv_heads / 8 tp) * 128 * 2 (K+V) * bytes * min(seq_len, window=512)
    # global: 12 layers * (8 kv_heads / 8 tp) * 128 * 2 (K+V) * bytes * seq_len (no window cap)
    expected_swa = 36 * 1 * 128 * 2 * bytes_per_elem * 512
    expected_global = 12 * 1 * 128 * 2 * bytes_per_elem * seq_len
    expected_total_bytes = expected_swa + expected_global

    assert model.get_kvcache_bytes_per_sequence(seq_len) == expected_total_bytes


def test_laguna_kvcache_max_tokens_inverts_window_capped_curve(laguna_model_path):
    model = _laguna_model(laguna_model_path, tp_size=2, moe_ep_size=2)
    seq_len = 8000
    budget = model.get_kvcache_bytes_per_sequence(seq_len)

    tokens = model.get_kvcache_max_tokens(budget)

    assert tokens == seq_len
    assert model.get_kvcache_bytes_per_sequence(tokens) <= budget
    assert model.get_kvcache_bytes_per_sequence(tokens + 1) > budget
    assert tokens > int(budget // model.get_kvcache_bytes_per_sequence(1))


def test_laguna_invalid_tp_for_swa_heads_raises(laguna_model_path):
    with pytest.raises(ValueError, match="attention head count 72 must be divisible"):
        _laguna_model(laguna_model_path, tp_size=16, moe_ep_size=16)


def test_laguna_invalid_gqa_head_ratio_raises(laguna_model_path):
    config_path = f"{laguna_model_path}/config.json"
    with open(config_path) as f:
        config_json = json.load(f)
    config_json["num_attention_heads_per_layer"] = [
        48 if layer_type == "full_attention" else 70 for layer_type in config_json["layer_types"]
    ]
    with open(config_path, "w") as f:
        json.dump(config_json, f)

    with pytest.raises(ValueError, match="num_kv_heads 8"):
        _laguna_model(laguna_model_path, tp_size=2, moe_ep_size=2)


def test_laguna_afd_partition_classifies_all_ops(laguna_model_path):
    model = _laguna_model(laguna_model_path)

    # allow_unknown_ops=False means the partitioner must classify every op; any
    # unrecognised op raises rather than being silently dropped.
    context_partition = build_afd_ops_partition(model, phase="context", allow_unknown_ops=False)
    generation_partition = build_afd_ops_partition(model, phase="generation", allow_unknown_ops=False)

    # SWA attention-gate GEMMs must land in the attention bucket, not FFN.
    assert any(op._name == "context_swa_attention_gate_gemm" for op in context_partition.attn_ops)
    assert not any(op._name == "context_swa_attention_gate_gemm" for op in context_partition.ffn_ops)
    assert any(op._name == "generation_swa_attention_gate_gemm" for op in generation_partition.attn_ops)
    assert not any(op._name == "generation_swa_attention_gate_gemm" for op in generation_partition.ffn_ops)
    # SWA shared-expert GEMMs must land in the FFN bucket, not attention.
    assert any(op._name == "context_swa_shared_gate_up_gemm" for op in context_partition.ffn_ops)
    assert not any(op._name == "context_swa_shared_gate_up_gemm" for op in context_partition.attn_ops)
    assert any(op._name == "generation_swa_shared_gate_up_gemm" for op in generation_partition.ffn_ops)
    assert not any(op._name == "generation_swa_shared_gate_up_gemm" for op in generation_partition.attn_ops)

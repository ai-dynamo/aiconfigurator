# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Architecture tests for the Gemma 4 vision tower and language adapter."""

import pytest

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.models import get_model

pytestmark = pytest.mark.unit

MODEL = "google/gemma-4-26B-A4B"


def _model_config(*, tp_size: int = 1, enable_encoder_dp: bool = True) -> config.ModelConfig:
    return config.ModelConfig(
        tp_size=tp_size,
        moe_tp_size=1,
        moe_ep_size=tp_size,
        enable_encoder_dp=enable_encoder_dp,
    )


def _op(model, name: str):
    return next(op for op in model.encoder_ops if op._name == name)


def test_gemma4_model_builds_checkpoint_accurate_vision_graph():
    model = get_model(MODEL, _model_config(), "trtllm")

    enc = model.encoder_config
    assert isinstance(enc, common.Gemma4VisionEncoderConfig)
    assert (enc.depth, enc.hidden_size, enc.num_heads, enc.head_dim) == (27, 1152, 16, 72)
    assert enc.intermediate_size == 4304
    assert enc.pooling_kernel_size == 3
    assert enc.soft_tokens_per_image == 280
    assert enc.projector_dims == ((1152, 2816),)

    names = {op._name for op in model.encoder_ops}
    assert {
        "encoder_patch_embed_gemm",
        "encoder_position_embedding",
        "encoder_qkv_norm_rope_2d",
        "encoder_attention",
        "encoder_ffn_gate_up_gemm",
        "encoder_ffn_act_mul",
        "encoder_ffn_down_gemm",
        "encoder_gemma4_pool_avg",
        "encoder_gemma4_pool_postprocess",
        "encoder_projector_pre_norm",
        "encoder_projector_fc0_gemm",
    } <= names

    # This is a gated Gemma MLP plus average pool + single adapter, not the
    # Qwen3-VL single-up-projection/PatchMerger/two-layer projector graph.
    assert "encoder_ffn1_gemm" not in names
    assert "encoder_projector_fc1_gemm" not in names
    assert _op(model, "encoder_patch_embed_gemm")._k == 3 * 16**2
    assert _op(model, "encoder_qkv_gemm")._n == 3 * 1152
    assert _op(model, "encoder_attention")._head_size == 72
    assert _op(model, "encoder_ffn_gate_up_gemm")._n == 2 * 4304
    assert _op(model, "encoder_gemma4_pool_avg")._scale_num_tokens == 9
    assert (_op(model, "encoder_projector_fc0_gemm")._n, _op(model, "encoder_projector_fc0_gemm")._k) == (
        2816,
        1152,
    )


def test_gemma4_encoder_weights_include_patch_position_vit_and_adapter():
    model = get_model(MODEL, _model_config(), "trtllm")

    expected_bf16_weights = (
        1152 * (3 * 16**2)  # patch projection
        + 2 * 10240 * 1152  # learned x/y position tables
        + 27
        * (
            (3 * 1152) * 1152  # QKV
            + 1152 * 1152  # attention output
            + (2 * 4304) * 1152  # gated MLP gate/up
            + 1152 * 4304  # gated MLP down
        )
        + 2816 * 1152  # vision-to-language projection
    )
    assert sum(op.get_weights() for op in model.encoder_ops) == expected_bf16_weights * 2


def test_gemma4_encoder_dp_replicates_compute_and_gathers_soft_tokens():
    model = get_model(MODEL, _model_config(tp_size=4, enable_encoder_dp=True), "trtllm")

    assert _op(model, "encoder_qkv_gemm")._n == 3 * 1152
    assert _op(model, "encoder_ffn_gate_up_gemm")._n == 2 * 4304
    gather = _op(model, "encoder_dp_all_gather")
    assert gather._num_gpus == 4
    assert gather._num_elements_per_token == 2816 * 4


def test_gemma4_encoder_tp_shards_tower_but_replicates_adapter_and_communicates():
    model = get_model(MODEL, _model_config(tp_size=4, enable_encoder_dp=False), "trtllm")

    names = {op._name for op in model.encoder_ops}
    assert "encoder_dp_all_gather" not in names
    assert _op(model, "encoder_qkv_gemm")._n == 3 * 1152 // 4
    assert _op(model, "encoder_attention")._n == 16 // 4
    assert _op(model, "encoder_ffn_gate_up_gemm")._n == 2 * 4304 // 4
    # vLLM uses ReplicatedLinear for Gemma4MultimodalEmbedder and HF's
    # vision TP plan excludes this projection.
    assert _op(model, "encoder_projector_fc0_gemm")._n == 2816
    assert _op(model, "encoder_ar_1")._tp_size == 4
    assert _op(model, "encoder_ar_2")._tp_size == 4
    assert "encoder_projector_ar" not in names

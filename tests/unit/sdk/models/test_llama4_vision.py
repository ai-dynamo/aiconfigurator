# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.resources as pkg_resources
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.models import HybridMoEModel, get_model
from aiconfigurator.sdk.utils import get_model_config_from_model_path

pytestmark = pytest.mark.unit

LLAMA4_CHECKPOINTS = (
    ("meta-llama/Llama-4-Scout-17B-16E-Instruct", 16, 48, 0),
    ("meta-llama/Llama-4-Maverick-17B-128E-Instruct", 128, 24, 24),
)


def _model_config(**overrides):
    values = {
        "tp_size": 1,
        "pp_size": 1,
        "attention_dp_size": 1,
        "moe_tp_size": 1,
        "moe_ep_size": 1,
    }
    values.update(overrides)
    return config.ModelConfig(**values)


@pytest.mark.parametrize("model_id,num_experts,moe_layers,dense_layers", LLAMA4_CHECKPOINTS)
def test_checkpoint_configs_preserve_text_and_exact_vision_shapes(model_id, num_experts, moe_layers, dense_layers):
    info = get_model_config_from_model_path(model_id)

    assert info["architecture"] == "Llama4ForConditionalGeneration"
    assert info["layers"] == 48
    assert info["hidden_size"] == 5120
    assert info["num_experts"] == num_experts

    hybrid = info["extra_params"]
    assert isinstance(hybrid, common.HybridMoEConfig)
    assert sum(hybrid.moe_layer_freq) == moe_layers
    assert hybrid.moe_layer_freq.count(0) == dense_layers

    vision = hybrid.vision_config
    assert isinstance(vision, common.VisionEncoderConfig)
    assert vision.depth == 34
    assert vision.hidden_size == 1408
    assert vision.num_heads == 16
    assert vision.intermediate_size == 5632
    assert vision.image_size == 336
    assert vision.patch_size == 14
    assert vision.spatial_merge_size == 2
    assert vision.out_hidden_size == 5120
    assert vision.projector_dims == ((5632, 4096), (4096, 4096), (4096, 5120))
    assert vision.num_channels == 3
    assert vision.has_cls_token
    assert vision.max_num_tiles == 16
    assert vision.add_global_tile


@pytest.mark.parametrize("model_id,_,__,___", LLAMA4_CHECKPOINTS)
def test_bundled_checkpoint_json_preserves_llama4_special_tokens_and_vision_metadata(model_id, _, __, ___):
    config_path = (
        pkg_resources.files("aiconfigurator_core") / "model_configs" / (f"{model_id.replace('/', '--')}_config.json")
    )
    checkpoint = json.loads(config_path.read_text())

    assert checkpoint["boi_token_index"] == 200080
    assert checkpoint["eoi_token_index"] == 200081
    assert checkpoint["image_token_index"] == 200092
    assert checkpoint["vision_config"] == {
        "attention_dropout": 0.0,
        "hidden_act": "gelu",
        "hidden_size": 1408,
        "image_size": 336,
        "initializer_range": 0.02,
        "intermediate_size": 5632,
        "model_type": "llama4_vision_model",
        "multi_modal_projector_bias": False,
        "norm_eps": 1e-5,
        "num_channels": 3,
        "num_hidden_layers": 34,
        "num_attention_heads": 16,
        "patch_size": 14,
        "pixel_shuffle_ratio": 0.5,
        "projector_dropout": 0.0,
        "projector_input_dim": 4096,
        "projector_output_dim": 4096,
        "rope_theta": 10000,
        "torch_dtype": "bfloat16",
        "vision_feature_layer": -1,
        "vision_feature_select_strategy": "default",
        "vision_output_dim": 4096,
    }


@pytest.mark.parametrize("model_id,_,__,___", LLAMA4_CHECKPOINTS)
def test_both_checkpoints_build_vision_and_hybrid_text_ops(model_id, _, __, ___):
    model = get_model(model_id, _model_config(), "trtllm")

    assert isinstance(model, HybridMoEModel)
    assert model.encoder_ops
    assert model.context_ops
    assert model.generation_ops
    names = {op._name for op in model.encoder_ops}
    assert {
        "encoder_patch_embedding_gemm",
        "encoder_qkv_gemm",
        "encoder_attention",
        "encoder_ffn1_gemm",
        "encoder_ffn2_gemm",
        "encoder_projector_pixel_shuffle",
        "encoder_projector_adapter_fc0_gemm",
        "encoder_projector_adapter_fc1_gemm",
        "encoder_projector_adapter_ar",
        "encoder_projector_mm_gemm",
    } <= names


def test_scout_encoder_operation_shapes_match_engine_modules():
    model = get_model(LLAMA4_CHECKPOINTS[0][0], _model_config(), "trtllm")
    by_name = {op._name: op for op in model.encoder_ops}

    assert (by_name["encoder_patch_embedding_gemm"]._n, by_name["encoder_patch_embedding_gemm"]._k) == (
        1408,
        3 * 14 * 14,
    )
    assert (by_name["encoder_qkv_gemm"]._n, by_name["encoder_qkv_gemm"]._k) == (3 * 1408, 1408)
    assert by_name["encoder_qkv_gemm"]._scale_factor == 34
    assert by_name["encoder_attention"]._n == 16
    assert by_name["encoder_attention"]._head_size == 88
    assert by_name["encoder_attention"]._scale_factor == 34
    assert (by_name["encoder_ffn1_gemm"]._n, by_name["encoder_ffn1_gemm"]._k) == (5632, 1408)
    assert (by_name["encoder_projector_adapter_fc0_gemm"]._n, by_name["encoder_projector_adapter_fc0_gemm"]._k) == (
        4096,
        5632,
    )
    assert (by_name["encoder_projector_adapter_fc1_gemm"]._n, by_name["encoder_projector_adapter_fc1_gemm"]._k) == (
        4096,
        4096,
    )
    assert (by_name["encoder_projector_mm_gemm"]._n, by_name["encoder_projector_mm_gemm"]._k) == (5120, 4096)


def test_llama4_encoder_tp_models_both_engine_communication_boundaries():
    model = get_model(
        LLAMA4_CHECKPOINTS[0][0],
        _model_config(tp_size=8, moe_tp_size=1, moe_ep_size=8, enable_encoder_dp=False),
        "trtllm",
    )
    by_name = {op._name: op for op in model.encoder_ops}

    assert "encoder_patch_embedding_all_gather" in by_name
    assert by_name["encoder_projector_adapter_ar"]._tp_size == 8
    assert "encoder_projector_mm_all_gather" in by_name
    assert "encoder_dp_all_gather" not in by_name


def test_llama4_encoder_dp_models_exit_all_gather_only():
    model = get_model(
        LLAMA4_CHECKPOINTS[0][0],
        _model_config(tp_size=8, moe_tp_size=1, moe_ep_size=8, enable_encoder_dp=True),
        "trtllm",
    )
    names = {op._name for op in model.encoder_ops}

    assert "encoder_dp_all_gather" in names
    assert "encoder_patch_embedding_all_gather" not in names
    assert "encoder_projector_mm_all_gather" not in names


def test_single_tile_image_produces_nonzero_engine_and_text_tokens():
    enc_cfg = get_model_config_from_model_path(LLAMA4_CHECKPOINTS[0][0])["extra_params"].vision_config
    workload = BaseBackend._encoder_workload_per_visual(
        RuntimeConfig(image_height=336, image_width=336, num_images_per_request=1),
        enc_cfg,
    )

    assert workload.patch_tokens_per_sequence == 576
    assert workload.transformer_tokens_per_sequence == 577
    assert workload.output_tokens_per_sequence == 144
    assert workload.output_tokens_per_image == 144
    assert workload.sequences_per_image == 1


def test_four_local_tiles_add_engine_global_tile_and_720_text_tokens():
    enc_cfg = get_model_config_from_model_path(LLAMA4_CHECKPOINTS[0][0])["extra_params"].vision_config
    workload = BaseBackend._encoder_workload_per_visual(
        RuntimeConfig(image_height=672, image_width=672, num_images_per_request=1),
        enc_cfg,
    )

    assert workload.sequences_per_image == 5
    assert workload.output_tokens_per_image == 5 * 144


@pytest.mark.parametrize("model_id,_,__,___", LLAMA4_CHECKPOINTS)
def test_nonzero_image_workload_reaches_encoder_and_text_context(model_id, _, __, ___):
    model = get_model(model_id, _model_config(), "trtllm")
    result = MagicMock(__float__=lambda self: 1.0, energy=2.0, source="silicon")
    for op in model.encoder_ops:
        op.query = MagicMock(return_value=result)
    database = SimpleNamespace(backend="trtllm", version="test", system="h200_sxm")
    runtime = RuntimeConfig(batch_size=1, isl=256, osl=16, image_height=336, image_width=336)

    latency, energy, source, image_tokens = TRTLLMBackend()._run_encoder_phase(model, database, runtime, 1)

    assert image_tokens == 144
    assert sum(latency.values()) > 0
    assert sum(energy.values()) > 0
    assert source
    by_name = {op._name: op for op in model.encoder_ops}
    assert by_name["encoder_patch_embedding_gemm"].query.call_args.kwargs["s"] == 576
    assert by_name["encoder_attention"].query.call_args.kwargs["s"] == 577
    assert by_name["encoder_projector_mm_gemm"].query.call_args.kwargs["s"] == 144


def test_static_ttft_memory_and_energy_include_llama4_encoder():
    model = get_model(LLAMA4_CHECKPOINTS[0][0], _model_config(), "trtllm")
    result = MagicMock(__float__=lambda self: 1.0, energy=2.0, source="silicon")
    for op in model.encoder_ops + model.context_ops + model.generation_ops:
        op.query = MagicMock(return_value=result)
    database = SimpleNamespace(
        backend="trtllm",
        version="test",
        system="h200_sxm",
        system_spec={
            "gpu": {"mem_capacity": 16 * (1 << 40)},
            "misc": {"nccl_mem": {1: 0}, "other_mem": 0},
        },
    )
    runtime = RuntimeConfig(
        batch_size=1,
        isl=256,
        osl=16,
        image_height=336,
        image_width=336,
        engine_step_backend="python",
    )

    summary = TRTLLMBackend().run_static(model, database, runtime, mode="static")
    row = summary.get_result_dict()

    assert row["encoder_latency"] > 0
    assert summary.get_encoder_power_avg() > 0
    assert row["encoder_memory"] > 0
    assert row["ttft"] == pytest.approx(row["encoder_latency"] + row["context_latency"])

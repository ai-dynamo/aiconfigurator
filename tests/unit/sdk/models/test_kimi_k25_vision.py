# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kimi K2.5 image/video encoder parsing, construction, and runtime tests."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.performance_result import PerformanceResult
from aiconfigurator.sdk.utils import _parse_hf_config_json, get_model_config_from_model_path

pytestmark = pytest.mark.unit

_KIMI_MODELS = ("moonshotai/Kimi-K2.5", "nvidia/Kimi-K2.5-NVFP4")


def _model_config(tp_size: int = 1, *, enable_encoder_dp: bool = True) -> config.ModelConfig:
    return config.ModelConfig(
        tp_size=tp_size,
        attention_dp_size=1,
        moe_tp_size=tp_size,
        moe_ep_size=1,
        enable_encoder_dp=enable_encoder_dp,
    )


@pytest.mark.parametrize("model_id", _KIMI_MODELS)
def test_real_vision_config_is_preserved_alongside_language_config(model_id):
    info = get_model_config_from_model_path(model_id)

    assert "text_config" in info["raw_config"]
    assert info["raw_config"]["vision_config"]["video_attn_type"] == "spatial_temporal"
    assert info["extra_params"] == {
        "v_head_dim": 128,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
    }
    assert isinstance(info["encoder_config"], common.VisionEncoderConfig)


@pytest.mark.parametrize("model_id", _KIMI_MODELS)
def test_kimi_vision_geometry_matches_published_checkpoint(model_id):
    enc = get_model_config_from_model_path(model_id)["encoder_config"]

    assert (enc.depth, enc.hidden_size, enc.num_heads, enc.intermediate_size) == (27, 1152, 16, 4304)
    assert (enc.patch_size, enc.temporal_patch_size, enc.spatial_merge_size) == (14, 1, 2)
    assert enc.projector_dims == ((4608, 4608), (4608, 7168))
    assert enc.out_hidden_size == 7168
    assert enc.partial_rotary_factor == 1.0
    assert enc.video_attention_type == "spatial_temporal"
    assert enc.pool_temporal is True
    assert enc.final_norm is True
    assert enc.projector_pre_norm is True


@pytest.mark.parametrize("model_id", _KIMI_MODELS)
def test_kimi_builds_spatial_temporal_vit_patch_merger_and_projector(model_id):
    model = get_model(model_id, _model_config(), "trtllm")
    names = {op._name for op in model.encoder_ops}

    assert {
        "encoder_patch_embed_gemm",
        "encoder_pos_embed",
        "encoder_qkv_gemm",
        "encoder_attention",
        "encoder_rope_apply",
        "encoder_final_norm",
        "encoder_merger_pre_norm",
        "encoder_patch_merge_pool",
        "encoder_projector_fc0_gemm",
        "encoder_projector_fc1_gemm",
        "encoder_projector_ar",
    } <= names


@pytest.mark.parametrize(
    "model_id,language_modes",
    [
        (
            "moonshotai/Kimi-K2.5",
            (
                common.GEMMQuantMode.bfloat16,
                common.MoEQuantMode.int4_wo,
                common.FMHAQuantMode.bfloat16,
                common.KVCacheQuantMode.bfloat16,
            ),
        ),
        (
            "nvidia/Kimi-K2.5-NVFP4",
            (
                common.GEMMQuantMode.nvfp4,
                common.MoEQuantMode.nvfp4,
                common.FMHAQuantMode.fp8,
                common.KVCacheQuantMode.fp8,
            ),
        ),
    ],
)
def test_language_checkpoint_modes_are_retained_while_encoder_stays_bf16(model_id, language_modes):
    model_cfg = _model_config()
    model = get_model(model_id, model_cfg, "trtllm")

    assert (
        model_cfg.gemm_quant_mode,
        model_cfg.moe_quant_mode,
        model_cfg.fmha_quant_mode,
        model_cfg.kvcache_quant_mode,
    ) == language_modes
    assert model.encoder_config.gemm_quant_mode == "bfloat16"
    assert model.encoder_config.fmha_quant_mode == "bfloat16"
    assert {op._quant_mode for op in model.encoder_ops if hasattr(op, "_quant_mode")} == {common.GEMMQuantMode.bfloat16}
    encoder_attention = next(op for op in model.encoder_ops if op._name == "encoder_attention")
    assert encoder_attention._fmha_quant_mode == common.FMHAQuantMode.bfloat16


def test_model_level_quantization_must_explicitly_exclude_kimi_vision_components():
    raw = deepcopy(get_model_config_from_model_path("nvidia/Kimi-K2.5-NVFP4")["raw_config"])
    raw["quantization_config"]["ignore"] = ["language_model.lm_head"]

    with pytest.raises(ValueError, match="does not explicitly exclude both vision_tower and mm_projector"):
        _parse_hf_config_json(raw)


def test_encoder_parallelism_models_required_communication():
    dp_model = get_model("moonshotai/Kimi-K2.5", _model_config(tp_size=2), "trtllm")
    tp_model = get_model(
        "moonshotai/Kimi-K2.5",
        _model_config(tp_size=2, enable_encoder_dp=False),
        "trtllm",
    )

    dp_names = {op._name for op in dp_model.encoder_ops}
    tp_ops = {op._name: op for op in tp_model.encoder_ops}
    assert "encoder_dp_all_gather" in dp_names
    assert tp_ops["encoder_ar_1"]._tp_size == 2
    assert tp_ops["encoder_ar_2"]._tp_size == 2
    assert tp_ops["encoder_projector_ar"]._tp_size == 2


def test_video_frame_count_must_be_positive():
    enc = get_model_config_from_model_path("moonshotai/Kimi-K2.5")["encoder_config"]
    runtime = config.RuntimeConfig(image_height=448, image_width=448)
    runtime.num_frames_per_visual = 0

    with pytest.raises(ValueError, match="num_frames_per_visual must be positive"):
        BaseBackend._encoder_pre_merge_per_visual(runtime, enc)


@pytest.mark.parametrize("model_id", _KIMI_MODELS)
def test_kimi_image_and_video_runtime_cover_latency_memory_energy_and_ttft(model_id):
    model = get_model(model_id, _model_config(), "trtllm")
    backend = TRTLLMBackend()
    database = SimpleNamespace(
        backend="trtllm",
        version="structural-test",
        system="b200_sxm",
        system_spec={
            "gpu": {"mem_capacity": 1024 * (1 << 30)},
            "misc": {"nccl_mem": {1: 0}, "other_mem": 0},
        },
    )

    def _shape_scaled_result(*_args, **kwargs):
        x = kwargs["x"]
        return PerformanceResult(x / 1_000_000.0, energy=x / 100_000.0, source="structural-test")

    for op in model.context_ops + model.generation_ops + model.encoder_ops:
        op.query = MagicMock(side_effect=_shape_scaled_result)

    image_runtime = config.RuntimeConfig(
        batch_size=1,
        isl=128,
        osl=1,
        image_height=448,
        image_width=448,
        num_frames_per_visual=1,
        engine_step_backend="python",
    )
    video_runtime = config.RuntimeConfig(
        batch_size=1,
        isl=128,
        osl=1,
        image_height=448,
        image_width=448,
        num_frames_per_visual=8,
        engine_step_backend="python",
    )

    enc = model.encoder_config
    assert BaseBackend._encoder_pre_merge_per_visual(image_runtime, enc) == (256, 1024)
    assert BaseBackend._encoder_pre_merge_per_visual(video_runtime, enc) == (256, 8192)
    assert BaseBackend._visual_context_tokens(model, image_runtime) == 256
    assert BaseBackend._visual_context_tokens(model, video_runtime) == 256

    image_summary = backend.run_static(model, database, image_runtime, mode="static_ctx")
    image_attention_call = next(op for op in model.encoder_ops if op._name == "encoder_attention").query.call_args
    video_summary = backend.run_static(model, database, video_runtime, mode="static_ctx")
    video_attention_call = next(op for op in model.encoder_ops if op._name == "encoder_attention").query.call_args

    assert image_attention_call.kwargs["s"] == 1024
    assert video_attention_call.kwargs["s"] == 8192
    assert sum(image_summary.get_encoder_latency_dict().values()) > 0
    assert sum(image_summary.get_encoder_energy_wms_dict().values()) > 0
    assert sum(video_summary.get_encoder_latency_dict().values()) > sum(
        image_summary.get_encoder_latency_dict().values()
    )
    assert sum(video_summary.get_encoder_energy_wms_dict().values()) > sum(
        image_summary.get_encoder_energy_wms_dict().values()
    )
    assert video_summary.get_encoder_memory()["activations"] > image_summary.get_encoder_memory()["activations"]
    assert video_summary.get_result_dict()["ttft"] > image_summary.get_result_dict()["ttft"]

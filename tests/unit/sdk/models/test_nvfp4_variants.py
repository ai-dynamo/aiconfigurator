# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage for the current NVIDIA NVFP4 support-matrix checkpoints."""

import dataclasses

import pytest

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.task_v2 import Task
from aiconfigurator.sdk.utils import get_model_config_from_model_path

pytestmark = pytest.mark.unit


NVFP4_VARIANTS = (
    "nvidia/Qwen3.6-35B-A3B-NVFP4",
    "nvidia/Qwen3.6-27B-NVFP4",
    "nvidia/Qwen3.5-397B-A17B-NVFP4",
    "nvidia/Qwen3.5-122B-A10B-NVFP4",
    "nvidia/Gemma-4-31B-IT-NVFP4",
    "nvidia/Gemma-4-26B-A4B-NVFP4",
    "nvidia/Kimi-K2.6-NVFP4",
    "nvidia/Kimi-K2.7-Code-NVFP4",
    "nvidia/DeepSeek-V4-Flash-NVFP4",
    "nvidia/DeepSeek-V4-Pro-NVFP4",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
    "nvidia/MiniMax-M3-NVFP4",
)


def _model_config():
    return config.ModelConfig(
        tp_size=1,
        attention_dp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
    )


@pytest.mark.parametrize("hf_id", NVFP4_VARIANTS)
def test_nvfp4_variant_is_registered_for_default_matrix(hf_id):
    assert hf_id in common.DefaultHFModels
    assert hf_id in common.SupportMatrixHFModels


@pytest.mark.parametrize("hf_id", NVFP4_VARIANTS)
def test_nvfp4_variant_loads_offline_with_quant_metadata(hf_id, monkeypatch):
    import aiconfigurator.sdk.utils as sdk_utils

    def _no_network(*args, **kwargs):
        raise AssertionError("network path reached")

    monkeypatch.setattr(sdk_utils, "_download_hf_config", _no_network, raising=False)
    sdk_utils._load_model_config_from_model_path.cache_clear()
    loaded = get_model_config_from_model_path(hf_id)

    assert loaded["architecture"] in common.ARCHITECTURE_TO_MODEL_FAMILY
    raw_config = loaded["raw_config"]
    assert raw_config.get("hf_quant_config") or raw_config.get("quantization_config")
    sdk_utils._load_model_config_from_model_path.cache_clear()


@pytest.mark.parametrize(
    "hf_id,gemm_mode,moe_mode,kv_mode,fmha_mode",
    [
        (
            "nvidia/Qwen3.6-35B-A3B-NVFP4",
            common.GEMMQuantMode.fp8_static,
            common.MoEQuantMode.w4a16_nvfp4,
            common.KVCacheQuantMode.fp8,
            common.FMHAQuantMode.fp8,
        ),
        (
            "nvidia/Qwen3.6-27B-NVFP4",
            common.GEMMQuantMode.fp8_static,
            common.MoEQuantMode.bfloat16,
            common.KVCacheQuantMode.fp8,
            common.FMHAQuantMode.fp8,
        ),
        *[
            (
                hf_id,
                common.GEMMQuantMode.nvfp4,
                common.MoEQuantMode.nvfp4,
                common.KVCacheQuantMode.fp8,
                common.FMHAQuantMode.fp8,
            )
            for hf_id in (
                "nvidia/Qwen3.5-397B-A17B-NVFP4",
                "nvidia/Qwen3.5-122B-A10B-NVFP4",
                "nvidia/Gemma-4-31B-IT-NVFP4",
                "nvidia/Gemma-4-26B-A4B-NVFP4",
                "nvidia/Kimi-K2.6-NVFP4",
                "nvidia/Kimi-K2.7-Code-NVFP4",
                "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
            )
        ],
        *[
            (
                hf_id,
                common.GEMMQuantMode.fp8_block,
                common.MoEQuantMode.nvfp4,
                common.KVCacheQuantMode.fp8,
                common.FMHAQuantMode.bfloat16,
            )
            for hf_id in (
                "nvidia/DeepSeek-V4-Flash-NVFP4",
                "nvidia/DeepSeek-V4-Pro-NVFP4",
            )
        ],
        (
            "nvidia/MiniMax-M3-NVFP4",
            # MXFP8 prices as fp8_block (owner decision 2026-08-09; DSV4
            # precedent, and the shipped M3 MSA tables carry the fp8_block
            # gemm tier this lane resolves to — see helpers.py).
            common.GEMMQuantMode.fp8_block,
            common.MoEQuantMode.nvfp4,
            common.KVCacheQuantMode.bfloat16,
            common.FMHAQuantMode.bfloat16,
        ),
    ],
)
def test_nvfp4_variant_infers_checkpoint_quant_modes(hf_id, gemm_mode, moe_mode, kv_mode, fmha_mode):
    model_config = _model_config()

    get_model(hf_id, model_config, backend_name="trtllm")

    assert model_config.gemm_quant_mode == gemm_mode
    assert model_config.moe_quant_mode == moe_mode
    assert model_config.kvcache_quant_mode == kv_mode
    assert model_config.fmha_quant_mode == fmha_mode


@pytest.mark.parametrize(
    "hf_id,ffn_names",
    [
        (
            "nvidia/Qwen3.6-35B-A3B-NVFP4",
            ("context_gdn_shared_gate_up_gemm", "generation_full_shared_down_gemm"),
        ),
        (
            "nvidia/Qwen3.6-27B-NVFP4",
            ("context_gdn_gate_ffn1_gemm", "generation_full_ffn2_gemm"),
        ),
    ],
)
def test_qwen36_uses_fp8_projections_and_w4a16_nvfp4_ffn(hf_id, ffn_names):
    model = get_model(hf_id, _model_config(), backend_name="trtllm")
    by_name = {op._name: op for op in model.context_ops + model.generation_ops}

    for name in ("context_gdn_in_proj_gemm", "context_qkv_gemm", "generation_proj_gemm"):
        assert by_name[name]._quant_mode == common.GEMMQuantMode.fp8_static
    for name in ffn_names:
        assert by_name[name]._quant_mode == common.GEMMQuantMode.w4a16_nvfp4
    if hf_id == "nvidia/Qwen3.6-35B-A3B-NVFP4":
        assert by_name["context_gdn_moe"]._quant_mode == common.MoEQuantMode.w4a16_nvfp4


def test_qwen36_task_preserves_inferred_provenance_for_mixed_precision_split():
    task = Task(
        model_path="nvidia/Qwen3.6-27B-NVFP4",
        system_name="gb300",
        backend_name="sglang",
        backend_version="0.5.16",  # next slot; 0.5.12-era rows reach it via backward fill
        total_gpus=32,
    )
    model_config = task.build_model_config(role="agg")
    model_config = dataclasses.replace(model_config, tp_size=1)
    model = get_model(task.model_path, model_config, task.backend_name)
    by_name = {op._name: op for op in model.context_ops + model.generation_ops}

    assert model_config._gemm_quant_mode_is_explicit is False
    assert by_name["context_gdn_in_proj_gemm"]._quant_mode == common.GEMMQuantMode.fp8_static
    assert by_name["context_gdn_gate_ffn1_gemm"]._quant_mode == common.GEMMQuantMode.w4a16_nvfp4


def test_model_config_provenance_does_not_shift_positional_quant_modes():
    model_config = config.ModelConfig(
        2,
        1,
        common.GEMMQuantMode.fp8,
        common.MoEQuantMode.nvfp4,
        common.KVCacheQuantMode.fp8,
        common.FMHAQuantMode.fp8,
    )

    assert model_config.gemm_quant_mode == common.GEMMQuantMode.fp8
    assert model_config.moe_quant_mode == common.MoEQuantMode.nvfp4
    assert model_config.kvcache_quant_mode == common.KVCacheQuantMode.fp8
    assert model_config.fmha_quant_mode == common.FMHAQuantMode.fp8
    assert model_config._gemm_quant_mode_is_explicit is None


def test_qwen36_task_preserves_explicit_gemm_override():
    task = Task(
        model_path="nvidia/Qwen3.6-27B-NVFP4",
        system_name="gb300",
        backend_name="sglang",
        backend_version="0.5.16",  # next slot; 0.5.12-era rows reach it via backward fill
        total_gpus=32,
        gemm_quant_mode=common.GEMMQuantMode.fp8_static,
    )
    model_config = task.build_model_config(role="agg")
    model_config = dataclasses.replace(model_config, tp_size=1)
    model = get_model(task.model_path, model_config, task.backend_name)
    by_name = {op._name: op for op in model.context_ops + model.generation_ops}

    assert model_config._gemm_quant_mode_is_explicit is True
    assert by_name["context_gdn_gate_ffn1_gemm"]._quant_mode == common.GEMMQuantMode.fp8_static


def test_qwen36_afd_prefill_preserves_explicit_gemm_override():
    task = Task(
        serving_mode="afd",
        model_path="nvidia/Qwen3.6-27B-NVFP4",
        system_name="b200_sxm",
        backend_name="trtllm",
        total_gpus=16,
        afd_combined_with_pd=True,
        gemm_quant_mode=common.GEMMQuantMode.fp8_static,
    )
    model_config = task.sweep_afd_kwargs(database=None)["prefill_model_config"]
    model = get_model(task.model_path, model_config, task.backend_name)
    by_name = {op._name: op for op in model.context_ops + model.generation_ops}

    assert model_config._gemm_quant_mode_is_explicit is True
    assert by_name["context_gdn_gate_ffn1_gemm"]._quant_mode == common.GEMMQuantMode.fp8_static


@pytest.mark.parametrize(
    "hf_id,ffn_name",
    [
        ("nvidia/Qwen3.6-27B-NVFP4", "generation_full_ffn2_gemm"),
        ("nvidia/Qwen3.6-35B-A3B-NVFP4", "generation_full_shared_down_gemm"),
    ],
)
@pytest.mark.parametrize("gemm_mode", [common.GEMMQuantMode.bfloat16, common.GEMMQuantMode.fp8_static])
def test_qwen36_preserves_explicit_global_gemm_override(hf_id, ffn_name, gemm_mode):
    model_config = _model_config()
    model_config.gemm_quant_mode = gemm_mode

    model = get_model(hf_id, model_config, backend_name="trtllm")
    by_name = {op._name: op for op in model.context_ops + model.generation_ops}

    for name in ("context_gdn_in_proj_gemm", "context_qkv_gemm", ffn_name):
        assert by_name[name]._quant_mode == gemm_mode


def test_w4a16_nvfp4_uses_scale_aware_weight_only_profile():
    for mode in (common.GEMMQuantMode.w4a16_nvfp4, common.MoEQuantMode.w4a16_nvfp4):
        assert mode.value.memory == 9 / 16
        assert mode.value.compute == 1
        assert mode.value.compute_dtype == "bfloat16"


def test_qwen35_nvfp4_exclusions_keep_attention_and_shared_experts_bf16():
    model = get_model("nvidia/Qwen3.5-122B-A10B-NVFP4", _model_config(), backend_name="trtllm")
    by_name = {op._name: op for op in model.context_ops + model.generation_ops}

    assert by_name["context_gdn_in_proj_gemm"]._quant_mode == common.GEMMQuantMode.bfloat16
    assert by_name["context_qkv_gemm"]._quant_mode == common.GEMMQuantMode.bfloat16
    assert by_name["context_gdn_shared_gate_up_gemm"]._quant_mode == common.GEMMQuantMode.bfloat16
    assert by_name["context_gdn_moe"]._quant_mode == common.MoEQuantMode.nvfp4


def test_gemma4_exclusions_keep_attention_and_mlp_bf16():
    model = get_model("nvidia/Gemma-4-26B-A4B-NVFP4", _model_config(), backend_name="trtllm")
    by_name = {op._name: op for op in model.context_ops + model.generation_ops}

    assert by_name["context_swa_qkv_gemm"]._quant_mode == common.GEMMQuantMode.bfloat16
    assert by_name["context_swa_shared_mlp_gate_up_gemm"]._quant_mode == common.GEMMQuantMode.bfloat16
    assert by_name["context_swa_moe"]._quant_mode == common.MoEQuantMode.bfloat16


def test_kimi_k26_exclusions_keep_attention_and_shared_experts_bf16():
    model = get_model("nvidia/Kimi-K2.6-NVFP4", _model_config(), backend_name="vllm")
    by_name = {op._name: op for op in model.context_ops + model.generation_ops}
    attention_block = by_name["context_mla_block"]

    assert attention_block._primary._gemm_quant_mode == common.GEMMQuantMode.bfloat16
    assert all(
        op._quant_mode == common.GEMMQuantMode.bfloat16
        for op in attention_block._fallback
        if hasattr(op, "_quant_mode")
    )
    assert by_name["context_shared_gate_up_gemm"]._quant_mode == common.GEMMQuantMode.bfloat16
    assert by_name["context_moe"]._quant_mode == common.MoEQuantMode.nvfp4


def test_minimax_m3_uses_mxfp8_lane_for_non_experts_and_nvfp4_for_experts():
    model = get_model("nvidia/MiniMax-M3-NVFP4", _model_config(), backend_name="trtllm")
    by_name = {op._name: op for op in model.context_ops + model.generation_ops}

    # MXFP8 non-expert lane prices as fp8_block (owner decision 2026-08-09).
    assert by_name["context_shared_gate_up_gemm"]._quant_mode == common.GEMMQuantMode.fp8_block
    assert by_name["context_moe"]._quant_mode == common.MoEQuantMode.nvfp4


@pytest.mark.parametrize(
    "hf_id",
    ("nvidia/DeepSeek-V4-Flash-NVFP4", "nvidia/DeepSeek-V4-Pro-NVFP4"),
)
def test_dsv4_nvfp4_experts_skip_native_mxfp4_backend_remap(hf_id):
    from aiconfigurator.sdk.models.helpers import resolve_dsv4_moe_arch_mode

    assert resolve_dsv4_moe_arch_mode(hf_id, "b200_sxm", "sglang") is None
    assert resolve_dsv4_moe_arch_mode(hf_id, "h200_sxm", "sglang") is None


@pytest.mark.parametrize(
    "hf_id",
    ("nvidia/DeepSeek-V4-Flash-NVFP4", "nvidia/DeepSeek-V4-Pro-NVFP4"),
)
def test_dsv4_nvfp4_experts_preserve_fp8_block_nonexpert_lane(hf_id):
    model_config = _model_config()
    model = get_model(hf_id, model_config, backend_name="trtllm")

    assert model_config.gemm_quant_mode == common.GEMMQuantMode.fp8_block
    assert model_config.moe_quant_mode == common.MoEQuantMode.nvfp4
    module_modes = {
        op._gemm_quant_mode for op in model.context_ops + model.generation_ops if hasattr(op, "_gemm_quant_mode")
    }
    assert module_modes == {common.GEMMQuantMode.fp8_block}

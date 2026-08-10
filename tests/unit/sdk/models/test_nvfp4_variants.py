# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage for the current NVIDIA NVFP4 support-matrix checkpoints."""

import pytest

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.models import get_model
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
            common.MoEQuantMode.nvfp4,
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
                common.GEMMQuantMode.nvfp4,
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
            common.GEMMQuantMode.fp8,
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
            ("context_gdn_shared_up_gemm", "generation_full_shared_down_gemm"),
        ),
        (
            "nvidia/Qwen3.6-27B-NVFP4",
            ("context_gdn_gate_ffn1_gemm", "generation_full_ffn2_gemm"),
        ),
    ],
)
def test_qwen36_uses_fp8_projections_and_nvfp4_ffn(hf_id, ffn_names):
    model = get_model(hf_id, _model_config(), backend_name="trtllm")
    by_name = {op._name: op for op in model.context_ops + model.generation_ops}

    for name in ("context_gdn_in_proj_gemm", "context_qkv_gemm", "generation_proj_gemm"):
        assert by_name[name]._quant_mode == common.GEMMQuantMode.fp8_static
    for name in ffn_names:
        assert by_name[name]._quant_mode == common.GEMMQuantMode.nvfp4


def test_qwen36_preserves_explicit_global_gemm_override():
    model_config = _model_config()
    model_config.gemm_quant_mode = common.GEMMQuantMode.bfloat16

    model = get_model("nvidia/Qwen3.6-27B-NVFP4", model_config, backend_name="trtllm")
    by_name = {op._name: op for op in model.context_ops + model.generation_ops}

    for name in ("context_gdn_in_proj_gemm", "context_qkv_gemm", "generation_full_ffn2_gemm"):
        assert by_name[name]._quant_mode == common.GEMMQuantMode.bfloat16


def test_minimax_m3_uses_mxfp8_lane_for_non_experts_and_nvfp4_for_experts():
    model = get_model("nvidia/MiniMax-M3-NVFP4", _model_config(), backend_name="trtllm")
    by_name = {op._name: op for op in model.context_ops + model.generation_ops}

    assert by_name["context_shared_gate_up_gemm"]._quant_mode == common.GEMMQuantMode.fp8
    assert by_name["context_moe"]._quant_mode == common.MoEQuantMode.nvfp4


@pytest.mark.parametrize(
    "hf_id",
    ("nvidia/DeepSeek-V4-Flash-NVFP4", "nvidia/DeepSeek-V4-Pro-NVFP4"),
)
def test_dsv4_nvfp4_experts_skip_native_mxfp4_backend_remap(hf_id):
    from aiconfigurator.sdk.models.helpers import resolve_dsv4_moe_arch_mode

    assert resolve_dsv4_moe_arch_mode(hf_id, "b200_sxm", "sglang") is None
    assert resolve_dsv4_moe_arch_mode(hf_id, "h200_sxm", "sglang") is None

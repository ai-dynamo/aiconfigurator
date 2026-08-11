# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.config import ModelConfig
from aiconfigurator_core.sdk.models.helpers import resolve_nvfp4_for_system

pytestmark = pytest.mark.unit


def _model_config(**overrides) -> ModelConfig:
    defaults = dict(tp_size=8, moe_tp_size=1, moe_ep_size=8)
    defaults.update(overrides)
    return ModelConfig(**defaults)


def test_nvfp4_remapped_to_nvfp4_wo_on_hopper_vllm():
    """vLLM >= 0.21.0 supports sw-dequant on Hopper → remap fires."""
    mc = _model_config(
        gemm_quant_mode=common.GEMMQuantMode.nvfp4,
        moe_quant_mode=common.MoEQuantMode.nvfp4,
    )
    resolve_nvfp4_for_system(mc, "h100_sxm", backend_name="vllm", version="0.21.0")
    assert mc.gemm_quant_mode == common.GEMMQuantMode.nvfp4_wo
    assert mc.moe_quant_mode == common.MoEQuantMode.nvfp4_wo


def test_nvfp4_not_remapped_on_hopper_trtllm():
    """TRT-LLM has no Hopper NVFP4 support → remap must NOT fire."""
    mc = _model_config(
        gemm_quant_mode=common.GEMMQuantMode.nvfp4,
        moe_quant_mode=common.MoEQuantMode.nvfp4,
    )
    resolve_nvfp4_for_system(mc, "h100_sxm", backend_name="trtllm", version="1.3.0rc20")
    assert mc.gemm_quant_mode == common.GEMMQuantMode.nvfp4
    assert mc.moe_quant_mode == common.MoEQuantMode.nvfp4


def test_nvfp4_not_remapped_on_hopper_old_vllm():
    """vLLM < 0.21.0 does not support sw-dequant on Hopper → remap must NOT fire."""
    mc = _model_config(
        gemm_quant_mode=common.GEMMQuantMode.nvfp4,
        moe_quant_mode=common.MoEQuantMode.nvfp4,
    )
    resolve_nvfp4_for_system(mc, "h100_sxm", backend_name="vllm", version="0.19.0")
    assert mc.gemm_quant_mode == common.GEMMQuantMode.nvfp4
    assert mc.moe_quant_mode == common.MoEQuantMode.nvfp4


def test_nvfp4_unchanged_on_blackwell():
    mc = _model_config(
        gemm_quant_mode=common.GEMMQuantMode.nvfp4,
        moe_quant_mode=common.MoEQuantMode.nvfp4,
    )
    resolve_nvfp4_for_system(mc, "b200_sxm", backend_name="trtllm", version="1.3.0rc20")
    assert mc.gemm_quant_mode == common.GEMMQuantMode.nvfp4
    assert mc.moe_quant_mode == common.MoEQuantMode.nvfp4


def test_bfloat16_unchanged_on_hopper():
    mc = _model_config(
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        moe_quant_mode=common.MoEQuantMode.bfloat16,
    )
    resolve_nvfp4_for_system(mc, "h100_sxm", backend_name="vllm", version="0.21.0")
    assert mc.gemm_quant_mode == common.GEMMQuantMode.bfloat16
    assert mc.moe_quant_mode == common.MoEQuantMode.bfloat16


def test_nvfp4_wo_memory_matches_nvfp4():
    assert common.GEMMQuantMode.nvfp4_wo.value.memory == common.GEMMQuantMode.nvfp4.value.memory
    assert common.MoEQuantMode.nvfp4_wo.value.memory == common.MoEQuantMode.nvfp4.value.memory


def test_nvfp4_wo_compute_matches_bfloat16():
    assert common.GEMMQuantMode.nvfp4_wo.value.compute == common.GEMMQuantMode.bfloat16.value.compute
    assert common.MoEQuantMode.nvfp4_wo.value.compute == common.MoEQuantMode.bfloat16.value.compute

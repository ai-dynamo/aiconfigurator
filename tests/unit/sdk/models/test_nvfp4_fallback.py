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


def test_nvfp4_remapped_to_nvfp4_wo_on_hopper():
    """nvfp4 → nvfp4_wo unconditionally on non-Blackwell (hardware fact)."""
    mc = _model_config(
        gemm_quant_mode=common.GEMMQuantMode.nvfp4,
        moe_quant_mode=common.MoEQuantMode.nvfp4,
    )
    resolve_nvfp4_for_system(mc, "h100_sxm")
    assert mc.gemm_quant_mode == common.GEMMQuantMode.nvfp4_wo
    assert mc.moe_quant_mode == common.MoEQuantMode.nvfp4_wo


def test_nvfp4_unchanged_on_blackwell():
    """Blackwell has native FP4 TC — nvfp4 stays as-is."""
    mc = _model_config(
        gemm_quant_mode=common.GEMMQuantMode.nvfp4,
        moe_quant_mode=common.MoEQuantMode.nvfp4,
    )
    resolve_nvfp4_for_system(mc, "b200_sxm")
    assert mc.gemm_quant_mode == common.GEMMQuantMode.nvfp4
    assert mc.moe_quant_mode == common.MoEQuantMode.nvfp4


def test_bfloat16_unchanged_on_hopper():
    mc = _model_config(
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        moe_quant_mode=common.MoEQuantMode.bfloat16,
    )
    resolve_nvfp4_for_system(mc, "h100_sxm")
    assert mc.gemm_quant_mode == common.GEMMQuantMode.bfloat16
    assert mc.moe_quant_mode == common.MoEQuantMode.bfloat16


def test_nvfp4_wo_memory_matches_nvfp4():
    assert common.GEMMQuantMode.nvfp4_wo.value.memory == common.GEMMQuantMode.nvfp4.value.memory
    assert common.MoEQuantMode.nvfp4_wo.value.memory == common.MoEQuantMode.nvfp4.value.memory


def test_nvfp4_wo_compute_matches_bfloat16():
    assert common.GEMMQuantMode.nvfp4_wo.value.compute == common.GEMMQuantMode.bfloat16.value.compute
    assert common.MoEQuantMode.nvfp4_wo.value.compute == common.MoEQuantMode.bfloat16.value.compute


def test_static_estimate_calls_resolve_nvfp4(monkeypatch):
    """The static estimate path must call resolve_nvfp4_for_system.

    Without the remap, nvfp4 on non-Blackwell hits MissingSystemFlopsError
    on fp4_tc_flops. This verifies _run_static_estimate has the call.
    """
    import inspect

    import aiconfigurator.cli.api as api

    source = inspect.getsource(api._run_static_estimate)
    assert "resolve_nvfp4_for_system" in source, "_run_static_estimate must call resolve_nvfp4_for_system"

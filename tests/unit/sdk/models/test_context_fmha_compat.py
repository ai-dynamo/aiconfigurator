# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the DeepSeek-V3 / Kimi context-role FMHA rule.

NVBug 6401867: the single-point ``cli estimate`` / AFD path resolved fp8 FMHA
for DeepSeek-V3 context MLA (no perf data) and crashed with a
PerfDataNotAvailableError traceback. ``resolve_context_fmha_compat`` mirrors the
role-aware downgrade that ``task_v2`` (the sweep path) already applies.
"""

import pytest

import aiconfigurator.sdk.models.helpers as helpers
from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.models import resolve_context_fmha_compat

pytestmark = pytest.mark.unit

# DeepSeek-V3 ships fp8_block weights → inference resolves FMHA to fp8.
_V3_FP8_RAW = {"quant_algo": "fp8_block"}
_V3_BF16_RAW = {"quant_algo": None}


@pytest.fixture
def fake_model_info(monkeypatch):
    """Patch _get_model_info so the helper resolves a chosen (arch, raw_config)."""

    def _install(architecture, raw_config):
        monkeypatch.setattr(
            helpers,
            "_get_model_info",
            lambda _model_path: {"architecture": architecture, "raw_config": raw_config},
        )

    return _install


def _mc(fmha=None):
    return config.ModelConfig(fmha_quant_mode=fmha)


def test_context_role_inferred_fp8_downgrades_to_bf16(fake_model_info):
    """Auto-inferred fp8 FMHA on a V3 context role falls back to bf16."""
    fake_model_info("DeepseekV3ForCausalLM", _V3_FP8_RAW)
    mc = _mc(fmha=None)
    resolve_context_fmha_compat(mc, "deepseek-ai/DeepSeek-V3", is_context_role=True)
    assert mc.fmha_quant_mode == common.FMHAQuantMode.bfloat16


def test_context_role_explicit_fp8_raises(fake_model_info):
    """Explicit fp8 FMHA on a V3 context role raises a concise error, no traceback."""
    fake_model_info("DeepseekV3ForCausalLM", _V3_FP8_RAW)
    mc = _mc(fmha=common.FMHAQuantMode.fp8)
    with pytest.raises(ValueError, match="does not support fp8 FMHA"):
        resolve_context_fmha_compat(mc, "deepseek-ai/DeepSeek-V3", is_context_role=True)


def test_generation_role_keeps_fp8(fake_model_info):
    """Generation-only roles (static_gen / decode) keep fp8 — no downgrade, no error."""
    fake_model_info("DeepseekV3ForCausalLM", _V3_FP8_RAW)
    # Explicit fp8 must NOT raise for a gen role.
    mc = _mc(fmha=common.FMHAQuantMode.fp8)
    resolve_context_fmha_compat(mc, "deepseek-ai/DeepSeek-V3", is_context_role=False)
    assert mc.fmha_quant_mode == common.FMHAQuantMode.fp8
    # Auto-inferred case: helper leaves it for get_model to resolve to fp8.
    mc_auto = _mc(fmha=None)
    resolve_context_fmha_compat(mc_auto, "deepseek-ai/DeepSeek-V3", is_context_role=False)
    assert mc_auto.fmha_quant_mode is None


def test_context_role_explicit_bf16_is_untouched(fake_model_info):
    """An explicit non-fp8 request is respected (no error, no change)."""
    fake_model_info("KimiK25ForConditionalGeneration", _V3_FP8_RAW)
    mc = _mc(fmha=common.FMHAQuantMode.bfloat16)
    resolve_context_fmha_compat(mc, "moonshotai/Kimi-K2.5", is_context_role=True)
    assert mc.fmha_quant_mode == common.FMHAQuantMode.bfloat16


def test_non_v3_architecture_is_untouched(fake_model_info):
    """Architectures without the context-fp8 limitation are left to get_model."""
    fake_model_info("Qwen3MoeForCausalLM", _V3_FP8_RAW)
    mc = _mc(fmha=None)
    resolve_context_fmha_compat(mc, "Qwen/Qwen3-235B", is_context_role=True)
    assert mc.fmha_quant_mode is None


def test_bf16_v3_checkpoint_needs_no_downgrade(fake_model_info):
    """A bf16 V3 checkpoint infers no fp8, so the helper is a no-op."""
    fake_model_info("DeepseekV3ForCausalLM", _V3_BF16_RAW)
    mc = _mc(fmha=None)
    resolve_context_fmha_compat(mc, "deepseek-ai/DeepSeek-V3-bf16", is_context_role=True)
    assert mc.fmha_quant_mode is None

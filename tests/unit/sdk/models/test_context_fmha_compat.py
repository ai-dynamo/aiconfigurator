# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the estimate-path data-driven context-FMHA guard.

NVBug 6401867: the single-point ``cli estimate`` / AFD path resolved fp8 FMHA
for DeepSeek-V3 context MLA (no perf data) and crashed with a
PerfDataNotAvailableError traceback. ``resolve_context_fmha_by_data`` mirrors
the resolve-time data fallback that ``task_v2`` (the sweep path) applies: the
perf DB's fmha-keyed context table decides, not a hand-written model list.
"""

from types import SimpleNamespace

import pytest

import aiconfigurator.sdk.models.helpers as helpers
from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.models import resolve_context_fmha_by_data

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


def _db(**supported):
    """Database stub exposing only supported_quant_mode."""
    return SimpleNamespace(supported_quant_mode=supported)


# V3 on trtllm consults the "context_mla" op key.
_BF16_ONLY_DB = _db(context_mla=["bfloat16"])
_FP8_DB = _db(context_mla=["bfloat16", "fp8"])
_NO_INFO_DB = _db()


def test_context_role_inferred_fp8_downgrades_to_bf16(fake_model_info, caplog):
    """Auto-inferred fp8 FMHA with a bf16-only context table falls back to bf16."""
    fake_model_info("DeepseekV3ForCausalLM", _V3_FP8_RAW)
    mc = _mc(fmha=None)
    with caplog.at_level("WARNING"):
        resolve_context_fmha_by_data(mc, "deepseek-ai/DeepSeek-V3", _BF16_ONLY_DB, "trtllm", is_context_role=True)
    assert mc.fmha_quant_mode == common.FMHAQuantMode.bfloat16
    assert any("falling back to bfloat16" in r.message for r in caplog.records)


def test_context_role_inferred_fp8_kept_when_data_exists(fake_model_info):
    """With an fp8 slice in the context table, the inference survives (left to get_model)."""
    fake_model_info("DeepseekV3ForCausalLM", _V3_FP8_RAW)
    mc = _mc(fmha=None)
    resolve_context_fmha_by_data(mc, "deepseek-ai/DeepSeek-V3", _FP8_DB, "trtllm", is_context_role=True)
    assert mc.fmha_quant_mode is None


def test_context_role_explicit_fp8_raises(fake_model_info):
    """Explicit fp8 FMHA with no fp8 slice raises a concise error, no traceback."""
    fake_model_info("DeepseekV3ForCausalLM", _V3_FP8_RAW)
    mc = _mc(fmha=common.FMHAQuantMode.fp8)
    with pytest.raises(ValueError, match="has no 'context_mla' perf data"):
        resolve_context_fmha_by_data(mc, "deepseek-ai/DeepSeek-V3", _BF16_ONLY_DB, "trtllm", is_context_role=True)


def test_generation_role_keeps_fp8(fake_model_info):
    """Generation-only roles (static_gen / decode) keep fp8 — no downgrade, no error."""
    fake_model_info("DeepseekV3ForCausalLM", _V3_FP8_RAW)
    # Explicit fp8 must NOT raise for a gen role.
    mc = _mc(fmha=common.FMHAQuantMode.fp8)
    resolve_context_fmha_by_data(mc, "deepseek-ai/DeepSeek-V3", _BF16_ONLY_DB, "trtllm", is_context_role=False)
    assert mc.fmha_quant_mode == common.FMHAQuantMode.fp8
    # Auto-inferred case: helper leaves it for get_model to resolve to fp8.
    mc_auto = _mc(fmha=None)
    resolve_context_fmha_by_data(mc_auto, "deepseek-ai/DeepSeek-V3", _BF16_ONLY_DB, "trtllm", is_context_role=False)
    assert mc_auto.fmha_quant_mode is None


def test_context_role_explicit_bf16_is_untouched(fake_model_info):
    """An explicit non-fp8 request is respected (no error, no change)."""
    fake_model_info("KimiK25ForConditionalGeneration", _V3_FP8_RAW)
    mc = _mc(fmha=common.FMHAQuantMode.bfloat16)
    resolve_context_fmha_by_data(mc, "moonshotai/Kimi-K2.5", _BF16_ONLY_DB, "trtllm", is_context_role=True)
    assert mc.fmha_quant_mode == common.FMHAQuantMode.bfloat16


def test_no_db_information_is_untouched(fake_model_info):
    """Without table info (synthetic what-if system), the inference is kept."""
    fake_model_info("DeepseekV3ForCausalLM", _V3_FP8_RAW)
    mc = _mc(fmha=None)
    resolve_context_fmha_by_data(mc, "deepseek-ai/DeepSeek-V3", _NO_INFO_DB, "trtllm", is_context_role=True)
    assert mc.fmha_quant_mode is None
    # Explicit fp8 is also left for the query path to surface (no early raise).
    mc_explicit = _mc(fmha=common.FMHAQuantMode.fp8)
    resolve_context_fmha_by_data(mc_explicit, "deepseek-ai/DeepSeek-V3", _NO_INFO_DB, "trtllm", is_context_role=True)
    assert mc_explicit.fmha_quant_mode == common.FMHAQuantMode.fp8


def test_bf16_v3_checkpoint_needs_no_downgrade(fake_model_info):
    """A bf16 V3 checkpoint infers no fp8, so the helper is a no-op."""
    fake_model_info("DeepseekV3ForCausalLM", _V3_BF16_RAW)
    mc = _mc(fmha=None)
    resolve_context_fmha_by_data(mc, "deepseek-ai/DeepSeek-V3-bf16", _BF16_ONLY_DB, "trtllm", is_context_role=True)
    assert mc.fmha_quant_mode is None


def test_generic_arch_consults_context_attention(fake_model_info, caplog):
    """Non-MLA families consult the generic context_attention table."""
    fake_model_info("Qwen3MoeForCausalLM", {"quant_algo": "fp8"})
    db = _db(context_attention=["bfloat16"])
    mc = _mc(fmha=None)
    with caplog.at_level("WARNING"):
        resolve_context_fmha_by_data(mc, "Qwen/Qwen3-235B", db, "sglang", is_context_role=True)
    assert mc.fmha_quant_mode == common.FMHAQuantMode.bfloat16
    assert any("context_attention" in r.message for r in caplog.records)

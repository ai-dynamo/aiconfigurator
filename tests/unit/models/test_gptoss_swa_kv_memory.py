# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""gpt-oss KV-cache memory must honor the hybrid SWA layout.

gpt-oss-120b: 36 layers, half banded at window 128 and half global, 8 KV heads,
head_size 64. The timing path (MOEModel's GptOssForCausalLM branch) already
splits layers this way; these tests pin the memory path to the same layout so
the two cannot drift apart again (the drift produced a ~2x KV overcharge and
false OOMs at long ISL).
"""

import pytest

from aiconfigurator_core.sdk import config as sdk_config
from aiconfigurator_core.sdk.models import get_model


GPTOSS = "openai/gpt-oss-120b"

# Model geometry (from the HF config, mirrored in AIC's model info).
LAYERS = 36
KV_HEADS = 8
HEAD_SIZE = 64
WINDOW = 128
NUM_SWA = LAYERS // 2
NUM_GLOBAL = LAYERS - NUM_SWA


def _model(tp_size: int = 1, kvcache_bytes_per_elem: int = 1):
    model_config = sdk_config.ModelConfig(tp_size=tp_size, moe_tp_size=1, moe_ep_size=1)
    if kvcache_bytes_per_elem == 1:
        model_config.kvcache_quant_mode = sdk_config.common.KVCacheQuantMode.fp8
    model = get_model(GPTOSS, model_config, "vllm")
    return model


def _expected_bytes(seq_len: int, bytes_per_elem: int = 1, tp: int = 1) -> float:
    kv_per_gpu = (KV_HEADS + tp - 1) // tp
    per_layer_token = kv_per_gpu * HEAD_SIZE * 2 * bytes_per_elem
    return per_layer_token * (NUM_SWA * min(seq_len, WINDOW) + NUM_GLOBAL * seq_len)


def test_gptoss_kv_bytes_long_sequence_pins_hybrid_layout():
    """At ISL+OSL = 65,936 (the 64k agentic recipe shape), fp8 KV, TP1."""
    model = _model()
    seq_len = 65_936
    got = model.get_kvcache_bytes_per_sequence(seq_len)
    expected = _expected_bytes(seq_len)
    assert got == pytest.approx(expected, rel=1e-9)
    # Regression guard: the linear (all-layers-full-context) value is ~2x.
    linear = seq_len * LAYERS * 2 * KV_HEADS * HEAD_SIZE
    assert got < 0.52 * linear


def test_gptoss_kv_bytes_below_window_matches_linear():
    """Below the 128-token window the hybrid and linear formulas agree."""
    model = _model()
    seq_len = 100
    got = model.get_kvcache_bytes_per_sequence(seq_len)
    linear = seq_len * LAYERS * 2 * KV_HEADS * HEAD_SIZE
    assert got == pytest.approx(linear, rel=1e-9)


def test_gptoss_kv_max_tokens_inverts_the_piecewise_curve():
    """Capacity inverse must follow the window-capped curve, not the seq_len=1 slope."""
    model = _model()
    seq_len = 50_000
    budget = model.get_kvcache_bytes_per_sequence(seq_len)
    max_tokens = model.get_kvcache_max_tokens(budget)
    assert abs(max_tokens - seq_len) <= 1


def test_gptoss_deepep_path_also_gets_hybrid_layout():
    """SGLangEPMOEModel subclasses BaseModel directly and must not regress to linear KV."""
    model_config = sdk_config.ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1)
    model_config.kvcache_quant_mode = sdk_config.common.KVCacheQuantMode.fp8
    model_config.moe_backend = "deepep_moe"
    model = get_model(GPTOSS, model_config, "sglang")
    assert type(model).__name__ == "SGLangEPMOEModel"
    seq_len = 65_936
    got = model.get_kvcache_bytes_per_sequence(seq_len)
    assert got == pytest.approx(_expected_bytes(seq_len), rel=1e-9)
    budget = model.get_kvcache_bytes_per_sequence(50_000)
    assert abs(model.get_kvcache_max_tokens(budget) - 50_000) <= 1

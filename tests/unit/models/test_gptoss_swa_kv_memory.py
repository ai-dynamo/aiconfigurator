# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""gpt-oss KV-cache memory must honor the hybrid SWA layout.

gpt-oss-120b: 36 layers, half banded at window 128 and half global, 8 KV heads,
head_size 64. The timing path (MOEModel's GptOssForCausalLM branch) already
splits layers this way; these tests pin the memory path to the same layout so
the two cannot drift apart again (the drift produced a ~2x KV overcharge and
false OOMs at long ISL).
"""

from types import SimpleNamespace

import pytest

from aiconfigurator_core.sdk import config as sdk_config
from aiconfigurator_core.sdk.backends.factory import get_backend
from aiconfigurator_core.sdk.models import get_model

GPTOSS = "openai/gpt-oss-120b"

# Model geometry (from the HF config, mirrored in AIC's model info).
LAYERS = 36
KV_HEADS = 8
HEAD_SIZE = 64
WINDOW = 128
NUM_SWA = LAYERS // 2
NUM_GLOBAL = LAYERS - NUM_SWA


def _model(tp_size: int = 1, kvcache_bytes_per_elem: int = 1, nextn: int = 0):
    model_config = sdk_config.ModelConfig(tp_size=tp_size, moe_tp_size=1, moe_ep_size=1)
    if kvcache_bytes_per_elem == 1:
        model_config.kvcache_quant_mode = sdk_config.common.KVCacheQuantMode.fp8
    model_config.nextn = nextn
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


@pytest.mark.parametrize("nextn", [0, 1, 3], ids=("spec-off", "nextn1", "nextn3"))
def test_gptoss_kv_bytes_is_nextn_independent(nextn):
    """Speculative decoding must not re-linearize the window-capped KV curve.

    Draft tokens are a per-step compute cost, not extra resident KV, so the
    hybrid-SWA byte count is identical with spec decode on and off.
    """
    model = _model(nextn=nextn)
    seq_len = 65_936  # 65536 ISL + 400 OSL, the 64k agentic recipe shape
    assert model.get_kvcache_bytes_per_sequence(seq_len) == pytest.approx(_expected_bytes(seq_len), rel=1e-9)


def test_gptoss_backend_memory_kv_stays_window_aware_under_spec_decode():
    """The breakdown's ``kvcache`` (base_backend._get_memory_usage, the sole KV
    sizing site) must follow the window-capped curve with ``nextn > 0``.

    Pins the recipe operating point: batch 48 at ISL 65536 / OSL 400, fp8 KV,
    TP1 -> 54.4 GiB, not the ~2x all-layers-full-context value that false-OOMs
    the shipped 8xB200 agg recipe.
    """
    batch_size, isl, osl = 48, 65_536, 400
    seq_len = isl + osl
    model = _model(nextn=3)
    database = SimpleNamespace(system_spec={"misc": {"nccl_mem": {1: 0}, "other_mem": 0}})

    memory = get_backend("vllm")._get_memory_usage(
        model, database, batch_size=batch_size, beam_width=1, isl=isl, osl=osl
    )

    expected_gib = batch_size * _expected_bytes(seq_len) / (1 << 30)
    assert memory["kvcache"] == pytest.approx(expected_gib, rel=1e-9)
    assert memory["kvcache"] == pytest.approx(54.4, abs=0.1)
    linear_gib = batch_size * seq_len * LAYERS * 2 * KV_HEADS * HEAD_SIZE / (1 << 30)
    assert memory["kvcache"] < 0.52 * linear_gib


def test_gptoss_kv_bytes_scale_linearly_with_kvcache_dtype():
    """Guard against misreading a dtype difference as a layout regression.

    The window-aware bf16 figure (2.268 GiB/seq at 65,936) is within 0.2% of the
    *linear* fp8 figure (2.264 GiB/seq) that the hybrid override removed, so the
    two are easy to confuse when reading a CLI breakdown. Pin that the only
    difference between the two dtypes is the element size.
    """
    seq_len = 65_936
    fp8 = _model(kvcache_bytes_per_elem=1).get_kvcache_bytes_per_sequence(seq_len)
    bf16 = _model(kvcache_bytes_per_elem=2).get_kvcache_bytes_per_sequence(seq_len)
    assert bf16 == pytest.approx(2 * fp8, rel=1e-9)
    assert bf16 == pytest.approx(_expected_bytes(seq_len, bytes_per_elem=2), rel=1e-9)


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

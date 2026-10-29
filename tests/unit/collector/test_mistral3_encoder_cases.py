# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Medium-3.5 collection profile.

The Pixtral vision encoder is head_dim=104, which no base encoder sweep covers
and which the SDK requires as an exact ``EncoderKey`` partition. This pins the
model-correlated expansion: the encoder-attention TP shards resolve to the
Pixtral head counts at head_dim=104, and the text attention row keeps head_dim
128.
"""

import pytest

from collector.case_generator import (
    get_attention_context_shape_sweeps,
    get_attention_encoder_head_configs,
    get_attention_encoder_shape_sweeps,
    get_attention_head_configs,
)
from collector.model_cases import build_collection_case_plan

pytestmark = pytest.mark.unit

_MODEL_PATH = "mistralai/Mistral-Medium-3.5-128B"


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
def test_encoder_attention_resolves_pixtral_head_dim_104(monkeypatch, backend):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", _MODEL_PATH)
    configs = {
        (cfg.num_heads, cfg.head_dim)
        for sweep in get_attention_encoder_shape_sweeps(backend)
        for cfg in get_attention_encoder_head_configs(sweep)
    }
    # 16 heads sharded by TP {1,2,4,8,16} -> {16,8,4,2,1}, all at head_dim 104.
    assert configs == {(16, 104), (8, 104), (4, 104), (2, 104), (1, 104)}


@pytest.mark.parametrize(
    ("backend", "expected"),
    [("sglang", True), ("vllm", True), ("trtllm", False)],
)
def test_encoder_attention_activation_excludes_trtllm(backend, expected):
    # trtllm has no FMHA kernel for Pixtral head_dim=104, so encoder_attention
    # is deliberately not activated in its plan (only sglang and vllm collect it).
    plan = build_collection_case_plan(backend=backend, model_path=_MODEL_PATH)
    assert plan.has_op("encoder_attention") is expected


def test_text_attention_keeps_head_dim_128(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", _MODEL_PATH)
    configs = {
        (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.window_size)
        for sweep in get_attention_context_shape_sweeps("trtllm")
        for cfg in get_attention_head_configs(sweep, phase="context")
    }
    # 96 q / 8 kv heads, head_dim 128, sharded by TP {1,2,4,8,16}; a shard is
    # kept only when local q heads stay an integer multiple of local kv heads.
    expected = {
        (96 // tp, (8 + tp - 1) // tp, 128, 0) for tp in (1, 2, 4, 8, 16) if (96 // tp) % ((8 + tp - 1) // tp) == 0
    }
    assert configs == expected

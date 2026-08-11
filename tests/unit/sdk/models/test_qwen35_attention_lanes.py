# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 carries the ``attention_backend`` lane override into its attention ops.

AIC-1715: ``ModelConfig.attention_backend`` is the user-facing knob that heads the
attention lane precedence order. Qwen3.5 is the first dense model plumbed for it,
so both the context and the generation attention ops must receive it.
"""

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk import config as sdk_config
from aiconfigurator.sdk.models.qwen35 import Qwen35Model

pytestmark = pytest.mark.unit


def _qwen35_config():
    """Two-layer hybrid (one GDN + one full-attention) dense Qwen3.5."""
    return common.Qwen35Config(
        layer_types=("linear_attention", "full_attention"),
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_num_value_heads=32,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )


def _build_model(attention_backend):
    model_config = sdk_config.ModelConfig(
        tp_size=1,
        pp_size=1,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        moe_quant_mode=common.MoEQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        attention_backend=attention_backend,
    )
    return Qwen35Model(
        "Qwen/Qwen3.5-Test",
        "QWEN35",
        "Qwen35ForCausalLM",
        2,  # num_layers
        32,  # num_heads
        8,  # num_kv_heads
        128,  # head_size
        4096,  # hidden_size
        12288,  # inter_size
        151936,  # vocab_size
        32768,  # context_length
        model_config,
        _qwen35_config(),
    )


def _attention_ops(model):
    from aiconfigurator.sdk.operations.attention import ContextAttention, GenerationAttention

    ctx = [op for op in model.context_ops if isinstance(op, ContextAttention)]
    gen = [op for op in model.generation_ops if isinstance(op, GenerationAttention)]
    assert ctx and gen, "the full-attention layer must produce both attention ops"
    return ctx, gen


def test_attention_backend_reaches_both_attention_ops():
    """The knob flows into context AND generation attention (not just one side)."""
    ctx_ops, gen_ops = _attention_ops(_build_model("trtllm_mha"))

    assert all(op._attention_backend == "trtllm_mha" for op in ctx_ops)
    assert all(op._attention_backend == "trtllm_mha" for op in gen_ops)


def test_attention_backend_defaults_to_no_override():
    """Unset knob means no override: the framework-default lane heads the order."""
    ctx_ops, gen_ops = _attention_ops(_build_model(None))

    assert all(op._attention_backend is None for op in ctx_ops)
    assert all(op._attention_backend is None for op in gen_ops)


def test_model_config_attention_backend_defaults_to_none():
    """``ModelConfig`` must not force a lane: the default is "no override"."""
    assert sdk_config.ModelConfig().attention_backend is None

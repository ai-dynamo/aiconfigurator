# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 carries the ``attention_backend`` lane override into its attention ops.

AIC-1715: ``ModelConfig.attention_backend`` is the user-facing knob that heads the
attention lane precedence order. Since the pyo3 op unification, models are built
WITHOUT a database handle (pure shape graphs); ops no longer store the override
directly (``ContextAttention``/``GenerationAttention`` have no Python
``__init__``). The knob rides ``model.config.attention_backend`` — read
generically for every model family, not per-op — until
``engine.py::_resolve_attention_lane_orders`` resolves it (with a database) at
spec-build time and sets each op's ``_lane_order``. This file checks both
halves: the knob survives model construction, and resolution heads the walk
with it.
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
        backend_name="sglang",
    )


def _attention_ops(model):
    from aiconfigurator.sdk.operations.attention import ContextAttention, GenerationAttention

    ctx = [op for op in model.context_ops if isinstance(op, ContextAttention)]
    gen = [op for op in model.generation_ops if isinstance(op, GenerationAttention)]
    assert ctx and gen, "the full-attention layer must produce both attention ops"
    return ctx, gen


def test_attention_backend_reaches_both_attention_ops():
    """The knob survives model construction and heads BOTH ops' resolved lane
    order once ``build_engine_spec_json`` resolves it against a real database
    (context AND generation, not just one side) — the modern equivalent of the
    retired per-op ``_attention_backend`` attribute check. A database is
    required: resolution reads backend/version/sm_version/systems_root off it
    (``database=None`` always yields the always-valid ``["default"]``, override
    or not — see the no-override twin below)."""
    from aiconfigurator.sdk.perf_database import get_database
    from aiconfigurator_core.sdk.engine import _resolve_attention_lane_orders

    model = _build_model("trtllm_mha")
    assert model.config.attention_backend == "trtllm_mha"
    ctx_ops, gen_ops = _attention_ops(model)

    database = get_database("b200_sxm", "sglang", "0.5.14")
    _resolve_attention_lane_orders(ctx_ops, database, model.config.attention_backend)
    _resolve_attention_lane_orders(gen_ops, database, model.config.attention_backend)

    assert all(op._lane_order[0] == "trtllm_mha" for op in ctx_ops)
    assert all(op._lane_order[0] == "trtllm_mha" for op in gen_ops)


def test_attention_backend_defaults_to_no_override():
    """Unset knob means no override: the framework-default lane heads the order
    (a ``None`` database still resolves the always-valid ``["default"]``)."""
    from aiconfigurator_core.sdk.engine import _resolve_attention_lane_orders

    model = _build_model(None)
    assert model.config.attention_backend is None
    ctx_ops, gen_ops = _attention_ops(model)

    _resolve_attention_lane_orders(ctx_ops, None, model.config.attention_backend)
    _resolve_attention_lane_orders(gen_ops, None, model.config.attention_backend)

    assert all(op._lane_order == ["default"] for op in ctx_ops)
    assert all(op._lane_order == ["default"] for op in gen_ops)


def test_model_config_attention_backend_defaults_to_none():
    """``ModelConfig`` must not force a lane: the default is "no override"."""
    assert sdk_config.ModelConfig().attention_backend is None

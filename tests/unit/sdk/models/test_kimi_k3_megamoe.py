# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kimi-K3 MegaMoE modeling contracts."""

import pytest

from aiconfigurator.sdk import common, models
from aiconfigurator.sdk import config as sdk_config
from aiconfigurator.sdk.operations import DeepSeekV4MegaMoEModule, MoE, MoEDispatch

pytestmark = pytest.mark.unit


def _model_config(*, tp=1, moe_tp=1, moe_ep=8, attn_dp=8, moe_backend=None):
    return sdk_config.ModelConfig(
        tp_size=tp,
        pp_size=1,
        moe_tp_size=moe_tp,
        moe_ep_size=moe_ep,
        attention_dp_size=attn_dp,
        moe_backend=moe_backend,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        moe_quant_mode=common.MoEQuantMode.w4a8_mxfp4_mxfp8,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
    )


def test_kimi_k3_megamoe_replaces_decomposed_moe_with_module():
    model = models.get_model("moonshotai/Kimi-K3", _model_config(moe_backend="megamoe"), "sglang")
    pre_dispatch = "sglang_jit"
    names = {op._name: op for op in model.generation_ops}
    assert "generation_moe" not in names
    assert "generation_moe_pre_dispatch" not in names
    assert "generation_moe_post_dispatch" not in names
    module = names["generation_megamoe"]
    assert isinstance(module, DeepSeekV4MegaMoEModule)
    assert module._hidden_size == 3584
    assert module._inter_size == 3072
    assert module._topk == 16
    assert module._num_experts == 896
    assert module._moe_tp_size == 1
    assert module._moe_ep_size == 8
    assert module._pre_dispatch == pre_dispatch
    assert "generation_router_gemm" in names
    assert "generation_latent_down_gemm" in names
    assert "generation_latent_up_gemm" in names
    assert "generation_shared_gate_up_gemm" in names


def test_kimi_k3_default_moe_path_stays_decomposed():
    model = models.get_model("moonshotai/Kimi-K3", _model_config(), "vllm")
    kinds = {type(op) for op in model.generation_ops}
    assert MoE in kinds
    assert MoEDispatch in kinds
    assert DeepSeekV4MegaMoEModule not in kinds


def test_kimi_k3_megamoe_rejects_moe_tp():
    with pytest.raises(ValueError, match="moe_tp_size=1"):
        models.get_model(
            "moonshotai/Kimi-K3",
            _model_config(tp=2, moe_tp=2, moe_ep=4, attn_dp=4, moe_backend="megamoe"),
            "sglang",
        )


def test_kimi_k3_megamoe_vllm_uses_vllm_pre_dispatch():
    model = models.get_model("moonshotai/Kimi-K3", _model_config(moe_backend="megamoe"), "vllm")
    names = {op._name: op for op in model.generation_ops}
    assert "generation_moe" not in names
    module = names["generation_megamoe"]
    assert isinstance(module, DeepSeekV4MegaMoEModule)
    assert module._pre_dispatch == "vllm"
    assert module._hidden_size == 3584
    assert module._inter_size == 3072
    assert module._topk == 16
    assert module._num_experts == 896
    assert module._moe_tp_size == 1
    assert module._moe_ep_size == 8


def test_kimi_k3_megamoe_vllm_agg_answers_via_compiled_engine():
    """E2E regression for the family-first MegaMoE data dir.

    The unified table lives at ``gb300/megamoe/vllm/0.27.0/``, not the legacy
    ``<system>/<backend>/<version>`` dir. The compiled engine used to hard-
    point its single primary at the legacy dir, so every MegaMoE query
    reported "data not loaded" and the CLI sweep returned zero feasible
    configs while the Python step answered the same points fine.
    """
    from aiconfigurator.sdk import perf_database, rust_engine_step
    from aiconfigurator.sdk.backends.factory import get_backend
    from aiconfigurator.sdk.config import RuntimeConfig

    database = perf_database.get_database("gb300", "vllm", "0.27.0")
    if not rust_engine_step.should_use_rust_engine_step(RuntimeConfig(), database):
        pytest.skip("compiled engine unavailable: python fallback would mask the family-resolution regression")
    model = models.get_model(
        "moonshotai/Kimi-K3",
        _model_config(moe_ep=16, attn_dp=16, moe_backend="megamoe"),
        "vllm",
    )
    summary = get_backend("vllm").run_agg(
        model=model,
        database=database,
        runtime_config=RuntimeConfig(isl=4000, osl=1000, batch_size=8),
        ctx_tokens=4000,
    )
    result = summary.get_result_dict()
    assert result["ttft"] > 0 and result["tpot"] > 0
    # The fused module must have answered from measured silicon rows — if the
    # compiled engine regresses back to the legacy dir the query dies before
    # this breakdown exists at all.
    sources = summary.get_per_ops_source()
    assert sources is not None
    megamoe_sources = {ops[op] for ops in sources.values() for op in ops if op.endswith("_megamoe")}
    assert megamoe_sources == {"silicon"}, f"megamoe sources: {megamoe_sources}"


def test_kimi_k3_megamoe_rejects_unsupported_backend():
    with pytest.raises(ValueError, match="SGLang and vLLM"):
        models.get_model("moonshotai/Kimi-K3", _model_config(moe_backend="megamoe"), "trtllm")


def test_kimi_k3_identity_and_remap_follow_local_checkpoint_sizes(tmp_path):
    """A local/offline K3 checkpoint mount (/data/models/...) must get the same
    identity and Blackwell quant remap as the hub id (CodeRabbit review)."""
    import json

    from aiconfigurator.sdk.models import helpers

    cfg = {
        "architectures": ["KimiK3ForConditionalGeneration"],
        "model_type": "kimi_k3",
        "num_hidden_layers": 4,
        "hidden_size": 7168,
        "num_attention_heads": 96,
        "num_key_value_heads": 96,
        "vocab_size": 163840,
        "max_position_embeddings": 1048576,
        "head_dim": 128,
        "num_experts": 896,
        "topk": 16,
        "linear_attn_config": {"kda_layers": [1, 2, 3], "num_heads": 96, "head_dim": 128},
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))

    assert helpers._is_kimi_k3_checkpoint(str(tmp_path))
    assert not helpers._is_kimi_k3_checkpoint(str(tmp_path / "missing_child"))
    # vllm megamoe on Blackwell -> fused w4a8 key; plain vllm -> no remap.
    assert (
        helpers.resolve_kimi_k3_moe_arch_mode(str(tmp_path), "gb300", "vllm", "megamoe")
        == common.MoEQuantMode.w4a8_mxfp4_mxfp8
    )
    assert helpers.resolve_kimi_k3_moe_arch_mode(str(tmp_path), "gb300", "vllm", None) is None
    assert (
        helpers.resolve_kimi_k3_moe_arch_mode(str(tmp_path), "gb300", "sglang", None)
        == common.MoEQuantMode.w4a8_mxfp4_mxfp8
    )

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for StepFun Step-3.7-Flash (STEP3P7).

Covers the three things this model adds on top of HybridMoEModel:
  * architecture -> family mapping (Step3p7 / Step3p5 -> STEP3P7),
  * the hybrid layer split parsed from ``layer_types`` (33 SWA + 12 global at
    the real 45-layer 1:3 recipe) plus dense-first-K MoE frequency, and
  * the window-capped KV curve: at 16K context the SWA-512 layers cap while the
    global layers grow, giving ~0.29x the KV of an all-global model.
"""

import pytest

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.models import _architecture_to_model_family
from aiconfigurator.sdk.models.base import _MODEL_REGISTRY
from aiconfigurator.sdk.models.step3p7 import Step3p7Model
from aiconfigurator.sdk.utils import _parse_hf_config_json

pytestmark = pytest.mark.unit


def _step37_hf_config():
    """Real Step-3.7-Flash shape: 45 layers, 1:3 global:SWA, SWA window 512.

    layer_types is ``full`` every 4th layer (indices 0,4,8,...,44) -> 12 global,
    33 sliding. first_k_dense_replace=3 -> first 3 layers dense, rest MoE.
    """
    layer_types = ["full_attention" if i % 4 == 0 else "sliding_attention" for i in range(45)]
    return {
        "architectures": ["Step3p7FlashForCausalLM"],
        "num_hidden_layers": 45,
        "hidden_size": 4096,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 128896,
        "max_position_embeddings": 65536,
        "intermediate_size": 11264,
        "sliding_window": 512,
        "layer_types": layer_types,
        "first_k_dense_replace": 3,
        "num_experts": 288,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 1280,
        "share_expert_dim": 1280,
        "n_shared_experts": 1,
        "num_nextn_predict_layers": 3,
    }


class TestStep3p7ArchMapping:
    def test_architecture_maps_to_step3p7_family(self):
        assert _architecture_to_model_family("Step3p7FlashForCausalLM") == "STEP3P7"
        assert _architecture_to_model_family("Step3p5FlashForCausalLM") == "STEP3P7"

    def test_registered_class_is_step3p7_model(self):
        assert _MODEL_REGISTRY.get("STEP3P7") is Step3p7Model
        assert "STEP3P7" in common.ModelFamily


class TestStep3p7ConfigParse:
    def test_parse_yields_hybrid_config(self):
        result = _parse_hf_config_json(_step37_hf_config())

        assert result["architecture"] == "Step3p7FlashForCausalLM"
        assert result["layers"] == 45
        assert result["n"] == 64
        assert result["n_kv"] == 8
        assert result["d"] == 128
        assert result["topk"] == 8
        assert result["num_experts"] == 288
        assert result["moe_inter_size"] == 1280

        cfg = result["extra_params"]
        assert isinstance(cfg, common.HybridMoEConfig)
        # 1:3 global:SWA -> 12 global, 33 sliding.
        assert sum(cfg.attn_layer_pattern) == 12
        assert cfg.attn_layer_pattern.count(0) == 33
        # first_k_dense_replace=3 -> first 3 dense, remaining 42 MoE.
        assert cfg.moe_layer_freq.count(0) == 3
        assert sum(cfg.moe_layer_freq) == 42
        assert cfg.sliding_window_size == 512

    def test_layer_types_length_mismatch_raises(self):
        cfg = _step37_hf_config()
        cfg["layer_types"] = cfg["layer_types"][:10]  # 10 != 45
        with pytest.raises(ValueError, match="layer_types length"):
            _parse_hf_config_json(cfg)

    def test_invalid_layer_type_raises(self):
        cfg = _step37_hf_config()
        cfg["layer_types"][1] = "linear_attention"
        with pytest.raises(ValueError, match="must contain only"):
            _parse_hf_config_json(cfg)


class TestStep3p7KVCache:
    @staticmethod
    def _make_model(tp_size=1):
        result = _parse_hf_config_json(_step37_hf_config())
        model_config = config.ModelConfig(
            tp_size=tp_size,
            pp_size=1,
            moe_tp_size=tp_size,
            moe_ep_size=1,
            attention_dp_size=1,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            moe_quant_mode=common.MoEQuantMode.bfloat16,
        )
        model = Step3p7Model(
            result["topk"],
            result["num_experts"],
            result["moe_inter_size"],
            "stepfun-ai/Step-3.7-Flash",
            "STEP3P7",
            result["architecture"],
            result["layers"],
            result["n"],
            result["n_kv"],
            result["d"],
            result["hidden_size"],
            result["inter_size"],
            result["vocab"],
            result["context"],
            model_config,
        )
        model._share_expert_dim = 1280
        model.set_hybrid_config(result["extra_params"])
        return model

    def test_swa_global_counts(self):
        model = self._make_model()
        assert model._swa_global_counts() == (33, 12)

    def test_kv_ratio_at_16k_is_about_0_29(self):
        """SWA-512 layers cap while global layers grow -> ~0.29x an all-global model."""
        model = self._make_model()
        seq = 16384
        step_kv = model.get_kvcache_bytes_per_sequence(seq)
        # All-global reference at the same head geometry: every layer grows linearly.
        per = model._kv_per_layer_per_token()
        full_kv = model._num_layers * per * seq
        # 33*512 + 12*16384 = 213504 window-token-layers vs 45*16384 = 737280.
        assert step_kv == 33 * per * 512 + 12 * per * seq
        assert step_kv / full_kv == pytest.approx(0.2896, abs=1e-3)

    def test_kv_below_window_is_linear(self):
        model = self._make_model()
        per_token = model.get_kvcache_bytes_per_sequence(1)
        budget = model.get_kvcache_bytes_per_sequence(512)  # <= window, still linear
        assert model.get_kvcache_max_tokens(budget) == int(budget // per_token) == 512

    def test_max_tokens_follows_window_capped_curve(self):
        model = self._make_model()
        budget = model.get_kvcache_bytes_per_sequence(65536)
        tokens = model.get_kvcache_max_tokens(budget)
        assert model.get_kvcache_bytes_per_sequence(tokens) <= budget
        assert model.get_kvcache_bytes_per_sequence(tokens + 1) > budget

    def test_shared_expert_ops_appended(self):
        model = self._make_model()
        ctx = [o for o in model.context_ops if "shared_expert" in getattr(o, "_name", "")]
        gen = [o for o in model.generation_ops if "shared_expert" in getattr(o, "_name", "")]
        # dense gate_up + act + down = 3 ops on each side.
        assert len(ctx) == 3
        assert len(gen) == 3


def test_step3p7_parses_the_published_checkpoint_schema():
    """Parse the schema the published checkpoint actually ships.

    Four things differ from the curated fixture and each one silently broke the
    model: the architecture strings are Step3p7ForConditionalGeneration /
    Step3p5ForCausalLM (not the *Flash* spellings this repo invented), the decoder
    is nested under text_config, layer_types is sized over the decoder PLUS the 3
    MTP predict layers, and MoE placement comes from moe_layers_enum rather than
    first_k_dense_replace.
    """
    hf_config = {
        "architectures": ["Step3p7ForConditionalGeneration"],
        "model_type": "step3p7",
        "text_config": {
            "architectures": ["Step3p5ForCausalLM"],
            "model_type": "step3p5",
            "num_hidden_layers": 6,
            "num_nextn_predict_layers": 2,
            "hidden_size": 4096,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 11264,
            "sliding_window": 512,
            "vocab_size": 128896,
            "max_position_embeddings": 65536,
            # 8 entries = 6 decoder layers + 2 MTP predict layers
            "layer_types": [
                "full_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
            "moe_layers_enum": "2,3,4,5",
            "moe_num_experts": 288,
            "num_experts": 288,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 1280,
            "attention_other_setting": {"num_attention_heads": 96, "head_dim": 128},
        },
    }

    parsed = _parse_hf_config_json(hf_config)
    extra = parsed["extra_params"]

    # sliding layers take the declared 96 query heads, not the global 64
    assert extra.swa_num_heads == 96
    assert extra.swa_head_dim == 128
    # layer_types trimmed to the 6 decoder layers, MTP tail dropped
    assert len(extra.attn_layer_pattern) == 6
    assert extra.attn_layer_pattern == (1, 0, 0, 1, 0, 0)
    # MoE placement from moe_layers_enum, so layers 0-1 stay dense
    assert extra.moe_layer_freq == (0, 0, 1, 1, 1, 1)
    assert extra.sliding_window_size == 512


def test_shared_expert_ops_are_cp_audited():
    """Shared-expert ops must carry the same seq_split as the rest of the pipeline.

    They are appended after ``_build_context_ops`` has already wired context
    parallelism. Before the fix they kept ``_seq_split=1`` while the inherited
    dense FFN ops carried ``cp``, so every rank was charged the full
    shared-expert token count; they also bypassed the un-audited-op guard, so it
    surfaced as a wrong number rather than an error.
    """
    cp = 2
    result = _parse_hf_config_json(_step37_hf_config())
    model_config = config.ModelConfig(
        tp_size=1,
        pp_size=1,
        moe_tp_size=1,
        moe_ep_size=cp,  # tp * dp * cp must equal moe_tp * moe_ep
        attention_dp_size=1,
        cp_size=cp,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        moe_quant_mode=common.MoEQuantMode.bfloat16,
    )
    model = Step3p7Model(
        result["topk"],
        result["num_experts"],
        result["moe_inter_size"],
        "stepfun-ai/Step-3.7-Flash",
        "STEP3P7",
        result["architecture"],
        result["layers"],
        result["n"],
        result["n_kv"],
        result["d"],
        result["hidden_size"],
        result["inter_size"],
        result["vocab"],
        result["context"],
        model_config,
    )
    model._share_expert_dim = 1280
    model.set_hybrid_config(result["extra_params"])

    shared = [o for o in model.context_ops if "context_shared_expert" in getattr(o, "_name", "")]
    assert shared, "expected shared-expert context ops"
    split = {getattr(o, "_seq_split", None) for o in shared}
    assert split == {cp}, f"shared-expert ops must split by cp={cp}, got {split}"

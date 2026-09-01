# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense-family draft schemes on Qwen3-8B: DSpark (dense branch), DFlash,
and EAGLE-3 (chain + tree).

Ground facts pinned from the three community draft checkpoints:

* deepseek-ai/dspark_qwen3_8b_block7 @03326e50 — 5 full-attention Qwen3
  layers, block_size 7, target_layer_ids [1,9,17,25,33], markov rank 256,
  OWN embed + lm_head; safetensors 4 742 170 330 B.
* z-lab/Qwen3-8B-DFlash-b16 @9b41424b — the block-identical 5-layer stack,
  block_size 16 (anchor + 15 masks), same target_layer_ids, no sampling
  heads and no embed/head; safetensors 2 097 259 104 B.
* Tengyunw/qwen3_8b_eagle3 — one decoder layer with 2h QKV input,
  fc(3h->h), lm_head over draft_vocab 32000; pytorch_model.bin
  799 493 246 B.

Weight assertions close against the actual checkpoint byte counts — the
strongest structural check available without GPUs.
"""

from __future__ import annotations

import pytest

from aiconfigurator_core.sdk import common, models
from aiconfigurator_core.sdk import config as sdk_config
from aiconfigurator_core.sdk.speculation import SpeculationConfig
from aiconfigurator_core.sdk.speculation.dflash import DFlashScheme
from aiconfigurator_core.sdk.speculation.dspark import DSparkScheme
from aiconfigurator_core.sdk.speculation.eagle import EagleScheme

pytestmark = pytest.mark.unit

# Real values from deepseek-ai/dspark_qwen3_8b_block7 config.json.
DSPARK_8B_CONFIG = {
    "architectures": ["Qwen3DSparkModel"],
    "model_type": "qwen3",
    "block_size": 7,
    "target_layer_ids": [1, 9, 17, 25, 33],
    "markov_rank": 256,
    "num_hidden_layers": 5,
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "sliding_window": None,
    "use_sliding_window": False,
}
DSPARK_8B_CKPT_BYTES = 4_742_170_330

# Real values from z-lab/Qwen3-8B-DFlash-b16 config.json.
DFLASH_CONFIG = {
    "architectures": ["DFlashDraftModel"],
    "model_type": "qwen3",
    "block_size": 16,
    "dflash_config": {"mask_token_id": 151669, "target_layer_ids": [1, 9, 17, 25, 33]},
    "num_hidden_layers": 5,
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "sliding_window": None,
    "use_sliding_window": False,
}
DFLASH_CKPT_BYTES = 2_097_259_104

# Real values from Tengyunw/qwen3_8b_eagle3 config.json.
EAGLE3_CONFIG = {
    "architectures": ["LlamaForCausalLMEagle3"],
    "model_type": "llama",
    "num_hidden_layers": 1,
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "draft_vocab_size": 32000,
    "sliding_window": None,
    "use_sliding_window": False,
}
EAGLE3_CKPT_BYTES = 799_493_246


def _model_config(kind: str, draft_config: dict, **params) -> sdk_config.ModelConfig:
    return sdk_config.ModelConfig(
        tp_size=1,
        pp_size=1,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        speculation=SpeculationConfig(kind=kind, params=params, draft_config=draft_config),
    )


def _q8b(kind: str, draft_config: dict, **params):
    return models.get_model("Qwen/Qwen3-8B", _model_config(kind, draft_config, **params), "vllm")


class TestDSparkDense:
    def test_param_resolution_from_dense_config(self):
        model = _q8b("dspark", DSPARK_8B_CONFIG)
        scheme = model.spec_scheme
        assert isinstance(scheme, DSparkScheme)
        assert scheme.num_draft_tokens == 7  # block_size
        assert scheme.num_draft_layers == 5
        assert scheme.markov_rank == 256
        assert scheme.target_layer_ids == (1, 9, 17, 25, 33)
        assert scheme.verify_width() == 8
        assert scheme.draft_geometry is not None

    def test_op_graph_is_dense_not_v4(self):
        model = _q8b("dspark", DSPARK_8B_CONFIG)
        specs = model.spec_scheme.build_draft_generation_ops(model)
        names = [s.op._name for s in specs]
        assert "dspark_qkv_gemm" in names
        assert "dspark_gate_ffn1_gemm" in names
        assert not any("moe" in n or "mhc" in n for n in names)
        by_name = {s.op._name: s for s in specs}
        # Full-attention draft: no window-capped s override, backbone width N.
        assert by_name["dspark_attention"].query_overrides is None
        assert by_name["dspark_attention"].tokens_per_request == 7
        # main_proj fuses 5 target hiddens: k = 5h.
        assert by_name["dspark_main_proj"].op._k == 5 * 4096
        # Markov chain still sequential: count = N.
        assert by_name["dspark_markov_bias_gemm"].op._scale_factor == 7

    def test_weights_close_against_checkpoint(self):
        model = _q8b("dspark", DSPARK_8B_CONFIG)
        w = model.spec_scheme.draft_weights_bytes(model)
        # Own embed + head counted (unlike the V4 draft, which aliases them).
        assert w == pytest.approx(DSPARK_8B_CKPT_BYTES, rel=0.01)

    def test_draft_kv_full_attention_scales_with_seq(self):
        model = _q8b("dspark", DSPARK_8B_CONFIG)
        scheme = model.spec_scheme
        kv_1k = scheme.draft_kv_bytes_per_sequence(model, 1_000)
        kv_10k = scheme.draft_kv_bytes_per_sequence(model, 10_000)
        assert kv_10k == pytest.approx(10 * kv_1k)  # no window cap
        # 5 layers x 2 (K+V) x 8 kv heads x 128 head_dim x 2 B (bf16)
        assert kv_1k == pytest.approx(5 * 1_000 * 2 * 8 * 128 * 2)

    def test_dense_config_rejected_on_v4_family(self):
        cfg = sdk_config.ModelConfig(
            tp_size=8,
            pp_size=1,
            moe_tp_size=1,
            moe_ep_size=8,
            gemm_quant_mode=common.GEMMQuantMode.fp8_block,
            moe_quant_mode=common.MoEQuantMode.nvfp4,
            kvcache_quant_mode=common.KVCacheQuantMode.fp8,
            fmha_quant_mode=common.FMHAQuantMode.fp8,
            speculation=SpeculationConfig(kind="dspark", params={}, draft_config=DSPARK_8B_CONFIG),
        )
        with pytest.raises(ValueError, match="model families"):
            models.get_model("deepseek-ai/DeepSeek-V4-Flash", cfg, "sglang")


class TestDFlash:
    def test_param_resolution(self):
        model = _q8b("dflash", DFLASH_CONFIG)
        scheme = model.spec_scheme
        assert isinstance(scheme, DFlashScheme)
        # block_size counts the anchor: b16 -> 15 drafts by default.
        assert scheme.num_draft_tokens == 15
        assert scheme.verify_width() == 16
        deployed = _q8b("dflash", DFLASH_CONFIG, num_draft_tokens=7).spec_scheme
        assert deployed.verify_width() == 8

    def test_single_shot_block_no_markov(self):
        model = _q8b("dflash", DFLASH_CONFIG, num_draft_tokens=7)
        specs = model.spec_scheme.build_draft_generation_ops(model)
        names = [s.op._name for s in specs]
        assert not any("markov" in n for n in names)
        by_name = {s.op._name: s for s in specs}
        # ONE parallel forward over anchor + N masks.
        assert by_name["dflash_attention"].tokens_per_request == 8
        # Logits only over the N drafted positions (target-aliased head).
        assert by_name["dflash_head_gemm"].tokens_per_request == 7
        assert by_name["dflash_main_proj"].op._k == 5 * 4096

    def test_weights_close_against_checkpoint(self):
        model = _q8b("dflash", DFLASH_CONFIG)
        w = model.spec_scheme.draft_weights_bytes(model)
        # Blocks + main_proj only: no embed/head/sampling weights.
        assert w == pytest.approx(DFLASH_CKPT_BYTES, rel=0.01)

    def test_missing_dflash_marker_raises(self):
        bad = {k: v for k, v in DFLASH_CONFIG.items() if k != "dflash_config"}
        with pytest.raises(ValueError, match="target_layer_ids"):
            DFlashScheme.from_configs(
                _model_config("dflash", DFLASH_CONFIG),
                SpeculationConfig(kind="dflash", params={}, draft_config=bad),
            )


class TestEagle3:
    def test_chain_param_resolution(self):
        model = _q8b("eagle3", EAGLE3_CONFIG, num_speculative_tokens=3)
        scheme = model.spec_scheme
        assert isinstance(scheme, EagleScheme)
        assert scheme.tree_shape == (1, 1, 1)
        assert not scheme.is_tree
        assert scheme.verify_width() == 4
        assert scheme.draft_vocab_size == 32000

    def test_tree_shape_generalizes_verify_budget(self):
        scheme = _q8b("eagle3", EAGLE3_CONFIG, tree_shape=[4, 8, 8]).spec_scheme
        assert scheme.is_tree
        assert scheme.num_draft_tokens == 20
        assert scheme.verify_width() == 21  # token budget, not depth + 1

    def test_chain_op_graph(self):
        model = _q8b("eagle3", EAGLE3_CONFIG, num_speculative_tokens=3)
        specs = model.spec_scheme.build_draft_generation_ops(model)
        names = [s.op._name for s in specs]
        # fc fusion once per round; K sequential single-token levels.
        assert names.count("eagle3_fc") == 1
        assert names.count("eagle3_qkv_gemm") == 3
        by_name = {s.op._name: s for s in specs}
        assert by_name["eagle3_fc"].op._k == 3 * 4096  # 3 aux hiddens
        assert by_name["eagle3_qkv_gemm"].op._k == 2 * 4096  # cat(emb, hidden)
        assert by_name["eagle3_head_gemm"].op._n == 32000  # reduced draft vocab
        assert all(s.tokens_per_request == 1 for s in specs)

    def test_tree_levels_carry_their_widths(self):
        model = _q8b("eagle3", EAGLE3_CONFIG, tree_shape=[4, 8, 8])
        specs = model.spec_scheme.build_draft_generation_ops(model)
        widths = [s.tokens_per_request for s in specs if s.op._name == "eagle3_attention"]
        assert widths == [4, 8, 8]

    def test_weights_close_against_checkpoint(self):
        model = _q8b("eagle3", EAGLE3_CONFIG)
        w = model.spec_scheme.draft_weights_bytes(model)
        assert w == pytest.approx(EAGLE3_CKPT_BYTES, rel=0.01)

    def test_draft_kv_single_layer(self):
        model = _q8b("eagle3", EAGLE3_CONFIG)
        kv = model.spec_scheme.draft_kv_bytes_per_sequence(model, 1_000)
        assert kv == pytest.approx(1 * 1_000 * 2 * 8 * 128 * 2)

    def test_invalid_tree_shape_raises(self):
        with pytest.raises(ValueError, match="tree_shape"):
            EagleScheme.from_configs(
                _model_config("eagle3", EAGLE3_CONFIG),
                SpeculationConfig(kind="eagle3", params={"tree_shape": [4, 0]}, draft_config=EAGLE3_CONFIG),
            )


class TestNgram:
    def test_pure_verify_width_scheme(self):
        model = _q8b("ngram", None, num_speculative_tokens=8)
        scheme = model.spec_scheme
        assert scheme.verify_width() == 9
        assert scheme.build_draft_generation_ops(model) == []
        assert scheme.build_draft_context_ops(model) == []
        assert scheme.draft_weights_bytes(model) == 0.0
        assert scheme.draft_kv_bytes_per_sequence(model, 10_000) == 0.0

    def test_requires_explicit_k(self):
        from aiconfigurator_core.sdk.speculation.ngram import NgramScheme

        with pytest.raises(ValueError, match="num_speculative_tokens"):
            NgramScheme.from_configs(None, SpeculationConfig(kind="ngram", params={}))


class TestEagleVerifyBudget:
    def test_budget_decouples_draft_width_from_verify(self):
        # SGLang semantics: steps=5, topk=8 drafts 40 nodes but prunes the
        # verify forward to num_draft_tokens=32 (root included).
        scheme = _q8b("eagle3", EAGLE3_CONFIG, tree_shape=[8] * 5, verify_token_budget=32).spec_scheme
        assert scheme.num_draft_tokens == 40  # draft-side compute
        assert scheme.verify_width() == 32  # target-side verify tokens
        widths = [
            s.tokens_per_request
            for s in scheme.build_draft_generation_ops(_q8b("eagle3", EAGLE3_CONFIG))
            if s.op._name == "eagle3_attention"
        ]
        assert widths == [8] * 5

    def test_default_budget_is_everything_drafted_plus_root(self):
        scheme = _q8b("eagle3", EAGLE3_CONFIG, tree_shape=[4, 8, 8]).spec_scheme
        assert scheme.verify_width() == 21

    def test_budget_may_exceed_forwarded_width(self):
        # SGLang selects verify tokens from the logit-branch candidate pool,
        # which is larger than the forwarded width: (steps=3, topk=2) forwards
        # 6 nodes but num_draft_tokens=8 is a legal deployment.
        scheme = _q8b("eagle3", EAGLE3_CONFIG, tree_shape=[2, 2, 2], verify_token_budget=8).spec_scheme
        assert scheme.verify_width() == 8

    def test_budget_bounds(self):
        with pytest.raises(ValueError, match="verify_token_budget"):
            _q8b("eagle3", EAGLE3_CONFIG, tree_shape=[2, 2], verify_token_budget=1)


# Real values from RedHatAI/DeepSeek-V4-Flash-speculator.dflash config.json
# (speculators-style checkpoint; safetensors header decomposed 2026-08-07).
DFLASH_V4_CONFIG = {
    "architectures": ["DFlashDraftModel"],
    "speculators_model_type": "dflash",
    "block_size": 8,
    "draft_vocab_size": 32000,
    "aux_hidden_state_layer_ids": [3, 13, 23, 32, 42],
    "transformer_layer_config": {
        "model_type": "llama",
        "hc_mult": 4,
        "hidden_size": 4096,
        "intermediate_size": 2048,
        "num_hidden_layers": 5,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "sliding_window": 2048,
    },
}
DFLASH_V4_CKPT_BYTES = 3_607_596_760


class TestDFlashV4Speculators:
    def _v4(self, **params):
        cfg = sdk_config.ModelConfig(
            tp_size=1,
            pp_size=1,
            moe_tp_size=1,
            moe_ep_size=1,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            moe_quant_mode=common.MoEQuantMode.bfloat16,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            speculation=SpeculationConfig(kind="dflash", params=params, draft_config=DFLASH_V4_CONFIG),
        )
        return models.get_model("deepseek-ai/DeepSeek-V4-Flash", cfg, "vllm")

    def test_param_resolution(self):
        scheme = self._v4().spec_scheme
        assert isinstance(scheme, DFlashScheme)
        assert scheme.num_draft_tokens == 7  # block 8 = anchor + 7 drafts
        assert scheme.verify_width() == 8
        assert scheme.target_layer_ids == (3, 13, 23, 32, 42)
        assert scheme.injection_streams == 4  # mHC hc_mult streams
        assert scheme.draft_vocab_size == 32000
        assert scheme.owns_embed_and_head
        # MQA + tiny MLP + SWA-2048 draft geometry
        g = scheme.draft_geometry
        assert (g.num_heads, g.num_kv_heads, g.head_dim, g.inter_size) == (64, 1, 256, 2048)
        assert g.sliding_window == 2048

    def test_injection_fc_reads_hc_streams(self):
        model = self._v4()
        specs = model.spec_scheme.build_draft_generation_ops(model)
        by_name = {s.op._name: s for s in specs}
        # fc [4096, 81920]: 5 aux layers x hc_mult 4 x target h 4096
        assert by_name["dflash_main_proj"].op._k == 5 * 4 * 4096
        # reduced draft vocab head, not the target's 129k
        assert by_name["dflash_head_gemm"].op._n == 32000

    def test_weights_close_against_checkpoint(self):
        model = self._v4()
        w = model.spec_scheme.draft_weights_bytes(model)
        # blocks + fc + full-vocab embed (129280x4096) + 32k head
        assert w == pytest.approx(DFLASH_V4_CKPT_BYTES, rel=0.01)

    def test_draft_kv_window_capped(self):
        model = self._v4()
        scheme = model.spec_scheme
        assert scheme.draft_kv_bytes_per_sequence(model, 10_000) == scheme.draft_kv_bytes_per_sequence(model, 100_000)
        # 5 layers x 2048 window x 2(K+V) x 1 kv head x 256 head_dim x 2 B
        assert scheme.draft_kv_bytes_per_sequence(model, 100_000) == pytest.approx(5 * 2048 * 2 * 1 * 256 * 2)


class TestNgramTriggerRate:
    def test_measured_two_parameter_yield(self):
        # Measured on vLLM x H100 gsm8k greedy: p = 0.301, E+1|drafted = 2.15
        # -> per-round progress 1.347 (flat across c8/16/32).
        model = _q8b("ngram", None, num_speculative_tokens=8, trigger_rate=0.301)
        scheme = model.spec_scheme
        assert scheme.expected_progress(1.15) == pytest.approx(1.0 + 0.301 * 1.15)

    def test_default_is_always_draft(self):
        scheme = _q8b("ngram", None, num_speculative_tokens=8).spec_scheme
        assert scheme.trigger_rate == 1.0
        assert scheme.expected_progress(1.15) == pytest.approx(2.15)

    def test_trigger_rate_bounds(self):
        from aiconfigurator_core.sdk.speculation.ngram import NgramScheme

        with pytest.raises(ValueError, match="trigger_rate"):
            NgramScheme(num_speculative_tokens=8, trigger_rate=0.0)

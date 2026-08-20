# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DSpark scheme tests (DeepSeek-V4 first target).

Ground facts pinned from deepseek-ai/DeepSeek-V4-Flash-DSpark @62af8fff
(safetensors index) and vLLM models/deepseek_v4/nvidia/dspark.py:
3 full-width draft blocks (mtp.{0,1,2}); main_proj = 3h -> h; markov head
rank 256; draft attention window-capped (SWA 128); one parallel draft
forward of N query tokens per round; verify width N + 1.
"""

from __future__ import annotations

import pytest

from aiconfigurator_core.sdk import common, models
from aiconfigurator_core.sdk import config as sdk_config
from aiconfigurator_core.sdk.speculation import SpeculationConfig
from aiconfigurator_core.sdk.speculation.dspark import DSparkScheme

pytestmark = pytest.mark.unit

# Real values from the V4-Flash DSpark draft config.json.
DRAFT_CONFIG = {
    "dspark_block_size": 5,
    "dspark_target_layer_ids": [40, 41, 42],
    "dspark_markov_rank": 256,
    "dspark_noise_token_id": 128799,
}


def _spec_config(**params) -> SpeculationConfig:
    return SpeculationConfig(kind="dspark", params=params, draft_config=DRAFT_CONFIG)


def test_v4_config_requires_target_layer_ids():
    # An absent field must not silently price main_proj at width h.
    partial = {k: v for k, v in DRAFT_CONFIG.items() if k != "dspark_target_layer_ids"}
    with pytest.raises(ValueError, match="dspark_target_layer_ids"):
        DSparkScheme.from_configs(None, SpeculationConfig(kind="dspark", params={}, draft_config=partial))


def _model_config(speculation: SpeculationConfig | None = None) -> sdk_config.ModelConfig:
    return sdk_config.ModelConfig(
        tp_size=8,
        pp_size=1,
        moe_tp_size=1,
        moe_ep_size=8,
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        moe_quant_mode=common.MoEQuantMode.nvfp4,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.fp8,
        speculation=speculation,
    )


def _v4_model_with_dspark(**params):
    cfg = _model_config(speculation=_spec_config(**params))
    return models.get_model("deepseek-ai/DeepSeek-V4-Flash", cfg, "sglang")


class TestParamResolution:
    def test_defaults_from_draft_config(self):
        scheme = DSparkScheme.from_configs(_model_config(), _spec_config())
        assert scheme.num_draft_tokens == 5  # dspark_block_size
        assert scheme.num_draft_layers == 3
        assert scheme.markov_rank == 256
        assert scheme.verify_width() == 6

    def test_user_num_draft_tokens_overrides_block_size(self):
        scheme = DSparkScheme.from_configs(_model_config(), _spec_config(num_draft_tokens=7))
        assert scheme.num_draft_tokens == 7
        assert scheme.verify_width() == 8

    def test_missing_draft_config_raises(self):
        with pytest.raises(ValueError):
            DSparkScheme.from_configs(_model_config(), SpeculationConfig(kind="dspark", params={}))


class TestValidation:
    def test_attaches_to_v4_model(self):
        model = _v4_model_with_dspark(num_draft_tokens=7)
        assert isinstance(model.spec_scheme, DSparkScheme)
        assert model.spec_scheme.verify_width() == 8
        # Materialized width channel: the engine widens the decode batch by
        # (_nextn + 1) = verify_width. MTP LAYER scaling must stay off — op
        # counts were built with config.nextn=0 before materialization.
        assert model._nextn == 7
        assert getattr(model, "_mtp_scale_factor", 1.0) == pytest.approx(1.0)
        assert any(op._name.startswith("draft_") for op in model.generation_ops)

    def test_rejects_non_v4_family(self):
        cfg = sdk_config.ModelConfig(
            tp_size=1,
            pp_size=1,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            speculation=_spec_config(),
        )
        with pytest.raises(ValueError):
            models.get_model("Qwen/Qwen3-32B", cfg, "trtllm")


class TestDraftOpGraph:
    def test_generation_op_inventory(self):
        model = _v4_model_with_dspark(num_draft_tokens=7)
        scheme = model.spec_scheme
        specs = scheme.build_draft_generation_ops(model)
        names = [s.op._name for s in specs]

        assert "dspark_main_proj" in names
        assert "dspark_attention" in names
        assert "dspark_head_gemm" in names
        assert "dspark_markov_bias_gemm" in names

        by_name = {s.op._name: s for s in specs}
        # Backbone forward runs at the draft block width N.
        assert by_name["dspark_attention"].tokens_per_request == 7
        # Injection projection runs once per request per round.
        assert by_name["dspark_main_proj"].tokens_per_request == 1
        # Draft attention KV is window-capped: pinned s, independent of target isl.
        overrides = by_name["dspark_attention"].query_overrides
        assert overrides is not None and overrides["s"] <= 128 + 7
        # Markov chain: N sequential steps folded as count=N on one op.
        assert by_name["dspark_markov_bias_gemm"].op._scale_factor == 7

    def test_context_precompute_ops_exist(self):
        model = _v4_model_with_dspark()
        specs = model.spec_scheme.build_draft_context_ops(model)
        assert specs, "DSpark needs a context-KV precompute pass"
        names = [s.op._name for s in specs]
        assert any("attention" in n for n in names)


class TestAccounting:
    def test_draft_weights_cross_check_against_index_diff(self):
        """Unique draft bytes ~= 10.9 GB (3 x 3.635 GB from the safetensors
        index diff at revision 62af8fff).

        draft_weights_bytes() returns PER-GPU resident bytes: sharded ops
        (attention/MoE/shared/router, tp8/ep8) contribute 1/8 of their
        unique bytes; replicated ops (main_proj, markov head) contribute in
        full on every GPU. Reconstruct the unique total accordingly.
        """
        model = _v4_model_with_dspark()
        per_gpu = model.spec_scheme.draft_weights_bytes(model)

        h = model._hidden_size
        vocab = model._vocab_size
        rank = model.spec_scheme.markov_rank
        # main_proj: fp8 (1 B/param) 3h -> h; markov: two vocab x rank bf16.
        replicated = h * (3 * h) * 1.0 + 2 * vocab * rank * 2.0
        unique_total = (per_gpu - replicated) * 8 + replicated
        assert unique_total == pytest.approx(10.9e9, rel=0.15)

    def test_draft_kv_is_window_capped(self):
        model = _v4_model_with_dspark()
        scheme = model.spec_scheme
        short = scheme.draft_kv_bytes_per_sequence(model, 64)
        capped_a = scheme.draft_kv_bytes_per_sequence(model, 10_000)
        capped_b = scheme.draft_kv_bytes_per_sequence(model, 1_000_000)
        assert 0 < short < capped_a
        assert capped_a == capped_b  # window cap
        # 3 layers x 128 window x head_dim 512 x 1 byte (fp8) minimum
        assert capped_a >= 3 * 128 * 512 * 1

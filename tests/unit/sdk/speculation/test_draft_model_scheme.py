# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone draft-model scheme (classic two-model speculative decoding).

The draft is instantiated through the model registry, so its op graph,
weights and KV accounting come from the draft model itself — validated
here with the real Qwen/Qwen3-0.6B drafting for Qwen/Qwen3-8B."""

from __future__ import annotations

import pytest

from aiconfigurator_core.sdk import common, models
from aiconfigurator_core.sdk import config as sdk_config
from aiconfigurator_core.sdk.speculation import SpeculationConfig
from aiconfigurator_core.sdk.speculation.draft_model import DraftModelScheme

pytestmark = pytest.mark.unit


def _q8b(**params):
    cfg = sdk_config.ModelConfig(
        tp_size=1,
        pp_size=1,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        speculation=SpeculationConfig(kind="draft_model", params=params, draft_model_path="Qwen/Qwen3-0.6B"),
    )
    return models.get_model("Qwen/Qwen3-8B", cfg, "vllm")


class TestDraftModelScheme:
    def test_param_resolution(self):
        model = _q8b(num_speculative_tokens=3)
        scheme = model.spec_scheme
        assert isinstance(scheme, DraftModelScheme)
        assert scheme.verify_width() == 4
        assert scheme.draft_model_path == "Qwen/Qwen3-0.6B"

    def test_k_required(self):
        with pytest.raises(ValueError, match="num_speculative_tokens"):
            _q8b()

    def test_draft_ops_are_the_full_small_model_times_k(self):
        model = _q8b(num_speculative_tokens=3)
        specs = model.spec_scheme.build_draft_generation_ops(model)
        by_name = {s.op._name: s for s in specs}
        # Full model graph: embedding + layers + logits head all present.
        assert "generation_embedding" in by_name
        assert "generation_logits_gemm" in by_name
        # K sequential steps folded into op counts: 28 layers x K=3.
        assert by_name["generation_qkv_gemm"].op._scale_factor == 28 * 3
        # Draft geometry is the 0.6B's, not the target's: qkv n at h=1024,
        # 16 q heads + 2x8 kv heads, head_dim 128.
        assert by_name["generation_qkv_gemm"].op._k == 1024
        assert all(s.tokens_per_request == 1 for s in specs)

    def test_weights_are_the_draft_checkpoint(self):
        model = _q8b(num_speculative_tokens=3)
        w = model.spec_scheme.draft_weights_bytes(model)
        # Qwen3-0.6B bf16 ~= 1.50 GB; K-scaling must not inflate weights.
        assert w == pytest.approx(1.503e9, rel=0.02)
        model5 = _q8b(num_speculative_tokens=5)
        w5 = model5.spec_scheme.draft_weights_bytes(model5)
        assert w5 == pytest.approx(w)

    def test_kv_uses_draft_models_own_accounting(self):
        model = _q8b(num_speculative_tokens=3)
        kv = model.spec_scheme.draft_kv_bytes_per_sequence(model, 1_000)
        # 28 layers x 2(K+V) x 8 kv heads x 128 head_dim x 2 B (bf16)
        assert kv == pytest.approx(28 * 1_000 * 2 * 8 * 128 * 2)

    def test_context_ops_prefill_the_draft(self):
        model = _q8b(num_speculative_tokens=3)
        specs = model.spec_scheme.build_draft_context_ops(model)
        names = [s.op._name for s in specs]
        assert "context_attention" in names

    def test_unsupported_backend_rejected(self):
        cfg = sdk_config.ModelConfig(
            tp_size=1,
            pp_size=1,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            speculation=SpeculationConfig(
                kind="draft_model",
                params={"num_speculative_tokens": 3},
                draft_model_path="Qwen/Qwen3-0.6B",
            ),
        )
        with pytest.raises(ValueError, match="backends"):
            models.get_model("Qwen/Qwen3-8B", cfg, "trtllm")

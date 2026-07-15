# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for SDK utility functions.

Tests HuggingFace config parsing and model config retrieval.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.models import Gemma4MixModel, HybridMoEModel
from aiconfigurator.sdk.utils import (
    _parse_hf_config_json,
    enumerate_parallel_config,
    enumerate_ttft_tpot_constraints,
    get_model_config_from_model_path,
)

pytestmark = pytest.mark.unit


class TestParseHFConfig:
    """Test HuggingFace config parsing."""

    def test_parse_llama_config(self):
        """Test parsing a Llama model config."""
        config = {
            "architectures": ["LlamaForCausalLM"],
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "intermediate_size": 14336,
            "vocab_size": 128256,
            "max_position_embeddings": 131072,
            "num_experts_per_tok": 0,
        }

        result = _parse_hf_config_json(config)

        assert result["architecture"] == "LlamaForCausalLM"  # architecture
        assert result["layers"] == 32  # num_layers
        assert result["n"] == 32  # num_heads
        assert result["n_kv"] == 8  # num_kv_heads
        assert result["hidden_size"] == 4096  # hidden_size
        assert result["inter_size"] == 14336  # inter_size
        assert result["vocab"] == 128256  # vocab_size

    def test_parse_moe_config(self):
        """Test parsing a MoE model config."""
        config = {
            "architectures": ["MixtralForCausalLM"],
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "intermediate_size": 14336,
            "vocab_size": 32000,
            "max_position_embeddings": 32768,
            "num_experts_per_tok": 2,
            "num_local_experts": 8,
            "moe_intermediate_size": 14336,
        }

        result = _parse_hf_config_json(config)

        assert result["architecture"] == "MixtralForCausalLM"  # architecture
        assert result["topk"] == 2  # topk
        assert result["num_experts"] == 8  # num_experts
        assert result["moe_inter_size"] == 14336  # moe_inter_size

    def test_parse_deepseek_config(self):
        """Test parsing a DeepSeek model config."""
        config = {
            "architectures": ["DeepseekV3ForCausalLM"],
            "num_hidden_layers": 61,
            "num_key_value_heads": 128,
            "hidden_size": 7168,
            "num_attention_heads": 128,
            "intermediate_size": 18432,
            "vocab_size": 129280,
            "max_position_embeddings": 4096,
            "num_experts_per_tok": 8,
            "n_routed_experts": 256,
            "moe_intermediate_size": 2048,
        }

        result = _parse_hf_config_json(config)

        assert result["architecture"] == "DeepseekV3ForCausalLM"  # architecture
        assert result["num_experts"] == 256  # num_experts from n_routed_experts

    def test_parse_config_with_head_dim(self):
        """Test parsing config that explicitly provides head_dim."""
        config = {
            "architectures": ["LlamaForCausalLM"],
            "num_hidden_layers": 64,
            "num_key_value_heads": 8,
            "hidden_size": 5120,
            "num_attention_heads": 64,
            "intermediate_size": 25600,
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
            "num_experts_per_tok": 0,
            "head_dim": 80,  # Explicit head_dim
        }

        result = _parse_hf_config_json(config)

        assert result["d"] == 80  # head_dim

    def test_parse_nemotronh_config(self):
        """Test parsing a NemotronH hybrid model config (Mamba + MoE + Transformer)."""
        config = {
            "architectures": ["NemotronHForCausalLM"],
            "num_hidden_layers": 52,
            "num_key_value_heads": 2,
            "hidden_size": 2688,
            "num_attention_heads": 32,
            "intermediate_size": 1856,
            "vocab_size": 131072,
            "max_position_embeddings": 262144,
            "num_experts_per_tok": 6,
            "n_routed_experts": 128,
            "moe_intermediate_size": 1856,
            "head_dim": 128,
            # NemotronH-specific fields
            "hybrid_override_pattern": "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
            "mamba_num_heads": 64,
            "mamba_head_dim": 64,
            "ssm_state_size": 128,
            "conv_kernel": 4,
            "n_groups": 8,
            "chunk_size": 128,
            "moe_shared_expert_intermediate_size": 3712,
        }

        result = _parse_hf_config_json(config)

        assert result["architecture"] == "NemotronHForCausalLM"  # architecture
        assert result["layers"] == 52  # num_layers
        assert result["hidden_size"] == 2688  # hidden_size
        assert result["topk"] == 6  # topk (num_experts_per_tok)
        assert result["num_experts"] == 128  # num_experts (n_routed_experts)
        # extra_params should be NemotronHConfig
        extra_params = result["extra_params"]
        assert extra_params is not None
        assert hasattr(extra_params, "hybrid_override_pattern")
        assert extra_params.hybrid_override_pattern == "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
        assert extra_params.mamba_num_heads == 64
        assert extra_params.moe_shared_expert_intermediate_size == 3712

    def test_parse_nemotronh_without_moe(self):
        """Test parsing a NemotronH config without MoE layers (no 'E' in pattern)."""
        config = {
            "architectures": ["NemotronHForCausalLM"],
            "num_hidden_layers": 118,
            "num_key_value_heads": 8,
            "hidden_size": 8192,
            "num_attention_heads": 64,
            "intermediate_size": 32768,
            "vocab_size": 131072,
            "max_position_embeddings": 8192,
            "attention_head_dim": 128,  # Uses attention_head_dim instead of head_dim
            # NemotronH-specific fields (no MoE)
            "hybrid_override_pattern": "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-",
            "mamba_num_heads": 256,
            "mamba_head_dim": 64,
            "ssm_state_size": 256,
            "conv_kernel": 4,
            "n_groups": 8,
            "chunk_size": 128,
        }

        result = _parse_hf_config_json(config)

        assert result["architecture"] == "NemotronHForCausalLM"  # architecture
        assert result["layers"] == 118  # num_layers
        assert result["hidden_size"] == 8192  # hidden_size
        assert result["d"] == 128  # head_dim from attention_head_dim
        # extra_params should be NemotronHConfig with moe_shared_expert_intermediate_size=0
        extra_params = result["extra_params"]
        assert extra_params is not None
        assert "E" not in extra_params.hybrid_override_pattern  # No MoE layers
        assert extra_params.moe_shared_expert_intermediate_size == 0

    def test_parse_nemotronh_layers_block_type_config(self):
        """Test parsing NemotronH configs that use explicit layer block names."""
        config = {
            "architectures": ["NemotronHForCausalLM"],
            "num_key_value_heads": 2,
            "hidden_size": 8192,
            "num_attention_heads": 64,
            "intermediate_size": 5120,
            "vocab_size": 131072,
            "max_position_embeddings": 262144,
            "num_experts_per_tok": 22,
            "n_routed_experts": 512,
            "moe_intermediate_size": 5120,
            "head_dim": 128,
            "layers_block_type": ["mamba", "moe", "mamba", "attention", "mlp"],
            "mamba_num_heads": 256,
            "mamba_head_dim": 64,
            "ssm_state_size": 128,
            "conv_kernel": 4,
            "n_groups": 8,
            "chunk_size": 128,
            "moe_shared_expert_intermediate_size": 10240,
        }

        result = _parse_hf_config_json(config)

        assert result["architecture"] == "NemotronHForCausalLM"
        assert result["layers"] == 5
        assert result["hidden_size"] == 8192
        assert result["topk"] == 22
        assert result["num_experts"] == 512
        extra_params = result["extra_params"]
        assert isinstance(extra_params, common.NemotronHConfig)
        assert extra_params.hybrid_override_pattern == "MEM*-"
        assert extra_params.mamba_num_heads == 256
        assert extra_params.moe_shared_expert_intermediate_size == 10240

    def test_parse_qwen35_dense_config(self):
        """Test parsing Qwen3.5-27B (dense hybrid) config → Qwen35Config."""
        # Mimics Qwen/Qwen3.5-27B HF config structure (params nested under text_config).
        # 64 layers: 48 linear_attention + 16 full_attention (3:1 ratio).
        layer_types = ["linear_attention"] * 3 + ["full_attention"]
        layer_types = layer_types * 16  # 64 layers total (48 GDN + 16 GQA)
        config = {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {
                "num_hidden_layers": 64,
                "num_attention_heads": 24,
                "num_key_value_heads": 4,
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "vocab_size": 151936,
                "max_position_embeddings": 32768,
                "head_dim": 256,
                "layer_types": layer_types,
                "linear_num_key_heads": 16,
                "linear_key_head_dim": 128,
                "linear_num_value_heads": 48,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
            },
        }

        result = _parse_hf_config_json(config)

        assert result["architecture"] == "Qwen3_5ForConditionalGeneration"
        assert result["layers"] == 64
        assert result["hidden_size"] == 5120
        assert result["inter_size"] == 17408
        assert result["n"] == 24
        assert result["n_kv"] == 4
        assert result["d"] == 256

        extra_params = result["extra_params"]
        assert isinstance(extra_params, common.Qwen35Config)
        assert len(extra_params.layer_types) == 64
        assert extra_params.layer_types.count("linear_attention") == 48
        assert extra_params.layer_types.count("full_attention") == 16
        assert extra_params.linear_num_key_heads == 16
        assert extra_params.linear_key_head_dim == 128
        assert extra_params.linear_num_value_heads == 48
        assert extra_params.linear_value_head_dim == 128
        assert extra_params.linear_conv_kernel_dim == 4
        # Dense model: no MoE routing
        assert extra_params.topk == 0
        assert extra_params.num_experts == 0
        # For dense models moe_inter_size falls back to intermediate_size
        assert extra_params.moe_inter_size == 17408
        assert extra_params.shared_expert_inter_size == 0

    def test_parse_qwen35_moe_config(self):
        """Test parsing Qwen3.5-35B-A3B (MoE hybrid) config → Qwen35Config with MoE fields."""
        # 40 layers: 30 linear_attention + 10 full_attention (3:1 ratio).
        layer_types = ["linear_attention"] * 3 + ["full_attention"]
        layer_types = layer_types * 10  # 40 layers total (30 GDN + 10 GQA)
        config = {
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "text_config": {
                "num_hidden_layers": 40,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "hidden_size": 2048,
                "vocab_size": 151936,
                "max_position_embeddings": 32768,
                "head_dim": 256,
                "layer_types": layer_types,
                "linear_num_key_heads": 16,
                "linear_key_head_dim": 128,
                "linear_num_value_heads": 32,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                # MoE fields
                "num_experts_per_tok": 8,
                "num_experts": 256,
                "moe_intermediate_size": 512,
                "shared_expert_intermediate_size": 512,
            },
        }

        result = _parse_hf_config_json(config)

        assert result["architecture"] == "Qwen3_5MoeForConditionalGeneration"
        assert result["layers"] == 40
        assert result["hidden_size"] == 2048
        assert result["topk"] == 8
        assert result["num_experts"] == 256

        extra_params = result["extra_params"]
        assert isinstance(extra_params, common.Qwen35Config)
        assert len(extra_params.layer_types) == 40
        assert extra_params.layer_types.count("linear_attention") == 30
        assert extra_params.layer_types.count("full_attention") == 10
        assert extra_params.linear_num_value_heads == 32
        assert extra_params.topk == 8
        assert extra_params.num_experts == 256
        assert extra_params.moe_inter_size == 512
        assert extra_params.shared_expert_inter_size == 512

    def test_parse_qwen35_layer_types_length_mismatch_raises(self):
        """Test that mismatched layer_types length raises ValueError."""
        layer_types = ["linear_attention"] * 3 + ["full_attention"]  # 4 entries, not 64
        config = {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {
                "num_hidden_layers": 64,
                "num_attention_heads": 24,
                "num_key_value_heads": 4,
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "vocab_size": 151936,
                "max_position_embeddings": 32768,
                "head_dim": 256,
                "layer_types": layer_types,  # length 4, not 64 → should raise
                "linear_num_key_heads": 16,
                "linear_key_head_dim": 128,
                "linear_num_value_heads": 48,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
            },
        }
        with pytest.raises(ValueError, match="layer_types length"):
            _parse_hf_config_json(config)

    def test_parse_llama4_scout_config(self):
        """Test Llama 4 Scout (VLM, step=1: all-MoE) → HybridMoEConfig with alternating attn pattern."""
        config = {
            "architectures": ["Llama4ForConditionalGeneration"],
            "model_type": "llama4",
            "text_config": {
                "num_hidden_layers": 48,
                "hidden_size": 5120,
                "num_attention_heads": 40,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "intermediate_size": 8192,
                "intermediate_size_mlp": 16384,
                "vocab_size": 202048,
                "max_position_embeddings": 10485760,
                "num_experts_per_tok": 1,
                "num_local_experts": 16,
                "interleave_moe_layer_step": 1,
                "attention_chunk_size": 8192,
            },
        }

        result = _parse_hf_config_json(config)

        assert result["architecture"] == "Llama4ForConditionalGeneration"
        assert result["layers"] == 48
        assert result["num_experts"] == 16
        assert result["moe_inter_size"] == 8192
        cfg = result["extra_params"]
        assert cfg is not None
        # step=1: all layers MoE
        assert all(m == 1 for m in cfg.moe_layer_freq)
        # alternating local(0)/global(1): even=0, odd=1
        assert cfg.attn_layer_pattern == tuple(i % 2 for i in range(48))
        assert cfg.sliding_window_size == 8192
        assert cfg.dense_inter_size == 16384
        # Llama 4 uses same dims for all layers → all four dim fields are 0
        assert cfg.swa_num_kv_heads == 0
        assert cfg.swa_head_dim == 0

    def test_parse_llama4_maverick_config(self):
        """Test Llama 4 Maverick (VLM, step=2: alternating MoE/dense) → HybridMoEConfig."""
        config = {
            "architectures": ["Llama4ForConditionalGeneration"],
            "model_type": "llama4",
            "text_config": {
                "num_hidden_layers": 48,
                "hidden_size": 5120,
                "num_attention_heads": 40,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "intermediate_size": 8192,
                "intermediate_size_mlp": 16384,
                "vocab_size": 202048,
                "max_position_embeddings": 1048576,
                "num_experts_per_tok": 1,
                "num_local_experts": 128,
                "interleave_moe_layer_step": 2,
                "attention_chunk_size": 8192,
            },
        }

        result = _parse_hf_config_json(config)

        assert result["architecture"] == "Llama4ForConditionalGeneration"
        assert result["num_experts"] == 128
        cfg = result["extra_params"]
        # step=2: odd layers are MoE (1), even layers are dense (0)
        assert sum(cfg.moe_layer_freq) == 24  # 24 MoE layers
        assert cfg.moe_layer_freq.count(0) == 24  # 24 dense layers
        assert cfg.dense_inter_size == 16384

    def test_parse_mimov2flash_config(self):
        """Test MiMo-V2-Flash (explicit per-layer patterns, different SWA/global dims) → HybridMoEConfig."""
        hybrid_pattern = [0, 1, 1, 1, 1, 0, 1, 1, 1, 1]  # 10-layer test
        moe_freq = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        hf_config = {
            "architectures": ["MiMoV2FlashForCausalLM"],
            "num_hidden_layers": 10,
            "hidden_size": 4096,
            "num_attention_heads": 64,
            "num_key_value_heads": 4,
            "head_dim": 192,
            "v_head_dim": 128,
            "intermediate_size": 16384,
            "vocab_size": 152576,
            "max_position_embeddings": 262144,
            "num_experts_per_tok": 8,
            "num_local_experts": 64,
            "moe_intermediate_size": 2048,
            "hybrid_layer_pattern": hybrid_pattern,
            "moe_layer_freq": moe_freq,
            "swa_num_key_value_heads": 8,
            "swa_head_dim": 192,
            "swa_v_head_dim": 128,
            "sliding_window_size": 128,
        }

        result = _parse_hf_config_json(hf_config)

        assert result["architecture"] == "MiMoV2FlashForCausalLM"
        assert result["layers"] == 10
        cfg = result["extra_params"]
        assert cfg is not None
        assert cfg.attn_layer_pattern == tuple(hybrid_pattern)
        assert cfg.moe_layer_freq == tuple(moe_freq)
        assert cfg.swa_num_kv_heads == 8
        assert cfg.swa_head_dim == 192
        assert cfg.swa_v_head_dim == 128
        assert cfg.global_v_head_dim == 128  # from v_head_dim
        assert cfg.sliding_window_size == 128
        assert cfg.dense_inter_size == 0  # dense layers use model-level inter_size

    def test_mimo_pattern_length_mismatch_raises(self):
        """Test that mismatched hybrid pattern length raises ValueError."""
        hf_config = {
            "architectures": ["MiMoV2FlashForCausalLM"],
            "num_hidden_layers": 10,
            "hidden_size": 4096,
            "num_attention_heads": 64,
            "num_key_value_heads": 4,
            "head_dim": 192,
            "intermediate_size": 16384,
            "vocab_size": 152576,
            "max_position_embeddings": 262144,
            "num_experts_per_tok": 8,
            "num_local_experts": 64,
            "moe_intermediate_size": 2048,
            "hybrid_layer_pattern": [0, 1, 1],  # wrong length
            "moe_layer_freq": [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
        with pytest.raises(ValueError, match="pattern length mismatch"):
            _parse_hf_config_json(hf_config)

    def test_llama4_invalid_step_raises(self):
        """Test that interleave_moe_layer_step <= 0 raises ValueError."""
        hf_config = {
            "architectures": ["Llama4ForConditionalGeneration"],
            "text_config": {
                "num_hidden_layers": 8,
                "hidden_size": 1024,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 128,
                "intermediate_size": 2048,
                "vocab_size": 10000,
                "max_position_embeddings": 4096,
                "num_experts_per_tok": 1,
                "num_local_experts": 4,
                "interleave_moe_layer_step": 0,
                "attention_chunk_size": 512,
            },
        }
        with pytest.raises(ValueError, match="positive integer"):
            _parse_hf_config_json(hf_config)

    @staticmethod
    def _gemma4_text_config(layer_types):
        """Minimal valid Gemma 4 HF config wrapping the text_config branch."""
        return {
            "architectures": ["Gemma4ForConditionalGeneration"],
            "text_config": {
                "num_hidden_layers": len(layer_types),
                "hidden_size": 2816,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "num_global_key_value_heads": 2,
                "head_dim": 256,
                "global_head_dim": 512,
                "intermediate_size": 2112,
                "moe_intermediate_size": 704,
                "vocab_size": 262144,
                "max_position_embeddings": 262144,
                "top_k_experts": 8,
                "num_experts": 128,
                "layer_types": layer_types,
                "sliding_window": 1024,
                "attention_k_eq_v": True,
            },
        }

    def test_parse_gemma4_config(self):
        """Real google/gemma-4-26B-A4B shape: 5:1 SWA:global, k_eq_v on global, top_k_experts."""
        # Exact pattern from config.json: [SWA, SWA, SWA, SWA, SWA, global] x 5 = 30 layers.
        layer_types = ["sliding_attention"] * 5 + ["full_attention"]
        layer_types = layer_types * 5
        hf_config = self._gemma4_text_config(layer_types)

        result = _parse_hf_config_json(hf_config)

        assert result["architecture"] == "Gemma4ForConditionalGeneration"
        assert result["layers"] == 30
        assert result["n"] == 16
        assert result["n_kv"] == 8  # SWA default at top level
        assert result["d"] == 256  # SWA default at top level
        assert result["hidden_size"] == 2816
        assert result["inter_size"] == 2112  # shared dense MLP intermediate
        assert result["moe_inter_size"] == 704  # per routed expert
        assert result["topk"] == 8  # picked up from top_k_experts fallback
        assert result["num_experts"] == 128
        assert result["vocab"] == 262144

        cfg = result["extra_params"]
        assert isinstance(cfg, common.Gemma4MixConfig)
        assert cfg.layer_types == tuple(layer_types)
        assert cfg.layer_types.count("sliding_attention") == 25
        assert cfg.layer_types.count("full_attention") == 5
        assert cfg.layer_types[-1] == "full_attention"
        assert cfg.swa_num_kv_heads == 8
        assert cfg.swa_head_dim == 256
        assert cfg.global_num_kv_heads == 2
        assert cfg.global_head_dim == 512
        assert cfg.sliding_window_size == 1024
        assert cfg.attention_k_eq_v is True

    def test_gemma4_layer_types_length_mismatch_raises(self):
        """layer_types length must equal num_hidden_layers."""
        hf_config = self._gemma4_text_config(["sliding_attention"] * 5)
        hf_config["text_config"]["num_hidden_layers"] = 30  # model claims 30, config has 5
        with pytest.raises(ValueError, match="layer_types length"):
            _parse_hf_config_json(hf_config)

    def test_gemma4_invalid_layer_type_raises(self):
        """layer_types must contain only sliding_attention / full_attention."""
        hf_config = self._gemma4_text_config(["sliding_attention", "linear_attention", "full_attention"])
        with pytest.raises(ValueError, match="must contain only"):
            _parse_hf_config_json(hf_config)


class TestGemma4MixModelBuilder:
    """Builder-level tests that verify Gemma4MixModel wiring through set_gemma4_config."""

    @staticmethod
    def _make_model_config(tp_size=1, moe_tp_size=1, moe_ep_size=1):
        return config.ModelConfig(
            tp_size=tp_size,
            pp_size=1,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
            attention_dp_size=1,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            moe_quant_mode=common.MoEQuantMode.bfloat16,
        )

    @staticmethod
    def _make_model(model_config, layer_types=None, attention_k_eq_v=True):
        """Build a Gemma4MixModel at the gemma-4-26B-A4B shape (or a custom layer pattern)."""
        if layer_types is None:
            layer_types = (["sliding_attention"] * 5 + ["full_attention"]) * 5
        cfg = common.Gemma4MixConfig(
            layer_types=tuple(layer_types),
            swa_num_kv_heads=8,
            swa_head_dim=256,
            global_num_kv_heads=2,
            global_head_dim=512,
            sliding_window_size=1024,
            attention_k_eq_v=attention_k_eq_v,
        )
        model = Gemma4MixModel(
            8,  # topk
            128,  # num_experts
            704,  # moe_inter_size
            "google/gemma-4-26B-A4B",  # model_path
            "GEMMA4MIX",  # model_family
            "Gemma4ForConditionalGeneration",  # architecture
            len(layer_types),  # num_layers
            16,  # num_heads
            8,  # num_kv_heads (SWA default)
            256,  # head_size (SWA default)
            2816,  # hidden_size
            2112,  # inter_size (shared MLP)
            262144,  # vocab_size
            262144,  # context_length
            model_config,
            None,  # extra_params (we use set_gemma4_config instead)
        )
        model.set_gemma4_config(cfg)
        return model

    @staticmethod
    def _make_dense_model():
        """Build a dense Gemma 4 variant (e.g. gemma-4-31B-it: topk=0, num_experts=None):
        every layer is just shared dense MLP + attention, no routed-MoE block."""
        layer_types = (["sliding_attention"] * 5 + ["full_attention"]) * 2  # 12 layers
        cfg = common.Gemma4MixConfig(
            layer_types=tuple(layer_types),
            swa_num_kv_heads=16,
            swa_head_dim=256,
            global_num_kv_heads=4,
            global_head_dim=512,
            sliding_window_size=1024,
            attention_k_eq_v=True,
        )
        model = Gemma4MixModel(
            0,  # topk = 0 (dense)
            None,  # num_experts = None (dense)
            21504,  # moe_inter_size (unused for dense, but kept aligned with HF config)
            "google/gemma-4-31B-it",
            "GEMMA4MIX",
            "Gemma4ForConditionalGeneration",
            len(layer_types),
            32,
            16,
            256,
            5376,
            21504,
            262144,
            262144,
            TestGemma4MixModelBuilder._make_model_config(),  # tp=1, moe_tp=1, moe_ep=1
            None,
        )
        model.set_gemma4_config(cfg)
        return model

    def test_builds_both_layer_recipes(self):
        """30-layer 5:1 SWA:global pattern emits both recipes with shared-MLP + MoE ops."""
        model = self._make_model(self._make_model_config())
        op_names = {op._name for op in model.context_ops}
        # SWA recipe
        assert "context_swa_qkv_gemm" in op_names
        assert "context_swa_shared_mlp_gate_up_gemm" in op_names
        assert "context_swa_shared_mlp_down_gemm" in op_names
        assert "context_swa_router_gemm" in op_names
        # Global recipe
        assert "context_global_qkv_gemm" in op_names
        assert "context_global_shared_mlp_gate_up_gemm" in op_names
        assert "context_global_router_gemm" in op_names

    def test_global_qkv_omits_v_when_k_eq_v(self):
        """attention_k_eq_v=True drops V from the global-layer QKV-GEMM output width."""
        model = self._make_model(self._make_model_config(), attention_k_eq_v=True)
        global_qkv = next(op for op in model.context_ops if op._name == "context_global_qkv_gemm")
        swa_qkv = next(op for op in model.context_ops if op._name == "context_swa_qkv_gemm")
        # Global: Q (16*512) + K (2*512) = 9216, no V buffer.
        assert global_qkv._n == 16 * 512 + 2 * 512
        # SWA always has separate K and V: Q (16*256) + K (8*256) + V (8*256) = 8192.
        assert swa_qkv._n == 16 * 256 + 8 * 256 * 2

    def test_global_qkv_includes_v_when_k_eq_v_false(self):
        """attention_k_eq_v=False (defensive default) keeps V in the global QKV-out width."""
        model = self._make_model(self._make_model_config(), layer_types=["full_attention"], attention_k_eq_v=False)
        qkv = next(op for op in model.context_ops if op._name == "context_global_qkv_gemm")
        # With V re-enabled: Q + K + V = 16*512 + 2*512*2 = 10240.
        assert qkv._n == 16 * 512 + 2 * 512 * 2

    def test_kvcache_bytes_window_caps_swa(self):
        """SWA contribution caps at sliding_window_size; global grows linearly with seq_len."""
        model = self._make_model(self._make_model_config())

        # bf16, TP=1 hand derivation:
        #   per-token SWA layer = 8 KV * 256 dim * 2 (K+V) * 2 bytes = 8192
        #   per-token global layer = 2 KV * 512 dim * 2 (K+V) * 2 bytes = 4096
        def expected(seq_len):
            swa_seq = min(seq_len, 1024)
            return 25 * 8192 * swa_seq + 5 * 4096 * seq_len

        for seq_len in (512, 1024, 4096, 65536, 262144):
            assert model.get_kvcache_bytes_per_sequence(seq_len) == expected(seq_len)

    def test_kvcache_bytes_at_256k_in_expected_range(self):
        """Architectural payoff: full 256K context KV fits in ~5 GiB on a single GPU at bf16."""
        model = self._make_model(self._make_model_config())
        gib = model.get_kvcache_bytes_per_sequence(262144) / (1024**3)
        assert 4.5 <= gib <= 5.5, f"expected ~5.2 GiB, got {gib:.3f}"

    def test_kvcache_bytes_tp_sharding(self):
        """KV heads round up per GPU; TP=8 with KV-heads (8,2) → (1,1) per GPU."""
        model = self._make_model(self._make_model_config(tp_size=8, moe_ep_size=8))
        seq_len = 262144
        # SWA: 8/8 = 1 KV per GPU.  Global: ceil(2/8) = 1 KV per GPU.
        # SWA per-token per-layer per-GPU = 1*256*2*2 = 1024.  Global = 1*512*2*2 = 2048.
        expected = 25 * 1024 * 1024 + 5 * 2048 * seq_len
        assert model.get_kvcache_bytes_per_sequence(seq_len) == expected

    def test_get_kvcache_max_tokens_follows_window_capped_curve(self):
        """The capacity inverse caps SWA layers at the window instead of using the
        (larger) seq_len=1 slope, so it fits far more tokens past the window."""
        model = self._make_model(self._make_model_config())
        budget = model.get_kvcache_bytes_per_sequence(65536)  # >> 1024 window
        tokens = model.get_kvcache_max_tokens(budget)
        # Exact monotonic inverse: `tokens` fits, one more token does not.
        assert tokens == 65536
        assert model.get_kvcache_bytes_per_sequence(tokens) <= budget
        assert model.get_kvcache_bytes_per_sequence(tokens + 1) > budget
        # The seq_len=1 extrapolation under-counts: SWA layers are charged forever.
        per_token = model.get_kvcache_bytes_per_sequence(1)
        assert tokens > int(budget // per_token)

    def test_get_kvcache_max_tokens_linear_below_window(self):
        """Below the window every layer still grows, so the inverse is plain
        floor-division by the per-token size."""
        model = self._make_model(self._make_model_config())
        per_token = model.get_kvcache_bytes_per_sequence(1)
        budget = model.get_kvcache_bytes_per_sequence(512)  # < 1024 window
        assert model.get_kvcache_max_tokens(budget) == int(budget // per_token) == 512

    def test_get_kvcache_max_tokens_saturated_caps_at_context_length(self):
        """A fully window-capped cache (all SWA, no global layers) saturates: KV
        stops growing past the window, so memory never binds and capacity is
        bounded by the model context length -- not an arbitrary doubling step."""
        # No full_attention layer -> every layer caps at the 1024 window.
        model = self._make_model(self._make_model_config(), layer_types=["sliding_attention"] * 6)
        # KV is flat past the window: confirm we are actually in the saturated regime.
        assert model.get_kvcache_bytes_per_sequence(2048) == model.get_kvcache_bytes_per_sequence(1024)
        saturated = model.get_kvcache_bytes_per_sequence(model._context_length)
        # Any budget at/above the saturated size returns the context length, and is
        # stable across wildly different (large) budgets rather than tracking 2^k.
        assert model.get_kvcache_max_tokens(saturated) == model._context_length == 262144
        assert model.get_kvcache_max_tokens(saturated * 1000) == model._context_length

    def test_set_gemma4_config_rejects_wrong_type(self):
        """Passing a HybridMoEConfig (or any non-Gemma4MixConfig) raises."""
        model = Gemma4MixModel(
            8,
            128,
            704,
            "test",
            "GEMMA4MIX",
            "Gemma4ForConditionalGeneration",
            2,
            16,
            8,
            256,
            2816,
            2112,
            262144,
            262144,
            self._make_model_config(),
            None,
        )
        with pytest.raises(ValueError, match="requires a Gemma4MixConfig"):
            model.set_gemma4_config(common.HybridMoEConfig(attn_layer_pattern=(0, 1), moe_layer_freq=(1, 1)))

    def test_set_gemma4_config_rejects_wrong_layer_count(self):
        """layer_types length must match num_layers passed at construction."""
        model = Gemma4MixModel(
            8,
            128,
            704,
            "test",
            "GEMMA4MIX",
            "Gemma4ForConditionalGeneration",
            30,
            16,
            8,
            256,
            2816,
            2112,
            262144,
            262144,
            self._make_model_config(),
            None,
        )
        bad_cfg = common.Gemma4MixConfig(
            layer_types=("sliding_attention",) * 5,  # only 5, but num_layers=30
            swa_num_kv_heads=8,
            swa_head_dim=256,
            global_num_kv_heads=2,
            global_head_dim=512,
            sliding_window_size=1024,
        )
        with pytest.raises(ValueError, match="layer_types length"):
            model.set_gemma4_config(bad_cfg)

    def test_dense_variant_builds_without_moe_ops(self):
        """Dense Gemma 4 variants (e.g. gemma-4-31B-it: topk=0, num_experts=None) have
        no routed-MoE block: every layer is just shared dense MLP + attention.

        Regression test for the assertion crash when num_experts is None.
        """
        model = self._make_dense_model()

        op_names = {op._name for op in model.context_ops}
        # Shared dense MLP ops MUST be present.
        assert "context_swa_shared_mlp_gate_up_gemm" in op_names
        assert "context_global_shared_mlp_gate_up_gemm" in op_names
        # Routed-MoE ops MUST NOT be present.
        assert not any("moe" in n.lower() or "router" in n.lower() for n in op_names), (
            f"dense variant should not emit MoE/router ops, found: "
            f"{[n for n in op_names if 'moe' in n.lower() or 'router' in n.lower()]}"
        )
        gen_names = {op._name for op in model.generation_ops}
        assert not any("moe" in n.lower() or "router" in n.lower() for n in gen_names)

    def test_dense_variant_memory_usage_no_crash(self):
        """Dense Gemma 4 (num_experts=None) must flow through _get_memory_usage without
        crashing on the missing MoE workspace; activations fall back to MIN_ACTIVATION_BYTES."""
        model = self._make_dense_model()

        database = SimpleNamespace(
            system_spec={
                "misc": {
                    "nccl_mem": {1: 0},
                    "other_mem": 0,
                }
            }
        )
        memory = BaseBackend()._get_memory_usage(
            model,
            database,
            batch_size=1,
            beam_width=1,
            isl=1,
            osl=1,
        )

        assert memory["activations"] == pytest.approx(BaseBackend.MIN_ACTIVATION_BYTES / (1 << 30))
        assert memory["total"] > 0

    def test_dense_variant_rejects_moe_ep_gt_1(self):
        """Dense Gemma 4 has no experts, so any moe_ep_size > 1 must be rejected
        (otherwise pareto search would enumerate equivalent dense configurations)."""
        bad_config = config.ModelConfig(
            tp_size=2,
            pp_size=1,
            moe_tp_size=1,
            moe_ep_size=2,
            attention_dp_size=1,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            moe_quant_mode=common.MoEQuantMode.bfloat16,
        )
        with pytest.raises(AssertionError, match="moe_ep_size=1"):
            Gemma4MixModel(
                0,
                None,
                21504,
                "google/gemma-4-31B-it",
                "GEMMA4MIX",
                "Gemma4ForConditionalGeneration",
                2,
                32,
                16,
                256,
                5376,
                21504,
                262144,
                262144,
                bad_config,
                None,
            )


class TestHybridMoEModelBuilder:
    """Builder-level tests that verify HybridMoEModel wiring through set_hybrid_config."""

    @staticmethod
    def _make_model_config():
        return config.ModelConfig(
            tp_size=1,
            pp_size=1,
            moe_tp_size=1,
            moe_ep_size=1,
            attention_dp_size=1,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            moe_quant_mode=common.MoEQuantMode.bfloat16,
        )

    def test_mimov2flash_model_builds_all_three_layer_types(self):
        """MiMo-V2-Flash config produces context/generation ops for global_moe, swa_moe, swa_dense."""
        hybrid_cfg = common.HybridMoEConfig(
            attn_layer_pattern=(0, 1, 1, 1, 1, 0, 1, 1, 1, 1),
            moe_layer_freq=(0, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            swa_num_kv_heads=8,
            swa_head_dim=192,
            swa_v_head_dim=128,
            global_v_head_dim=128,
            sliding_window_size=128,
        )
        model = HybridMoEModel(
            8,
            64,
            2048,  # topk, num_experts, moe_inter_size
            "test-model",
            "HYBRIDMOE",
            "MiMoV2FlashForCausalLM",
            10,
            64,
            4,
            192,
            4096,
            16384,
            152576,
            262144,
            self._make_model_config(),
        )
        model.set_hybrid_config(hybrid_cfg)

        assert len(model.context_ops) > 0
        assert len(model.generation_ops) > 0
        op_names = [op._name for op in model.context_ops]
        assert any("global" in n for n in op_names), "Missing global attention ops"
        assert any("swa" in n and "moe" in n for n in op_names), "Missing swa_moe ops"
        assert any("swa" in n and "dense" in n for n in op_names), "Missing swa_dense ops"

    def test_llama4_scout_model_builds_global_and_swa_moe(self):
        """Llama 4 Scout (step=1, all MoE) produces global_moe + swa_moe ops."""
        layers = 8
        hybrid_cfg = common.HybridMoEConfig(
            attn_layer_pattern=tuple(i % 2 for i in range(layers)),
            moe_layer_freq=tuple(1 for _ in range(layers)),
            sliding_window_size=8192,
            dense_inter_size=16384,
        )
        model = HybridMoEModel(
            1,
            16,
            8192,  # topk, num_experts, moe_inter_size
            "test-model",
            "HYBRIDMOE",
            "Llama4ForConditionalGeneration",
            layers,
            40,
            8,
            128,
            5120,
            8192,
            202048,
            10485760,
            self._make_model_config(),
        )
        model.set_hybrid_config(hybrid_cfg)

        assert len(model.context_ops) > 0
        assert len(model.generation_ops) > 0
        op_names = [op._name for op in model.context_ops]
        assert any("global" in n for n in op_names)
        assert any("swa" in n for n in op_names)
        assert not any("dense" in n for n in op_names), "Scout has no dense layers"

    def test_llama4_maverick_model_builds_global_moe_and_swa_dense(self):
        """Llama 4 Maverick (step=2) produces global_moe + swa_dense ops."""
        layers = 8
        step = 2
        hybrid_cfg = common.HybridMoEConfig(
            attn_layer_pattern=tuple(i % 2 for i in range(layers)),
            moe_layer_freq=tuple(1 if (i + 1) % step == 0 else 0 for i in range(layers)),
            sliding_window_size=8192,
            dense_inter_size=16384,
        )
        model = HybridMoEModel(
            1,
            128,
            8192,  # topk, num_experts, moe_inter_size
            "test-model",
            "HYBRIDMOE",
            "Llama4ForConditionalGeneration",
            layers,
            40,
            8,
            128,
            5120,
            8192,
            202048,
            1048576,
            self._make_model_config(),
        )
        model.set_hybrid_config(hybrid_cfg)

        assert len(model.context_ops) > 0
        op_names = [op._name for op in model.context_ops]
        assert any("global" in n and "moe" in n for n in op_names)
        assert any("dense" in n for n in op_names), "Maverick needs dense FFN ops"


class TestGetModelConfigFromHFID:
    """Test getting model config from HuggingFace ID."""

    @patch("aiconfigurator.sdk.utils._download_hf_json")
    @patch("aiconfigurator.sdk.utils._download_hf_config")
    def test_successful_download(self, mock_download, mock_download_quant):
        """Test successful download from HuggingFace."""
        mock_config = {
            "architectures": ["LlamaForCausalLM"],
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "intermediate_size": 14336,
            "vocab_size": 128256,
            "max_position_embeddings": 131072,
            "num_experts_per_tok": 0,
        }
        mock_download.return_value = mock_config

        mock_download_quant.return_value = None

        model_id = "acme/Fake-Model-32B"
        result = get_model_config_from_model_path(model_id)

        assert result["architecture"] == "LlamaForCausalLM"  # architecture
        mock_download.assert_called_once_with(model_id)


class TestSafeMkdir:
    """Test safe_mkdir utility (existing tests can be expanded if needed)."""

    def test_safe_mkdir_exists(self):
        """Test that safe_mkdir function exists and is importable."""
        from aiconfigurator.sdk.utils import safe_mkdir

        assert callable(safe_mkdir)


class TestEnumerateTTFTTPOTConstraints:
    """Tests for request-latency driven TTFT/TPOT enumeration."""

    def test_constraints_respect_request_latency_and_include_explicit_ttft(self):
        """Passing request_latency + ttft yields tuples below the latency budget."""
        constraints = enumerate_ttft_tpot_constraints(osl=500, request_latency=12000, ttft=4000)

        expected_tpot = (12000 - 4000) / (500 - 1)
        assert any(ttft == 4000 and tpot == pytest.approx(expected_tpot) for ttft, tpot in constraints)
        assert all(ttft < 12000 for ttft, _ in constraints)
        assert all(tpot > 0 for _, tpot in constraints)

    def test_constraints_default_to_95_percent_ttft_when_not_provided(self):
        """When ttft is omitted, we fall back to 95% of request latency."""
        constraints = enumerate_ttft_tpot_constraints(osl=50, request_latency=1000)

        expected_ttft = 0.95 * 1000
        derived_pair = next((pair for pair in constraints if pair[0] == pytest.approx(expected_ttft)), None)
        assert derived_pair is not None
        assert derived_pair[1] == pytest.approx((1000 - 950) / (50 - 1))


class TestParseCompressedTensorsQuant:
    """Tests for parse_compressed_tensors_quant and its _categorize_ignore_pattern helper."""

    @staticmethod
    def _make_quant_config(num_bits: int, w_type: str, ignore: list | None = None) -> dict:
        return {
            "config_groups": {
                "group_0": {
                    "weights": {"num_bits": num_bits, "type": w_type},
                    "input_activations": None,
                    "output_activations": None,
                    "targets": ["Linear"],
                }
            },
            "ignore": ignore or [],
            "quant_method": "compressed-tensors",
        }

    # --- input guards ---

    def test_none_input(self):
        from aiconfigurator.sdk.utils import parse_compressed_tensors_quant

        assert parse_compressed_tensors_quant(None) == (None, frozenset())

    def test_non_dict_input(self):
        from aiconfigurator.sdk.utils import parse_compressed_tensors_quant

        assert parse_compressed_tensors_quant("not-a-dict") == (None, frozenset())

    def test_empty_config_groups(self):
        from aiconfigurator.sdk.utils import parse_compressed_tensors_quant

        assert parse_compressed_tensors_quant({"config_groups": {}}) == (None, frozenset())

    # --- base_algo detection ---

    def test_int4_base_algo(self):
        from aiconfigurator.sdk.utils import parse_compressed_tensors_quant

        algo, _ = parse_compressed_tensors_quant(self._make_quant_config(4, "int"))
        assert algo == "int4_wo"

    def test_int8_base_algo(self):
        from aiconfigurator.sdk.utils import parse_compressed_tensors_quant

        algo, _ = parse_compressed_tensors_quant(self._make_quant_config(8, "int"))
        assert algo == "int8_wo"

    def test_fp8_base_algo(self):
        from aiconfigurator.sdk.utils import parse_compressed_tensors_quant

        algo, _ = parse_compressed_tensors_quant(self._make_quant_config(8, "float"))
        assert algo == "fp8"

    # --- ignored_categories detection ---

    def test_no_ignore_empty_set(self):
        from aiconfigurator.sdk.utils import parse_compressed_tensors_quant

        _, ignored = parse_compressed_tensors_quant(self._make_quant_config(4, "int", ignore=[]))
        assert ignored == frozenset()

    def test_attention_pattern_detected(self):
        from aiconfigurator.sdk.utils import parse_compressed_tensors_quant

        _, ignored = parse_compressed_tensors_quant(self._make_quant_config(4, "int", ignore=["re:.*self_attn.*"]))
        assert "attention" in ignored

    def test_routing_experts_pattern_detected(self):
        from aiconfigurator.sdk.utils import parse_compressed_tensors_quant

        _, ignored = parse_compressed_tensors_quant(
            self._make_quant_config(4, "int", ignore=["re:.*mlp\\.experts\\..*"])
        )
        assert "routing_experts" in ignored

    def test_shared_experts_pattern_detected_not_routing(self):
        """re:.*shared_experts.* must categorize as shared_experts, NOT routing_experts."""
        from aiconfigurator.sdk.utils import parse_compressed_tensors_quant

        _, ignored = parse_compressed_tensors_quant(self._make_quant_config(4, "int", ignore=["re:.*shared_experts.*"]))
        assert "shared_experts" in ignored
        assert "routing_experts" not in ignored

    def test_dense_mlp_pattern_detected(self):
        from aiconfigurator.sdk.utils import parse_compressed_tensors_quant

        _, ignored = parse_compressed_tensors_quant(
            self._make_quant_config(4, "int", ignore=["re:.*mlp\\.(gate|up|gate_up|down)_proj.*"])
        )
        assert "dense_mlp" in ignored

    def test_lm_head_pattern_detected(self):
        from aiconfigurator.sdk.utils import parse_compressed_tensors_quant

        _, ignored = parse_compressed_tensors_quant(self._make_quant_config(4, "int", ignore=["lm_head"]))
        assert "lm_head" in ignored

    # --- Kimi K2.5 end-to-end ---

    def test_kimi_k25_actual_config(self):
        """Kimi K2.5: attention + shared_experts + dense_mlp + lm_head all ignored.
        Only routing experts are quantized → those four categories in ignored set."""
        from aiconfigurator.sdk.utils import parse_compressed_tensors_quant

        cfg = self._make_quant_config(
            4,
            "int",
            ignore=[
                "lm_head",
                "re:.*self_attn.*",
                "re:.*shared_experts.*",
                "re:.*mlp\\.(gate|up|gate_up|down)_proj.*",
            ],
        )
        base_algo, ignored = parse_compressed_tensors_quant(cfg)
        assert base_algo == "int4_wo"
        assert ignored == frozenset({"attention", "shared_experts", "dense_mlp", "lm_head"})
        assert "routing_experts" not in ignored

    # --- models.py mapping (end-to-end via _infer_quant_modes_from_raw_config) ---

    def test_kimi_k25_maps_to_correct_sdk_modes(self):
        """Kimi K2.5 compressed-tensors config → gemm=bfloat16, moe=int4_wo."""
        from aiconfigurator.sdk import common
        from aiconfigurator.sdk.models import _infer_quant_modes_from_raw_config

        raw_config = {
            "quant_algo": "compressed-tensors",
            "quantization_config": self._make_quant_config(
                4,
                "int",
                ignore=[
                    "lm_head",
                    "re:.*self_attn.*",
                    "re:.*shared_experts.*",
                    "re:.*mlp\\.(gate|up|gate_up|down)_proj.*",
                ],
            ),
        }
        overrides = _infer_quant_modes_from_raw_config(raw_config)
        assert overrides.get("gemm_quant_mode") is None  # not set → falls back to bfloat16
        assert overrides.get("moe_quant_mode") == common.MoEQuantMode.int4_wo

    def test_all_layers_quantized_maps_both_modes(self):
        """No ignore list → both gemm and moe overrides are set."""
        from aiconfigurator.sdk import common
        from aiconfigurator.sdk.models import _infer_quant_modes_from_raw_config

        raw_config = {
            "quant_algo": "compressed-tensors",
            "quantization_config": self._make_quant_config(4, "int", ignore=[]),
        }
        overrides = _infer_quant_modes_from_raw_config(raw_config)
        assert overrides.get("gemm_quant_mode") == common.GEMMQuantMode.int4_wo
        assert overrides.get("moe_quant_mode") == common.MoEQuantMode.int4_wo

    def test_modelopt_mixed_precision_config_groups_map_sdk_modes(self):
        """ModelOpt MIXED_PRECISION config groups map FP8 dense layers and NVFP4 routed experts."""
        from aiconfigurator.sdk import common
        from aiconfigurator.sdk.models import _infer_quant_modes_from_raw_config
        from aiconfigurator.sdk.utils import _attach_inferred_quant_fields

        raw_config = _attach_inferred_quant_fields(
            {
                "quantization_config": {
                    "quant_algo": "MIXED_PRECISION",
                    "kv_cache_scheme": {"type": "float", "num_bits": 8},
                    "config_groups": {
                        "group_0": {
                            "weights": {"dynamic": False, "num_bits": 8, "type": "float"},
                            "input_activations": {"dynamic": False, "num_bits": 8, "type": "float"},
                            "targets": [
                                "backbone.layers.0.mixer.in_proj",
                                "backbone.layers.1.mixer.shared_experts.up_proj",
                            ],
                        },
                        "group_1": {
                            "weights": {
                                "dynamic": False,
                                "group_size": 16,
                                "num_bits": 4,
                                "type": "float",
                            },
                            "input_activations": {
                                "dynamic": False,
                                "group_size": 16,
                                "num_bits": 4,
                                "type": "float",
                            },
                            "targets": ["backbone.layers.1.mixer.experts.0.up_proj"],
                        },
                    },
                },
            }
        )

        overrides = _infer_quant_modes_from_raw_config(raw_config)

        assert raw_config["quant_algo"] == "mixed_precision"
        assert overrides["gemm_quant_mode"] == common.GEMMQuantMode.fp8_static
        assert overrides["moe_quant_mode"] == common.MoEQuantMode.nvfp4
        assert overrides["kvcache_quant_mode"] == common.KVCacheQuantMode.fp8
        assert overrides["fmha_quant_mode"] == common.FMHAQuantMode.fp8

    def test_modelopt_mixed_precision_hf_quant_layers_map_sdk_modes(self):
        """Standalone hf_quant_config.json MIXED_PRECISION metadata no longer raises unsupported quant_algo."""
        from aiconfigurator.sdk import common
        from aiconfigurator.sdk.models import _infer_quant_modes_from_raw_config
        from aiconfigurator.sdk.utils import _attach_inferred_quant_fields

        raw_config = _attach_inferred_quant_fields(
            {
                "hf_quant_config": {
                    "quantization": {
                        "quant_algo": "MIXED_PRECISION",
                        "kv_cache_quant_algo": "FP8",
                        "quantized_layers": {
                            "backbone.layers.0.mixer.in_proj": {"quant_algo": "FP8"},
                            "backbone.layers.1.mixer.shared_experts.up_proj": {"quant_algo": "FP8"},
                            "backbone.layers.1.mixer.experts.0.up_proj": {"quant_algo": "NVFP4"},
                        },
                    }
                },
            }
        )

        overrides = _infer_quant_modes_from_raw_config(raw_config)

        assert raw_config["quant_algo"] == "mixed_precision"
        assert overrides["gemm_quant_mode"] == common.GEMMQuantMode.fp8_static
        assert overrides["moe_quant_mode"] == common.MoEQuantMode.nvfp4
        assert overrides["kvcache_quant_mode"] == common.KVCacheQuantMode.fp8
        assert overrides["fmha_quant_mode"] == common.FMHAQuantMode.fp8

    def test_expert_dtype_fp4_is_deepseek_v4_specific(self):
        from aiconfigurator.sdk import common
        from aiconfigurator.sdk.models import _infer_quant_modes_from_raw_config

        raw_config = {"expert_dtype": "fp4"}

        llama_overrides = _infer_quant_modes_from_raw_config(raw_config, "LlamaForCausalLM")
        deepseek_v4_overrides = _infer_quant_modes_from_raw_config(raw_config, "DeepseekV4ForCausalLM")

        assert "moe_quant_mode" not in llama_overrides
        assert deepseek_v4_overrides["moe_quant_mode"] == common.MoEQuantMode.w4a8_mxfp4_mxfp8


class TestEnumerateParallelConfigSGLangMoE:
    """Test enumerate_parallel_config for SGLang MoE scenarios."""

    def test_sglang_non_wideep_moe_includes_moe_ep_gt_1(self):
        """Test that SGLang + enable_wideep=False includes configs with moe_ep > 1."""
        configs = enumerate_parallel_config(
            num_gpu_list=[1, 2, 4, 8],
            tp_list=[1, 2, 4, 8],
            pp_list=[1],
            dp_list=[1, 2, 4, 8],
            moe_tp_list=[1, 2, 4, 8],
            moe_ep_list=[1, 2, 4, 8],
            is_moe=True,
            backend=common.BackendName.sglang,
            enable_wideep=False,
        )
        assert len(configs) > 0, "Should generate at least one config"
        moe_ep_values = [c[4] for c in configs]
        assert any(ep > 1 for ep in moe_ep_values), (
            f"Should include at least one config with moe_ep > 1, got moe_ep values: {set(moe_ep_values)}"
        )

    def test_sglang_wideep_moe_excludes_moe_tp_gt_1(self):
        """Test that SGLang + enable_wideep=True excludes configs with moe_tp > 1."""
        configs = enumerate_parallel_config(
            num_gpu_list=[8, 16, 32],
            tp_list=[1, 2, 4, 8],
            pp_list=[1],
            dp_list=[1, 2, 4, 8, 16, 32],
            moe_tp_list=[1, 2, 4, 8],
            moe_ep_list=[8, 16, 32],
            is_moe=True,
            backend=common.BackendName.sglang,
            enable_wideep=True,
        )
        assert len(configs) > 0, "Should generate at least one config"
        # All configs should have moe_tp == 1 (EP-only for wideep)
        for c in configs:
            assert c[3] == 1, f"WideEP config should have moe_tp=1, got {c}"

    def test_sglang_non_wideep_moe_allows_mixed_tp_ep(self):
        """Test that SGLang + enable_wideep=False allows configs with both moe_tp > 1 and moe_ep > 1."""
        configs = enumerate_parallel_config(
            num_gpu_list=[1, 2, 4, 8],
            tp_list=[1, 2, 4, 8],
            pp_list=[1],
            dp_list=[1, 2, 4, 8],
            moe_tp_list=[1, 2, 4, 8],
            moe_ep_list=[1, 2, 4, 8],
            is_moe=True,
            backend=common.BackendName.sglang,
            enable_wideep=False,
        )
        # Should include configs with moe_ep == 1 (pure TP)
        has_pure_tp = any(c[4] == 1 and c[3] > 1 for c in configs)
        # Should include configs with moe_ep > 1
        has_ep_gt_1 = any(c[4] > 1 for c in configs)
        # Should include truly mixed configs (both moe_tp > 1 and moe_ep > 1)
        has_mixed = any(c[3] > 1 and c[4] > 1 for c in configs)
        assert has_pure_tp, "Should include pure TP configs (moe_ep=1, moe_tp>1)"
        assert has_ep_gt_1, "Should include configs with moe_ep > 1"
        assert has_mixed, "Should include mixed configs with both moe_tp > 1 and moe_ep > 1"

    def test_sglang_deepep_intranode_excludes_moe_tp_gt_1(self):
        """SGLang + moe_backend=deepep_moe + enable_wideep=False excludes moe_tp > 1."""
        configs = enumerate_parallel_config(
            num_gpu_list=[1, 2, 4, 8],
            tp_list=[1, 2, 4, 8],
            pp_list=[1],
            dp_list=[1, 2, 4, 8],
            moe_tp_list=[1, 2, 4, 8],
            moe_ep_list=[1, 2, 4, 8],
            is_moe=True,
            backend=common.BackendName.sglang,
            enable_wideep=False,
            moe_backend="deepep_moe",
        )
        assert len(configs) > 0, "Should generate at least one config"
        for c in configs:
            assert c[3] == 1, f"DeepEP config should have moe_tp=1, got {c}"
        # Should still include ep > 1 configs
        moe_ep_values = [c[4] for c in configs]
        assert any(ep > 1 for ep in moe_ep_values), "Should include configs with moe_ep > 1"


class TestEnumerateParallelConfigVLLMMoE:
    """Test enumerate_parallel_config for vLLM MoE scenarios."""

    def test_vllm_excludes_simultaneous_moe_tp_and_ep(self):
        """vLLM should never have both moe_tp > 1 and moe_ep > 1."""
        configs = enumerate_parallel_config(
            num_gpu_list=[1, 2, 4, 8],
            tp_list=[1, 2, 4, 8],
            pp_list=[1],
            dp_list=[1, 2, 4, 8],
            moe_tp_list=[1, 2, 4, 8],
            moe_ep_list=[1, 2, 4, 8],
            is_moe=True,
            backend=common.BackendName.vllm,
        )
        assert len(configs) > 0, "Should generate at least one config"
        for c in configs:
            tp, pp, dp, moe_tp, moe_ep, cp = c
            assert not (moe_tp > 1 and moe_ep > 1), f"vLLM should not have both moe_tp > 1 and moe_ep > 1, got {c}"

    def test_vllm_allows_pure_moe_tp(self):
        """vLLM should allow configs with moe_tp > 1 and moe_ep == 1."""
        configs = enumerate_parallel_config(
            num_gpu_list=[1, 2, 4, 8],
            tp_list=[1, 2, 4, 8],
            pp_list=[1],
            dp_list=[1, 2, 4, 8],
            moe_tp_list=[1, 2, 4, 8],
            moe_ep_list=[1, 2, 4, 8],
            is_moe=True,
            backend=common.BackendName.vllm,
        )
        has_pure_tp = any(c[3] > 1 and c[4] == 1 for c in configs)
        assert has_pure_tp, "Should include pure MoE TP configs (moe_tp>1, moe_ep=1)"

    def test_vllm_allows_pure_moe_ep(self):
        """vLLM should allow configs with moe_tp == 1 and moe_ep > 1."""
        configs = enumerate_parallel_config(
            num_gpu_list=[1, 2, 4, 8],
            tp_list=[1, 2, 4, 8],
            pp_list=[1],
            dp_list=[1, 2, 4, 8],
            moe_tp_list=[1, 2, 4, 8],
            moe_ep_list=[1, 2, 4, 8],
            is_moe=True,
            backend=common.BackendName.vllm,
        )
        has_pure_ep = any(c[3] == 1 and c[4] > 1 for c in configs)
        assert has_pure_ep, "Should include pure MoE EP configs (moe_tp=1, moe_ep>1)"


# ── Qwen3VL config parsing constants ──────────────────────────────────────────

_QWEN3VL_ARCH = "Qwen3VLForConditionalGeneration"

# Minimal Qwen3VL config matching the actual downloaded structure
_QWEN3VL_HF_CONFIG = {
    "architectures": [_QWEN3VL_ARCH],
    "image_token_id": 151655,
    "model_type": "qwen3_vl",
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "dtype": "bfloat16",
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 5120,
        "initializer_range": 0.02,
        "intermediate_size": 25600,
        "max_position_embeddings": 262144,
        "model_type": "qwen3_vl_text",
        "num_attention_heads": 64,
        "num_hidden_layers": 64,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_theta": 5000000,
        "use_cache": True,
        "vocab_size": 151936,
    },
    "tie_word_embeddings": False,
    "vision_config": {
        "depth": 27,
        "hidden_size": 1152,
        "num_heads": 16,
        "intermediate_size": 4304,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "out_hidden_size": 5120,
    },
}


class TestQwen3VLConfigParsing:
    """Test that _parse_hf_config_json correctly unwraps text_config for Qwen3VL."""

    def test_architecture_preserved(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["architecture"] == _QWEN3VL_ARCH

    def test_llm_layers_from_text_config(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["layers"] == 64

    def test_llm_hidden_size_from_text_config(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["hidden_size"] == 5120

    def test_llm_attention_heads_from_text_config(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["n"] == 64

    def test_llm_kv_heads_from_text_config(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["n_kv"] == 8

    def test_llm_head_dim_from_text_config(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["d"] == 128

    def test_llm_inter_size_from_text_config(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["inter_size"] == 25600

    def test_llm_vocab_from_text_config(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["vocab"] == 151936

    def test_not_moe(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["topk"] == 0
        assert result["num_experts"] == 0


class TestQwen3VLVisionEncoderParsing:
    """Test that vision_config is captured before text_config unwrap and parsed correctly."""

    def test_extra_params_is_vision_encoder_config(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert isinstance(result["extra_params"], common.VisionEncoderConfig)

    def test_vision_encoder_depth(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["extra_params"].depth == 27

    def test_vision_encoder_hidden_size(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["extra_params"].hidden_size == 1152

    def test_vision_encoder_num_heads(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["extra_params"].num_heads == 16

    def test_vision_encoder_intermediate_size(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["extra_params"].intermediate_size == 4304

    def test_vision_encoder_patch_size(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["extra_params"].patch_size == 16

    def test_vision_encoder_temporal_patch_size(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["extra_params"].temporal_patch_size == 2

    def test_vision_encoder_spatial_merge_size(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["extra_params"].spatial_merge_size == 2

    def test_vision_encoder_out_hidden_size_matches_llm(self):
        """out_hidden_size must equal LLM hidden_size for the projection to be valid."""
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["extra_params"].out_hidden_size == result["hidden_size"]

    def test_vision_config_not_lost_after_text_config_unwrap(self):
        """Regression: vision_config must be captured before the text_config overwrite."""
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["extra_params"] is not None

    def test_missing_vision_config_does_not_raise(self):
        """If vision_config is absent, extra_params should be None, not an error."""
        cfg = {**_QWEN3VL_HF_CONFIG}
        cfg.pop("vision_config")
        result = _parse_hf_config_json(cfg)
        assert result["extra_params"] is None

    def test_vision_encoder_deepstack_visual_indexes_default(self):
        """deepstack_visual_indexes defaults to empty tuple when absent from vision_config."""
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["extra_params"].deepstack_visual_indexes == ()

    def test_vision_encoder_deepstack_visual_indexes_populated(self):
        """deepstack_visual_indexes is parsed as a tuple when present in vision_config."""
        vision_cfg = {**_QWEN3VL_HF_CONFIG["vision_config"], "deepstack_visual_indexes": [8, 17, 26]}
        cfg = {**_QWEN3VL_HF_CONFIG, "vision_config": vision_cfg}
        result = _parse_hf_config_json(cfg)
        assert result["extra_params"].deepstack_visual_indexes == (8, 17, 26)

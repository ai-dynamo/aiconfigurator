# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for SDK utility functions.

Tests HuggingFace config parsing and model config retrieval.
"""

from unittest.mock import patch

import pytest

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.models import HybridMoEModel
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
            gemm_quant_mode=common.GEMMQuantMode.float16,
            kvcache_quant_mode=common.KVCacheQuantMode.float16,
            fmha_quant_mode=common.FMHAQuantMode.float16,
            moe_quant_mode=common.MoEQuantMode.float16,
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

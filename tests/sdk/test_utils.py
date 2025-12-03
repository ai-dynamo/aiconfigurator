# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for SDK utility functions.

Tests HuggingFace config parsing and model config retrieval.
"""

from unittest.mock import patch

import pytest

from aiconfigurator.sdk.utils import (
    _parse_hf_config_json,
    enumerate_ttft_tpot_constraints,
    get_model_config_from_hf_id,
)


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

        assert result[0] == "LLAMA"  # model_family
        assert result[1] == 32  # num_layers
        assert result[2] == 32  # num_heads
        assert result[3] == 8  # num_kv_heads
        assert result[5] == 4096  # hidden_size
        assert result[6] == 14336  # inter_size
        assert result[7] == 128256  # vocab_size

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

        assert result[0] == "MOE"  # model_family
        assert result[9] == 2  # topk
        assert result[10] == 8  # num_experts
        assert result[11] == 14336  # moe_inter_size

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

        assert result[0] == "DEEPSEEK"  # model_family
        assert result[10] == 256  # num_experts from n_routed_experts

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

        assert result[4] == 80  # head_dim


class TestGetModelConfigFromHFID:
    """Test getting model config from HuggingFace ID."""

    @patch("aiconfigurator.sdk.utils._download_hf_config")
    def test_successful_download(self, mock_download):
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

        result = get_model_config_from_hf_id("Qwen/Qwen3-32B-FP8")

        assert result[0] == "LLAMA"
        mock_download.assert_called_once_with("Qwen/Qwen3-32B-FP8")


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

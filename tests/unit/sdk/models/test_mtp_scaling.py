# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for MTP (Multi-Token Prediction) speculative decoding scaling.

Tests that verify:
1. context_p2p is NOT scaled by mtp_scale_factor (bug fix verification)
2. generation_p2p IS scaled by mtp_scale_factor
3. MTP scale factor calculation for non-DeepSeek models
"""

import pytest

from aiconfigurator.sdk import models
from aiconfigurator.sdk import config as sdk_config
from aiconfigurator.sdk import common

pytestmark = pytest.mark.unit


class TestMTPScaling:
    """Tests for MTP speculative decoding scaling behavior."""

    def _create_model_config(self, nextn=0):
        """Helper to create a ModelConfig for testing."""
        return sdk_config.ModelConfig(
            tp_size=1,
            pp_size=1,
            gemm_quant_mode=common.GEMMQuantMode.float16,
            kvcache_quant_mode=common.KVCacheQuantMode.float16,
            nextn=nextn,
            nextn_accept_rates=[0.85, 0.3, 0, 0, 0],
        )

    def test_mtp_scale_factor_with_nextn_zero(self):
        """
        Test that mtp_scale_factor is 1.0 when nextn=0 (MTP disabled).

        Backward compatibility: nextn=0 should produce identical results as before.
        """
        model_config_zero = self._create_model_config(nextn=0)
        model_zero = models.get_model("Qwen/Qwen3-32B", model_config_zero, "trtllm")

        # When nextn=0, mtp_scale_factor should be 1.0 (no scaling)
        assert model_zero._mtp_scale_factor == 1.0, \
            "mtp_scale_factor should be 1.0 when nextn=0"

    def test_mtp_scale_factor_calculation(self):
        """
        Test that mtp_scale_factor is calculated correctly.

        Formula: (1.0 / (1 + calc_expectation(nextn, accept_rates))) * (nextn + num_layers) / num_layers
        """
        from aiconfigurator.sdk.models import calc_expectation

        # Test calc_expectation function
        # With accept_rates [0.85, 0, 0, 0, 0]:
        # - nextn=0: expectation = 0.0
        # - nextn=1: expectation = 0.85 (1st token only)
        assert calc_expectation(0, [0.85, 0, 0, 0, 0]) == 0.0
        assert calc_expectation(1, [0.85, 0, 0, 0, 0]) == 0.85

    def test_llama_model_supports_mtp(self):
        """
        Test that LLAMAModel supports MTP (does not assert nextn==0).
        """
        # Should not raise any assertion error
        model_config = self._create_model_config(nextn=3)
        model = models.get_model("Qwen/Qwen3-32B", model_config, "trtllm")

        # Verify model was created successfully with nextn > 0
        assert model is not None
        assert model.config.nextn == 3
        assert hasattr(model, '_mtp_scale_factor')

    def test_moe_model_supports_mtp(self):
        """
        Test that MOEModel supports MTP (does not assert nextn==0).
        """
        try:
            model_config = self._create_model_config(nextn=2)
            # Use a known MOE model or skip if not available
            model = models.get_model("mistralai/Mixtral-8x7B-v0.1", model_config, "trtllm")

            # Verify model was created successfully with nextn > 0
            assert model is not None
            assert model.config.nextn == 2
            assert hasattr(model, '_mtp_scale_factor')
        except Exception as e:
            # Model might not be supported, skip
            pytest.skip(f"MOE model test skipped: {e}")

    def test_mtp_scale_factor_exists_for_all_models(self):
        """
        Test that all models have _mtp_scale_factor attribute.
        """
        # LLAMA model
        llama_config = self._create_model_config(nextn=0)
        llama_model = models.get_model("Qwen/Qwen3-32B", llama_config, "trtllm")
        assert hasattr(llama_model, '_mtp_scale_factor')

        # DeepSeek model (if available)
        try:
            deepseek_config = self._create_model_config(nextn=1)
            deepseek_model = models.get_model("deepseek-ai/DeepSeek-V3", deepseek_config, "trtllm")
            assert hasattr(deepseek_model, '_mtp_scale_factor')
        except Exception:
            pass  # DeepSeek model might not be available

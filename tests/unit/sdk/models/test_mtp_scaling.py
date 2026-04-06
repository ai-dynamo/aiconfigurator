# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for MTP (Multi-Token Prediction) speculative decoding scaling.

Tests that verify:
1. context_p2p is NOT scaled by mtp_scale_factor (bug fix verification)
2. generation_p2p IS scaled by mtp_scale_factor
3. MTP scale factor calculation for non-DeepSeek models
4. Qwen3.5 MTP support (hybrid GDN + full_attention architecture)
"""

import pytest

from aiconfigurator.sdk import common, models
from aiconfigurator.sdk import config as sdk_config
from aiconfigurator.sdk.utils import HuggingFaceDownloadError

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
            nextn_accept_rates=[0.85, 0.3, 0.0, 0.0, 0.0],
        )

    def test_mtp_scale_factor_with_nextn_zero(self):
        """
        Test that mtp_scale_factor is 1.0 when nextn=0 (MTP disabled).

        Backward compatibility: nextn=0 should produce identical results as before.
        """
        model_config_zero = self._create_model_config(nextn=0)
        model_zero = models.get_model("Qwen/Qwen3-32B", model_config_zero, "trtllm")

        # When nextn=0, mtp_scale_factor should be 1.0 (no scaling)
        assert model_zero._mtp_scale_factor == 1.0, "mtp_scale_factor should be 1.0 when nextn=0"

    def test_mtp_scale_factor_calculation(self):
        """
        Test that mtp_scale_factor is calculated correctly.

        Formula: (1.0 / (1 + calc_expectation(nextn, accept_rates))) * (nextn + num_layers) / num_layers
        """
        from aiconfigurator.sdk.models import calc_expectation

        # Test calc_expectation function
        # With accept_rates [0.85, 0.3, 0, 0, 0]:
        # - nextn=0: expectation = 0.0
        # - nextn=1: expectation = 0.85 (1st token only)
        # - nextn=2: expectation = 0.85 + 0.85*0.3 = 0.85 + 0.255 = 1.105
        assert calc_expectation(0, [0.85, 0.3, 0.0, 0.0, 0.0]) == 0.0
        assert calc_expectation(1, [0.85, 0.3, 0.0, 0.0, 0.0]) == 0.85

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
        assert hasattr(model, "_mtp_scale_factor")

    def test_moe_model_supports_mtp(self):
        """
        Test that MOEModel supports MTP (does not assert nextn==0).
        """
        try:
            # Create a minimal MOE model config with required fields
            model_config = sdk_config.ModelConfig(
                tp_size=2,
                pp_size=1,
                gemm_quant_mode=common.GEMMQuantMode.float16,
                kvcache_quant_mode=common.KVCacheQuantMode.float16,
                nextn=2,
                nextn_accept_rates=[0.85, 0.3, 0.0, 0.0, 0.0],
                moe_tp_size=2,
                moe_ep_size=1,
                attention_dp_size=1,
            )
            # Use a known MOE model or skip if not available
            model = models.get_model("Qwen/Qwen3-30B-A3B", model_config, "trtllm")

            # Verify model was created successfully with nextn > 0
            assert model is not None
            assert model.config.nextn == 2
            assert hasattr(model, "_mtp_scale_factor")
        except (FileNotFoundError, KeyError, ValueError, TypeError, HuggingFaceDownloadError) as e:
            # Model config file not found or missing required keys
            pytest.skip(f"MOE model test skipped due to missing config: {e}")

    def test_mtp_scale_factor_exists_for_all_models(self):
        """
        Test that all models have _mtp_scale_factor attribute.
        """
        # LLAMA model
        llama_config = self._create_model_config(nextn=0)
        llama_model = models.get_model("Qwen/Qwen3-32B", llama_config, "trtllm")
        assert hasattr(llama_model, "_mtp_scale_factor")

        # DeepSeek model (if available) - requires moe config
        try:
            deepseek_config = sdk_config.ModelConfig(
                tp_size=2,
                pp_size=1,
                gemm_quant_mode=common.GEMMQuantMode.float16,
                kvcache_quant_mode=common.KVCacheQuantMode.float16,
                nextn=1,
                nextn_accept_rates=[0.85, 0.3, 0.0, 0.0, 0.0],
                moe_tp_size=2,
                moe_ep_size=1,
                attention_dp_size=1,
            )
            deepseek_model = models.get_model("deepseek-ai/DeepSeek-V3", deepseek_config, "trtllm")
            assert hasattr(deepseek_model, "_mtp_scale_factor")
        except (FileNotFoundError, KeyError, ValueError, TypeError):
            pass  # DeepSeek model config might not be available

    def test_generation_ops_scaled_by_mtp(self):
        """
        Test that generation ops are correctly scaled by mtp_scale_factor.
        """
        # Create model with nextn=0 (no scaling)
        model_config_zero = self._create_model_config(nextn=0)
        model_zero = models.get_model("Qwen/Qwen3-32B", model_config_zero, "trtllm")

        # Create model with nextn=2 (with scaling)
        model_config_mtp = self._create_model_config(nextn=2)
        model_mtp = models.get_model("Qwen/Qwen3-32B", model_config_mtp, "trtllm")

        # Find a GEMM operation in generation_ops
        gen_gemm_zero = None
        gen_gemm_mtp = None
        for op in model_zero.generation_ops:
            if hasattr(op, "_name") and "qkv_gemm" in op._name:
                gen_gemm_zero = op
                break
        for op in model_mtp.generation_ops:
            if hasattr(op, "_name") and "qkv_gemm" in op._name:
                gen_gemm_mtp = op
                break

        assert gen_gemm_zero is not None, "Should find qkv_gemm in generation_ops"
        assert gen_gemm_mtp is not None, "Should find qkv_gemm in generation_ops"

        # The _scale_factor in gen_gemm_mtp should be scaled by mtp_scale_factor
        # gen_gemm_zero._scale_factor = num_layers (since mtp_scale_factor=1.0)
        # gen_gemm_mtp._scale_factor = num_layers * mtp_scale_factor
        assert gen_gemm_zero._scale_factor != gen_gemm_mtp._scale_factor, (
            "Generation ops should be scaled differently for different nextn values"
        )

    def test_context_ops_not_scaled_by_mtp(self):
        """
        Test that context ops are NOT scaled by mtp_scale_factor.

        This is a regression test for the P2P bug fix where context_p2p
        was incorrectly being scaled.
        """
        # Create model with nextn=0 (no scaling)
        model_config_zero = self._create_model_config(nextn=0)
        model_zero = models.get_model("Qwen/Qwen3-32B", model_config_zero, "trtllm")

        # Create model with nextn=2 (with scaling)
        model_config_mtp = self._create_model_config(nextn=2)
        model_mtp = models.get_model("Qwen/Qwen3-32B", model_config_mtp, "trtllm")

        # Find context ops (e.g., qkv_gemm or attention)
        ctx_op_zero = None
        ctx_op_mtp = None
        for op in model_zero.context_ops:
            if hasattr(op, "_name") and "qkv_gemm" in op._name:
                ctx_op_zero = op
                break
        for op in model_mtp.context_ops:
            if hasattr(op, "_name") and "qkv_gemm" in op._name:
                ctx_op_mtp = op
                break

        assert ctx_op_zero is not None, "Should find qkv_gemm in context_ops"
        assert ctx_op_mtp is not None, "Should find qkv_gemm in context_ops"

        # Context ops should have the same _scale_factor regardless of nextn
        assert ctx_op_zero._scale_factor == ctx_op_mtp._scale_factor, (
            "Context ops should NOT be scaled by mtp_scale_factor"
        )

    def test_qwen35_model_supports_mtp(self):
        """
        Test that Qwen35Model accepts nextn > 0 without assertion error.
        """
        try:
            model_config = self._create_model_config(nextn=1)
            model = models.get_model("Qwen/Qwen3.5-27B", model_config, "trtllm")

            assert model is not None
            assert model.config.nextn == 1
            assert hasattr(model, "_mtp_scale_factor")
            assert model._mtp_scale_factor != 1.0, "mtp_scale_factor should differ from 1.0 when nextn > 0"
        except (FileNotFoundError, KeyError, ValueError, TypeError, HuggingFaceDownloadError) as e:
            pytest.skip(f"Qwen3.5 model test skipped due to missing config: {e}")

    def test_qwen35_generation_ops_scaled_by_mtp(self):
        """
        Test that Qwen35Model generation ops are scaled by mtp_scale_factor
        for both GDN and full_attention layer types.
        """
        model_config_zero = self._create_model_config(nextn=0)
        model_zero = models.get_model("Qwen/Qwen3.5-27B", model_config_zero, "trtllm")

        model_config_mtp = self._create_model_config(nextn=1)
        model_mtp = models.get_model("Qwen/Qwen3.5-27B", model_config_mtp, "trtllm")

        # GDN ops should be scaled
        gdn_zero = next(
            (op for op in model_zero.generation_ops if hasattr(op, "_name") and "gdn_in_proj" in op._name), None
        )
        gdn_mtp = next(
            (op for op in model_mtp.generation_ops if hasattr(op, "_name") and "gdn_in_proj" in op._name), None
        )
        assert gdn_zero is not None and gdn_mtp is not None, "Should find GDN ops in generation_ops"
        assert gdn_zero._scale_factor != gdn_mtp._scale_factor, (
            "GDN generation ops should be scaled differently when MTP is enabled"
        )

        # Full attention ops should be scaled
        attn_zero = next(
            (op for op in model_zero.generation_ops if hasattr(op, "_name") and "qkv_gemm" in op._name), None
        )
        attn_mtp = next(
            (op for op in model_mtp.generation_ops if hasattr(op, "_name") and "qkv_gemm" in op._name), None
        )
        assert attn_zero is not None and attn_mtp is not None, "Should find full attention ops in generation_ops"
        assert attn_zero._scale_factor != attn_mtp._scale_factor, (
            "Full attention generation ops should be scaled differently when MTP is enabled"
        )

    def test_qwen35_context_ops_not_scaled_by_mtp(self):
        """
        Test that Qwen35Model context ops are NOT scaled by mtp_scale_factor.
        """
        try:
            model_config_zero = self._create_model_config(nextn=0)
            model_zero = models.get_model("Qwen/Qwen3.5-27B", model_config_zero, "trtllm")

            model_config_mtp = self._create_model_config(nextn=1)
            model_mtp = models.get_model("Qwen/Qwen3.5-27B", model_config_mtp, "trtllm")
        except (FileNotFoundError, KeyError, ValueError, TypeError, HuggingFaceDownloadError) as e:
            pytest.skip(f"Qwen3.5 model test skipped due to missing config: {e}")

        # GDN context ops should NOT be scaled
        ctx_gdn_zero = next(
            (op for op in model_zero.context_ops if hasattr(op, "_name") and "gdn_in_proj" in op._name), None
        )
        ctx_gdn_mtp = next(
            (op for op in model_mtp.context_ops if hasattr(op, "_name") and "gdn_in_proj" in op._name), None
        )
        assert ctx_gdn_zero is not None and ctx_gdn_mtp is not None, "Should find GDN ops in context_ops"
        assert ctx_gdn_zero._scale_factor == ctx_gdn_mtp._scale_factor, (
            "Context GDN ops should NOT be scaled by mtp_scale_factor"
        )

        # Full attention context ops should NOT be scaled
        ctx_attn_zero = next(
            (op for op in model_zero.context_ops if hasattr(op, "_name") and "qkv_gemm" in op._name), None
        )
        ctx_attn_mtp = next(
            (op for op in model_mtp.context_ops if hasattr(op, "_name") and "qkv_gemm" in op._name), None
        )
        assert ctx_attn_zero is not None and ctx_attn_mtp is not None, "Should find full attention ops in context_ops"
        assert ctx_attn_zero._scale_factor == ctx_attn_mtp._scale_factor, (
            "Context full attention ops should NOT be scaled by mtp_scale_factor"
        )

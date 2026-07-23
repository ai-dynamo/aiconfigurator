# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for MTP (Multi-Token Prediction) speculative decoding scaling.

Tests that verify:
1. the mtp_scale_factor helper models compute-side nextn only
2. generation ops ARE scaled by mtp_scale_factor while context ops are NOT
   (context_p2p bug-fix regression), incl. the Qwen3.5 hybrid GDN arch
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
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            nextn=nextn,
        )

    def test_mtp_scale_factor_calculation(self):
        """
        Test the mtp_scale_factor helper.

        Formula: (nextn + num_layers) / num_layers
        """
        from aiconfigurator.sdk.models import mtp_scale_factor

        assert mtp_scale_factor(0, 64) == 1.0
        assert mtp_scale_factor(2, 64) == pytest.approx((2 + 64) / 64)
        assert mtp_scale_factor(1, 64) == pytest.approx((1 + 64) / 64)
        assert mtp_scale_factor(3, 61) == pytest.approx((3 + 61) / 61)

    def test_model_config_contains_compute_side_nextn_only(self):
        """Core model configuration does not carry workload acceptance."""
        model_config = sdk_config.ModelConfig(
            tp_size=1,
            pp_size=1,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            nextn=2,
        )

        model = models.get_model("Qwen/Qwen3-32B", model_config, "trtllm")

        assert not hasattr(model_config, "nextn_accepted")
        assert not hasattr(model, "_nextn_accepted")
        assert model._mtp_scale_factor == pytest.approx((2 + model._num_layers) / model._num_layers)

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

    def test_p2p_scaling_split_between_phases(self):
        """context_p2p must NOT carry the MTP scale while generation_p2p must
        (the original context-P2P regression), asserted on the P2P ops directly."""

        def build(nextn):
            mc = sdk_config.ModelConfig(
                tp_size=1,
                pp_size=2,
                gemm_quant_mode=common.GEMMQuantMode.bfloat16,
                kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
                nextn=nextn,
            )
            return models.get_model("Qwen/Qwen3-32B", mc, "trtllm")

        def p2p_scale(model, name):
            ops_list = model.context_ops if name == "context_p2p" else model.generation_ops
            op = next(op for op in ops_list if getattr(op, "_name", "") == name)
            return op._scale_factor

        baseline, mtp = build(0), build(2)
        assert p2p_scale(mtp, "context_p2p") == p2p_scale(baseline, "context_p2p")
        assert p2p_scale(mtp, "generation_p2p") != p2p_scale(baseline, "generation_p2p")

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

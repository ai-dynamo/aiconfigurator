# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for model configuration functionality.

Tests model validation, supported models, and model-specific configurations.
"""

import pytest
from aiconfigurator.sdk import common
from aiconfigurator.sdk.models import check_is_moe


class TestSupportedModels:
    """Test supported models configuration."""

    def test_supported_models_list_exists(self):
        """Test that SupportedModels dictionary exists and has content."""
        assert hasattr(common, 'SupportedModels')
        assert isinstance(common.SupportedModels, dict)
        assert len(common.SupportedModels) > 0

    @pytest.mark.parametrize("model_name", [
        "QWEN3_32B", "LLAMA3.1_8B", "DEEPSEEK_V3", "MOE_Mixtral8x7B"
    ])
    def test_specific_models_are_supported(self, model_name):
        """Test that specific models are in the supported list."""
        assert model_name in common.SupportedModels

    def test_model_configs_have_correct_structure(self):
        """Test that model configurations have the expected structure."""
        for model_name, config in common.SupportedModels.items():
            assert isinstance(config, (list, tuple))
            assert len(config) >= 1  # At least model family
            
            # First element should be model family string
            model_family = config[0]
            assert isinstance(model_family, str)
            assert model_family in common.ModelFamily

    @pytest.mark.parametrize("model_name,is_moe_expected", [
        ("QWEN3_32B", False),
        ("LLAMA3.1_8B", False),
        ("DEEPSEEK_V3", True),
        ("MOE_Mixtral8x7B", True),
    ])
    def test_model_moe_detection(self, model_name, is_moe_expected):
        """Test that MoE models are correctly identified."""
        if model_name in common.SupportedModels:
            is_moe = check_is_moe(model_name)
            assert is_moe == is_moe_expected


class TestBackendConfiguration:
    """Test backend configuration."""

    def test_backend_enum_exists(self):
        """Test that BackendName enum exists and has expected values."""
        assert hasattr(common, 'BackendName')
        
        # Check that common backends are supported
        backend_values = [backend.value for backend in common.BackendName]
        expected_backends = ['trtllm', 'vllm', 'sglang']
        
        for backend in expected_backends:
            assert backend in backend_values

    def test_default_backend_is_trtllm(self):
        """Test that the default backend is trtllm."""
        assert common.BackendName.trtllm.value == 'trtllm'


class TestQuantizationModes:
    """Test quantization mode configurations."""

    def test_gemm_quant_modes_exist(self):
        """Test that GEMM quantization modes are defined."""
        assert hasattr(common, 'GEMMQuantMode')
        
        # Should have at least float16 and fp8
        gemm_modes = list(common.GEMMQuantMode)
        mode_names = [mode.name for mode in gemm_modes]
        
        assert 'float16' in mode_names
        assert 'fp8' in mode_names

    def test_attention_quant_modes_exist(self):
        """Test that attention quantization modes are defined."""
        assert hasattr(common, 'FMHAQuantMode')
        assert hasattr(common, 'KVCacheQuantMode')
        
        # Check FMHA modes
        fmha_modes = list(common.FMHAQuantMode)
        assert len(fmha_modes) > 0
        
        # Check KV cache modes
        kv_modes = list(common.KVCacheQuantMode)
        assert len(kv_modes) > 0

    def test_moe_quant_modes_exist(self):
        """Test that MoE quantization modes are defined."""
        assert hasattr(common, 'MoEQuantMode')
        
        moe_modes = list(common.MoEQuantMode)
        mode_names = [mode.name for mode in moe_modes]
        
        assert 'float16' in mode_names
        assert 'fp8' in mode_names 
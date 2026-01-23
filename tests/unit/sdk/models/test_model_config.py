# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for model configuration functionality.

Tests model validation, default models, and model-specific configurations.
"""

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.models import check_is_moe, get_model_family
from aiconfigurator.sdk.utils import get_model_config_from_model_path

pytestmark = pytest.mark.unit


class TestSupportedModels:
    """Test default models configuration from support_matrix.csv."""

    def test_get_default_models_function_exists(self):
        """Test that get_default_models function exists and returns content."""
        assert hasattr(common, "get_default_models")
        models = common.get_default_models()
        assert isinstance(models, set)
        assert len(models) > 0

    @pytest.mark.parametrize(
        "hf_id",
        [
            "Qwen/Qwen3-32B",
            "meta-llama/Meta-Llama-3.1-8B",
            "deepseek-ai/DeepSeek-V3",
            "mistralai/Mixtral-8x7B-v0.1",
        ],
    )
    def test_specific_models_are_in_default_list(self, hf_id):
        """Test that specific models are in the default list."""
        models = common.get_default_models()
        assert hf_id in models

    def test_model_configs_have_correct_structure(self):
        """Test that model configurations have the expected structure."""
        for hf_id in common.DefaultHFModels:
            config = get_model_config_from_model_path(hf_id)
            assert isinstance(config, (list, tuple))
            assert len(config) >= 1  # At least architecture

            # First element should be architecture string
            architecture = config[0]
            assert isinstance(architecture, str)
            assert architecture in common.ARCHITECTURE_TO_MODEL_FAMILY

    @pytest.mark.parametrize(
        "hf_id,is_moe_expected",
        [
            ("Qwen/Qwen3-32B", False),
            ("meta-llama/Meta-Llama-3.1-8B", False),
            ("deepseek-ai/DeepSeek-V3", True),
            ("mistralai/Mixtral-8x7B-v0.1", True),
        ],
    )
    def test_model_moe_detection(self, hf_id, is_moe_expected):
        """Test that MoE models are correctly identified."""
        is_moe = check_is_moe(hf_id)
        assert is_moe == is_moe_expected


class TestHFModelSupport:
    """Test HuggingFace model ID support."""

    def test_default_hf_models_exists(self):
        """Test that DefaultHFModels set exists and has content."""
        assert hasattr(common, "DefaultHFModels")
        assert isinstance(common.DefaultHFModels, set)
        assert len(common.DefaultHFModels) > 0

    def test_hf_models_have_valid_architecture(self):
        """Test that all HF model IDs have valid architecture mapping."""
        for hf_id in common.DefaultHFModels:
            config = get_model_config_from_model_path(hf_id)
            architecture = config[0]
            assert architecture in common.ARCHITECTURE_TO_MODEL_FAMILY

    @pytest.mark.parametrize(
        "hf_id,expected_family",
        [
            ("Qwen/Qwen2.5-7B", "LLAMA"),
            ("meta-llama/Meta-Llama-3.1-8B", "LLAMA"),
            ("deepseek-ai/DeepSeek-V3", "DEEPSEEK"),
            ("mistralai/Mixtral-8x7B-v0.1", "MOE"),
        ],
    )
    def test_hf_id_resolves_to_correct_model_family(self, hf_id, expected_family):
        """Test that HF IDs resolve to the correct model family."""
        family = get_model_family(hf_id)
        assert family == expected_family

    @pytest.mark.parametrize(
        "hf_id,is_moe_expected",
        [
            ("Qwen/Qwen2.5-7B", False),
            ("meta-llama/Meta-Llama-3.1-8B", False),
            ("deepseek-ai/DeepSeek-V3", True),
            ("mistralai/Mixtral-8x7B-v0.1", True),
        ],
    )
    def test_hf_id_moe_detection(self, hf_id, is_moe_expected):
        """Test that MoE models are correctly identified via HF ID."""
        is_moe = check_is_moe(hf_id)
        assert is_moe == is_moe_expected


class TestBackendConfiguration:
    """Test backend configuration."""

    def test_backend_enum_exists(self):
        """Test that BackendName enum exists and has expected values."""
        assert hasattr(common, "BackendName")

        # Check that common backends are supported
        backend_values = [backend.value for backend in common.BackendName]
        expected_backends = ["trtllm", "vllm", "sglang"]

        for backend in expected_backends:
            assert backend in backend_values

    def test_default_backend_is_trtllm(self):
        """Test that the default backend is trtllm."""
        assert common.BackendName.trtllm.value == "trtllm"


class TestQuantizationModes:
    """Test quantization mode configurations."""

    def test_gemm_quant_modes_exist(self):
        """Test that GEMM quantization modes are defined."""
        assert hasattr(common, "GEMMQuantMode")

        # Should have at least float16 and fp8
        gemm_modes = list(common.GEMMQuantMode)
        mode_names = [mode.name for mode in gemm_modes]

        assert "float16" in mode_names
        assert "fp8" in mode_names

    def test_attention_quant_modes_exist(self):
        """Test that attention quantization modes are defined."""
        assert hasattr(common, "FMHAQuantMode")
        assert hasattr(common, "KVCacheQuantMode")

        # Check FMHA modes
        fmha_modes = list(common.FMHAQuantMode)
        assert len(fmha_modes) > 0

        # Check KV cache modes
        kv_modes = list(common.KVCacheQuantMode)
        assert len(kv_modes) > 0

    def test_moe_quant_modes_exist(self):
        """Test that MoE quantization modes are defined."""
        assert hasattr(common, "MoEQuantMode")

        moe_modes = list(common.MoEQuantMode)
        mode_names = [mode.name for mode in moe_modes]

        assert "float16" in mode_names
        assert "fp8" in mode_names

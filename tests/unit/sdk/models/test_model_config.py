# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for model configuration functionality.

Tests model validation, default models, and model-specific configurations.
"""

from unittest.mock import patch

import pytest

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.models import check_is_moe, get_model, get_model_family
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

            # First element should be architecture string that maps to a valid model family
            architecture = config[0]
            assert isinstance(architecture, str)
            assert architecture in common.ARCHITECTURE_TO_MODEL_FAMILY, (
                f"Model {hf_id} has unknown architecture: {architecture}. "
                f"Supported architectures: {list(common.ARCHITECTURE_TO_MODEL_FAMILY.keys())}"
            )

    @pytest.mark.parametrize(
        "hf_id,is_moe_expected",
        [
            ("Qwen/Qwen3-32B", False),
            ("meta-llama/Meta-Llama-3.1-8B", False),
            ("deepseek-ai/DeepSeek-V3", True),
            ("mistralai/Mixtral-8x7B-v0.1", True),
            # NemotronH: check hybrid_override_pattern for 'E' (MoE layers)
            ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", True),  # Has 'E' in pattern
            ("nvidia/Nemotron-H-56B-Base-8K", False),  # No 'E' in pattern (only M, *, -)
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
            ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "NEMOTRONH"),
            ("nvidia/Nemotron-H-56B-Base-8K", "NEMOTRONH"),
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
            # NemotronH: is_moe depends on 'E' in hybrid_override_pattern
            ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", True),  # Has 'E' (MoE layers)
            ("nvidia/Nemotron-H-56B-Base-8K", False),  # No 'E' (Mamba + Attention + MLP only)
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
        assert "fp8_static" in mode_names

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


class TestMOEModelFP8BlockQuantizationValidation:
    """Test MOEModel._validate_fp8_block_quantized_moe_config() method."""

    @pytest.mark.parametrize(
        "moe_quant_mode,moe_tp_size,quantization_config,should_raise,test_id",
        [
            # Valid fp8_block config: 1536/4 = 384, 384 % 128 = 0
            (
                common.MoEQuantMode.fp8_block,
                4,
                {"weight_block_size": [128, 128]},
                False,
                "valid_fp8_block",
            ),
            # Invalid fp8_block config: 1536/8 = 192, 192 % 128 = 64
            (
                common.MoEQuantMode.fp8_block,
                8,
                {"weight_block_size": [128, 128]},
                True,
                "invalid_fp8_block",
            ),
            # Skip validation for float16 (even with invalid moe_tp)
            (
                common.MoEQuantMode.float16,
                8,
                {"weight_block_size": [128, 128]},
                False,
                "skip_validation_float16",
            ),
            # Skip validation for fp8 non-block mode
            (
                common.MoEQuantMode.fp8,
                8,
                {"weight_block_size": [128, 128]},
                False,
                "skip_validation_fp8_no_block",
            ),
            # Default block size when not in config: 1536/4 = 384, 384 % 128 = 0
            (
                common.MoEQuantMode.fp8_block,
                4,
                None,
                False,
                "default_block_size",
            ),
        ],
    )
    @patch("aiconfigurator.sdk.models._get_model_info")
    @patch("aiconfigurator.sdk.utils._load_model_config_from_model_path")
    def test_fp8_block_quantization_validation(
        self,
        mock_load_config,
        mock_get_info,
        moe_quant_mode,
        moe_tp_size,
        quantization_config,
        should_raise,
        test_id,
    ):
        """Parametrized test for fp8_block quantization validation."""
        # Setup mocks
        mock_get_info.return_value = (
            "MixtralForCausalLM",  # architecture
            32,  # layers
            32,  # n
            8,  # n_kv
            128,  # d
            4096,  # hidden
            14336,  # inter
            32000,  # vocab
            32768,  # context
            2,  # topk
            8,  # num_experts
            1536,  # moe_inter_size
            None,  # extra_params
        )
        config_dict = {"moe_intermediate_size": 1536}
        if quantization_config is not None:
            config_dict["quantization_config"] = quantization_config
        mock_load_config.return_value = config_dict

        # Create model config (tp_size * attention_dp_size must equal moe_tp_size * moe_ep_size)
        model_config = config.ModelConfig()
        model_config.moe_quant_mode = moe_quant_mode
        model_config.tp_size = moe_tp_size
        model_config.moe_tp_size = moe_tp_size
        model_config.moe_ep_size = 1
        model_config.attention_dp_size = 1

        # Test validation
        if should_raise:
            with pytest.raises(ValueError, match="Invalid quantized MoE configuration"):
                get_model("Qwen/Qwen3-235B-A22B", model_config, "trtllm")
        else:
            model = get_model("Qwen/Qwen3-235B-A22B", model_config, "trtllm")
            assert model is not None

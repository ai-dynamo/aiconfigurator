# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for model configuration functionality.

Tests model validation, default models, and model-specific configurations.
"""

from unittest.mock import patch

import pytest

from aiconfigurator.sdk import common, config, models
from aiconfigurator.sdk.models import LLAMAModel, Qwen3VLModel, check_is_moe, get_model, get_model_family
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
            assert isinstance(config, dict)
            assert "architecture" in config

            # First element should be architecture string that maps to a valid model family
            architecture = config["architecture"]
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
            ("deepseek-ai/DeepSeek-V3.2", True),
            ("zai-org/GLM-5", True),
            ("Qwen/Qwen3-30B-A3B", True),
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
            architecture = config["architecture"]
            assert architecture in common.ARCHITECTURE_TO_MODEL_FAMILY

    @pytest.mark.parametrize(
        "hf_id,expected_family",
        [
            ("Qwen/Qwen3-32B", "LLAMA"),
            ("meta-llama/Meta-Llama-3.1-8B", "LLAMA"),
            ("deepseek-ai/DeepSeek-V3", "DEEPSEEK"),
            ("deepseek-ai/DeepSeek-V3.2", "DEEPSEEKV32"),
            ("zai-org/GLM-5", "DEEPSEEKV32"),
            ("Qwen/Qwen3-30B-A3B", "MOE"),
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
            ("Qwen/Qwen3-32B", False),
            ("meta-llama/Meta-Llama-3.1-8B", False),
            ("deepseek-ai/DeepSeek-V3", True),
            ("deepseek-ai/DeepSeek-V3.2", True),
            ("zai-org/GLM-5", True),
            ("Qwen/Qwen3-30B-A3B", True),
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
        mock_get_info.return_value = {
            "architecture": "MixtralForCausalLM",
            "layers": 32,
            "n": 32,
            "n_kv": 8,
            "d": 128,
            "hidden_size": 4096,
            "inter_size": 14336,
            "vocab": 32000,
            "context": 32768,
            "topk": 2,
            "num_experts": 8,
            "moe_inter_size": 1536,
            "extra_params": None,
            "raw_config": {},
        }
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


class TestGetModelMOESGLangDispatch:
    """Test get_model() dispatch logic for MOE family with SGLang backend.

    Dispatch keys on moe_backend (communication path), not enable_wideep (scale intent).
    """

    def test_sglang_moe_deepep_returns_sglang_ep_moe_model(self):
        """DeepEP backend (inter-node, enable_wideep=True) → SGLangEPMOEModel."""
        model_config = config.ModelConfig(
            tp_size=1,
            pp_size=1,
            gemm_quant_mode=common.GEMMQuantMode.float16,
            kvcache_quant_mode=common.KVCacheQuantMode.float16,
            moe_tp_size=1,
            moe_ep_size=8,
            attention_dp_size=8,
            enable_wideep=True,
            moe_backend="deepep_moe",
        )
        model = models.get_model("Qwen/Qwen3-235B-A22B", model_config, "sglang")
        assert isinstance(model, models.SGLangEPMOEModel)

    def test_sglang_moe_deepep_intranode_returns_sglang_ep_moe_model(self):
        """DeepEP intra-node (enable_wideep=False, moe_backend=deepep_moe) → SGLangEPMOEModel."""
        model_config = config.ModelConfig(
            tp_size=1,
            pp_size=1,
            gemm_quant_mode=common.GEMMQuantMode.float16,
            kvcache_quant_mode=common.KVCacheQuantMode.float16,
            moe_tp_size=1,
            moe_ep_size=4,
            attention_dp_size=4,
            enable_wideep=False,
            moe_backend="deepep_moe",
        )
        model = models.get_model("Qwen/Qwen3-235B-A22B", model_config, "sglang")
        assert isinstance(model, models.SGLangEPMOEModel)

    def test_sglang_moe_no_deepep_returns_moe_model(self):
        """Standard comm (no moe_backend) → MOEModel."""
        model_config = config.ModelConfig(
            tp_size=2,
            pp_size=1,
            gemm_quant_mode=common.GEMMQuantMode.float16,
            kvcache_quant_mode=common.KVCacheQuantMode.float16,
            moe_tp_size=1,
            moe_ep_size=2,
            attention_dp_size=1,
            enable_wideep=False,
        )
        model = models.get_model("Qwen/Qwen3-235B-A22B", model_config, "sglang")
        assert isinstance(model, models.MOEModel)
        assert not isinstance(model, models.SGLangEPMOEModel)

    def test_trtllm_moe_returns_moe_model(self):
        """trtllm always → MOEModel (moe_backend irrelevant for non-sglang)."""
        model_config = config.ModelConfig(
            tp_size=2,
            pp_size=1,
            gemm_quant_mode=common.GEMMQuantMode.float16,
            kvcache_quant_mode=common.KVCacheQuantMode.float16,
            moe_tp_size=2,
            moe_ep_size=1,
            attention_dp_size=1,
            enable_wideep=True,
        )
        model = models.get_model("Qwen/Qwen3-235B-A22B", model_config, "trtllm")
        assert isinstance(model, models.MOEModel)
        assert not isinstance(model, models.SGLangEPMOEModel)


# ── Qwen3VL constants ──────────────────────────────────────────────────────────

_QWEN3VL_ARCH = "Qwen3VLForConditionalGeneration"
_VL_MODELS = [
    "Qwen/Qwen3-VL-32B-Instruct",
    "Qwen/Qwen3-VL-32B-Thinking",
]


class TestQwen3VLRegistration:
    """Test that Qwen3VL architecture is correctly registered in common.py."""

    def test_architecture_in_model_family_map(self):
        assert _QWEN3VL_ARCH in common.ARCHITECTURE_TO_MODEL_FAMILY

    def test_architecture_maps_to_llama_family(self):
        assert common.ARCHITECTURE_TO_MODEL_FAMILY[_QWEN3VL_ARCH] == "LLAMA"

    def test_architecture_in_multimodal_text_config_key(self):
        assert _QWEN3VL_ARCH in common.MULTIMODAL_TEXT_CONFIG_KEY

    def test_multimodal_text_config_key_is_text_config(self):
        assert common.MULTIMODAL_TEXT_CONFIG_KEY[_QWEN3VL_ARCH] == "text_config"

    @pytest.mark.parametrize("model_id", _VL_MODELS)
    def test_model_ids_in_default_hf_models(self, model_id):
        assert model_id in common.DefaultHFModels


class TestQwen3VLPredownloadedConfig:
    """Test get_model_config_from_model_path using the cached config.json files."""

    @pytest.mark.parametrize("model_id", _VL_MODELS)
    def test_config_loads_without_error(self, model_id):
        result = get_model_config_from_model_path(model_id)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("model_id", _VL_MODELS)
    def test_config_has_correct_architecture(self, model_id):
        result = get_model_config_from_model_path(model_id)
        assert result["architecture"] == _QWEN3VL_ARCH

    @pytest.mark.parametrize("model_id", _VL_MODELS)
    def test_config_has_correct_llm_params(self, model_id):
        result = get_model_config_from_model_path(model_id)
        assert result["layers"] == 64
        assert result["hidden_size"] == 5120
        assert result["n"] == 64
        assert result["n_kv"] == 8
        assert result["d"] == 128

    @pytest.mark.parametrize("model_id", _VL_MODELS)
    def test_extra_params_is_vision_encoder_config(self, model_id):
        result = get_model_config_from_model_path(model_id)
        assert isinstance(result["extra_params"], common.VisionEncoderConfig)

    @pytest.mark.parametrize("model_id", _VL_MODELS)
    def test_vision_encoder_params_from_downloaded_config(self, model_id):
        result = get_model_config_from_model_path(model_id)
        enc = result["extra_params"]
        assert enc.depth == 27
        assert enc.hidden_size == 1152
        assert enc.patch_size == 16
        assert enc.spatial_merge_size == 2
        assert enc.out_hidden_size == result["hidden_size"]

    @pytest.mark.parametrize("model_id", _VL_MODELS)
    def test_both_variants_have_identical_architecture(self, model_id):
        """Instruct and Thinking are fine-tunes of the same base — configs must match."""
        result = get_model_config_from_model_path(model_id)
        assert result["layers"] == 64
        assert result["vocab"] == 151936


class TestQwen3VLModel:
    """Test Qwen3VLModel class and get_model() factory for VL architecture."""

    @pytest.fixture
    def model_config(self):
        return config.ModelConfig()

    @pytest.fixture
    def vl_model(self, model_config):
        return get_model("Qwen/Qwen3-VL-32B-Instruct", model_config, "trtllm")

    def test_base_model_has_encoder_ops(self, model_config):
        """encoder_ops must be present on all models, not just VL ones."""
        model = get_model("Qwen/Qwen3-32B", model_config, "trtllm")
        assert hasattr(model, "encoder_ops")
        assert isinstance(model.encoder_ops, list)

    def test_non_vl_llama_has_empty_encoder_ops(self, model_config):
        model = get_model("Qwen/Qwen3-32B", model_config, "trtllm")
        assert len(model.encoder_ops) == 0

    def test_get_model_returns_qwen3vl_instance(self, vl_model):
        assert isinstance(vl_model, Qwen3VLModel)

    def test_get_model_vl_is_subclass_of_llama(self, vl_model):
        assert isinstance(vl_model, LLAMAModel)

    def test_vl_model_has_encoder_ops_populated(self, vl_model):
        assert len(vl_model.encoder_ops) > 0

    def test_vl_model_has_context_ops_populated(self, vl_model):
        """LLM context ops must still be present from LLAMAModel parent."""
        assert len(vl_model.context_ops) > 0

    def test_vl_model_has_generation_ops_populated(self, vl_model):
        """LLM generation ops must still be present from LLAMAModel parent."""
        assert len(vl_model.generation_ops) > 0

    def test_encoder_op_names(self, vl_model):
        """All expected encoder op names must be present."""
        names = [op._name for op in vl_model.encoder_ops]
        assert "encoder_qkv_gemm" in names
        assert "encoder_attention" in names
        assert "encoder_proj_gemm" in names
        assert "encoder_ffn1_gemm" in names
        assert "encoder_ffn2_gemm" in names
        assert "encoder_merger_fc1" in names
        assert "encoder_merger_act" in names
        assert "encoder_merger_fc2" in names

    def test_encoder_op_names_do_not_overlap_with_llm(self, vl_model):
        """Encoder op names must be distinct from LLM context op names."""
        encoder_names = {op._name for op in vl_model.encoder_ops}
        context_names = {op._name for op in vl_model.context_ops}
        assert encoder_names.isdisjoint(context_names)

    def test_vl_model_has_encoder_config_attribute(self, vl_model):
        """encoder_config must be stored on the model for use in _run_encoder."""
        assert hasattr(vl_model, "encoder_config")

    def test_vl_encoder_config_is_vision_encoder_config(self, vl_model):
        assert isinstance(vl_model.encoder_config, common.VisionEncoderConfig)

    def test_vl_encoder_config_depth(self, vl_model):
        assert vl_model.encoder_config.depth == 27

    def test_vl_encoder_config_patch_size(self, vl_model):
        assert vl_model.encoder_config.patch_size == 16

    def test_vl_encoder_config_spatial_merge_size(self, vl_model):
        assert vl_model.encoder_config.spatial_merge_size == 2

    def test_vl_encoder_config_out_hidden_size_matches_llm(self, vl_model):
        """out_hidden_size must equal LLM hidden_size for the projection to work."""
        assert vl_model.encoder_config.out_hidden_size == 5120

    @pytest.mark.parametrize("model_id", _VL_MODELS)
    def test_both_vl_variants_return_qwen3vl_model(self, model_id, model_config):
        model = get_model(model_id, model_config, "trtllm")
        assert isinstance(model, Qwen3VLModel)

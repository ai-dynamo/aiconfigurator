# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for model configuration functionality.

Tests model validation, default models, and model-specific configurations.
"""

from collections import Counter
from typing import ClassVar
from unittest.mock import patch

import pytest

import aiconfigurator.sdk.operations as ops
from aiconfigurator.sdk import common, config, models
from aiconfigurator.sdk.models import (
    LLAMAModel,
    Qwen3VLModel,
    Qwen3VLMoEModel,
    check_is_moe,
    get_model,
    get_model_family,
)
from aiconfigurator.sdk.performance_result import PerformanceResult
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
            "deepseek-ai/DeepSeek-V4-Flash",
            "deepseek-ai/DeepSeek-V4-Pro",
            "sgl-project/DeepSeek-V4-Flash-FP8",
            "sgl-project/DeepSeek-V4-Pro-FP8",
            "zai-org/GLM-5-FP8",
            "nvidia/GLM-5-NVFP4",
            "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
            "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8",
            "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
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
            ("deepseek-ai/DeepSeek-V4-Flash", True),
            ("deepseek-ai/DeepSeek-V4-Pro", True),
            ("sgl-project/DeepSeek-V4-Flash-FP8", True),
            ("sgl-project/DeepSeek-V4-Pro-FP8", True),
            ("zai-org/GLM-5", True),
            ("zai-org/GLM-5-FP8", True),
            ("nvidia/GLM-5-NVFP4", True),
            ("Qwen/Qwen3-30B-A3B", True),
            ("Qwen/Qwen3-VL-32B-Instruct", False),
            ("Qwen/Qwen3-VL-30B-A3B-Instruct", True),
            ("Qwen/Qwen3-VL-235B-A22B-Instruct", True),
            # NemotronH: check hybrid_override_pattern for 'E' (MoE layers)
            ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", True),  # Has 'E' in pattern
            ("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", True),  # Has 'E' in derived pattern
            ("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8", True),  # Has 'E' in derived pattern
            ("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4", True),  # Has 'E' in derived pattern
            ("nvidia/Nemotron-H-56B-Base-8K", False),  # No 'E' in pattern (only M, *, -)
        ],
    )
    def test_model_moe_detection(self, hf_id, is_moe_expected):
        """Test that MoE models are correctly identified."""
        is_moe = check_is_moe(hf_id)
        assert is_moe == is_moe_expected


class TestMOEParallelismResolution:
    """Regression tests for SDK-side MoE parallelism defaults."""

    def test_missing_moe_tp_size_is_inferred_for_minimax_nvfp4(self):
        model_config = config.ModelConfig(
            tp_size=1,
            attention_dp_size=1,
            moe_tp_size=None,
            moe_ep_size=1,
        )

        model = get_model("nvidia/MiniMax-M2.7-NVFP4", model_config, backend_name="vllm")

        assert model.model_family == "MOE"
        assert model_config.moe_tp_size == 1
        assert model_config.moe_ep_size == 1

    def test_minimax_m3_builds_with_msa_and_moe(self):
        """MiniMax-M3 registers as its own family and wires the MSA attention op + MoE."""
        from aiconfigurator.sdk.operations.msa import ContextMSAModule

        model_config = config.ModelConfig(tp_size=1, attention_dp_size=1, moe_tp_size=1, moe_ep_size=1)
        model = get_model("MiniMaxAI/MiniMax-M3", model_config, backend_name="trtllm")
        assert model.model_family == "MINIMAXM3"
        ctx = {op._name: op for op in model.context_ops}
        assert "context_attention" in ctx and "context_moe" in ctx
        assert isinstance(ctx["context_attention"], ContextMSAModule)  # MSA, not plain attention

    def test_nemotron_h_mtp_scales_generation_only(self):
        """Nemotron-3 ships num_nextn_predict_layers=1; MTP must build (no assert) and
        scale only the generation path. nextn=0 must be an exact 1.0 no-op."""

        def build(nextn):
            mc = config.ModelConfig(
                tp_size=8,
                moe_tp_size=1,
                moe_ep_size=8,
                nextn=nextn,
            )
            return get_model("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", mc, backend_name="trtllm")

        baseline = build(0)
        mtp = build(1)
        assert baseline._mtp_scale_factor == 1.0
        assert mtp._mtp_scale_factor > 1.0

        baseline_context = {op._name: op._scale_factor for op in baseline.context_ops}
        mtp_context = {op._name: op._scale_factor for op in mtp.context_ops}
        assert mtp_context["context_mamba_norm"] == baseline_context["context_mamba_norm"]

        baseline_generation = {op._name: op._scale_factor for op in baseline.generation_ops}
        mtp_generation = {op._name: op._scale_factor for op in mtp.generation_ops}
        for name in ("generation_embedding", "generation_mamba_norm", "generation_logits_gemm"):
            assert mtp_generation[name] == pytest.approx(baseline_generation[name] * mtp._mtp_scale_factor)

    def test_both_missing_moe_parallelism_raises_clear_error(self):
        model_config = config.ModelConfig(
            tp_size=4,
            attention_dp_size=2,
            moe_tp_size=None,
            moe_ep_size=None,
        )

        with pytest.raises(ValueError, match="At least one of moe_tp_size or moe_ep_size must be set"):
            get_model("Qwen/Qwen3-235B-A22B", model_config, backend_name="trtllm")

    @pytest.mark.parametrize(
        "tp_size,attention_dp_size,moe_tp_size,moe_ep_size,expected_moe_tp_size,expected_moe_ep_size",
        [
            (4, 2, None, 2, 4, 2),
            (4, 2, 2, None, 2, 4),
            (2, 4, None, 4, 2, 4),
            (2, 4, 1, None, 1, 8),
        ],
    )
    def test_partial_moe_parallelism_is_inferred_for_nontrivial_widths(
        self,
        tp_size,
        attention_dp_size,
        moe_tp_size,
        moe_ep_size,
        expected_moe_tp_size,
        expected_moe_ep_size,
    ):
        model_config = config.ModelConfig(
            tp_size=tp_size,
            attention_dp_size=attention_dp_size,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
        )

        get_model("Qwen/Qwen3-235B-A22B", model_config, backend_name="trtllm")

        assert model_config.moe_tp_size == expected_moe_tp_size
        assert model_config.moe_ep_size == expected_moe_ep_size

    def test_uninferrable_moe_parallelism_raises_clear_error(self):
        model_config = config.ModelConfig(
            tp_size=3,
            attention_dp_size=1,
            moe_tp_size=None,
            moe_ep_size=2,
        )

        with pytest.raises(ValueError, match="Cannot infer moe_tp_size"):
            get_model("Qwen/Qwen3-235B-A22B", model_config, backend_name="trtllm")

    def test_dense_model_does_not_resolve_moe_parallelism(self):
        model_config = config.ModelConfig(
            tp_size=1,
            attention_dp_size=1,
            moe_tp_size=None,
            moe_ep_size=None,
        )

        model = get_model("Qwen/Qwen3-32B", model_config, backend_name="trtllm")

        assert model.model_family == "LLAMA"
        assert model_config.moe_tp_size is None
        assert model_config.moe_ep_size is None


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
            ("deepseek-ai/DeepSeek-V4-Flash", "DEEPSEEKV4"),
            ("deepseek-ai/DeepSeek-V4-Pro", "DEEPSEEKV4"),
            ("sgl-project/DeepSeek-V4-Flash-FP8", "DEEPSEEKV4"),
            ("sgl-project/DeepSeek-V4-Pro-FP8", "DEEPSEEKV4"),
            ("zai-org/GLM-5", "DEEPSEEKV32"),
            ("zai-org/GLM-5-FP8", "DEEPSEEKV32"),
            ("nvidia/GLM-5-NVFP4", "DEEPSEEKV32"),
            ("Qwen/Qwen3-30B-A3B", "MOE"),
            ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "NEMOTRONH"),
            ("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", "NEMOTRONH"),
            ("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8", "NEMOTRONH"),
            ("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4", "NEMOTRONH"),
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
            ("deepseek-ai/DeepSeek-V4-Flash", True),
            ("deepseek-ai/DeepSeek-V4-Pro", True),
            ("sgl-project/DeepSeek-V4-Flash-FP8", True),
            ("sgl-project/DeepSeek-V4-Pro-FP8", True),
            ("zai-org/GLM-5", True),
            ("zai-org/GLM-5-FP8", True),
            ("nvidia/GLM-5-NVFP4", True),
            ("Qwen/Qwen3-30B-A3B", True),
            ("Qwen/Qwen3-VL-32B-Instruct", False),
            ("Qwen/Qwen3-VL-30B-A3B-Instruct", True),
            ("Qwen/Qwen3-VL-235B-A22B-Instruct", True),
            # NemotronH: is_moe depends on 'E' in hybrid_override_pattern
            ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", True),  # Has 'E' (MoE layers)
            ("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", True),  # Has 'E' in derived pattern
            ("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8", True),  # Has 'E' in derived pattern
            ("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4", True),  # Has 'E' in derived pattern
            ("nvidia/Nemotron-H-56B-Base-8K", False),  # No 'E' (Mamba + Attention + MLP only)
        ],
    )
    def test_hf_id_moe_detection(self, hf_id, is_moe_expected):
        """Test that MoE models are correctly identified via HF ID."""
        is_moe = check_is_moe(hf_id)
        assert is_moe == is_moe_expected

    @pytest.mark.parametrize(
        "hf_id",
        [
            "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
            "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8",
            "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
        ],
    )
    def test_nemotron_ultra_config_shape(self, hf_id):
        """Test Nemotron 3 Ultra layer-block config parsing."""
        model_info = get_model_config_from_model_path(hf_id)

        assert model_info["architecture"] == "NemotronHForCausalLM"
        assert model_info["layers"] == 108
        assert model_info["hidden_size"] == 8192
        assert model_info["inter_size"] == 5120
        assert model_info["topk"] == 22
        assert model_info["num_experts"] == 512

        extra = model_info["extra_params"]
        assert isinstance(extra, common.NemotronHConfig)
        assert Counter(extra.hybrid_override_pattern) == {"M": 48, "E": 48, "*": 12}
        assert extra.mamba_num_heads == 256
        assert extra.mamba_head_dim == 64
        assert extra.moe_shared_expert_intermediate_size == 10240

    @pytest.mark.parametrize(
        "hf_id,expected_gemm_quant,expected_moe_quant,expected_kvcache_quant,expected_fmha_quant",
        [
            (
                "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
                common.GEMMQuantMode.bfloat16,
                common.MoEQuantMode.bfloat16,
                common.KVCacheQuantMode.bfloat16,
                common.FMHAQuantMode.bfloat16,
            ),
            (
                "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8",
                common.GEMMQuantMode.fp8_static,
                common.MoEQuantMode.fp8,
                common.KVCacheQuantMode.fp8,
                common.FMHAQuantMode.fp8,
            ),
            (
                "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
                common.GEMMQuantMode.nvfp4,
                common.MoEQuantMode.nvfp4,
                common.KVCacheQuantMode.fp8,
                common.FMHAQuantMode.fp8,
            ),
        ],
    )
    def test_nemotron_ultra_quant_defaults(
        self,
        hf_id,
        expected_gemm_quant,
        expected_moe_quant,
        expected_kvcache_quant,
        expected_fmha_quant,
    ):
        """Test official Nemotron 3 Ultra precision-specific quant defaults."""
        model_config = config.ModelConfig(
            tp_size=8,
            pp_size=1,
            moe_tp_size=1,
            moe_ep_size=8,
            attention_dp_size=1,
        )
        model = get_model(hf_id, model_config, backend_name="trtllm")

        assert model.model_family == "NEMOTRONH"
        assert model_config.gemm_quant_mode == expected_gemm_quant
        assert model_config.moe_quant_mode == expected_moe_quant
        assert model_config.kvcache_quant_mode == expected_kvcache_quant
        assert model_config.fmha_quant_mode == expected_fmha_quant
        assert sum(op._scale_factor for op in model.context_ops if op._name == "context_mamba_norm") == 48
        assert sum(op._scale_factor for op in model.context_ops if op._name == "context_moe_norm") == 48
        assert sum(op._scale_factor for op in model.context_ops if op._name == "context_attn_norm") == 12

    @pytest.mark.parametrize(
        "hf_id,expected_layers,expected_hidden,expected_index_topk,expected_ratio_counts,expected_moe_quant",
        [
            (
                "deepseek-ai/DeepSeek-V4-Flash",
                43,
                4096,
                512,
                {0: 2, 4: 21, 128: 20},
                common.MoEQuantMode.w4a8_mxfp4_mxfp8,
            ),
            (
                "deepseek-ai/DeepSeek-V4-Pro",
                61,
                7168,
                1024,
                {4: 30, 128: 31},
                common.MoEQuantMode.w4a8_mxfp4_mxfp8,
            ),
            (
                "sgl-project/DeepSeek-V4-Flash-FP8",
                43,
                4096,
                512,
                {0: 2, 4: 21, 128: 20},
                common.MoEQuantMode.fp8_block,
            ),
            (
                "sgl-project/DeepSeek-V4-Pro-FP8",
                61,
                7168,
                1024,
                {4: 30, 128: 31},
                common.MoEQuantMode.fp8_block,
            ),
        ],
    )
    def test_deepseek_v4_config_shape_and_quant(
        self,
        hf_id,
        expected_layers,
        expected_hidden,
        expected_index_topk,
        expected_ratio_counts,
        expected_moe_quant,
    ):
        model_info = get_model_config_from_model_path(hf_id)
        assert model_info["architecture"] == "DeepseekV4ForCausalLM"
        assert model_info["layers"] == expected_layers
        assert model_info["hidden_size"] == expected_hidden
        assert model_info["topk"] == 6
        assert model_info["num_experts"] in {256, 384}

        extra = model_info["extra_params"]
        assert isinstance(extra, common.DeepSeekV4Config)
        assert extra.index_topk == expected_index_topk
        assert extra.hc_mult == 4
        observed_ratio_counts = {ratio: extra.compress_ratios.count(ratio) for ratio in set(extra.compress_ratios)}
        assert observed_ratio_counts == expected_ratio_counts

        model_config = config.ModelConfig(
            tp_size=1,
            moe_tp_size=1,
            moe_ep_size=1,
            nextn=1,
        )
        model = get_model(hf_id, model_config, backend_name="trtllm")
        assert model.model_family == "DEEPSEEKV4"
        assert model_config.gemm_quant_mode == common.GEMMQuantMode.fp8_block
        assert model_config.moe_quant_mode == expected_moe_quant
        assert model_config.kvcache_quant_mode == common.KVCacheQuantMode.fp8
        assert model_config.fmha_quant_mode == common.FMHAQuantMode.bfloat16
        assert sum(op._scale_factor for op in model.context_ops if op._name == "context_attention") == expected_layers
        op_ratio_counts = Counter()
        for op in model.context_ops:
            if op._name == "context_attention":
                op_ratio_counts[op._compress_ratio] += op._scale_factor
        assert op_ratio_counts[0] == 0
        assert op_ratio_counts[4] == expected_ratio_counts.get(4, 0)
        assert op_ratio_counts[128] == expected_ratio_counts.get(128, 0) + expected_ratio_counts.get(0, 0)

    def test_deepseek_v4_kvcache_bytes_include_csa_indexer_cache_and_decode_buffers(self):
        model_config = config.ModelConfig(
            tp_size=8,
            moe_tp_size=1,
            moe_ep_size=8,
            attention_dp_size=1,
            nextn=1,
        )
        model = get_model("sgl-project/DeepSeek-V4-Pro-FP8", model_config, backend_name="trtllm")
        seq_len = 4096
        extra = model.extra_params

        expected = 0.0
        without_indexer = 0.0
        cache_entry_bytes = extra.head_dim * model_config.kvcache_quant_mode.value.memory
        for ratio in extra.compress_ratios:
            local_bytes = min(seq_len, extra.sliding_window) * cache_entry_bytes
            expected += local_bytes
            without_indexer += local_bytes
            if ratio:
                compressed_bytes = (seq_len // ratio) * cache_entry_bytes
                expected += compressed_bytes
                without_indexer += compressed_bytes
                coff = 2 if ratio == 4 else 1
                buffer_bytes = 2 * ratio * coff * extra.head_dim * 4
                expected += buffer_bytes
                without_indexer += buffer_bytes
                if ratio == 4:
                    expected += (seq_len // ratio) * common.deepseek_v4_indexer_cache_entry_bytes(extra.index_head_dim)
                    expected += 2 * ratio * 2 * extra.index_head_dim * 4

        assert model.get_kvcache_bytes_per_sequence(seq_len) == expected
        assert expected > without_indexer

    def test_deepseek_v4_shared_expert_ops_are_tp_sharded(self):
        model_config = config.ModelConfig(
            tp_size=4,
            moe_tp_size=1,
            moe_ep_size=4,
            attention_dp_size=1,
            nextn=1,
        )
        model = get_model("sgl-project/DeepSeek-V4-Pro-FP8", model_config, backend_name="trtllm")
        local_inter_size = model._moe_inter_size // model_config.tp_size

        context_gate = next(op for op in model.context_ops if op._name == "context_shared_gate_up_gemm")
        context_act = next(op for op in model.context_ops if op._name == "context_shared_act_gate")
        context_down = next(op for op in model.context_ops if op._name == "context_shared_ffn2_gemm")
        generation_overlap = next(op for op in model.generation_ops if op._name == "generation_moe_overlap")
        generation_gate = next(op for op in generation_overlap._group_b if op._name == "generation_shared_gate_up_gemm")
        generation_act = next(op for op in generation_overlap._group_b if op._name == "generation_shared_act_gate")
        generation_down = next(op for op in generation_overlap._group_b if op._name == "generation_shared_ffn2_gemm")

        assert context_gate._n == 2 * local_inter_size
        assert context_act._dim_in == 2 * local_inter_size
        assert context_act._dim_out == local_inter_size
        assert context_down._k == local_inter_size
        assert generation_gate._n == 2 * local_inter_size
        assert generation_act._dim_in == 2 * local_inter_size
        assert generation_act._dim_out == local_inter_size
        assert generation_down._k == local_inter_size

    def test_deepseek_v4_sglang_megamoe_backend_uses_megamoe_module(self):
        model_config = config.ModelConfig(
            tp_size=1,
            moe_tp_size=1,
            moe_ep_size=8,
            attention_dp_size=8,
            moe_backend="megamoe",
            workload_distribution="uniform",
            nextn=1,
        )
        model = get_model("deepseek-ai/DeepSeek-V4-Pro", model_config, backend_name="sglang")

        context_names = [op._name for op in model.context_ops]
        assert "context_megamoe" in context_names
        assert "context_moe_pre_dispatch" not in context_names
        assert "context_moe_post_dispatch" not in context_names

        generation_overlap = next(op for op in model.generation_ops if op._name == "generation_moe_overlap")
        generation_names = [op._name for op in generation_overlap._group_a]
        assert "generation_megamoe" in generation_names
        assert "generation_moe_pre_dispatch" not in generation_names
        generation_megamoe = next(op for op in generation_overlap._group_a if op._name == "generation_megamoe")
        assert generation_megamoe._workload_distribution == "balanced"

    def test_deepseek_v4_sglang_deepep_backend_keeps_decomposed_moe(self):
        model_config = config.ModelConfig(
            tp_size=1,
            moe_tp_size=1,
            moe_ep_size=8,
            attention_dp_size=8,
            moe_backend="deepep_moe",
            nextn=1,
        )
        model = get_model("deepseek-ai/DeepSeek-V4-Pro", model_config, backend_name="sglang")

        context_names = [op._name for op in model.context_ops]
        assert "context_megamoe" not in context_names
        assert "context_moe_pre_dispatch" in context_names
        assert "context_moe_post_dispatch" in context_names

    def test_deepseek_v4_megamoe_module_query_uses_local_rank_tokens(self):
        class FakeDatabase:
            system_spec: ClassVar[dict[str, dict[str, int]]] = {"gpu": {"sm_version": 100}}

            def query_dsv4_megamoe_module(self, **kwargs):
                self.kwargs = kwargs
                return PerformanceResult(2.0, energy=3.0)

        database = FakeDatabase()
        op = ops.DeepSeekV4MegaMoEModule(
            "test_megamoe",
            2,
            hidden_size=7168,
            inter_size=3072,
            topk=6,
            num_experts=384,
            moe_tp_size=1,
            moe_ep_size=8,
            quant_mode=common.MoEQuantMode.w4a8_mxfp4_mxfp8,
            workload_distribution="uniform",
        )

        result = op.query(database, x=16)

        assert float(result) == 4.0
        assert result.energy == 6.0
        assert database.kwargs["num_tokens"] == 16
        assert database.kwargs["workload_distribution"] == "balanced"
        assert database.kwargs["source_policy"] == "random"

    def test_deepseek_v4_megamoe_module_rejects_non_blackwell_database(self):
        class FakeDatabase:
            system_spec: ClassVar[dict[str, dict[str, int]]] = {"gpu": {"sm_version": 90}}

        op = ops.DeepSeekV4MegaMoEModule(
            "test_megamoe",
            1,
            hidden_size=7168,
            inter_size=3072,
            topk=6,
            num_experts=384,
            moe_tp_size=1,
            moe_ep_size=8,
            quant_mode=common.MoEQuantMode.w4a8_mxfp4_mxfp8,
            workload_distribution="power_law_1.01",
        )

        with pytest.raises(ValueError, match="Blackwell"):
            op.query(FakeDatabase(), x=16)

    def test_deepseek_v32_kvcache_bytes_include_indexer_cache(self):
        model_config = config.ModelConfig(
            tp_size=8,
            moe_tp_size=1,
            moe_ep_size=8,
            attention_dp_size=1,
            kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        )
        model = get_model("deepseek-ai/DeepSeek-V3.2", model_config, backend_name="trtllm")
        seq_len = 4096
        extra = model.extra_params
        indexer_bytes = common.indexer_cache_entry_bytes(extra["index_head_dim"])

        expected = (
            model._num_layers
            * seq_len
            * (
                extra["kv_lora_rank"] * model_config.kvcache_quant_mode.value.memory
                + extra["qk_rope_head_dim"] * common.GEMMQuantMode.bfloat16.value.memory
                + indexer_bytes
            )
        )
        old_without_indexer = (
            model._num_layers
            * seq_len
            * (
                extra["kv_lora_rank"] * model_config.kvcache_quant_mode.value.memory
                + extra["qk_rope_head_dim"] * common.GEMMQuantMode.bfloat16.value.memory
            )
        )

        assert model.get_kvcache_bytes_per_sequence(seq_len) == expected
        assert expected > old_without_indexer

    @pytest.mark.parametrize(
        "hf_id,expected_gemm_quant,expected_moe_quant",
        [
            ("zai-org/GLM-5-FP8", common.GEMMQuantMode.fp8_block, common.MoEQuantMode.fp8_block),
            ("nvidia/GLM-5-NVFP4", common.GEMMQuantMode.nvfp4, common.MoEQuantMode.nvfp4),
        ],
    )
    def test_glm5_quantized_cached_config_uses_deepseek_v32_family(
        self,
        hf_id,
        expected_gemm_quant,
        expected_moe_quant,
    ):
        model_info = get_model_config_from_model_path(hf_id)
        assert model_info["architecture"] == "GlmMoeDsaForCausalLM"

        model_config = config.ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1)
        model = get_model(hf_id, model_config, backend_name="sglang")

        assert model.model_family == "DEEPSEEKV32"
        assert model_config.gemm_quant_mode == expected_gemm_quant
        assert model_config.moe_quant_mode == expected_moe_quant
        assert model_config.fmha_quant_mode == common.FMHAQuantMode.bfloat16

    def test_glm5_nvfp4_dsa_attention_uses_unquantized_projection_tables(self):
        model_config = config.ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1)
        model = get_model("nvidia/GLM-5-NVFP4", model_config, backend_name="sglang")

        context_dsa = next(op for op in model.context_ops if op._name == "context_attention")
        generation_dsa = next(op for op in model.generation_ops if op._name == "generation_attention")

        assert model_config.gemm_quant_mode == common.GEMMQuantMode.nvfp4
        assert model_config.moe_quant_mode == common.MoEQuantMode.nvfp4
        assert context_dsa._gemm_quant_mode == common.GEMMQuantMode.bfloat16
        assert generation_dsa._gemm_quant_mode == common.GEMMQuantMode.bfloat16


class TestKVCacheElementsPerToken:
    """Regression tests for ``BaseModel.get_kvcache_elements_per_token``.

    Guards against the bug where MLA models other than DeepSeek (notably
    KIMIK25 / Kimi K2.5) fell through to the GQA branch in the backend
    memory model, overestimating per-token KV cache by ~6x and capping the
    feasible batch size in the agg sweep.
    """

    @staticmethod
    def _build_model(hf_id: str, tp_size: int, **extra):
        model_config = config.ModelConfig(tp_size=tp_size, pp_size=1, attention_dp_size=1, **extra)
        return models.get_model(hf_id, model_config, backend_name="trtllm")

    @pytest.mark.parametrize(
        "hf_id,tp,moe_kw,expected_family,expected_elems",
        [
            # MLA path: 61 layers * (kv_lora_rank=512 + qk_rope_head_dim=64) = 35136
            (
                "nvidia/Kimi-K2.5-NVFP4",
                4,
                {"moe_tp_size": 2, "moe_ep_size": 2},
                "KIMIK25",
                35136,
            ),
            (
                "deepseek-ai/DeepSeek-V3",
                4,
                {"moe_tp_size": 1, "moe_ep_size": 4},
                "DEEPSEEK",
                35136,
            ),
            (
                "deepseek-ai/DeepSeek-V3.2",
                4,
                {"moe_tp_size": 1, "moe_ep_size": 4},
                "DEEPSEEKV32",
                35136,
            ),
            # GQA path: num_kv_heads_per_gpu * head_size * num_layers * 2
            ("meta-llama/Meta-Llama-3.1-8B", 1, {}, "LLAMA", 8 * 128 * 32 * 2),
            ("Qwen/Qwen3-32B", 4, {}, "LLAMA", 2 * 128 * 64 * 2),
            (
                "Qwen/Qwen3-30B-A3B",
                4,
                {"moe_tp_size": 1, "moe_ep_size": 4},
                "MOE",
                1 * 128 * 48 * 2,
            ),
        ],
    )
    def test_kvcache_elements_per_token(self, hf_id, tp, moe_kw, expected_family, expected_elems):
        model = self._build_model(hf_id, tp, **moe_kw)
        assert model.model_family == expected_family
        assert model.get_kvcache_elements_per_token() == expected_elems

    @pytest.mark.parametrize(
        "hf_id",
        [
            "nvidia/Kimi-K2.5-NVFP4",
            "deepseek-ai/DeepSeek-V3",
            "deepseek-ai/DeepSeek-V3.2",
        ],
    )
    def test_mla_dims_exposed_via_extra_params(self, hf_id):
        """Parser must expose kv_lora_rank/qk_rope_head_dim so the KV cache
        size is data-driven instead of relying on the 512/64 fallback."""
        parsed = get_model_config_from_model_path(hf_id)
        extra = parsed["extra_params"]
        assert isinstance(extra, dict), f"{hf_id}: extra_params should be a dict"
        assert extra.get("kv_lora_rank") == 512, f"{hf_id}: kv_lora_rank not extracted"
        assert extra.get("qk_rope_head_dim") == 64, f"{hf_id}: qk_rope_head_dim not extracted"

    def test_kimik25_does_not_use_gqa_branch(self):
        """Direct regression for the original concurrency cap bug: with the
        GQA branch, KIMIK25 would compute 16*112*61*2 = 218624 elems/token at
        TP=4 (a ~6.2x overestimate). The MLA branch must produce 35136."""
        model = self._build_model("nvidia/Kimi-K2.5-NVFP4", tp_size=4, moe_tp_size=2, moe_ep_size=2)
        gqa_elems = (
            ((model._num_kv_heads + model.config.tp_size - 1) // model.config.tp_size)
            * model._head_size
            * model._num_layers
            * 2
        )
        assert gqa_elems != model.get_kvcache_elements_per_token()
        assert model.get_kvcache_elements_per_token() == model._num_layers * (512 + 64)


class TestGetKvcacheMaxTokens:
    """``Model.get_kvcache_max_tokens`` -- the capacity-sizing inverse of
    ``get_kvcache_bytes_per_sequence``.

    Linear-growth models (GQA / MLA) must invert to exact floor-division by the
    per-token size; non-linear models (DeepSeek-V4's window-capped + compressed
    attention) must follow the true piecewise curve via the monotonic search,
    which also fits strictly more tokens than the seq_len=1 extrapolation.
    """

    @staticmethod
    def _build_model(hf_id: str, tp_size: int, **extra):
        model_config = config.ModelConfig(tp_size=tp_size, pp_size=1, attention_dp_size=1, **extra)
        return models.get_model(hf_id, model_config, backend_name="trtllm")

    @pytest.mark.parametrize(
        "hf_id,tp,moe_kw",
        [
            ("meta-llama/Meta-Llama-3.1-8B", 1, {}),  # GQA, linear
            ("Qwen/Qwen3-32B", 4, {}),  # GQA, linear
            ("deepseek-ai/DeepSeek-V3.2", 4, {"moe_tp_size": 1, "moe_ep_size": 4}),  # MLA, linear
        ],
    )
    def test_linear_models_invert_to_floor_division(self, hf_id, tp, moe_kw):
        model = self._build_model(hf_id, tp, **moe_kw)
        per_token = model.get_kvcache_bytes_per_sequence(1)
        for seq_len in (1, 137, 4096, 200_000):
            budget = model.get_kvcache_bytes_per_sequence(seq_len)
            assert model.get_kvcache_max_tokens(budget) == int(budget // per_token) == seq_len

    def test_zero_or_sub_token_budget_returns_zero(self):
        model = self._build_model("Qwen/Qwen3-32B", 4)
        assert model.get_kvcache_max_tokens(0) == 0
        assert model.get_kvcache_max_tokens(model.get_kvcache_bytes_per_sequence(1) - 1) == 0

    def test_deepseek_v4_inverts_nonlinear_curve(self):
        """DeepSeek-V4 caps local attention at its window and compresses past it
        (plus fixed decode-state buffers), so its KV growth is non-linear; the
        inverse follows that curve and beats the seq_len=1 extrapolation."""
        model_config = config.ModelConfig(
            tp_size=8,
            moe_tp_size=1,
            moe_ep_size=8,
            attention_dp_size=1,
            nextn=1,
        )
        model = models.get_model("sgl-project/DeepSeek-V4-Pro-FP8", model_config, backend_name="trtllm")
        window = model.extra_params.sliding_window
        budget = model.get_kvcache_bytes_per_sequence(window * 8)  # well past the window
        tokens = model.get_kvcache_max_tokens(budget)
        # Exact monotonic inverse: `tokens` fits, `tokens + 1` does not.
        assert model.get_kvcache_bytes_per_sequence(tokens) <= budget
        assert model.get_kvcache_bytes_per_sequence(tokens + 1) > budget
        # The seq_len=1 slope (inflated by the fixed buffers + uncapped local KV)
        # under-counts capacity; the curve-aware inverse fits more.
        per_token = model.get_kvcache_bytes_per_sequence(1)
        assert tokens > int(budget // per_token)


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

        # Should have at least bfloat16 and fp8
        gemm_modes = list(common.GEMMQuantMode)
        mode_names = [mode.name for mode in gemm_modes]

        assert "bfloat16" in mode_names
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

        assert "bfloat16" in mode_names
        assert "fp8" in mode_names

    @pytest.mark.parametrize(
        "hf_id,backend_name",
        [
            ("deepseek-ai/DeepSeek-V3", "trtllm"),
            ("deepseek-ai/DeepSeek-V3", "sglang"),
            ("nvidia/Kimi-K2.5-NVFP4", "trtllm"),
            ("nvidia/Kimi-K2.5-NVFP4", "sglang"),
        ],
    )
    def test_deepseek_v3_and_kimi_keep_fp8_fmha_for_supported_backends(self, hf_id, backend_name):
        model_info = get_model_config_from_model_path(hf_id)
        model_config = config.ModelConfig()

        models._apply_model_quant_defaults(
            model_config,
            model_info["raw_config"],
            model_info["architecture"],
            backend_name,
        )

        assert model_config.kvcache_quant_mode == common.KVCacheQuantMode.fp8
        assert model_config.fmha_quant_mode == common.FMHAQuantMode.fp8

    def test_vllm_still_uses_bfloat16_fmha_tables_for_quantized_models(self):
        model_info = get_model_config_from_model_path("deepseek-ai/DeepSeek-V3")
        model_config = config.ModelConfig()

        models._apply_model_quant_defaults(
            model_config,
            model_info["raw_config"],
            model_info["architecture"],
            "vllm",
        )

        assert model_config.kvcache_quant_mode == common.KVCacheQuantMode.fp8
        assert model_config.fmha_quant_mode == common.FMHAQuantMode.bfloat16


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
            # Skip validation for bfloat16 (even with invalid moe_tp)
            (
                common.MoEQuantMode.bfloat16,
                8,
                {"weight_block_size": [128, 128]},
                False,
                "skip_validation_bfloat16",
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
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
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
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
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
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
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
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            moe_tp_size=2,
            moe_ep_size=1,
            attention_dp_size=1,
            enable_wideep=True,
        )
        model = models.get_model("Qwen/Qwen3-235B-A22B", model_config, "trtllm")
        assert isinstance(model, models.MOEModel)
        assert not isinstance(model, models.SGLangEPMOEModel)


class TestDeepSeekTPAllReduce:
    """vLLM TP allreduce coverage in DeepSeekModel context+generation ops."""

    @staticmethod
    def _build(hf_id: str, backend: str, tp_size: int):
        model_config = config.ModelConfig(
            tp_size=tp_size,
            pp_size=1,
            moe_tp_size=1,
            moe_ep_size=tp_size,
            attention_dp_size=1,
        )
        return models.get_model(hf_id, model_config, backend_name=backend)

    def test_vllm_has_generation_tp_allreduce_scaled_by_2x_num_layers(self):
        model = self._build("nvidia/Kimi-K2.5-NVFP4", "vllm", tp_size=4)
        ar_ops = [op for op in model.generation_ops if op._name == "generation_tp_allreduce"]
        assert len(ar_ops) == 1, "vLLM DeepSeekModel must emit one generation_tp_allreduce op"
        ar = ar_ops[0]
        assert ar._tp_size == 4
        assert ar._h == model._hidden_size
        assert ar._scale_factor == pytest.approx(2 * model._num_layers * model._mtp_scale_factor)

    def test_vllm_deepseek_v3_also_emits_tp_allreduce(self):
        # DEEPSEEK family (non-KIMIK25) goes through the create() fallback;
        # confirm backend_name is threaded so DS-V3 + vLLM also emits the op.
        # Also confirm the parser exposes v_head_dim (128 for the MLA arch) so
        # the existing vLLM attention-swap doesn't silently use head_size=56.
        model = self._build("deepseek-ai/DeepSeek-V3", "vllm", tp_size=4)
        gen_ar = [op for op in model.generation_ops if op._name == "generation_tp_allreduce"]
        ctx_ar = [op for op in model.context_ops if op._name == "context_tp_allreduce"]
        assert len(gen_ar) == 1 and gen_ar[0]._tp_size == 4
        assert len(ctx_ar) == 1 and ctx_ar[0]._tp_size == 4
        assert model._vllm_head_size == 128

    def test_vllm_has_context_tp_allreduce_scaled_by_2x_num_layers(self):
        model = self._build("nvidia/Kimi-K2.5-NVFP4", "vllm", tp_size=4)
        ar_ops = [op for op in model.context_ops if op._name == "context_tp_allreduce"]
        assert len(ar_ops) == 1, "vLLM DeepSeekModel must emit one context_tp_allreduce op"
        ar = ar_ops[0]
        assert ar._tp_size == 4
        assert ar._h == model._hidden_size
        # context_ops are NOT mtp-scaled (matches the rest of context_ops).
        assert ar._scale_factor == pytest.approx(2 * model._num_layers)

    def test_vllm_tp1_keeps_op_but_query_is_zero(self):
        # The op stays in the list for tp_size=1 (CustomAllReduce.query handles the
        # short-circuit), so the model has uniform shape regardless of TP.
        model = self._build("nvidia/Kimi-K2.5-NVFP4", "vllm", tp_size=1)
        gen_ar = [op for op in model.generation_ops if op._name == "generation_tp_allreduce"]
        ctx_ar = [op for op in model.context_ops if op._name == "context_tp_allreduce"]
        assert len(gen_ar) == 1 and gen_ar[0]._tp_size == 1
        assert len(ctx_ar) == 1 and ctx_ar[0]._tp_size == 1

    def test_trtllm_narrow_ep_does_not_emit_op(self):
        # Issue is scoped to vLLM; TRT-LLM narrow-EP path through DeepSeekModel
        # must not gain a spurious tp_allreduce op (its allreduce is modeled
        # elsewhere — or, like today, is a separate latent gap to be tracked).
        model = self._build("deepseek-ai/DeepSeek-V3", "trtllm", tp_size=4)
        assert not any(op._name == "generation_tp_allreduce" for op in model.generation_ops)
        assert not any(op._name == "context_tp_allreduce" for op in model.context_ops)

    def test_sglang_narrow_ep_does_not_emit_op(self):
        model = self._build("deepseek-ai/DeepSeek-V3", "sglang", tp_size=4)
        assert not any(op._name == "generation_tp_allreduce" for op in model.generation_ops)
        assert not any(op._name == "context_tp_allreduce" for op in model.context_ops)


# ── Qwen3VL constants ──────────────────────────────────────────────────────────

_QWEN3VL_ARCH = "Qwen3VLForConditionalGeneration"
_QWEN3VL_MOE_ARCH = "Qwen3VLMoeForConditionalGeneration"
_VL_MODELS = [
    "Qwen/Qwen3-VL-32B-Instruct",
    "Qwen/Qwen3-VL-32B-Thinking",
]


class TestQwen3VLRegistration:
    """Test that Qwen3VL architecture is correctly registered in common.py."""

    def test_architecture_in_model_family_map(self):
        assert _QWEN3VL_ARCH in common.ARCHITECTURE_TO_MODEL_FAMILY

    def test_architecture_maps_to_qwen3vl_family(self):
        assert common.ARCHITECTURE_TO_MODEL_FAMILY[_QWEN3VL_ARCH] == "QWEN3VL"

    def test_moe_architecture_maps_to_qwen3vl_moe_family(self):
        assert common.ARCHITECTURE_TO_MODEL_FAMILY[_QWEN3VL_MOE_ARCH] == "QWEN3VL_MOE"

    def test_qwen3vl_families_are_registered(self):
        assert "QWEN3VL" in common.ModelFamily
        assert "QWEN3VL_MOE" in common.ModelFamily

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

    def test_get_model_returns_qwen3vl_moe_instance(self):
        model_config = config.ModelConfig(moe_tp_size=1, moe_ep_size=1)
        model = get_model("Qwen/Qwen3-VL-30B-A3B-Instruct", model_config, "trtllm")
        assert isinstance(model, Qwen3VLMoEModel)

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
        assert "encoder_projector_fc0_gemm" in names
        assert "encoder_projector_fc0_act" in names
        assert "encoder_projector_fc1_gemm" in names
        assert "encoder_projector_ar" in names

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


class TestDSAAttentionQuantExclusion:
    """GLM-5 DSA: detect ModelOpt keeping attention projections unquantized (bf16).

    Regression for the nvidia/GLM-5-NVFP4 exclude format — layer-prefixed
    ``model.layers.N.self_attn*`` globs and the ``ignore`` key — which the
    earlier full-projection-name substring check missed, silently modeling DSA
    attention at NVFP4 instead of bf16.
    """

    @staticmethod
    def _excluded(raw):
        from aiconfigurator.sdk.models.deepseek_v32 import (
            _dsa_attention_modules_excluded_from_quant,
        )

        return _dsa_attention_modules_excluded_from_quant(raw)

    def test_modelopt_layer_prefixed_self_attn_globs(self):
        raw = {
            "quantization_config": {
                "quant_algo": "NVFP4",
                "ignore": ["lm_head", "model.layers.10.self_attn*"],
            },
            "hf_quant_config": {
                "quantization": {
                    "exclude_modules": [
                        "lm_head",
                        "model.layers.10.self_attn*",
                        "model.layers.10.mlp.shared_experts*",
                    ]
                }
            },
        }
        assert self._excluded(raw) is True

    def test_full_projection_names_backward_compatible(self):
        raw = {"quantization_config": {"modules_to_not_convert": ["model.layers.0.self_attn.q_a_proj"]}}
        assert self._excluded(raw) is True

    def test_ignore_key_regex_pattern(self):
        raw = {"quantization_config": {"ignore": ["re:.*self_attn.*"]}}
        assert self._excluded(raw) is True

    def test_hf_quant_ignore_key_pattern(self):
        raw = {"hf_quant_config": {"quantization": {"ignore": ["model.layers.3.self_attn*"]}}}
        assert self._excluded(raw) is True

    def test_attention_quantized_returns_false(self):
        raw = {"quantization_config": {"exclude_modules": ["lm_head", "model.layers.0.mlp.shared_experts.gate"]}}
        assert self._excluded(raw) is False

    def test_empty_config_returns_false(self):
        assert self._excluded({}) is False

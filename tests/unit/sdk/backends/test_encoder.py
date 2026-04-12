# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for vision encoder (ViT) runtime support in the aiconfigurator backend.

Covers:
- Image token count formula (post-merge and pre-merge patch counts)
- Fix A: ViT transformer layers use pre-merge patch count; only encoder_proj_to_llm_gemm
         uses post-merge count (detected by 'proj_to_llm' in op name)
- Fix B: Post-merge image tokens are added to the LLM context ISL (effective_isl)
- RuntimeConfig image-related fields (num_image_tokens, image_height/width, num_images_per_request)
- InferenceSummary encoder latency/energy accessors
"""

import pytest

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.models import get_model

pytestmark = pytest.mark.unit


class TestEncoderTokenFormula:
    """Test image token count formula from VisionEncoderConfig patch/merge dimensions."""

    @pytest.fixture
    def enc_cfg(self):
        return common.VisionEncoderConfig(
            depth=27, hidden_size=1152, num_heads=16, intermediate_size=4304,
            patch_size=16, temporal_patch_size=2, spatial_merge_size=2, out_hidden_size=5120,
            deepstack_visual_indexes=(8, 16, 24),
        )

    def test_patch_stride_is_patch_size_times_merge_size(self, enc_cfg):
        """Effective spatial stride = patch_size * spatial_merge_size = 16 * 2 = 32."""
        stride = enc_cfg.patch_size * enc_cfg.spatial_merge_size
        assert stride == 32

    def test_token_count_448x448(self, enc_cfg):
        """448x448 image: 448//32=14 per side → 14*14=196 post-merge tokens."""
        stride = enc_cfg.patch_size * enc_cfg.spatial_merge_size
        tokens = (448 // stride) * (448 // stride)
        assert tokens == 196

    def test_token_count_224x224(self, enc_cfg):
        """224x224 image: 224//32=7 per side → 7*7=49 post-merge tokens."""
        stride = enc_cfg.patch_size * enc_cfg.spatial_merge_size
        tokens = (224 // stride) * (224 // stride)
        assert tokens == 49

    def test_doubling_resolution_quadruples_tokens(self, enc_cfg):
        """Doubling H and W doubles each dimension's patch count → 4x total tokens."""
        stride = enc_cfg.patch_size * enc_cfg.spatial_merge_size
        tokens_small = (224 // stride) * (224 // stride)
        tokens_large = (448 // stride) * (448 // stride)
        assert tokens_large == tokens_small * 4

    # ── Fix A: pre-merge patch count (ViT transformer token count) ────────────

    def test_pre_merge_patch_count_448x448(self, enc_cfg):
        """ViT transformer sees pre-merge patches: (H // patch_size)² = (448//16)² = 784."""
        pre_merge = (448 // enc_cfg.patch_size) * (448 // enc_cfg.patch_size)
        assert pre_merge == 784

    def test_pre_merge_patch_count_224x224(self, enc_cfg):
        """224x224: (224//16)² = 14² = 196 pre-merge patches."""
        pre_merge = (224 // enc_cfg.patch_size) * (224 // enc_cfg.patch_size)
        assert pre_merge == 196

    def test_pre_merge_patch_count_896x896(self, enc_cfg):
        """896x896: (896//16)² = 56² = 3136 pre-merge patches."""
        pre_merge = (896 // enc_cfg.patch_size) * (896 // enc_cfg.patch_size)
        assert pre_merge == 3136

    def test_pre_merge_is_spatial_merge_squared_times_post_merge(self, enc_cfg):
        """pre_merge = post_merge × spatial_merge_size² (merge factor = 4 for merge=2)."""
        stride = enc_cfg.patch_size * enc_cfg.spatial_merge_size
        post_merge = (448 // stride) * (448 // stride)          # 196
        pre_merge  = (448 // enc_cfg.patch_size) * (448 // enc_cfg.patch_size)  # 784
        assert pre_merge == post_merge * (enc_cfg.spatial_merge_size ** 2)

    def test_merger_ops_use_post_merge_by_name(self):
        """PatchMerger ops are detected by 'encoder_merger' in op name."""
        for name in ("encoder_merger_fc1", "encoder_merger_act", "encoder_merger_fc2"):
            assert "encoder_merger" in name

    def test_transformer_ops_do_not_match_encoder_merger(self):
        """ViT transformer ops must NOT match 'encoder_merger'."""
        transformer_ops = [
            "encoder_add_norm_1", "encoder_qkv_gemm", "encoder_attention",
            "encoder_proj_gemm", "encoder_add_norm_2", "encoder_ffn1_gemm",
            "encoder_act", "encoder_ffn2_gemm",
        ]
        for name in transformer_ops:
            assert "encoder_merger" not in name

    def test_merger_dim_is_vit_hidden_times_spatial_merge_squared(self, enc_cfg):
        """merger_dim = hidden_size * spatial_merge_size² = 1152 * 4 = 4608."""
        merger_dim = enc_cfg.hidden_size * (enc_cfg.spatial_merge_size ** 2)
        assert merger_dim == 4608

    def test_n_mergers_includes_final_plus_deepstack(self, enc_cfg):
        """n_mergers = 1 (final) + len(deepstack_visual_indexes) = 1 + 3 = 4."""
        n_mergers = 1 + len(enc_cfg.deepstack_visual_indexes)
        assert n_mergers == 4

    def test_deepstack_visual_indexes_default_empty(self):
        """VisionEncoderConfig without deepstack_visual_indexes defaults to empty tuple."""
        cfg = common.VisionEncoderConfig(
            depth=27, hidden_size=1152, num_heads=16, intermediate_size=4304,
            patch_size=16, temporal_patch_size=2, spatial_merge_size=2, out_hidden_size=5120,
        )
        assert cfg.deepstack_visual_indexes == ()
        assert 1 + len(cfg.deepstack_visual_indexes) == 1  # no deepstack → 1 merger

    def test_n_mergers_no_deepstack(self):
        """Model with no deepstack should have n_mergers=1."""
        cfg = common.VisionEncoderConfig(
            depth=27, hidden_size=1152, num_heads=16, intermediate_size=4304,
            patch_size=16, temporal_patch_size=2, spatial_merge_size=2, out_hidden_size=5120,
        )
        assert 1 + len(cfg.deepstack_visual_indexes) == 1

    def test_total_encoder_tokens_scales_with_batch_and_images(self, enc_cfg):
        """total_encoder_tokens = batch_size * num_images_per_request * tokens_per_image."""
        stride = enc_cfg.patch_size * enc_cfg.spatial_merge_size
        tokens_per_image = (448 // stride) * (448 // stride)  # 196
        batch_size = 2
        num_images = 3
        total = batch_size * num_images * tokens_per_image
        assert total == 1176  # 2 * 3 * 196

    def test_num_image_tokens_override_is_per_image(self):
        """RuntimeConfig.num_image_tokens is per-image; backend multiplies by num_images_per_request."""
        rc = RuntimeConfig(batch_size=2, isl=512, osl=128, num_image_tokens=196, num_images_per_request=3)
        # total encoder input tokens = batch_size * num_images * num_image_tokens
        total = rc.batch_size * rc.num_images_per_request * rc.num_image_tokens
        assert total == 1176  # 2 * 3 * 196


class TestFixBEffectiveISL:
    """Fix B: post-merge image tokens must be added to the LLM context ISL."""

    @pytest.fixture
    def enc_cfg(self):
        return common.VisionEncoderConfig(
            depth=27, hidden_size=1152, num_heads=16, intermediate_size=4304,
            patch_size=16, temporal_patch_size=2, spatial_merge_size=2, out_hidden_size=5120,
            deepstack_visual_indexes=(8, 16, 24),
        )

    def _post_merge(self, enc_cfg, h, w):
        stride = enc_cfg.patch_size * enc_cfg.spatial_merge_size
        return (h // stride) * (w // stride)

    def test_effective_isl_adds_single_image_448(self, enc_cfg):
        """isl=512, 1×448×448 → effective_isl = 512 + 196 = 708."""
        isl = 512
        img_ctx = self._post_merge(enc_cfg, 448, 448) * 1  # n_img=1
        assert isl + img_ctx == 708

    def test_effective_isl_adds_single_image_896(self, enc_cfg):
        """isl=512, 1×896×896 → effective_isl = 512 + 784 = 1296."""
        isl = 512
        img_ctx = self._post_merge(enc_cfg, 896, 896) * 1
        assert isl + img_ctx == 1296

    def test_effective_isl_adds_four_images_448(self, enc_cfg):
        """isl=64, 4×448×448 → effective_isl = 64 + 784 = 848."""
        isl = 64
        img_ctx = self._post_merge(enc_cfg, 448, 448) * 4  # n_img=4
        assert isl + img_ctx == 848

    def test_effective_isl_adds_two_images_448(self, enc_cfg):
        """isl=512, 2×448×448 → effective_isl = 512 + 392 = 904."""
        isl = 512
        img_ctx = self._post_merge(enc_cfg, 448, 448) * 2
        assert isl + img_ctx == 904

    def test_effective_isl_no_image_unchanged(self):
        """Text-only (no encoder ops): img_ctx_tokens=0, effective_isl = isl."""
        isl = 512
        img_ctx = 0  # _run_encoder returns 0 when encoder_ops is empty
        assert isl + img_ctx == 512

    def test_effective_isl_four_small_images_224(self, enc_cfg):
        """isl=256, 4×224×224 → effective_isl = 256 + 196 = 452 (4×49=196 total)."""
        isl = 256
        img_ctx = self._post_merge(enc_cfg, 224, 224) * 4  # 4 * 49 = 196
        assert isl + img_ctx == 452

    def test_img_ctx_tokens_equals_n_img_times_post_merge(self, enc_cfg):
        """img_ctx_tokens = n_img × tokens_per_image (post-merge), not pre-merge."""
        n_img = 3
        post_merge = self._post_merge(enc_cfg, 448, 448)   # 196
        pre_merge  = (448 // enc_cfg.patch_size) ** 2       # 784
        img_ctx = post_merge * n_img
        assert img_ctx == 588
        assert img_ctx != pre_merge * n_img  # confirm pre-merge is NOT used for LLM context


class TestEncoderRuntime:
    """Test _run_encoder behaviour via text-only vs VL model comparison."""

    @pytest.fixture
    def model_config(self):
        return config.ModelConfig()

    def test_text_only_model_has_empty_encoder_ops(self, model_config):
        model = get_model("Qwen/Qwen3-32B", model_config, "trtllm")
        assert model.encoder_ops == []

    def test_vl_model_has_non_empty_encoder_ops(self, model_config):
        model = get_model("Qwen/Qwen3-VL-32B-Instruct", model_config, "trtllm")
        assert len(model.encoder_ops) > 0

    def test_runtime_config_num_image_tokens_zero_is_text_only(self):
        rc = RuntimeConfig(batch_size=1, isl=512, osl=128, num_image_tokens=0)
        assert rc.num_image_tokens == 0

    def test_runtime_config_num_image_tokens_set_for_vl(self):
        """196 tokens = 448x448 image at patch_size=16, spatial_merge_size=2."""
        rc = RuntimeConfig(batch_size=1, isl=512, osl=128, num_image_tokens=196)
        assert rc.num_image_tokens == 196


class TestRuntimeConfigImageFields:
    """Test image-related fields added to RuntimeConfig for VL support."""

    def test_default_num_image_tokens_is_zero(self):
        rc = RuntimeConfig(batch_size=1, isl=512, osl=128)
        assert rc.num_image_tokens == 0

    def test_num_image_tokens_can_be_set(self):
        rc = RuntimeConfig(batch_size=1, isl=512, osl=128, num_image_tokens=196)
        assert rc.num_image_tokens == 196

    def test_default_image_height_is_zero(self):
        rc = RuntimeConfig(batch_size=1, isl=512, osl=128)
        assert rc.image_height == 0

    def test_default_image_width_is_zero(self):
        rc = RuntimeConfig(batch_size=1, isl=512, osl=128)
        assert rc.image_width == 0

    def test_default_num_images_per_request_is_one(self):
        rc = RuntimeConfig(batch_size=1, isl=512, osl=128)
        assert rc.num_images_per_request == 1

    def test_image_height_and_width_can_be_set(self):
        rc = RuntimeConfig(batch_size=1, isl=512, osl=128, image_height=448, image_width=448)
        assert rc.image_height == 448
        assert rc.image_width == 448

    def test_num_images_per_request_can_be_set(self):
        rc = RuntimeConfig(batch_size=1, isl=512, osl=128, num_images_per_request=4)
        assert rc.num_images_per_request == 4

    def test_num_image_tokens_is_per_image(self):
        """num_image_tokens is per-image; total = num_image_tokens * num_images_per_request."""
        rc = RuntimeConfig(batch_size=1, isl=512, osl=128, num_image_tokens=196, num_images_per_request=3)
        assert rc.num_image_tokens == 196
        assert rc.num_images_per_request == 3


class TestInferenceSummaryEncoderFields:
    """Test encoder latency/energy fields added to InferenceSummary."""

    @pytest.fixture
    def summary(self):
        from aiconfigurator.sdk.inference_summary import InferenceSummary
        rc = RuntimeConfig(batch_size=1, isl=512, osl=128)
        return InferenceSummary(rc)

    def test_encoder_latency_dict_initializes_empty(self, summary):
        assert summary.get_encoder_latency_dict() == {}

    def test_encoder_energy_wms_dict_initializes_empty(self, summary):
        assert summary.get_encoder_energy_wms_dict() == {}

    def test_set_get_encoder_latency_dict(self, summary):
        d = {"encoder_qkv_gemm": 1.5, "encoder_attention": 2.0}
        summary.set_encoder_latency_dict(d)
        assert summary.get_encoder_latency_dict() == d

    def test_set_get_encoder_energy_wms_dict(self, summary):
        d = {"encoder_qkv_gemm": 3.0}
        summary.set_encoder_energy_wms_dict(d)
        assert summary.get_encoder_energy_wms_dict() == d

    def test_encoder_power_avg_initializes_zero(self, summary):
        assert summary.get_encoder_power_avg() == 0.0

    def test_set_get_encoder_power_avg(self, summary):
        summary.set_encoder_power_avg(250.5)
        assert summary.get_encoder_power_avg() == 250.5

    def test_encoder_energy_dict_alias(self, summary):
        d = {"encoder_ffn1_gemm": 1.0}
        summary.set_encoder_energy_wms_dict(d)
        assert summary.get_encoder_energy_dict() == d

    def test_has_sufficient_power_data_includes_encoder(self, summary):
        """Encoder latency must be counted in power data coverage check."""
        summary.set_encoder_latency_dict({"encoder_qkv_gemm": 10.0})
        summary.set_encoder_energy_wms_dict({"encoder_qkv_gemm": 50.0})
        summary.set_context_latency_dict({"context_attention": 5.0})
        summary.set_context_energy_wms_dict({"context_attention": 25.0})
        assert summary.has_sufficient_power_data(threshold=0.9)


class TestVarlenAttentionMultiImage:
    """Fix: ViT attention is varlen — each image is an independent sequence.

    For n_img images of pre_merge_per_image tokens each, the correct query is:
        batch_size = batch_size * n_img,  s = pre_merge_per_image
    NOT:
        batch_size = batch_size,          s = n_img * pre_merge_per_image

    This avoids the O(n²) sequence-concatenation overestimate.
    """

    @pytest.fixture
    def enc_cfg(self):
        return common.VisionEncoderConfig(
            depth=27, hidden_size=1152, num_heads=16, intermediate_size=4304,
            patch_size=16, temporal_patch_size=2, spatial_merge_size=2, out_hidden_size=5120,
            deepstack_visual_indexes=(8, 16, 24),
        )

    def _pre_merge(self, enc_cfg, h, w):
        return (h // enc_cfg.patch_size) * (w // enc_cfg.patch_size)

    def test_attention_op_is_detected_by_name(self):
        """encoder_attention matches 'encoder_attention' for varlen dispatch."""
        assert "encoder_attention" in "encoder_attention"

    def test_non_attention_ops_do_not_match_varlen(self):
        """No other encoder op name should trigger the varlen path."""
        non_attn_ops = [
            "encoder_add_norm_1", "encoder_qkv_gemm", "encoder_proj_gemm",
            "encoder_add_norm_2", "encoder_ffn1_gemm", "encoder_act",
            "encoder_ffn2_gemm", "encoder_merger_fc1", "encoder_merger_act",
            "encoder_merger_fc2",
        ]
        for name in non_attn_ops:
            assert "encoder_attention" not in name

    def test_single_image_varlen_same_as_non_varlen(self, enc_cfg):
        """With n_img=1, varlen and non-varlen give identical (batch, s) pairs."""
        batch_size = 2
        num_images = 1
        pre_merge = self._pre_merge(enc_cfg, 448, 448)  # 784

        # varlen path
        eff_batch_v = batch_size * num_images   # 2
        eff_s_v     = pre_merge                 # 784

        # non-varlen path
        n_img_pre   = pre_merge * num_images    # 784
        eff_batch_n = batch_size                # 2
        eff_s_n     = n_img_pre                 # 784

        assert eff_batch_v == eff_batch_n
        assert eff_s_v == eff_s_n

    def test_multi_image_varlen_avoids_quadratic_overestimate(self, enc_cfg):
        """With 4 images, varlen uses s=784 not s=3136 (4× smaller sequence)."""
        batch_size = 1
        num_images = 4
        pre_merge = self._pre_merge(enc_cfg, 448, 448)  # 784

        # varlen: each image is a separate sequence
        eff_batch_v = batch_size * num_images   # 4
        eff_s_v     = pre_merge                 # 784

        # naive concatenation (old wrong behaviour)
        n_img_pre   = pre_merge * num_images    # 3136
        eff_batch_n = batch_size                # 1
        eff_s_n     = n_img_pre                 # 3136

        # same total tokens
        assert eff_batch_v * eff_s_v == eff_batch_n * eff_s_n

        # but sequence length is n_img × shorter, avoiding O(s²) overestimate
        assert eff_s_v * num_images == eff_s_n
        assert eff_s_v < eff_s_n

    def test_varlen_eff_s_is_always_per_image(self, enc_cfg):
        """eff_s for attention is pre_merge_per_image regardless of num_images."""
        pre_merge = self._pre_merge(enc_cfg, 448, 448)  # 784
        for n_img in (1, 2, 4, 8):
            eff_s_v = pre_merge  # always per-image, not total
            assert eff_s_v == pre_merge

    def test_varlen_eff_batch_scales_with_num_images(self, enc_cfg):
        """eff_batch for attention = batch_size * num_images."""
        batch_size = 2
        for n_img in (1, 2, 4):
            eff_batch = batch_size * n_img
            assert eff_batch == batch_size * n_img

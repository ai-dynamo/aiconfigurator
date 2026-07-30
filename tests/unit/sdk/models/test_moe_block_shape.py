# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MoEBlockShape and the derived MoE fields in ``_get_model_info``.

The expectations are pinned against the shipped HF ``config.json`` fixtures in
``aiconfigurator_core/model_configs`` (loaded by model path, same as the other
model-config tests). DeepSeek-R1 is the parity oracle: 61 hidden layers with
``first_k_dense_replace=3`` and ``n_shared_experts=1`` must derive to 58 MoE
layers and 1 shared expert.
"""

import dataclasses

import pytest

from aiconfigurator.sdk.models.blocks import MoEBlockShape
from aiconfigurator.sdk.models.helpers import _get_model_info

pytestmark = pytest.mark.unit


class TestGetModelInfoDerivedMoeFields:
    """``_get_model_info`` derives num_shared_experts / num_moe_layers per checkpoint."""

    @pytest.mark.parametrize(
        "hf_id,expected_shared,expected_moe_layers",
        [
            # 61 hidden layers, first_k_dense_replace=3, n_shared_experts=1
            ("deepseek-ai/DeepSeek-R1", 1, 58),
            # no shared-expert / dense-first keys: every layer is MoE
            ("Qwen/Qwen3-235B-A22B", 0, 94),
            # 61 hidden layers, first_k_dense_replace=1, n_shared_experts=1
            ("moonshotai/Kimi-K2-Instruct", 1, 60),
            # classic MoE, no shared experts, MoE on all 32 layers
            ("mistralai/Mixtral-8x7B-v0.1", 0, 32),
            # no shared experts, MoE on all 36 layers
            ("openai/gpt-oss-120b", 0, 36),
            # dense model: both derived fields are 0
            ("meta-llama/Meta-Llama-3.1-8B", 0, 0),
            # interleave_moe_layer_step=2 (nested text_config): 24 of 48 layers
            ("meta-llama/Llama-4-Maverick-17B-128E-Instruct", 0, 24),
            # n_shared_experts is an explicit null; per-layer moe_layer_freq list sums to 47
            ("XiaomiMiMo/MiMo-V2-Flash", 0, 47),
            # shared_expert_intermediate_size == moe_intermediate_size (nested): 1 shared expert
            ("Qwen/Qwen3.5-35B-A3B", 1, 40),
            # NemotronH hybrid pattern: 23 'E' (MoE) layers of 52; n_shared_experts=1
            ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", 1, 23),
        ],
    )
    def test_derived_fields_match_fixture_configs(self, hf_id, expected_shared, expected_moe_layers):
        info = _get_model_info(hf_id)
        assert info["num_shared_experts"] == expected_shared
        assert info["num_moe_layers"] == expected_moe_layers

    def test_derived_fields_are_ints(self):
        info = _get_model_info("deepseek-ai/DeepSeek-R1")
        assert isinstance(info["num_shared_experts"], int)
        assert isinstance(info["num_moe_layers"], int)


class TestMoEBlockShape:
    """MoEBlockShape.from_model_info on the real fixture configs."""

    def test_deepseek_r1_shape(self):
        shape = MoEBlockShape.from_model_info(_get_model_info("deepseek-ai/DeepSeek-R1"))
        assert shape == MoEBlockShape(
            hidden_size=7168,
            moe_inter_size=2048,
            topk=8,
            num_experts=256,
            num_shared_experts=1,
            num_moe_layers=58,
            is_gated=True,
        )

    def test_qwen3_235b_shape(self):
        shape = MoEBlockShape.from_model_info(_get_model_info("Qwen/Qwen3-235B-A22B"))
        assert shape == MoEBlockShape(
            hidden_size=4096,
            moe_inter_size=1536,
            topk=8,
            num_experts=128,
            num_shared_experts=0,
            num_moe_layers=94,
            is_gated=True,
        )

    def test_dense_model_raises_value_error(self):
        with pytest.raises(ValueError, match="not a MoE model"):
            MoEBlockShape.from_model_info(_get_model_info("meta-llama/Meta-Llama-3.1-8B"))

    def test_frozen_dataclass(self):
        shape = MoEBlockShape.from_model_info(_get_model_info("deepseek-ai/DeepSeek-R1"))
        with pytest.raises(dataclasses.FrozenInstanceError):
            shape.topk = 4

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TrtLLMWideEPMoE operation."""

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import TrtLLMWideEPMoE

pytestmark = pytest.mark.unit


class TestTrtLLMWideEPMoE:
    """Test cases for TrtLLMWideEPMoE class."""

    def test_initialization_with_default_num_slots(self):
        """Test TrtLLMWideEPMoE initialization with default num_slots."""
        moe = TrtLLMWideEPMoE(
            name="test_wideep_moe",
            scale_factor=2.0,
            hidden_size=2048,
            inter_size=8192,
            topk=2,
            num_experts=8,
            moe_tp_size=2,
            moe_ep_size=2,
            quant_mode=common.MoEQuantMode.bfloat16,
            workload_distribution="power_law_1.01_eplb",
            attention_dp_size=1,
        )

        assert moe._name == "test_wideep_moe"
        assert moe._scale_factor == 2.0
        assert moe._hidden_size == 2048
        assert moe._inter_size == 8192
        assert moe._topk == 2
        assert moe._num_experts == 8
        assert moe._num_slots == 8  # Should default to num_experts
        assert moe._moe_tp_size == 2
        assert moe._moe_ep_size == 2
        assert moe._is_gated  # Default value

    def test_initialization_with_custom_num_slots(self):
        """Test TrtLLMWideEPMoE initialization with custom num_slots."""
        moe = TrtLLMWideEPMoE(
            name="test_wideep_moe",
            scale_factor=1.0,
            hidden_size=2048,
            inter_size=8192,
            topk=2,
            num_experts=8,
            num_slots=16,  # Custom num_slots > num_experts
            moe_tp_size=1,
            moe_ep_size=1,
            quant_mode=common.MoEQuantMode.nvfp4,
            workload_distribution="power_law_1.01_eplb",
            attention_dp_size=2,
            is_gated=False,
        )

        assert moe._num_slots == 16
        assert not moe._is_gated
        assert moe._attention_dp_size == 2

    def test_weight_calculation_gated(self):
        """Test weight calculation for gated MoE."""
        moe = TrtLLMWideEPMoE(
            name="test_moe",
            scale_factor=1.0,
            hidden_size=1024,
            inter_size=4096,
            topk=2,
            num_experts=8,
            moe_tp_size=2,
            moe_ep_size=2,
            quant_mode=common.MoEQuantMode.bfloat16,
            workload_distribution="uniform",
            attention_dp_size=1,
            is_gated=True,
        )

        # For gated: 3 GEMMs * hidden_size * inter_size * num_experts * memory_bytes / tp / ep
        expected_weights = (1024 * 4096 * 8 * 2 * 3) // 2 // 2
        assert moe._weights == expected_weights
        assert moe.get_weights() == expected_weights  # scale_factor = 1.0

    def test_weight_calculation_non_gated(self):
        """Test weight calculation for non-gated MoE."""
        moe = TrtLLMWideEPMoE(
            name="test_moe",
            scale_factor=2.0,
            hidden_size=1024,
            inter_size=4096,
            topk=2,
            num_experts=8,
            moe_tp_size=2,
            moe_ep_size=2,
            quant_mode=common.MoEQuantMode.bfloat16,
            workload_distribution="uniform",
            attention_dp_size=1,
            is_gated=False,
        )

        # For non-gated: 2 GEMMs * hidden_size * inter_size * num_experts * memory_bytes / tp / ep
        expected_weights = (1024 * 4096 * 8 * 2 * 2) // 2 // 2
        assert moe._weights == expected_weights
        assert moe.get_weights() == expected_weights * 2.0  # scale_factor = 2.0

    def _make(self, **overrides):
        base = dict(
            name="test_moe",
            scale_factor=1.0,
            hidden_size=2048,
            inter_size=8192,
            topk=2,
            num_experts=8,
            moe_tp_size=2,
            moe_ep_size=2,
            quant_mode=common.MoEQuantMode.bfloat16,
            workload_distribution="power_law_1.01_eplb",
            attention_dp_size=1,
        )
        base.update(overrides)
        return TrtLLMWideEPMoE(**base)

    def test_query_is_retired(self):
        moe = self._make(moe_tp_size=1)
        with pytest.raises(NotImplementedError, match="ModeledEPMoE"):
            moe.query(object(), x=16)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

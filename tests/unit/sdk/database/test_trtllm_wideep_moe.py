# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TrtLLMWideEPMoE operation."""

from unittest.mock import MagicMock, patch

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import PerformanceResult, TrtLLMWideEPMoE

pytestmark = pytest.mark.unit


class TestTrtLLMWideEPMoE:
    """Test cases for TrtLLMWideEPMoE class."""

    @pytest.fixture
    def mock_database(self):
        """Create a mock database for testing."""
        mock_db = MagicMock()
        mock_db.backend = "trtllm"
        # Mock query_wideep_moe_compute to return a PerformanceResult-like object
        mock_result = MagicMock()
        mock_result.__float__ = MagicMock(return_value=10.5)
        mock_result.energy = 2.5
        mock_db.query_wideep_moe_compute.return_value = mock_result
        return mock_db

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
            quant_mode=common.MoEQuantMode.float16,
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
            quant_mode=common.MoEQuantMode.float16,
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
            quant_mode=common.MoEQuantMode.float16,
            workload_distribution="uniform",
            attention_dp_size=1,
            is_gated=False,
        )

        # For non-gated: 2 GEMMs * hidden_size * inter_size * num_experts * memory_bytes / tp / ep
        expected_weights = (1024 * 4096 * 8 * 2 * 2) // 2 // 2
        assert moe._weights == expected_weights
        assert moe.get_weights() == expected_weights * 2.0  # scale_factor = 2.0

    def test_query_basic(self, mock_database):
        """Test basic query functionality."""
        moe = TrtLLMWideEPMoE(
            name="test_moe",
            scale_factor=1.0,
            hidden_size=2048,
            inter_size=8192,
            topk=2,
            num_experts=8,
            moe_tp_size=2,
            moe_ep_size=2,
            quant_mode=common.MoEQuantMode.float16,
            workload_distribution="power_law_1.01_eplb",
            attention_dp_size=1,
        )

        result = moe.query(mock_database, x=16)

        # Verify database was called correctly
        mock_database.query_wideep_moe_compute.assert_called_once_with(
            num_tokens=16,  # x * attention_dp_size = 16 * 1
            hidden_size=2048,
            inter_size=8192,
            topk=2,
            num_experts=8,
            num_slots=8,  # defaults to num_experts
            moe_tp_size=2,
            moe_ep_size=2,
            quant_mode=common.MoEQuantMode.float16,
            workload_distribution="power_law_1.01_eplb",
        )

        # Verify result
        assert isinstance(result, PerformanceResult)
        assert float(result) == 10.5  # PerformanceResult IS the latency value
        assert result.energy == 2.5  # mock energy value

    def test_query_with_attention_dp_scaling(self, mock_database):
        """Test query with attention_dp_size scaling."""
        moe = TrtLLMWideEPMoE(
            name="test_moe",
            scale_factor=1.0,
            hidden_size=2048,
            inter_size=8192,
            topk=2,
            num_experts=8,
            moe_tp_size=1,
            moe_ep_size=1,
            quant_mode=common.MoEQuantMode.float16,
            workload_distribution="uniform",
            attention_dp_size=4,  # This should scale the input tokens
        )

        moe.query(mock_database, x=16)

        # Verify tokens were scaled by attention_dp_size
        mock_database.query_wideep_moe_compute.assert_called_once()
        call_args = mock_database.query_wideep_moe_compute.call_args[1]
        assert call_args["num_tokens"] == 64  # 16 * 4

    def test_query_with_scale_factor(self, mock_database):
        """Test query with scale_factor applied to results."""
        moe = TrtLLMWideEPMoE(
            name="test_moe",
            scale_factor=3.0,
            hidden_size=2048,
            inter_size=8192,
            topk=2,
            num_experts=8,
            moe_tp_size=1,
            moe_ep_size=1,
            quant_mode=common.MoEQuantMode.float16,
            workload_distribution="uniform",
            attention_dp_size=1,
        )

        result = moe.query(mock_database, x=16)

        # Verify scale_factor was applied
        assert float(result) == 31.5  # 10.5 * 3.0 (PerformanceResult IS the latency)
        assert result.energy == 7.5  # 2.5 * 3.0

    def test_query_with_quant_mode_override(self, mock_database):
        """Test query with quantization mode override."""
        moe = TrtLLMWideEPMoE(
            name="test_moe",
            scale_factor=1.0,
            hidden_size=2048,
            inter_size=8192,
            topk=2,
            num_experts=8,
            moe_tp_size=1,
            moe_ep_size=1,
            quant_mode=common.MoEQuantMode.float16,  # Original mode
            workload_distribution="uniform",
            attention_dp_size=1,
        )

        # Override quant_mode in query
        moe.query(mock_database, x=16, quant_mode=common.MoEQuantMode.nvfp4)

        # Verify override was used
        call_args = mock_database.query_wideep_moe_compute.call_args[1]
        assert call_args["quant_mode"] == common.MoEQuantMode.nvfp4

    def test_query_with_custom_num_slots(self, mock_database):
        """Test query with custom num_slots for EPLB."""
        moe = TrtLLMWideEPMoE(
            name="test_moe",
            scale_factor=1.0,
            hidden_size=2048,
            inter_size=8192,
            topk=2,
            num_experts=8,
            num_slots=12,  # Custom slots for EPLB
            moe_tp_size=1,
            moe_ep_size=1,
            quant_mode=common.MoEQuantMode.float16,
            workload_distribution="power_law_1.2_eplb",
            attention_dp_size=1,
        )

        moe.query(mock_database, x=16)

        # Verify custom num_slots was used
        call_args = mock_database.query_wideep_moe_compute.call_args[1]
        assert call_args["num_slots"] == 12

    @patch("aiconfigurator.sdk.operations.logger")
    def test_query_debug_logging(self, mock_logger, mock_database):
        """Test that debug logging is called during query."""
        moe = TrtLLMWideEPMoE(
            name="test_moe",
            scale_factor=1.0,
            hidden_size=2048,
            inter_size=8192,
            topk=2,
            num_experts=8,
            num_slots=16,
            moe_tp_size=1,
            moe_ep_size=1,
            quant_mode=common.MoEQuantMode.float16,
            workload_distribution="uniform",
            attention_dp_size=1,
        )

        moe.query(mock_database, x=16)

        # Verify debug logging was called
        mock_logger.debug.assert_called_with("TrtLLMWideEPMoE: Querying compute with num_slots=16")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

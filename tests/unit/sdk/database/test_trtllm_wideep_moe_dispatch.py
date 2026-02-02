# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TrtLLMWideEPMoEDispatch operation."""

import pytest
from unittest.mock import MagicMock, patch

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import TrtLLMWideEPMoEDispatch, PerformanceResult

pytestmark = pytest.mark.unit


class TestTrtLLMWideEPMoEDispatch:
    """Test cases for TrtLLMWideEPMoEDispatch class."""
    
    @pytest.fixture
    def mock_database(self):
        """Create a mock database for testing."""
        mock_db = MagicMock()
        mock_db.backend = "trtllm"
        mock_db.system_spec = {
            "gpu": {"sm_version": 100},
            "node": {"num_gpus_per_node": 8}
        }
        # Mock query_wideep_alltoall to return different values for different ops
        def mock_alltoall(op_name, **kwargs):
            mock_result = MagicMock()
            if op_name == "alltoall_prepare":
                mock_result.__float__ = MagicMock(return_value=5.0)
            elif op_name == "alltoall_dispatch":
                mock_result.__float__ = MagicMock(return_value=7.0)
            elif op_name == "alltoall_combine":
                mock_result.__float__ = MagicMock(return_value=6.0)
            elif op_name == "alltoall_combine_low_precision":
                mock_result.__float__ = MagicMock(return_value=4.5)
            return mock_result
        
        mock_db.query_wideep_alltoall.side_effect = mock_alltoall
        return mock_db
    
    def test_initialization_pre_dispatch(self):
        """Test initialization for pre-dispatch phase."""
        dispatch = TrtLLMWideEPMoEDispatch(
            name="test_pre_dispatch",
            scale_factor=1.0,
            hidden_size=2048,
            topk=2,
            num_experts=8,
            moe_tp_size=2,
            moe_ep_size=4,
            attention_dp_size=1,
            pre_dispatch=True,
            quant_mode=common.MoEQuantMode.float16,
        )
        
        assert dispatch._name == "test_pre_dispatch"
        assert dispatch._hidden_size == 2048
        assert dispatch._topk == 2
        assert dispatch._num_experts == 8
        assert dispatch._moe_tp_size == 2
        assert dispatch._moe_ep_size == 4
        assert dispatch._pre_dispatch == True
        assert dispatch._quant_mode == common.MoEQuantMode.float16
        assert dispatch._use_low_precision_combine == False  # Default
        assert dispatch._node_num is None  # Default
        assert dispatch.num_gpus == 8  # 2 * 4
        assert dispatch._weights == 0.0
    
    def test_initialization_post_dispatch_with_options(self):
        """Test initialization for post-dispatch phase with custom options."""
        dispatch = TrtLLMWideEPMoEDispatch(
            name="test_post_dispatch",
            scale_factor=2.0,
            hidden_size=1024,
            topk=4,
            num_experts=16,
            moe_tp_size=1,
            moe_ep_size=8,
            attention_dp_size=2,
            pre_dispatch=False,
            quant_mode=common.MoEQuantMode.nvfp4,
            use_low_precision_combine=True,
            node_num=2,
        )
        
        assert dispatch._pre_dispatch == False
        assert dispatch._use_low_precision_combine == True
        assert dispatch._node_num == 2
        assert dispatch._scale_factor == 2.0
        assert dispatch.num_gpus == 8  # 1 * 8
    
    def test_get_weights(self):
        """Test that get_weights returns 0 for dispatch operations."""
        dispatch = TrtLLMWideEPMoEDispatch(
            name="test_dispatch",
            scale_factor=10.0,
            hidden_size=2048,
            topk=2,
            num_experts=8,
            moe_tp_size=1,
            moe_ep_size=1,
            attention_dp_size=1,
            pre_dispatch=True,
            quant_mode=common.MoEQuantMode.float16,
        )
        
        assert dispatch.get_weights() == 0.0
    
    def test_query_pre_dispatch(self, mock_database):
        """Test query for pre-dispatch phase (prepare + dispatch)."""
        dispatch = TrtLLMWideEPMoEDispatch(
            name="test_dispatch",
            scale_factor=1.0,
            hidden_size=2048,
            topk=2,
            num_experts=8,
            moe_tp_size=2,
            moe_ep_size=4,
            attention_dp_size=1,
            pre_dispatch=True,
            quant_mode=common.MoEQuantMode.float16,
        )
        
        result = dispatch.query(mock_database, x=16)
        
        # Verify both prepare and dispatch were called
        assert mock_database.query_wideep_alltoall.call_count == 2
        
        # Check prepare call
        prepare_call = mock_database.query_wideep_alltoall.call_args_list[0]
        assert prepare_call[1]["op_name"] == "alltoall_prepare"
        assert prepare_call[1]["num_tokens"] == 16
        assert prepare_call[1]["hidden_size"] == 2048
        assert prepare_call[1]["topk"] == 2
        assert prepare_call[1]["num_experts"] == 8
        assert prepare_call[1]["moe_ep_size"] == 4
        assert prepare_call[1]["quant_mode"] == common.MoEQuantMode.float16
        assert prepare_call[1]["node_num"] is None
        
        # Check dispatch call
        dispatch_call = mock_database.query_wideep_alltoall.call_args_list[1]
        assert dispatch_call[1]["op_name"] == "alltoall_dispatch"
        
        # Verify result is sum of prepare + dispatch
        assert isinstance(result, PerformanceResult)
        assert float(result) == 12.0  # 5.0 + 7.0
        assert result.energy == 0.0  # No energy for comm ops
    
    def test_query_post_dispatch_standard(self, mock_database):
        """Test query for post-dispatch phase with standard combine."""
        dispatch = TrtLLMWideEPMoEDispatch(
            name="test_dispatch",
            scale_factor=1.0,
            hidden_size=2048,
            topk=2,
            num_experts=8,
            moe_tp_size=1,
            moe_ep_size=4,
            attention_dp_size=1,
            pre_dispatch=False,
            quant_mode=common.MoEQuantMode.float16,
            use_low_precision_combine=False,
        )
        
        result = dispatch.query(mock_database, x=16)
        
        # Verify only combine was called
        assert mock_database.query_wideep_alltoall.call_count == 1
        
        combine_call = mock_database.query_wideep_alltoall.call_args_list[0]
        assert combine_call[1]["op_name"] == "alltoall_combine"
        assert combine_call[1]["num_tokens"] == 16
        
        assert float(result) == 6.0
        assert result.energy == 0.0
    
    def test_query_post_dispatch_low_precision(self, mock_database):
        """Test query for post-dispatch phase with low precision combine."""
        dispatch = TrtLLMWideEPMoEDispatch(
            name="test_dispatch",
            scale_factor=1.0,
            hidden_size=2048,
            topk=2,
            num_experts=8,
            moe_tp_size=1,
            moe_ep_size=4,
            attention_dp_size=1,
            pre_dispatch=False,
            quant_mode=common.MoEQuantMode.nvfp4,
            use_low_precision_combine=True,
        )
        
        result = dispatch.query(mock_database, x=16)
        
        # Verify low precision combine was called
        combine_call = mock_database.query_wideep_alltoall.call_args_list[0]
        assert combine_call[1]["op_name"] == "alltoall_combine_low_precision"
        
        assert float(result) == 4.5
        assert result.energy == 0.0
    
    def test_query_with_scale_factor(self, mock_database):
        """Test query with scale_factor applied."""
        dispatch = TrtLLMWideEPMoEDispatch(
            name="test_dispatch",
            scale_factor=3.0,
            hidden_size=2048,
            topk=2,
            num_experts=8,
            moe_tp_size=2,
            moe_ep_size=4,
            attention_dp_size=1,
            pre_dispatch=True,
            quant_mode=common.MoEQuantMode.float16,
        )
        
        result = dispatch.query(mock_database, x=16)
        
        # Verify scale_factor was applied to the sum
        assert float(result) == 36.0  # (5.0 + 7.0) * 3.0
        assert result.energy == 0.0
    
    def test_query_with_custom_node_num(self, mock_database):
        """Test query with custom node_num."""
        dispatch = TrtLLMWideEPMoEDispatch(
            name="test_dispatch",
            scale_factor=1.0,
            hidden_size=2048,
            topk=2,
            num_experts=8,
            moe_tp_size=1,
            moe_ep_size=8,
            attention_dp_size=1,
            pre_dispatch=True,
            quant_mode=common.MoEQuantMode.float16,
            node_num=4,  # Custom node count
        )
        
        result = dispatch.query(mock_database, x=16)
        
        # Verify node_num was passed to queries
        for call in mock_database.query_wideep_alltoall.call_args_list:
            assert call[1]["node_num"] == 4
    
    @patch('aiconfigurator.sdk.operations.logger')
    def test_query_debug_logging_pre_dispatch(self, mock_logger, mock_database):
        """Test debug logging for pre-dispatch."""
        dispatch = TrtLLMWideEPMoEDispatch(
            name="test_dispatch",
            scale_factor=1.0,
            hidden_size=2048,
            topk=2,
            num_experts=8,
            moe_tp_size=1,
            moe_ep_size=1,
            attention_dp_size=1,
            pre_dispatch=True,
            quant_mode=common.MoEQuantMode.float16,
        )
        
        result = dispatch.query(mock_database, x=16)
        
        mock_logger.debug.assert_called_with(
            "TrtLLMWideEPMoEDispatch: Pre-dispatch with standard precision"
        )
    
    @patch('aiconfigurator.sdk.operations.logger')
    def test_query_debug_logging_post_dispatch_low_precision(self, mock_logger, mock_database):
        """Test debug logging for post-dispatch with low precision."""
        dispatch = TrtLLMWideEPMoEDispatch(
            name="test_dispatch",
            scale_factor=1.0,
            hidden_size=2048,
            topk=2,
            num_experts=8,
            moe_tp_size=1,
            moe_ep_size=1,
            attention_dp_size=1,
            pre_dispatch=False,
            quant_mode=common.MoEQuantMode.nvfp4,
            use_low_precision_combine=True,
        )
        
        result = dispatch.query(mock_database, x=16)
        
        mock_logger.debug.assert_called_with(
            "TrtLLMWideEPMoEDispatch: Post-dispatch with low-precision combine"
        )
    
    def test_different_quant_modes(self, mock_database):
        """Test with different quantization modes."""
        quant_modes = [
            common.MoEQuantMode.float16,
            common.MoEQuantMode.nvfp4,
            common.MoEQuantMode.fp8_block,
        ]
        
        for quant_mode in quant_modes:
            dispatch = TrtLLMWideEPMoEDispatch(
                name="test_dispatch",
                scale_factor=1.0,
                hidden_size=2048,
                topk=2,
                num_experts=8,
                moe_tp_size=1,
                moe_ep_size=1,
                attention_dp_size=1,
                pre_dispatch=True,
                quant_mode=quant_mode,
            )
            
            # Clear previous calls
            mock_database.query_wideep_alltoall.reset_mock()
            
            result = dispatch.query(mock_database, x=16)
            
            # Verify correct quant_mode was passed
            for call in mock_database.query_wideep_alltoall.call_args_list:
                assert call[1]["quant_mode"] == quant_mode


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
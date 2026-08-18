# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TrtLLMWideEPMoEDispatch operation."""

from unittest.mock import MagicMock

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import TrtLLMWideEPMoEDispatch

pytestmark = pytest.mark.unit


class TestTrtLLMWideEPMoEDispatch:
    """Test cases for TrtLLMWideEPMoEDispatch class."""

    @pytest.fixture
    def mock_database(self):
        """Create a mock database for testing."""
        mock_db = MagicMock()
        mock_db.backend = "trtllm"
        mock_db.system_spec = {"gpu": {"sm_version": 100}, "node": {"num_gpus_per_node": 8}}

        # Mock query_trtllm_alltoall to return different values for different ops
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

        mock_db.query_trtllm_alltoall.side_effect = mock_alltoall
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
            quant_mode=common.MoEQuantMode.bfloat16,
        )

        assert dispatch._name == "test_pre_dispatch"
        assert dispatch._hidden_size == 2048
        assert dispatch._topk == 2
        assert dispatch._num_experts == 8
        assert dispatch._moe_tp_size == 2
        assert dispatch._moe_ep_size == 4
        assert dispatch._pre_dispatch
        assert dispatch._quant_mode == common.MoEQuantMode.bfloat16
        assert not dispatch._use_low_precision_combine  # Default
        assert dispatch._node_num is None  # Default
        assert dispatch.num_gpus == 8  # 2 * 4
        assert dispatch.get_weights() == 0.0

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

        assert not dispatch._pre_dispatch
        assert dispatch._use_low_precision_combine
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
            quant_mode=common.MoEQuantMode.bfloat16,
        )

        assert dispatch.get_weights() == 0.0

    def test_query_is_a_retired_tombstone(self):
        """The per-op query retired with the Python per-call query stack
        (#1357 PR-5): the per-phase alltoall rows stay loadable
        (database._trtllm_alltoall_data — see test_moe_block_builder_large_ep
        for the raw-row oracle) and live models express dispatch through the
        compiled MoEDispatch op, so this class's query has NO engine twin and
        must fail loudly rather than silently answer."""
        op = TrtLLMWideEPMoEDispatch(
            name="test_dispatch",
            scale_factor=1.0,
            hidden_size=2048,
            topk=2,
            num_experts=8,
            moe_tp_size=2,
            moe_ep_size=4,
            attention_dp_size=1,
            pre_dispatch=True,
            quant_mode=common.MoEQuantMode.bfloat16,
        )
        with pytest.raises(NotImplementedError, match="retired with the Python per-call query stack"):
            op.query(object(), x=16)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

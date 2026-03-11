# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MoEDispatch communication logic across SM versions."""

from unittest.mock import MagicMock

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import MoEDispatch, PerformanceResult

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_db(sm_version=100, num_gpus_per_node=8):
    """Create a mock database configured as trtllm backend."""
    db = MagicMock()
    db.backend = "trtllm"
    db.system_spec = {
        "gpu": {"sm_version": sm_version},
        "node": {"num_gpus_per_node": num_gpus_per_node},
    }
    db.query_trtllm_alltoall.return_value = PerformanceResult(3.0)
    db.query_nccl.return_value = PerformanceResult(2.0)
    db.query_custom_allreduce.return_value = PerformanceResult(1.5)
    return db


def _make_dispatch(
    moe_tp_size=1,
    moe_ep_size=8,
    attention_dp_size=8,
    pre_dispatch=True,
    quant_mode=None,
    moe_backend=None,
    reduce_results=True,
    hidden_size=7168,
    topk=8,
    num_experts=256,
):
    """Helper to build MoEDispatch with given parallelism config."""
    return MoEDispatch(
        name="test_dispatch",
        scale_factor=1,
        hidden_size=hidden_size,
        topk=topk,
        num_experts=num_experts,
        moe_tp_size=moe_tp_size,
        moe_ep_size=moe_ep_size,
        attention_dp_size=attention_dp_size,
        pre_dispatch=pre_dispatch,
        quant_mode=quant_mode,
        moe_backend=moe_backend,
        reduce_results=reduce_results,
    )


# ===================================================================
# SM == 100 path tests
# ===================================================================


class TestEnableAlltoallConditions:
    """Test the enable_alltoall gating logic (SM100 only)."""

    def test_alltoall_enabled_default_backend(self):
        """alltoall enabled when moe_backend=None, dp>1, moe_tp=1, quant_mode set."""
        db = _make_mock_db(sm_version=100)
        dispatch = _make_dispatch(
            moe_tp_size=1,
            moe_ep_size=8,
            attention_dp_size=8,
            pre_dispatch=True,
            quant_mode=common.MoEQuantMode.fp8,
        )
        dispatch.query(db, x=16)
        db.query_trtllm_alltoall.assert_called_once()

    def test_alltoall_enabled_cutlass_backend(self):
        """alltoall enabled when moe_backend='CUTLASS' and quant_mode set."""
        db = _make_mock_db(sm_version=100)
        dispatch = _make_dispatch(moe_backend="CUTLASS", pre_dispatch=True, quant_mode=common.MoEQuantMode.fp8)
        dispatch.query(db, x=16)
        db.query_trtllm_alltoall.assert_called_once()

    def test_alltoall_requires_quant_mode_when_enabled(self):
        """TRTLLM alltoall path fails fast when quant_mode is missing."""
        db = _make_mock_db(sm_version=100)
        dispatch = _make_dispatch(moe_tp_size=1, moe_ep_size=8, attention_dp_size=8, pre_dispatch=True, quant_mode=None)
        with pytest.raises(ValueError, match="requires quant_mode"):
            dispatch.query(db, x=16)

        db.query_trtllm_alltoall.assert_not_called()
        db.query_nccl.assert_not_called()

    def test_alltoall_disabled_deepep_backend(self):
        """alltoall disabled when moe_backend='deepep' (not in allowed set)."""
        db = _make_mock_db(sm_version=100)
        dispatch = _make_dispatch(moe_backend="deepep", pre_dispatch=True)
        dispatch.query(db, x=16)
        db.query_trtllm_alltoall.assert_not_called()

    def test_alltoall_disabled_when_dp_is_1(self):
        """alltoall disabled when attention_dp_size=1 (only this condition disables it)."""
        db = _make_mock_db(sm_version=100)
        dispatch = _make_dispatch(moe_tp_size=1, moe_ep_size=8, attention_dp_size=1, pre_dispatch=True)
        dispatch.query(db, x=16)
        db.query_trtllm_alltoall.assert_not_called()

    def test_alltoall_disabled_when_moe_tp_gt_1(self):
        """alltoall disabled when moe_tp_size>1 (only this condition disables it)."""
        db = _make_mock_db(sm_version=100)
        dispatch = _make_dispatch(moe_tp_size=2, moe_ep_size=4, attention_dp_size=8, pre_dispatch=True)
        dispatch.query(db, x=16)
        db.query_trtllm_alltoall.assert_not_called()


class TestSm100AlltoallPath:
    """Test the alltoall communication path (SM100, enable_alltoall=True)."""

    def test_pre_dispatch_calls_alltoall_dispatch(self):
        """Pre-dispatch uses alltoall_dispatch op."""
        db = _make_mock_db(sm_version=100)
        dispatch = _make_dispatch(pre_dispatch=True, quant_mode=common.MoEQuantMode.fp8)
        result = dispatch.query(db, x=16)

        db.query_trtllm_alltoall.assert_called_once()
        call_kwargs = db.query_trtllm_alltoall.call_args[1]
        assert call_kwargs["op_name"] == "alltoall_dispatch"
        assert call_kwargs["quant_mode"] == common.MoEQuantMode.fp8
        assert float(result) == 3.0

    def test_post_dispatch_calls_alltoall_combine(self):
        """Post-dispatch uses alltoall_combine op."""
        db = _make_mock_db(sm_version=100)
        dispatch = _make_dispatch(pre_dispatch=False, quant_mode=common.MoEQuantMode.fp8)
        result = dispatch.query(db, x=16)

        db.query_trtllm_alltoall.assert_called_once()
        call_kwargs = db.query_trtllm_alltoall.call_args[1]
        assert call_kwargs["op_name"] == "alltoall_combine"
        assert float(result) == 3.0

    def test_nvfp4_quant_mode_forwarded(self):
        """nvfp4 quant_mode is correctly forwarded to alltoall (not replaced by default)."""
        db = _make_mock_db(sm_version=100)
        dispatch = _make_dispatch(pre_dispatch=True, quant_mode=common.MoEQuantMode.nvfp4)
        dispatch.query(db, x=16)

        call_kwargs = db.query_trtllm_alltoall.call_args[1]
        assert call_kwargs["quant_mode"] == common.MoEQuantMode.nvfp4

    def test_moe_backend_forwarded_to_alltoall(self):
        """moe_backend is forwarded to query_trtllm_alltoall."""
        db = _make_mock_db(sm_version=100)
        dispatch = _make_dispatch(pre_dispatch=True, moe_backend="CUTLASS", quant_mode=common.MoEQuantMode.fp8)
        dispatch.query(db, x=16)

        call_kwargs = db.query_trtllm_alltoall.call_args[1]
        assert call_kwargs["moe_backend"] == "CUTLASS"


class TestSm100DpFallbackPath:
    """Test SM100 DP>1 path when alltoall is NOT enabled."""

    def test_pre_dispatch_uses_all_gather(self):
        """Pre-dispatch DP>1 without alltoall uses all_gather."""
        db = _make_mock_db(sm_version=100)
        dispatch = _make_dispatch(moe_backend="deepep", pre_dispatch=True, attention_dp_size=8)
        dispatch.query(db, x=16)

        db.query_nccl.assert_called_once()
        assert db.query_nccl.call_args[0][2] == "all_gather"

    def test_post_dispatch_uses_reduce_scatter(self):
        """Post-dispatch DP>1 without alltoall uses reduce_scatter."""
        db = _make_mock_db(sm_version=100)
        dispatch = _make_dispatch(moe_backend="deepep", pre_dispatch=False, attention_dp_size=8)
        dispatch.query(db, x=16)

        db.query_nccl.assert_called_once()
        assert db.query_nccl.call_args[0][2] == "reduce_scatter"


class TestSm100QuantAwareVolume:
    """Test quantize-aware communication volume calculation (SM100 DP fallback)."""

    def _get_all_gather_volume(self, quant_mode):
        """Helper: run pre-dispatch on DP fallback path, return the all_gather volume arg."""
        db = _make_mock_db(sm_version=100)
        dispatch = _make_dispatch(
            moe_backend="deepep",
            pre_dispatch=True,
            attention_dp_size=8,
            quant_mode=quant_mode,
            hidden_size=1024,
        )
        dispatch.query(db, x=16)
        return db.query_nccl.call_args[0][3]

    def test_nvfp4_compressed_volume(self):
        """nvfp4: volume/4 + volume/32, scaled by dp_size."""
        volume = 16 * 1024
        expected_x = volume / 4
        expected_sf = volume / 4 / 8
        expected_total = (expected_x + expected_sf) * 8
        actual = self._get_all_gather_volume(common.MoEQuantMode.nvfp4)
        assert actual == pytest.approx(expected_total)

    def test_fp8_compressed_volume(self):
        """fp8: volume/2, no scale factor, scaled by dp_size."""
        volume = 16 * 1024
        expected_total = (volume / 2 + 0) * 8
        actual = self._get_all_gather_volume(common.MoEQuantMode.fp8)
        assert actual == pytest.approx(expected_total)

    def test_bf16_full_volume(self):
        """bf16 / unknown: full volume, scaled by dp_size."""
        volume = 16 * 1024
        expected_total = volume * 8
        actual = self._get_all_gather_volume(None)
        assert actual == pytest.approx(expected_total)


class TestSm100TpFallbackPath:
    """Test SM100 TP>1 path (no alltoall, no DP)."""

    def test_pre_dispatch_tp_reduce_results_true(self):
        """TP>1 with reduce_results=True calls custom_allreduce."""
        db = _make_mock_db(sm_version=100, num_gpus_per_node=8)
        dispatch = _make_dispatch(
            moe_tp_size=8,
            moe_ep_size=1,
            attention_dp_size=1,
            pre_dispatch=True,
            reduce_results=True,
        )
        dispatch.query(db, x=16)
        db.query_custom_allreduce.assert_called_once()

    def test_pre_dispatch_tp_reduce_results_false(self):
        """TP>1 with reduce_results=False has zero comm latency."""
        db = _make_mock_db(sm_version=100)
        dispatch = _make_dispatch(
            moe_tp_size=8,
            moe_ep_size=1,
            attention_dp_size=1,
            pre_dispatch=True,
            reduce_results=False,
        )
        result = dispatch.query(db, x=16)
        db.query_custom_allreduce.assert_not_called()
        db.query_nccl.assert_not_called()
        assert float(result) == 0.0

    def test_post_dispatch_tp_reduce_results_false(self):
        """Post-dispatch TP>1 with reduce_results=False also has zero comm."""
        db = _make_mock_db(sm_version=100)
        dispatch = _make_dispatch(
            moe_tp_size=8,
            moe_ep_size=1,
            attention_dp_size=1,
            pre_dispatch=False,
            reduce_results=False,
        )
        result = dispatch.query(db, x=16)
        assert float(result) == 0.0

    def test_nvl72_uses_nccl_allreduce(self):
        """On NVL72 with >4 GPUs, TP path uses NCCL all_reduce instead of custom."""
        db = _make_mock_db(sm_version=100, num_gpus_per_node=72)
        dispatch = _make_dispatch(
            moe_tp_size=8,
            moe_ep_size=1,
            attention_dp_size=1,
            pre_dispatch=True,
            reduce_results=True,
        )
        dispatch.query(db, x=16)
        db.query_nccl.assert_called_once()
        assert db.query_nccl.call_args[0][2] == "all_reduce"
        db.query_custom_allreduce.assert_not_called()


class TestSm100NoCommPath:
    """Test zero communication when neither TP nor DP applies (SM100)."""

    def test_single_gpu_zero_comm(self):
        """tp=1, dp=1 on SM100 -> zero communication."""
        db = _make_mock_db(sm_version=100)
        dispatch = _make_dispatch(
            moe_tp_size=1,
            moe_ep_size=1,
            attention_dp_size=1,
            pre_dispatch=True,
        )
        result = dispatch.query(db, x=16)
        db.query_trtllm_alltoall.assert_not_called()
        db.query_nccl.assert_not_called()
        db.query_custom_allreduce.assert_not_called()
        assert float(result) == 0.0


# ===================================================================
# SM < 100 path tests
# ===================================================================


class TestSmLt100PreDispatch:
    """Test SM<100 pre-dispatch paths (tp > dp priority, no alltoall)."""

    def test_tp_gt1_uses_custom_allreduce(self):
        """SM<100 pre-dispatch: tp>1 -> custom_allreduce."""
        db = _make_mock_db(sm_version=90)
        dispatch = _make_dispatch(
            moe_tp_size=8,
            moe_ep_size=8,
            attention_dp_size=1,
            pre_dispatch=True,
        )
        dispatch.query(db, x=16)
        db.query_custom_allreduce.assert_called_once()
        db.query_nccl.assert_not_called()
        db.query_trtllm_alltoall.assert_not_called()

    def test_dp_gt1_uses_all_gather(self):
        """SM<100 pre-dispatch: dp>1, tp=1 -> all_gather with volume * dp_size."""
        db = _make_mock_db(sm_version=90)
        dispatch = _make_dispatch(
            moe_tp_size=1,
            moe_ep_size=8,
            attention_dp_size=8,
            pre_dispatch=True,
            hidden_size=1024,
        )
        dispatch.query(db, x=16)

        db.query_nccl.assert_called_once()
        args = db.query_nccl.call_args[0]
        assert args[2] == "all_gather"
        assert args[3] == 16 * 1024 * 8  # volume * dp_size

    def test_single_gpu_zero_comm(self):
        """SM<100 pre-dispatch: tp=1, dp=1 -> zero."""
        db = _make_mock_db(sm_version=90)
        dispatch = _make_dispatch(
            moe_tp_size=1,
            moe_ep_size=1,
            attention_dp_size=1,
            pre_dispatch=True,
        )
        result = dispatch.query(db, x=16)
        db.query_custom_allreduce.assert_not_called()
        db.query_nccl.assert_not_called()
        assert float(result) == 0.0


class TestSmLt100PostDispatch:
    """Test SM<100 post-dispatch paths."""

    def test_tp_gt1_uses_custom_allreduce(self):
        """SM<100 post-dispatch: tp>1 -> custom_allreduce."""
        db = _make_mock_db(sm_version=90)
        dispatch = _make_dispatch(
            moe_tp_size=8,
            moe_ep_size=8,
            attention_dp_size=1,
            pre_dispatch=False,
        )
        dispatch.query(db, x=16)
        db.query_custom_allreduce.assert_called_once()

    def test_dp_gt1_uses_reduce_scatter(self):
        """SM<100 post-dispatch: dp>1, tp=1 -> reduce_scatter with volume * dp_size."""
        db = _make_mock_db(sm_version=90)
        dispatch = _make_dispatch(
            moe_tp_size=1,
            moe_ep_size=8,
            attention_dp_size=8,
            pre_dispatch=False,
            hidden_size=1024,
        )
        dispatch.query(db, x=16)

        db.query_nccl.assert_called_once()
        args = db.query_nccl.call_args[0]
        assert args[2] == "reduce_scatter"
        assert args[3] == 16 * 1024 * 8  # volume * dp_size

    def test_single_gpu_zero_comm(self):
        """SM<100 post-dispatch: tp=1, dp=1 -> zero."""
        db = _make_mock_db(sm_version=90)
        dispatch = _make_dispatch(
            moe_tp_size=1,
            moe_ep_size=1,
            attention_dp_size=1,
            pre_dispatch=False,
        )
        result = dispatch.query(db, x=16)
        assert float(result) == 0.0


class TestSmLt100NoAlltoall:
    """Verify SM<100 never uses alltoall regardless of config."""

    def test_dp_config_no_alltoall(self):
        """SM<100 with dp>1, moe_tp=1 still does NOT use alltoall."""
        db = _make_mock_db(sm_version=90)
        dispatch = _make_dispatch(
            moe_tp_size=1,
            moe_ep_size=8,
            attention_dp_size=8,
            pre_dispatch=True,
        )
        dispatch.query(db, x=16)
        db.query_trtllm_alltoall.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

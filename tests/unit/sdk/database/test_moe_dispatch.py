# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MoEDispatch communication logic across SM versions."""

import logging
from unittest.mock import MagicMock, call

import pytest
import torch

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import MoEDispatch, PerformanceResult

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_db(sm_version=100, num_gpus_per_node=8, backend="trtllm"):
    """Create a mock database configured for MoEDispatch tests."""
    db = MagicMock()
    db.backend = backend
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


@pytest.mark.skipif(torch.xpu.is_available(), reason="skip for xpu")
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


@pytest.mark.skipif(torch.xpu.is_available(), reason="skip for xpu")
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


@pytest.mark.skipif(torch.xpu.is_available(), reason="skip for xpu")
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


@pytest.mark.skipif(torch.xpu.is_available(), reason="skip for xpu")
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


@pytest.mark.skipif(torch.xpu.is_available(), reason="skip for xpu")
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


@pytest.mark.skipif(torch.xpu.is_available(), reason="skip for xpu")
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


@pytest.mark.skipif(torch.xpu.is_available(), reason="skip for xpu")
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


@pytest.mark.skipif(torch.xpu.is_available(), reason="skip for xpu")
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


@pytest.mark.skipif(torch.xpu.is_available(), reason="skip for xpu")
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


@pytest.mark.skipif(torch.xpu.is_available(), reason="skip for xpu")
class TestSGLangNonDeepEPAttentionTpDp:
    """Test SGLang non-DeepEP combined attention TPxDP communication."""

    def test_pre_dispatch_uses_reduce_scatter_then_all_gather(self):
        db = _make_mock_db(sm_version=90, backend="sglang")
        db.query_nccl.side_effect = [PerformanceResult(2.0), PerformanceResult(3.0)]
        dispatch = _make_dispatch(
            moe_tp_size=4,
            moe_ep_size=2,
            attention_dp_size=2,
            pre_dispatch=True,
            hidden_size=1024,
        )

        result = dispatch.query(db, x=16)

        volume = 16 * 1024
        assert float(result) == 5.0
        assert db.query_nccl.call_count == 2
        db.query_nccl.assert_has_calls(
            [
                call(common.CommQuantMode.half, 4, "reduce_scatter", volume),
                call(common.CommQuantMode.half, 8, "all_gather", volume * 2),
            ]
        )
        db.query_custom_allreduce.assert_not_called()

    def test_post_dispatch_uses_reduce_scatter_then_all_gather(self):
        db = _make_mock_db(sm_version=90, backend="sglang")
        db.query_nccl.side_effect = [PerformanceResult(2.0), PerformanceResult(3.0)]
        dispatch = _make_dispatch(
            moe_tp_size=4,
            moe_ep_size=2,
            attention_dp_size=2,
            pre_dispatch=False,
            hidden_size=1024,
        )

        result = dispatch.query(db, x=16)

        volume = 16 * 1024
        assert float(result) == 5.0
        assert db.query_nccl.call_count == 2
        db.query_nccl.assert_has_calls(
            [
                call(common.CommQuantMode.half, 8, "reduce_scatter", volume * 2),
                call(common.CommQuantMode.half, 4, "all_gather", volume),
            ]
        )
        db.query_custom_allreduce.assert_not_called()


@pytest.mark.skipif(torch.xpu.is_available(), reason="skip for xpu")
class TestWideEpDeepEpLlNodeNumFallback:
    """node_num resolution in the WideEP DeepEP tables.

    The contract has three cases and they must stay distinguishable:

    * A measured node scale answers from its own rows, silently, tagged
      ``"silicon"``. Real multi-node data must never be downgraded.
    * An unmeasured multi-node scale (node_num > 1) still answers -- refusing
      would remove WideEP from every system whose DeepEP tables carry
      node_num=1 only -- but it answers from node_num=1 rows, warns with the
      measured error for that scale, and tags the dedicated WideEP
      communication node-1 fallback source so the CLI can mark the config row.
      What is forbidden is substituting *silently*.
    * A sub-node scale (node_num < 1, what MoEDispatch.query derives for a
      config smaller than one node) keeps the older, quieter fallback and is
      NOT an estimate: it stays intra-node in both directions.
    """

    @staticmethod
    def _leaf(latency):
        return {128: {"latency": latency, "power": 0.0, "energy": 0.0}}

    def _db_with_ll_data(self):
        db = MagicMock()
        # node_num=1 -> 100us, node_num=2 -> 200us for (hidden=7168, topk=8, experts=256)
        db._wideep_deepep_ll_data = {
            1: {7168: {8: {256: self._leaf(100.0)}}},
            2: {7168: {8: {256: self._leaf(200.0)}}},
        }
        db._default_database_mode = common.DatabaseMode.SILICON
        # _interp_pr(lat, energy=...) -> echo the latency so tests can assert which bucket was
        # used, and tag "silicon" the way the real one does so source assertions are meaningful.
        db._interp_pr = lambda lat, energy=0.0: PerformanceResult(lat, energy=energy, source="silicon")
        return db

    def _query(self, db, node_num, num_experts=256):
        return MoEDispatch._query_wideep_deepep_ll_table(
            db,
            node_num=node_num,
            num_tokens=128,
            num_experts=num_experts,
            topk=8,
            hidden_size=7168,
            database_mode=common.DatabaseMode.SILICON,
        )

    @pytest.fixture(autouse=True)
    def _reset_warning_dedupe(self):
        """The once-per-scale dedupe is process-global; isolate it per test."""
        MoEDispatch._wideep_comm_node1_fallback_logged.clear()
        yield
        MoEDispatch._wideep_comm_node1_fallback_logged.clear()

    def test_measured_multi_node_scale_is_neither_warned_nor_marked(self, monkeypatch, caplog):
        """node_num=2 is collected here: real data, used as-is, still "silicon"."""
        monkeypatch.setattr(MoEDispatch, "load_data", lambda database: None)
        db = self._db_with_ll_data()
        with caplog.at_level(logging.WARNING, logger="aiconfigurator_core.sdk.operations.moe"):
            res = self._query(db, node_num=2)
        assert float(res) == pytest.approx(0.2)  # 200us/1000, the node_num=2 bucket
        assert res.source == "silicon"
        assert caplog.records == []

    def test_unmeasured_multi_node_scale_substitutes_but_marks_fallback(self, monkeypatch):
        """node_num=4 is absent -> node_num=1 rows, tagged so the caller can surface it."""
        monkeypatch.setattr(MoEDispatch, "load_data", lambda database: None)
        db = self._db_with_ll_data()
        res = self._query(db, node_num=4)
        assert float(res) == pytest.approx(0.1)  # 100us/1000, the node_num=1 bucket
        assert res.source == common.WIDEEP_COMM_NODE1_FALLBACK_SOURCE

    def test_unmeasured_multi_node_scale_warns_with_the_measured_error(self, monkeypatch, caplog):
        """The warning names the scale and quotes the measured ratio for it."""
        monkeypatch.setattr(MoEDispatch, "load_data", lambda database: None)
        db = self._db_with_ll_data()
        with caplog.at_level(logging.WARNING, logger="aiconfigurator_core.sdk.operations.moe"):
            self._query(db, node_num=4)
        message = caplog.records[0].getMessage()
        assert "node_num=4" in message
        assert "FALLING BACK" in message
        assert common.WIDEEP_COMM_NODE1_FALLBACK_SOURCE in message
        # 3.1x is the measured node4/node1 median for the ll table; asserting the
        # figure keeps the message honest if _MEASURED_NODE_SCALING is edited.
        assert "3.1x" in message

    def test_warning_magnitude_tracks_the_node_scale(self, monkeypatch, caplog):
        """node_num=2 and node_num=8 are different propositions and say so."""
        monkeypatch.setattr(MoEDispatch, "load_data", lambda database: None)
        db = self._db_with_ll_data()
        db._wideep_deepep_ll_data.pop(2)  # force node_num=2 to extrapolate too
        with caplog.at_level(logging.WARNING, logger="aiconfigurator_core.sdk.operations.moe"):
            self._query(db, node_num=2)
            self._query(db, node_num=8)
        messages = [r.getMessage() for r in caplog.records]
        assert len(messages) == 2  # one per scale, not one per table
        assert "2.2x" in messages[0]
        assert "3.5x" in messages[1]

    def test_beyond_the_largest_measured_scale_is_stated_as_a_floor(self, monkeypatch, caplog):
        """Nobody measured node_num=16, so the message must not invent a figure for it."""
        monkeypatch.setattr(MoEDispatch, "load_data", lambda database: None)
        db = self._db_with_ll_data()
        with caplog.at_level(logging.WARNING, logger="aiconfigurator_core.sdk.operations.moe"):
            self._query(db, node_num=16)
        message = caplog.records[0].getMessage()
        assert "no campaign measured node_num=16" in message
        assert "lower bound" in message
        assert "3.5x" in message  # quoted as the node_num=8 floor, not as node_num=16

    def test_multi_node_warning_is_once_per_scale_not_once_per_query(self, monkeypatch, caplog):
        """A sweep hits each scale thousands of times; the log must not."""
        monkeypatch.setattr(MoEDispatch, "load_data", lambda database: None)
        db = self._db_with_ll_data()
        with caplog.at_level(logging.WARNING, logger="aiconfigurator_core.sdk.operations.moe"):
            for _ in range(5):
                self._query(db, node_num=4)
        assert len(caplog.records) == 1

    def test_sub_node_falls_back_to_node_num_1(self, monkeypatch):
        """node_num < 1 (a config smaller than one node) still uses node_num=1..."""
        monkeypatch.setattr(MoEDispatch, "load_data", lambda database: None)
        db = self._db_with_ll_data()
        res = self._query(db, node_num=4 / 8)  # MoEDispatch.query: num_gpus / num_gpus_per_node
        assert float(res) == pytest.approx(0.1)  # 100us/1000, node_num=1 fallback

    def test_sub_node_fallback_is_not_an_estimate(self, monkeypatch):
        """...and is not marked: no cross-node collective is being modelled away."""
        monkeypatch.setattr(MoEDispatch, "load_data", lambda database: None)
        db = self._db_with_ll_data()
        assert self._query(db, node_num=4 / 8).source == "silicon"

    def test_sub_node_fallback_keeps_its_own_warning(self, monkeypatch, caplog):
        """The sub-node message stays distinct from the multi-node one."""
        monkeypatch.setattr(MoEDispatch, "load_data", lambda database: None)
        db = self._db_with_ll_data()
        with caplog.at_level(logging.WARNING, logger="aiconfigurator_core.sdk.operations.moe"):
            self._query(db, node_num=4 / 8)
            self._query(db, node_num=4 / 8)
        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert "sub-node" in message
        assert "EXTRAPOLATING" not in message

    def test_node_num_1_no_fallback(self, monkeypatch):
        """node_num=1 uses its own data directly."""
        monkeypatch.setattr(MoEDispatch, "load_data", lambda database: None)
        db = self._db_with_ll_data()
        res = self._query(db, node_num=1)
        assert float(res) == pytest.approx(0.1)
        assert res.source == "silicon"

    def test_node_num_1_missing_shape_raises(self, monkeypatch):
        """node_num=1 with no rows for the shape -> not-available (as before)."""
        from aiconfigurator.sdk.errors import PerfDataNotAvailableError

        monkeypatch.setattr(MoEDispatch, "load_data", lambda database: None)
        db = self._db_with_ll_data()
        with pytest.raises(PerfDataNotAvailableError):
            self._query(db, node_num=1, num_experts=999)  # experts=999 absent at every node_num

    def test_neither_present_raises(self, monkeypatch):
        """No rows at the requested scale AND none at node_num=1 -> still not-available.

        Marking an estimate requires something to estimate from; with nothing to
        substitute, the config is dropped from the sweep as it always was.
        """
        from aiconfigurator.sdk.errors import PerfDataNotAvailableError

        monkeypatch.setattr(MoEDispatch, "load_data", lambda database: None)
        db = self._db_with_ll_data()
        with pytest.raises(PerfDataNotAvailableError):
            self._query(db, node_num=4, num_experts=999)  # experts=999 absent at every node_num

    def test_normal_table_shares_the_same_path(self, monkeypatch, caplog):
        """The normal-mode table resolves node_num through the same helper, so it
        substitutes, warns and marks identically -- with its own measured ratios,
        which diverge sharply from ll's at node 8 (6.2x vs 3.5x)."""
        monkeypatch.setattr(MoEDispatch, "load_data", lambda database: None)
        db = MagicMock()
        db._wideep_deepep_normal_data = {1: {7168: {8: {256: {20: self._leaf(100.0)}}}}}
        db._default_database_mode = common.DatabaseMode.SILICON
        db._interp_pr = lambda lat, energy=0.0: PerformanceResult(lat, energy=energy, source="silicon")

        def query(node_num):
            return MoEDispatch._query_wideep_deepep_normal_table(
                db,
                node_num=node_num,
                num_tokens=128,
                num_experts=256,
                topk=8,
                hidden_size=7168,
                sms=20,
                database_mode=common.DatabaseMode.SILICON,
            )

        with caplog.at_level(logging.WARNING, logger="aiconfigurator_core.sdk.operations.moe"):
            fallback_result = query(node_num=8)
        assert float(fallback_result) == pytest.approx(0.1)
        assert fallback_result.source == common.WIDEEP_COMM_NODE1_FALLBACK_SOURCE
        assert "wideep_deepep_normal" in caplog.records[0].getMessage()
        assert "6.2x" in caplog.records[0].getMessage()

        exact = query(node_num=1)
        assert float(exact) == pytest.approx(0.1)
        assert exact.source == "silicon"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

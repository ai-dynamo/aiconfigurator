# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AFD communication operations:

- :class:`AFDTransfer` — cross-pool bidirectional DMA
- :class:`AFDFAllGather` — F-node intra-node AllGather
- :class:`AFDFReduceScatter` — F-node intra-node ReduceScatter
- :class:`AFDCombine` — A-side cross-EP local reduce
"""

from __future__ import annotations

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import (
    AFDCombine,
    AFDFAllGather,
    AFDFReduceScatter,
    AFDTransfer,
    _afd_send_prob,
)
from aiconfigurator.sdk.performance_result import PerformanceResult

pytestmark = pytest.mark.unit


class _StubDatabase:
    """Minimal PerfDatabase stub returning deterministic latencies."""

    def __init__(self) -> None:
        self.p2p_calls: list[int] = []
        self.nccl_calls: list[tuple[common.CommQuantMode, int, str, int]] = []
        self.mem_calls: list[int] = []

    def query_p2p(self, message_bytes: int) -> PerformanceResult:
        self.p2p_calls.append(int(message_bytes))
        return PerformanceResult(latency=float(message_bytes) / 1.0e9, energy=0.0)

    def query_nccl(self, quant, tp, op_name, message_size):
        self.nccl_calls.append((quant, int(tp), str(op_name), int(message_size)))
        return PerformanceResult(latency=float(message_size) / 1.0e9, energy=0.0)

    def query_mem_op(self, total_bytes: int) -> PerformanceResult:
        self.mem_calls.append(int(total_bytes))
        return PerformanceResult(latency=float(total_bytes) / 1.0e9, energy=0.0)


# ---------------------------------------------------------------------------
# _afd_send_prob tests
# ---------------------------------------------------------------------------


class TestAfdSendProb:
    def test_dense_returns_one_over_nf(self):
        assert _afd_send_prob(0, 0, 4) == pytest.approx(0.25)

    def test_single_node_returns_one(self):
        assert _afd_send_prob(256, 8, 1) == pytest.approx(1.0)

    def test_moe_topk_greater_than_other_experts(self):
        assert _afd_send_prob(8, 8, 2) == 1.0

    def test_moe_normal_case(self):
        prob = _afd_send_prob(256, 8, 4)
        assert 0 < prob < 1

    def test_zero_experts_returns_uniform(self):
        assert _afd_send_prob(0, 8, 4) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# AFDTransfer tests
# ---------------------------------------------------------------------------


class TestAFDTransfer:
    def _make(self, direction="a2f", **overrides) -> AFDTransfer:
        base = dict(
            name="afd_transfer",
            scale_factor=1.0,
            direction=direction,
            hidden_size=1024,
            n_a_workers=4,
            n_f_workers=16,
            gpus_per_node=8,
            num_experts=0,
            topk=0,
            comm_quant_mode=common.CommQuantMode.half,
            comm_overhead_factor=1.0,
        )
        base.update(overrides)
        return AFDTransfer(**base)

    def test_returns_performance_result(self):
        db = _StubDatabase()
        op = self._make()
        result = op.query(db, x=32)
        assert isinstance(result, PerformanceResult)

    def test_single_direction_latency(self):
        db = _StubDatabase()
        a2f = self._make(direction="a2f")
        f2a = self._make(direction="f2a")
        r_a2f = a2f.query(db, x=32)
        r_f2a = f2a.query(db, x=32)
        assert float(r_a2f) == pytest.approx(float(r_f2a))

    def test_direction_property(self):
        op_a2f = self._make(direction="a2f")
        op_f2a = self._make(direction="f2a")
        assert op_a2f.direction == "a2f"
        assert op_f2a.direction == "f2a"

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="direction"):
            self._make(direction="both")

    def test_dense_per_link_bytes(self):
        db = _StubDatabase()
        op = self._make(n_a_workers=4, n_f_workers=16, gpus_per_node=8, hidden_size=1024)
        op.query(db, x=32)
        nf = 2  # 16 / 8
        p_send = _afd_send_prob(0, 0, nf)  # 1/nf = 0.5
        # per-link = single A-rank's tokens * p_send * hidden * bpe
        expected_bytes = int(p_send * 32 * 1024 * 2)
        assert db.p2p_calls[0] == expected_bytes

    def test_moe_selective_per_link_bytes(self):
        db = _StubDatabase()
        op = self._make(num_experts=256, topk=8)
        op.query(db, x=32)
        nf = 2
        p_send = _afd_send_prob(256, 8, nf)
        # per-link = single A-rank's 32 tokens
        expected_bytes = int(p_send * 32 * 1024 * 2)
        assert db.p2p_calls[0] == expected_bytes

    def test_num_f_nodes_property(self):
        op = self._make(n_f_workers=20, gpus_per_node=8)
        assert op.num_f_nodes == 3
        op_single = self._make(n_f_workers=4, gpus_per_node=8)
        assert op_single.num_f_nodes == 1

    def test_prefill_scales_linearly(self):
        op = self._make()
        db1 = _StubDatabase()
        db2 = _StubDatabase()
        r_decode = op.query(db1, x=32)
        r_prefill = op.query(db2, x=32 * 4096)
        assert float(r_prefill) == pytest.approx(float(r_decode) * 4096, rel=1e-3)


# ---------------------------------------------------------------------------
# AFDFAllGather tests
# ---------------------------------------------------------------------------


class TestAFDFAllGather:
    def _make(self, **overrides) -> AFDFAllGather:
        base = dict(
            name="afd_f_allgather",
            scale_factor=1.0,
            hidden_size=1024,
            n_a_workers=4,
            n_f_workers=16,
            gpus_per_node=8,
            num_experts=0,
            topk=0,
            comm_quant_mode=common.CommQuantMode.half,
            rank_mapping="one_to_one",
        )
        base.update(overrides)
        return AFDFAllGather(**base)

    def test_returns_performance_result(self):
        db = _StubDatabase()
        op = self._make()
        result = op.query(db, x=32)
        assert isinstance(result, PerformanceResult)

    def test_single_gpu_node_returns_zero(self):
        db = _StubDatabase()
        op = self._make(n_f_workers=1, gpus_per_node=1)
        result = op.query(db, x=32)
        assert float(result) == 0.0
        assert db.nccl_calls == []

    def test_broadcast_mapping_returns_zero(self):
        db = _StubDatabase()
        op = self._make(rank_mapping="broadcast")
        result = op.query(db, x=32)
        assert float(result) == 0.0

    def test_one_to_one_queries_nccl_allgather(self):
        db = _StubDatabase()
        op = self._make(n_f_workers=16, gpus_per_node=8)
        op.query(db, x=32)
        assert len(db.nccl_calls) == 1
        assert db.nccl_calls[0][2] == "all_gather"
        assert db.nccl_calls[0][1] == 8  # min(16, 8) = 8 GPUs in node

    def test_ep8_tp1_still_needs_allgather(self):
        db = _StubDatabase()
        op = self._make(n_f_workers=8, gpus_per_node=8)
        result = op.query(db, x=32)
        assert float(result) > 0.0
        assert db.nccl_calls[0][1] == 8

    def test_message_size_is_per_rank_chunk(self):
        """``query_nccl`` takes the per-rank sendcount, not the per-F-node total.

        AllGather participants are the ``f_local = min(n_f_workers,
        gpus_per_node)`` GPUs in a single F-node; each one contributes
        ``tokens_per_f_node * hidden_size / f_local`` elements. Passing
        the un-divided per-node total would over-report bandwidth by
        ``f_local``x and silently flip the comm-vs-compute bottleneck.
        """
        db = _StubDatabase()
        op = self._make(n_a_workers=4, n_f_workers=16, gpus_per_node=8, hidden_size=1024)
        op.query(db, x=32)
        total = 32 * 4
        nf = 2
        f_local = 8  # min(n_f_workers, gpus_per_node) = min(16, 8)
        p_send = _afd_send_prob(0, 0, nf)
        expected_msg = int(p_send * total * 1024 / f_local)
        assert db.nccl_calls[0][3] == expected_msg

    def test_invalid_rank_mapping_raises(self):
        with pytest.raises(ValueError, match="rank_mapping"):
            self._make(rank_mapping="ring")


# ---------------------------------------------------------------------------
# AFDFReduceScatter tests
# ---------------------------------------------------------------------------


class TestAFDFReduceScatter:
    def _make(self, **overrides) -> AFDFReduceScatter:
        base = dict(
            name="afd_f_reduce_scatter",
            scale_factor=1.0,
            hidden_size=1024,
            n_a_workers=4,
            n_f_workers=16,
            gpus_per_node=8,
            num_experts=0,
            topk=0,
            comm_quant_mode=common.CommQuantMode.half,
            rank_mapping="one_to_one",
        )
        base.update(overrides)
        return AFDFReduceScatter(**base)

    def test_returns_performance_result(self):
        db = _StubDatabase()
        op = self._make()
        result = op.query(db, x=32)
        assert isinstance(result, PerformanceResult)

    def test_single_gpu_node_returns_zero(self):
        db = _StubDatabase()
        op = self._make(n_f_workers=1, gpus_per_node=1)
        result = op.query(db, x=32)
        assert float(result) == 0.0

    def test_ep8_tp1_still_needs_reduce_scatter(self):
        db = _StubDatabase()
        op = self._make(n_f_workers=8, gpus_per_node=8)
        result = op.query(db, x=32)
        assert float(result) > 0.0
        assert db.nccl_calls[0][1] == 8
        assert db.nccl_calls[0][2] == "reduce_scatter"

    def test_one_to_one_queries_nccl_reduce_scatter(self):
        db = _StubDatabase()
        op = self._make(n_f_workers=16, gpus_per_node=8)
        op.query(db, x=32)
        assert len(db.nccl_calls) == 1
        assert db.nccl_calls[0][2] == "reduce_scatter"
        assert db.nccl_calls[0][1] == 8  # min(16, 8) = 8 GPUs in node

    def test_invalid_rank_mapping_raises(self):
        with pytest.raises(ValueError, match="rank_mapping"):
            self._make(rank_mapping="bogus")


# ---------------------------------------------------------------------------
# AFDCombine tests
# ---------------------------------------------------------------------------


class TestAFDCombine:
    def _make(self, **overrides) -> AFDCombine:
        base = dict(
            name="afd_combine",
            scale_factor=1.0,
            hidden_size=1024,
            tp_a=1,
            f_moe_ep_size=1,
            comm_quant_mode=common.CommQuantMode.half,
        )
        base.update(overrides)
        return AFDCombine(**base)

    def test_returns_performance_result(self):
        db = _StubDatabase()
        op = self._make(f_moe_ep_size=4)
        result = op.query(db, x=32)
        assert isinstance(result, PerformanceResult)

    def test_dense_ep1_returns_zero(self):
        db = _StubDatabase()
        op = self._make(f_moe_ep_size=1)
        result = op.query(db, x=32)
        assert float(result) == 0.0
        assert db.mem_calls == []

    def test_ep_gt1_calls_mem_op(self):
        db = _StubDatabase()
        op = self._make(f_moe_ep_size=4, tp_a=1)
        op.query(db, x=32)
        assert len(db.mem_calls) == 1
        expected_bytes = (4 + 1) * 32 * 1024 * 2
        assert db.mem_calls[0] == expected_bytes

    def test_tp_a_divides_tokens(self):
        db = _StubDatabase()
        op = self._make(f_moe_ep_size=4, tp_a=2)
        op.query(db, x=32)
        expected_bytes = (4 + 1) * 16 * 1024 * 2
        assert db.mem_calls[0] == expected_bytes

    def test_prefill_scales_linearly(self):
        op = self._make(f_moe_ep_size=4)
        db1 = _StubDatabase()
        db2 = _StubDatabase()
        r_decode = op.query(db1, x=32)
        r_prefill = op.query(db2, x=32 * 4096)
        assert float(r_prefill) == pytest.approx(float(r_decode) * 4096, rel=1e-3)


# ---------------------------------------------------------------------------
# Numerical equivalence with old monolithic AFDTransfer behavior
# ---------------------------------------------------------------------------


class TestNumericalEquivalence:
    """Verify the 5-op split with unidirectional transfers and token-dim AG/RS.

    A-side is DP: each A-rank sends full hidden_size per token.
    F-side AllGather/ReduceScatter operate along the token dimension
    across all GPUs in a node (determined by gpus_per_node, not TP).
    Two separate AFDTransfer instances model A→F and F→A independently.
    """

    def _query_split(
        self, *, x, n_a_workers=4, n_f_workers=16, gpus_per_node=8, tp_a=1, f_moe_ep_size=1, num_experts=0, topk=0
    ):
        db = _StubDatabase()
        hidden_size = 1024
        qm = common.CommQuantMode.half
        common_kw = dict(
            hidden_size=hidden_size,
            n_a_workers=n_a_workers,
            n_f_workers=n_f_workers,
            gpus_per_node=gpus_per_node,
            num_experts=num_experts,
            topk=topk,
            comm_quant_mode=qm,
            comm_overhead_factor=1.0,
        )

        a2f = AFDTransfer(name="a2f", scale_factor=1.0, direction="a2f", **common_kw)
        f2a = AFDTransfer(name="f2a", scale_factor=1.0, direction="f2a", **common_kw)
        ag = AFDFAllGather(
            name="afd_f_allgather",
            scale_factor=1.0,
            hidden_size=hidden_size,
            n_a_workers=n_a_workers,
            n_f_workers=n_f_workers,
            gpus_per_node=gpus_per_node,
            num_experts=num_experts,
            topk=topk,
            comm_quant_mode=qm,
            rank_mapping="one_to_one",
        )
        rs = AFDFReduceScatter(
            name="afd_f_reduce_scatter",
            scale_factor=1.0,
            hidden_size=hidden_size,
            n_a_workers=n_a_workers,
            n_f_workers=n_f_workers,
            gpus_per_node=gpus_per_node,
            num_experts=num_experts,
            topk=topk,
            comm_quant_mode=qm,
            rank_mapping="one_to_one",
        )
        combine = AFDCombine(
            name="afd_combine",
            scale_factor=1.0,
            hidden_size=hidden_size,
            tp_a=tp_a,
            f_moe_ep_size=f_moe_ep_size,
            comm_quant_mode=qm,
        )

        t_a2f = a2f.query(db, x=x)
        t_f2a = f2a.query(db, x=x)
        t_ag = ag.query(db, x=x)
        t_rs = rs.query(db, x=x)
        t_comb = combine.query(db, x=x)

        return {
            "t_a2f": float(t_a2f),
            "t_f2a": float(t_f2a),
            "t_c": float(t_a2f) + float(t_f2a),
            "ag": float(t_ag),
            "rs": float(t_rs),
            "combine": float(t_comb),
        }

    def test_dense_8gpu_node(self):
        r = self._query_split(x=32, n_f_workers=16, gpus_per_node=8, f_moe_ep_size=1)
        assert r["t_a2f"] == pytest.approx(r["t_f2a"])
        assert r["t_c"] == pytest.approx(r["t_a2f"] + r["t_f2a"])
        assert r["combine"] == 0.0
        assert r["ag"] > 0.0
        assert r["rs"] > 0.0

    def test_moe_ep4(self):
        r = self._query_split(
            x=32,
            n_f_workers=16,
            gpus_per_node=8,
            f_moe_ep_size=4,
            num_experts=256,
            topk=8,
        )
        assert r["t_c"] == pytest.approx(r["t_a2f"] + r["t_f2a"])
        assert r["combine"] > 0.0
        assert r["ag"] > 0.0
        assert r["rs"] > 0.0

    def test_single_gpu_node_zeroes_collectives(self):
        r = self._query_split(x=32, n_f_workers=1, gpus_per_node=1, f_moe_ep_size=1)
        assert r["ag"] == 0.0
        assert r["rs"] == 0.0
        assert r["combine"] == 0.0
        assert r["t_a2f"] > 0.0

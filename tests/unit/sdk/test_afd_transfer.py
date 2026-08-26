# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AFD communication operations:

- :class:`AFDTransfer` — cross-pool bidirectional DMA
- :class:`AFDFAllGather` — F-node intra-node AllGather
- :class:`AFDFReduceScatter` — F-node intra-node ReduceScatter
- :class:`AFDCombine` — A-side cross-EP local reduce

These assert the Python-side TOPOLOGY math (send probability, per-link and
per-rank volumes, zero gates). The per-message latency comes from the
compiled engine via ``afd_transfer._engine_comm_query`` — stubbed here at
that seam, recording the probe twin op each leg builds. Value fidelity
against the engine is covered by ``tests/cross_package/
the frozen parity goldens (afd-* cases were pinned on a real database
through the migration window).
"""

from __future__ import annotations

import json

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import (
    AFDCombine,
    AFDFAllGather,
    AFDFReduceScatter,
    AFDTransfer,
    _afd_send_prob,
    afd_dest_ep_ranks,
)
from aiconfigurator.sdk.performance_result import PerformanceResult
from aiconfigurator_core.sdk.operations import afd_transfer as afd_transfer_module
from aiconfigurator_core.sdk.operations.communication import NCCL, P2P
from aiconfigurator_core.sdk.operations.elementwise import ElementWise

pytestmark = pytest.mark.unit


def _half_ceil(num_bytes: int) -> int:
    """The bf16-element count a byte volume becomes on a probe op."""
    return -(-int(num_bytes) // 2)


class _EngineStub:
    """Stands in for ``_engine_comm_query``: records each probe twin op and
    returns a deterministic latency proportional to its message volume."""

    def __init__(self) -> None:
        self.p2p_calls: list[int] = []  # P2P probe hidden_size (= ceil(bytes/2))
        self.nccl_calls: list[tuple[common.CommQuantMode, int, str, int]] = []
        self.mem_calls: list[int] = []  # ElementWise probe dim_in (= ceil(bytes/2))

    def __call__(self, database, op) -> PerformanceResult:
        if isinstance(op, P2P):
            self.p2p_calls.append(int(op._h))
            volume = op._h * 2
        elif isinstance(op, NCCL):
            self.nccl_calls.append(
                (op._comm_quant_mode, int(op._num_gpus), str(op._nccl_op), int(op._num_elements_per_token))
            )
            volume = op._num_elements_per_token
        elif isinstance(op, ElementWise):
            # AFD builds the mem twin as ElementWise(dim_in=ceil(bytes/2),
            # dim_out=0); the Rust op folds dims to bytes_per_token =
            # (dim_in + dim_out) * 2, so dim_in recovers exactly.
            bytes_per_token = json.loads(op._spec_json())["Elementwise"]["bytes_per_token"]
            dim_in = int(bytes_per_token) // 2
            self.mem_calls.append(dim_in)
            volume = bytes_per_token
        else:  # pragma: no cover — a new leg must extend this stub deliberately
            raise TypeError(f"unexpected probe op {type(op).__name__}")
        return PerformanceResult(latency=float(volume) / 1.0e9, energy=0.0)


@pytest.fixture()
def engine(monkeypatch) -> _EngineStub:
    stub = _EngineStub()
    monkeypatch.setattr(afd_transfer_module, "_engine_comm_query", stub)
    return stub


_DB = object()  # the stub never touches the database


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

    def test_returns_performance_result(self, engine):
        op = self._make()
        result = op.query(_DB, x=32)
        assert isinstance(result, PerformanceResult)

    def test_single_direction_latency(self, engine):
        a2f = self._make(direction="a2f")
        f2a = self._make(direction="f2a")
        r_a2f = a2f.query(_DB, x=32)
        r_f2a = f2a.query(_DB, x=32)
        assert float(r_a2f) == pytest.approx(float(r_f2a))

    def test_direction_property(self):
        op_a2f = self._make(direction="a2f")
        op_f2a = self._make(direction="f2a")
        assert op_a2f.direction == "a2f"
        assert op_f2a.direction == "f2a"

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="direction"):
            self._make(direction="both")

    def test_dense_aggregate_egress_bytes(self, engine):
        op = self._make(n_a_workers=4, n_f_workers=16, gpus_per_node=8, hidden_size=1024)
        op.query(_DB, x=32)
        nf = 2  # 16 / 8
        p_send = _afd_send_prob(0, 0, nf)  # 1/nf = 0.5
        # Aggregate egress of one A-rank: nf destinations sharing one NIC.
        # Dense: nf * (1/nf) = 1, i.e. the whole batch leaves the NIC once.
        expected_bytes = int(nf * p_send * 32 * 1024 * 2)
        assert engine.p2p_calls[0] == _half_ceil(expected_bytes)
        assert expected_bytes == 32 * 1024 * 2

    def test_moe_selective_aggregate_egress_bytes(self, engine):
        op = self._make(num_experts=256, topk=8)
        op.query(_DB, x=32)
        nf = 2
        p_send = _afd_send_prob(256, 8, nf)
        # Aggregate over both destination nodes; each gets the deduped subset.
        expected_bytes = int(nf * p_send * 32 * 1024 * 2)
        assert engine.p2p_calls[0] == _half_ceil(expected_bytes)

    def test_wide_a_pool_is_billed_at_the_f_side_ingress(self, engine):
        # 64 A ranks feeding 16 F GPUs: each F GPU handles 4x what one A rank
        # sends, so the F side sets the pace.
        self._make(n_a_workers=64, n_f_workers=16, gpus_per_node=8).query(_DB, x=32)
        self._make(n_a_workers=4, n_f_workers=16, gpus_per_node=8).query(_DB, x=32)
        assert engine.p2p_calls[0] == 4 * engine.p2p_calls[1]

    def test_num_f_nodes_property(self):
        op = self._make(n_f_workers=20, gpus_per_node=8)
        assert op.num_f_nodes == 3
        op_single = self._make(n_f_workers=4, gpus_per_node=8)
        assert op_single.num_f_nodes == 1

    def test_prefill_scales_linearly(self, engine):
        op = self._make()
        r_decode = op.query(_DB, x=32)
        r_prefill = op.query(_DB, x=32 * 4096)
        assert float(r_prefill) == pytest.approx(float(r_decode) * 4096, rel=1e-3)


# ---------------------------------------------------------------------------
# dispatch_mode="a_side_routing" (DeepEP low-latency style direct dispatch)
# ---------------------------------------------------------------------------


def test_afd_dest_ep_ranks_anchors():
    assert afd_dest_ep_ranks(256, 8, 1) == pytest.approx(1.0)  # single rank: one destination
    assert 1.0 < afd_dest_ep_ranks(256, 8, 16) <= min(8, 16)  # bounded by min(topk, ep)
    assert afd_dest_ep_ranks(16, 4, 16) == pytest.approx(4.0)  # one expert/rank: exactly topk


class TestAFDTransferASideRouting:
    """A-side routing bills one A-rank's *aggregate* egress across all
    destination EP ranks; ``f_side_routing`` aggregates over the F nodes."""

    def _make(self, direction="a2f", **overrides) -> AFDTransfer:
        base = dict(
            name="afd_transfer",
            scale_factor=1.0,
            direction=direction,
            hidden_size=1024,
            n_a_workers=4,
            n_f_workers=32,
            gpus_per_node=8,
            num_experts=256,
            topk=8,
            comm_quant_mode=common.CommQuantMode.half,
            dispatch_mode="a_side_routing",
            f_moe_ep_size=16,
        )
        base.update(overrides)
        return AFDTransfer(**base)

    def test_a2f_bytes_include_fanout_and_tp_replication(self, engine):
        op = self._make(direction="a2f")
        op.query(_DB, x=32)
        n_dest = afd_dest_ep_ranks(256, 8, 16)
        moe_tp = 32 // 16
        assert engine.p2p_calls[0] == _half_ceil(int(n_dest * moe_tp * 32 * 1024 * 2))

    def test_f2a_returns_one_partial_per_ep_rank(self, engine):
        op = self._make(direction="f2a")
        op.query(_DB, x=32)
        n_dest = afd_dest_ep_ranks(256, 8, 16)
        assert engine.p2p_calls[0] == _half_ceil(int(n_dest * 32 * 1024 * 2))

    def test_invalid_dispatch_mode_raises(self):
        with pytest.raises(ValueError, match="dispatch_mode"):
            self._make(dispatch_mode="a_side")

    def test_dense_model_rejected(self):
        with pytest.raises(ValueError, match="requires a MoE model"):
            self._make(num_experts=0, topk=0)

    def test_f_side_routing_unchanged_by_new_kwargs(self, engine):
        """Passing the new knobs must not perturb the default topology."""
        self._make(direction="a2f", dispatch_mode="f_side_routing", f_moe_ep_size=16).query(_DB, x=32)
        nf = 4  # 32 F-GPUs / 8 per node
        p_send = _afd_send_prob(256, 8, nf)
        assert engine.p2p_calls[0] == _half_ceil(int(nf * p_send * 32 * 1024 * 2))


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
        )
        base.update(overrides)
        return AFDFAllGather(**base)

    def test_returns_performance_result(self, engine):
        op = self._make()
        result = op.query(_DB, x=32)
        assert isinstance(result, PerformanceResult)

    def test_single_gpu_node_returns_zero(self, engine):
        op = self._make(n_f_workers=1, gpus_per_node=1)
        result = op.query(_DB, x=32)
        assert float(result) == 0.0
        assert engine.nccl_calls == []

    def test_a_side_routing_returns_zero(self, engine):
        # A-side routing sends tokens straight to their experts' GPUs, so
        # there is nothing left to redistribute inside the F node.
        op = self._make(dispatch_mode="a_side_routing", num_experts=256, topk=8)
        assert float(op.query(_DB, x=32)) == 0.0
        assert engine.nccl_calls == []

    def test_one_to_one_queries_nccl_allgather(self, engine):
        op = self._make(n_f_workers=16, gpus_per_node=8)
        op.query(_DB, x=32)
        assert len(engine.nccl_calls) == 1
        assert engine.nccl_calls[0][2] == "all_gather"
        assert engine.nccl_calls[0][1] == 8  # min(16, 8) = 8 GPUs in node

    def test_ep8_tp1_still_needs_allgather(self, engine):
        op = self._make(n_f_workers=8, gpus_per_node=8)
        result = op.query(_DB, x=32)
        assert float(result) > 0.0
        assert engine.nccl_calls[0][1] == 8

    def test_message_size_is_per_rank_chunk(self, engine):
        """The NCCL probe carries the per-rank sendcount, not the per-F-node total.

        AllGather participants are the ``f_local = min(n_f_workers,
        gpus_per_node)`` GPUs in a single F-node; each one contributes
        ``tokens_per_f_node * hidden_size / f_local`` elements. Passing
        the un-divided per-node total would over-report bandwidth by
        ``f_local``x and silently flip the comm-vs-compute bottleneck.
        """
        op = self._make(n_a_workers=4, n_f_workers=16, gpus_per_node=8, hidden_size=1024)
        op.query(_DB, x=32)
        total = 32 * 4
        nf = 2
        f_local = 8  # min(n_f_workers, gpus_per_node) = min(16, 8)
        p_send = _afd_send_prob(0, 0, nf)
        expected_msg = int(p_send * total * 1024 / f_local)
        assert engine.nccl_calls[0][3] == expected_msg

    def test_invalid_dispatch_mode_raises(self):
        with pytest.raises(ValueError, match="dispatch_mode"):
            self._make(dispatch_mode="ring")


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
        )
        base.update(overrides)
        return AFDFReduceScatter(**base)

    def test_returns_performance_result(self, engine):
        op = self._make()
        result = op.query(_DB, x=32)
        assert isinstance(result, PerformanceResult)

    def test_single_gpu_node_returns_zero(self, engine):
        op = self._make(n_f_workers=1, gpus_per_node=1)
        result = op.query(_DB, x=32)
        assert float(result) == 0.0

    def test_a_side_routing_returns_zero(self, engine):
        # Each F-GPU returns its own partials straight to the owning A-rank.
        op = self._make(dispatch_mode="a_side_routing", num_experts=256, topk=8)
        assert float(op.query(_DB, x=32)) == 0.0
        assert engine.nccl_calls == []

    def test_ep8_tp1_still_needs_reduce_scatter(self, engine):
        op = self._make(n_f_workers=8, gpus_per_node=8)
        result = op.query(_DB, x=32)
        assert float(result) > 0.0
        assert engine.nccl_calls[0][1] == 8
        assert engine.nccl_calls[0][2] == "reduce_scatter"

    def test_one_to_one_queries_nccl_reduce_scatter(self, engine):
        op = self._make(n_f_workers=16, gpus_per_node=8)
        op.query(_DB, x=32)
        assert len(engine.nccl_calls) == 1
        assert engine.nccl_calls[0][2] == "reduce_scatter"
        assert engine.nccl_calls[0][1] == 8  # min(16, 8) = 8 GPUs in node

    def test_invalid_dispatch_mode_raises(self):
        with pytest.raises(ValueError, match="dispatch_mode"):
            self._make(dispatch_mode="bogus")


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

    def test_returns_performance_result(self, engine):
        op = self._make(f_moe_ep_size=4)
        result = op.query(_DB, x=32)
        assert isinstance(result, PerformanceResult)

    def test_dense_ep1_returns_zero(self, engine):
        op = self._make(f_moe_ep_size=1)
        result = op.query(_DB, x=32)
        assert float(result) == 0.0
        assert engine.mem_calls == []

    def test_ep_gt1_calls_mem_op(self, engine):
        op = self._make(f_moe_ep_size=4, tp_a=1)
        op.query(_DB, x=32)
        assert len(engine.mem_calls) == 1
        expected_bytes = (4 + 1) * 32 * 1024 * 2
        assert engine.mem_calls[0] == _half_ceil(expected_bytes)

    def test_tp_a_divides_tokens(self, engine):
        op = self._make(f_moe_ep_size=4, tp_a=2)
        op.query(_DB, x=32)
        expected_bytes = (4 + 1) * 16 * 1024 * 2
        assert engine.mem_calls[0] == _half_ceil(expected_bytes)

    def test_prefill_scales_linearly(self, engine):
        op = self._make(f_moe_ep_size=4)
        r_decode = op.query(_DB, x=32)
        r_prefill = op.query(_DB, x=32 * 4096)
        assert float(r_prefill) == pytest.approx(float(r_decode) * 4096, rel=1e-3)

    def test_a_side_routing_reduces_only_reached_ep_ranks(self, engine):
        # Only the EP ranks holding one of the token's top-k experts send a
        # partial back, so the reduce is over the expected fan-out, not over
        # every EP rank.
        op = self._make(f_moe_ep_size=16, dispatch_mode="a_side_routing", num_experts=256, topk=8)
        op.query(_DB, x=32)
        n_dest = afd_dest_ep_ranks(256, 8, 16)
        assert engine.mem_calls[0] == _half_ceil(int((n_dest + 1) * 32 * 1024 * 2))
        assert engine.mem_calls[0] < _half_ceil((16 + 1) * 32 * 1024 * 2)  # cheaper than f_side_routing


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
        )
        combine = AFDCombine(
            name="afd_combine",
            scale_factor=1.0,
            hidden_size=hidden_size,
            tp_a=tp_a,
            f_moe_ep_size=f_moe_ep_size,
            comm_quant_mode=qm,
        )

        t_a2f = a2f.query(_DB, x=x)
        t_f2a = f2a.query(_DB, x=x)
        t_ag = ag.query(_DB, x=x)
        t_rs = rs.query(_DB, x=x)
        t_comb = combine.query(_DB, x=x)

        return {
            "t_a2f": float(t_a2f),
            "t_f2a": float(t_f2a),
            "t_c": float(t_a2f) + float(t_f2a),
            "ag": float(t_ag),
            "rs": float(t_rs),
            "combine": float(t_comb),
        }

    def test_dense_8gpu_node(self, engine):
        r = self._query_split(x=32, n_f_workers=16, gpus_per_node=8, f_moe_ep_size=1)
        assert r["t_a2f"] == pytest.approx(r["t_f2a"])
        assert r["t_c"] == pytest.approx(r["t_a2f"] + r["t_f2a"])
        assert r["combine"] == 0.0
        assert r["ag"] > 0.0
        assert r["rs"] > 0.0

    def test_moe_ep4(self, engine):
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

    def test_single_gpu_node_zeroes_collectives(self, engine):
        r = self._query_split(x=32, n_f_workers=1, gpus_per_node=1, f_moe_ep_size=1)
        assert r["ag"] == 0.0
        assert r["rs"] == 0.0
        assert r["combine"] == 0.0
        assert r["t_a2f"] > 0.0

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the consolidated :class:`AFDTransfer` op."""

from __future__ import annotations

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import AFDTransfer
from aiconfigurator.sdk.performance_result import PerformanceResult

pytestmark = pytest.mark.unit


class _StubDatabase:
    """Minimal PerfDatabase stub for AFDTransfer unit tests.

    Returns deterministic latencies proportional to message size so that
    the relative ordering of returned values is predictable, while
    keeping the test independent of any real perf-table content.
    """

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


def _make(**overrides) -> AFDTransfer:
    base = dict(
        name="afd_transfer",
        hidden_size=1024,
        n_a_workers=4,
        n_f_workers=16,
        gpus_per_node=8,
        tp_a=1,
        tp_f=8,
        f_moe_ep_size=1,
        topk=1,
        num_experts=1,
        comm_quant_mode=common.CommQuantMode.half,
        comm_overhead_factor=1.0,
        transfer_mode="p2p",
        rank_mapping="one_to_one",
    )
    base.update(overrides)
    return AFDTransfer(**base)


def test_query_returns_expected_dict_shape():
    db = _StubDatabase()
    op = _make()
    brk = op.query(db, b_total=128, a_batch_size=32)
    assert set(brk.keys()) == {"t_a2f", "t_f2a", "t_a", "t_f"}
    assert set(brk["t_a2f"].keys()) == {"afd_transfer_a2f"}
    assert set(brk["t_f2a"].keys()) == {"afd_transfer_f2a"}
    assert set(brk["t_a"].keys()) == {"afd_combine"}
    assert set(brk["t_f"].keys()) == {"afd_f_allgather", "afd_f_reduce_scatter"}


def test_a2f_equals_f2a_under_symmetric_assumption():
    db = _StubDatabase()
    op = _make()
    brk = op.query(db, b_total=128, a_batch_size=32)
    assert brk["t_a2f"]["afd_transfer_a2f"] == brk["t_f2a"]["afd_transfer_f2a"]


def test_dense_tp1_zeroes_all_collectives():
    """Dense (EP=1) + tp_f=1 → AG/RS/Combine all zero, only cross-pool runs."""
    db = _StubDatabase()
    op = _make(tp_f=1, f_moe_ep_size=1)
    brk = op.query(db, b_total=128, a_batch_size=32)
    assert brk["t_a"]["afd_combine"] == 0.0
    assert brk["t_f"]["afd_f_allgather"] == 0.0
    assert brk["t_f"]["afd_f_reduce_scatter"] == 0.0
    assert brk["t_a2f"]["afd_transfer_a2f"] > 0.0
    assert db.nccl_calls == []
    assert db.mem_calls == []


def test_dense_tp_f8_runs_ag_and_rs_but_no_combine():
    """1:1 dense with tp_f>1 → AG and RS > 0, Combine = 0."""
    db = _StubDatabase()
    op = _make(tp_f=8, f_moe_ep_size=1)
    brk = op.query(db, b_total=128, a_batch_size=32)
    assert brk["t_a"]["afd_combine"] == 0.0
    assert brk["t_f"]["afd_f_allgather"] > 0.0
    assert brk["t_f"]["afd_f_reduce_scatter"] > 0.0
    op_names = {call[2] for call in db.nccl_calls}
    assert op_names == {"all_gather", "reduce_scatter"}


def test_moe_one_to_one_runs_ag_rs_and_combine():
    """1:1 MoE → AG, RS, and Combine all > 0."""
    db = _StubDatabase()
    op = _make(
        tp_f=8,
        f_moe_ep_size=4,
        transfer_mode="moe_selective",
        topk=8,
        num_experts=256,
    )
    brk = op.query(db, b_total=128, a_batch_size=32)
    assert brk["t_a"]["afd_combine"] > 0.0
    assert brk["t_f"]["afd_f_allgather"] > 0.0
    assert brk["t_f"]["afd_f_reduce_scatter"] > 0.0


def test_broadcast_mapping_zeroes_f_collectives_only():
    """Non-1:1 (broadcast) → F-side AG/RS = 0; combine and cross-pool still run."""
    db = _StubDatabase()
    op = _make(
        tp_f=8,
        f_moe_ep_size=4,
        transfer_mode="moe_selective",
        topk=8,
        num_experts=256,
        rank_mapping="broadcast",
    )
    brk = op.query(db, b_total=128, a_batch_size=32)
    assert brk["t_f"]["afd_f_allgather"] == 0.0
    assert brk["t_f"]["afd_f_reduce_scatter"] == 0.0
    assert brk["t_a"]["afd_combine"] > 0.0
    assert brk["t_a2f"]["afd_transfer_a2f"] > 0.0
    assert db.nccl_calls == []


def test_prefill_scales_linearly_with_isl():
    """Cross-pool DMA latency scales linearly with the token volume kwarg.

    Stub returns latency proportional to message bytes, and the stub
    ignores any bandwidth saturation curve, so a 4096x volume should
    produce a strictly larger latency (well above the small base).
    """
    db_decode = _StubDatabase()
    db_prefill = _StubDatabase()
    op = _make(
        tp_f=8,
        f_moe_ep_size=4,
        transfer_mode="moe_selective",
        topk=8,
        num_experts=256,
    )
    decode = op.query(db_decode, b_total=192, a_batch_size=64)
    prefill = op.query(db_prefill, b_total=192 * 4096, a_batch_size=64 * 4096)

    decode_a2f = decode["t_a2f"]["afd_transfer_a2f"]
    prefill_a2f = prefill["t_a2f"]["afd_transfer_a2f"]
    assert prefill_a2f == pytest.approx(decode_a2f * 4096, rel=1e-3)

    decode_combine = decode["t_a"]["afd_combine"]
    prefill_combine = prefill["t_a"]["afd_combine"]
    assert prefill_combine == pytest.approx(decode_combine * 4096, rel=1e-3)


def test_a_batch_size_defaults_from_b_total_when_missing():
    """If caller omits ``a_batch_size``, it is derived as ``b_total / n_a_workers``."""
    db = _StubDatabase()
    op = _make(
        tp_a=1,
        n_a_workers=4,
        tp_f=8,
        f_moe_ep_size=4,
        transfer_mode="moe_selective",
        topk=8,
        num_experts=256,
    )
    explicit = op.query(_StubDatabase(), b_total=128, a_batch_size=32)
    derived = op.query(db, b_total=128)
    assert (
        explicit["t_a"]["afd_combine"]
        == pytest.approx(derived["t_a"]["afd_combine"], rel=1e-9)
    )


def test_invalid_transfer_mode_raises():
    with pytest.raises(ValueError, match="transfer_mode"):
        _make(transfer_mode="bogus")


def test_invalid_rank_mapping_raises():
    with pytest.raises(ValueError, match="rank_mapping"):
        _make(rank_mapping="ring")


def test_num_f_nodes_uses_physical_topology():
    """``num_f_nodes`` derives from ``ceil(n_f_workers / gpus_per_node)``."""
    op = _make(n_f_workers=20, gpus_per_node=8)
    assert op.num_f_nodes == 3
    op_single = _make(n_f_workers=4, gpus_per_node=8)
    assert op_single.num_f_nodes == 1

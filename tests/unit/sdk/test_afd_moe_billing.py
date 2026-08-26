# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two F-pool billing defects, both of which inflated ``t_f_layer``.

**A. The F-node AllGather/ReduceScatter fired without a TP group.**
The pair was gated on ``min(n_f_workers, f_gpus_per_node) > 1`` -- a property
of the *fabric*. A token-dimension collective presupposes a *TP group*: under
pure expert parallelism every F rank owns its own experts and there is nothing
to gather along tokens. ``f_moe_tp_size`` is now an existence gate; it never
touches sizing.

**B. The model-internal MoE dispatch was billed twice.**
Under AFD that all-to-all *is* the cross-pool A<->F transfer; it does not run
in addition to it. ``build_afd_ops_partition`` drops a bare ``MoEDispatch`` via
its skip list, but one nested inside an ``OverlapOp`` survived: the skip marker
only removes it from that op's classification vote, while
``OverlapOp.query()`` (latency = ``max(sum(group_a), sum(group_b))``) still
sums it into F-pool latency. It never reached ``skipped_ops`` either, so the
audit trail showed nothing dropped -- which is why the cost went unnoticed.

Measured on gb200/sglang: pure EP saves 76.1 us/layer of AG+RS; DeepSeek-V3
(shared experts, so the MoE ops sit inside an OverlapOp) drops 8.7% of t_f,
while Qwen3-235B-A22B (no shared expert, so the ops are flat and the skip list
already worked) is unchanged to the last digit.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aiconfigurator.sdk import operations
from aiconfigurator.sdk.afd_partition import AFDOpsPartition
from aiconfigurator.sdk.config import AFDConfig
from aiconfigurator.sdk.inference_session import (
    _is_moe_dispatch_op,
    _strip_moe_dispatch_from_partition,
)
from aiconfigurator.sdk.operations import AFDFAllGather, AFDFReduceScatter
from aiconfigurator.sdk.performance_result import PerformanceResult
from aiconfigurator_core.sdk.operations import afd_transfer as afd_transfer_module

pytestmark = pytest.mark.unit


class _EngineStub:
    """Stands in for ``_engine_comm_query``: records each probe twin op and
    returns a fixed latency.

    The engine itself rejects duck-typed databases (``_require_real_database``),
    so a collective-gate test cannot hand a synthetic object to the real path.
    Intercepting the one function that reaches the engine keeps the gate logic
    under test without needing perf data on disk.
    """

    def __init__(self) -> None:
        self.calls: list = []

    def __call__(self, database, op):
        self.calls.append(op)
        return PerformanceResult(latency=1.0, energy=0.0)


_DB = object()  # never inspected: the stub short-circuits before the engine


def _real_database():
    """A real ``PerfDatabase``: the engine loads perf tables from disk and
    ``_require_real_database`` rejects stand-ins, so any assertion that goes
    through the real single-op path needs one."""
    from aiconfigurator.sdk import perf_database

    return perf_database.get_database("h200_sxm", "sglang", "0.5.16")


@pytest.fixture()
def engine(monkeypatch) -> _EngineStub:
    stub = _EngineStub()
    monkeypatch.setattr(afd_transfer_module, "_engine_comm_query", stub)
    return stub


_COLLECTIVE_KW = {
    "hidden_size": 4096,
    "n_a_workers": 12,
    "n_f_workers": 16,
    "gpus_per_node": 8,
    "f_gpus_per_node": 8,
    "num_experts": 128,
    "topk": 8,
}


def _collectives(f_moe_tp_size):
    kw = dict(_COLLECTIVE_KW)
    if f_moe_tp_size is not None:
        kw["f_moe_tp_size"] = f_moe_tp_size
    return (
        AFDFAllGather(name="ag", scale_factor=1.0, **kw),
        AFDFReduceScatter(name="rs", scale_factor=1.0, **kw),
    )


# ---------------------------------------------------------------------------
# A. F-side AllGather / ReduceScatter TP-group gate
# ---------------------------------------------------------------------------


class TestFCollectiveTpGroupGate:
    def test_pure_ep_returns_zero(self, engine):
        """``f_moe_tp_size == 1``: no TP group, so nothing to exchange."""
        ag, rs = _collectives(1)
        assert float(ag.query(_DB, x=256)) == 0.0
        assert float(rs.query(_DB, x=256)) == 0.0
        assert engine.calls == [], "a zero-cost collective must not reach the engine"

    @pytest.mark.parametrize("tp", [2, 4, 8, 16])
    def test_with_a_tp_group_the_cost_stands(self, engine, tp):
        ag, rs = _collectives(tp)
        assert float(ag.query(_DB, x=256)) > 0.0
        assert float(rs.query(_DB, x=256)) > 0.0

    def test_unset_preserves_historical_behavior(self, engine):
        """``0`` means "caller did not say" and must not change any result.

        This is what keeps every pre-existing call site and test double intact;
        folding the default into 1 would silently zero their collectives.
        """
        unset_ag, unset_rs = _collectives(None)
        zero_ag, zero_rs = _collectives(0)
        tp_ag, tp_rs = _collectives(2)

        assert float(unset_ag.query(_DB, x=256)) == float(tp_ag.query(_DB, x=256))
        assert float(unset_rs.query(_DB, x=256)) == float(tp_rs.query(_DB, x=256))
        assert float(zero_ag.query(_DB, x=256)) == float(tp_ag.query(_DB, x=256))
        assert float(zero_rs.query(_DB, x=256)) == float(tp_rs.query(_DB, x=256))

    @pytest.mark.parametrize(
        ("tp", "expected"),
        [(0, True), (1, False), (2, True), (8, True), (-3, True)],
    )
    def test_has_tp_group_truth_table(self, tp, expected):
        ag, rs = _collectives(tp)
        assert ag.has_tp_group is expected
        assert rs.has_tp_group is expected

    def test_single_f_gpu_still_wins_regardless_of_tp_width(self, engine):
        """The pre-existing ``f_local <= 1`` gate keeps priority."""
        kw = dict(_COLLECTIVE_KW, n_f_workers=1, f_gpus_per_node=1, f_moe_tp_size=8)
        assert float(AFDFAllGather(name="ag", scale_factor=1.0, **kw).query(_DB, x=256)) == 0.0
        assert float(AFDFReduceScatter(name="rs", scale_factor=1.0, **kw).query(_DB, x=256)) == 0.0

    def test_broadcast_mapping_still_wins(self, engine):
        kw = dict(_COLLECTIVE_KW, f_moe_tp_size=8, rank_mapping="broadcast")
        assert float(AFDFAllGather(name="ag", scale_factor=1.0, **kw).query(_DB, x=256)) == 0.0

    def test_gate_does_not_change_sizing(self, engine):
        """Different TP widths above 1 must produce identical cost.

        The gate is about *existence*; the collective's participant count is
        still ``f_gpus_in_node``. If sizing ever starts depending on
        ``f_moe_tp_size`` this test fails and the docstring is wrong.
        """
        values = {float(_collectives(tp)[0].query(_DB, x=256)) for tp in (2, 4, 8, 16)}
        assert len(values) == 1


class TestAfdConfigFMoeTpSize:
    """``tp_f / EP``, derived so it follows the F pool under hetero A/F."""

    @staticmethod
    def _cfg(**kw):
        base = {"n_a_nodes": 2, "n_f_nodes": 2, "gpus_per_node": 8, "tp_a": 2, "a_batch_size": 8}
        base.update(kw)
        return AFDConfig(**base)

    @pytest.mark.parametrize(
        ("f_moe_ep_size", "expected"),
        [(16, 1), (8, 2), (4, 4), (2, 8), (1, 16)],
    )
    def test_derivation_from_tp_f_over_ep(self, f_moe_ep_size, expected):
        # n_f_nodes=2 x gpus_per_node=8 -> tp_f = n_f_workers = 16
        cfg = self._cfg(f_moe_ep_size=f_moe_ep_size)
        assert cfg.n_f_workers == 16
        assert cfg.f_moe_tp_size == expected

    def test_pure_ep_is_exactly_one(self):
        assert self._cfg(f_moe_ep_size=16).f_moe_tp_size == 1

    def test_follows_the_f_pool_node_width_under_hetero(self):
        """Hetero A/F: the F pool's own ``f_gpus_per_node`` drives tp_f."""
        cfg = self._cfg(gpus_per_node=4, a_gpus_per_node=4, f_gpus_per_node=8, tp_a=2, f_moe_ep_size=8)
        assert cfg.n_f_workers == 2 * 8  # F side, not the top-level 4
        assert cfg.f_moe_tp_size == 2

    def test_never_below_one(self):
        """EP wider than tp_f is nonsense but must not yield 0 (== falsy)."""
        cfg = self._cfg(n_f_nodes=1, gpus_per_node=8, f_moe_ep_size=64)
        assert cfg.f_moe_tp_size == 1


class TestSessionWiresTheGate:
    """Only the F-node collectives take the gate; the P2P legs must not."""

    def _comm_ops(self, f_moe_ep_size):
        from aiconfigurator.sdk import common
        from aiconfigurator.sdk.inference_session import AFDInferenceSession

        cfg = AFDConfig(
            n_a_nodes=2,
            n_f_nodes=1,
            gpus_per_node=8,
            tp_a=2,
            a_batch_size=8,
            f_moe_ep_size=f_moe_ep_size,
        )
        session = AFDInferenceSession.__new__(AFDInferenceSession)
        session._afd_config = cfg
        session._a_model_config = SimpleNamespace(comm_quant_mode=common.CommQuantMode.half)
        model = SimpleNamespace(_hidden_size=4096, _num_experts=128, _topk=8)
        return cfg, session._build_afd_comm_ops(model, model)

    def test_pure_ep_config_disables_the_collectives(self):
        cfg, ops = self._comm_ops(f_moe_ep_size=8)  # tp_f=8, EP=8 -> TP=1
        assert cfg.f_moe_tp_size == 1
        assert ops.f_ag.has_tp_group is False
        assert ops.f_rs.has_tp_group is False

    def test_tp_group_config_keeps_them(self):
        cfg, ops = self._comm_ops(f_moe_ep_size=4)  # tp_f=8, EP=4 -> TP=2
        assert cfg.f_moe_tp_size == 2
        assert ops.f_ag.has_tp_group is True
        assert ops.f_rs.has_tp_group is True

    def test_cross_pool_legs_do_not_accept_the_gate(self):
        """``AFDTransfer`` has no ``f_moe_tp_size``; passing it would TypeError.

        Guards against someone moving the argument into the ``shared`` dict,
        which feeds all four ops.
        """
        import inspect

        from aiconfigurator.sdk.operations import AFDTransfer

        assert "f_moe_tp_size" not in inspect.signature(AFDTransfer.__init__).parameters
        _cfg, ops = self._comm_ops(f_moe_ep_size=8)
        assert not hasattr(ops.a2f, "has_tp_group")
        assert not hasattr(ops.f2a, "has_tp_group")


# ---------------------------------------------------------------------------
# B. Nested MoE dispatch double-billing
# ---------------------------------------------------------------------------


def _named_op(name: str, weight: int = 1):
    """Minimal engine-backed leaf carrying only a name.

    ``_strip_moe_dispatch_from_partition`` classifies by name and rebuilds
    composites, so the double needs nothing else. Rust composites reject
    children that are not engine-backed, which rules out a hand-rolled
    ``Operation`` subclass -- the base has no Python constructor. ``weight``
    scales the op's token volume so a test can reason about relative cost.
    """
    return operations.ElementWise(name, 1.0, weight, 0)


def _partition(ffn_ops, skipped_ops=None):
    return AFDOpsPartition(
        phase="generation",
        attn_ops=[],
        ffn_ops=list(ffn_ops),
        boundary_ops=[],
        skipped_ops=list(skipped_ops or []),
    )


class TestIsMoeDispatchOp:
    @pytest.mark.parametrize(
        "name",
        ["generation_moe_dispatch", "GENERATION_DISPATCH", "pre_dispatch_gemm"],
    )
    def test_matches_dispatch_names(self, name):
        assert _is_moe_dispatch_op(_named_op(name)) is True

    @pytest.mark.parametrize("name", ["generation_moe_gemm", "generation_attention", "combine"])
    def test_rejects_others(self, name):
        assert _is_moe_dispatch_op(_named_op(name)) is False

    def test_tolerates_a_nameless_object(self):
        assert _is_moe_dispatch_op(object()) is False


class TestStripMoeDispatch:
    def test_strips_a_dispatch_nested_in_an_overlap_op(self):
        gemm_a, dispatch, gemm_b = _named_op("moe_gemm"), _named_op("moe_dispatch"), _named_op("shared_gemm")
        overlap = operations.OverlapOp("moe_overlap", [gemm_a, dispatch], [gemm_b])
        part = _strip_moe_dispatch_from_partition(_partition([overlap]))

        assert len(part.ffn_ops) == 1
        rebuilt = part.ffn_ops[0]
        assert isinstance(rebuilt, operations.OverlapOp)
        inner = list(rebuilt._group_a) + list(rebuilt._group_b)
        assert [o._name for o in inner] == ["moe_gemm", "shared_gemm"]

    def test_removing_the_dispatch_lowers_the_overlap_cost_by_its_own_share(self):
        """The delta must equal the dispatch's own cost, not something else.

        OverlapOp latency is ``max(sum(a), sum(b))``, so with the dispatch in
        the heavier group the drop is exactly its contribution. The leaves are
        engine-backed, so the reference is the standalone dispatch cost rather
        than a hand-picked latency constant.
        """
        db = _real_database()
        gemm, dispatch = _named_op("moe_gemm", 5), _named_op("moe_dispatch", 2)
        shared = _named_op("shared", 1)
        before = operations.OverlapOp("moe_overlap", [gemm, dispatch], [shared])
        before_cost = float(before._engine_query(db, x=64))

        part = _strip_moe_dispatch_from_partition(_partition([before]))
        after_cost = float(part.ffn_ops[0]._engine_query(db, x=64))

        dispatch_cost = float(dispatch._engine_query(db, x=64))
        # group_a is the heavier side both before and after, so the whole
        # delta is the dispatch leg.
        assert before_cost > after_cost
        assert before_cost - after_cost == pytest.approx(dispatch_cost)

    def test_bare_dispatch_at_top_level_is_removed_and_recorded(self):
        dispatch = _named_op("moe_dispatch")
        part = _strip_moe_dispatch_from_partition(_partition([dispatch, _named_op("moe_gemm")]))
        assert [o._name for o in part.ffn_ops] == ["moe_gemm"]
        assert dispatch in part.skipped_ops

    def test_an_all_dispatch_overlap_op_is_dropped_entirely(self):
        overlap = operations.OverlapOp("d_overlap", [_named_op("moe_dispatch")], [_named_op("moe_dispatch_2")])
        part = _strip_moe_dispatch_from_partition(_partition([overlap]))
        assert part.ffn_ops == []

    def test_recursion_reaches_a_nested_overlap_op(self):
        inner = operations.OverlapOp("inner", [_named_op("moe_dispatch")], [_named_op("moe_gemm")])
        outer = operations.OverlapOp("outer", [inner], [_named_op("shared_gemm")])
        part = _strip_moe_dispatch_from_partition(_partition([outer]))

        rebuilt_inner = part.ffn_ops[0]._group_a[0]
        names = [o._name for o in list(rebuilt_inner._group_a) + list(rebuilt_inner._group_b)]
        assert names == ["moe_gemm"]

    def test_a_partition_without_dispatch_is_left_alone(self):
        """No dispatch anywhere in the subtree means no rebuild.

        Object identity is not usable as the probe: reading ``_group_a`` off a
        composite returns engine-rebuilt children every time, so even an
        untouched tree fails an ``is`` check. What must hold is that the
        top-level list entries are the very objects handed in, and that nothing
        was recorded as skipped.
        """
        gemm = _named_op("moe_gemm")
        overlap = operations.OverlapOp("ov", [_named_op("a")], [_named_op("b")])
        part = _strip_moe_dispatch_from_partition(_partition([gemm, overlap]))

        assert len(part.ffn_ops) == 2
        assert part.ffn_ops[0] is gemm
        assert part.ffn_ops[1] is overlap
        assert part.skipped_ops == []

    def test_existing_skipped_ops_are_preserved(self):
        already = _named_op("generation_ar_1")
        dispatch = _named_op("moe_dispatch")
        part = _strip_moe_dispatch_from_partition(_partition([dispatch], skipped_ops=[already]))
        assert already in part.skipped_ops
        assert dispatch in part.skipped_ops

    def test_attn_ops_are_untouched(self):
        """Only ``ffn_ops`` feeds F-pool compute, so only it is rewritten."""
        part = AFDOpsPartition(
            phase="generation",
            attn_ops=[_named_op("moe_dispatch")],
            ffn_ops=[],
            boundary_ops=[],
            skipped_ops=[],
        )
        result = _strip_moe_dispatch_from_partition(part)
        assert [o._name for o in result.attn_ops] == ["moe_dispatch"]


# ---------------------------------------------------------------------------
# C. Scope on real models -- the fix must land exactly where the nesting is
# ---------------------------------------------------------------------------


def _nested_dispatch_names(partition):
    """Names of dispatch ops reachable *inside* an OverlapOp in ``ffn_ops``."""

    def walk(op, depth=0):
        yield depth, op
        if isinstance(op, operations.OverlapOp):
            for inner in list(op._group_a) + list(op._group_b):
                yield from walk(inner, depth + 1)

    return [o._name for top in partition.ffn_ops for depth, o in walk(top) if depth > 0 and _is_moe_dispatch_op(o)]


def _generation_partition(model_path, *, tp=8, ep=8):
    from aiconfigurator.sdk import config as cfgmod
    from aiconfigurator.sdk.afd_partition import build_afd_ops_partition
    from aiconfigurator.sdk.models import get_model

    model_config = cfgmod.ModelConfig(tp_size=tp, moe_tp_size=tp // ep, moe_ep_size=ep)
    model = get_model(model_path, model_config, "sglang")
    return build_afd_ops_partition(model, phase="generation")


class TestRealModelScope:
    """The nesting is model-dependent, and so is the fix's reach.

    ``deepseek*`` / ``minimax_m3`` wrap the routed MoE ops (dispatch included)
    in an ``OverlapOp`` unconditionally; ``qwen35`` only does so when the model
    has shared experts. Qwen3-235B-A22B has none, so its ops stay flat and the
    partitioner's skip list already removed the dispatch -- nothing left to
    strip. Asserting both directions is what separates "the fix works" from
    "the fix fires everywhere".

    Both cases need the HF config, so they share the same environment
    dependency as ``tests/unit/sdk/models/`` -- skipped rather than failed when
    the config cannot be fetched.
    """

    @staticmethod
    def _partition_or_skip(model_path):
        try:
            return _generation_partition(model_path)
        except Exception as exc:  # environment dependency, not a logic failure
            if "Download" in type(exc).__name__ or "download" in str(exc).lower():
                pytest.skip(f"{model_path}: HF config unavailable ({type(exc).__name__})")
            raise

    def test_deepseek_v3_has_a_nested_dispatch_that_gets_stripped(self):
        """Shared experts -> OverlapOp -> the dispatch was billed as F compute."""
        before = self._partition_or_skip("deepseek-ai/DeepSeek-V3")
        assert _nested_dispatch_names(before), "expected a nested dispatch to strip"

        after = _strip_moe_dispatch_from_partition(self._partition_or_skip("deepseek-ai/DeepSeek-V3"))
        assert _nested_dispatch_names(after) == []
        assert len(after.skipped_ops) >= len(before.skipped_ops)

    def test_qwen3_235b_has_no_nested_dispatch_and_is_untouched(self):
        """No shared expert -> flat ops -> the skip list already handled it.

        ``ffn_ops`` must come back element-identical: if this ever starts
        rebuilding, the strip is reaching further than the defect it fixes.
        """
        before = self._partition_or_skip("Qwen/Qwen3-235B-A22B")
        assert _nested_dispatch_names(before) == []
        original_ops = list(before.ffn_ops)
        original_skipped = list(before.skipped_ops)

        after = _strip_moe_dispatch_from_partition(before)
        assert len(after.ffn_ops) == len(original_ops)
        assert all(new is old for new, old in zip(after.ffn_ops, original_ops, strict=True))
        assert list(after.skipped_ops) == original_skipped

    def test_the_partitioner_already_excludes_a_bare_dispatch(self):
        """Spec item: a top-level dispatch is never re-recorded by the strip.

        The skip list keeps bare ``MoEDispatch`` out of ``ffn_ops`` in the first
        place, so ``_strip`` finds nothing at the top level and adds nothing to
        ``skipped_ops``. The only thing it can remove is the nested copy.
        """
        part = self._partition_or_skip("Qwen/Qwen3-235B-A22B")
        assert not any(_is_moe_dispatch_op(op) for op in part.ffn_ops), (
            "a bare dispatch reached ffn_ops -- the skip list regressed"
        )
        skipped_before = len(part.skipped_ops)
        _strip_moe_dispatch_from_partition(part)
        assert len(part.skipped_ops) == skipped_before

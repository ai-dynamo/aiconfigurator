# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Super-node (NVL72-class) fabric tiering for the AFD cross-pool transfer.

A rack-scale system has three fabric tiers, not one: NVLink inside a node,
NVSwitch inside the rack, and a scale-out fabric between racks.
``SystemSpec.get_p2p_bandwidth`` already selected among them, but nothing on
the AFD path reached it -- ``P2P._query_p2p_table`` hardcoded
``inter_node_bw``, so a topology that left the scale-up domain was priced as
if it had not. ``inter_rack_latency`` was declared in gb200/gb300 and read by
nobody.

Covered here:

1. Both tier selectors, at the boundaries, including the no-rack-tier
   fallback.
2. ``inter_rack_latency`` actually reaching the latency term -- a guard
   against it silently becoming a dead field again.
3. ``num_gpus=None`` preserving the legacy flat pricing bit-for-bit, which is
   what every pipeline-parallel caller relies on.
4. The AFD session computing the A+F span and only handing it to the
   cross-pool legs.
"""

from __future__ import annotations

import logging
import pathlib
from types import SimpleNamespace
from typing import ClassVar

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import AFDTransfer
from aiconfigurator.sdk.performance_result import PerformanceResult
from aiconfigurator_core.sdk import system_spec as system_spec_module
from aiconfigurator_core.sdk.operations import afd_transfer as afd_transfer_module
from aiconfigurator_core.sdk.system_spec import SystemSpec

pytestmark = pytest.mark.unit


# Mirrors gb200.yaml's topology: 4 GPUs/node, 72/rack, NVLink inside the rack
# and a 9x slower scale-out fabric between racks.
#
# The latencies here are deliberately NOT copied from gb200.yaml. That file pairs
# p2p_latency 10us (raised by hand as a "nonofficial correction") with
# inter_rack_latency 5us (textbook InfiniBand), which makes crossing racks look
# faster. Tier selection has nothing to do with that data defect, so this fixture
# states a consistent pair and ``_INVERTED_RACK_NODE_SPEC`` covers the clamp.
_RACK_NODE_SPEC = {
    "num_gpus_per_node": 4,
    "num_gpus_per_rack": 72,
    "intra_node_bw": 900e9,
    "inter_node_bw": 900e9,
    "inter_rack_bw": 100e9,
    "p2p_latency": 10e-6,
    "inter_rack_latency": 30e-6,
}

# What gb200/gb300 actually ship: cross-rack latency below same-rack.
_INVERTED_RACK_NODE_SPEC = dict(_RACK_NODE_SPEC, inter_rack_latency=5e-6)

# Mirrors h200_sxm.yaml: 8 GPUs/node and no rack tier at all.
_FLAT_NODE_SPEC = {
    "num_gpus_per_node": 8,
    "intra_node_bw": 450e9,
    "inter_node_bw": 50e9,
    "p2p_latency": 10e-6,
}


def _spec(node_spec: dict) -> SystemSpec:
    return SystemSpec({"node": dict(node_spec)})


class _SpecDatabase:
    """A node spec plus the two tier selectors, for spec-level assertions.

    The P2P analytic formula now lives in the compiled engine, so this class no
    longer models pricing -- it only carries a synthetic node spec so the tier
    *selection* can be asserted without perf data on disk. Anything that needs
    an actual latency goes through ``_tiered_latency`` below.
    """

    def __init__(self, node_spec: dict, mode: common.DatabaseMode = common.DatabaseMode.EMPIRICAL) -> None:
        self.system_spec = _spec(node_spec)
        self._default_database_mode = mode

    def _get_p2p_bandwidth(self, num_gpus: int) -> float:
        return self.system_spec.get_p2p_bandwidth(num_gpus)

    def _get_p2p_latency(self, num_gpus: int) -> float:
        return self.system_spec.get_p2p_latency(num_gpus)


def _tiered_latency(node_spec: dict, message_bytes: int, span: int | None, *, sol: bool = False) -> float:
    """The engine's P2P formula evaluated against ``node_spec``.

    Mirrors the Rust ``P2POp::query`` shape: ``span=None`` keeps flat
    inter-node pricing, otherwise both terms come from the tier selectors.
    Used to state the *expected* value in tests -- the engine itself is
    exercised end to end by the real-database cases.
    """
    spec = _spec(node_spec)
    if span is None:
        bw, lat = node_spec["inter_node_bw"], node_spec["p2p_latency"]
    else:
        bw, lat = spec.get_p2p_bandwidth(span), spec.get_p2p_latency(span)
    return (message_bytes / max(bw, 1.0) + (0.0 if sol else lat)) * 1000.0


class TestBandwidthTierSelection:
    @pytest.mark.parametrize(
        ("num_gpus", "expected"),
        [
            (1, 900e9),  # inside a node
            (4, 900e9),  # exactly one node
            (5, 900e9),  # crosses nodes, still inside the rack
            (72, 900e9),  # exactly one rack
            (73, 100e9),  # first GPU past the rack
            (144, 100e9),  # two racks
        ],
    )
    def test_bandwidth_tiers_at_boundaries(self, num_gpus, expected):
        assert _spec(_RACK_NODE_SPEC).get_p2p_bandwidth(num_gpus) == expected

    @pytest.mark.parametrize(
        ("num_gpus", "expected"),
        [
            (1, 450e9),
            (8, 450e9),
            (9, 50e9),
            (1000, 50e9),  # no rack tier -> never escalates past inter_node
        ],
    )
    def test_bandwidth_without_rack_tier_never_escalates(self, num_gpus, expected):
        assert _spec(_FLAT_NODE_SPEC).get_p2p_bandwidth(num_gpus) == expected


class TestLatencyTierSelection:
    @pytest.mark.parametrize("num_gpus", [1, 4, 5, 72])
    def test_within_rack_uses_p2p_latency(self, num_gpus):
        assert _spec(_RACK_NODE_SPEC).get_p2p_latency(num_gpus) == 10e-6

    @pytest.mark.parametrize("num_gpus", [73, 144])
    def test_across_racks_uses_inter_rack_latency(self, num_gpus):
        assert _spec(_RACK_NODE_SPEC).get_p2p_latency(num_gpus) == 30e-6

    @pytest.mark.parametrize("num_gpus", [1, 8, 9, 1000])
    def test_without_rack_tier_falls_back_to_p2p_latency(self, num_gpus):
        assert _spec(_FLAT_NODE_SPEC).get_p2p_latency(num_gpus) == 10e-6

    def test_missing_inter_rack_latency_falls_back(self):
        node_spec = {k: v for k, v in _RACK_NODE_SPEC.items() if k != "inter_rack_latency"}
        # Rack tier declared but no latency for it: keep p2p_latency rather
        # than inventing a number.
        assert _spec(node_spec).get_p2p_latency(1000) == 10e-6

    def test_inter_rack_latency_is_actually_consumed(self):
        """Guard against ``inter_rack_latency`` regressing to a dead field.

        Changing only that key must move the cross-rack latency and leave the
        within-rack latency alone.
        """
        # Still above the fixture's 30us, so the clamp cannot mask the change.
        slow = dict(_RACK_NODE_SPEC, inter_rack_latency=500e-6)

        assert _tiered_latency(slow, 1024, 144) > _tiered_latency(_RACK_NODE_SPEC, 1024, 144)
        assert _tiered_latency(slow, 1024, 72) == _tiered_latency(_RACK_NODE_SPEC, 1024, 72)


class TestFlatPricingPreserved:
    """``num_gpus=None`` must reproduce the pre-tiering formula exactly."""

    @pytest.mark.parametrize("message_bytes", [1, 1024, 1 << 20, 1 << 24])
    def test_none_span_matches_inter_node_formula(self, message_bytes):
        expected = (message_bytes / _RACK_NODE_SPEC["inter_node_bw"] + _RACK_NODE_SPEC["p2p_latency"]) * 1000
        assert _tiered_latency(_RACK_NODE_SPEC, message_bytes, None) == pytest.approx(expected, rel=1e-12)

    def test_none_span_ignores_rack_tier(self):
        """A 144-GPU payload priced without a span stays on ``inter_node_bw``.

        This is why the span is opt-in: pipeline-parallel P2P moves between
        two adjacent ranks, so the deployment size is not its span.
        """
        flat = _tiered_latency(_RACK_NODE_SPEC, 1 << 20, None)
        assert flat == _tiered_latency(_RACK_NODE_SPEC, 1 << 20, 72)
        assert flat != _tiered_latency(_RACK_NODE_SPEC, 1 << 20, 144)

    def test_sol_mode_also_honors_the_span(self):
        within = _tiered_latency(_RACK_NODE_SPEC, 1 << 24, 72, sol=True)
        across = _tiered_latency(_RACK_NODE_SPEC, 1 << 24, 144, sol=True)
        # SOL drops the latency constant, so the 9x bandwidth gap is exact.
        assert across == pytest.approx(within * 9.0, rel=1e-9)


class _SpanRecordingEngine:
    """Stands in for ``_engine_comm_query``, recording the probe's span.

    The tier now lives on the probe op the AFD leg builds, so what a wiring
    test must observe is ``P2P.span_gpus`` on that probe -- not an argument to
    a database method (the analytic formula moved into the engine, and
    ``query_p2p`` no longer exists). Latency is keyed on the database object so
    a bottleneck test can still make the two sides disagree.
    """

    def __init__(self, factors: dict | None = None) -> None:
        self.spans: list[int | None] = []
        self.databases: list = []
        self._factors = factors or {}

    def __call__(self, database, op):
        self.spans.append(op._span_gpus if hasattr(op, "_span_gpus") else _probe_span(op))
        self.databases.append(database)
        volume = int(op._h) * 2
        return PerformanceResult(latency=float(volume) * self._factors.get(id(database), 1.0), energy=0.0)


def _probe_span(op) -> int | None:
    """The span recorded on a Rust-backed ``P2P`` probe."""
    import json

    spec = json.loads(op._spec_json())
    inner = spec.get("P2P") or next(iter(spec.values()))
    return inner.get("span_gpus")


class TestAFDTransferSpan:
    _KW: ClassVar[dict] = {
        "hidden_size": 4096,
        "n_a_workers": 12,
        "n_f_workers": 4,
        "gpus_per_node": 4,
        "f_gpus_per_node": 4,
        "num_experts": 128,
        "topk": 8,
    }

    @pytest.fixture()
    def engine(self, monkeypatch) -> _SpanRecordingEngine:
        stub = _SpanRecordingEngine()
        monkeypatch.setattr(afd_transfer_module, "_engine_comm_query", stub)
        return stub

    def test_span_is_forwarded_to_the_probe(self, engine):
        op = AFDTransfer(name="a2f", scale_factor=1.0, direction="a2f", span_gpus=72, **self._KW)
        op.query(object(), x=64)
        assert engine.spans == [72]

    def test_no_span_leaves_the_probe_untiered(self, engine):
        """An unset span must reach the engine as ``None``, not as a number.

        The engine keys flat-vs-tiered pricing off exactly that, so a probe
        that invented a span would silently reprice every pipeline-parallel
        caller.
        """
        op = AFDTransfer(name="a2f", scale_factor=1.0, direction="a2f", **self._KW)
        assert op.span_gpus is None
        assert float(op.query(object(), x=64)) > 0.0
        assert engine.spans == [None]

    def test_span_reaches_both_sides_under_hetero(self, engine):
        """Bottleneck pricing and tiering compose: both sides get the span."""
        a_db, f_db = object(), object()
        op = AFDTransfer(name="a2f", scale_factor=1.0, direction="a2f", span_gpus=144, **self._KW)
        op.query(a_db, x=64, peer_database=f_db)
        assert engine.spans == [144, 144]
        assert engine.databases == [a_db, f_db]

    @pytest.mark.parametrize("span", [0, None])
    def test_falsy_span_means_flat_pricing(self, span):
        op = AFDTransfer(name="a2f", scale_factor=1.0, direction="a2f", span_gpus=span, **self._KW)
        assert op.span_gpus is None


class TestHeteroTimesSupernode:
    """The two features are orthogonal and must compose.

    Rack width is a per-system hardware fact, so under hetero A/F the two
    sides can resolve the *same* span to *different* tiers. Bottleneck pricing
    then has to pick the slower of the two tier-resolved latencies, not the A
    side's.

    The engine owns the formula now, so these tests drive it through the stub
    with a per-database factor standing in for "this side's fabric is slower".
    Tier *selection* itself is covered by TestBandwidthTierSelection /
    TestLatencyTierSelection against the real spec helpers.
    """

    _KW: ClassVar[dict] = {
        "hidden_size": 4096,
        "n_a_workers": 68,
        "n_f_workers": 4,
        "gpus_per_node": 4,
        "f_gpus_per_node": 8,
        "num_experts": 128,
        "topk": 8,
    }

    A_DB = object()
    F_DB = object()

    @pytest.fixture()
    def engine(self, monkeypatch) -> _SpanRecordingEngine:
        # F is the slower fabric: at span 144 the A side escalates to its
        # inter-rack link while F, having no rack tier, stays on its slower
        # inter-node link.
        stub = _SpanRecordingEngine({id(self.A_DB): 1.0, id(self.F_DB): 2.0})
        monkeypatch.setattr(afd_transfer_module, "_engine_comm_query", stub)
        return stub

    def _op(self, span):
        return AFDTransfer(name="a2f", scale_factor=1.0, direction="a2f", span_gpus=span, **self._KW)

    def test_tier_selection_differs_per_side_at_the_same_span(self):
        """A=rack-aware and F=flat resolve span 144 to different tiers.

        Each side reads its own spec, so the same span lands on different
        bandwidth *and* different latency. Asserting an ordering between the two
        totals would depend on payload -- and the previous version of this test
        only got one because the rack fixture carried an inverted cross-rack
        latency. Pin the per-side tier resolution instead.
        """
        a_db = _SpecDatabase(_RACK_NODE_SPEC)
        f_db = _SpecDatabase(_FLAT_NODE_SPEC)
        assert a_db._get_p2p_bandwidth(144) == 100e9  # rack tier -> inter_rack
        assert f_db._get_p2p_bandwidth(144) == 50e9  # no rack tier -> inter_node

        # Latency follows the same split: the rack-aware side pays its declared
        # cross-rack figure, the flat side stays on p2p_latency.
        assert _spec(_RACK_NODE_SPEC).get_p2p_latency(144) == _RACK_NODE_SPEC["inter_rack_latency"]
        assert _spec(_FLAT_NODE_SPEC).get_p2p_latency(144) == _FLAT_NODE_SPEC["p2p_latency"]

    def test_slower_side_sets_the_price(self, engine):
        op = self._op(144)
        priced = float(op.query(self.A_DB, x=1024, peer_database=self.F_DB))
        a_only = float(op.query(self.A_DB, x=1024))
        f_only = float(op.query(self.F_DB, x=1024))

        assert priced == pytest.approx(max(a_only, f_only))
        assert priced == pytest.approx(f_only)  # F is the bottleneck here
        assert priced > a_only

    def test_bottleneck_can_be_either_side(self, engine):
        """Symmetry: swapping which DB is primary must not change the price."""
        op = self._op(144)
        forward = float(op.query(self.A_DB, x=1024, peer_database=self.F_DB))
        reverse = float(op.query(self.F_DB, x=1024, peer_database=self.A_DB))
        assert forward == pytest.approx(reverse)

    def test_both_sides_are_priced_at_the_same_span(self, engine):
        """Tiering is not dropped on the peer leg."""
        self._op(144).query(self.A_DB, x=1024, peer_database=self.F_DB)
        assert engine.spans == [144, 144]

    def test_homogeneous_span_pricing_is_unaffected_by_a_same_object_peer(self, engine):
        """Passing the same DB as peer is a no-op, span or not."""
        op = self._op(144)
        assert float(op.query(self.A_DB, x=1024, peer_database=self.A_DB)) == float(op.query(self.A_DB, x=1024))


class TestSessionSpanWiring:
    """The session owns span derivation; the op only carries it."""

    def _comm_ops(self, *, n_a_nodes, n_f_nodes, a_gpus_per_node=4, f_gpus_per_node=4):
        from aiconfigurator.sdk.config import AFDConfig
        from aiconfigurator.sdk.inference_session import AFDInferenceSession

        cfg = AFDConfig(
            n_a_nodes=n_a_nodes,
            n_f_nodes=n_f_nodes,
            gpus_per_node=a_gpus_per_node,
            a_gpus_per_node=a_gpus_per_node,
            f_gpus_per_node=f_gpus_per_node,
            tp_a=1,
            f_moe_ep_size=1,
        )
        session = AFDInferenceSession.__new__(AFDInferenceSession)
        session._afd_config = cfg
        session._a_model_config = SimpleNamespace(comm_quant_mode=common.CommQuantMode.half)
        model = SimpleNamespace(_hidden_size=4096, _num_experts=128, _topk=8)
        return session._build_afd_comm_ops(model, model)

    @pytest.mark.parametrize(
        ("n_a_nodes", "n_f_nodes", "expected_span"),
        [
            (3, 1, 16),  # 4 nodes x 4 GPUs -- comfortably inside one rack
            (17, 1, 72),  # exactly the NVL72 domain, FastAFD's largest ratio
            (18, 1, 76),  # one node past it -- now a cross-rack deployment
        ],
    )
    def test_span_is_total_a_plus_f_gpus(self, n_a_nodes, n_f_nodes, expected_span):
        ops = self._comm_ops(n_a_nodes=n_a_nodes, n_f_nodes=n_f_nodes)
        assert ops.a2f.span_gpus == expected_span
        assert ops.f2a.span_gpus == expected_span

    def test_span_uses_per_pool_node_widths(self):
        """Under hetero A/F the two pools can have different node widths."""
        ops = self._comm_ops(n_a_nodes=2, n_f_nodes=1, a_gpus_per_node=4, f_gpus_per_node=8)
        assert ops.a2f.span_gpus == 2 * 4 + 1 * 8

    def test_only_cross_pool_legs_carry_a_span(self):
        """F-side AG/RS is intra-node and a_combine is a local HBM reduce.

        Neither can cross a rack, so neither takes a span -- giving them one
        would price a node-local collective on the scale-out fabric.
        """
        ops = self._comm_ops(n_a_nodes=17, n_f_nodes=1)
        for op in (ops.f_ag, ops.f_rs, ops.a_combine):
            assert not hasattr(op, "span_gpus")


# ---------------------------------------------------------------------------
# The compiled engine actually honors the span
# ---------------------------------------------------------------------------

_ENGINE_SYSTEM, _ENGINE_BACKEND, _ENGINE_VERSION = "gb200", "sglang", "0.5.16"


@pytest.fixture(scope="module")
def rack_database():
    """A real rack-tiered ``PerfDatabase`` (gb200: 4 GPUs/node, 72/rack)."""
    from aiconfigurator.sdk.perf_database import get_database

    db = get_database(_ENGINE_SYSTEM, _ENGINE_BACKEND, _ENGINE_VERSION)
    if db is None:
        pytest.skip(f"{_ENGINE_SYSTEM}/{_ENGINE_BACKEND}/{_ENGINE_VERSION} data missing")
    node = db.system_spec["node"]
    if not node.get("num_gpus_per_rack"):
        pytest.skip(f"{_ENGINE_SYSTEM} declares no rack tier")
    return db


def _engine_p2p(database, message_bytes: int, span: int | None) -> float:
    """One P2P probe through the real engine, exactly as ``AFDTransfer`` does."""
    from aiconfigurator_core.sdk.operations.communication import P2P

    probe = P2P("probe", 1.0, -(-message_bytes // 2), 2, span_gpus=span)
    return float(afd_transfer_module._engine_comm_query(database, probe))


class TestEngineHonorsTheSpan:
    """End-to-end: the tier selection lives in Rust, so assert against it.

    Every other test here states the *expected* latency with a Python mirror of
    the formula (``_tiered_latency``). That cannot catch the engine drifting
    away from the spec -- it would move in lockstep with the mirror. These
    cases go through ``_engine_comm_query`` on a real database and compare with
    the values the system spec implies, so a Rust-side regression in
    ``P2POp::query`` (or a ``span_gpus`` that stops reaching it) fails here.
    """

    _BYTES = 1 << 20

    def test_within_rack_matches_flat_pricing(self, rack_database):
        """Inside the rack the span must not change the price.

        The intra-rack tier IS ``inter_node_bw`` + ``p2p_latency`` -- the same
        pair flat pricing uses -- so this equality is exact by construction,
        not a property of gb200's particular numbers.
        """
        per_rack = rack_database.system_spec["node"]["num_gpus_per_rack"]
        flat = _engine_p2p(rack_database, self._BYTES, None)
        assert _engine_p2p(rack_database, self._BYTES, per_rack) == pytest.approx(flat, rel=1e-12)

    def test_crossing_the_rack_repricing_matches_the_spec(self, rack_database):
        """Past the rack the engine must switch to the inter-rack pair."""
        node = rack_database.system_spec["node"]
        per_rack = node["num_gpus_per_rack"]
        inter_rack_bw = node.get("inter_rack_bw")
        if not inter_rack_bw or inter_rack_bw == node["inter_node_bw"]:
            pytest.skip(f"{_ENGINE_SYSTEM} has no distinct inter_rack_bw")

        # Derive the latency through ``get_p2p_latency`` rather than reading
        # ``inter_rack_latency`` directly: that method owns the tier definition,
        # including the clamp that stops a spec with an inverted pair from
        # reporting a cross-rack speedup. Reading the raw key would restate the
        # rule and diverge from it the moment the rule changes.
        latency = rack_database.system_spec.get_p2p_latency(per_rack + 1)
        # ``AFDTransfer`` rounds the payload to bf16 elements, so mirror that.
        expected = ((-(-self._BYTES // 2)) * 2 / inter_rack_bw + latency) * 1000.0

        across = _engine_p2p(rack_database, self._BYTES, per_rack + 1)
        assert across == pytest.approx(expected, rel=1e-9)
        assert across > _engine_p2p(rack_database, self._BYTES, per_rack)

    def test_unset_span_reproduces_the_pre_tiering_formula(self, rack_database):
        """The default path must be untouched by the tiering work.

        A rack-tiered system is the strictest case: if the engine ever started
        inferring a span, this is where it would show.
        """
        node = rack_database.system_spec["node"]
        expected = ((-(-self._BYTES // 2)) * 2 / node["inter_node_bw"] + node["p2p_latency"]) * 1000.0
        assert _engine_p2p(rack_database, self._BYTES, None) == pytest.approx(expected, rel=1e-9)

    def test_afd_transfer_pricing_reacts_to_its_own_span(self, rack_database):
        """The whole chain: AFDTransfer -> probe -> engine tier selection."""
        per_rack = rack_database.system_spec["node"]["num_gpus_per_rack"]
        kw = dict(
            hidden_size=4096,
            n_a_workers=8,
            n_f_workers=8,
            gpus_per_node=4,
            f_gpus_per_node=4,
            num_experts=0,
            topk=0,
        )

        def priced(span):
            op = AFDTransfer(name="a2f", scale_factor=1.0, direction="a2f", span_gpus=span, **kw)
            return float(op.query(rack_database, x=1024))

        within, across = priced(per_rack), priced(per_rack + 1)
        node = rack_database.system_spec["node"]
        if node.get("inter_rack_bw") in (None, node["inter_node_bw"]):
            pytest.skip(f"{_ENGINE_SYSTEM} has no distinct inter_rack_bw")
        assert across > within
        assert priced(None) == pytest.approx(within, rel=1e-12)


class TestCrossRackLatencyCannotBeatSameRack:
    """`inter_rack_latency` below `p2p_latency` must never report a speedup.

    Nothing read `inter_rack_latency` before the tiering landed, so a spec could
    carry an inverted pair without consequence -- and gb200/gb300 do, because
    their `p2p_latency` was raised by hand as a calibration knob while
    `inter_rack_latency` kept the textbook InfiniBand value.
    """

    @staticmethod
    def _spec(p2p, inter_rack, per_rack=72):
        """Build a rack-tiered spec with an explicit latency pair.

        Derived from ``_RACK_NODE_SPEC`` so the topology stays in one place and
        only the two latencies under test vary.
        """
        node = dict(_RACK_NODE_SPEC, num_gpus_per_rack=per_rack, p2p_latency=p2p)
        if inter_rack is None:
            node.pop("inter_rack_latency", None)
        else:
            node["inter_rack_latency"] = inter_rack
        return SystemSpec({"node": node})

    def test_inverted_spec_is_clamped_to_p2p(self):
        spec = self._spec(p2p=10e-6, inter_rack=5e-6)
        assert spec.get_p2p_latency(76) == pytest.approx(10e-6)

    def test_inverted_spec_warns(self, caplog):
        spec = self._spec(p2p=10e-6, inter_rack=5e-6)
        with caplog.at_level(logging.WARNING):
            spec.get_p2p_latency(76)
        assert "inter_rack_latency" in caplog.text
        assert "clamping" in caplog.text

    def test_a_consistent_spec_is_left_alone(self):
        """Guards the opposite error: a clamp that swallows the real value."""
        spec = self._spec(p2p=10e-6, inter_rack=20e-6)
        assert spec.get_p2p_latency(76) == pytest.approx(20e-6)

    def test_a_consistent_spec_does_not_warn(self, caplog):
        spec = self._spec(p2p=10e-6, inter_rack=20e-6)
        with caplog.at_level(logging.WARNING):
            spec.get_p2p_latency(76)
        assert "inter_rack_latency" not in caplog.text

    def test_equal_values_are_not_treated_as_inverted(self):
        spec = self._spec(p2p=10e-6, inter_rack=10e-6)
        assert spec.get_p2p_latency(76) == pytest.approx(10e-6)

    def test_same_rack_is_unaffected_by_the_guard(self):
        spec = self._spec(p2p=10e-6, inter_rack=5e-6)
        assert spec.get_p2p_latency(72) == pytest.approx(10e-6)

    def test_missing_inter_rack_falls_back(self):
        spec = self._spec(p2p=10e-6, inter_rack=None)
        assert spec.get_p2p_latency(76) == pytest.approx(10e-6)

    def test_latency_is_monotonic_across_the_rack_boundary_on_shipped_specs(self):
        """The property that matters, asserted on the real yaml rather than a
        hand-built dict: going wider may cost more, never less."""
        for name in ("gb200", "gb300"):
            spec = _load_shipped_spec(name)
            per_rack = spec["node"]["num_gpus_per_rack"]
            inside = spec.get_p2p_latency(per_rack)
            outside = spec.get_p2p_latency(per_rack * 2)
            assert outside >= inside, f"{name}: cross-rack {outside} < same-rack {inside}"


def _load_shipped_spec(name):
    """Load a system yaml the way ``PerfDatabase`` does."""
    import yaml

    root = pathlib.Path(system_spec_module.__file__).resolve().parents[1] / "systems"
    with open(root / f"{name}.yaml", encoding="utf-8") as handle:
        return SystemSpec(yaml.load(handle, Loader=yaml.SafeLoader))

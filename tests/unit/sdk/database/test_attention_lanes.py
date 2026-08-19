# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the attention kernel-lane precedence machinery (AIC-1715/1716).

The per-call Python query stack that used to walk these tables directly
retired with the deprecation-cleanup PR (#1357 PR-5 / Phase 3 PR7); per-op
attention VALUES now come only from the compiled Rust engine, anchored by the
frozen parity goldens (single-oracle rule, see
``.claude/rules/rust-core/parity.md``). What stays Python-owned, and is
tested here, is lane PRECEDENCE RESOLUTION: given a database identity
(backend/version/sm_version), an optional ``attention_backend`` override, and
a loaded table's own kernel_source lanes, compute the concrete walk order
(override -> framework default -> remaining known lanes -> ``"default"`` ->
the table's own leftover lanes, donor tiers ranked by measured coverage).
That order is serialized verbatim onto the op spec and the Rust engine
replays it without any lane-vocabulary knowledge of its own.

Rebase-4 review (AIC-1715/1716, Blocker 1): the donor/leftover density tiers
were inert in production because ``resolved_lane_order_for_op`` read density
off ``database._context_attention_data`` / ``_generation_attention_data`` --
the lane-BLIND ``table_view`` rehydration (enum-keyed top level, first-lane-
wins), never the real by-lane query structure. ``_StubDatabase`` below
mirrors that real collapsed shape (instead of a lane-keyed dict that made the
old tests blind to exactly this bug), and ``resolved_lane_order_for_op``'s
density fetch is routed to the stub's own ``lane_density`` through the same
``fetch_attention_lane_density`` seam production uses (see
``_route_lane_density_through_the_stub`` below) -- so the tests exercise the
real wiring, not just ``lane_walk_order``'s ranking math in isolation.
"""

import copy
import pickle

import pytest

from aiconfigurator.sdk.common import FMHAQuantMode, KVCacheQuantMode

pytestmark = pytest.mark.unit

# Fixed physical-key components used across tests
QM = FMHAQuantMode.bfloat16
KCD = KVCacheQuantMode.bfloat16
KV_N = 0  # n == kv_n → normalised to 0 by loader
HEAD = 128
WIN = 0
N = 8
B = 1

# ---------------------------------------------------------------------------
# Lane-aware query paths (AIC-1715 Task 3)
# ---------------------------------------------------------------------------

_LANE_DEFAULTS_YAML = """\
# Test fixture: minimal copy of attention_lane_defaults.yaml
sglang:
  "0.5.14":
    90: fa3
    100: triton
    103: triton
    120: flashinfer
"""

# The two lanes carry the SAME grid with latencies differing by a constant
# factor, so any homogeneous interpolation reproduces the ratio exactly.
_SLOW_LATENCY = 1.0
_FAST_LATENCY = 0.25
_LANE_RATIO = _FAST_LATENCY / _SLOW_LATENCY

_CTX_DEPTH = 5  # [fmha][kv][kv_n][head][window]
_GEN_DEPTH = 4  # [kv][kv_n][head][window]

_GRID_N = (8, 16)
_GRID_S = (32, 64, 128)
_GRID_B = (1, 2)


@pytest.fixture
def lane_systems_root(tmp_path):
    """systems_root holding the framework-default lane map (sglang 0.5.14)."""
    root = tmp_path / "systems"
    root.mkdir()
    (root / "attention_lane_defaults.yaml").write_text(_LANE_DEFAULTS_YAML, encoding="utf-8")
    return str(root)


class _LoadedLanes(dict):
    """Minimal stand-in for ``LoadedOpData``: a mapping that is always loaded."""

    loaded = True

    def raise_if_not_loaded(self):
        return None


def _leaf(latency):
    return {"latency": latency, "power": 0.0, "energy": 0.0}


def _ctx_lane(latency, head_size=HEAD):
    """[fmha][kv][kv_n][head][window][n][s][b] slice for one lane."""
    return {
        QM: {
            KCD: {
                KV_N: {
                    head_size: {WIN: {n: {s: {b: _leaf(latency) for b in _GRID_B} for s in _GRID_S} for n in _GRID_N}}
                }
            }
        }
    }


def _gen_lane(latency, head_size=HEAD):
    """[kv][kv_n][head][window][n][b][s] slice for one lane."""
    return {
        KCD: {
            KV_N: {head_size: {WIN: {n: {b: {s: _leaf(latency) for s in _GRID_S} for b in _GRID_B} for n in _GRID_N}}}
        }
    }


def _count_leaves(node) -> int:
    """Number of ``{"latency": ..., ...}`` leaves nested anywhere below ``node``."""
    if isinstance(node, dict) and "latency" in node:
        return 1
    return sum(_count_leaves(child) for child in node.values())


def _slices_at(node, remaining: int) -> list:
    """Every subtree exactly ``remaining`` dict-levels below ``node`` (one entry
    per distinct physical-key coordinate -- a "slice")."""
    if remaining == 0:
        return [node]
    out: list = []
    for child in node.values():
        out.extend(_slices_at(child, remaining - 1))
    return out


def _density(table: dict, depth: int) -> dict[str, tuple[int, int]]:
    """Test-only: ``{lane: (slice_count, row_count)}`` from a lane-keyed
    fixture table, mirroring what the Rust engine's
    ``AttentionTable::context_lanes``/``generation_lanes`` compute from the
    real by-lane query structure (``lane_density<K>`` in
    ``perf_database/attention.rs``). Production gets this from
    ``engine_table_view.fetch_attention_lane_density``; these fixtures build
    it directly from a hand-constructed lane-keyed table instead, since there
    is no live Rust engine in these lightweight unit tests (same reason
    ``conftest.py``'s ``stub_perf_db``/``comprehensive_perf_db`` fixtures
    route ``fetch_table_view`` rather than build a real one).
    """
    out: dict[str, tuple[int, int]] = {}
    for lane, blob in table.items():
        slices = _slices_at(blob, depth)
        out[lane] = (len(slices), sum(_count_leaves(s) for s in slices))
    return out


class _StubDatabase:
    """Minimal PerfDatabase stand-in exercising the real lane-resolution surface.

    ``resolve_lane_order`` reads backend/version/sm_version/systems_root
    (unchanged from before). ``_context_attention_data`` /
    ``_generation_attention_data`` mirror the REAL post-rebase shape: lane-
    BLIND, enum-keyed (first lane wins by construction order) -- exactly the
    ``table_view`` rehydration that made Blocker 1's donor/leftover tiers
    read ``{}`` in production. The REAL by-lane density (mirroring
    ``AttentionTable::context_lanes``/``generation_lanes``) lives separately
    on ``self.lane_density``, reachable only through
    ``fetch_attention_lane_density`` (routed to it by
    ``_route_lane_density_through_the_stub`` below) -- never through the two
    blind attributes, so a regression that reads density off them again
    would make these tests fail exactly like production did.
    """

    def __init__(self, systems_root, context_lanes=None, generation_lanes=None, *, sm_version=103):
        self.system = "test_system"
        self.backend = "sglang"
        self.version = "0.5.14"
        self.systems_root = systems_root
        self.enable_shared_layer = False
        self.system_spec = {
            "data_dir": "data",
            "gpu": {
                "sm_version": sm_version,
                "mem_bw": 1.0e6,
                "bfloat16_tc_flops": 1.0e6,
                "fp8_tc_flops": 2.0e6,
            },
        }
        context_lanes = context_lanes or {}
        generation_lanes = generation_lanes or {}
        # Lane-blind rehydration stand-in (table_view shape): first lane
        # (dict insertion order) wins per enum-keyed coordinate, exactly like
        # the real collapsed view. Never read for density/ranking.
        self._context_attention_data = _LoadedLanes(next(iter(context_lanes.values()), {}))
        self._generation_attention_data = _LoadedLanes(next(iter(generation_lanes.values()), {}))
        self._raw_generation_attention_data = self._generation_attention_data
        # The REAL by-lane density source, fed through
        # fetch_attention_lane_density by the routing fixture below.
        self.lane_density = {
            "_context_attention_data": _density(context_lanes, _CTX_DEPTH),
            "_generation_attention_data": _density(generation_lanes, _GEN_DEPTH),
        }


@pytest.fixture(autouse=True)
def _route_lane_density_through_the_stub(monkeypatch):
    """``resolved_lane_order_for_op`` fetches density through
    ``engine_table_view.fetch_attention_lane_density`` -> the live Rust
    engine in production. Route ``_StubDatabase`` instances to their own
    ``lane_density`` attribute instead -- the same no-live-engine-in-a-unit-
    test seam ``tests/unit/sdk/database/conftest.py`` already uses for
    ``fetch_table_view`` (``stub_perf_db`` / ``comprehensive_perf_db``).
    """
    import aiconfigurator_core.sdk.engine_table_view as _etv

    real_fetch = _etv.fetch_attention_lane_density

    def _fetch(database, attribute):
        if isinstance(database, _StubDatabase):
            return database.lane_density.get(attribute, {})
        return real_fetch(database, attribute)

    monkeypatch.setattr(_etv, "fetch_attention_lane_density", _fetch)


def test_engine_spec_schema_version_is_fourteen():
    """The lane_order field is an always-serialized positional payload change."""
    from aiconfigurator.sdk import engine

    assert engine.ENGINE_SPEC_SCHEMA_VERSION == 14


def test_lanes_outside_the_known_vocabulary_stay_reachable():
    """Collected ``kernel_source`` labels are richer than the resolver's lane
    vocabulary — trtllm ships ``torch_flow*``, vllm ships ``vllm_*``, and neither
    has a ``"default"`` lane. Those rows must still be reachable (they are the
    ONLY rows there), after every named lane in the resolved order."""
    from aiconfigurator_core.sdk.operations.attention import lane_walk_order

    raw = {"torch_flow": _ctx_lane(_SLOW_LATENCY), "torch_flow_flashinfer": _ctx_lane(_FAST_LATENCY)}

    # Hand-specified orders are honoured verbatim (no tier split), leftovers
    # ride behind them; the two here are equally dense, so the name breaks the tie.
    order = lane_walk_order(_density(raw, _CTX_DEPTH), ("triton", "default"))
    assert order == ("triton", "default", "torch_flow", "torch_flow_flashinfer")
    assert lane_walk_order({}, ("triton", "default")) == ("triton", "default")


def test_donor_tier_prefers_the_data_richest_lane_over_the_alphabetic_one(lane_systems_root):
    """Gap-fill donors are ranked by measured coverage, not by name.

    On gb200/sglang the resolver's alphabetical donor tier let ``flashinfer``
    (10 slices / 2 584 rows) preempt ``trtllm_mha`` (64 / 31 141) purely because
    "f" < "t". The map-resolved head lane keeps its position; only the donor
    tiers re-order.
    """
    from aiconfigurator_core.sdk.operations.attention import lane_walk_order, resolve_lane_order

    sparse = _ctx_lane(_SLOW_LATENCY, head_size=64)  # 1 slice
    dense = _ctx_lane(_FAST_LATENCY, head_size=64)
    for extra_head in (256, 512):
        dense[QM][KCD][KV_N][extra_head] = _ctx_lane(_FAST_LATENCY, head_size=extra_head)[QM][KCD][KV_N][extra_head]

    db = _StubDatabase(lane_systems_root, context_lanes={"flashinfer": sparse, "trtllm_mha": dense})

    order = lane_walk_order(db.lane_density["_context_attention_data"], resolve_lane_order(db))

    assert order[0] == "triton", "the map-resolved head lane keeps its position"
    assert order.index("trtllm_mha") < order.index("flashinfer"), "denser donor must precede the sparser one"
    assert order[-1] == "default", "'default' stays the last resort of the known-lane tier"


def test_ties_on_slice_count_are_broken_by_row_count():
    """vllm's context table carries ``…trtllmprefill`` and ``…trtllmdecode`` with
    an IDENTICAL slice footprint; only the row count identifies the substantive
    lane, and a name tie-break would hand the context table to the decode variant."""
    from aiconfigurator_core.sdk.operations.attention import lane_walk_order

    prefill = _ctx_lane(_SLOW_LATENCY)  # full (n, s, b) grid
    decode = _ctx_lane(_FAST_LATENCY)
    # Same single slice, far fewer measured points.
    decode[QM][KCD][KV_N][HEAD][WIN] = {_GRID_N[0]: {_GRID_S[0]: {_GRID_B[0]: _leaf(_FAST_LATENCY)}}}

    raw = {"vllm_flashinfer_trtllmdecode": decode, "vllm_flashinfer_trtllmprefill": prefill}

    order = lane_walk_order(_density(raw, _CTX_DEPTH), ("triton", "default"))
    leftovers = [lane for lane in order if lane.startswith("vllm_")]
    assert leftovers == ["vllm_flashinfer_trtllmprefill", "vllm_flashinfer_trtllmdecode"]


def test_pinned_override_head_is_exempt_from_donor_density_ranking(lane_systems_root):
    """An EXPLICIT ``attention_backend`` override heads the walk even when a
    donor lane is denser — density ranks donors, never the pin.

    Regression (AIC-1715/1716): the tier split used to be reconstructed from the
    flat tuple, and a pin of ``fa3`` is byte-identical to the unpinned
    alphabetical donor tier (``fa3`` sorts first), so the override collapsed into
    the donor tier and the densest lane took the head.
    """
    from aiconfigurator_core.sdk.operations.attention import lane_walk_order, resolve_lane_order

    sparse = _ctx_lane(_SLOW_LATENCY, head_size=64)
    dense = _ctx_lane(_FAST_LATENCY, head_size=64)
    for extra_head in (256, 512):
        dense[QM][KCD][KV_N][extra_head] = _ctx_lane(_FAST_LATENCY, head_size=extra_head)[QM][KCD][KV_N][extra_head]
    # sm 999 has no entry in the fixture map: the override is the ONLY pin.
    db = _StubDatabase(lane_systems_root, context_lanes={"fa3": sparse, "trtllm_mha": dense}, sm_version=999)

    order = lane_walk_order(db.lane_density["_context_attention_data"], resolve_lane_order(db, "fa3"))

    assert order[0] == "fa3", f"the explicit override must head the walk; got {order}"
    assert order[1] == "trtllm_mha", f"the denser donor ranks first WITHIN the donor tier; got {order}"


def test_pinned_framework_default_head_is_exempt_from_donor_density_ranking(lane_systems_root):
    """The framework-default map lane heads the walk even when a donor is denser.

    Same regression as the override case, on the sm90 map entry (``fa3``): the
    reconstruction classified the pinned head as donor tier, so the densest lane
    silently replaced the framework default the map exists to express.
    """
    from aiconfigurator_core.sdk.operations.attention import lane_walk_order, resolve_lane_order

    sparse = _ctx_lane(_SLOW_LATENCY, head_size=64)
    dense = _ctx_lane(_FAST_LATENCY, head_size=64)
    for extra_head in (256, 512):
        dense[QM][KCD][KV_N][extra_head] = _ctx_lane(_FAST_LATENCY, head_size=extra_head)[QM][KCD][KV_N][extra_head]
    db = _StubDatabase(lane_systems_root, context_lanes={"fa3": sparse, "trtllm_mha": dense}, sm_version=90)

    order = lane_walk_order(db.lane_density["_context_attention_data"], resolve_lane_order(db))

    assert order[0] == "fa3", f"sglang 0.5.14 @ sm90 maps to fa3; it must head the walk; got {order}"
    assert order[1] == "trtllm_mha", f"the denser donor ranks first WITHIN the donor tier; got {order}"
    assert order[-1] == "default", "'default' stays the last resort of the known-lane tier"


# ---------------------------------------------------------------------------
# Production-wiring regressions (rebase-4 review, Blocker 1): the tests above
# probe lane_walk_order's ranking math directly, given a density dict the
# test already computed correctly -- which is exactly why they stayed green
# while the real bug (resolved_lane_order_for_op reading density off the
# lane-blind _context_attention_data/_generation_attention_data instead of
# fetch_attention_lane_density) shipped undetected. These two exercise the
# actual entry point end to end against a lane-blind stub.
# ---------------------------------------------------------------------------


def test_resolved_lane_order_for_op_ranks_donors_from_real_density_not_the_blind_view(lane_systems_root):
    """``resolved_lane_order_for_op`` (called from ``engine.py`` at spec-build
    time) must rank donor lanes from the REAL by-lane density, never from
    ``database._context_attention_data`` directly. That attribute is
    lane-BLIND on this stub (enum-keyed, first-lane-wins) exactly like the
    real post-rebase rehydration; reading it for density yields ``{}`` and
    donors fall back to alphabetical order (``'flashinfer'`` before
    ``'trtllm_mha'``) -- the production bug this guards against.
    """
    from aiconfigurator_core.sdk.operations.attention import resolved_lane_order_for_op

    sparse = _ctx_lane(_SLOW_LATENCY, head_size=64)  # 1 slice
    dense = _ctx_lane(_FAST_LATENCY, head_size=64)
    for extra_head in (256, 512):
        dense[QM][KCD][KV_N][extra_head] = _ctx_lane(_FAST_LATENCY, head_size=extra_head)[QM][KCD][KV_N][extra_head]
    db = _StubDatabase(lane_systems_root, context_lanes={"flashinfer": sparse, "trtllm_mha": dense})

    # Structural guard: the stub really does mirror the lane-blind shape.
    assert QM in db._context_attention_data
    assert "flashinfer" not in db._context_attention_data
    assert "trtllm_mha" not in db._context_attention_data

    order = resolved_lane_order_for_op(db, "_context_attention_data")

    assert order[0] == "triton", "the map-resolved head lane keeps its position"
    assert order.index("trtllm_mha") < order.index("flashinfer"), (
        f"denser donor must precede the sparser one; got {order} (alphabetical "
        "order here means density was read from the lane-blind view)"
    )


def test_resolved_lane_order_for_op_surfaces_leftover_lanes_on_an_unmapped_backend(lane_systems_root):
    """An unmapped (backend, version) resolves an EMPTY pinned order (no
    override, no framework-default map entry), so every lane in the walk
    comes from the leftover tier. Under Blocker 1 this collapsed to
    ``["default"]`` in production for exactly this scenario shape
    (h200_sxm/trtllm, b200_sxm/vllm-0.19.0 in the real-world report).
    """
    from aiconfigurator_core.sdk.operations.attention import resolved_lane_order_for_op

    raw = {"torch_flow": _ctx_lane(_SLOW_LATENCY), "torch_flow_flashinfer": _ctx_lane(_FAST_LATENCY)}
    db = _StubDatabase(lane_systems_root, context_lanes=raw)
    db.version = "0.9.9"  # not in the fixture map -> resolver order is empty

    order = resolved_lane_order_for_op(db, "_context_attention_data")

    assert "torch_flow" in order and "torch_flow_flashinfer" in order, (
        f"table's own kernel_source lanes must stay reachable; got {order}"
    )
    assert order != ["default"], "must not collapse to the always-valid fallback when real lanes exist"


# ---------------------------------------------------------------------------
# AIC-1762: per-architecture default lane, end-to-end through
# resolved_lane_order_for_op (the choke point engine.py's
# build_engine_spec_json / _resolve_attention_lane_orders calls once a
# database is in hand — see that module's docstring). Since the pyo3 op
# unification these ops have no Python __init__/query() of their own, so
# "op.query() end-to-end" (the pre-rebase shape of this test group) is
# replaced by exercising the same production choke point directly, exactly
# like test_resolved_lane_order_for_op_* above.
# ---------------------------------------------------------------------------

_ARCH_LANE_DEFAULTS_YAML = """\
# Test fixture: global sglang/0.5.14 map + one architecture entry at 0.5.17.
sglang:
  "0.5.14":
    90: fa3
    100: triton
    103: triton
    120: flashinfer
architectures:
  Qwen3_5MoeForCausalLM:
    sglang:
      "0.5.17":
        90: fa3
        100: trtllm_mha
        103: trtllm_mha
        120: flashinfer
"""

_MAX_ARCHITECTURE = "Qwen3_5MoeForCausalLM"
_CONDGEN_ARCHITECTURE = "Qwen3_5MoeForConditionalGeneration"  # 397B/35B/122B — deliberately NOT declared


@pytest.fixture
def arch_lane_systems_root(tmp_path):
    """systems_root holding the global map plus the Max architecture entry."""
    root = tmp_path / "systems"
    root.mkdir()
    (root / "attention_lane_defaults.yaml").write_text(_ARCH_LANE_DEFAULTS_YAML, encoding="utf-8")
    return str(root)


def _max_db(systems_root, **kwargs):
    """A gb300-like (sm103) stub database at sglang 0.5.17 (the Max version)."""
    db = _StubDatabase(systems_root, sm_version=103, **kwargs)
    db.version = "0.5.17"
    return db


def test_resolved_lane_order_for_op_selects_architecture_default_without_override(arch_lane_systems_root):
    """A Max-architecture context op with no attention_backend override must
    resolve trtllm_mha (its own serving-true default) instead of the
    inherited triton entry — the I1 fix, end-to-end through the production
    choke point."""
    from aiconfigurator_core.sdk.operations.attention import resolved_lane_order_for_op

    db = _max_db(
        arch_lane_systems_root,
        context_lanes={"triton": _ctx_lane(_SLOW_LATENCY), "trtllm_mha": _ctx_lane(_FAST_LATENCY)},
    )

    no_architecture = resolved_lane_order_for_op(db, "_context_attention_data", None, None)
    max_architecture = resolved_lane_order_for_op(db, "_context_attention_data", None, _MAX_ARCHITECTURE)

    assert no_architecture[0] == "triton", f"no architecture: unchanged inherited triton; got {no_architecture}"
    assert max_architecture[0] == "trtllm_mha", f"Max architecture: own trtllm_mha default; got {max_architecture}"


def test_resolved_lane_order_for_op_architecture_default_generation_twin(arch_lane_systems_root):
    """Generation twin of the context architecture-default test above."""
    from aiconfigurator_core.sdk.operations.attention import resolved_lane_order_for_op

    db = _max_db(
        arch_lane_systems_root,
        generation_lanes={"triton": _gen_lane(_SLOW_LATENCY), "trtllm_mha": _gen_lane(_FAST_LATENCY)},
    )

    no_architecture = resolved_lane_order_for_op(db, "_generation_attention_data", None, None)
    max_architecture = resolved_lane_order_for_op(db, "_generation_attention_data", None, _MAX_ARCHITECTURE)

    assert no_architecture[0] == "triton"
    assert max_architecture[0] == "trtllm_mha"


def test_resolved_lane_order_for_op_override_wins_over_architecture_default(arch_lane_systems_root):
    """An explicit override still wins outright, even for a Max-architecture
    op — explicit intent stays first-class (owner design)."""
    from aiconfigurator_core.sdk.operations.attention import resolved_lane_order_for_op

    db = _max_db(
        arch_lane_systems_root,
        context_lanes={"triton": _ctx_lane(_SLOW_LATENCY), "trtllm_mha": _ctx_lane(_FAST_LATENCY)},
    )

    order = resolved_lane_order_for_op(db, "_context_attention_data", "triton", _MAX_ARCHITECTURE)

    assert order[0] == "triton", f"explicit override must beat the architecture default; got {order}"
    assert order.count("triton") == 1, "override lane must not be duplicated by a later (donor-tier) step"


def test_resolved_lane_order_for_op_397b_architecture_is_unaffected(arch_lane_systems_root):
    """The 397B architecture string has no entry under architectures: — a
    query carrying it must resolve byte-identically to one carrying none at
    all (both fall through to the untouched global map)."""
    from aiconfigurator_core.sdk.operations.attention import resolved_lane_order_for_op

    db = _max_db(
        arch_lane_systems_root,
        context_lanes={"triton": _ctx_lane(_SLOW_LATENCY), "trtllm_mha": _ctx_lane(_FAST_LATENCY)},
    )

    no_architecture = resolved_lane_order_for_op(db, "_context_attention_data", None, None)
    conditional_generation = resolved_lane_order_for_op(db, "_context_attention_data", None, _CONDGEN_ARCHITECTURE)

    assert no_architecture == conditional_generation, (
        f"397B architecture must be byte-identical to no architecture; got {conditional_generation} "
        f"vs {no_architecture}"
    )
    assert no_architecture[0] == "triton"


def test_lane_order_cache_distinguishes_by_architecture(arch_lane_systems_root):
    """Regression guard for the LRU memo (`_lane_order_cached`): two
    architectures sharing the same database identity and override must NOT
    collide in the cache — each resolves its own order."""
    from aiconfigurator_core.sdk.operations.attention import resolve_lane_order

    db = _max_db(arch_lane_systems_root)

    max_order = resolve_lane_order(db, None, _MAX_ARCHITECTURE)
    condgen_order = resolve_lane_order(db, None, _CONDGEN_ARCHITECTURE)

    assert max_order[0] == "trtllm_mha", f"got {max_order}"
    assert condgen_order[0] == "triton", f"got {condgen_order}"


# ---------------------------------------------------------------------------
# _lane_order pickle/deepcopy round-trip (rebase-4 review, minor 4): the two
# ops encode it asymmetrically in __getnewargs_ex__ (py_ops.rs) --
# ContextAttention rides it as the 11th POSITIONAL __new__ arg (after
# cp_size), GenerationAttention rides it in the KWARGS dict instead (position
# 8 there is use_qk_norm, a different, never-round-tripped parameter, so a
# positional 8th slot would bind to the wrong thing). Both encodings must
# still round-trip the resolved order through pickle and deepcopy, which
# construct a fresh instance via __new__(*args, **kwargs) rather than
# copying attributes directly.
# ---------------------------------------------------------------------------


def _context_op_with_lane_order(order):
    from aiconfigurator.sdk import common
    from aiconfigurator_core.sdk.operations.attention import ContextAttention

    op = ContextAttention(
        "ctx_attn", 1.0, 32, 8, common.KVCacheQuantMode.fp8, common.FMHAQuantMode.fp8, 0, 128, False, 1
    )
    op._lane_order = list(order)
    return op


def _generation_op_with_lane_order(order):
    from aiconfigurator.sdk import common
    from aiconfigurator_core.sdk.operations.attention import GenerationAttention

    op = GenerationAttention("gen_attn", 1.0, 32, 8, common.KVCacheQuantMode.fp8, 0, 128, False)
    op._lane_order = list(order)
    return op


@pytest.mark.parametrize(
    ("build_op", "order"),
    [
        (_context_op_with_lane_order, ["trtllm_mha", "triton", "flashinfer", "fa3", "fla", "default"]),
        (_generation_op_with_lane_order, ["triton", "trtllm_mha", "flashinfer", "fa3", "fla", "default"]),
    ],
    ids=["context_positional_slot_11", "generation_kwargs"],
)
def test_lane_order_survives_pickle_and_deepcopy_round_trip(build_op, order):
    op = build_op(order)
    assert op._lane_order == order  # sanity: the setter under test actually took

    pickled = pickle.loads(pickle.dumps(op))
    assert pickled._lane_order == order, (
        f"__getnewargs_ex__ dropped _lane_order across pickle; got {pickled._lane_order}"
    )

    copied = copy.deepcopy(op)
    assert copied._lane_order == order, (
        f"__getnewargs_ex__ dropped _lane_order across deepcopy; got {copied._lane_order}"
    )

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
"""

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


class _StubDatabase:
    """Minimal PerfDatabase stand-in exercising the real lane-resolution surface.

    Only the attributes ``resolve_lane_order`` / ``lane_walk_order`` touch are
    provided (backend/version/sm_version/systems_root plus the raw lane-keyed
    tables); there is no query-path delegation here — per-op attention values
    come only from the compiled Rust engine (single-oracle rule).
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
        self._context_attention_data = _LoadedLanes(context_lanes or {})
        self._generation_attention_data = _LoadedLanes(generation_lanes or {})
        self._raw_generation_attention_data = self._generation_attention_data


def test_engine_spec_schema_version_is_fourteen():
    """The lane_order field is an always-serialized positional payload change."""
    from aiconfigurator.sdk import engine

    assert engine.ENGINE_SPEC_SCHEMA_VERSION == 14


def test_lanes_outside_the_known_vocabulary_stay_reachable(lane_systems_root):
    """Collected ``kernel_source`` labels are richer than the resolver's lane
    vocabulary — trtllm ships ``torch_flow*``, vllm ships ``vllm_*``, and neither
    has a ``"default"`` lane. Those rows must still be reachable (they are the
    ONLY rows there), after every named lane in the resolved order."""
    from aiconfigurator.sdk.operations.attention import lane_walk_order

    db = _StubDatabase(
        lane_systems_root,
        context_lanes={"torch_flow": _ctx_lane(_SLOW_LATENCY), "torch_flow_flashinfer": _ctx_lane(_FAST_LATENCY)},
    )

    # Hand-specified orders are honoured verbatim (no tier split), leftovers
    # ride behind them; the two here are equally dense, so the name breaks the tie.
    order = lane_walk_order(db._context_attention_data, ("triton", "default"), _CTX_DEPTH)
    assert order == ("triton", "default", "torch_flow", "torch_flow_flashinfer")
    assert lane_walk_order({}, ("triton", "default"), _CTX_DEPTH) == ("triton", "default")


def test_donor_tier_prefers_the_data_richest_lane_over_the_alphabetic_one(lane_systems_root):
    """Gap-fill donors are ranked by measured coverage, not by name.

    On gb200/sglang the resolver's alphabetical donor tier let ``flashinfer``
    (10 slices / 2 584 rows) preempt ``trtllm_mha`` (64 / 31 141) purely because
    "f" < "t". The map-resolved head lane keeps its position; only the donor
    tiers re-order.
    """
    from aiconfigurator.sdk.operations.attention import lane_walk_order, resolve_lane_order

    sparse = _ctx_lane(_SLOW_LATENCY, head_size=64)  # 1 slice
    dense = _ctx_lane(_FAST_LATENCY, head_size=64)
    for extra_head in (256, 512):  # 3 slices total -> strictly denser
        dense[QM][KCD][KV_N][extra_head] = _ctx_lane(_FAST_LATENCY, head_size=extra_head)[QM][KCD][KV_N][extra_head]

    db = _StubDatabase(lane_systems_root, context_lanes={"flashinfer": sparse, "trtllm_mha": dense})

    order = lane_walk_order(db._context_attention_data, resolve_lane_order(db), _CTX_DEPTH)

    assert order[0] == "triton", "the map-resolved head lane keeps its position"
    assert order.index("trtllm_mha") < order.index("flashinfer"), "denser donor must precede the sparser one"
    assert order[-1] == "default", "'default' stays the last resort of the known-lane tier"


def test_ties_on_slice_count_are_broken_by_row_count(lane_systems_root):
    """vllm's context table carries ``…trtllmprefill`` and ``…trtllmdecode`` with
    an IDENTICAL slice footprint; only the row count identifies the substantive
    lane, and a name tie-break would hand the context table to the decode variant."""
    from aiconfigurator.sdk.operations.attention import lane_walk_order

    prefill = _ctx_lane(_SLOW_LATENCY)  # full (n, s, b) grid
    decode = _ctx_lane(_FAST_LATENCY)
    # Same single slice, far fewer measured points.
    decode[QM][KCD][KV_N][HEAD][WIN] = {_GRID_N[0]: {_GRID_S[0]: {_GRID_B[0]: _leaf(_FAST_LATENCY)}}}

    db = _StubDatabase(
        lane_systems_root,
        context_lanes={"vllm_flashinfer_trtllmdecode": decode, "vllm_flashinfer_trtllmprefill": prefill},
    )

    order = lane_walk_order(db._context_attention_data, ("triton", "default"), _CTX_DEPTH)
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
    from aiconfigurator.sdk.operations.attention import lane_walk_order, resolve_lane_order

    sparse = _ctx_lane(_SLOW_LATENCY, head_size=64)
    dense = _ctx_lane(_FAST_LATENCY, head_size=64)
    for extra_head in (256, 512):
        dense[QM][KCD][KV_N][extra_head] = _ctx_lane(_FAST_LATENCY, head_size=extra_head)[QM][KCD][KV_N][extra_head]
    # sm 999 has no entry in the fixture map: the override is the ONLY pin.
    db = _StubDatabase(lane_systems_root, context_lanes={"fa3": sparse, "trtllm_mha": dense}, sm_version=999)

    order = lane_walk_order(db._context_attention_data, resolve_lane_order(db, "fa3"), _CTX_DEPTH)

    assert order[0] == "fa3", f"the explicit override must head the walk; got {order}"
    assert order[1] == "trtllm_mha", f"the denser donor ranks first WITHIN the donor tier; got {order}"


def test_pinned_framework_default_head_is_exempt_from_donor_density_ranking(lane_systems_root):
    """The framework-default map lane heads the walk even when a donor is denser.

    Same regression as the override case, on the sm90 map entry (``fa3``): the
    reconstruction classified the pinned head as donor tier, so the densest lane
    silently replaced the framework default the map exists to express.
    """
    from aiconfigurator.sdk.operations.attention import lane_walk_order, resolve_lane_order

    sparse = _ctx_lane(_SLOW_LATENCY, head_size=64)
    dense = _ctx_lane(_FAST_LATENCY, head_size=64)
    for extra_head in (256, 512):
        dense[QM][KCD][KV_N][extra_head] = _ctx_lane(_FAST_LATENCY, head_size=extra_head)[QM][KCD][KV_N][extra_head]
    db = _StubDatabase(lane_systems_root, context_lanes={"fa3": sparse, "trtllm_mha": dense}, sm_version=90)

    order = lane_walk_order(db._context_attention_data, resolve_lane_order(db), _CTX_DEPTH)

    assert order[0] == "fa3", f"sglang 0.5.14 @ sm90 maps to fa3; it must head the walk; got {order}"
    assert order[1] == "trtllm_mha", f"the denser donor ranks first WITHIN the donor tier; got {order}"
    assert order[-1] == "default", "'default' stays the last resort of the known-lane tier"

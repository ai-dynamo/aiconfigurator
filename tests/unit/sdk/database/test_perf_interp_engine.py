# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioural tests for the perf_interp resolver engine (v2).

Synthetic tables whose latency follows the op's physics exactly, so the
engine's answers have closed-form truths: RAW lerp is exact for linear physics,
SQRT lerp is exact for ~seq^2, and util-hold is exact when the boundary util
matches the query's util (SOL carries the growth).
"""

import math

import pytest

from aiconfigurator.sdk import perf_interp
from aiconfigurator.sdk.errors import InterpolationDataNotAvailableError

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# GEMM-like: scattered (n, k) sites + m curves; latency = c*m*n*k (linear in m)
# ---------------------------------------------------------------------------

_C = 1e-9


def _gemm_lat(m, n, k):
    return _C * m * n * k


def _gemm_sol(m, n, k):
    return _gemm_lat(m, n, k)  # util == 1 everywhere -> holds are exact


def _gemm_table():
    """Coarse symmetric shapes + ONE asymmetric dense-m site (3072, 5120) that
    forms no Cartesian product with the rest (the P0-1 stress case)."""
    data = {}
    for m in (256, 4096):
        for n in (2048, 4096):
            for k in (2048, 4096):
                data.setdefault(m, {}).setdefault(n, {})[k] = {"latency": _gemm_lat(m, n, k), "energy": 0.0}
    for m in (512, 1024, 2048):
        data.setdefault(m, {}).setdefault(3072, {})[5120] = {"latency": _gemm_lat(m, 3072, 5120), "energy": 0.0}
    return data


def _gemm_cfg():
    return perf_interp.gemm_config(sol_fn=_gemm_sol)


def _lat(result):
    return result["latency"] if isinstance(result, dict) else float(result)


def test_exact_hit_returns_leaf_verbatim():
    data = _gemm_table()
    result = perf_interp.query(_gemm_cfg(), data, 512, 3072, 5120)
    assert result is data[512][3072][5120]  # identity, not a copy


def test_exact_hit_supports_legacy_float_leaves():
    data = {8: {128: {128: 0.5}}, 16: {128: {128: 0.9}}}
    assert perf_interp.query(_gemm_cfg(), data, 8, 128, 128) == 0.5
    # in-curve lerp on the site's own m-curve: (0.5 + 0.9) / 2
    assert _lat(perf_interp.query(_gemm_cfg(), data, 12, 128, 128)) == pytest.approx(0.7)


def test_asymmetric_site_interpolates_on_its_own_curve():
    # m=768 lies between the anchor's own m=512 and m=1024 samples; the coarse
    # shapes never contaminate the exact-site path.
    lat = _lat(perf_interp.query(_gemm_cfg(), _gemm_table(), 768, 3072, 5120))
    assert lat == pytest.approx(_gemm_lat(768, 3072, 5120))


def test_site_curve_util_hold_beyond_sweep():
    # m=8192 is past the anchor site's sweep (<=2048): hold util, SOL ~ m.
    lat = _lat(perf_interp.query(_gemm_cfg(), _gemm_table(), 8192, 3072, 5120))
    assert lat == pytest.approx(_gemm_lat(8192, 3072, 5120))


def test_unknown_site_transfers_util_from_neighbours():
    # (4000, 4000) is not collected; nearest sites are the symmetric 4096 ones
    # (well within the 0.25-octave gate). util==1 everywhere -> exact transfer.
    lat = _lat(perf_interp.query(_gemm_cfg(), _gemm_table(), 1000, 4000, 4000))
    assert lat == pytest.approx(_gemm_lat(1000, 4000, 4000), rel=1e-6)


def test_far_site_is_a_structured_miss():
    # (16384, 512) is ~2.6 octaves from every collected shape -> miss, not a
    # guess. (Also the mixed-direction case: n is scale-up beyond the frontier
    # but k is BELOW every collected k, so the frontier fallback must not fire.)
    with pytest.raises(InterpolationDataNotAvailableError):
        perf_interp.query(_gemm_cfg(), _gemm_table(), 1000, 16384, 512)


def test_scale_up_beyond_frontier_holds_util():
    # Gemma-4-26B-A4B tp1 LM head (issue #1415): (n=262144, k=2816) is ~2.005
    # octaves from the nearest collected site (n caps at 65536 in every GPU
    # database) — past the distance gate — but lies beyond the frontier in the
    # scale-up direction, so the engine holds the frontier util instead of
    # missing. util==1 fixture -> the hold is exact.
    data = {}
    for m in (16, 32, 64):
        for n, k in ((65536, 2560), (65536, 3072), (4096, 2816)):
            data.setdefault(m, {}).setdefault(n, {})[k] = {"latency": _gemm_lat(m, n, k), "energy": 0.0}
    lat = _lat(perf_interp.query(_gemm_cfg(), data, 32, 262144, 2816))
    assert lat == pytest.approx(_gemm_lat(32, 262144, 2816), rel=1e-6)


def test_interior_hole_beyond_gate_still_misses():
    # (8192, 8192) is dominated by the collected (65536, 65536) corner but >2
    # octaves from every site — a sparse interior hole, not a frontier query.
    # The gate must still refuse to guess.
    data = {}
    for m in (16, 32, 64):
        for n, k in ((256, 256), (65536, 65536)):
            data.setdefault(m, {}).setdefault(n, {})[k] = {"latency": _gemm_lat(m, n, k), "energy": 0.0}
    with pytest.raises(InterpolationDataNotAvailableError):
        perf_interp.query(_gemm_cfg(), data, 32, 8192, 8192)


def test_incomparable_notch_inside_bounding_box_still_misses():
    # (4096, 4096) sits in the notch between (256, 65536) and (65536, 256): no
    # site dominates it, yet it exceeds the collected maximum on NO axis — an
    # interior hole in the Pareto staircase, not a scale-up frontier query.
    # The waiver must not fire; the distance gate stands.
    data = {}
    for m in (16, 32, 64):
        for n, k in ((256, 65536), (65536, 256)):
            data.setdefault(m, {}).setdefault(n, {})[k] = {"latency": _gemm_lat(m, n, k), "energy": 0.0}
    with pytest.raises(InterpolationDataNotAvailableError):
        perf_interp.query(_gemm_cfg(), data, 32, 4096, 4096)


def test_sparse_stub_multi_axis_overflow_still_misses():
    # Real-table regression (PR #1419 review): rtx_pro_6000_server's fp8_block
    # gemm table only collects n,k in 32..128; Qwen3-32B's logits GEMM
    # (m=1, n=151936, k=5120) sits ~12 octaves away with BOTH site axes past
    # the collected maximum. Holding a launch-bound stub's util across that
    # gap fabricated 24,506 ms against a 0.543 ms truth — simultaneous
    # multi-axis overflow must stay a structured miss.
    data = {}
    for m in (1, 2, 4):
        for n in (32, 64, 128):
            for k in (32, 64, 128):
                data.setdefault(m, {}).setdefault(n, {})[k] = {"latency": _gemm_lat(m, n, k), "energy": 0.0}
    with pytest.raises(InterpolationDataNotAvailableError):
        perf_interp.query(_gemm_cfg(), data, 1, 151936, 5120)


def test_waiver_admissibility_computed_over_coverage_candidates():
    # PR #1419 review: with site (1, 1) covering only m<=2 and site (64, 64)
    # covering m=16..64, the query (m=32, n=1024, k=1) used to pass a GLOBAL
    # frontier check (n above the global max, k at the global min) while the
    # only coverage-eligible anchor (64, 64) required a k=64 -> k=1 scale-DOWN
    # transfer. Admissibility must be computed over the coverage-eligible
    # candidates, where k=1 sits below the minimum -> miss.
    data = {}
    for m in (1, 2):
        data.setdefault(m, {}).setdefault(1, {})[1] = {"latency": _gemm_lat(m, 1, 1), "energy": 0.0}
    for m in (16, 32, 64):
        data.setdefault(m, {}).setdefault(64, {})[64] = {"latency": _gemm_lat(m, 64, 64), "energy": 0.0}
    with pytest.raises(InterpolationDataNotAvailableError):
        perf_interp.query(_gemm_cfg(), data, 32, 1024, 1)


def test_coverage_filter_prefers_sites_that_span_the_query_m():
    # Site A (2048, 2048) only has tiny m (decode-like sweep, m <= 4) with a
    # BAD tail util; site B (2000, 2000) covers large m with util == 1. A naive
    # nearest-site pick would take A (closer); the coverage filter must use B.
    data = {}
    for m in (1, 2, 4):
        data.setdefault(m, {}).setdefault(2048, {})[2048] = {"latency": 100.0 * _gemm_lat(m, 2048, 2048), "energy": 0.0}
    for m in (256, 1024, 8192):
        data.setdefault(m, {}).setdefault(2000, {})[2000] = {"latency": _gemm_lat(m, 2000, 2000), "energy": 0.0}
    lat = _lat(perf_interp.query(_gemm_cfg(), data, 1024, 2100, 2100))
    assert lat == pytest.approx(_gemm_lat(1024, 2100, 2100), rel=1e-6)


def test_k_tail_median_resists_boundary_sawtooth():
    # The last collected m is a sawtooth outlier (2x latency). A single-point
    # boundary anchor would double every extrapolated value; the k_tail=3
    # median must side with the two clean neighbours.
    data = {}
    for m, bad in ((256, False), (512, False), (768, False), (1024, True)):
        lat = _gemm_lat(m, 1000, 1000) * (2.0 if bad else 1.0)
        data.setdefault(m, {}).setdefault(1000, {})[1000] = {"latency": lat, "energy": 0.0}
    lat = _lat(perf_interp.query(_gemm_cfg(), data, 4096, 1000, 1000))
    assert lat == pytest.approx(_gemm_lat(4096, 1000, 1000), rel=1e-6)


# ---------------------------------------------------------------------------
# Attention-like: (num_heads, seq, batch) grid, corner-truncated; lat ~ n*b*s^2
# ---------------------------------------------------------------------------


def _attn_lat(n, s, b):
    return 1e-6 * n * b * s * s


def _attn_cfg():
    return perf_interp.context_attention_config(sol_fn=_attn_lat)  # util == 1


def _attn_table():
    """Staircase: the larger the seq, the fewer batches collected."""
    present = {512: (1, 2, 4, 8), 1024: (1, 2, 4, 8), 2048: (1, 2, 4), 4096: (1, 2)}
    return {
        n: {s: {b: {"latency": _attn_lat(n, s, b), "energy": 0.0} for b in bs} for s, bs in present.items()}
        for n in (8, 16)
    }


def test_grid_sqrt_blend_is_exact_for_quadratic_seq():
    # seq=1536 between 1024 and 2048: sqrt(lat) is linear in s -> exact.
    lat = _lat(perf_interp.query(_attn_cfg(), _attn_table(), 8, 1536, 2))
    assert lat == pytest.approx(_attn_lat(8, 1536, 2))


def test_transform_applies_only_along_its_axis():
    # batch=3 between 2 and 4 (seq exact): latency is LINEAR in batch, so the
    # blend must be raw-exact — sqrt (configured for the seq axis) must NOT
    # distort the batch axis. Curvature is per-axis (LOO: global sqrt 9.4% vs
    # per-axis 2.0% interior on real data).
    lat = _lat(perf_interp.query(_attn_cfg(), _attn_table(), 8, 1024, 3))
    assert lat == pytest.approx(_attn_lat(8, 1024, 3))


def test_grid_corner_hole_is_util_hold():
    # (seq=4096, batch=8) sits inside the truncated corner (batch <= 2 there):
    # hold the frontier util, SOL restores the batch scaling.
    lat = _lat(perf_interp.query(_attn_cfg(), _attn_table(), 8, 4096, 8))
    assert lat == pytest.approx(_attn_lat(8, 4096, 8))


def test_grid_beyond_max_seq_tracks_sol_growth():
    lat = _lat(perf_interp.query(_attn_cfg(), _attn_table(), 8, 16384, 1))
    assert lat == pytest.approx(_attn_lat(8, 16384, 1))


def test_grid_ragged_branch_is_dropped_not_fatal():
    # seq=1536 at batch=8: the 2048 branch lacks batch=8 (staircase) -> the
    # engine must degrade to the surviving 1024 branch, not crash.
    lat = _lat(perf_interp.query(_attn_cfg(), _attn_table(), 8, 1536, 8))
    assert math.isfinite(lat) and lat > 0


def test_grid_single_survivor_gets_sol_ratio_correction():
    """When one bracket branch drops, the survivor's value is re-scaled by the
    SOL ratio along the dropped axis (util held across it) -- a plain clamp
    measured -41% median on one-sided seq-row LOO folds (seq^2 physics).

    seq=1536 at batch=8: 2048 branch lacks b=8 -> survivor is the 1024 branch
    (exact leaf), corrected by SOL(8,1536,8)/SOL(8,1024,8). The fixture's SOL
    equals its latency (util == 1), so the correction reproduces the formula
    exactly at the un-collected seq.
    """
    lat = _lat(perf_interp.query(_attn_cfg(), _attn_table(), 8, 1536, 8))
    assert lat == pytest.approx(_attn_lat(8, 1024, 8) * _attn_lat(8, 1536, 8) / _attn_lat(8, 1024, 8))
    assert lat == pytest.approx(_attn_lat(8, 1536, 8))


def test_grid_empty_table_is_a_miss():
    with pytest.raises(InterpolationDataNotAvailableError):
        perf_interp.query(_attn_cfg(), {}, 8, 512, 1)


# ---------------------------------------------------------------------------
# Multi-axis hold: joint-log kNN util transfer (the B200 gen-attn report case)
# ---------------------------------------------------------------------------
#
# Staircase with a REGIME SPLIT: the deep row (b=128) is collected at exact
# physics (saturated, util 1) while the short row (b=256) ends early in a
# 1.4x-latency regime (unsaturated row end, like the real 128K-token-capped
# b>=256 rows). Every query below is past the frontier on both rows.


def _gen_lat(n, b, s):
    return 1e-6 * n * b * s  # decode physics: linear in both


def _gen_cfg():
    return perf_interp.generation_attention_config(sol_fn=_gen_lat)


def _gen_split_table():
    table = {64: {}}
    table[64][128] = {s: {"latency": _gen_lat(64, 128, s), "energy": 0.0} for s in (512, 1024, 2048)}
    table[64][256] = {s: {"latency": 1.4 * _gen_lat(64, 256, s), "energy": 0.0} for s in (128, 256, 512)}
    return table


def test_grid_hold_is_continuous_across_outer_midpoint():
    """The nearest-path snap flipped anchors at the bracket midpoint (b=192 ->
    row 128, b=193 -> row 256), a +36.9% cliff on the real B200 table where
    measured hardware moves +0.17%. The kNN hold must stay continuous: the
    192->193 step may not exceed the SOL growth by more than a percent."""
    table = _gen_split_table()
    lat_192 = _lat(perf_interp.query(_gen_cfg(), table, 64, 192, 4096))
    lat_193 = _lat(perf_interp.query(_gen_cfg(), table, 64, 193, 4096))
    sol_step = _gen_lat(64, 193, 4096) / _gen_lat(64, 192, 4096)
    assert lat_193 / lat_192 == pytest.approx(sol_step, abs=0.01)


def test_grid_hold_prefers_nearby_saturated_evidence():
    """A query deep past the short row's end (b=256, s=4096) must not inherit
    that row's unsaturated 1.4x boundary util verbatim (the old snap did,
    +41% on the reported B200 case): the joint-log nearest leaves are the deep
    saturated ones, so the answer lands near physics."""
    lat = _lat(perf_interp.query(_gen_cfg(), _gen_split_table(), 64, 256, 4096))
    assert lat / _gen_lat(64, 256, 4096) < 1.15  # snap gave 1.4x
    assert lat / _gen_lat(64, 256, 4096) > 0.95


def test_grid_hold_is_axis_order_independent():
    """The hold works on joint coordinates, not the nesting order: the same
    data nested [n][b][s] and [n][s][b] must answer a past-frontier query
    identically (this is what makes the generation and context table layouts
    behave the same past the staircase)."""
    table = _gen_split_table()
    swapped = {64: {}}
    for b, row in table[64].items():
        for s, leaf in row.items():
            swapped[64].setdefault(s, {})[b] = leaf
    cfg_swapped = perf_interp.OpInterpConfig(
        axes=("num_heads", "seq_len", "batch"),
        resolver=perf_interp.Grid(),
        sol_fn=lambda n, s, b: _gen_lat(n, b, s),
    )
    lat = _lat(perf_interp.query(_gen_cfg(), table, 64, 200, 4096))
    lat_swapped = _lat(perf_interp.query(cfg_swapped, swapped, 64, 4096, 200))
    assert lat == pytest.approx(lat_swapped, rel=1e-9)


def test_grid_hold_boundary_ties_carry_zero_weight():
    """Power-of-two grids produce EXACT distance ties. Under the tapered
    weights, leaves sitting exactly on the support radius R (the 5th valid
    distance) contribute ZERO — so which side of the nn_leaves cutoff a tied
    leaf lands on cannot matter, and the blend is independent of axis order
    and table insertion order without any tie-ordering rule.

    Fixture: query (n=64, b=32, s=1) sits below the collected s range (hold).
    One leaf at distance 1, then a FOUR-way tie at sqrt(2); the 5th valid
    distance IS sqrt(2), so every tie member weighs 0 and the answer equals
    the nearest anchor's held util exactly. The b/s-swapped nesting must
    answer identically."""
    lat_of = {
        (64, 32, 2): 1.00,  # d=1: the only anchor with non-zero weight
        (64, 16, 2): 1.10,  # d=sqrt(2) x4, distinct latencies so any leaked
        (64, 64, 2): 1.30,  # tie weight would shift the blend
        (32, 32, 2): 1.50,
        (128, 32, 2): 1.70,
        (64, 32, 4): 2.00,  # d=2, beyond the support: excluded
    }
    table = {}
    for (n, b, s), lat in lat_of.items():
        table.setdefault(n, {}).setdefault(b, {})[s] = {"latency": lat, "energy": 0.0}

    def sol(n, b, s):
        return 1.0  # constant SOL isolates the anchor selection

    cfg = perf_interp.OpInterpConfig(axes=("num_heads", "batch", "seq_len"), resolver=perf_interp.Grid(), sol_fn=sol)

    expected = lat_of[(64, 32, 2)]  # util held from the single live anchor
    lat = _lat(perf_interp.query(cfg, table, 64, 32, 1))
    assert lat == pytest.approx(expected, rel=1e-9)

    swapped = {}
    for (n, b, s), leaf_lat in lat_of.items():
        swapped.setdefault(n, {}).setdefault(s, {})[b] = {"latency": leaf_lat, "energy": 0.0}
    cfg_swapped = perf_interp.OpInterpConfig(
        axes=("num_heads", "seq_len", "batch"),
        resolver=perf_interp.Grid(),
        sol_fn=lambda n, s, b: 1.0,
    )
    lat_swapped = _lat(perf_interp.query(cfg_swapped, swapped, 64, 1, 32))
    assert lat == pytest.approx(lat_swapped, rel=1e-12)


def test_grid_hold_is_continuous_across_rank_swaps():
    """The support-radius taper sends a neighbour's weight to ZERO as it
    leaves the nn_leaves selection, so the estimate is continuous when leaves
    4 and 5 exchange rank — a hard cutoff traded them at full weight (review
    finding: epsilon steps across the swap moved the estimate ~10x on an
    adversarial fixture).

    Fixture: five (a, b) leaves at b=8, query (x, 32) beyond the b range;
    L4=(16,.) and L5=(1024,.) swap ranks at x=128 and their latencies differ
    100x, so any leaked full-weight swap is a huge jump. The estimate across
    x = 128*(1 +- 1e-6) must move by less than 0.01%."""
    lat_of = {(128, 8): 1.0, (96, 8): 1.1, (160, 8): 1.2, (16, 8): 0.5, (1024, 8): 50.0}
    table = {}
    for (a, b), lat in lat_of.items():
        table.setdefault(a, {})[b] = {"latency": lat, "energy": 0.0}

    def sol(a, b):
        return 1.0

    cfg = perf_interp.OpInterpConfig(axes=("a", "b"), resolver=perf_interp.Grid(), sol_fn=sol)
    lo = _lat(perf_interp.query(cfg, table, 128.0 * (1 - 1e-6), 32))
    hi = _lat(perf_interp.query(cfg, table, 128.0 * (1 + 1e-6), 32))
    assert abs(hi - lo) / lo < 1e-4, f"rank swap discontinuity: {lo} vs {hi}"


def test_config_rejects_bad_transform_axis_and_k_tail():
    with pytest.raises(ValueError, match="transform_axis"):
        perf_interp.OpInterpConfig(
            axes=("num_heads", "seq_len", "batch"),
            resolver=perf_interp.Grid(),
            sol_fn=_attn_lat,
            value_transform=perf_interp.ValueTransform.SQRT,
            transform_axis="seq",  # typo of "seq_len" -- must fail loudly, not silently no-op
        )
    with pytest.raises(ValueError, match="k_tail"):
        perf_interp.OpInterpConfig(
            axes=("num_heads", "seq_len", "batch"),
            resolver=perf_interp.Grid(k_tail=0),
            sol_fn=_attn_lat,
        )
    with pytest.raises(ValueError, match="nn_leaves"):
        perf_interp.OpInterpConfig(
            axes=("num_heads", "seq_len", "batch"),
            resolver=perf_interp.Grid(nn_leaves=0),
            sol_fn=_attn_lat,
        )
    with pytest.raises(ValueError, match="duplicate"):
        perf_interp.OpInterpConfig(
            axes=("seq_len", "seq_len", "batch"),
            resolver=perf_interp.Grid(),
            sol_fn=_attn_lat,
        )
    with pytest.raises(ValueError, match="UTIL"):
        perf_interp.OpInterpConfig(
            axes=("num_heads", "seq_len", "batch"),
            resolver=perf_interp.Grid(),
            sol_fn=_attn_lat,
            value_transform=perf_interp.ValueTransform.UTIL,
        )


# ---------------------------------------------------------------------------
# 4-axis (DSA/CSA-like): [num_heads][prefix][seq][batch] — the past-KV axis
# ---------------------------------------------------------------------------


def _dsa_lat(n, p, s, b):
    return 1e-6 * n * b * s * (s + p)  # grows with past-KV; linear per axis pair


def _dsa_cfg():
    return perf_interp.OpInterpConfig(
        axes=("num_heads", "prefix", "seq_len", "batch"),
        resolver=perf_interp.Grid(),
        sol_fn=_dsa_lat,  # util == 1 -> holds are exact
        value_transform=perf_interp.ValueTransform.RAW,
    )


def _dsa_table():
    return {
        n: {
            p: {s: {b: {"latency": _dsa_lat(n, p, s, b), "energy": 0.0} for b in (1, 4)} for s in (512, 1024, 2048)}
            for p in (0, 1024, 4096)
        }
        for n in (8, 16)
    }


def test_four_axis_exact_hit():
    data = _dsa_table()
    assert perf_interp.query(_dsa_cfg(), data, 8, 1024, 512, 4) is data[8][1024][512][4]


def test_four_axis_interior_blend_across_prefix():
    # prefix=2048 between collected 1024 and 4096; all other axes exact.
    lat = _lat(perf_interp.query(_dsa_cfg(), _dsa_table(), 8, 2048, 512, 4))
    lo, hi = _dsa_lat(8, 1024, 512, 4), _dsa_lat(8, 4096, 512, 4)
    expected = lo + (hi - lo) * (2048 - 1024) / (4096 - 1024)  # RAW lerp on the prefix axis
    assert lat == pytest.approx(expected)


def test_four_axis_util_hold_beyond_prefix_range():
    # prefix=16384 beyond the collected range: hold util, SOL carries growth.
    lat = _lat(perf_interp.query(_dsa_cfg(), _dsa_table(), 8, 16384, 512, 1))
    assert lat == pytest.approx(_dsa_lat(8, 16384, 512, 1))


# ---------------------------------------------------------------------------
# 1-axis (the 1-D/2-D convergence wave: comm/moe-tokens/bmm curves)
# ---------------------------------------------------------------------------


def _curve_lat(size):
    return 2e-9 * size  # bandwidth-bound: linear


def _curve_cfg():
    return perf_interp.OpInterpConfig(
        axes=("message_bytes",),
        resolver=perf_interp.Grid(),
        sol_fn=_curve_lat,  # util == 1 -> holds exact
    )


def _curve_table():
    return {s: {"latency": _curve_lat(s), "energy": 0.0} for s in (1024, 4096, 16384, 65536)}


def test_one_axis_exact_and_interp():
    data = _curve_table()
    assert perf_interp.query(_curve_cfg(), data, 4096) is data[4096]
    lat = _lat(perf_interp.query(_curve_cfg(), data, 8192))
    assert lat == pytest.approx(_curve_lat(8192))


def test_one_axis_util_hold_beyond_range():
    lat = _lat(perf_interp.query(_curve_cfg(), _curve_table(), 1 << 20))
    assert lat == pytest.approx(_curve_lat(1 << 20))


def test_data_unavailable_error_subclasses_value_error():
    """Existing ``except ValueError`` callers must keep working."""
    assert issubclass(InterpolationDataNotAvailableError, ValueError)


def test_get_value_dict_and_legacy_float_leaves():
    leaf = {"latency": 1.5, "power": 2.0}
    assert perf_interp.get_value(leaf) == 1.5
    assert perf_interp.get_value(leaf, "power") == 2.0
    assert perf_interp.get_value(leaf, "energy") == 0.0  # absent -> 0
    # Legacy format: raw float is latency, other metrics are 0
    assert perf_interp.get_value(0.7) == 0.7
    assert perf_interp.get_value(0.7, "power") == 0.0

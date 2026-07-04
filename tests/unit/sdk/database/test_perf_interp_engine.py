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
from aiconfigurator.sdk.interpolation import InterpolationDataNotAvailableError

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
    # (16384, 512) is ~2.6 octaves from every collected shape -> miss, not a guess.
    with pytest.raises(InterpolationDataNotAvailableError):
        perf_interp.query(_gemm_cfg(), _gemm_table(), 1000, 16384, 512)


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


def test_grid_empty_table_is_a_miss():
    with pytest.raises(InterpolationDataNotAvailableError):
        perf_interp.query(_attn_cfg(), {}, 8, 512, 1)


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

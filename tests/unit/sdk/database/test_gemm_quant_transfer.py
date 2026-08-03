# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GEMM quant-transfer ladder (shared quant-transfer primitive).

The mechanism acceptance contract for adding a data-less quant: with only an
enum member and a util-LEVEL line, HYBRID/EMPIRICAL must resolve end-to-end
through the borrow relations while SILICON keeps rejecting. ``int4_wo``
(profile (0.5, 1), no fixture data) plays the synthetic new quant — the
ladder keys only off the profile and the level table, so an existing member
without data exercises exactly the add-a-quant path (e.g. #1392's nvfp4_wo).
"""

import math

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.errors import EmpiricalNotImplementedError, PerfDataNotAvailableError
from aiconfigurator.sdk.operations import util_empirical
from aiconfigurator.sdk.operations.gemm import _GEMM_QUANT_UTIL_LEVEL

pytestmark = pytest.mark.unit


def _sol(db, m, n, k, quant):
    return float(db.query_gemm(m, n, k, quant, database_mode=common.DatabaseMode.SOL))


def _assert_dataless(db, quant):
    """Guard the fixture premise: ``quant`` must have no collected GEMM table
    (otherwise the ladder tests below stop exercising the borrow and pass for
    the wrong reason)."""
    with pytest.raises(PerfDataNotAvailableError):
        db.query_gemm(16, 256, 512, quant, database_mode=common.DatabaseMode.SILICON)


def test_xquant_borrows_same_profile_sibling(comprehensive_perf_db):
    """sq (1,2) has no data; fp8 (1,2) does. Same profile => identical SOL
    coefficients => the borrowed estimate at a collected point equals fp8's
    measured value exactly, tagged xquant (no level-table involvement)."""
    db = comprehensive_perf_db
    m, n, k = 16, 256, 512  # collected point in the fixture grid

    fp8_silicon = float(db.query_gemm(m, n, k, common.GEMMQuantMode.fp8, database_mode=common.DatabaseMode.SILICON))
    with util_empirical.capture_provenance() as tags:
        borrowed = float(db.query_gemm(m, n, k, common.GEMMQuantMode.sq, database_mode=common.DatabaseMode.HYBRID))

    assert "xquant" in tags
    assert math.isclose(borrowed, fp8_silicon, rel_tol=1e-9)


def test_xprofile_weight_only_borrows_bf16_not_fp8(comprehensive_perf_db):
    """The nvfp4_wo guarantee, exercised via int4_wo (0.5, 1): a weight-only
    quant runs the bf16 compute family, so with BOTH bfloat16 and fp8
    collected the reference must be bfloat16 (prefer_same_compute ordering,
    never a file-order tie-break into fp8), rescaled by e(0.5,1)/e(2,1)."""
    db = comprehensive_perf_db
    m, n, k = 16, 256, 512
    quant = common.GEMMQuantMode.int4_wo
    _assert_dataless(db, quant)

    with util_empirical.capture_provenance() as tags:
        borrowed = float(db.query_gemm(m, n, k, quant, database_mode=common.DatabaseMode.HYBRID))
    assert "xprofile" in tags

    bf16_silicon = float(
        db.query_gemm(m, n, k, common.GEMMQuantMode.bfloat16, database_mode=common.DatabaseMode.SILICON)
    )
    util_bf16 = _sol(db, m, n, k, common.GEMMQuantMode.bfloat16) / bf16_silicon
    ratio = _GEMM_QUANT_UTIL_LEVEL[(0.5, 1)] / _GEMM_QUANT_UTIL_LEVEL[(2, 1)]
    expected_from_bf16 = _sol(db, m, n, k, quant) / (util_bf16 * ratio)
    assert math.isclose(borrowed, expected_from_bf16, rel_tol=1e-9)

    # And it must NOT be the fp8-derived value (guards against the ordering
    # silently regressing to profile-L1 / file-order selection).
    fp8_silicon = float(db.query_gemm(m, n, k, common.GEMMQuantMode.fp8, database_mode=common.DatabaseMode.SILICON))
    util_fp8 = _sol(db, m, n, k, common.GEMMQuantMode.fp8) / fp8_silicon
    ratio_fp8 = _GEMM_QUANT_UTIL_LEVEL[(0.5, 1)] / _GEMM_QUANT_UTIL_LEVEL[(1, 2)]
    expected_from_fp8 = _sol(db, m, n, k, quant) / (util_fp8 * ratio_fp8)
    assert not math.isclose(borrowed, expected_from_fp8, rel_tol=1e-6)


def test_xprofile_is_policy_gated(comprehensive_perf_db):
    """balanced (XSHAPE+XQUANT) finds nothing for (0.5, 1) — no same-profile
    sibling — and must raise the typed empirical miss, not silently borrow."""
    db = comprehensive_perf_db
    _assert_dataless(db, common.GEMMQuantMode.int4_wo)
    db.set_transfer_policy("balanced")
    try:
        with pytest.raises(EmpiricalNotImplementedError):
            db.query_gemm(16, 256, 512, common.GEMMQuantMode.int4_wo, database_mode=common.DatabaseMode.HYBRID)
    finally:
        db.set_transfer_policy(None)


def test_silicon_still_rejects_dataless_quant(comprehensive_perf_db):
    """SILICON never borrows: a data-less quant stays a typed perf-data miss."""
    with pytest.raises(PerfDataNotAvailableError):
        comprehensive_perf_db.query_gemm(
            16, 256, 512, common.GEMMQuantMode.int4_wo, database_mode=common.DatabaseMode.SILICON
        )


def test_hybrid_is_invariant_for_covered_quants(comprehensive_perf_db):
    """The ladder only extends behind the existing miss: covered queries are
    bit-unchanged under HYBRID."""
    db = comprehensive_perf_db
    for quant in (common.GEMMQuantMode.bfloat16, common.GEMMQuantMode.fp8):
        silicon = float(db.query_gemm(16, 256, 512, quant, database_mode=common.DatabaseMode.SILICON))
        hybrid = float(db.query_gemm(16, 256, 512, quant, database_mode=common.DatabaseMode.HYBRID))
        assert hybrid == silicon


def test_level_table_covers_every_gemm_quant_profile():
    """Every GEMMQuantMode profile must have a calibrated level line — the
    gate refuses XPROFILE admission for unlisted profiles by design, so a
    profile silently missing here would strand its quants in HYBRID."""
    for quant in common.GEMMQuantMode:
        assert util_empirical.quant_profile(quant) in _GEMM_QUANT_UTIL_LEVEL, quant

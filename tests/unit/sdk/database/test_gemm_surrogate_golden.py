# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Golden characterization of GEMM query behavior.

Locks the current ``query_gemm`` outputs (exact hit / 1-D-M interp / 3-D interp /
extrapolation) so the upcoming PerfSurrogate façade can be proven bit-identical.
Uses the deterministic ``stub_perf_db`` GEMM table (m∈{64,128}, n∈{128,256},
k∈{256,512})."""

import pytest

from aiconfigurator.sdk import common

pytestmark = pytest.mark.unit

# (m, n, k) probes: exact hits, 1-D-M interpolation at exact (n,k), interior 3-D
# interpolation, and M-axis extrapolation beyond the grid.
PROBES = [
    (64, 128, 256),  # exact
    (128, 256, 512),  # exact
    (96, 128, 256),  # 1-D M interp at exact (n,k)
    (96, 256, 512),  # 1-D M interp at exact (n,k)
    (96, 192, 384),  # interior 3-D interp
    (200, 128, 256),  # M extrapolation at exact (n,k)
]

# Captured from the current implementation (stub system spec → SOL-clamped end to
# end; this golden guards the façade refactor against any output change).
GOLDEN: dict[tuple[int, int, int], float] = {
    (64, 128, 256): 4194304.0,
    (128, 256, 512): 33554432.0,
    (96, 128, 256): 6291456.0,
    (96, 256, 512): 25165824.0,
    (96, 192, 384): 14705077.77806446,
    (200, 128, 256): 13107200.0,
}


def test_gemm_query_golden(stub_perf_db):
    db = stub_perf_db
    qm = common.GEMMQuantMode.bfloat16
    got = {}
    for m, n, k in PROBES:
        db.query_gemm.cache_clear()
        got[(m, n, k)] = float(db.query_gemm(m, n, k, qm))
    if not GOLDEN:
        # capture mode: print and skip until goldens are filled in
        for shape, lat in got.items():
            print(f"GOLDEN[{shape}] = {lat!r}")
        pytest.skip("golden not yet populated — see printed values")
    for shape, lat in got.items():
        assert lat == pytest.approx(GOLDEN[shape], rel=1e-12), f"{shape}: {lat} != {GOLDEN[shape]}"

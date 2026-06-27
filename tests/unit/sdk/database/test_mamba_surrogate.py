# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import LoadedOpData

pytestmark = pytest.mark.unit

_MAMBA_KEY = (4096, 128, 4, 128, 64, 8, 128)
_GDN_KEY = (4096, 16, 128, 16, 128, 4)


def _metric(latency: float) -> dict[str, float]:
    return {"latency": latency, "power": 10.0, "energy": latency * 10.0}


def test_mamba_context_sparse_curve_mesh_and_local_exterior(mutable_comprehensive_perf_db):
    db = mutable_comprehensive_perf_db
    table = {
        1: {1: _metric(1.0), 4: _metric(4.0), 8: _metric(8.0)},
        2: {1: _metric(2.0), 8: _metric(16.0)},
        4: {1: _metric(4.0), 4: _metric(16.0)},
    }
    db._mamba2_data = LoadedOpData(
        {"causal_conv1d_fn": {"context": {_MAMBA_KEY: table}}},
        common.PerfDataFilename.mamba2,
        "synthetic-mamba",
    )
    db._sparse_surrogate_cache = {}

    curve = db.query_mamba2("context", "causal_conv1d_fn", 2, 4, *_MAMBA_KEY)
    mesh = db.query_mamba2("context", "causal_conv1d_fn", 3, 4, *_MAMBA_KEY)
    exterior = db.query_mamba2("context", "causal_conv1d_fn", 4, 8, *_MAMBA_KEY)

    assert float(curve) == pytest.approx(8.0)
    assert float(mesh) == pytest.approx(12.0)
    assert float(exterior) == pytest.approx(32.0)
    assert exterior.energy == pytest.approx(320.0)

    missing_model = db.query_mamba2(
        "context",
        "causal_conv1d_fn",
        1,
        4,
        12345,
        *_MAMBA_KEY[1:],
    )
    assert missing_model.source == "sol"


def test_gdn_collector_alias_nearest_category_and_batch_exterior(mutable_comprehensive_perf_db):
    db = mutable_comprehensive_perf_db
    low = (_GDN_KEY[0], _GDN_KEY[1], _GDN_KEY[2], 8, _GDN_KEY[4], _GDN_KEY[5])
    data = {
        "fused_recurrent_gated_delta_rule": {
            "generation": {
                low: {1: _metric(8.0), 3: _metric(12.0)},
                _GDN_KEY: {1: _metric(16.0), 3: _metric(20.0)},
            }
        }
    }
    db._gdn_data = LoadedOpData(data, common.PerfDataFilename.gdn, "synthetic-gdn")
    db._sparse_surrogate_cache = {}

    interior = db.query_gdn(
        "generation",
        "fused_sigmoid_gating_delta_rule_update",
        2,
        None,
        _GDN_KEY[0],
        _GDN_KEY[1],
        _GDN_KEY[2],
        14,
        _GDN_KEY[4],
        _GDN_KEY[5],
    )
    exterior = db.query_gdn(
        "generation",
        "fused_sigmoid_gating_delta_rule_update",
        6,
        None,
        _GDN_KEY[0],
        _GDN_KEY[1],
        _GDN_KEY[2],
        14,
        _GDN_KEY[4],
        _GDN_KEY[5],
    )

    assert float(interior) == pytest.approx(18.0)
    assert float(exterior) == pytest.approx(40.0)
    assert exterior.energy == pytest.approx(400.0)

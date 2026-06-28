# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import mamba
from aiconfigurator.sdk.perf_database import LoadedOpData

pytestmark = pytest.mark.unit

_MAMBA_KEY = (4096, 128, 4, 128, 64, 8, 128)
_GDN_KEY = (4096, 16, 128, 16, 128, 4)


def _metric(latency: float) -> dict[str, float]:
    return {"latency": latency, "power": 10.0, "energy": latency * 10.0}


def _mamba_row(phase: str, latency: float, *, batch_size: int = 1, seq_len: int = 1) -> dict[str, object]:
    return {
        "kernel_source": "causal_conv1d_fn" if phase == "context" else "causal_conv1d_update",
        "phase": phase,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "d_model": _MAMBA_KEY[0],
        "d_state": _MAMBA_KEY[1],
        "d_conv": _MAMBA_KEY[2],
        "nheads": _MAMBA_KEY[3],
        "head_dim": _MAMBA_KEY[4],
        "n_groups": _MAMBA_KEY[5],
        "chunk_size": _MAMBA_KEY[6],
        "latency": latency,
        "power": 10.0,
    }


def test_load_mamba2_data_keeps_generation_row_and_context_shape(monkeypatch):
    rows = [_mamba_row("generation", 2.0), _mamba_row("context", 3.0, seq_len=4)]
    monkeypatch.setattr(mamba, "_read_filtered_rows", lambda _: rows)

    data = mamba.load_mamba2_data("unused")

    assert data["causal_conv1d_update"]["generation"][_MAMBA_KEY][1] == _metric(2.0)
    assert data["causal_conv1d_fn"]["context"][_MAMBA_KEY][1][4] == _metric(3.0)


def test_load_mamba2_data_generation_conflict_keeps_first_source(monkeypatch):
    rows = [_mamba_row("generation", 2.0), _mamba_row("generation", 99.0)]
    monkeypatch.setattr(mamba, "_read_filtered_rows", lambda _: rows)

    data = mamba.load_mamba2_data("unused")

    assert data["causal_conv1d_update"]["generation"][_MAMBA_KEY][1] == _metric(2.0)


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


def test_gdn_chunked_sol_counts_partial_chunk(mutable_comprehensive_perf_db):
    db = mutable_comprehensive_perf_db
    db._gdn_data = LoadedOpData({}, common.PerfDataFilename.gdn, "empty-gdn")

    result = db.query_gdn("context", "chunk_gated_delta_rule", 1, 65, 1, 1, 1, 1, 1, 1)

    expected_bytes = 402  # 65 tokens plus two read/write h_chunks buffers.
    expected = expected_bytes / db.system_spec["gpu"]["mem_bw"] * 1000
    assert float(result) == pytest.approx(expected)

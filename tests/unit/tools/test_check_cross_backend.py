# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for check_cross_backend's review findings:
op_name partitioning, NaN kernel_source visibility, multi-component latency,
count-aware baseline ratchet, and systematic-offset uniformity."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "perf_database"))
from check_cross_backend import (
    _load_op_table,
    _SchemaUnsupportedError,
    detect_systematic_offsets,
    evaluate_gate,
    snapshot_baseline,
)

pytestmark = pytest.mark.unit


def _write(tmp_path: Path, header: str, rows: list[str]) -> Path:
    p = tmp_path / "some_perf.txt"
    p.write_text("\n".join([header, *rows]) + "\n")
    return p


def test_op_name_partitions_the_min_reduction(tmp_path):
    """Two logical ops sharing the same numeric shape (mla_gen_pre/post in
    mla_bmm_perf) must stay separate rows, never be min-merged or compared."""
    path = _write(
        tmp_path,
        "framework,version,kernel_source,op_name,latency,num_tokens",
        [
            "sglang,1.0,k,mla_gen_pre,2.0,16",
            "sglang,1.0,k,mla_gen_post,1.0,16",
        ],
    )
    table, _, _ = _load_op_table(path, "sglang", "1.0")
    assert "op_name" in table.shape_cols
    assert len(table.frame) == 2
    assert sorted(table.frame["op_name"]) == ["mla_gen_post", "mla_gen_pre"]


def test_nan_kernel_source_nonpositive_rows_are_counted(tmp_path):
    """A corrupt (latency <= 0) row must not vanish because its label is
    missing — value_counts drops NaN groups by default."""
    path = _write(
        tmp_path,
        "framework,version,kernel_source,latency,m",
        [
            "vllm,1.0,,0.0,16",
            "vllm,1.0,k,1.0,16",
        ],
    )
    table, npos_by_ks, _ = _load_op_table(path, "vllm", "1.0")
    assert npos_by_ks == {"<unknown>": 1}
    assert table is not None


def test_component_latency_is_summed_like_the_consumer(tmp_path):
    """wideep_deepep_ll stores combine+dispatch; the runtime consumer sums
    them, so the checker must audit the sum (and derived bandwidth columns
    must stay out of the shape key)."""
    path = _write(
        tmp_path,
        "framework,version,kernel_source,combine_avg_t_us,dispatch_avg_t_us,combine_bandwidth_gbps,num_token",
        ["sglang,1.0,deepep,3.0,4.0,100.0,16"],
    )
    table, _, _ = _load_op_table(path, "sglang", "1.0")
    assert float(table.frame["latency"].iloc[0]) == 7.0
    assert table.shape_cols == ["num_token"]
    assert table.noise_scale == 1000.0  # microsecond-unit components


def test_unsupported_schema_raises_explicitly(tmp_path):
    path = _write(
        tmp_path,
        "framework,version,kernel_source,some_timing,m",
        ["vllm,1.0,k,1.0,16"],
    )
    with pytest.raises(_SchemaUnsupportedError):
        _load_op_table(path, "vllm", "1.0")


def _npos(system: str, rows: int) -> dict:
    return {
        "kind": "nonpositive_latency",
        "system": system,
        "op_file": "x_perf.parquet",
        "backend": "sglang",
        "version": "1.0",
        "rows": rows,
        "by_kernel_source": {"k": rows},
    }


def test_ratchet_flags_count_growth_inside_a_known_group():
    """Baselining 1 nonpositive row must NOT suppress the same group growing
    to 500 rows — the excess is new findings."""
    thresholds = {"nonpositive_latency": 0}
    baseline = snapshot_baseline([_npos("h200_sxm", 1)])
    breaches, tallies, suppressed = evaluate_gate([_npos("h200_sxm", 500)], thresholds, baseline)
    assert tallies["nonpositive_latency"] == 499
    assert suppressed == 1
    assert breaches == {"nonpositive_latency": 499}
    # And the unchanged group stays fully suppressed.
    breaches, tallies, _ = evaluate_gate([_npos("h200_sxm", 1)], thresholds, baseline)
    assert not breaches and not tallies


def test_ratchet_point_findings_suppressed_by_key():
    point = {
        "kind": "pair_outlier",
        "system": "h200_sxm",
        "op_file": "gemm_perf.parquet",
        "pair": "vllm/0.24.0 vs trtllm/1.3.0rc10",
        "shape": {"m": 1, "n": 2, "k": 3},
    }
    thresholds = {"pair_outlier": 0}
    baseline = snapshot_baseline([point])
    assert evaluate_gate([point], thresholds, baseline)[0] == {}
    # Version bumps must not invalidate the key.
    bumped = {**point, "pair": "vllm/0.25.0 vs trtllm/1.4.0"}
    assert evaluate_gate([bumped], thresholds, baseline)[0] == {}
    # A different shape IS a new finding.
    other = {**point, "shape": {"m": 9, "n": 2, "k": 3}}
    assert evaluate_gate([other], thresholds, baseline)[0] == {"pair_outlier": 1}


def _pair_summary(system: str, ratio: float) -> dict:
    return {
        "kind": "pair_summary",
        "system": system,
        "op_file": "x_perf.parquet",
        "pair": "vllm/1.0 vs trtllm/1.0",
        "median_ratio": ratio,
        "kernel_sources_a": ["ka"],
        "kernel_sources_b": ["kb"],
    }


def test_systematic_offset_requires_near_constant_multiplier():
    """Same-direction 1.2x/4x/20x is not a configuration-level offset."""
    scattered = [_pair_summary(s, r) for s, r in [("a", 1.2), ("b", 4.0), ("c", 20.0)]]
    assert detect_systematic_offsets(scattered, 1.15, 3, 2.0, []) == []
    uniform = [_pair_summary(s, r) for s, r in [("a", 1.4), ("b", 1.5), ("c", 1.6)]]
    offsets = detect_systematic_offsets(uniform, 1.15, 3, 2.0, [])
    assert len(offsets) == 1 and offsets[0]["slow_backend"] == "vllm"

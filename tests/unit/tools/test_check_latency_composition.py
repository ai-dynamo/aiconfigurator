# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The scan must derive a reproducible latency metric for every perf table.

Two properties are covered:

* Latency composition — the DeepEP dispatch/combine tables have no single
  `latency` column: the SDK sums component columns at load time
  (sdk/operations/moe.py). If the scan picked one component instead, the
  manifest's row counts and divergence stats would describe a fragment of the
  latency, and the normal-mode table would be dropped entirely for having no
  recognised latency column.
* Merge order — `record_row` is last-write-wins per (shape, framework), so the
  merge must walk files in a fixed order or identical runs disagree.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "perf_database"))
from check_kernel_source import (
    _build_shape_key,
    _compose_latency,
    _resolve_latency_columns,
    _scan_one_file,
    scan,
)

pytestmark = pytest.mark.unit

_LL_HEADER = [
    "framework",
    "version",
    "device",
    "op_name",
    "node_num",
    "kernel_source",
    "hidden_size",
    "num_token",
    "num_topk",
    "num_experts",
    "combine_avg_t_us",
    "combine_bandwidth_gbps",
    "dispatch_avg_t_us",
    "dispatch_bandwidth_gbps",
]

_NORMAL_HEADER = [
    "framework",
    "version",
    "device",
    "op_name",
    "node_num",
    "kernel_source",
    "hidden_size",
    "num_token",
    "num_topk",
    "num_experts",
    "dispatch_sms",
    "dispatch_transmit_us",
    "dispatch_notify_us",
    "combine_sms",
    "combine_transmit_us",
    "combine_notify_us",
]


def test_single_latency_column_tables_keep_their_metric():
    assert _resolve_latency_columns(["hidden_size", "latency"]) == ("latency",)
    assert _resolve_latency_columns(["hidden_size", "avg_ms"]) == ("avg_ms",)


def test_latency_wins_over_avg_ms_when_both_present():
    assert _resolve_latency_columns(["latency", "avg_ms"]) == ("latency",)


def test_deepep_ll_resolves_to_dispatch_plus_combine():
    assert _resolve_latency_columns(_LL_HEADER) == ("dispatch_avg_t_us", "combine_avg_t_us")


def test_deepep_normal_resolves_to_all_four_components():
    assert _resolve_latency_columns(_NORMAL_HEADER) == (
        "dispatch_transmit_us",
        "dispatch_notify_us",
        "combine_transmit_us",
        "combine_notify_us",
    )


def test_unknown_schema_still_yields_no_metric():
    assert _resolve_latency_columns(["hidden_size", "bandwidth_gbps"]) is None


def test_compose_latency_sums_components():
    row = {
        "dispatch_transmit_us": 1.5,
        "dispatch_notify_us": 0.25,
        "combine_transmit_us": 2.0,
        "combine_notify_us": 0.25,
    }
    assert _compose_latency(row, tuple(row)) == pytest.approx(4.0)


@pytest.mark.parametrize("bad", [None, ""])
def test_compose_latency_rejects_missing_component(bad):
    """A partial sum would silently understate latency, so refuse to compose."""
    row = {"dispatch_avg_t_us": 1.0, "combine_avg_t_us": bad}
    assert _compose_latency(row, ("dispatch_avg_t_us", "combine_avg_t_us")) is None


def test_compose_latency_rejects_unparseable_component():
    row = {"dispatch_avg_t_us": 1.0, "combine_avg_t_us": "n/a"}
    assert _compose_latency(row, ("dispatch_avg_t_us", "combine_avg_t_us")) is None


def test_shape_key_excludes_every_latency_component():
    row = dict.fromkeys(_NORMAL_HEADER, "x")
    latency_cols = _resolve_latency_columns(_NORMAL_HEADER)
    keys = dict(_build_shape_key(row, _NORMAL_HEADER, latency_cols))
    assert not set(keys) & set(latency_cols)
    # Shape dimensions the SDK keys on must survive.
    assert {"node_num", "hidden_size", "num_token", "num_topk", "num_experts", "dispatch_sms"} <= set(keys)


def _write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    lines = [",".join(header)]
    lines.extend(",".join(str(row.get(column, "")) for column in header) for row in rows)
    path.write_text("\n".join(lines) + "\n")


def test_scan_one_file_records_composed_latency(tmp_path):
    """End-to-end: a normal-schema table is scanned, not skipped for lack of `latency`."""
    row = {
        "framework": "sglang",
        "version": "0.5.10",
        "kernel_source": "deepep",
        "node_num": 1,
        "hidden_size": 7168,
        "num_token": 128,
        "num_topk": 8,
        "num_experts": 256,
        "dispatch_sms": 24,
        "dispatch_transmit_us": 10.0,
        "dispatch_notify_us": 1.0,
        "combine_sms": 24,
        "combine_transmit_us": 20.0,
        "combine_notify_us": 2.0,
    }
    path = tmp_path / "wideep_deepep_normal_perf.txt"
    _write_csv(path, _NORMAL_HEADER, [row])

    result = _scan_one_file("h200_sxm", "sglang", "0.5.10", path)

    assert result.rows_scanned == 1
    assert result.rows_skipped == 0
    assert len(result.records) == 1
    *_, latency = result.records[0]
    assert latency == pytest.approx(33.0)


def test_merge_order_is_file_order_not_completion_order(tmp_path):
    """Two versions measuring one shape must always resolve to the later file's value.

    `record_row` overwrites the per-(shape, framework) latency, so merging in
    thread-completion order made `median_pct_divergence` flap between runs.
    """
    header = ["framework", "version", "kernel_source", "hidden_size", "latency"]
    for version, latency in (("0.1", 1.0), ("0.2", 2.0)):
        version_dir = tmp_path / "h200_sxm" / "gemm" / "sglang" / version
        version_dir.mkdir(parents=True)
        _write_csv(
            version_dir / "gemm_perf.parquet.txt",
            header,
            [
                {
                    "framework": "sglang",
                    "version": version,
                    "kernel_source": "k",
                    "hidden_size": 4096,
                    "latency": latency,
                }
            ],
        )

    for _ in range(5):
        groups = scan(tmp_path)
        (group,) = groups.values()
        (by_framework,) = group.latency_by_shape_framework.values()
        assert by_framework["sglang"] == pytest.approx(2.0)

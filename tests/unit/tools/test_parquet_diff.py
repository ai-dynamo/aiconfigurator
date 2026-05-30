# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pyarrow as pa
import pytest

pytestmark = pytest.mark.unit

PARQUET_DIFF = Path(__file__).resolve().parents[3] / "tools" / "perf_database" / "parquet_diff.py"


@pytest.fixture
def parquet_diff_module():
    spec = importlib.util.spec_from_file_location("parquet_diff", PARQUET_DIFF)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _snapshot(parquet_diff_module, path: str, rows: list[dict[str, object]]):
    return parquet_diff_module.Snapshot(path=path, table=pa.Table.from_pylist(rows))


def test_legacy_perf_policy_flags_added_and_modified_text_files(parquet_diff_module):
    entries = [
        parquet_diff_module.DiffEntry("A", "src/aiconfigurator/systems/data/h100/gemm_perf.txt"),
        parquet_diff_module.DiffEntry("M", "src/aiconfigurator/systems/data/h100/moe_perf.txt"),
        parquet_diff_module.DiffEntry("D", "src/aiconfigurator/systems/data/h100/nccl_perf.txt"),
        parquet_diff_module.DiffEntry("A", "src/aiconfigurator/systems/data/h100/gemm_perf.parquet"),
    ]

    legacy_changes = parquet_diff_module.find_legacy_perf_changes(entries)

    assert [entry.status for entry in legacy_changes] == ["A", "M"]
    assert parquet_diff_module.should_fail_strict([], legacy_changes)


def test_legacy_perf_policy_allows_text_file_deletions(parquet_diff_module):
    entries = [
        parquet_diff_module.DiffEntry("D", "src/aiconfigurator/systems/data/h100/gemm_perf.txt"),
    ]

    legacy_changes = parquet_diff_module.find_legacy_perf_changes(entries)
    report = parquet_diff_module.render_report(
        base_ref="origin/main",
        head_ref="HEAD",
        entries=entries,
        comparisons=[],
        legacy_perf_changes=legacy_changes,
    )

    assert legacy_changes == []
    assert not parquet_diff_module.should_fail_strict([], legacy_changes)
    assert "- Legacy `*_perf.txt` files added or modified: 0" in report


def test_row_diff_writes_added_removed_and_modified_artifacts(parquet_diff_module, tmp_path):
    base = _snapshot(
        parquet_diff_module,
        "gemm_perf.parquet",
        [
            {"framework": "vllm", "gemm_dtype": "fp8", "m": 1, "n": 16, "k": 16, "latency": 1.0},
            {"framework": "vllm", "gemm_dtype": "fp8", "m": 2, "n": 16, "k": 16, "latency": 2.0},
            {"framework": "vllm", "gemm_dtype": "fp8", "m": 3, "n": 16, "k": 16, "latency": 3.0},
        ],
    )
    head = _snapshot(
        parquet_diff_module,
        "gemm_perf.parquet",
        [
            {"framework": "vllm", "gemm_dtype": "fp8", "m": 1, "n": 16, "k": 16, "latency": 1.5},
            {"framework": "vllm", "gemm_dtype": "fp8", "m": 2, "n": 16, "k": 16, "latency": 2.0},
            {"framework": "vllm", "gemm_dtype": "fp8", "m": 4, "n": 16, "k": 16, "latency": 4.0},
        ],
    )

    row_diff = parquet_diff_module._diff_snapshots(
        "src/aiconfigurator/systems/data/h100/vllm/0.19.0/gemm_perf.parquet",
        base,
        head,
        detail_dir=tmp_path,
    )

    assert row_diff.added_rows == 1
    assert row_diff.removed_rows == 1
    assert row_diff.modified_rows == 1
    assert set(row_diff.detail_files) == {"added", "removed", "modified"}

    modified_path = tmp_path / row_diff.detail_files["modified"]
    with modified_path.open(newline="") as f:
        modified_rows = list(csv.DictReader(f))

    assert modified_rows == [
        {
            "framework": "vllm",
            "gemm_dtype": "fp8",
            "m": "1",
            "n": "16",
            "k": "16",
            "latency__base": "1.0",
            "latency__head": "1.5",
        }
    ]


def test_row_diff_pairs_duplicate_keys_within_key(parquet_diff_module, tmp_path):
    base = _snapshot(
        parquet_diff_module,
        "moe_perf.parquet",
        [
            {"framework": "sglang", "moe_dtype": "fp8", "num_tokens": 1, "latency": 1.0},
            {"framework": "sglang", "moe_dtype": "fp8", "num_tokens": 1, "latency": 1.1},
        ],
    )
    head = _snapshot(
        parquet_diff_module,
        "moe_perf.parquet",
        [
            {"framework": "sglang", "moe_dtype": "fp8", "num_tokens": 1, "latency": 1.0},
            {"framework": "sglang", "moe_dtype": "fp8", "num_tokens": 1, "latency": 1.2},
        ],
    )

    row_diff = parquet_diff_module._diff_snapshots(
        "src/aiconfigurator/systems/data/h100/sglang/0.5.10/moe_perf.parquet",
        base,
        head,
        detail_dir=tmp_path,
    )

    assert row_diff.added_rows == 0
    assert row_diff.removed_rows == 0
    assert row_diff.modified_rows == 1
    assert row_diff.note == "duplicate keys; unmatched rows paired within key"


def test_report_includes_row_level_counts(parquet_diff_module):
    comparison = parquet_diff_module.Comparison(
        path="src/aiconfigurator/systems/data/h100/vllm/0.19.0/gemm_perf.parquet",
        base_path="src/aiconfigurator/systems/data/h100/vllm/0.19.0/gemm_perf.parquet",
        status="M",
        base_rows=3,
        head_rows=3,
        columns_match=True,
        content_match=False,
        base_hash="aaa",
        head_hash="bbb",
        row_diff=parquet_diff_module.RowDiff(
            key_columns=["framework", "gemm_dtype", "m", "n", "k"],
            added_rows=1,
            removed_rows=1,
            modified_rows=1,
            detail_files={"added": "gemm.added.csv", "removed": "gemm.removed.csv", "modified": "gemm.modified.csv"},
            detail_previews={
                "modified": "framework,gemm_dtype,m,n,k,latency__base,latency__head\nvllm,fp8,1,16,16,1.0,1.5"
            },
        ),
    )

    report = parquet_diff_module.render_report(
        base_ref="origin/main",
        head_ref="HEAD",
        entries=[parquet_diff_module.DiffEntry("M", comparison.path)],
        comparisons=[comparison],
        legacy_perf_changes=[],
    )

    assert "- Row-level changes: +1 / -1 / ~1" in report
    assert "| M | src/aiconfigurator/systems/data/h100/vllm/0.19.0/gemm_perf.parquet | 3 | 3 | 1 | 1 | 1 |" in report
    assert "Exact row-level CSVs are attached" in report
    assert "### Per-File Row Diff Preview" in report
    assert "latency__base,latency__head" in report

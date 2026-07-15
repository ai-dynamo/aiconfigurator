# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import importlib.util
import math
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
        parquet_diff_module.DiffEntry("A", "aic-core/src/aiconfigurator_core/systems/data/h100/gemm_perf.txt"),
        parquet_diff_module.DiffEntry("M", "aic-core/src/aiconfigurator_core/systems/data/h100/moe_perf.txt"),
        parquet_diff_module.DiffEntry("D", "aic-core/src/aiconfigurator_core/systems/data/h100/nccl_perf.txt"),
        parquet_diff_module.DiffEntry("A", "aic-core/src/aiconfigurator_core/systems/data/h100/gemm_perf.parquet"),
    ]

    legacy_changes = parquet_diff_module.find_legacy_perf_changes(entries)

    assert [entry.status for entry in legacy_changes] == ["A", "M"]
    assert parquet_diff_module.should_fail_strict([], legacy_changes)


def test_legacy_perf_policy_allows_text_file_deletions(parquet_diff_module):
    entries = [
        parquet_diff_module.DiffEntry("D", "aic-core/src/aiconfigurator_core/systems/data/h100/gemm_perf.txt"),
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
        "aic-core/src/aiconfigurator_core/systems/data/h100/vllm/0.19.0/gemm_perf.parquet",
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
        "aic-core/src/aiconfigurator_core/systems/data/h100/sglang/0.5.10/moe_perf.parquet",
        base,
        head,
        detail_dir=tmp_path,
    )

    assert row_diff.added_rows == 0
    assert row_diff.removed_rows == 0
    assert row_diff.modified_rows == 1
    assert row_diff.note == "duplicate keys; unmatched rows paired within key"


def test_measurement_only_columns_use_full_row_delta(parquet_diff_module, tmp_path):
    base = _snapshot(parquet_diff_module, "latency_perf.parquet", [{"latency": 1.0}, {"latency": 2.0}])
    head = _snapshot(parquet_diff_module, "latency_perf.parquet", [{"latency": 2.0}, {"latency": 3.0}])

    row_diff = parquet_diff_module._diff_snapshots(
        "aic-core/src/aiconfigurator_core/systems/data/h100/vllm/0.19.0/latency_perf.parquet",
        base,
        head,
        detail_dir=tmp_path,
    )

    assert row_diff.key_columns == []
    assert row_diff.added_rows == 1
    assert row_diff.removed_rows == 1
    assert row_diff.modified_rows == 0
    assert row_diff.note == "unavailable keys; exact full-row add/remove diff used"


def test_full_row_delta_preserves_original_nested_values(parquet_diff_module):
    row = {"latency": float("nan"), "power": {"rails": ["gpu", "soc"]}}

    added_rows, removed_rows = parquet_diff_module._counter_rows_delta(
        base_rows=[],
        head_rows=[row],
        columns=["latency", "power"],
    )
    unmatched_rows = parquet_diff_module._unmatched_full_rows(
        [row],
        matched_rows=parquet_diff_module.Counter(),
        columns=["latency", "power"],
    )

    assert removed_rows == []
    assert added_rows[0] is row
    assert unmatched_rows[0] is row
    assert math.isnan(added_rows[0]["latency"])
    assert isinstance(added_rows[0]["power"], dict)


def test_full_diff_artifacts_include_every_changed_perf_file(parquet_diff_module, tmp_path, monkeypatch):
    entries = [
        parquet_diff_module.DiffEntry("M", "aic-core/src/aiconfigurator_core/systems/data/h100/gemm_perf.parquet"),
        parquet_diff_module.DiffEntry("A", "aic-core/src/aiconfigurator_core/systems/data/h100/new_perf.parquet"),
        parquet_diff_module.DiffEntry("D", "aic-core/src/aiconfigurator_core/systems/data/h100/moe_perf.txt"),
        parquet_diff_module.DiffEntry(
            "R100",
            "aic-core/src/aiconfigurator_core/systems/data/h100/nccl_perf.parquet",
            old_path="aic-core/src/aiconfigurator_core/systems/data/h100/comm_perf.parquet",
        ),
    ]
    file_lines = {
        ("base", "aic-core/src/aiconfigurator_core/systems/data/h100/gemm_perf.parquet"): [
            "framework,latency",
            "vllm,1.0",
        ],
        ("head", "aic-core/src/aiconfigurator_core/systems/data/h100/gemm_perf.parquet"): [
            "framework,latency",
            "vllm,1.5",
        ],
        ("head", "aic-core/src/aiconfigurator_core/systems/data/h100/new_perf.parquet"): [
            "framework,latency",
            "trtllm,3.0",
        ],
        ("base", "aic-core/src/aiconfigurator_core/systems/data/h100/moe_perf.txt"): [
            "framework,latency",
            "sglang,2.0",
        ],
        ("base", "aic-core/src/aiconfigurator_core/systems/data/h100/comm_perf.parquet"): [
            "framework,bandwidth",
            "nccl,900",
        ],
        ("head", "aic-core/src/aiconfigurator_core/systems/data/h100/nccl_perf.parquet"): [
            "framework,bandwidth",
            "nccl,950",
        ],
    }

    monkeypatch.setattr(parquet_diff_module, "_git_file_exists", lambda ref, path: (ref, path) in file_lines)
    monkeypatch.setattr(parquet_diff_module, "_file_text_lines", lambda ref, path: file_lines[(ref, path)])

    artifacts = parquet_diff_module._write_full_diff_artifacts(
        detail_dir=tmp_path,
        base_ref="base",
        head_ref="head",
        entries=entries,
    )
    parquet_diff_module._write_diff_manifest(tmp_path, artifacts)

    assert [artifact.diff_file for artifact in artifacts] == [
        "diffs/aic-core/src/aiconfigurator_core/systems/data/h100/gemm_perf.parquet.diff",
        "diffs/aic-core/src/aiconfigurator_core/systems/data/h100/new_perf.parquet.diff",
        "diffs/aic-core/src/aiconfigurator_core/systems/data/h100/moe_perf.txt.diff",
        "diffs/aic-core/src/aiconfigurator_core/systems/data/h100/nccl_perf.parquet.diff",
    ]
    assert "+vllm,1.5" in (tmp_path / artifacts[0].diff_file).read_text()
    assert "+trtllm,3.0" in (tmp_path / artifacts[1].diff_file).read_text()
    assert "-sglang,2.0" in (tmp_path / artifacts[2].diff_file).read_text()
    assert "comm_perf.parquet" in (tmp_path / artifacts[3].diff_file).read_text()
    assert "nccl_perf.parquet" in (tmp_path / artifacts[3].diff_file).read_text()
    assert "-nccl,900" in (tmp_path / artifacts[3].diff_file).read_text()
    assert "+nccl,950" in (tmp_path / artifacts[3].diff_file).read_text()

    with (tmp_path / "changed-files.csv").open(newline="") as f:
        manifest_rows = list(csv.DictReader(f))

    assert manifest_rows == [
        {
            "status": "M",
            "file": "aic-core/src/aiconfigurator_core/systems/data/h100/gemm_perf.parquet",
            "old_file": "",
            "full_diff_file": ("diffs/aic-core/src/aiconfigurator_core/systems/data/h100/gemm_perf.parquet.diff"),
        },
        {
            "status": "A",
            "file": "aic-core/src/aiconfigurator_core/systems/data/h100/new_perf.parquet",
            "old_file": "",
            "full_diff_file": ("diffs/aic-core/src/aiconfigurator_core/systems/data/h100/new_perf.parquet.diff"),
        },
        {
            "status": "D",
            "file": "aic-core/src/aiconfigurator_core/systems/data/h100/moe_perf.txt",
            "old_file": "",
            "full_diff_file": "diffs/aic-core/src/aiconfigurator_core/systems/data/h100/moe_perf.txt.diff",
        },
        {
            "status": "R100",
            "file": "aic-core/src/aiconfigurator_core/systems/data/h100/nccl_perf.parquet",
            "old_file": "aic-core/src/aiconfigurator_core/systems/data/h100/comm_perf.parquet",
            "full_diff_file": ("diffs/aic-core/src/aiconfigurator_core/systems/data/h100/nccl_perf.parquet.diff"),
        },
    ]


def test_report_includes_row_level_counts(parquet_diff_module):
    comparison = parquet_diff_module.Comparison(
        path="aic-core/src/aiconfigurator_core/systems/data/h100/vllm/0.19.0/gemm_perf.parquet",
        base_path="aic-core/src/aiconfigurator_core/systems/data/h100/vllm/0.19.0/gemm_perf.parquet",
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
                "added": "framework,gemm_dtype,m,n,k,latency\nvllm,fp8,4,16,16,4.0",
                "modified": "framework,gemm_dtype,m,n,k,latency__base,latency__head\nvllm,fp8,1,16,16,1.0,1.5",
            },
        ),
    )

    report = parquet_diff_module.render_report(
        base_ref="origin/main",
        head_ref="HEAD",
        entries=[parquet_diff_module.DiffEntry("M", comparison.path)],
        comparisons=[comparison],
        legacy_perf_changes=[],
        full_diff_artifacts=[
            parquet_diff_module.FullDiffArtifact(
                status="M",
                path=comparison.path,
                old_path=None,
                diff_file="diffs/aic-core/src/aiconfigurator_core/systems/data/h100/vllm/0.19.0/gemm_perf.parquet.diff",
            )
        ],
    )

    assert "- Row-level changes: +1 / -1 / ~1" in report
    assert "- Full per-file diff artifacts: 1 file under `parquet-diff-details/diffs/`" in report
    assert "### Other Parquet Changes" not in report
    assert "| M | aic-core/src/aiconfigurator_core/systems/data/h100/vllm/0.19.0/gemm_perf.parquet |" not in report
    assert "Full per-file unified diffs: `parquet-diff-details/diffs/` (1 file)" in report
    assert "Exact row-level CSVs: `parquet-diff-details/` (listed in `summary.csv`)" in report
    assert "**added rows** - full CSV: `parquet-diff-details/gemm.added.csv`" in report
    assert "vllm,fp8,4,16,16,4.0" in report
    assert "**modified rows** - full CSV: `parquet-diff-details/gemm.modified.csv`" in report
    assert "vllm,fp8,1,16,16,1.0,1.5" in report

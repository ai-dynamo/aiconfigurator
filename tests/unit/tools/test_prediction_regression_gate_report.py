# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the prediction-regression-gate snapshot report / gate driver."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tools.prediction_regression_gate import compare, grid, report

pytestmark = pytest.mark.unit

COMBO = "h200_sxm/trtllm/1.3.0rc10.csv"
TIER1_HEADER = [*compare.KEY_FIELDS, "status", "value_ms", "err"]


def _tier1_row(status: str = "OK", value_ms: str = "10.000000", err: str = "", **overrides) -> dict:
    row = dict.fromkeys(compare.KEY_FIELDS, "1")
    row.update(model="org/model", quant="default", phase="ctx", moe_tp="", moe_ep="")
    row.update(status=status, value_ms=value_ms, err=err)
    row.update(overrides)
    return row


def _write_snapshot(root: Path, combos: dict[str, list[dict]], tier2_rows: list[dict] | None = None) -> Path:
    for relpath, rows in combos.items():
        path = root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TIER1_HEADER, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    if tier2_rows is not None:
        with (root / "tier2.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "status", "ttft_ms", "tpot_ms"], lineterminator="\n")
            writer.writeheader()
            writer.writerows(tier2_rows)
    return root


def test_combo_only_on_one_side_reported_not_blocking(tmp_path: Path) -> None:
    old = _write_snapshot(tmp_path / "old", {COMBO: [_tier1_row()]})
    new = _write_snapshot(tmp_path / "new", {COMBO: [_tier1_row()], "b200_sxm/vllm/0.19.0.csv": [_tier1_row()]})
    results = report.compare_snapshots(old, new, rtol=1e-4)
    categories = [d.category for r in results for d in r.diffs]
    assert categories == ["ROWS_ADDED"]
    assert not any(c in compare.BLOCKING_CATEGORIES for c in categories)


def test_tier1_and_tier2_both_compared(tmp_path: Path) -> None:
    tier2 = [{"id": "dense_agg_c8", "status": "OK", "ttft_ms": "100.0", "tpot_ms": "10.0"}]
    tier2_broken = [{"id": "dense_agg_c8", "status": "ValueError", "ttft_ms": "", "tpot_ms": ""}]
    old = _write_snapshot(tmp_path / "old", {COMBO: [_tier1_row()]}, tier2)
    new = _write_snapshot(tmp_path / "new", {COMBO: [_tier1_row(status="DATA_MISS", value_ms="")]}, tier2_broken)
    results = report.compare_snapshots(old, new, rtol=1e-4)
    categories = sorted(d.category for r in results for d in r.diffs)
    assert categories == ["REGRESSION", "REGRESSION"]  # one tier-1, one tier-2


def test_main_exit_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def run(old: Path, new: Path) -> int:
        monkeypatch.setattr(
            "sys.argv",
            ["report.py", "--old", str(old), "--new", str(new), "--report-dir", str(tmp_path / "rep")],
        )
        return report.main()

    clean_old = _write_snapshot(tmp_path / "o1", {COMBO: [_tier1_row()]})
    clean_new = _write_snapshot(tmp_path / "n1", {COMBO: [_tier1_row()]})
    assert run(clean_old, clean_new) == 0

    drift_new = _write_snapshot(tmp_path / "n2", {COMBO: [_tier1_row(value_ms="12.000000")]})
    assert run(clean_old, drift_new) == 0  # drift reports, never blocks

    broken_new = _write_snapshot(tmp_path / "n3", {COMBO: [_tier1_row(status="INVALID", value_ms="", err="E")]})
    assert run(clean_old, broken_new) == 1  # OK -> INVALID blocks

    no_harness_old = tmp_path / "o-empty"
    no_harness_old.mkdir()
    assert run(no_harness_old, clean_new) == 0  # degraded: old side predates harness
    assert "no snapshot" in (tmp_path / "rep" / "summary.md").read_text()


def _write_silicon(path: Path, rows: list[dict]) -> Path:
    fields = [
        "id",
        "status",
        "predicted_ttft_ms",
        "predicted_tpot_ms",
        "silicon_ttft_ms",
        "silicon_tpot_ms",
        "ttft_rel_err",
        "tpot_rel_err",
        "predicted_with_version",
        "err",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({**dict.fromkeys(fields, ""), **row} for row in rows)
    return path


def test_silicon_section_reports_accuracy_movement(tmp_path: Path) -> None:
    old = _write_silicon(
        tmp_path / "old.csv",
        [{"id": "ref1", "status": "OK", "ttft_rel_err": "0.30", "tpot_rel_err": "0.10"}],
    )
    new = _write_silicon(
        tmp_path / "new.csv",
        [{"id": "ref1", "status": "OK", "ttft_rel_err": "0.10", "tpot_rel_err": "0.10"}],
    )
    section = report.render_silicon_section(old, new)
    assert section is not None
    assert "accuracy moved on 1 point(s)" in section
    assert "+30.0% -> +10.0%" in section


def test_silicon_section_degrades_without_old_side(tmp_path: Path) -> None:
    new = _write_silicon(
        tmp_path / "new.csv", [{"id": "ref1", "status": "OK", "ttft_rel_err": "0.2", "tpot_rel_err": "0.1"}]
    )
    section = report.render_silicon_section(tmp_path / "missing.csv", new)
    assert section is not None and "1/1 refs predicted" in section
    assert report.render_silicon_section(tmp_path / "missing.csv", tmp_path / "also-missing.csv") is None


def test_version_sort_is_version_aware() -> None:
    ordered = sorted(["0.5.9", "0.5.10", "1.3.0", "1.3.0rc10", "1.3.0rc2"], key=grid._version_sort_key)
    assert ordered == ["0.5.9", "0.5.10", "1.3.0rc2", "1.3.0rc10", "1.3.0"]

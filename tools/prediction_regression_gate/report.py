#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare two prediction-regression-gate snapshots and produce the gate report.

Inputs are two snapshot directories (see collect_static.py / run_tier2.py):

  <side>/<system>/<backend>/<version>.csv   tier-1 combos
  <side>/tier2.csv                          tier-2 scheduling configs

Outputs:
  <report-dir>/drift_report.csv   every difference, category-tagged
  <report-dir>/summary.md         markdown for the CI step summary / PR comment

Exit code is 1 only if a blocking difference exists (REGRESSION: a combo that
ran OK on the old side stopped working on the new side). Everything else —
drift, gains, added/removed rows — is reported for human review, never blocked
on: with old-vs-new there is no baseline to refresh, the report itself is the
review artifact.

If the old side has no snapshot (base revision predates the harness), the
report degrades to new-side statistics and exits 0.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))  # make `tools.` importable when run as a script

from tools.accuracy_tracking.run_silicon import SILICON_FILENAME
from tools.prediction_regression_gate import compare
from tools.prediction_regression_gate.run_tier2 import TIER2_FILENAME

CATEGORY_ORDER = ("REGRESSION", "STATUS_CHANGE", "DRIFT", "GAIN", "ROWS_ADDED", "ROWS_REMOVED")
MAX_INLINE_DIFFS = 50


def _combo_relpaths(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    return {str(p.relative_to(root)) for p in root.rglob("*.csv") if len(p.relative_to(root).parts) == 3}


def _snapshot_stats(root: Path) -> dict[str, int]:
    stats: dict[str, int] = {}
    for relpath in _combo_relpaths(root):
        with (root / relpath).open(newline="") as f:
            for row in csv.DictReader(f):
                stats[row["status"]] = stats.get(row["status"], 0) + 1
    return stats


def compare_snapshots(old_dir: Path, new_dir: Path, rtol: float) -> list[compare.ComboResult]:
    results: list[compare.ComboResult] = []
    old_combos, new_combos = _combo_relpaths(old_dir), _combo_relpaths(new_dir)

    for relpath in sorted(old_combos - new_combos):
        results.append(
            compare.ComboResult(
                relpath,
                diffs=[compare.Diff(relpath, ("*",), "ROWS_REMOVED", "combo present on old side, absent on new")],
            )
        )
    for relpath in sorted(new_combos - old_combos):
        results.append(
            compare.ComboResult(
                relpath,
                diffs=[compare.Diff(relpath, ("*",), "ROWS_ADDED", "combo produced on new side, absent on old")],
            )
        )
    for relpath in sorted(old_combos & new_combos):
        results.append(compare.compare_combo(relpath, old_dir / relpath, new_dir / relpath, rtol=rtol))

    old_tier2, new_tier2 = old_dir / TIER2_FILENAME, new_dir / TIER2_FILENAME
    if old_tier2.exists() and new_tier2.exists():
        results.append(compare.compare_tier2(old_tier2, new_tier2, rtol=rtol))
    elif old_tier2.exists() or new_tier2.exists():
        side = "new" if new_tier2.exists() else "old"
        results.append(
            compare.ComboResult(
                "tier2",
                diffs=[
                    compare.Diff(
                        "tier2",
                        ("*",),
                        "ROWS_ADDED" if side == "new" else "ROWS_REMOVED",
                        f"tier2 snapshot only present on {side} side",
                    )
                ],
            )
        )
    return results


def _load_silicon(path: Path) -> dict[str, dict]:
    with path.open(newline="") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def _median_abs_err(rows: list[dict], field: str) -> float:
    errs = sorted(abs(float(r[field])) for r in rows)
    return errs[len(errs) // 2] if errs else 0.0


def render_silicon_section(old_path: Path, new_path: Path) -> str | None:
    """Accuracy-vs-silicon section (report-only; the nightly tracked metric)."""
    if not new_path.exists():
        return None
    new = _load_silicon(new_path)
    new_ok = [r for r in new.values() if r["status"] == "OK"]

    lines = ["### Accuracy vs silicon (report-only)", ""]
    lines.append(
        f"- new: {len(new_ok)}/{len(new)} refs predicted; median |rel err| "
        f"ttft {_median_abs_err(new_ok, 'ttft_rel_err'):.1%}, tpot {_median_abs_err(new_ok, 'tpot_rel_err'):.1%}"
    )

    old = _load_silicon(old_path) if old_path.exists() else {}
    old_ok = [r for r in old.values() if r["status"] == "OK"]
    if old:
        lines.append(
            f"- old: {len(old_ok)}/{len(old)} refs predicted; median |rel err| "
            f"ttft {_median_abs_err(old_ok, 'ttft_rel_err'):.1%}, tpot {_median_abs_err(old_ok, 'tpot_rel_err'):.1%}"
        )
        moved = []
        for ref_id in sorted(new.keys() & old.keys()):
            o, n = old[ref_id], new[ref_id]
            if o["status"] != n["status"]:
                moved.append(f"`{ref_id}`: {o['status']} -> {n['status']}")
                continue
            if n["status"] != "OK":
                continue
            for metric in ("ttft_rel_err", "tpot_rel_err"):
                if abs(float(n[metric]) - float(o[metric])) > 1e-6:
                    moved.append(f"`{ref_id}`: {metric} {float(o[metric]):+.1%} -> {float(n[metric]):+.1%}")
        if moved:
            lines.append(f"- accuracy moved on {len(moved)} point(s):")
            lines.extend(f"  - {entry}" for entry in moved[:MAX_INLINE_DIFFS])
        else:
            lines.append("- no accuracy movement vs old side")
    lines.append("")
    return "\n".join(lines)


def render_markdown(results: list[compare.ComboResult], blocking: list[compare.Diff]) -> str:
    total_counts: dict[str, int] = {}
    for result in results:
        for category, count in result.counts.items():
            total_counts[category] = total_counts.get(category, 0) + count

    lines = ["## AIC Prediction Regression Gate (old vs new)", ""]
    rows_compared = sum(r.rows_compared for r in results)
    lines.append(f"Compared {rows_compared} rows across {len(results)} combos.")
    lines.append("")

    if not total_counts:
        lines.append("**No differences.**")
        return "\n".join(lines)

    lines.append("| category | count | gate |")
    lines.append("|---|---|---|")
    for category in CATEGORY_ORDER:
        if category in total_counts:
            gate = "**blocking**" if category in compare.BLOCKING_CATEGORIES else "report-only"
            lines.append(f"| {category} | {total_counts[category]} | {gate} |")
    lines.append("")

    if blocking:
        lines.append(f"### Blocking regressions ({len(blocking)})")
        lines.append("")
        for diff in blocking[:MAX_INLINE_DIFFS]:
            lines.append(f"- `{diff.render()}`")
        if len(blocking) > MAX_INLINE_DIFFS:
            lines.append(f"- ... and {len(blocking) - MAX_INLINE_DIFFS} more (see drift_report.csv artifact)")
        lines.append("")

    reported = [d for r in results for d in r.diffs if d.category not in compare.BLOCKING_CATEGORIES]
    if reported:
        lines.append(f"### Reported changes ({len(reported)})")
        lines.append("")
        by_combo: dict[str, dict[str, int]] = {}
        for diff in reported:
            combo_counts = by_combo.setdefault(diff.combo, {})
            combo_counts[diff.category] = combo_counts.get(diff.category, 0) + 1
        for combo in sorted(by_combo):
            counts = ", ".join(f"{cat}: {n}" for cat, n in sorted(by_combo[combo].items()))
            lines.append(f"- `{combo}` — {counts}")
        lines.append("")
        lines.append("Full detail in the `drift_report.csv` artifact.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old", type=Path, required=True, help="Snapshot dir collected on the base revision.")
    parser.add_argument("--new", type=Path, required=True, help="Snapshot dir collected on the head revision.")
    parser.add_argument("--report-dir", type=Path, default=Path("gate_report"))
    parser.add_argument("--rtol", type=float, default=compare.DEFAULT_RTOL)
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.report_dir / "summary.md"

    if not _combo_relpaths(args.new):
        print(f"error: new-side snapshot {args.new} is empty or missing", file=sys.stderr)
        return 2

    if not _combo_relpaths(args.old):
        stats = _snapshot_stats(args.new)
        summary = (
            "## AIC Prediction Regression Gate (old vs new)\n\n"
            "Old side has no snapshot (base revision predates the prediction-regression-gate harness).\n"
            f"New-side statistics: {stats}\n"
        )
        summary_path.write_text(summary)
        print(summary)
        return 0

    results = compare_snapshots(args.old, args.new, rtol=args.rtol)
    blocking = [d for r in results for d in r.diffs if d.category in compare.BLOCKING_CATEGORIES]

    compare.write_report(results, args.report_dir / "drift_report.csv")
    summary = render_markdown(results, blocking)
    silicon_section = render_silicon_section(args.old / SILICON_FILENAME, args.new / SILICON_FILENAME)
    if silicon_section:
        summary = f"{summary}\n\n{silicon_section}"
    summary_path.write_text(summary + "\n")
    print(summary)

    if blocking:
        print(f"\nFAIL: {len(blocking)} blocking regression(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

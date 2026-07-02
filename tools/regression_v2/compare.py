# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Baseline-vs-current comparison for the tier-1 static regression gate.

Pure CSV diffing — no SDK imports, so the gate itself is fast and testable.

Diff categories (all blocking; the fix is always "regenerate the baseline in
this PR so the diff is reviewed at rest"):

  REGRESSION   OK -> DATA_MISS / INVALID (a working combo stopped working)
  GAIN         DATA_MISS / INVALID -> OK (coverage gained; baseline stale)
  DRIFT        OK -> OK but |rel change| > rtol
  STATUS_CHANGE  non-OK status or error type changed
  ROWS_ADDED / ROWS_REMOVED  grid or model list changed
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

KEY_FIELDS = ("model", "tp", "pp", "adp", "moe_tp", "moe_ep", "quant", "phase", "bs", "isl")
DEFAULT_RTOL = 1e-4


@dataclass
class Diff:
    combo: str
    key: tuple
    category: str
    detail: str

    def render(self) -> str:
        key_text = "/".join(str(part) for part in self.key)
        return f"[{self.category}] {self.combo} :: {key_text} :: {self.detail}"


@dataclass
class ComboResult:
    combo: str
    diffs: list[Diff] = field(default_factory=list)
    rows_compared: int = 0

    @property
    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for diff in self.diffs:
            counts[diff.category] = counts.get(diff.category, 0) + 1
        return counts


def load_rows(path: Path) -> dict[tuple, dict]:
    with path.open(newline="") as f:
        return {tuple(row[k] for k in KEY_FIELDS): row for row in csv.DictReader(f)}


def compare_combo(combo: str, baseline_path: Path, current_path: Path, rtol: float = DEFAULT_RTOL) -> ComboResult:
    result = ComboResult(combo=combo)
    baseline = load_rows(baseline_path)
    current = load_rows(current_path)

    for key in baseline.keys() - current.keys():
        result.diffs.append(Diff(combo, key, "ROWS_REMOVED", "row present in baseline, absent in current run"))
    for key in current.keys() - baseline.keys():
        result.diffs.append(Diff(combo, key, "ROWS_ADDED", "row produced by current run, absent in baseline"))

    for key in baseline.keys() & current.keys():
        result.rows_compared += 1
        base, cur = baseline[key], current[key]
        base_status, cur_status = base["status"], cur["status"]

        if base_status == "OK" and cur_status != "OK":
            result.diffs.append(
                Diff(combo, key, "REGRESSION", f"OK ({base['value_ms']} ms) -> {cur_status} {cur['err']}".rstrip())
            )
        elif base_status != "OK" and cur_status == "OK":
            result.diffs.append(Diff(combo, key, "GAIN", f"{base_status} -> OK ({cur['value_ms']} ms)"))
        elif base_status == "OK" and cur_status == "OK":
            base_value, cur_value = float(base["value_ms"]), float(cur["value_ms"])
            denom = max(abs(base_value), 1e-9)
            rel = abs(cur_value - base_value) / denom
            if rel > rtol:
                result.diffs.append(
                    Diff(combo, key, "DRIFT", f"{base_value:.6f} -> {cur_value:.6f} ms (rel {rel:.2%})")
                )
        elif base_status != cur_status or base["err"] != cur["err"]:
            result.diffs.append(
                Diff(combo, key, "STATUS_CHANGE", f"{base_status} {base['err']} -> {cur_status} {cur['err']}".rstrip())
            )
    return result


def summarize(results: list[ComboResult], max_examples: int = 20) -> str:
    total_counts: dict[str, int] = {}
    for result in results:
        for category, count in result.counts.items():
            total_counts[category] = total_counts.get(category, 0) + count
    lines = [f"compared {sum(r.rows_compared for r in results)} rows across {len(results)} combos"]
    if not total_counts:
        lines.append("no differences")
        return "\n".join(lines)
    lines.append(f"differences: {total_counts}")
    shown = 0
    for result in results:
        for diff in result.diffs:
            if shown >= max_examples:
                lines.append(f"... and {sum(total_counts.values()) - shown} more (see report artifact)")
                return "\n".join(lines)
            lines.append(diff.render())
            shown += 1
    return "\n".join(lines)


def write_report(results: list[ComboResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["combo", *KEY_FIELDS, "category", "detail"])
        for result in results:
            for diff in result.diffs:
                writer.writerow([diff.combo, *diff.key, diff.category, diff.detail])

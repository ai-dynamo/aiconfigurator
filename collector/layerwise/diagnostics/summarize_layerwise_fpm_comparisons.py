#!/usr/bin/env python3
"""Summarize AIC-vs-FPM comparison CSVs into one accuracy table.

The comparison tools emit row-level CSVs. This helper reads a small manifest of
the row-level artifacts and rebuilds the curated summary table used while
tracking layerwise accuracy.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUMMARY_FIELDS = [
    "model",
    "tp",
    "moe_tp",
    "ep",
    "phase",
    "rows",
    "mape_pct",
    "max_abs_error_pct",
    "worst_shape",
    "worst_error_pct",
    "comparison_csv",
]


@dataclass(frozen=True)
class ManifestEntry:
    """One summary row to compute from a comparison CSV."""

    model: str
    tp: str
    moe_tp: str
    ep: str
    phase: str
    source_phase: str
    comparison_csv: Path
    comparison_csv_label: str


def _repo_root() -> Path:
    """Return the repository root from this diagnostics script location."""

    return Path(__file__).resolve().parents[3]


def _resolve_path(path_text: str, *, base_dir: Path) -> Path:
    """Resolve an absolute path or repo-relative path."""

    path = Path(path_text)
    if path.is_absolute():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return _repo_root() / path


def load_manifest(path: Path) -> list[ManifestEntry]:
    """Load summary entries from a CSV manifest."""

    entries: list[ManifestEntry] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            phase = str(row["phase"])
            source_phase = str(row.get("source_phase") or phase)
            comparison_csv_label = str(row["comparison_csv"])
            entries.append(
                ManifestEntry(
                    model=str(row["model"]),
                    tp=str(row["tp"]),
                    moe_tp=str(row["moe_tp"]),
                    ep=str(row["ep"]),
                    phase=phase,
                    source_phase=source_phase,
                    comparison_csv=_resolve_path(comparison_csv_label, base_dir=path.parent),
                    comparison_csv_label=comparison_csv_label,
                )
            )
    return entries


def _float(row: dict[str, Any], key: str) -> float:
    """Parse a required numeric CSV field."""

    return float(row[key])


def summarize_entry(entry: ManifestEntry, *, allow_empty: bool = False) -> dict[str, Any]:
    """Summarize one manifest entry into rows/MAPE/max/worst fields."""

    if not entry.comparison_csv.exists():
        raise FileNotFoundError(entry.comparison_csv)

    with entry.comparison_csv.open(newline="") as f:
        rows = [row for row in csv.DictReader(f) if str(row.get("phase", "")) == entry.source_phase]

    if not rows:
        if not allow_empty:
            raise ValueError(f"no rows for phase={entry.source_phase!r} in {entry.comparison_csv}")
        return {
            "model": entry.model,
            "tp": entry.tp,
            "moe_tp": entry.moe_tp,
            "ep": entry.ep,
            "phase": entry.phase,
            "rows": 0,
            "mape_pct": "",
            "max_abs_error_pct": "",
            "worst_shape": "",
            "worst_error_pct": "",
            "comparison_csv": entry.comparison_csv_label,
        }

    error_values = [_float(row, "error_pct") for row in rows]
    abs_errors = [abs(value) for value in error_values]
    worst_index = max(range(len(rows)), key=lambda index: abs_errors[index])
    worst_row = rows[worst_index]
    return {
        "model": entry.model,
        "tp": entry.tp,
        "moe_tp": entry.moe_tp,
        "ep": entry.ep,
        "phase": entry.phase,
        "rows": len(rows),
        "mape_pct": statistics.fmean(abs_errors),
        "max_abs_error_pct": abs_errors[worst_index],
        "worst_shape": worst_row.get("shape", ""),
        "worst_error_pct": error_values[worst_index],
        "comparison_csv": entry.comparison_csv_label,
    }


def summarize_manifest(path: Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    """Summarize all entries in a manifest."""

    return [summarize_entry(entry, allow_empty=allow_empty) for entry in load_manifest(path)]


def write_summary(rows: list[dict[str, Any]], output: Path) -> None:
    """Write summary rows to a CSV file."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="CSV manifest of comparison artifacts.")
    parser.add_argument("--output", type=Path, required=True, help="Output summary CSV.")
    parser.add_argument("--allow-empty", action="store_true", help="Emit blank summary rows for missing phases.")
    args = parser.parse_args(argv)

    write_summary(summarize_manifest(args.manifest, allow_empty=args.allow_empty), args.output)


if __name__ == "__main__":
    main()

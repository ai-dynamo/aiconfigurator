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
    "median_abs_error_pct",
    "p90_abs_error_pct",
    "p95_abs_error_pct",
    "max_abs_error_pct",
    "within_5",
    "within_10",
    "wmape_pct",
    "weighted_bias_pct",
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


def _maybe_float(row: dict[str, Any], key: str) -> float | None:
    """Parse an optional numeric CSV field."""

    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def _percentile(values: list[float], percentile: float) -> float:
    """Return a linearly interpolated percentile from a non-empty list."""

    if not values:
        raise ValueError("percentile requires at least one value")
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _weighted_error_metrics(rows: list[dict[str, Any]]) -> tuple[float | str, float | str]:
    """Return wMAPE and weighted bias when FPM/AIC latency columns are available."""

    fpm_sum = 0.0
    abs_delta_sum = 0.0
    signed_delta_sum = 0.0
    for row in rows:
        fpm_ms = _maybe_float(row, "fpm_ms")
        aic_ms = _maybe_float(row, "aic_ms")
        if fpm_ms is None or aic_ms is None:
            return "", ""
        if fpm_ms <= 0.0:
            continue
        fpm_sum += fpm_ms
        delta = aic_ms - fpm_ms
        abs_delta_sum += abs(delta)
        signed_delta_sum += delta
    if fpm_sum <= 0.0:
        return "", ""
    return abs_delta_sum / fpm_sum * 100.0, signed_delta_sum / fpm_sum * 100.0


def _summarize_rows(
    *,
    model: str,
    tp: str,
    moe_tp: str,
    ep: str,
    phase: str,
    rows: list[dict[str, Any]],
    comparison_csv: str,
    allow_empty: bool = False,
) -> dict[str, Any]:
    """Summarize already-filtered comparison rows."""

    if not rows:
        if not allow_empty:
            raise ValueError(f"no rows for phase={phase!r}")
        return {
            "model": model,
            "tp": tp,
            "moe_tp": moe_tp,
            "ep": ep,
            "phase": phase,
            "rows": 0,
            "mape_pct": "",
            "median_abs_error_pct": "",
            "p90_abs_error_pct": "",
            "p95_abs_error_pct": "",
            "max_abs_error_pct": "",
            "within_5": "",
            "within_10": "",
            "wmape_pct": "",
            "weighted_bias_pct": "",
            "worst_shape": "",
            "worst_error_pct": "",
            "comparison_csv": comparison_csv,
        }

    error_values = [_float(row, "error_pct") for row in rows]
    abs_errors = [abs(value) for value in error_values]
    worst_index = max(range(len(rows)), key=lambda index: abs_errors[index])
    worst_row = rows[worst_index]
    wmape_pct, weighted_bias_pct = _weighted_error_metrics(rows)
    return {
        "model": model,
        "tp": tp,
        "moe_tp": moe_tp,
        "ep": ep,
        "phase": phase,
        "rows": len(rows),
        "mape_pct": statistics.fmean(abs_errors),
        "median_abs_error_pct": statistics.median(abs_errors),
        "p90_abs_error_pct": _percentile(abs_errors, 90.0),
        "p95_abs_error_pct": _percentile(abs_errors, 95.0),
        "max_abs_error_pct": abs_errors[worst_index],
        "within_5": f"{sum(value <= 5.0 for value in abs_errors)}/{len(abs_errors)}",
        "within_10": f"{sum(value <= 10.0 for value in abs_errors)}/{len(abs_errors)}",
        "wmape_pct": wmape_pct,
        "weighted_bias_pct": weighted_bias_pct,
        "worst_shape": worst_row.get("shape", ""),
        "worst_error_pct": error_values[worst_index],
        "comparison_csv": comparison_csv,
    }


def summarize_entry(entry: ManifestEntry, *, allow_empty: bool = False) -> dict[str, Any]:
    """Summarize one manifest entry into rows/MAPE/max/worst fields."""

    if not entry.comparison_csv.exists():
        raise FileNotFoundError(entry.comparison_csv)

    with entry.comparison_csv.open(newline="") as f:
        rows = [row for row in csv.DictReader(f) if str(row.get("phase", "")) == entry.source_phase]

    try:
        return _summarize_rows(
            model=entry.model,
            tp=entry.tp,
            moe_tp=entry.moe_tp,
            ep=entry.ep,
            phase=entry.phase,
            rows=rows,
            comparison_csv=entry.comparison_csv_label,
            allow_empty=allow_empty,
        )
    except ValueError as exc:
        if not rows:
            raise ValueError(f"no rows for phase={entry.source_phase!r} in {entry.comparison_csv}") from exc
        raise


def summarize_manifest(
    path: Path,
    *,
    allow_empty: bool = False,
    include_overall: bool = False,
    overall_model: str = "ALL",
) -> list[dict[str, Any]]:
    """Summarize all entries in a manifest."""

    entries = load_manifest(path)
    summary_rows = [summarize_entry(entry, allow_empty=allow_empty) for entry in entries]
    if not include_overall:
        return summary_rows

    overall_rows: list[dict[str, Any]] = []
    for entry in entries:
        if not entry.comparison_csv.exists():
            raise FileNotFoundError(entry.comparison_csv)
        with entry.comparison_csv.open(newline="") as f:
            overall_rows.extend(
                row for row in csv.DictReader(f) if str(row.get("phase", "")) == entry.source_phase
            )
    summary_rows.append(
        _summarize_rows(
            model=overall_model,
            tp="all",
            moe_tp="all",
            ep="all",
            phase="all",
            rows=overall_rows,
            comparison_csv="",
            allow_empty=allow_empty,
        )
    )
    return summary_rows


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
    parser.add_argument("--include-overall", action="store_true", help="Append one aggregate row across the manifest.")
    parser.add_argument("--overall-model", default="ALL", help="Model label for the aggregate row.")
    args = parser.parse_args(argv)

    write_summary(
        summarize_manifest(
            args.manifest,
            allow_empty=args.allow_empty,
            include_overall=args.include_overall,
            overall_model=args.overall_model,
        ),
        args.output,
    )


if __name__ == "__main__":
    main()

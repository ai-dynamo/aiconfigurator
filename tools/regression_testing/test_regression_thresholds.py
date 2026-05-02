#!/usr/bin/env python3
import csv
import os
from pathlib import Path

SUMMARY_PATH = Path(
    os.environ.get(
        "AIC_COMPARISON_SUMMARY",
        Path(__file__).resolve().parent / "results" / "comparison_summary.csv",
    )
)
METRICS = ("ttft_mape_improvement_%", "tpot_mape_improvement_%")


def _regression_pct(value: str) -> float:
    if value == "":
        return 0.0
    improvement = float(value)
    return max(0.0, -improvement)


def test_mape_regression_within_thresholds() -> None:
    with SUMMARY_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))

    failures = []
    for row in rows:
        threshold = 5.0 if row["partition"] == "all" else 10.0
        for metric in METRICS:
            regression = _regression_pct(row[metric])
            if regression >= threshold:
                failures.append(f"{row['partition']} {metric}: regression={regression:.6f}% threshold<{threshold:.1f}%")

    assert not failures, "\n".join(failures)

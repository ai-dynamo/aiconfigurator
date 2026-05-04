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
SILICON_RESULT_PATH = Path(
    os.environ.get(
        "AIC_SILICON_RESULT",
        Path(__file__).resolve().parent / "results" / "silicon_result.csv",
    )
)
METRICS = ("ttft_mape_improvement_%", "tpot_mape_improvement_%")


def _regression_pct(value: str) -> float:
    if value == "":
        return 0.0
    improvement = float(value)
    return max(0.0, -improvement)


def _has_prediction(row: dict[str, str], prefix: str) -> bool:
    return bool(row.get(f"{prefix}_predicted_ttft_ms") and row.get(f"{prefix}_predicted_tpot_ms"))


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


def test_no_prediction_coverage_regression() -> None:
    with SILICON_RESULT_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))

    failures = []
    for row in rows:
        if _has_prediction(row, "old") and not _has_prediction(row, "new"):
            failures.append(
                f"{row.get('id', '<unknown>')}: old prediction succeeded but new prediction failed "
                f"({row.get('model_path', '<unknown>')}, {row.get('system', '<unknown>')}, "
                f"{row.get('backend', '<unknown>')}, {row.get('mode', '<unknown>')})"
            )

    assert not failures, "\n".join(failures)

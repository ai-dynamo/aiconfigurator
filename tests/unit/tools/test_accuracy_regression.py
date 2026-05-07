# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools" / "accuracy_regression_testing"

SILICON_FIELDS = [
    "mode",
    "model_path",
    "backend",
    "system",
    "silicon_ttft_ms",
    "silicon_tpot_ms",
    "old_predicted_ttft_ms",
    "new_predicted_ttft_ms",
    "old_predicted_tpot_ms",
    "new_predicted_tpot_ms",
]
SUMMARY_FIELDS = ["partition", "num_samples_added", "ttft_mape_improvement_%", "tpot_mape_improvement_%"]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def compare_module():
    return _load_module("compare_silicon_predictions", TOOLS_DIR / "compare_silicon_predictions.py")


def _silicon_row(silicon_ttft, silicon_tpot, old_ttft, new_ttft, old_tpot, new_tpot):
    def _str(value):
        return "" if value is None else str(value)

    return {
        "mode": "agg",
        "model_path": "meta/Llama-3.1-8B",
        "backend": "trtllm",
        "system": "h200_sxm",
        "silicon_ttft_ms": str(silicon_ttft),
        "silicon_tpot_ms": str(silicon_tpot),
        "old_predicted_ttft_ms": _str(old_ttft),
        "new_predicted_ttft_ms": _str(new_ttft),
        "old_predicted_tpot_ms": _str(old_tpot),
        "new_predicted_tpot_ms": _str(new_tpot),
    }


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _generate_summary(compare_module, tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    silicon_csv = tmp_path / "silicon.csv"
    summary_csv = tmp_path / "summary.csv"
    _write_csv(silicon_csv, rows, SILICON_FIELDS)
    valid_rows = compare_module.read_valid_rows(silicon_csv)
    compare_module.write_summary(valid_rows, summary_csv)
    return summary_csv


def _load_threshold_module(monkeypatch, summary_path: Path, silicon_path: Path):
    monkeypatch.setenv("AIC_COMPARISON_SUMMARY", str(summary_path))
    monkeypatch.setenv("AIC_SILICON_RESULT", str(silicon_path))
    return _load_module("regression_thresholds", TOOLS_DIR / "test_regression_thresholds.py")


def test_summary_reports_regression_when_new_is_worse(compare_module, tmp_path):
    rows = [
        _silicon_row(100, 10, old_ttft=105, new_ttft=130, old_tpot=10.5, new_tpot=13),
        _silicon_row(200, 20, old_ttft=210, new_ttft=260, old_tpot=21, new_tpot=26),
    ]
    summary_csv = _generate_summary(compare_module, tmp_path, rows)
    with summary_csv.open(newline="") as f:
        all_row = next(row for row in csv.DictReader(f) if row["partition"] == "all")
    assert float(all_row["ttft_mape_improvement_%"]) == pytest.approx(-25.0, abs=0.01)
    assert float(all_row["tpot_mape_improvement_%"]) == pytest.approx(-25.0, abs=0.01)


def test_summary_reports_improvement_when_new_is_better(compare_module, tmp_path):
    rows = [
        _silicon_row(100, 10, old_ttft=130, new_ttft=105, old_tpot=13, new_tpot=10.5),
        _silicon_row(200, 20, old_ttft=260, new_ttft=210, old_tpot=26, new_tpot=21),
    ]
    summary_csv = _generate_summary(compare_module, tmp_path, rows)
    with summary_csv.open(newline="") as f:
        all_row = next(row for row in csv.DictReader(f) if row["partition"] == "all")
    assert float(all_row["ttft_mape_improvement_%"]) == pytest.approx(25.0, abs=0.01)
    assert float(all_row["tpot_mape_improvement_%"]) == pytest.approx(25.0, abs=0.01)


def test_threshold_check_catches_regression(monkeypatch, compare_module, tmp_path):
    rows = [
        _silicon_row(100, 10, old_ttft=105, new_ttft=130, old_tpot=10.5, new_tpot=13),
        _silicon_row(200, 20, old_ttft=210, new_ttft=260, old_tpot=21, new_tpot=26),
    ]
    silicon_csv = tmp_path / "silicon.csv"
    _write_csv(silicon_csv, rows, SILICON_FIELDS)
    summary_csv = _generate_summary(compare_module, tmp_path, rows)
    threshold = _load_threshold_module(monkeypatch, summary_csv, silicon_csv)
    with pytest.raises(AssertionError, match="mape_improvement"):
        threshold.test_mape_regression_within_thresholds()


def test_threshold_check_passes_when_new_is_better(monkeypatch, compare_module, tmp_path):
    rows = [
        _silicon_row(100, 10, old_ttft=130, new_ttft=105, old_tpot=13, new_tpot=10.5),
        _silicon_row(200, 20, old_ttft=260, new_ttft=210, old_tpot=26, new_tpot=21),
    ]
    silicon_csv = tmp_path / "silicon.csv"
    _write_csv(silicon_csv, rows, SILICON_FIELDS)
    summary_csv = _generate_summary(compare_module, tmp_path, rows)
    threshold = _load_threshold_module(monkeypatch, summary_csv, silicon_csv)
    threshold.test_mape_regression_within_thresholds()


def test_coverage_regression_catches_lost_prediction(monkeypatch, compare_module, tmp_path):
    rows = [_silicon_row(100, 10, old_ttft=105, new_ttft=None, old_tpot=10.5, new_tpot=None)]
    silicon_csv = tmp_path / "silicon.csv"
    _write_csv(silicon_csv, rows, SILICON_FIELDS)
    summary_csv = tmp_path / "summary.csv"
    _write_csv(summary_csv, [], SUMMARY_FIELDS)
    threshold = _load_threshold_module(monkeypatch, summary_csv, silicon_csv)
    with pytest.raises(AssertionError, match="old prediction succeeded but new prediction failed"):
        threshold.test_no_prediction_coverage_regression()

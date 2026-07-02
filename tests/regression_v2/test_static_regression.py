# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tier-1 gate: committed run_static baselines vs a freshly collected run.

Pure CSV comparison — the expensive collection happens beforehand:

    python tools/regression_v2/collect_static_baseline.py --output-dir <dir> --jobs 8
    AIC_REGV2_CURRENT_DIR=<dir> pytest -m regression_v2 tests/regression_v2/test_static_regression.py

Any difference fails, with category-tagged examples. The fix is always to make
the change reviewable: regenerate the committed baseline in the same PR
(`collect_static_baseline.py --update`) so the diff shows exactly which
combos/points moved.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.regression_v2 import compare, grid

pytestmark = pytest.mark.regression_v2

BASELINE_DIR = Path(os.environ.get("AIC_REGV2_BASELINE_DIR", grid.DEFAULT_BASELINE_DIR))
CURRENT_DIR = os.environ.get("AIC_REGV2_CURRENT_DIR")
RTOL = float(os.environ.get("AIC_REGV2_RTOL", compare.DEFAULT_RTOL))
REPORT_PATH = os.environ.get("AIC_REGV2_REPORT")

_results: list[compare.ComboResult] = []


def _combo_relpaths() -> list[str]:
    paths: set[str] = set()
    for root in [BASELINE_DIR] + ([Path(CURRENT_DIR)] if CURRENT_DIR else []):
        if root.is_dir():
            # Combos are exactly <system>/<backend>/<version>.csv; other files
            # in the baseline dir (e.g. tier2_golden.csv) are not combos.
            paths.update(str(p.relative_to(root)) for p in root.rglob("*.csv") if len(p.relative_to(root).parts) == 3)
    return sorted(paths)


def _require_current_dir() -> Path:
    if not CURRENT_DIR:
        pytest.skip("AIC_REGV2_CURRENT_DIR not set; run collect_static_baseline.py first")
    current_dir = Path(CURRENT_DIR)
    if not current_dir.is_dir():
        pytest.fail(f"AIC_REGV2_CURRENT_DIR={current_dir} does not exist")
    return current_dir


@pytest.mark.parametrize("relpath", _combo_relpaths() or ["<no-baselines>"])
def test_static_baseline_matches(relpath: str) -> None:
    if relpath == "<no-baselines>":
        pytest.skip("no baseline or current CSVs found")
    current_dir = _require_current_dir()

    baseline_path = BASELINE_DIR / relpath
    current_path = current_dir / relpath
    if not baseline_path.exists():
        pytest.fail(
            f"combo {relpath} has no committed baseline. If this combo is new, run\n"
            f"  python tools/regression_v2/collect_static_baseline.py --update\n"
            f"and commit the new file."
        )
    if not current_path.exists():
        pytest.fail(
            f"combo {relpath} exists in the baseline but was not produced by the current run.\n"
            f"If the data/version was intentionally removed, delete the baseline file in the same PR."
        )

    result = compare.compare_combo(relpath, baseline_path, current_path, rtol=RTOL)
    _results.append(result)
    if result.diffs:
        pytest.fail(
            compare.summarize([result]) + "\n\nIf these changes are intended, refresh the baseline in this PR:\n"
            "  python tools/regression_v2/collect_static_baseline.py --update\n"
            "and commit the diff — it is the review artifact.",
            pytrace=False,
        )


def teardown_module(_module) -> None:
    if REPORT_PATH and _results:
        compare.write_report(_results, Path(REPORT_PATH))

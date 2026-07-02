# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tier-2 gate: scheduling-defense configs vs committed golden TTFT/TPOT.

Runs the curated configs in tools/regression_v2/tier2_configs.yaml through
``cli_estimate`` with production semantics (SILICON, shared layer on) and
compares against tests/baselines/regression_v2/tier2_golden.csv.

A config that raises is recorded as status=<ExceptionType> and pinned in the
golden too — an infeasible config becoming feasible (or vice versa) is a
scheduling/memory-model change and must be reviewed.

Refresh goldens:
  AIC_REGV2_UPDATE_GOLDEN=1 pytest -m regression_v2 tests/regression_v2/test_scheduling_regression.py
"""

from __future__ import annotations

import csv
import io
import os
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import yaml

from tools.regression_v2 import grid

pytestmark = pytest.mark.regression_v2

CONFIGS_PATH = grid.REPO_ROOT / "tools" / "regression_v2" / "tier2_configs.yaml"
GOLDEN_PATH = Path(os.environ.get("AIC_REGV2_TIER2_GOLDEN", grid.DEFAULT_BASELINE_DIR / "tier2_golden.csv"))
UPDATE_GOLDEN = os.environ.get("AIC_REGV2_UPDATE_GOLDEN") == "1"
RTOL = float(os.environ.get("AIC_REGV2_RTOL", 1e-4))
GOLDEN_FIELDS = ["id", "status", "ttft_ms", "tpot_ms"]


def _load_configs() -> list[dict]:
    with CONFIGS_PATH.open() as f:
        return yaml.safe_load(f)["configs"]


def _run_config(config: dict) -> dict:
    from aiconfigurator.cli.api import cli_estimate

    try:
        with redirect_stdout(io.StringIO()):
            result = cli_estimate(**config["kwargs"])
        return {
            "id": config["id"],
            "status": "OK",
            "ttft_ms": f"{result.ttft:.6f}",
            "tpot_ms": f"{result.tpot:.6f}",
        }
    except Exception as e:
        return {"id": config["id"], "status": type(e).__name__, "ttft_ms": "", "tpot_ms": ""}


@pytest.fixture(scope="module")
def current_results() -> dict[str, dict]:
    results = {config["id"]: _run_config(config) for config in _load_configs()}
    if UPDATE_GOLDEN:
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        with GOLDEN_PATH.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=GOLDEN_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(results[key] for key in sorted(results))
    return results


@pytest.fixture(scope="module")
def golden() -> dict[str, dict]:
    if not GOLDEN_PATH.exists():
        if UPDATE_GOLDEN:
            return {}
        pytest.fail(f"golden file missing: {GOLDEN_PATH}; generate with AIC_REGV2_UPDATE_GOLDEN=1")
    with GOLDEN_PATH.open(newline="") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def _relative_delta(base: str, cur: str) -> float:
    base_value, cur_value = float(base), float(cur)
    return abs(cur_value - base_value) / max(abs(base_value), 1e-9)


@pytest.mark.parametrize("config", _load_configs(), ids=lambda c: c["id"])
def test_scheduling_config_matches_golden(config, current_results, golden) -> None:
    if UPDATE_GOLDEN:
        pytest.skip("golden refreshed; review and commit tier2_golden.csv")
    config_id = config["id"]
    cur = current_results[config_id]
    gold = golden.get(config_id)
    refresh_hint = (
        "\nIf intended, refresh goldens in this PR:\n"
        "  AIC_REGV2_UPDATE_GOLDEN=1 pytest -m regression_v2 tests/regression_v2/test_scheduling_regression.py"
    )
    if gold is None:
        pytest.fail(f"{config_id}: no golden entry (new config)." + refresh_hint, pytrace=False)
    if cur["status"] != gold["status"]:
        pytest.fail(f"{config_id}: status changed {gold['status']} -> {cur['status']}" + refresh_hint, pytrace=False)
    if cur["status"] == "OK":
        for metric in ("ttft_ms", "tpot_ms"):
            rel = _relative_delta(gold[metric], cur[metric])
            if rel > RTOL:
                pytest.fail(
                    f"{config_id}: {metric} drift {gold[metric]} -> {cur[metric]} (rel {rel:.2%})" + refresh_hint,
                    pytrace=False,
                )


def test_golden_has_no_stale_entries(golden) -> None:
    if UPDATE_GOLDEN:
        pytest.skip("golden refreshed")
    stale = set(golden) - {config["id"] for config in _load_configs()}
    assert not stale, f"golden entries with no config (remove or restore): {sorted(stale)}"

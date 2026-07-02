#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the tier-2 scheduling-defense configs and snapshot the results.

Runs the curated configs in tier2_configs.yaml through ``cli_estimate`` with
production semantics (SILICON, shared layer on) and writes one CSV row per
config into ``<output-dir>/tier2.csv``.

A config that raises is recorded as status=<ExceptionType> — an infeasible
config becoming feasible (or vice versa) is a scheduling/memory-model change
the comparison report must show.

Like collect_static.py, this snapshots one revision; diff two snapshots with
report.py.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))  # make `tools.` importable when run as a script

from tools.regression_v2 import grid

CONFIGS_PATH = grid.REPO_ROOT / "tools" / "regression_v2" / "tier2_configs.yaml"
TIER2_FILENAME = "tier2.csv"
TIER2_FIELDS = ["id", "status", "ttft_ms", "tpot_ms"]


def load_configs() -> list[dict]:
    import yaml

    with CONFIGS_PATH.open() as f:
        return yaml.safe_load(f)["configs"]


def run_config(config: dict) -> dict:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("regv2_snapshot"),
        help=f"Directory to write {TIER2_FILENAME} into (shared with collect_static.py).",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    results = sorted((run_config(config) for config in load_configs()), key=lambda r: r["id"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / TIER2_FILENAME
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TIER2_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)

    statuses: dict[str, int] = {}
    for row in results:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    print(f"tier2: {len(results)} configs {statuses} -> {out_path} in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

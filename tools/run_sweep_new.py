#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Standalone runner that drives the new sweep path end-to-end.

Reads a YAML experiment file (same format as cli/example.yaml or
cli/exps/*.yaml), builds a new Task from it, loads the perf
database, calls sweep_agg or sweep_disagg, and writes the resulting
Pareto DataFrame to <output_dir>/pareto.csv.

This tool exists for parity verification against the legacy CLI: run
the same YAML through both paths and diff the two CSVs.  It is NOT
intended to become a permanent CLI subcommand.  Once parity is locked
and the new path is wired into ``cli/main.py``, this script is
disposable.

Usage:
    python tools/run_sweep_new.py --yaml path/to/exp.yaml \\
        --exp exp_agg_simplified \\
        --output-dir /tmp/sweep_new
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from aiconfigurator.sdk.task_v2 import Task

logger = logging.getLogger("sweep_new")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the new sweep path on a YAML experiment.")
    p.add_argument("--yaml", required=True, type=Path, help="Path to experiment YAML.")
    p.add_argument(
        "--exp",
        default=None,
        help=(
            "Top-level key within the YAML to load (when the YAML contains multiple experiments). "
            "If omitted, the YAML root is used directly."
        ),
    )
    p.add_argument("--output-dir", required=True, type=Path, help="Directory to write pareto.csv into.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def _load_yaml(yaml_path: Path, exp_key: str | None) -> dict:
    with yaml_path.open() as f:
        data = yaml.safe_load(f) or {}
    if exp_key is None:
        return data
    if exp_key not in data:
        raise SystemExit(f"--exp {exp_key!r} not found in {yaml_path} (keys: {sorted(data.keys())})")
    return data[exp_key]


def run(yaml_path: Path, exp_key: str | None, output_dir: Path) -> Path:
    """Run the new sweep path; return the path to the written CSV."""
    yaml_data = _load_yaml(yaml_path, exp_key)
    task = Task.from_yaml(yaml_data)
    task.validate()  # fail-fast on invalid YAML before loading DBs
    logger.info(
        "Loaded Task: mode=%s model=%s is_moe=%s family=%s",
        task.serving_mode,
        task.model_path or task.prefill_model_path,
        task.is_moe,
        task.model_family,
    )

    df = task.run()

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "pareto.csv"
    # Strip object columns (e.g. _per_ops_source) before CSV write — matches legacy behavior.
    df_to_write = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")
    df_to_write.to_csv(csv_path, index=False)
    logger.info("Wrote %d rows to %s", len(df_to_write), csv_path)
    return csv_path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    try:
        run(args.yaml, args.exp, args.output_dir)
    except Exception:
        logger.exception("run_sweep_new failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

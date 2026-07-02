#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Curate silicon reference points from the silicon sample.

Selects a small, deterministic subset of ``silicon_sample.csv`` (real e2e
measurements with provenance) to serve as accuracy anchors for run_silicon.py:
up to --per-group configs per (system, backend, mode), preferring distinct
models and configs with the most measured concurrency rows. Candidates are
verified by actually predicting them — only points that produce a prediction
today are useful accuracy anchors (a point that later stops predicting shows
up in run_silicon.py output as a non-OK status).

Output keeps the silicon_sample.csv schema verbatim, so refreshing after a new
silicon dump is: update silicon_sample.csv, re-run this script, review the
diff.

Usage:
  python tools/regression_v2/make_silicon_refs.py            # rewrite refs
  python tools/regression_v2/make_silicon_refs.py --check    # verify freshness
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))  # make `tools.` importable when run as a script

from tools.regression_v2 import grid

SAMPLE_PATH = grid.REPO_ROOT / "src" / "aiconfigurator" / "systems" / "silicon_sample.csv"
REFS_PATH = grid.REPO_ROOT / "tools" / "regression_v2" / "silicon_refs.csv"

PARALLEL_FIELDS = ("tp_size", "pp_size", "attention_dp_size", "moe_tp_size", "moe_ep_size")
DISAGG_FIELDS = tuple(
    f"{stage}_{field}" for stage in ("prefill", "decode") for field in (*PARALLEL_FIELDS, "batch_size", "num_workers")
)
# cli_estimate cannot run disagg without these (older dumps lack them).
REQUIRED_DISAGG_FIELDS = ("prefill_batch_size", "prefill_num_workers", "decode_batch_size", "decode_num_workers")


def _config_key(row: dict) -> tuple:
    """Deployment identity: everything except the concurrency point."""
    parallel = tuple(row[f] for f in PARALLEL_FIELDS) if row["mode"] == "agg" else tuple(row[f] for f in DISAGG_FIELDS)
    return (
        row["model_path"],
        row["system"],
        row["backend"],
        row["backend_version"],
        row["mode"],
        row["isl"],
        row["osl"],
        row["gemm_quant_mode"],
        row["moe_quant_mode"],
        parallel,
    )


def _predictable(row: dict, offline_models: set[str]) -> bool:
    """Structurally runnable by cli_estimate without network access."""
    if row["model_path"] not in offline_models:
        return False
    return row["mode"] != "disagg" or all(row.get(field) for field in REQUIRED_DISAGG_FIELDS)


def select_refs(rows: list[dict], per_group: int) -> list[dict]:
    offline_models = set(grid.bundled_models())
    configs: dict[tuple, list[dict]] = {}
    for row in rows:
        if _predictable(row, offline_models):
            configs.setdefault(_config_key(row), []).append(row)

    from tools.regression_v2.run_silicon import predict_ref

    refs: list[dict] = []
    groups: dict[tuple, list[tuple]] = {}
    for key in configs:
        groups.setdefault((key[1], key[2], key[4]), []).append(key)

    for group in sorted(groups):
        # Prefer configs with the most measured points (stable deployments),
        # then deterministic tie-break; at most one config per model.
        candidates = sorted(groups[group], key=lambda k: (-len(configs[k]), k))
        seen_models: set[str] = set()
        picked = 0
        for key in candidates:
            if picked >= per_group:
                break
            if key[0] in seen_models:
                continue
            # Median concurrency point of the chosen config.
            points = sorted(configs[key], key=lambda r: float(r["silicon_ttft_ms"]))
            candidate = points[len(points) // 2]
            status = predict_ref(candidate)["status"]
            if status != "OK":
                print(f"  skip {candidate['id']}: {status}", file=sys.stderr)
                continue
            seen_models.add(key[0])
            picked += 1
            refs.append(candidate)
    return sorted(refs, key=lambda r: r["id"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-group", type=int, default=2, help="Max configs per (system, backend, mode).")
    parser.add_argument("--check", action="store_true", help="Fail if the committed refs are stale.")
    args = parser.parse_args()

    with SAMPLE_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        refs = select_refs(list(reader), args.per_group)

    if args.check:
        with REFS_PATH.open(newline="") as f:
            committed = list(csv.DictReader(f))
        if committed != refs:
            print("silicon_refs.csv is stale; re-run make_silicon_refs.py", file=sys.stderr)
            return 1
        print(f"silicon_refs.csv is current ({len(refs)} refs)")
        return 0

    with REFS_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(refs)
    print(f"wrote {len(refs)} refs -> {REFS_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

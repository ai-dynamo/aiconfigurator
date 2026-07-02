#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Predict the silicon reference points and snapshot accuracy vs measured.

For every row in silicon_refs.csv (curated real e2e measurements, see
make_silicon_refs.py) runs ``cli_estimate`` and writes
``<output-dir>/silicon_accuracy.csv`` with predicted vs measured TTFT/TPOT and
per-metric relative error.

Prediction uses the measured backend_version when the local database has it
and falls back to the latest local data otherwise (V1 semantics); the version
actually used is recorded in predicted_with_version.

This is the accuracy layer of the regression stack: it answers "how far from
real hardware", per point. It never gates a PR — report.py renders it as a
report-only section (and old-vs-new sides show whether a PR moved each
point's error). It is also the nightly accuracy-tracking workload.
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

from tools.regression_v2.make_silicon_refs import DISAGG_FIELDS, PARALLEL_FIELDS, REFS_PATH

SILICON_FILENAME = "silicon_accuracy.csv"
SILICON_FIELDS = [
    "id",
    "status",
    "predicted_ttft_ms",
    "predicted_tpot_ms",
    "silicon_ttft_ms",
    "silicon_tpot_ms",
    "ttft_rel_err",
    "tpot_rel_err",
    "predicted_with_version",
    "err",
]


def _estimate_kwargs(row: dict) -> dict:
    fields = ("batch_size", *PARALLEL_FIELDS) if row["mode"] == "agg" else DISAGG_FIELDS
    return dict(
        model_path=row["model_path"],
        system_name=row["system"],
        mode=row["mode"],
        backend_name=row["backend"],
        backend_version=row["backend_version"] or None,
        isl=int(row["isl"]),
        osl=int(row["osl"]),
        gemm_quant_mode=row["gemm_quant_mode"] or None,
        moe_quant_mode=row["moe_quant_mode"] or None,
        **{field: int(row[field]) for field in fields if row.get(field)},
    )


def predict_ref(row: dict) -> dict:
    from aiconfigurator.cli.api import cli_estimate

    kwargs = _estimate_kwargs(row)

    def estimate() -> tuple[float, float]:
        with redirect_stdout(io.StringIO()):
            result = cli_estimate(**kwargs)
        return result.ttft, result.tpot

    out = dict.fromkeys(SILICON_FIELDS, "")
    out.update(id=row["id"], silicon_ttft_ms=row["silicon_ttft_ms"], silicon_tpot_ms=row["silicon_tpot_ms"])
    try:
        try:
            ttft, tpot = estimate()
        except ValueError as e:
            # Measured version has no local database: predict with the latest
            # local data instead (V1 semantics) and record the skew.
            if kwargs["backend_version"] is None or "Failed to load perf database" not in str(e):
                raise
            kwargs["backend_version"] = None
            ttft, tpot = estimate()
        measured_ttft, measured_tpot = float(row["silicon_ttft_ms"]), float(row["silicon_tpot_ms"])
        out.update(
            status="OK",
            predicted_ttft_ms=f"{ttft:.6f}",
            predicted_tpot_ms=f"{tpot:.6f}",
            ttft_rel_err=f"{(ttft - measured_ttft) / max(abs(measured_ttft), 1e-9):.6f}",
            tpot_rel_err=f"{(tpot - measured_tpot) / max(abs(measured_tpot), 1e-9):.6f}",
            predicted_with_version=kwargs["backend_version"] or "latest",
        )
    except Exception as e:
        out.update(status=type(e).__name__, err=str(e)[:200])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("regv2_snapshot"),
        help=f"Directory to write {SILICON_FILENAME} into (shared with the other collectors).",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    with REFS_PATH.open(newline="") as f:
        refs = list(csv.DictReader(f))
    results = [predict_ref(row) for row in refs]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / SILICON_FILENAME
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SILICON_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)

    statuses: dict[str, int] = {}
    for row in results:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    print(
        f"silicon: {len(results)} refs {statuses} -> {out_path} in {time.perf_counter() - t0:.1f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

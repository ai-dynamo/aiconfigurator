#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect the tier-1 run_static snapshot of the current revision.

For every (system, backend, version) combo with real local data, runs the
fixed grid from ``grid.py`` — model x parallelism x quant x shape — through
``InferenceSession.run_static_latency_only`` in SILICON mode with the shared
layer OFF, and writes one CSV per combo into ``--output-dir``.

Every grid row always produces exactly one output row with a three-state
status, so snapshots are rectangular and diffs are reviewable:

  OK         latency computed; value_ms holds the rounded scalar
  DATA_MISS  PerfDataNotAvailableError (combo lacks silicon for the request)
  INVALID    anything else (validation error, unsupported layout, ...);
             err holds the exception type name

The regression signal is old-vs-new: collect a snapshot on each revision
(each side runs its own copy of this script) and diff them with report.py.

Usage:
  # PR profile (latest data-carrying version per system/backend):
  python tools/prediction_regression_gate/collect_static.py --output-dir /tmp/gate-new

  # Restrict scope while iterating:
  python tools/prediction_regression_gate/collect_static.py --systems h200_sxm --backends trtllm --jobs 1
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))  # make `tools.` importable when run as a script

from tools.prediction_regression_gate import grid

CSV_HEADER = [
    "model",
    "tp",
    "pp",
    "adp",
    "moe_tp",
    "moe_ep",
    "quant",
    "phase",
    "bs",
    "isl",
    "status",
    "value_ms",
    "err",
]

# Pinned so the in-flight Rust engine-step default flip is a deliberate,
# reviewed baseline regeneration instead of ambient drift.
ENGINE_STEP_BACKEND = "python"


def _row_key(row: dict) -> tuple:
    return (
        row["model"],
        int(row["tp"]),
        int(row["pp"]),
        int(row["adp"]),
        row["moe_tp"] or "",
        row["moe_ep"] or "",
        row["quant"],
        row["phase"],
        int(row["bs"]),
        int(row["isl"]),
    )


def collect_combo(combo: grid.Combo) -> tuple[grid.Combo, list[dict], float]:
    """Run the full grid for one combo. Runs inside a worker process."""
    from aiconfigurator.cli.api import _build_model_config
    from aiconfigurator.sdk import config as sdk_config
    from aiconfigurator.sdk.backends.factory import get_backend
    from aiconfigurator.sdk.errors import PerfDataNotAvailableError
    from aiconfigurator.sdk.inference_session import InferenceSession
    from aiconfigurator.sdk.models import _get_model_info, get_model
    from aiconfigurator.sdk.perf_database import get_database_view

    t0 = time.perf_counter()
    database = get_database_view(
        combo.system,
        combo.backend,
        combo.version,
        database_mode="SILICON",
        shared_layer=False,
    )
    if database is None:
        raise RuntimeError(f"failed to load database for {combo}")
    backend = get_backend(combo.backend)

    rows: list[dict] = []
    for model_name in grid.bundled_models():
        try:
            is_moe = (_get_model_info(model_name).get("num_experts") or 0) > 0
        except Exception:
            is_moe = False
        parallel = grid.MOE_PARALLEL if is_moe else grid.DENSE_PARALLEL

        for tp, pp, adp, moe_tp, moe_ep in parallel:
            for quant_label, gemm_quant, moe_quant in grid.quant_variants_for(combo.system):
                base = {
                    "model": model_name,
                    "tp": tp,
                    "pp": pp,
                    "adp": adp,
                    "moe_tp": moe_tp if moe_tp is not None else "",
                    "moe_ep": moe_ep if moe_ep is not None else "",
                    "quant": quant_label,
                }
                points = [("ctx", bs, isl) for bs, isl in grid.PREFILL_POINTS] + [
                    ("gen", bs, isl) for bs, isl in grid.DECODE_POINTS
                ]

                session = None
                build_err: Exception | None = None
                try:
                    model_config = _build_model_config(
                        tp, pp, adp, moe_tp, moe_ep, gemm_quant, None, None, moe_quant if is_moe else None, None
                    )
                    model = get_model(model_name, model_config, combo.backend)
                    session = InferenceSession(model, database, backend)
                except Exception as e:  # config invalid for this model/backend: whole block INVALID
                    build_err = e

                for phase, bs, isl in points:
                    row = dict(base, phase=phase, bs=bs, isl=isl, status="", value_ms="", err="")
                    if session is None:
                        row["status"] = "INVALID"
                        row["err"] = type(build_err).__name__
                        rows.append(row)
                        continue
                    runtime_config = sdk_config.RuntimeConfig(
                        batch_size=bs,
                        isl=isl,
                        osl=grid.CTX_OSL if phase == "ctx" else grid.GEN_OSL,
                        engine_step_backend=ENGINE_STEP_BACKEND,
                    )
                    mode = "static_ctx" if phase == "ctx" else "static_gen"
                    try:
                        with redirect_stdout(io.StringIO()):
                            latency = session.run_static_latency_only(
                                runtime_config=runtime_config, mode=mode, stride=grid.STRIDE
                            )
                        row["status"] = "OK"
                        row["value_ms"] = f"{latency:.6f}"
                    except PerfDataNotAvailableError:
                        row["status"] = "DATA_MISS"
                    except Exception as e:
                        row["status"] = "INVALID"
                        row["err"] = type(e).__name__
                    rows.append(row)

    rows.sort(key=_row_key)
    return combo, rows, time.perf_counter() - t0


def write_combo_csv(output_dir: Path, combo: grid.Combo, rows: list[dict]) -> Path:
    path = output_dir / combo.relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--systems", nargs="*", default=None, help="Restrict to these systems.")
    parser.add_argument("--backends", nargs="*", default=None, help="Restrict to these backends.")
    parser.add_argument("--versions", choices=("latest", "all"), default="latest")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gate_snapshot"),
        help="Where to write combo CSVs.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir

    combos = grid.enumerate_combos(args.systems, args.backends, args.versions)
    if not combos:
        print("no combos matched (is the perf data present? try `git lfs pull`)", file=sys.stderr)
        return 2

    print(f"collecting {len(combos)} combos -> {output_dir} (jobs={args.jobs})", file=sys.stderr)
    t0 = time.perf_counter()
    status_totals: dict[str, int] = {}
    failures: list[str] = []

    def _consume(combo: grid.Combo, rows: list[dict], dt: float) -> None:
        write_combo_csv(output_dir, combo, rows)
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            status_totals[row["status"]] = status_totals.get(row["status"], 0) + 1
        print(
            f"  {combo.system}/{combo.backend}/{combo.version}: {len(rows)} rows {counts} in {dt:.1f}s", file=sys.stderr
        )

    if args.jobs <= 1:
        for combo in combos:
            try:
                _consume(*collect_combo(combo))
            except Exception as e:
                failures.append(f"{combo}: {type(e).__name__}: {e}")
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(collect_combo, combo): combo for combo in combos}
            for future in as_completed(futures):
                try:
                    _consume(*future.result())
                except Exception as e:
                    failures.append(f"{futures[future]}: {type(e).__name__}: {e}")

    print(
        f"done: {len(combos) - len(failures)}/{len(combos)} combos, {status_totals} in {time.perf_counter() - t0:.1f}s",
        file=sys.stderr,
    )
    for failure in failures:
        print(f"COMBO FAILED: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

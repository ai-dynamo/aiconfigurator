#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import csv
import sys
from contextlib import redirect_stdout
from pathlib import Path

from aiconfigurator.cli.api import cli_estimate

DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "aic-core/src/aiconfigurator_core/systems/silicon_sample.csv"
PARALLEL_FIELDS = ("tp_size", "pp_size", "attention_dp_size", "moe_tp_size", "moe_ep_size")
AGG_FIELDS = ("batch_size", *PARALLEL_FIELDS)
DISAGG_FIELDS = tuple(
    f"{stage}_{field}" for stage in ("prefill", "decode") for field in (*PARALLEL_FIELDS, "batch_size", "num_workers")
)
REQUIRED_DISAGG_FIELDS = ("prefill_batch_size", "prefill_num_workers", "decode_batch_size", "decode_num_workers")
EstimateArgs = dict[str, str | int | None]
EstimateCacheKey = tuple[tuple[str, str | int | None], ...]
_ESTIMATE_CACHE: dict[EstimateCacheKey, tuple[float, float] | Exception] = {}
_MISSING_DATABASE_VERSIONS: set[tuple[str, str, str]] = set()
_CACHE_HITS = 0
_CACHE_MISSES = 0


def _estimate_args(row: dict[str, str]) -> EstimateArgs:
    fields = AGG_FIELDS if row["mode"] == "agg" else DISAGG_FIELDS
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


def _estimate_cache_key(args: EstimateArgs) -> EstimateCacheKey:
    return tuple(sorted(args.items()))


def _run_estimate(args: EstimateArgs, row_id: str) -> tuple[float, float]:
    if args["mode"] == "disagg":
        for field in REQUIRED_DISAGG_FIELDS:
            if args.get(field) is None:
                raise ValueError(f"{field} is required for disagg mode.")

    database_version = args["backend_version"]
    if database_version is not None:
        missing_database_key = (
            str(args["system_name"]),
            str(args["backend_name"]),
            str(database_version),
        )
        if missing_database_key in _MISSING_DATABASE_VERSIONS:
            args["backend_version"] = None

    def run_estimate() -> tuple[float, float]:
        with redirect_stdout(sys.stderr):
            result = cli_estimate(**args)
        return result.ttft, result.tpot

    try:
        return run_estimate()
    except ValueError as e:
        if args["backend_version"] is None or "Failed to load perf database" not in str(e):
            raise
        _MISSING_DATABASE_VERSIONS.add(
            (
                str(args["system_name"]),
                str(args["backend_name"]),
                str(args["backend_version"]),
            )
        )
        print(
            f"{row_id}: missing database version {args['backend_version']}; retrying with AIC default database version",
            file=sys.stderr,
        )
        args["backend_version"] = None
        return run_estimate()


def _estimate(row: dict[str, str]) -> tuple[float, float]:
    global _CACHE_HITS, _CACHE_MISSES

    args = _estimate_args(row)
    cache_key = _estimate_cache_key(args)
    cached = _ESTIMATE_CACHE.get(cache_key)
    if cached is not None:
        _CACHE_HITS += 1
        if isinstance(cached, Exception):
            raise cached
        return cached

    _CACHE_MISSES += 1
    try:
        result = _run_estimate(args, row.get("id", "<unknown>"))
    except Exception as e:
        _ESTIMATE_CACHE[cache_key] = e
        raise
    _ESTIMATE_CACHE[cache_key] = result
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--aic-output-prefix", default="aic")
    args = parser.parse_args()
    ttft_col = f"{args.aic_output_prefix}_predicted_ttft_ms"
    tpot_col = f"{args.aic_output_prefix}_predicted_tpot_ms"
    with args.input.open(newline="") as f:
        reader = csv.DictReader(f)
        writer = csv.DictWriter(sys.stdout, fieldnames=[*(reader.fieldnames or []), ttft_col, tpot_col])
        writer.writeheader()
        for row in reader:
            try:
                row[ttft_col], row[tpot_col] = (f"{v:.6f}" for v in _estimate(row))
            except Exception as e:
                print(f"{row.get('id', '<unknown>')}: {e}", file=sys.stderr)
                row[ttft_col] = row[tpot_col] = ""
            writer.writerow(row)
    print(f"prediction cache: {_CACHE_HITS} hits, {_CACHE_MISSES} misses", file=sys.stderr)


if __name__ == "__main__":
    main()

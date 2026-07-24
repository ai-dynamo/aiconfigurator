#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish a validated DSv4 MegaMoE staging CSV with full V3 provenance."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import tempfile
from datetime import date
from pathlib import Path

import yaml

from collector import provenance
from collector.helper import convert_perf_csv_to_parquet

COLLECTOR_MODULE = "collector.sglang.collect_dsv4_megamoe"
PERF_FILE = "dsv4_megamoe_module_perf.txt"
TABLE_STEM = "dsv4_megamoe_module_perf"
SAMPLED_DISTRIBUTION = "power_law_sampled_1.9"
DEFAULT_SAMPLED_SEED_COUNT = 10


def _csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _require_validation_pass(summary_path: Path) -> None:
    if not summary_path.is_file():
        raise FileNotFoundError(f"validation summary not found: {summary_path}")
    lines = {line.strip() for line in summary_path.read_text(encoding="utf-8").splitlines()}
    if "VALIDATION=PASS" not in lines:
        raise RuntimeError(f"{summary_path}: refusing publication without VALIDATION=PASS")


def _read_staging_rows(staging_csv: Path) -> list[dict[str, str]]:
    if staging_csv.name != PERF_FILE:
        raise ValueError(f"expected canonical staging filename {PERF_FILE}, got {staging_csv.name}")
    if not staging_csv.is_file():
        raise FileNotFoundError(staging_csv)
    with staging_csv.open(newline="", encoding="utf-8") as perf_file:
        rows = list(csv.DictReader(perf_file))
    if not rows:
        raise RuntimeError(f"{staging_csv}: refusing to publish an empty perf table")
    return rows


def _runtime_metadata(*, version: str, image_ref: str) -> dict[str, str]:
    image, separator, digest = image_ref.partition("@")
    if not image:
        raise ValueError("container image must be non-empty")
    runtime = {"framework": "sglang", "version": version, "image": image}
    if separator:
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise ValueError(f"container image digest must be sha256:<64 lowercase hex>, got {digest!r}")
        runtime["image_digest"] = digest
    return runtime


def _logical_cases(args: argparse.Namespace) -> list[dict[str, int | str]]:
    phases = _csv_strings(args.phase_order)
    unknown_phases = sorted(set(phases) - {"context", "generation"})
    if unknown_phases:
        raise ValueError(f"unsupported phases in plan: {unknown_phases}")

    cases: list[dict[str, int | str]] = []
    for phase in phases:
        if phase == "context":
            ep_sizes = _csv_ints(args.prefill_ep_sizes)
            tokens = _csv_ints(args.prefill_tokens)
            fixed_cap = args.prefill_num_max_tokens_per_rank
        else:
            ep_sizes = _csv_ints(args.decode_ep_sizes)
            tokens = _csv_ints(args.decode_tokens)
            fixed_cap = args.decode_num_max_tokens_per_rank

        for ep_size in ep_sizes:
            for num_tokens in tokens:
                for distribution in _csv_strings(args.distributions):
                    cases.append(
                        {
                            "distribution": distribution,
                            "ep_size": ep_size,
                            "fixed_cap": fixed_cap,
                            "num_tokens": num_tokens,
                            "phase": phase,
                        }
                    )
    if not cases:
        raise RuntimeError("refusing publication with an empty expanded case plan")
    return cases


def _planned_case_ids(args: argparse.Namespace) -> list[str]:
    explicit_seeds = _csv_ints(args.routing_seeds)
    plan: list[str] = []
    for logical_case in _logical_cases(args):
        distribution = str(logical_case["distribution"])
        if explicit_seeds:
            seeds = explicit_seeds
        elif distribution == SAMPLED_DISTRIBUTION:
            seeds = list(range(args.routing_seed, args.routing_seed + DEFAULT_SAMPLED_SEED_COUNT))
        else:
            seeds = [args.routing_seed]
        for routing_seed in seeds:
            num_tokens = int(logical_case["num_tokens"])
            case = {
                "cap_policy": args.cap_policy,
                "distribution": distribution,
                "ep_size": logical_case["ep_size"],
                "include_routed_scale": args.include_routed_scale,
                "model_config": args.model_config,
                "num_iterations": args.num_iterations,
                "num_max_tokens_per_rank": (
                    num_tokens if args.cap_policy == "case_tokens" else logical_case["fixed_cap"]
                ),
                "num_tokens": num_tokens,
                "num_warmup": args.num_warmup,
                "phase": logical_case["phase"],
                "pre_dispatch": args.pre_dispatch,
                "renormalize_topk_weights": args.renormalize_topk_weights,
                "routing_seed": routing_seed,
                "source_policy": args.source_policy,
            }
            plan.append(f"{COLLECTOR_MODULE}:run_case:" + json.dumps(case, sort_keys=True, separators=(",", ":")))
    return plan


def _require_exact_logical_cases(rows: list[dict[str, str]], args: argparse.Namespace) -> None:
    expected = {
        (
            str(case["phase"]),
            str(case["ep_size"]),
            str(case["num_tokens"]),
            str(case["distribution"]),
        )
        for case in _logical_cases(args)
    }
    actual_keys = [
        (
            str(row.get("phase", "")).strip(),
            str(row.get("moe_ep_size", "")).strip(),
            str(row.get("num_tokens", "")).strip(),
            str(row.get("distribution", "")).strip(),
        )
        for row in rows
    ]
    actual = set(actual_keys)
    if len(actual_keys) != len(actual) or actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RuntimeError(
            "staging rows do not exactly match the requested logical case plan: "
            f"duplicates={len(actual_keys) - len(actual)} missing={missing} unexpected={unexpected}"
        )


def _existing_tables_for_merge(
    target_dir: Path,
    *,
    runtime_meta: dict[str, str],
) -> dict[str, dict]:
    if (target_dir / PERF_FILE).exists():
        raise RuntimeError(
            f"{target_dir / PERF_FILE}: packaged perf-data trees must not contain collector staging CSV files"
        )

    meta_path = target_dir / "collection_meta.yaml"
    parquet_stems = {path.stem for path in target_dir.glob("*.parquet")} if target_dir.is_dir() else set()
    if not meta_path.exists():
        other_parquets = parquet_stems - {TABLE_STEM}
        if other_parquets:
            raise RuntimeError(
                f"{target_dir}: cannot create provenance for {TABLE_STEM}; existing parquet table(s) "
                f"{sorted(other_parquets)} have no sidecar entries to preserve"
            )
        return {}

    try:
        existing = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as error:
        raise RuntimeError(f"{meta_path}: cannot parse existing provenance sidecar: {error}") from error
    if not isinstance(existing, dict):
        raise TypeError(f"{meta_path}: expected a YAML mapping")
    if existing.get("provenance") == "legacy":
        raise RuntimeError(
            f"{target_dir}: existing collection_meta.yaml is provenance: legacy; a fresh table cannot "
            "be merged into a legacy-tier directory. Publish to a fresh runtime directory or recollect "
            "and migrate every table in this directory together."
        )

    tables = existing.get("tables")
    if not isinstance(tables, dict):
        raise TypeError(f"{meta_path}: expected a tables mapping")
    uncovered = sorted(parquet_stems - set(tables))
    if uncovered:
        raise RuntimeError(f"{meta_path}: existing sidecar does not cover parquet table(s) {uncovered}")

    other_tables = set(tables) - {TABLE_STEM}
    if other_tables:
        existing_runtime = existing.get("runtime")
        if existing_runtime != runtime_meta:
            raise RuntimeError(
                f"{meta_path}: runtime metadata {existing_runtime!r} does not match this run "
                f"{runtime_meta!r}; replacing the top-level runtime would mis-attest existing tables "
                f"{sorted(other_tables)}"
            )
    return dict(tables)


def publish_validated(args: argparse.Namespace) -> tuple[Path, Path]:
    staging_csv = args.staging_csv.resolve()
    target_dir = args.target_dir.resolve()
    _require_validation_pass(args.validation_summary.resolve())
    rows = _read_staging_rows(staging_csv)
    versions = {str(row.get("version", "")).strip() for row in rows}
    if versions != {args.target_sglang_version}:
        raise RuntimeError(
            f"{staging_csv}: collected version must exactly match destination "
            f"{args.target_sglang_version}, got {sorted(versions)}"
        )
    _require_exact_logical_cases(rows, args)
    if not re.fullmatch(r"[0-9a-f]{40,64}", args.collector_ref):
        raise ValueError("collector_ref must be the exact 40-64 character lowercase Git commit hash")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.collector_hash):
        raise ValueError("collector_hash must be sha256:<64 lowercase hex>")

    runtime_meta = _runtime_metadata(version=args.target_sglang_version, image_ref=args.image)
    existing_tables = _existing_tables_for_merge(target_dir, runtime_meta=runtime_meta)
    case_ids = _planned_case_ids(args)

    staged_parquet = convert_perf_csv_to_parquet(staging_csv, delete_source=False)
    import pyarrow.parquet as pq

    parquet_rows = pq.read_metadata(staged_parquet).num_rows
    if parquet_rows != len(rows):
        raise RuntimeError(
            f"{staged_parquet}: parquet row count {parquet_rows} does not match staging CSV row count {len(rows)}"
        )

    table_entry = {
        "collector_ref": args.collector_ref,
        "collector_hash": args.collector_hash,
        "case_plan_hash": provenance.case_plan_hash(case_ids),
        "collected_at": date.today().isoformat(),
        "rows": parquet_rows,
        "status": provenance.derive_table_status(
            unresolved_failed_count=0,
            had_module_failure=False,
        ),
    }
    merged_tables = {**existing_tables, TABLE_STEM: table_entry}

    target_dir.mkdir(parents=True, exist_ok=True)
    target_parquet = target_dir / staged_parquet.name
    target_meta = target_dir / "collection_meta.yaml"
    with tempfile.TemporaryDirectory(prefix=".dsv4-megamoe-finalize-", dir=target_dir) as temp_name:
        temp_dir = Path(temp_name)
        temp_parquet = temp_dir / staged_parquet.name
        shutil.copy2(staged_parquet, temp_parquet)
        temp_meta = provenance.write_collection_meta(temp_dir, runtime_meta, merged_tables)

        backup_parquet = temp_dir / f"{staged_parquet.name}.previous"
        had_existing_parquet = target_parquet.exists()
        if had_existing_parquet:
            shutil.copy2(target_parquet, backup_parquet)

        os.replace(temp_parquet, target_parquet)
        try:
            os.replace(temp_meta, target_meta)
        except Exception:
            if had_existing_parquet:
                os.replace(backup_parquet, target_parquet)
            else:
                target_parquet.unlink(missing_ok=True)
            raise

    return target_parquet, target_meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-csv", required=True, type=Path)
    parser.add_argument("--validation-summary", required=True, type=Path)
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument("--target-sglang-version", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--collector-ref", required=True)
    parser.add_argument("--collector-hash", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--prefill-ep-sizes", required=True)
    parser.add_argument("--decode-ep-sizes", required=True)
    parser.add_argument("--prefill-tokens", required=True)
    parser.add_argument("--decode-tokens", required=True)
    parser.add_argument("--distributions", required=True)
    parser.add_argument("--phase-order", required=True)
    parser.add_argument("--routing-seed", required=True, type=int)
    parser.add_argument("--routing-seeds", default="")
    parser.add_argument("--source-policy", required=True)
    parser.add_argument("--pre-dispatch", required=True)
    parser.add_argument("--cap-policy", choices=["fixed", "case_tokens"], required=True)
    parser.add_argument("--prefill-num-max-tokens-per-rank", required=True, type=int)
    parser.add_argument("--decode-num-max-tokens-per-rank", required=True, type=int)
    parser.add_argument("--include-routed-scale", required=True, type=int, choices=[0, 1])
    parser.add_argument("--renormalize-topk-weights", required=True, type=int, choices=[0, 1])
    parser.add_argument("--num-warmup", required=True, type=int)
    parser.add_argument("--num-iterations", required=True, type=int)
    return parser


def main() -> None:
    target_parquet, target_meta = publish_validated(build_parser().parse_args())
    print(f"FINALIZED_TO={target_parquet}")
    print(f"PROVENANCE_TO={target_meta}")


if __name__ == "__main__":
    main()

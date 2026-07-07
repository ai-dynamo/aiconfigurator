#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

LOADER_KEY_FIELDS = [
    "phase",
    "kernel_source",
    "kernel_dtype",
    "moe_dtype",
    "pre_dispatch",
    "source_policy",
    "distribution",
    "topk",
    "num_experts",
    "num_fused_shared_experts",
    "hidden_size",
    "inter_size",
    "moe_tp_size",
    "moe_ep_size",
    "num_tokens",
]

ROW_INVARIANTS = [
    ("framework", "SGLang", "every row framework must be SGLang"),
    ("op_name", "dsv4_megamoe_module", "every row op_name must be dsv4_megamoe_module"),
    ("kernel_source", "deepgemm_megamoe", "every row kernel_source must be deepgemm_megamoe"),
    ("used_cuda_graph", "true", "every row must use CUDA Graph"),
    ("includes_gate_topk", "false", "rows must not include gate/topk latency"),
    ("includes_routed_scale", "true", "rows must include routed scale"),
]

POSITIVE_FLOAT_FIELDS = [
    ("latency", "every latency must be positive"),
]


def _csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _read_rows(perf_path: Path) -> list[dict[str, str]]:
    if not perf_path.exists():
        raise FileNotFoundError(f"perf file not found: {perf_path}")
    with perf_path.open(newline="") as f:
        return list(csv.DictReader(f))


def _expected_rows(
    *,
    prefill_eps: list[int],
    decode_eps: list[int],
    prefill_tokens: list[int],
    decode_tokens: list[int],
    distributions: list[str],
    phases: set[str],
) -> int:
    cases_per_token = len(distributions)
    expected = 0
    if "context" in phases:
        expected += len(prefill_eps) * len(prefill_tokens) * cases_per_token
    if "generation" in phases:
        expected += len(decode_eps) * len(decode_tokens) * cases_per_token
    return expected


def _safe_int(value: str | None) -> int | None:
    try:
        return int(value or "")
    except ValueError:
        return None


def _positive_float(value: str | None) -> bool:
    try:
        return float(value or "0") > 0
    except ValueError:
        return False


def merge_perf_files(input_root: Path, perf_file: str, output_path: Path) -> tuple[int, int]:
    paths = sorted(path for path in input_root.rglob(perf_file) if "merged" not in path.parts)
    fieldnames: list[str] = []
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue
            for field in reader.fieldnames:
                if field not in fieldnames:
                    fieldnames.append(field)
            rows.extend(row for row in reader if row and row.get("framework") != "framework")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return len(paths), len(rows)


def validate_perf_file(
    *,
    perf_path: Path,
    prefill_eps: list[int],
    decode_eps: list[int],
    prefill_tokens: list[int],
    decode_tokens: list[int],
    distributions: list[str],
    phases: set[str],
    summary_path: Path,
    target_version: str | None = None,
    allow_version_mismatch: bool = True,
    expect_single_perf_file: bool = False,
) -> tuple[str, list[str]]:
    rows = _read_rows(perf_path)
    expected = _expected_rows(
        prefill_eps=prefill_eps,
        decode_eps=decode_eps,
        prefill_tokens=prefill_tokens,
        decode_tokens=decode_tokens,
        distributions=distributions,
        phases=phases,
    )

    summary: list[str] = []
    errors: list[str] = []
    if expect_single_perf_file:
        perf_files = sorted(perf_path.parent.glob("*_perf.txt"))
        summary.append(f"perf_files={len(perf_files)}")
        if len(perf_files) != 1:
            errors.append(f"expected exactly one perf file, got {len(perf_files)}")
    summary.append(f"perf_file={perf_path.name}")
    summary.append("phases=" + ",".join(sorted(phases)))
    summary.append(f"total_rows={len(rows)} expected={expected}")
    summary.append("seed_samples=averaged_per_logical_case")
    if len(rows) != expected:
        errors.append(f"expected {expected} rows, got {len(rows)}")

    versions = sorted({row.get("version", "") for row in rows})
    summary.append("versions=" + ",".join(versions))
    if target_version and not allow_version_mismatch and versions != [target_version]:
        errors.append(f"version must be exactly {target_version}, got {versions}")

    for ep in sorted(set(prefill_eps + decode_eps)):
        for phase in ("context", "generation"):
            count = sum(1 for row in rows if _safe_int(row.get("moe_ep_size")) == ep and row.get("phase") == phase)
            summary.append(f"rows ep={ep} phase={phase} count={count}")

    summary.append("distributions=" + ",".join(sorted({row.get("distribution", "") for row in rows})))
    summary.append("op_names=" + ",".join(sorted({row.get("op_name", "") for row in rows})))
    summary.append("kernel_sources=" + ",".join(sorted({row.get("kernel_source", "") for row in rows})))
    summary.append("used_cuda_graph=" + ",".join(sorted({row.get("used_cuda_graph", "") for row in rows})))
    summary.append("includes_gate_topk=" + ",".join(sorted({row.get("includes_gate_topk", "") for row in rows})))
    summary.append("includes_routed_scale=" + ",".join(sorted({row.get("includes_routed_scale", "") for row in rows})))
    for phase in ("context", "generation"):
        caps = sorted(
            {
                row.get("num_max_tokens_per_rank", "") + "/effective" + row.get("effective_num_max_tokens_per_rank", "")
                for row in rows
                if row.get("phase") == phase
            }
        )
        summary.append(f"caps {phase}=" + ",".join(caps))
    counts = Counter((row.get("moe_ep_size", ""), row.get("phase", ""), row.get("distribution", "")) for row in rows)
    for key, value in sorted(counts.items(), key=lambda item: (_safe_int(item[0][0]) or -1, item[0][1], item[0][2])):
        summary.append(f"count ep={key[0]} phase={key[1]} distribution={key[2]} rows={value}")

    for field, expected_value, error in ROW_INVARIANTS:
        if any(row.get(field) != expected_value for row in rows):
            errors.append(error)
    for field, error in POSITIVE_FLOAT_FIELDS:
        if any(not _positive_float(row.get(field)) for row in rows):
            errors.append(error)

    loader_key_counts = Counter(tuple(row.get(field, "") for field in LOADER_KEY_FIELDS) for row in rows)
    duplicate_loader_keys = sum(1 for count in loader_key_counts.values() if count > 1)
    summary.append(f"duplicate_loader_keys={duplicate_loader_keys}")
    if duplicate_loader_keys:
        errors.append(f"duplicate loader key groups: {duplicate_loader_keys}")

    summary.extend(f"ERROR: {error}" for error in errors)
    summary.append("VALIDATION=" + ("FAIL" if errors else "PASS"))
    text = "\n".join(summary) + "\n"
    summary_path.write_text(text)
    return text, errors


def _add_validate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--perf-path", required=True, type=Path)
    parser.add_argument("--prefill-ep-sizes", required=True)
    parser.add_argument("--decode-ep-sizes", required=True)
    parser.add_argument("--prefill-tokens", required=True)
    parser.add_argument("--decode-tokens", required=True)
    parser.add_argument("--distributions", required=True)
    parser.add_argument("--phase-order", required=True)
    parser.add_argument("--summary-path", required=True, type=Path)
    parser.add_argument("--target-sglang-version", default="")
    parser.add_argument("--allow-version-mismatch", choices=["0", "1"], default="1")
    parser.add_argument("--expect-single-perf-file", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge and validate DSv4 MegaMoE perf files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--input-root", required=True, type=Path)
    merge_parser.add_argument("--perf-file", required=True)
    merge_parser.add_argument("--output", required=True, type=Path)

    validate_parser = subparsers.add_parser("validate")
    _add_validate_args(validate_parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "merge":
        merged_files, rows = merge_perf_files(args.input_root, args.perf_file, args.output)
        print(f"merged_files={merged_files} rows={rows} output={args.output}")
        return

    text, errors = validate_perf_file(
        perf_path=args.perf_path,
        prefill_eps=_csv_ints(args.prefill_ep_sizes),
        decode_eps=_csv_ints(args.decode_ep_sizes),
        prefill_tokens=_csv_ints(args.prefill_tokens),
        decode_tokens=_csv_ints(args.decode_tokens),
        distributions=_csv_strings(args.distributions),
        phases=set(_csv_strings(args.phase_order)),
        summary_path=args.summary_path,
        target_version=args.target_sglang_version or None,
        allow_version_mismatch=args.allow_version_mismatch == "1",
        expect_single_perf_file=args.expect_single_perf_file,
    )
    print(text, end="")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a collector coverage manifest from JSON or JSONL perf rows."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from collector.planner.coverage import (
    COVERAGE_CANONICAL_VERSION,
    CoverageHeader,
    CoverageManifestError,
    canonical_perf_table,
    write,
)
from collector.planner.physical_keys import PHYSICAL_KEY_REGISTRY, PhysicalKeyError, physical_row_key


def _rows_from_document(value: Any, *, source: Path) -> list[dict[str, Any]]:
    values = value if isinstance(value, list) else [value]
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(values, start=1):
        if not isinstance(row, dict):
            raise TypeError(f"{source}: row {index} must be a JSON object")
        rows.append(row)
    return rows


def load_input_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON array/object or a JSONL sequence of row objects."""

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if not text.strip():
        return []

    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise TypeError(f"{source}:{line_number}: JSONL row must be an object") from None
            rows.append(value)
        return rows
    return _rows_from_document(document, source=source)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--rows", dest="input_path", required=True, help="JSON or JSONL perf rows")
    parser.add_argument("--output", required=True, help="Output .jsonl.gz manifest")
    parser.add_argument(
        "--perf-file",
        "--perf-filename",
        dest="perf_file",
        required=True,
        help="Registered perf filename used to derive PhysicalRowKey values",
    )
    parser.add_argument("--source-git-ref", required=True, help="Immutable source commit or tag")
    parser.add_argument("--backend-variant", required=True)
    parser.add_argument("--framework-version", required=True)
    parser.add_argument("--gpu-type", required=True)
    parser.add_argument("--sm-version", required=True, type=int)
    parser.add_argument(
        "--canonical-version",
        type=int,
        default=COVERAGE_CANONICAL_VERSION,
        help=f"Manifest canonical serialization version (default: {COVERAGE_CANONICAL_VERSION})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        table = canonical_perf_table(args.perf_file)
        rows = load_input_rows(args.input_path)
        keys = []
        for row in rows:
            key = physical_row_key(args.perf_file, row)
            if key is None:
                # The table is checked above, so this protects against registry
                # drift rather than silently emitting an empty manifest.
                raise CoverageManifestError(f"unknown perf table: {args.perf_file}")
            keys.append(key)

        schema = PHYSICAL_KEY_REGISTRY[table]
        header = CoverageHeader(
            source_git_ref=args.source_git_ref,
            backend_variant=args.backend_variant,
            framework_version=args.framework_version,
            gpu_type=args.gpu_type,
            sm_version=args.sm_version,
            perf_table=table,
            key_schema_version=schema.version,
            canonical_version=args.canonical_version,
        )
        manifest = write(args.output, header, keys)
    except (CoverageManifestError, PhysicalKeyError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    print(f"wrote {len(manifest.keys)} physical keys to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

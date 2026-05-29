# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize perf parquet changes for pull-request review."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pc
import pyarrow.parquet as pq

PERF_DATA_PREFIX = "src/aiconfigurator/systems/data"
COMMENT_MARKER = "<!-- perf-parquet-diff-comment -->"
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"


@dataclass(frozen=True)
class DiffEntry:
    status: str
    path: str
    old_path: str | None = None


@dataclass
class Snapshot:
    path: str
    table: pa.Table

    @property
    def row_count(self) -> int:
        return self.table.num_rows

    @property
    def columns(self) -> list[str]:
        return self.table.schema.names

    @property
    def schema(self) -> list[str]:
        return [f"{field.name}: {field.type}" for field in self.table.schema]

    @property
    def content_hash(self) -> str:
        return _hash_table(self.table)


@dataclass
class Comparison:
    path: str
    base_path: str | None
    status: str
    base_rows: int | None
    head_rows: int | None
    columns_match: bool | None
    content_match: bool | None
    base_hash: str | None
    head_hash: str | None


def _git(args: list[str], *, input_data: bytes | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        input=input_data,
        capture_output=True,
        check=check,
    )


def _git_file_exists(ref: str, path: str) -> bool:
    return _git(["cat-file", "-e", f"{ref}:{path}"], check=False).returncode == 0


def _smudge_lfs_pointer(data: bytes) -> bytes:
    if not data.startswith(LFS_POINTER_PREFIX):
        return data

    proc = _git(["lfs", "smudge"], input_data=data, check=False)
    if proc.returncode == 0 and proc.stdout and not proc.stdout.startswith(LFS_POINTER_PREFIX):
        return proc.stdout
    return data


def _read_git_file(ref: str, path: str) -> bytes:
    proc = _git(["show", f"{ref}:{path}"])
    return _smudge_lfs_pointer(proc.stdout)


def _parse_diff(base_ref: str, head_ref: str, path_prefix: str) -> list[DiffEntry]:
    proc = _git(["diff", "--name-status", "--find-renames", f"{base_ref}...{head_ref}", "--", path_prefix])
    entries: list[DiffEntry] = []
    for line in proc.stdout.decode("utf-8").splitlines():
        if not line:
            continue
        parts = line.split("\t")
        status = parts[0]
        code = status[0]
        if code == "R":
            entries.append(DiffEntry(status=status, old_path=parts[1], path=parts[2]))
        else:
            entries.append(DiffEntry(status=status, path=parts[1]))
    return entries


def _read_snapshot(ref: str, path: str) -> Snapshot:
    data = _read_git_file(ref, path)
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        table = pq.read_table(pa.BufferReader(data))
        return Snapshot(path=path, table=table)

    table = pc.read_csv(pa.BufferReader(data))
    return Snapshot(path=path, table=table)


def _hash_table(table: pa.Table) -> str:
    table = table.combine_chunks()
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()[:16]


def _legacy_txt_path(path: str) -> str:
    return f"{path[: -len('.parquet')]}.txt"


def _compare(base_ref: str, head_ref: str, entry: DiffEntry) -> Comparison:
    head_snapshot = None if entry.status.startswith("D") else _read_snapshot(head_ref, entry.path)

    base_path: str | None = entry.old_path or entry.path
    if entry.status.startswith("A") and Path(entry.path).suffix == ".parquet":
        legacy_path = _legacy_txt_path(entry.path)
        if _git_file_exists(base_ref, legacy_path):
            base_path = legacy_path
        elif not _git_file_exists(base_ref, entry.path):
            base_path = None

    base_snapshot = None
    if base_path is not None and _git_file_exists(base_ref, base_path):
        base_snapshot = _read_snapshot(base_ref, base_path)

    return Comparison(
        path=entry.path,
        base_path=base_path,
        status=entry.status,
        base_rows=base_snapshot.row_count if base_snapshot else None,
        head_rows=head_snapshot.row_count if head_snapshot else None,
        columns_match=base_snapshot.columns == head_snapshot.columns if base_snapshot and head_snapshot else None,
        content_match=base_snapshot.table.equals(head_snapshot.table, check_metadata=False)
        if base_snapshot and head_snapshot
        else None,
        base_hash=base_snapshot.content_hash if base_snapshot else None,
        head_hash=head_snapshot.content_hash if head_snapshot else None,
    )


def _render_table(rows: list[list[object]]) -> list[str]:
    if not rows:
        return []
    header = rows[0]
    out = [
        "| " + " | ".join(str(value) for value in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows[1:]:
        out.append("| " + " | ".join(str(value) for value in row) + " |")
    return out


def render_report(
    *,
    base_ref: str,
    head_ref: str,
    entries: list[DiffEntry],
    comparisons: list[Comparison],
    legacy_perf_changes: list[DiffEntry],
) -> str:
    converted = [c for c in comparisons if c.base_path and c.base_path.endswith(".txt")]
    converted_ok = [c for c in converted if c.columns_match and c.content_match]
    converted_bad = [c for c in converted if not (c.columns_match and c.content_match)]
    added = [c for c in comparisons if c.base_path is None and not c.status.startswith("D")]
    modified = [
        c for c in comparisons if c.base_path and c.base_path.endswith(".parquet") and not c.status.startswith("D")
    ]
    deleted = [c for c in comparisons if c.status.startswith("D")]

    lines = [
        COMMENT_MARKER,
        "## Perf Parquet Diff Report",
        "",
        f"Compared `{base_ref}` to `{head_ref}` for `{PERF_DATA_PREFIX}`.",
        "",
        f"- Parquet files changed: {len(comparisons)}",
        f"- CSV-to-parquet conversions checked: {len(converted)}",
        f"- Conversions with matching columns and rows: {len(converted_ok)}",
        f"- New parquet files without a base CSV/parquet counterpart: {len(added)}",
        f"- Modified or renamed parquet files: {len(modified)}",
        f"- Deleted parquet files: {len(deleted)}",
        f"- Legacy `*_perf.txt` files added or modified: {len(legacy_perf_changes)}",
        "",
    ]

    if not entries:
        lines.append("No perf data changes found.")
        return "\n".join(lines) + "\n"

    if converted_bad:
        lines.extend(["### Conversion Mismatches", ""])
        rows: list[list[object]] = [["File", "Base rows", "Head rows", "Columns", "Content"]]
        for item in converted_bad[:50]:
            rows.append(
                [
                    item.path,
                    item.base_rows,
                    item.head_rows,
                    "match" if item.columns_match else "changed",
                    "match" if item.content_match else f"{item.base_hash} -> {item.head_hash}",
                ]
            )
        lines.extend(_render_table(rows))
        if len(converted_bad) > 50:
            lines.append(f"\n...and {len(converted_bad) - 50} more conversion mismatches.")
        lines.append("")

    if legacy_perf_changes:
        lines.extend(["### Legacy Text Perf Files Still Changed", ""])
        rows = [["Status", "File"]]
        for item in legacy_perf_changes[:50]:
            rows.append([item.status, item.path])
        lines.extend(_render_table(rows))
        if len(legacy_perf_changes) > 50:
            lines.append(f"\n...and {len(legacy_perf_changes) - 50} more legacy text perf changes.")
        lines.append("")

    interesting = [c for c in added + modified + deleted if c not in converted_bad]
    if interesting:
        lines.extend(["### Other Parquet Changes", ""])
        rows = [["Status", "File", "Base rows", "Head rows", "Content"]]
        for item in interesting[:75]:
            if item.content_match is None:
                content = "n/a"
            else:
                content = "match" if item.content_match else f"{item.base_hash} -> {item.head_hash}"
            rows.append([item.status, item.path, item.base_rows, item.head_rows, content])
        lines.extend(_render_table(rows))
        if len(interesting) > 75:
            lines.append(f"\n...and {len(interesting) - 75} more parquet changes.")
        lines.append("")

    if converted and not converted_bad and not legacy_perf_changes:
        lines.append("All detected CSV-to-parquet conversions preserve column names and Arrow table content.")

    return "\n".join(lines).rstrip() + "\n"


def find_legacy_perf_changes(entries: list[DiffEntry]) -> list[DiffEntry]:
    """Return added/modified legacy text perf files; deletions are migration-safe."""
    return [entry for entry in entries if entry.path.endswith("_perf.txt") and not entry.status.startswith("D")]


def should_fail_strict(comparisons: list[Comparison], legacy_perf_changes: list[DiffEntry]) -> bool:
    converted_bad = [
        item
        for item in comparisons
        if item.base_path and item.base_path.endswith(".txt") and not (item.columns_match and item.content_match)
    ]
    return bool(converted_bad or legacy_perf_changes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", default="origin/main")
    parser.add_argument("--head-ref", default="HEAD")
    parser.add_argument("--path-prefix", default=PERF_DATA_PREFIX)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-strict", action="store_true", help="Do not fail on conversion mismatches.")
    args = parser.parse_args()

    entries = _parse_diff(args.base_ref, args.head_ref, args.path_prefix)
    parquet_entries = [entry for entry in entries if entry.path.endswith(".parquet")]
    legacy_perf_changes = find_legacy_perf_changes(entries)
    comparisons = [_compare(args.base_ref, args.head_ref, entry) for entry in parquet_entries]
    report = render_report(
        base_ref=args.base_ref,
        head_ref=args.head_ref,
        entries=entries,
        comparisons=comparisons,
        legacy_perf_changes=legacy_perf_changes,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    else:
        sys.stdout.write(report)

    if not args.no_strict and should_fail_strict(comparisons, legacy_perf_changes):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

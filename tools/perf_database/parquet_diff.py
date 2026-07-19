# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize perf parquet changes for pull-request review."""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import io
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pc
import pyarrow.parquet as pq

PERF_DATA_PREFIX = "aic-core/src/aiconfigurator_core/systems/data"
COMMENT_MARKER = "<!-- perf-parquet-diff-comment -->"
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"
DEFAULT_DETAIL_DIR = "parquet-diff-details"
FULL_DIFF_SUBDIR = "diffs"
INLINE_PREVIEW_ROWS = 3
MAX_INLINE_PREVIEW_CHARS = 45_000
MEASUREMENT_COLUMNS = frozenset(
    {
        "latency",
        "power",
        "energy",
        "avg_ms",
        "dispatch_avg_t_us",
        "dispatch_bandwidth_gbps",
        "dispatch_notify_us",
        "dispatch_sms",
        "dispatch_transmit_us",
        "combine_avg_t_us",
        "combine_bandwidth_gbps",
        "combine_notify_us",
        "combine_sms",
        "combine_transmit_us",
    }
)


@dataclass(frozen=True)
class DiffEntry:
    status: str
    path: str
    old_path: str | None = None


@dataclass(frozen=True)
class FullDiffArtifact:
    status: str
    path: str
    old_path: str | None
    diff_file: str


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
class RowDiff:
    key_columns: list[str]
    added_rows: int
    removed_rows: int
    modified_rows: int
    detail_files: dict[str, str]
    note: str | None = None
    detail_previews: dict[str, str] = field(default_factory=dict)


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
    row_diff: RowDiff | None


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


def _row_dicts(table: pa.Table) -> list[dict[str, object]]:
    return table.combine_chunks().to_pylist()


def _merge_columns(*column_lists: list[str]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for column_list in column_lists:
        for column in column_list:
            if column not in seen:
                columns.append(column)
                seen.add(column)
    return columns


def _select_key_columns(base_columns: list[str], head_columns: list[str]) -> list[str]:
    common_columns = [column for column in base_columns if column in head_columns]
    key_columns = [column for column in common_columns if column not in MEASUREMENT_COLUMNS]
    return key_columns


def _freeze_value(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return ("__nan__",)
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze_value(item)) for key, item in value.items()))
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    return value


def _row_key(row: dict[str, object], columns: list[str]) -> tuple[object, ...]:
    return tuple(_freeze_value(row.get(column)) for column in columns)


def _sort_key(row: dict[str, object], columns: list[str]) -> tuple[str, ...]:
    return tuple(repr(_freeze_value(row.get(column))) for column in columns)


def _rows_equal(base_row: dict[str, object], head_row: dict[str, object], columns: list[str]) -> bool:
    return all(_freeze_value(base_row.get(column)) == _freeze_value(head_row.get(column)) for column in columns)


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, sort_keys=True)
    return value


def _rows_to_csv_text(rows: list[dict[str, object]], columns: list[str], *, limit: int | None = None) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows[:limit]:
        writer.writerow({column: _csv_value(row.get(column)) for column in columns})
    return output.getvalue().rstrip()


def _write_rows_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in columns})


def _modified_csv_rows(
    modified_rows: list[tuple[dict[str, object], dict[str, object]]],
    *,
    key_columns: list[str],
    compare_columns: list[str],
) -> tuple[list[str], list[dict[str, object]]]:
    changed_columns = [
        column
        for column in compare_columns
        if any(_freeze_value(base.get(column)) != _freeze_value(head.get(column)) for base, head in modified_rows)
    ]
    fieldnames = key_columns + [field for column in changed_columns for field in (f"{column}__base", f"{column}__head")]
    rows: list[dict[str, object]] = []
    for base, head in modified_rows:
        row: dict[str, object] = {column: _csv_value(head.get(column)) for column in key_columns}
        for column in changed_columns:
            row[f"{column}__base"] = _csv_value(base.get(column))
            row[f"{column}__head"] = _csv_value(head.get(column))
        rows.append(row)
    return fieldnames, rows


def _modified_rows_to_csv_text(
    modified_rows: list[tuple[dict[str, object], dict[str, object]]],
    *,
    key_columns: list[str],
    compare_columns: list[str],
    limit: int | None = None,
) -> str:
    fieldnames, rows = _modified_csv_rows(
        modified_rows,
        key_columns=key_columns,
        compare_columns=compare_columns,
    )
    return _rows_to_csv_text(rows, fieldnames, limit=limit)


def _write_modified_csv(
    path: Path,
    modified_rows: list[tuple[dict[str, object], dict[str, object]]],
    *,
    key_columns: list[str],
    compare_columns: list[str],
) -> None:
    fieldnames, rows = _modified_csv_rows(
        modified_rows,
        key_columns=key_columns,
        compare_columns=compare_columns,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _ensure_trailing_newline(text: str) -> str:
    if text and not text.endswith("\n"):
        return f"{text}\n"
    return text


def _file_text_lines(ref: str, path: str) -> list[str]:
    data = _read_git_file(ref, path)
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        table = pq.read_table(pa.BufferReader(data))
        text = _rows_to_csv_text(_row_dicts(table), table.schema.names)
    else:
        text = data.decode("utf-8", errors="replace")
    return _ensure_trailing_newline(text).splitlines()


def _render_file_diff(
    *,
    status: str,
    base_path: str | None,
    head_path: str | None,
    base_lines: list[str],
    head_lines: list[str],
) -> str:
    fromfile = f"a/{base_path}" if base_path else "/dev/null"
    tofile = f"b/{head_path}" if head_path else "/dev/null"
    diff = "\n".join(
        difflib.unified_diff(
            base_lines,
            head_lines,
            fromfile=fromfile,
            tofile=tofile,
            lineterm="",
        )
    )
    if diff:
        return f"{diff}\n"

    path = head_path or base_path or "unknown"
    return f"# {status} {path}\n# No content changes after perf text conversion.\n"


def _detail_output_path(detail_dir: Path, parquet_path: str, kind: str) -> tuple[Path, str]:
    relative_path = Path(f"{parquet_path}.{kind}.csv")
    return detail_dir / relative_path, relative_path.as_posix()


def _write_detail_rows(
    *,
    detail_dir: Path | None,
    parquet_path: str,
    kind: str,
    rows: list[dict[str, object]],
    columns: list[str],
) -> str | None:
    if not detail_dir or not rows:
        return None
    output_path, relative_path = _detail_output_path(detail_dir, parquet_path, kind)
    _write_rows_csv(output_path, rows, columns)
    return relative_path


def _write_detail_modified_rows(
    *,
    detail_dir: Path | None,
    parquet_path: str,
    modified_rows: list[tuple[dict[str, object], dict[str, object]]],
    key_columns: list[str],
    compare_columns: list[str],
) -> str | None:
    if not detail_dir or not modified_rows:
        return None
    output_path, relative_path = _detail_output_path(detail_dir, parquet_path, "modified")
    _write_modified_csv(output_path, modified_rows, key_columns=key_columns, compare_columns=compare_columns)
    return relative_path


def _full_diff_output_path(detail_dir: Path, entry: DiffEntry) -> tuple[Path, str]:
    relative_path = Path(FULL_DIFF_SUBDIR) / f"{entry.path}.diff"
    return detail_dir / relative_path, relative_path.as_posix()


def _write_full_diff_artifacts(
    *,
    detail_dir: Path,
    base_ref: str,
    head_ref: str,
    entries: list[DiffEntry],
) -> list[FullDiffArtifact]:
    artifacts: list[FullDiffArtifact] = []
    for entry in entries:
        base_path = None if entry.status.startswith("A") else entry.old_path or entry.path
        head_path = None if entry.status.startswith("D") else entry.path
        if base_path and not _git_file_exists(base_ref, base_path):
            base_path = None
        if head_path and not _git_file_exists(head_ref, head_path):
            head_path = None

        base_lines = _file_text_lines(base_ref, base_path) if base_path else []
        head_lines = _file_text_lines(head_ref, head_path) if head_path else []
        output_path, relative_path = _full_diff_output_path(detail_dir, entry)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            _render_file_diff(
                status=entry.status,
                base_path=base_path,
                head_path=head_path,
                base_lines=base_lines,
                head_lines=head_lines,
            )
        )
        artifacts.append(
            FullDiffArtifact(
                status=entry.status,
                path=entry.path,
                old_path=entry.old_path,
                diff_file=relative_path,
            )
        )
    return artifacts


def _has_duplicate_keys(rows: list[dict[str, object]], key_columns: list[str]) -> bool:
    counts = Counter(_row_key(row, key_columns) for row in rows)
    return any(count > 1 for count in counts.values())


def _counter_rows_delta(
    *,
    base_rows: list[dict[str, object]],
    head_rows: list[dict[str, object]],
    columns: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    base_counter = Counter(_row_key(row, columns) for row in base_rows)
    head_counter = Counter(_row_key(row, columns) for row in head_rows)
    base_by_key = _rows_by_key(base_rows, columns)
    head_by_key = _rows_by_key(head_rows, columns)
    added_rows: list[dict[str, object]] = []
    removed_rows: list[dict[str, object]] = []
    for key, count in sorted((head_counter - base_counter).items(), key=lambda item: repr(item[0])):
        added_rows.extend(head_by_key[key].popleft() for _ in range(count))
    for key, count in sorted((base_counter - head_counter).items(), key=lambda item: repr(item[0])):
        removed_rows.extend(base_by_key[key].popleft() for _ in range(count))
    return (
        sorted(added_rows, key=lambda row: _sort_key(row, columns)),
        sorted(removed_rows, key=lambda row: _sort_key(row, columns)),
    )


def _rows_by_key(
    rows: list[dict[str, object]],
    columns: list[str],
) -> defaultdict[tuple[object, ...], deque[dict[str, object]]]:
    rows_by_key: defaultdict[tuple[object, ...], deque[dict[str, object]]] = defaultdict(deque)
    for row in sorted(rows, key=lambda item: _sort_key(item, columns)):
        rows_by_key[_row_key(row, columns)].append(row)
    return rows_by_key


def _unmatched_full_rows(
    rows: list[dict[str, object]],
    *,
    matched_rows: Counter[tuple[object, ...]],
    columns: list[str],
) -> list[dict[str, object]]:
    counter = Counter(_row_key(row, columns) for row in rows)
    unmatched = counter - matched_rows
    rows_by_key = _rows_by_key(rows, columns)
    result: list[dict[str, object]] = []
    for key, count in sorted(unmatched.items(), key=lambda item: repr(item[0])):
        result.extend(rows_by_key[key].popleft() for _ in range(count))
    return sorted(result, key=lambda row: _sort_key(row, columns))


def _diff_rows_with_duplicate_keys(
    *,
    base_rows: list[dict[str, object]],
    head_rows: list[dict[str, object]],
    key_columns: list[str],
    columns: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[tuple[dict[str, object], dict[str, object]]]]:
    base_groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    head_groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in base_rows:
        base_groups[_row_key(row, key_columns)].append(row)
    for row in head_rows:
        head_groups[_row_key(row, key_columns)].append(row)

    added_rows: list[dict[str, object]] = []
    removed_rows: list[dict[str, object]] = []
    modified_rows: list[tuple[dict[str, object], dict[str, object]]] = []

    for key in sorted(set(base_groups) | set(head_groups), key=repr):
        base_group = base_groups.get(key, [])
        head_group = head_groups.get(key, [])
        if not base_group:
            added_rows.extend(head_group)
            continue
        if not head_group:
            removed_rows.extend(base_group)
            continue

        matched_rows = Counter(_row_key(row, columns) for row in base_group) & Counter(
            _row_key(row, columns) for row in head_group
        )
        unmatched_base = _unmatched_full_rows(base_group, matched_rows=matched_rows, columns=columns)
        unmatched_head = _unmatched_full_rows(head_group, matched_rows=matched_rows, columns=columns)
        pair_count = min(len(unmatched_base), len(unmatched_head))
        modified_rows.extend(zip(unmatched_base[:pair_count], unmatched_head[:pair_count], strict=True))
        removed_rows.extend(unmatched_base[pair_count:])
        added_rows.extend(unmatched_head[pair_count:])

    added_rows = sorted(added_rows, key=lambda row: _sort_key(row, columns))
    removed_rows = sorted(removed_rows, key=lambda row: _sort_key(row, columns))
    modified_rows = sorted(modified_rows, key=lambda pair: _sort_key(pair[1], columns))
    return added_rows, removed_rows, modified_rows


def _diff_snapshots(
    parquet_path: str,
    base_snapshot: Snapshot | None,
    head_snapshot: Snapshot | None,
    *,
    detail_dir: Path | None,
) -> RowDiff | None:
    if base_snapshot is None and head_snapshot is None:
        return None

    base_columns = base_snapshot.columns if base_snapshot else []
    head_columns = head_snapshot.columns if head_snapshot else []
    columns = _merge_columns(base_columns, head_columns)
    key_columns = _select_key_columns(base_columns or head_columns, head_columns or base_columns)
    detail_files: dict[str, str] = {}
    detail_previews: dict[str, str] = {}

    if base_snapshot is None:
        head_rows = sorted(_row_dicts(head_snapshot.table), key=lambda row: _sort_key(row, columns))
        detail_previews["added"] = _rows_to_csv_text(head_rows, columns, limit=INLINE_PREVIEW_ROWS)
        detail_file = _write_detail_rows(
            detail_dir=detail_dir,
            parquet_path=parquet_path,
            kind="added",
            rows=head_rows,
            columns=columns,
        )
        if detail_file:
            detail_files["added"] = detail_file
        return RowDiff(key_columns, len(head_rows), 0, 0, detail_files, detail_previews=detail_previews)

    if head_snapshot is None:
        base_rows = sorted(_row_dicts(base_snapshot.table), key=lambda row: _sort_key(row, columns))
        detail_previews["removed"] = _rows_to_csv_text(base_rows, columns, limit=INLINE_PREVIEW_ROWS)
        detail_file = _write_detail_rows(
            detail_dir=detail_dir,
            parquet_path=parquet_path,
            kind="removed",
            rows=base_rows,
            columns=columns,
        )
        if detail_file:
            detail_files["removed"] = detail_file
        return RowDiff(key_columns, 0, len(base_rows), 0, detail_files, detail_previews=detail_previews)

    base_rows = _row_dicts(base_snapshot.table)
    head_rows = _row_dicts(head_snapshot.table)
    if not key_columns:
        added_rows, removed_rows = _counter_rows_delta(base_rows=base_rows, head_rows=head_rows, columns=columns)
        added_rows = sorted(added_rows, key=lambda row: _sort_key(row, columns))
        removed_rows = sorted(removed_rows, key=lambda row: _sort_key(row, columns))
        for kind, rows in (("added", added_rows), ("removed", removed_rows)):
            if rows:
                detail_previews[kind] = _rows_to_csv_text(rows, columns, limit=INLINE_PREVIEW_ROWS)
            detail_file = _write_detail_rows(
                detail_dir=detail_dir,
                parquet_path=parquet_path,
                kind=kind,
                rows=rows,
                columns=columns,
            )
            if detail_file:
                detail_files[kind] = detail_file
        return RowDiff(
            key_columns,
            len(added_rows),
            len(removed_rows),
            0,
            detail_files,
            note="unavailable keys; exact full-row add/remove diff used",
            detail_previews=detail_previews,
        )

    duplicate_keys = _has_duplicate_keys(base_rows, key_columns) or _has_duplicate_keys(head_rows, key_columns)
    if duplicate_keys:
        added_rows, removed_rows, modified_rows = _diff_rows_with_duplicate_keys(
            base_rows=base_rows,
            head_rows=head_rows,
            key_columns=key_columns,
            columns=columns,
        )
        for kind, rows in (("added", added_rows), ("removed", removed_rows)):
            if rows:
                detail_previews[kind] = _rows_to_csv_text(rows, columns, limit=INLINE_PREVIEW_ROWS)
            detail_file = _write_detail_rows(
                detail_dir=detail_dir,
                parquet_path=parquet_path,
                kind=kind,
                rows=rows,
                columns=columns,
            )
            if detail_file:
                detail_files[kind] = detail_file

        compare_columns = [column for column in columns if column not in key_columns]
        detail_file = _write_detail_modified_rows(
            detail_dir=detail_dir,
            parquet_path=parquet_path,
            modified_rows=modified_rows,
            key_columns=key_columns,
            compare_columns=compare_columns,
        )
        if modified_rows:
            detail_previews["modified"] = _modified_rows_to_csv_text(
                modified_rows,
                key_columns=key_columns,
                compare_columns=compare_columns,
                limit=INLINE_PREVIEW_ROWS,
            )
        if detail_file:
            detail_files["modified"] = detail_file

        return RowDiff(
            key_columns,
            len(added_rows),
            len(removed_rows),
            len(modified_rows),
            detail_files,
            note="duplicate keys; unmatched rows paired within key",
            detail_previews=detail_previews,
        )

    base_by_key = {_row_key(row, key_columns): row for row in base_rows}
    head_by_key = {_row_key(row, key_columns): row for row in head_rows}
    added_keys = sorted(set(head_by_key) - set(base_by_key), key=repr)
    removed_keys = sorted(set(base_by_key) - set(head_by_key), key=repr)
    common_keys = sorted(set(base_by_key) & set(head_by_key), key=repr)
    added_rows = [head_by_key[key] for key in added_keys]
    removed_rows = [base_by_key[key] for key in removed_keys]
    modified_rows = [
        (base_by_key[key], head_by_key[key])
        for key in common_keys
        if not _rows_equal(base_by_key[key], head_by_key[key], columns)
    ]

    for kind, rows in (("added", added_rows), ("removed", removed_rows)):
        if rows:
            detail_previews[kind] = _rows_to_csv_text(rows, columns, limit=INLINE_PREVIEW_ROWS)
        detail_file = _write_detail_rows(
            detail_dir=detail_dir,
            parquet_path=parquet_path,
            kind=kind,
            rows=rows,
            columns=columns,
        )
        if detail_file:
            detail_files[kind] = detail_file

    compare_columns = [column for column in columns if column not in key_columns]
    detail_file = _write_detail_modified_rows(
        detail_dir=detail_dir,
        parquet_path=parquet_path,
        modified_rows=modified_rows,
        key_columns=key_columns,
        compare_columns=compare_columns,
    )
    if modified_rows:
        detail_previews["modified"] = _modified_rows_to_csv_text(
            modified_rows,
            key_columns=key_columns,
            compare_columns=compare_columns,
            limit=INLINE_PREVIEW_ROWS,
        )
    if detail_file:
        detail_files["modified"] = detail_file

    return RowDiff(
        key_columns,
        len(added_rows),
        len(removed_rows),
        len(modified_rows),
        detail_files,
        detail_previews=detail_previews,
    )


def _write_diff_manifest(detail_dir: Path, full_diff_artifacts: list[FullDiffArtifact]) -> None:
    rows = [
        {
            "status": artifact.status,
            "file": artifact.path,
            "old_file": artifact.old_path or "",
            "full_diff_file": artifact.diff_file,
        }
        for artifact in full_diff_artifacts
    ]
    if rows:
        _write_rows_csv(detail_dir / "changed-files.csv", rows, ["status", "file", "old_file", "full_diff_file"])


def _write_detail_summary(
    detail_dir: Path,
    comparisons: list[Comparison],
    full_diff_artifacts: list[FullDiffArtifact] | None = None,
) -> None:
    full_diff_files = {artifact.path: artifact.diff_file for artifact in full_diff_artifacts or []}
    rows: list[dict[str, object]] = []
    for comparison in comparisons:
        row_diff = comparison.row_diff
        if row_diff is None:
            continue
        rows.append(
            {
                "file": comparison.path,
                "status": comparison.status,
                "base_rows": comparison.base_rows,
                "head_rows": comparison.head_rows,
                "added_rows": row_diff.added_rows,
                "removed_rows": row_diff.removed_rows,
                "modified_rows": row_diff.modified_rows,
                "key_columns": ",".join(row_diff.key_columns),
                "detail_files": ",".join(row_diff.detail_files.values()),
                "full_diff_file": full_diff_files.get(comparison.path, ""),
                "note": row_diff.note or "",
            }
        )
    if rows:
        _write_rows_csv(
            detail_dir / "summary.csv",
            rows,
            [
                "file",
                "status",
                "base_rows",
                "head_rows",
                "added_rows",
                "removed_rows",
                "modified_rows",
                "key_columns",
                "detail_files",
                "full_diff_file",
                "note",
            ],
        )


def _legacy_txt_path(path: str) -> str:
    return f"{path[: -len('.parquet')]}.txt"


def _compare(base_ref: str, head_ref: str, entry: DiffEntry, detail_dir: Path | None = None) -> Comparison:
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

    row_diff = _diff_snapshots(entry.path, base_snapshot, head_snapshot, detail_dir=detail_dir)

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
        row_diff=row_diff,
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


def _render_inline_diff_previews(comparisons: list[Comparison]) -> list[str]:
    preview_lines = [
        "### Per-File Row Diff Preview",
        "",
        f"Showing the first {INLINE_PREVIEW_ROWS} rows per diff kind for each changed parquet file. "
        f"Full exact CSVs are in `{DEFAULT_DETAIL_DIR}/`.",
        "",
    ]
    budget_used = 0
    rendered = 0
    for comparison in comparisons:
        row_diff = comparison.row_diff
        if not row_diff or not row_diff.detail_previews:
            continue

        section = [
            f"<details><summary>{comparison.path}</summary>",
            "",
            f"- Rows: +{row_diff.added_rows} / -{row_diff.removed_rows} / ~{row_diff.modified_rows}",
            f"- Key columns: `{', '.join(row_diff.key_columns)}`" if row_diff.key_columns else "- Key columns: n/a",
        ]
        if row_diff.note:
            section.append(f"- Note: {row_diff.note}")
        section.append("")

        for kind, preview in row_diff.detail_previews.items():
            detail_file = row_diff.detail_files.get(kind)
            if detail_file:
                section.append(f"**{kind} rows** - full CSV: `{DEFAULT_DETAIL_DIR}/{detail_file}`")
            else:
                section.append(f"**{kind} rows**")
            section.extend(["", "```csv", preview, "```", ""])
        section.extend(["</details>", ""])

        section_text = "\n".join(section)
        if budget_used + len(section_text) > MAX_INLINE_PREVIEW_CHARS:
            remaining = len([c for c in comparisons if c.row_diff and c.row_diff.detail_previews]) - rendered
            preview_lines.append(
                f"...inline previews truncated after {rendered} files to keep the PR comment readable; "
                f"{remaining} more files are available in `{DEFAULT_DETAIL_DIR}/`."
            )
            preview_lines.append("")
            break

        preview_lines.append(section_text)
        budget_used += len(section_text)
        rendered += 1

    return preview_lines if rendered else []


def render_report(
    *,
    base_ref: str,
    head_ref: str,
    entries: list[DiffEntry],
    comparisons: list[Comparison],
    legacy_perf_changes: list[DiffEntry],
    full_diff_artifacts: list[FullDiffArtifact] | None = None,
) -> str:
    converted = [c for c in comparisons if c.base_path and c.base_path.endswith(".txt")]
    converted_ok = [c for c in converted if c.columns_match and c.content_match]
    converted_bad = [c for c in converted if not (c.columns_match and c.content_match)]
    added = [c for c in comparisons if c.base_path is None and not c.status.startswith("D")]
    modified = [
        c for c in comparisons if c.base_path and c.base_path.endswith(".parquet") and not c.status.startswith("D")
    ]
    deleted = [c for c in comparisons if c.status.startswith("D")]
    row_diffs = [c.row_diff for c in comparisons if c.row_diff]
    added_rows = sum(row_diff.added_rows for row_diff in row_diffs)
    removed_rows = sum(row_diff.removed_rows for row_diff in row_diffs)
    modified_rows = sum(row_diff.modified_rows for row_diff in row_diffs)

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
        f"- Row-level changes: +{added_rows} / -{removed_rows} / ~{modified_rows}",
    ]
    if full_diff_artifacts is not None:
        diff_artifact_label = "file" if len(full_diff_artifacts) == 1 else "files"
        lines.append(
            f"- Full per-file diff artifacts: {len(full_diff_artifacts)} {diff_artifact_label} under "
            f"`{DEFAULT_DETAIL_DIR}/{FULL_DIFF_SUBDIR}/`"
        )
    lines.append("")

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

    lines.extend(_render_inline_diff_previews(comparisons))

    has_row_detail_files = any(c.row_diff and c.row_diff.detail_files for c in comparisons)
    if full_diff_artifacts is not None or has_row_detail_files:
        lines.extend(["### Artifact Contents", ""])
        if full_diff_artifacts is not None:
            diff_artifact_label = "file" if len(full_diff_artifacts) == 1 else "files"
            lines.append(
                f"- Full per-file unified diffs: `{DEFAULT_DETAIL_DIR}/{FULL_DIFF_SUBDIR}/` "
                f"({len(full_diff_artifacts)} {diff_artifact_label})"
            )
        if has_row_detail_files:
            lines.append(f"- Exact row-level CSVs: `{DEFAULT_DETAIL_DIR}/` (listed in `summary.csv`)")
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
    parser.add_argument(
        "--detail-dir",
        type=Path,
        default=None,
        help="Directory for full diff and exact row-level CSV artifacts.",
    )
    parser.add_argument("--no-strict", action="store_true", help="Do not fail on conversion mismatches.")
    args = parser.parse_args()

    entries = _parse_diff(args.base_ref, args.head_ref, args.path_prefix)
    parquet_entries = [entry for entry in entries if entry.path.endswith(".parquet")]
    legacy_perf_changes = find_legacy_perf_changes(entries)
    comparisons = [
        _compare(args.base_ref, args.head_ref, entry, detail_dir=args.detail_dir) for entry in parquet_entries
    ]
    full_diff_artifacts = None
    if args.detail_dir:
        full_diff_artifacts = _write_full_diff_artifacts(
            detail_dir=args.detail_dir,
            base_ref=args.base_ref,
            head_ref=args.head_ref,
            entries=entries,
        )
        _write_diff_manifest(args.detail_dir, full_diff_artifacts)
        _write_detail_summary(args.detail_dir, comparisons, full_diff_artifacts)
    report = render_report(
        base_ref=args.base_ref,
        head_ref=args.head_ref,
        entries=entries,
        comparisons=comparisons,
        legacy_perf_changes=legacy_perf_changes,
        full_diff_artifacts=full_diff_artifacts,
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

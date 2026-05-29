# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Print a parquet perf table as CSV for git diff textconv."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"


def _smudge_lfs_pointer(data: bytes) -> bytes:
    result = subprocess.run(["git", "lfs", "smudge"], input=data, capture_output=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"Failed to smudge Git LFS pointer: {stderr}")
    if result.stdout.startswith(LFS_POINTER_PREFIX):
        raise RuntimeError("Git LFS smudge returned a pointer; fetch the LFS object before diffing")
    return result.stdout


def _read_parquet_table(path: Path):
    with path.open("rb") as f:
        prefix = f.read(len(LFS_POINTER_PREFIX))
    if prefix == LFS_POINTER_PREFIX:
        return pq.read_table(pa.BufferReader(_smudge_lfs_pointer(path.read_bytes())))
    return pq.read_table(path)


def parquet_to_csv(path: Path) -> None:
    table = _read_parquet_table(path)
    fieldnames = table.schema.names
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in table.to_pylist():
        writer.writerow({key: "" if value is None else value for key, value in row.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    parquet_to_csv(args.path)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except OSError:
            pass

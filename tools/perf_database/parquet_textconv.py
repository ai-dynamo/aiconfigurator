# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Print a parquet perf table as CSV for git diff textconv."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pyarrow.parquet as pq


def parquet_to_csv(path: Path) -> None:
    table = pq.read_table(path)
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

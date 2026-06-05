#!/usr/bin/env python3
"""Parse vLLM V1 iteration-detail logs into CSV.

Start vLLM with `--enable-logging-iteration-details`, tee stdout/stderr to a
log file, then run this parser over that log.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import TextIO

ITERATION_RE = re.compile(
    r"Iteration\((?P<iteration>\d+)\):\s+"
    r"(?P<num_context_requests>\d+)\s+context requests,\s+"
    r"(?P<num_context_tokens>\d+)\s+context tokens,\s+"
    r"(?P<num_decode_requests>\d+)\s+generation requests,\s+"
    r"(?P<num_decode_tokens>\d+)\s+generation tokens,\s+"
    r"iteration elapsed time:\s+"
    r"(?P<latency_ms>[0-9]+(?:\.[0-9]+)?)\s+ms"
)


CSV_COLUMNS = [
    "iteration",
    "num_context_requests",
    "num_context_tokens",
    "num_decode_requests",
    "num_decode_tokens",
    "latency_ms",
]


def _iter_lines(paths: list[Path]) -> Iterable[str]:
    if not paths:
        yield from sys.stdin
        return
    for path in paths:
        with path.open() as f:
            yield from f


def parse_lines(lines: Iterable[str]) -> Iterable[dict[str, str]]:
    for line in lines:
        match = ITERATION_RE.search(line)
        if match is not None:
            yield match.groupdict()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse vLLM --enable-logging-iteration-details logs to CSV"
    )
    parser.add_argument("logs", nargs="*", type=Path, help="Log file(s); stdin if omitted")
    parser.add_argument("-o", "--output", type=Path, help="Output CSV path; stdout if omitted")
    args = parser.parse_args()

    out_f: TextIO
    if args.output is None:
        out_f = sys.stdout
        close_out = False
    else:
        out_f = args.output.open("w", newline="")
        close_out = True

    try:
        writer = csv.DictWriter(out_f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(parse_lines(_iter_lines(args.logs)))
    finally:
        if close_out:
            out_f.close()


if __name__ == "__main__":
    main()

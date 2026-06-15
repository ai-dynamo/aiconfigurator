#!/usr/bin/env python3
"""Write classified FPM step rows from the raw detail CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

OUTPUT_COLUMNS = [
    "phase",
    "workload_segment",
    "counter_id",
    "worker_id",
    "dp_rank",
    "ctx_tokens",
    "ctx_requests",
    "ctx_kv_tokens",
    "decode_tokens",
    "decode_requests",
    "decode_kv_tokens",
    "mean_decode_kv_tokens",
    "queued_ctx_tokens",
    "queued_ctx_requests",
    "queued_decode_requests",
    "queued_decode_kv_tokens",
    "latency_ms",
]


def _as_int(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    return int(float(value)) if value not in ("", None) else 0


def _classify(row: dict[str, str]) -> str:
    existing = row.get("phase")
    if existing:
        return existing
    ctx_tokens = _as_int(row, "sum_prefill_tokens")
    decode_requests = _as_int(row, "num_decode_requests")
    if ctx_tokens > 0 and decode_requests > 0:
        return "mixed"
    if ctx_tokens > 0:
        return "context"
    if decode_requests > 0:
        return "decode"
    return "idle"


def summarize_rows(rows: list[dict[str, str]], *, include_idle: bool = False) -> list[dict[str, object]]:
    out = []
    for row in rows:
        phase = _classify(row)
        if phase == "idle" and not include_idle:
            continue

        decode_requests = _as_int(row, "num_decode_requests")
        decode_kv_tokens = _as_int(row, "sum_decode_kv_tokens")
        mean_decode_kv_tokens = decode_kv_tokens / decode_requests if decode_requests > 0 else 0.0
        out.append(
            {
                "phase": phase,
                "workload_segment": row.get("workload_segment", ""),
                "counter_id": _as_int(row, "counter_id"),
                "worker_id": row.get("worker_id", ""),
                "dp_rank": _as_int(row, "dp_rank"),
                "ctx_tokens": _as_int(row, "sum_prefill_tokens"),
                "ctx_requests": _as_int(row, "num_prefill_requests"),
                "ctx_kv_tokens": _as_int(row, "sum_prefill_kv_tokens"),
                "decode_tokens": decode_requests,
                "decode_requests": decode_requests,
                "decode_kv_tokens": decode_kv_tokens,
                "mean_decode_kv_tokens": f"{mean_decode_kv_tokens:.3f}",
                "queued_ctx_tokens": _as_int(row, "queued_prefill_tokens"),
                "queued_ctx_requests": _as_int(row, "queued_prefill_requests"),
                "queued_decode_requests": _as_int(row, "queued_decode_requests"),
                "queued_decode_kv_tokens": _as_int(row, "queued_decode_kv_tokens"),
                "latency_ms": row.get("latency_ms", ""),
            }
        )
    return out


def summarize_file(detail_path: Path, output_path: Path, *, include_idle: bool = False) -> int:
    with detail_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = summarize_rows(rows, include_idle=include_idle)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)
    return len(summary_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-idle", action="store_true")
    args = parser.parse_args()

    rows = summarize_file(Path(args.detail), Path(args.output), include_idle=args.include_idle)
    print(f"wrote_phase_rows={rows} output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

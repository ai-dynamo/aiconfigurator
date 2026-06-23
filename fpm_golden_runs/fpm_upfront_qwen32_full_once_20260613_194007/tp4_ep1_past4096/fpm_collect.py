#!/usr/bin/env python3
"""Collect raw Dynamo/vLLM ForwardPassMetrics rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import signal
import sys
import time

import zmq
from dynamo.common.forward_pass_metrics import decode

running = True


def stop(_signum, _frame):
    global running
    running = False


def classify_phase(sum_prefill_tokens: int, num_decode_requests: int) -> str:
    if sum_prefill_tokens > 0 and num_decode_requests > 0:
        return "mixed"
    if sum_prefill_tokens > 0:
        return "context"
    if num_decode_requests > 0:
        return "decode"
    return "idle"


def read_segment(segment_file: str | None) -> str:
    if not segment_file:
        return ""
    try:
        return Path(segment_file).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--detail-output", required=True)
    parser.add_argument("--segment-file", default=None)
    parser.add_argument("--idle-timeout", type=float, default=0.0)
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://127.0.0.1:{args.port}")
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.RCVTIMEO, 1000)

    last_row = time.monotonic()
    with open(args.output, "w", newline="") as f, open(
        args.detail_output, "w", newline=""
    ) as detail_f:
        writer = csv.writer(f)
        detail_writer = csv.writer(detail_f)
        writer.writerow(["num_context_tokens", "num_decode_tokens", "latency_ms"])
        detail_writer.writerow(
            [
                "phase",
                "workload_segment",
                "counter_id",
                "worker_id",
                "dp_rank",
                "sum_prefill_tokens",
                "num_prefill_requests",
                "sum_prefill_kv_tokens",
                "var_prefill_length",
                "num_decode_requests",
                "sum_decode_kv_tokens",
                "var_decode_kv_tokens",
                "queued_prefill_requests",
                "queued_prefill_tokens",
                "queued_var_prefill_length",
                "queued_decode_requests",
                "queued_decode_kv_tokens",
                "queued_var_decode_kv_tokens",
                "latency_ms",
            ]
        )
        f.flush()
        detail_f.flush()

        while running:
            try:
                _topic, _seq, payload = sock.recv_multipart()
            except zmq.Again:
                if args.idle_timeout > 0 and time.monotonic() - last_row > args.idle_timeout:
                    break
                continue

            metrics = decode(payload)
            if metrics is None or metrics.wall_time <= 0:
                continue

            scheduled = metrics.scheduled_requests
            queued = metrics.queued_requests
            latency_ms = f"{metrics.wall_time * 1000.0:.3f}"
            phase = classify_phase(
                int(scheduled.sum_prefill_tokens),
                int(scheduled.num_decode_requests),
            )
            workload_segment = read_segment(args.segment_file)
            row = [
                int(scheduled.sum_prefill_tokens),
                int(scheduled.num_decode_requests),
                latency_ms,
            ]
            detail_row = [
                phase,
                workload_segment,
                int(metrics.counter_id),
                metrics.worker_id,
                int(metrics.dp_rank),
                int(scheduled.sum_prefill_tokens),
                int(scheduled.num_prefill_requests),
                int(scheduled.sum_prefill_kv_tokens),
                f"{scheduled.var_prefill_length:.3f}",
                int(scheduled.num_decode_requests),
                int(scheduled.sum_decode_kv_tokens),
                f"{scheduled.var_decode_kv_tokens:.3f}",
                int(queued.num_prefill_requests),
                int(queued.sum_prefill_tokens),
                f"{queued.var_prefill_length:.3f}",
                int(queued.num_decode_requests),
                int(queued.sum_decode_kv_tokens),
                f"{queued.var_decode_kv_tokens:.3f}",
                latency_ms,
            ]
            writer.writerow(row)
            detail_writer.writerow(detail_row)
            f.flush()
            detail_f.flush()
            print(",".join(map(str, row)), flush=True)
            last_row = time.monotonic()

    sock.close(linger=0)
    return 0


if __name__ == "__main__":
    sys.exit(main())

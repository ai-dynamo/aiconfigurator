"""Parse vLLM native profiler NVTX ranges into per-iteration latency rows.

vLLM's CUDA profiler path emits one NVTX range per worker step:

  execute_context_<ctx_reqs>(<ctx_tokens>)_generation_<gen_reqs>(<gen_tokens>)

This script reads an Nsight Systems sqlite export and produces rows with the
shape from that range plus GPU timing from CUPTI kernels launched inside it.

Use `gpu_span_us` as the closest no-patch forward-pass iteration latency:
it is the time from the first GPU kernel start to the last GPU kernel end for
that worker step. `wall_us` is the host NVTX duration and can understate GPU
time when vLLM schedules work asynchronously.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sqlite3
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

_VLLM_EXEC_RE = re.compile(
    r"^execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)$"
)
_GLOBAL_PID_MASK = -16777216


@dataclass(frozen=True)
class IterRange:
    rank: int
    index: int
    tid: int
    start_ns: int
    end_ns: int
    num_context_requests: int
    num_context_tokens: int
    num_decode_requests: int
    num_decode_tokens: int


@dataclass
class KernelStats:
    first_start_ns: int | None = None
    last_end_ns: int | None = None
    sum_ns: int = 0
    count: int = 0

    def add(self, start_ns: int, end_ns: int) -> None:
        if self.first_start_ns is None or start_ns < self.first_start_ns:
            self.first_start_ns = start_ns
        if self.last_end_ns is None or end_ns > self.last_end_ns:
            self.last_end_ns = end_ns
        self.sum_ns += end_ns - start_ns
        self.count += 1

    @property
    def span_ns(self) -> int:
        if self.first_start_ns is None or self.last_end_ns is None:
            return 0
        return self.last_end_ns - self.first_start_ns


def _require_table(cur: sqlite3.Cursor, name: str) -> None:
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
    if cur.fetchone() is not None:
        return
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = ", ".join(row[0] for row in cur.fetchall())
    raise RuntimeError(f"missing required table {name}. Tables present: {tables}")


def _iter_ranges(cur: sqlite3.Cursor) -> list[IterRange]:
    _require_table(cur, "NVTX_EVENTS")
    cur.execute(
        "SELECT text, start, end, globalTid FROM NVTX_EVENTS "
        "WHERE text LIKE 'execute_context_%'"
    )
    raw_by_tid: dict[int, list[tuple[int, int, tuple[int, int, int, int]]]] = (
        defaultdict(list)
    )
    for text, start_ns, end_ns, tid in cur.fetchall():
        match = _VLLM_EXEC_RE.match(text or "")
        if not match or end_ns is None or end_ns <= start_ns:
            continue
        shape = tuple(int(g) for g in match.groups())
        raw_by_tid[tid].append((start_ns, end_ns, shape))

    tids = sorted(raw_by_tid)
    tid_to_rank = {tid: rank for rank, tid in enumerate(tids)}
    out: list[IterRange] = []
    for tid in tids:
        rows = sorted(raw_by_tid[tid], key=lambda row: row[0])
        for index, (start_ns, end_ns, shape) in enumerate(rows):
            ctx_reqs, ctx_tokens, gen_reqs, gen_tokens = shape
            out.append(
                IterRange(
                    rank=tid_to_rank[tid],
                    index=index,
                    tid=tid,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    num_context_requests=ctx_reqs,
                    num_context_tokens=ctx_tokens,
                    num_decode_requests=gen_reqs,
                    num_decode_tokens=gen_tokens,
                )
            )
    return out


def _kernel_rows(cur: sqlite3.Cursor) -> Iterable[tuple[int, int, int, int]]:
    """Yield (runtime_tid, runtime_start_ns, kernel_start_ns, kernel_end_ns)."""
    _require_table(cur, "CUPTI_ACTIVITY_KIND_RUNTIME")
    _require_table(cur, "CUPTI_ACTIVITY_KIND_KERNEL")
    cur.execute(
        """
        SELECT R.globalTid, R.start, K.start, K.end
        FROM CUPTI_ACTIVITY_KIND_KERNEL K
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME R
          ON K.correlationId = R.correlationId
         AND K.globalPid = (R.globalTid & ?)
        """,
        (_GLOBAL_PID_MASK,),
    )
    yield from cur.fetchall()


def _find_range(ranges: list[IterRange], host_time_ns: int) -> IterRange | None:
    # Per-tid list is sorted and usually small for profiling captures.
    for rng in ranges:
        if rng.start_ns <= host_time_ns < rng.end_ns:
            return rng
        if rng.start_ns > host_time_ns:
            break
    return None


def _kernel_stats_by_range(
    cur: sqlite3.Cursor, ranges: list[IterRange]
) -> dict[tuple[int, int], KernelStats]:
    ranges_by_tid: dict[int, list[IterRange]] = defaultdict(list)
    for rng in ranges:
        ranges_by_tid[rng.tid].append(rng)
    for tid in ranges_by_tid:
        ranges_by_tid[tid].sort(key=lambda rng: rng.start_ns)

    stats: dict[tuple[int, int], KernelStats] = defaultdict(KernelStats)
    seen: set[tuple[int, int, int, int]] = set()
    for tid, runtime_start_ns, kernel_start_ns, kernel_end_ns in _kernel_rows(cur):
        rng = _find_range(ranges_by_tid.get(tid, []), runtime_start_ns)
        if rng is None:
            continue
        # Defensive dedup for repeated sqlite rows in some graph traces.
        key = (tid, runtime_start_ns, kernel_start_ns, kernel_end_ns)
        if key in seen:
            continue
        seen.add(key)
        stats[(rng.rank, rng.index)].add(kernel_start_ns, kernel_end_ns)
    return stats


def _rank_rows(ranges: list[IterRange], stats: dict[tuple[int, int], KernelStats]):
    for rng in sorted(ranges, key=lambda r: (r.index, r.rank)):
        ks = stats.get((rng.rank, rng.index), KernelStats())
        yield {
            "iteration": rng.index,
            "rank": rng.rank,
            "num_context_requests": rng.num_context_requests,
            "num_context_tokens": rng.num_context_tokens,
            "num_decode_requests": rng.num_decode_requests,
            "num_decode_tokens": rng.num_decode_tokens,
            "wall_us": (rng.end_ns - rng.start_ns) / 1000.0,
            "gpu_span_us": ks.span_ns / 1000.0,
            "gpu_sum_us": ks.sum_ns / 1000.0,
            "kernel_count": ks.count,
            "start_ns": rng.start_ns,
            "end_ns": rng.end_ns,
        }


def _reduced_rows(rows: list[dict[str, float | int | str]], rank_reduce: str):
    grouped: dict[int, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["iteration"])].append(row)

    for iteration in sorted(grouped):
        group = grouped[iteration]
        base = group[0]
        if rank_reduce == "sum":
            gpu_span_us = sum(float(r["gpu_span_us"]) for r in group)
            gpu_sum_us = sum(float(r["gpu_sum_us"]) for r in group)
            wall_us = sum(float(r["wall_us"]) for r in group)
        else:
            gpu_span_us = max(float(r["gpu_span_us"]) for r in group)
            gpu_sum_us = max(float(r["gpu_sum_us"]) for r in group)
            wall_us = max(float(r["wall_us"]) for r in group)
        yield {
            "iteration": iteration,
            "rank": "all",
            "num_context_requests": base["num_context_requests"],
            "num_context_tokens": base["num_context_tokens"],
            "num_decode_requests": base["num_decode_requests"],
            "num_decode_tokens": base["num_decode_tokens"],
            "wall_us": wall_us,
            "gpu_span_us": gpu_span_us,
            "gpu_sum_us": gpu_sum_us,
            "kernel_count": sum(int(r["kernel_count"]) for r in group),
            "start_ns": min(int(r["start_ns"]) for r in group),
            "end_ns": max(int(r["end_ns"]) for r in group),
        }


def parse(sqlite_path: str, *, per_rank: bool = False, rank_reduce: str = "max"):
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    cur = con.cursor()
    ranges = _iter_ranges(cur)
    if not ranges:
        raise RuntimeError(
            "no vLLM execute_context_* NVTX ranges found. Start vLLM with "
            "`--profiler-config.profiler cuda`, run it under `nsys profile "
            "--trace=cuda,nvtx ...`, and bracket requests with /start_profile "
            "and /stop_profile."
        )
    stats = _kernel_stats_by_range(cur, ranges)
    con.close()

    rows = list(_rank_rows(ranges, stats))
    if not per_rank:
        rows = list(_reduced_rows(rows, rank_reduce))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("sqlite_path")
    ap.add_argument("--output", "-o", help="CSV output path. Defaults to stdout.")
    ap.add_argument("--per-rank", action="store_true", help="Emit one row per rank.")
    ap.add_argument(
        "--rank-reduce",
        choices=("max", "sum"),
        default="max",
        help="How to reduce TP ranks when --per-rank is not set. Default: max.",
    )
    args = ap.parse_args()

    if not os.path.exists(args.sqlite_path):
        raise FileNotFoundError(args.sqlite_path)

    rows = parse(args.sqlite_path, per_rank=args.per_rank, rank_reduce=args.rank_reduce)
    fieldnames = [
        "iteration",
        "rank",
        "num_context_requests",
        "num_context_tokens",
        "num_decode_requests",
        "num_decode_tokens",
        "wall_us",
        "gpu_span_us",
        "gpu_sum_us",
        "kernel_count",
        "start_ns",
        "end_ns",
    ]
    if args.output:
        with open(args.output, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

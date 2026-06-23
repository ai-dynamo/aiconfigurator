#!/usr/bin/env python3
"""Analyze communication/compute overlap in an Nsight Systems sqlite export.

The preferred input is a trace with `bench_step::*` NVTX ranges from
`vllm_step_marker.py`; use `--whole-trace` for a coarse trace-wide summary when
those markers are unavailable.
"""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

_COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
sys.path.insert(0, str(_COMMON_DIR))

from parse_nsys_step_sweep import (
    _DEFAULT_KERNEL_DROP,
    _GLOBAL_PID_MASK,
    _build_nvtx_lookups,
    _query_kernels,
    _step_of_with_starts,
)


_WHOLE_TRACE_STEP = (-1, 0, 0, 0)
_DEFAULT_COMM_RE = re.compile(
    _DEFAULT_KERNEL_DROP.pattern + r"|all[_-]?reduce|allreduce|nccl",
    re.IGNORECASE,
)


def _merge_intervals_ns(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted((int(s), int(e)) for s, e in intervals if e > s):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
    return merged


def _union_ns(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(end - start for start, end in _merge_intervals_ns(intervals))


def _overlap_ns(left: Iterable[tuple[int, int]], right: Iterable[tuple[int, int]]) -> int:
    left_merged = _merge_intervals_ns(left)
    right_merged = _merge_intervals_ns(right)
    i = j = 0
    total = 0
    while i < len(left_merged) and j < len(right_merged):
        ls, le = left_merged[i]
        rs, re = right_merged[j]
        total += max(0, min(le, re) - max(ls, rs))
        if le <= re:
            i += 1
        else:
            j += 1
    return total


def _span_ns(intervals: Iterable[tuple[int, int]]) -> int:
    values = [(s, e) for s, e in intervals if e > s]
    if not values:
        return 0
    return max(e for _, e in values) - min(s for s, _ in values)


def _format_us(ns: int | float) -> float:
    return float(ns) / 1000.0


def _build_step_indexes(step_wins_by_tid: dict[int, list[tuple[int, int, tuple[int, int, int, int]]]]):
    step_wins_by_pid: dict[int, list[tuple[int, int, tuple[int, int, int, int]]]] = defaultdict(list)
    all_step_wins: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for tid, wins in step_wins_by_tid.items():
        step_wins_by_pid[tid & _GLOBAL_PID_MASK].extend(wins)
        all_step_wins.extend(wins)
    for wins in step_wins_by_pid.values():
        wins.sort()
    all_step_wins.sort()
    return (
        step_wins_by_pid,
        all_step_wins,
        {tid: [s for s, _, _ in wins] for tid, wins in step_wins_by_tid.items()},
        {pid: [s for s, _, _ in wins] for pid, wins in step_wins_by_pid.items()},
        [s for s, _, _ in all_step_wins],
    )


def _find_step(
    tid: int,
    runtime_start: int,
    *,
    step_wins_by_tid: dict[int, list[tuple[int, int, tuple[int, int, int, int]]]],
    step_wins_by_pid: dict[int, list[tuple[int, int, tuple[int, int, int, int]]]],
    all_step_wins: list[tuple[int, int, tuple[int, int, int, int]]],
    step_starts_by_tid: dict[int, list[int]],
    step_starts_by_pid: dict[int, list[int]],
    all_step_starts: list[int],
) -> tuple[int, int, int, int] | None:
    step = _step_of_with_starts(
        step_wins_by_tid.get(tid, []),
        step_starts_by_tid.get(tid, []),
        runtime_start,
    )
    if step is not None:
        return step
    pid = tid & _GLOBAL_PID_MASK
    step = _step_of_with_starts(
        step_wins_by_pid.get(pid, []),
        step_starts_by_pid.get(pid, []),
        runtime_start,
    )
    if step is not None:
        return step
    return _step_of_with_starts(all_step_wins, all_step_starts, runtime_start)


def analyze_sqlite(
    sqlite_path: str | Path,
    *,
    comm_re: re.Pattern[str] = _DEFAULT_COMM_RE,
    batch_size: int | None = None,
    past_kv: int | None = None,
    per_pid: bool = False,
    whole_trace: bool = False,
) -> tuple[list[dict], dict]:
    path = Path(sqlite_path)
    if not path.exists():
        raise FileNotFoundError(path)

    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    cur = con.cursor()
    step_wins_by_tid: dict[int, list[tuple[int, int, tuple[int, int, int, int]]]] = defaultdict(list)
    if not whole_trace:
        step_wins_by_tid, _module_intervals = _build_nvtx_lookups(cur)
        if not step_wins_by_tid:
            raise RuntimeError("no bench_step NVTX ranges found; retry with --whole-trace for a coarse summary")
    (
        step_wins_by_pid,
        all_step_wins,
        step_starts_by_tid,
        step_starts_by_pid,
        all_step_starts,
    ) = _build_step_indexes(step_wins_by_tid)

    cur.execute("SELECT id, value FROM StringIds")
    string_ids = dict(cur.fetchall())

    groups: dict[tuple[tuple[int, int, int, int], int | None], dict[str, list]] = defaultdict(
        lambda: {"compute": [], "comm": []}
    )
    name_totals: dict[tuple[tuple[int, int, int, int], int | None], Counter[str]] = defaultdict(Counter)
    seen = set()
    outside_step = 0
    for row in _query_kernels(cur):
        cid, graph_node_id, kernel_start, kernel_end, short_name_id, tid, runtime_start, _cap_s, _cap_e = row
        key = (tid, cid, graph_node_id)
        if key in seen:
            continue
        seen.add(key)
        if whole_trace:
            step = _WHOLE_TRACE_STEP
        else:
            step = _find_step(
                tid,
                runtime_start,
                step_wins_by_tid=step_wins_by_tid,
                step_wins_by_pid=step_wins_by_pid,
                all_step_wins=all_step_wins,
                step_starts_by_tid=step_starts_by_tid,
                step_starts_by_pid=step_starts_by_pid,
                all_step_starts=all_step_starts,
            )
            if step is None:
                outside_step += 1
                continue
        step_n, bs, past, _run = step
        if batch_size is not None and bs != batch_size:
            continue
        if past_kv is not None and past != past_kv:
            continue
        name = string_ids.get(short_name_id, str(short_name_id))
        kind = "comm" if comm_re.search(name) else "compute"
        group_key = (step, (tid & _GLOBAL_PID_MASK) if per_pid else None)
        groups[group_key][kind].append((int(kernel_start), int(kernel_end), name))
        name_totals[group_key][name] += int(kernel_end) - int(kernel_start)
    con.close()

    rows = []
    for (step, pid), kernels_by_kind in groups.items():
        compute = kernels_by_kind["compute"]
        comm = kernels_by_kind["comm"]
        compute_intervals = [(s, e) for s, e, _ in compute]
        comm_intervals = [(s, e) for s, e, _ in comm]
        all_intervals = compute_intervals + comm_intervals
        compute_union = _union_ns(compute_intervals)
        comm_union = _union_ns(comm_intervals)
        total_union = _union_ns(all_intervals)
        overlap = _overlap_ns(compute_intervals, comm_intervals)
        comm_visible = max(0, total_union - compute_union)
        step_n, bs, past, run = step
        top_comm = Counter({name: ns for name, ns in name_totals[(step, pid)].items() if comm_re.search(name)})
        rows.append(
            {
                "step": step_n,
                "batch_size": bs,
                "past_kv": past,
                "measure_run": run,
                "pid": "" if pid is None else pid,
                "kernel_count": len(compute) + len(comm),
                "compute_kernels": len(compute),
                "comm_kernels": len(comm),
                "compute_gpu_us": _format_us(sum(e - s for s, e in compute_intervals)),
                "comm_gpu_us": _format_us(sum(e - s for s, e in comm_intervals)),
                "compute_union_us": _format_us(compute_union),
                "comm_union_us": _format_us(comm_union),
                "total_union_us": _format_us(total_union),
                "total_span_us": _format_us(_span_ns(all_intervals)),
                "comm_compute_overlap_us": _format_us(overlap),
                "comm_visible_us": _format_us(comm_visible),
                "comm_overlap_pct": (100.0 * overlap / comm_union) if comm_union else 0.0,
                "comm_visible_pct": (100.0 * comm_visible / comm_union) if comm_union else 0.0,
                "top_comm_kernels": ";".join(
                    f"{name}:{_format_us(ns):.3f}us" for name, ns in top_comm.most_common(5)
                ),
            }
        )
    rows.sort(key=lambda r: (r["step"], r["measure_run"], r["batch_size"], r["past_kv"], str(r["pid"])))
    return rows, {
        "sqlite": str(path),
        "groups": len(rows),
        "outside_step": outside_step,
        "deduped_kernels": len(seen),
        "whole_trace": whole_trace,
    }


def _print_table(rows: list[dict], metadata: dict) -> None:
    print(
        f"[overlap] groups={metadata['groups']} kernels={metadata['deduped_kernels']} "
        f"outside_step={metadata['outside_step']} whole_trace={metadata['whole_trace']}",
        file=sys.stderr,
    )
    if not rows:
        print("No matching kernel groups.", file=sys.stderr)
        return
    print(
        f"{'step':>8} {'bs':>5} {'past':>8} {'run':>4} {'pid':>12} "
        f"{'comp_us':>10} {'comm_us':>10} {'total_us':>10} "
        f"{'overlap_us':>11} {'visible_us':>11} {'visible%':>9} {'comm_k':>7}"
    )
    for row in rows:
        print(
            f"{row['step']:8d} {row['batch_size']:5d} {row['past_kv']:8d} {row['measure_run']:4d} "
            f"{str(row['pid']):>12} "
            f"{row['compute_union_us']:10.3f} {row['comm_union_us']:10.3f} "
            f"{row['total_union_us']:10.3f} {row['comm_compute_overlap_us']:11.3f} "
            f"{row['comm_visible_us']:11.3f} {row['comm_visible_pct']:8.1f}% "
            f"{row['comm_kernels']:7d}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite_path")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--past-kv", type=int)
    parser.add_argument("--per-pid", action="store_true")
    parser.add_argument("--whole-trace", action="store_true")
    parser.add_argument("--comm-regex", default=_DEFAULT_COMM_RE.pattern)
    parser.add_argument("--format", choices=("table", "csv"), default="table")
    args = parser.parse_args(argv)

    rows, metadata = analyze_sqlite(
        args.sqlite_path,
        comm_re=re.compile(args.comm_regex, re.IGNORECASE),
        batch_size=args.batch_size,
        past_kv=args.past_kv,
        per_pid=args.per_pid,
        whole_trace=args.whole_trace,
    )
    if args.format == "csv":
        fieldnames = list(rows[0].keys()) if rows else [
            "step",
            "batch_size",
            "past_kv",
            "measure_run",
            "pid",
            "kernel_count",
            "compute_kernels",
            "comm_kernels",
            "compute_gpu_us",
            "comm_gpu_us",
            "compute_union_us",
            "comm_union_us",
            "total_union_us",
            "total_span_us",
            "comm_compute_overlap_us",
            "comm_visible_us",
            "comm_overlap_pct",
            "comm_visible_pct",
            "top_comm_kernels",
        ]
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    else:
        _print_table(rows, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

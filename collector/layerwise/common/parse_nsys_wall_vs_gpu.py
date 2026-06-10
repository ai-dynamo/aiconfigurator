"""Per-target-iteration step: wall time (host NVTX) vs GPU kernel sum.

Wall time = `bench_step::*` NVTX range `end - start` (host timestamps) —
time from entering `GPUModelRunner.execute_model` to returning.
GPU sum = CUPTI kernel durations attributed to that step (via graphNodeId
in graph mode, or correlationId in eager mode).

Gap = wall - gpu_sum ≈ CPU overhead (Python dispatch, input prep, graph
launch, output sample) + host wait on kernels reaching GPU.

Under cuda graph mode the gap is tiny (~10-20 μs for `cudaGraphLaunch`).
Under eager mode the gap is the per-kernel launch overhead summed
(~5-10 μs x N_kernels), often dominant at small batch sizes.

Requires `vllm_step_marker` NVTX in the sqlite.
"""
import argparse
import re
import sqlite3
import sys
from collections import defaultdict

_BENCH_STEP_RE = re.compile(r"bench_step::N(\d+)::bs(\d+)::past(\d+)")
_NODE_MASK = 0xFFFFFFFF
_GLOBAL_PID_MASK = -16777216  # nsys globalTid with low 24 bits cleared


def _bench_step_ranges(cur):
    cur.execute(
        "SELECT text, start, end, globalTid FROM NVTX_EVENTS "
        "WHERE text LIKE 'bench_step::%'"
    )
    out = []
    for text, s, e, tid in cur.fetchall():
        m = _BENCH_STEP_RE.search(text or "")
        if m:
            out.append((int(m.group(1)), int(m.group(2)), int(m.group(3)),
                        s, e, tid))
    out.sort()
    return out


def _sum_gpu_graph(cur, step_ranges):
    """For graph mode: attribute kernels to (tid, step) via correlationId.

    Multi-rank sqlites (TP=N) have one bench_step NVTX per rank; kernel
    attribution must be keyed by (globalTid, step) to avoid cross-rank sum.
    """
    cur.execute(
        "SELECT correlationId, start, globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME"
    )
    rt_rows = cur.fetchall()

    step_ranges_by_tid = defaultdict(list)
    for n, bs, past, s, e, tid in step_ranges:
        step_ranges_by_tid[tid].append((s, e, n))
    for tid in step_ranges_by_tid:
        step_ranges_by_tid[tid].sort()

    corr_to_key = {}
    for cid, rs, tid in rt_rows:
        ranges = step_ranges_by_tid.get(tid, [])
        for s, e, n in ranges:
            if s <= rs < e:
                corr_to_key[(tid & _GLOBAL_PID_MASK, cid)] = (tid, n)
                break
            if s > rs:
                break

    gpu_ns = defaultdict(int)
    n_k = defaultdict(int)
    cur.execute(
        "SELECT correlationId, globalPid, start, end FROM CUPTI_ACTIVITY_KIND_KERNEL"
    )
    for cid, gpid, ks, ke in cur.fetchall():
        key = corr_to_key.get((gpid, cid))
        if key is None:
            continue
        gpu_ns[key] += ke - ks
        n_k[key] += 1
    return gpu_ns, n_k


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("sqlite_path")
    args = ap.parse_args()

    con = sqlite3.connect(args.sqlite_path)
    cur = con.cursor()

    steps = _bench_step_ranges(cur)
    if not steps:
        print("no bench_step::* NVTX ranges found — vllm_step_marker missing?",
              file=sys.stderr)
        sys.exit(1)

    gpu_ns, n_k = _sum_gpu_graph(cur, steps)
    con.close()

    tids = sorted({tid for _, _, _, _, _, tid in steps})
    tid_to_rank = {tid: i for i, tid in enumerate(tids)}
    multi_rank = len(tids) > 1

    header_cols = (f"{'step':>8} {'bs':>5} {'past':>7} "
                   f"{'wall_us':>10} {'gpu_us':>10} {'cpu_us':>10} "
                   f"{'cpu_frac':>9} {'n_kern':>7}")
    if multi_rank:
        header_cols = f"{'rank':>5} " + header_cols
    print(header_cols)
    print("-" * (85 if multi_rank else 78))

    # Sort by (step, rank) for readability.
    steps_sorted = sorted(steps, key=lambda r: (r[0], tid_to_rank[r[5]]))
    for n, bs, past, s, e, tid in steps_sorted:
        wall_us = (e - s) / 1000.0
        gpu_us = gpu_ns.get((tid, n), 0) / 1000.0
        cpu_us = wall_us - gpu_us
        frac = cpu_us / wall_us if wall_us > 0 else 0
        row = (f"{n:>8d} {bs:>5d} {past:>7d} "
               f"{wall_us:>10.2f} {gpu_us:>10.2f} {cpu_us:>10.2f} "
               f"{frac:>8.1%} {n_k.get((tid, n), 0):>7d}")
        if multi_rank:
            row = f"{tid_to_rank[tid]:>5d} " + row
        print(row)


if __name__ == "__main__":
    main()

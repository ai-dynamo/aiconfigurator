"""Per-step-iteration kernel attribution from an nsys sqlite.

Designed for runs produced by `vllm_step_marker.py`:
  outer NVTX range:  `bench_step::N<nnnnnnn>::bs<B>::past<pppppp>`
  inner NVTX ranges: `{'Module': '<dotted.module.path>'}` (PytHooks layerwise)

Output: one row per target iteration x Module (rollup-regex optional), showing
per-step kernel GPU time and kernel count.

Attribution approach (matches TensorRT-LLM's layer_wise_benchmarks/parse.py):
  For cuda-graph replay kernels, traditional correlationId→host-time won't
  hit any Module NVTX because host is inside a single `cudaGraphLaunch`.
  We use a two-step SQL JOIN through `CUDA_GRAPH_NODE_EVENTS` to recover
  the *stream-capture-time* timestamp of each kernel's original node:

    replay kernel.graphNodeId
      → CGE1.graphNodeId          (matches instantiate-time row)
      → CGE1.originalGraphNodeId  (stable template id)
      → CGE2.graphNodeId          (matches stream-capture-time row)
      → CGE2.start/end            (host time when NVTX stack was open)

  The JOIN is constrained on globalTid too so a merged multi-rank sqlite
  doesn't produce 8x cardinality.

  Eager kernels (graphNodeId IS NULL) take `capture_start = R.start`
  (runtime host time, which already falls inside Module NVTX).

Step attribution uses `R.start` (host time of cudaLaunchKernel/cudaGraphLaunch)
against the `bench_step::*` NVTX windows (outer marker is host-level).

Usage:
  python parse_nsys_step_sweep.py <sqlite> \\
      --rollup '(self_attn|mlp|input_layernorm|post_attention_layernorm)' \\
      --layer 3
"""

import argparse
import bisect
import os
import re
import sqlite3
import sys
from collections import defaultdict

_BENCH_STEP_RE = re.compile(r"bench_step::N(\d+)::bs(\d+)::past(\d+)(?:::run(\d+))?")
_MODULE_RE = re.compile(r"'Module':\s*'([^']+)'")
_GLOBAL_PID_MASK = -16777216  # nsys globalTid with low 24 bits cleared

_DEFAULT_KERNEL_DROP = re.compile(
    r"(ncclDevKernel|ncclKernel|all2all|deep_ep|ep_fuse|nvshmem|"
    r"multimem_all_reduce|one_shot_all_reduce|two_shot_all_reduce|"
    r"^dispatch$|^combine$)",
    re.IGNORECASE,
)
_FUSABLE_RMS_KERNEL_RE = re.compile(
    r"(rms_norm|add_mean_mul_pow_rsqrt)",
    re.IGNORECASE,
)


def _extract_module(text):
    if text is None:
        return None
    m = _MODULE_RE.search(text)
    return m.group(1) if m else None


def _extract_bench_step(text):
    if text is None:
        return None
    m = _BENCH_STEP_RE.search(text)
    if not m:
        return None
    run = int(m.group(4)) if m.group(4) is not None else 0
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), run


def _query_kernels(cur):
    """Yield (cid, kernel_start, kernel_end, short_name_id, tid,
              runtime_start, capture_start, capture_end).

    Uses TRT-LLM's 2-step JOIN via `originalGraphNodeId` for graph kernels.
    Constrains JOINs on globalTid so multi-rank sqlites stay 1:1 at the
    CGE1→CGE2 level.

    vLLM captures/instantiates the same graph template many times during
    warmup, so one replay kernel's graphNodeId can match hundreds of CGE1
    rows (one per instantiate). The JOIN result thus has duplicates per
    kernel. Callers must dedup by correlationId.
    """
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='CUDA_GRAPH_NODE_EVENTS'")
    has_graph = cur.fetchone() is not None

    query = """
    SELECT K.correlationId, K.graphNodeId, K.start, K.end, K.shortName,
           R.globalTid, R.start AS runtime_start,
           R.start AS capture_start, R.end AS capture_end
    FROM CUPTI_ACTIVITY_KIND_KERNEL K
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME R
      ON K.correlationId = R.correlationId
     AND K.globalPid = (R.globalTid & ?)
    WHERE K.graphNodeId IS NULL
    """
    if has_graph:
        query += """
        UNION ALL
        SELECT K.correlationId, K.graphNodeId, K.start, K.end, K.shortName,
               R.globalTid, R.start AS runtime_start,
               CGE2.start AS capture_start, CGE2.end AS capture_end
        FROM CUPTI_ACTIVITY_KIND_KERNEL K
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME R
          ON K.correlationId = R.correlationId
         AND K.globalPid = (R.globalTid & ?)
        LEFT JOIN CUDA_GRAPH_NODE_EVENTS CGE1
          ON K.graphNodeId = CGE1.graphNodeId
         AND R.globalTid = CGE1.globalTid
         AND CGE1.originalGraphNodeId IS NOT NULL
        LEFT JOIN CUDA_GRAPH_NODE_EVENTS CGE2
          ON CGE1.originalGraphNodeId = CGE2.graphNodeId
         AND CGE1.globalTid = CGE2.globalTid
        WHERE K.graphNodeId IS NOT NULL
        """
    params = (_GLOBAL_PID_MASK,) * (2 if has_graph else 1)
    cur.execute(query, params)
    return cur.fetchall()


def _has_table(cur, table):
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    )
    return cur.fetchone() is not None


def _build_nvtx_lookups(cur):
    """Return (step_wins_by_tid, module_intervals_by_tid).

    step_wins_by_tid[tid] = sorted list of (start, end, (n, bs, past))
    module_intervals_by_tid[tid] = sorted list of (start, end, module_name)
    """
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='NVTX_EVENTS'")
    if cur.fetchone() is None:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = ", ".join(row[0] for row in cur.fetchall())
        raise RuntimeError(
            "nsys sqlite has no NVTX_EVENTS table. Re-profile with "
            "`nsys profile --trace=cuda,nvtx --cuda-graph-trace=node ...` "
            "and make sure the vLLM layerwise plugin/patches are loaded. "
            f"Tables present: {tables or '<none>'}"
        )
    cur.execute(
        "SELECT text, start, end, globalTid FROM NVTX_EVENTS "
        "WHERE text IS NOT NULL AND (text LIKE 'bench_step::%' OR text LIKE '%Module%')"
    )
    step_wins = defaultdict(list)
    mod_ivs = defaultdict(list)
    for text, s, e, tid in cur.fetchall():
        step = _extract_bench_step(text)
        if step is not None:
            step_wins[tid].append((s, e, step))
            continue
        mod = _extract_module(text)
        if mod:
            mod_ivs[tid].append((s, e, mod))
    for tid in step_wins:
        step_wins[tid].sort()
    for tid in mod_ivs:
        mod_ivs[tid].sort()
    return step_wins, mod_ivs


def _count_nvtx_ranges(cur):
    cur.execute(
        "SELECT "
        "SUM(CASE WHEN text LIKE 'bench_step::%' THEN 1 ELSE 0 END), "
        "SUM(CASE WHEN text LIKE '%Module%' THEN 1 ELSE 0 END) "
        "FROM NVTX_EVENTS WHERE text IS NOT NULL"
    )
    step_count, module_count = cur.fetchone()
    return int(step_count or 0), int(module_count or 0)


def _step_of_with_starts(step_wins_for_tid, starts, host_time):
    """Binary search host_time against precomputed step-window starts."""
    if not step_wins_for_tid:
        return None
    idx = bisect.bisect_right(starts, host_time) - 1
    if idx < 0:
        return None
    s, e, step = step_wins_for_tid[idx]
    if s <= host_time < e:
        return step
    return None


def _sum_kernels(cur, kernel_drop_re):
    step_wins, mod_ivs = _build_nvtx_lookups(cur)
    step_wins_by_pid = defaultdict(list)
    all_step_wins = []
    for tid, wins in step_wins.items():
        step_wins_by_pid[tid & _GLOBAL_PID_MASK].extend(wins)
        all_step_wins.extend(wins)
    for pid in step_wins_by_pid:
        step_wins_by_pid[pid].sort()
    all_step_wins.sort()
    step_starts = {tid: [s for s, _, _ in wins] for tid, wins in step_wins.items()}
    mod_starts = {tid: [s for s, _, _ in ivs] for tid, ivs in mod_ivs.items()}
    step_starts_by_pid = {pid: [s for s, _, _ in wins] for pid, wins in step_wins_by_pid.items()}
    all_step_starts = [s for s, _, _ in all_step_wins]

    cur.execute("SELECT id, value FROM StringIds")
    sid = dict(cur.fetchall())

    # A single `cudaGraphLaunch` call fires every kernel in the graph with
    # the same host correlationId but distinct `graphNodeId` per kernel.
    # In multi-process nsys traces, correlationId is only unique within a
    # process, so include globalTid in the dedup key as well.
    # (tid, cid, gnid) → (ks, ke, short_id, tid, rt_start, step, dropped)
    kern_best_mod = {}  # (tid, cid, gnid) → (best_mod_start, best_mod_name)
    kern_meta = {}
    step_pid_fallback = 0
    step_global_fallback = 0
    for row in _query_kernels(cur):
        cid, gnid, ks, ke, short_id, tid, rt_start, cap_s, cap_e = row
        key = (tid, cid, gnid)
        if key not in kern_meta:
            name = sid.get(short_id, "")
            dropped = bool(kernel_drop_re.search(name))
            step = None
            if not dropped:
                step = _step_of_with_starts(step_wins.get(tid, []), step_starts.get(tid, []), rt_start)
                if step is None:
                    step = _step_of_with_starts(
                        step_wins_by_pid.get(tid & _GLOBAL_PID_MASK, []),
                        step_starts_by_pid.get(tid & _GLOBAL_PID_MASK, []),
                        rt_start,
                    )
                    if step is not None:
                        step_pid_fallback += 1
                if step is None:
                    step = _step_of_with_starts(all_step_wins, all_step_starts, rt_start)
                    if step is not None:
                        step_global_fallback += 1
            kern_meta[key] = (ks, ke, short_id, tid, rt_start, step, dropped)
        _, _, _, _, _, step, dropped = kern_meta[key]
        if dropped or step is None:
            continue
        if cap_s is None:
            continue
        if key in kern_best_mod:
            continue  # already attributed via another instantiate's candidate
        mod_start_name = _innermost_module_at_with_start(mod_ivs.get(tid, []), mod_starts.get(tid, []), cap_s, cap_e)
        if mod_start_name is None:
            continue
        kern_best_mod[key] = mod_start_name

    gpu_ns = defaultdict(int)
    rms_ns = defaultdict(int)
    span_start_ns = {}
    span_end_ns = {}
    n_k = defaultdict(int)
    n_rms_k = defaultdict(int)
    unmatched_step = 0
    unmatched_module = 0
    dropped_comm = 0

    for key, (ks, ke, short_id, tid, rt_start, step, dropped) in kern_meta.items():
        if dropped:
            dropped_comm += 1
            continue
        if step is None:
            unmatched_step += 1
            continue
        mod_entry = kern_best_mod.get(key)
        if mod_entry is None:
            unmatched_module += 1
            continue
        _, mod = mod_entry
        out_key = (step, mod, tid)
        gpu_ns[out_key] += ke - ks
        name = sid.get(short_id, "")
        if _FUSABLE_RMS_KERNEL_RE.search(name):
            rms_ns[out_key] += ke - ks
            n_rms_k[out_key] += 1
        span_start_ns[out_key] = min(span_start_ns.get(out_key, ks), ks)
        span_end_ns[out_key] = max(span_end_ns.get(out_key, ke), ke)
        n_k[out_key] += 1

    # Deployment-parity traces use CUDAGraphWrapper as the measured unit. Some
    # TP chunked-prefill windows have valid step ranges but no kernel with a
    # recoverable wrapper module. Keep those shapes usable by falling back to
    # the full step NVTX span for only the missing wrapper rows.
    wrapper_steps = {(step_meta, tid) for (step_meta, mod, tid) in gpu_ns if mod == "CUDAGraphWrapper"}
    for tid, wins in step_wins.items():
        for step_s, step_e, step in wins:
            if (step, tid) in wrapper_steps:
                continue
            out_key = (step, "CUDAGraphWrapper", tid)
            gpu_ns[out_key] += step_e - step_s
            span_start_ns[out_key] = min(span_start_ns.get(out_key, step_s), step_s)
            span_end_ns[out_key] = max(span_end_ns.get(out_key, step_e), step_e)
            n_k[out_key] += 1

    return (
        gpu_ns,
        rms_ns,
        span_start_ns,
        span_end_ns,
        n_k,
        n_rms_k,
        (
            unmatched_step,
            unmatched_module,
            dropped_comm,
            step_pid_fallback,
            step_global_fallback,
        ),
    )


def _sum_nvtx_ranges(cur):
    """Fallback attribution from NVTX span duration when CUPTI rows are absent."""
    step_wins, mod_ivs = _build_nvtx_lookups(cur)

    gpu_ns = defaultdict(int)
    rms_ns = defaultdict(int)
    span_start_ns = {}
    span_end_ns = {}
    n_k = defaultdict(int)
    n_rms_k = defaultdict(int)

    for tid, wins in step_wins.items():
        modules = mod_ivs.get(tid, [])
        module_starts = [s for s, _, _ in modules]
        for step_s, step_e, step in wins:
            emitted = False
            hi = bisect.bisect_right(module_starts, step_e)
            for i in range(hi):
                mod_s, mod_e, mod = modules[i]
                if mod_s < step_s or mod_e is None or mod_e > step_e:
                    continue
                out_key = (step, mod, tid)
                gpu_ns[out_key] += mod_e - mod_s
                span_start_ns[out_key] = min(span_start_ns.get(out_key, mod_s), mod_s)
                span_end_ns[out_key] = max(span_end_ns.get(out_key, mod_e), mod_e)
                n_k[out_key] += 1
                if mod == "CUDAGraphWrapper":
                    emitted = True
            if not emitted:
                # Some decode CUDA graph replays only expose child/output
                # module ranges. Keep deployment-parity rows usable by
                # attributing the full measured step to the wrapper.
                out_key = (step, "CUDAGraphWrapper", tid)
                gpu_ns[out_key] += step_e - step_s
                span_start_ns[out_key] = min(span_start_ns.get(out_key, step_s), step_s)
                span_end_ns[out_key] = max(span_end_ns.get(out_key, step_e), step_e)
                n_k[out_key] += 1

    return gpu_ns, rms_ns, span_start_ns, span_end_ns, n_k, n_rms_k


def _innermost_module_at_with_start(mod_ivs_for_tid, starts, capture_start, capture_end):
    """Return the innermost Module NVTX range enclosing the capture interval."""
    hi = bisect.bisect_right(starts, capture_start)
    for i in range(hi - 1, -1, -1):
        s, e, name = mod_ivs_for_tid[i]
        if e < capture_end:
            continue
        return (s, name)
    return None


def _rollup_rows(
    gpu_ns,
    rms_ns,
    span_start_ns,
    span_end_ns,
    n_k,
    n_rms_k,
    rollup,
    layer,
    per_rank,
    rank_reduce,
):
    """Convert raw (step, module, tid) stats into structured rollup rows."""
    tids = sorted({tid for (_, _, tid) in gpu_ns})
    tid_to_rank = {tid: i for i, tid in enumerate(tids)}

    rollup_re = re.compile(rollup)
    per_rank_ns = defaultdict(int)
    per_rank_rms_ns = defaultdict(int)
    per_rank_start_ns = {}
    per_rank_end_ns = {}
    per_rank_k = defaultdict(int)
    per_rank_rms_k = defaultdict(int)
    for (step_meta, mod, tid), ns in gpu_ns.items():
        m = rollup_re.search(mod or "")
        if not m:
            continue
        roll_key = m.groups() if m.groups() else (m.group(0),)
        if layer is not None and roll_key and str(layer) != str(roll_key[0]):
            continue
        rank = tid_to_rank[tid]
        key = (step_meta, roll_key, rank)
        per_rank_ns[key] += ns
        per_rank_rms_ns[key] += rms_ns.get((step_meta, mod, tid), 0)
        raw_key = (step_meta, mod, tid)
        s = span_start_ns[raw_key]
        e = span_end_ns[raw_key]
        per_rank_start_ns[key] = min(per_rank_start_ns.get(key, s), s)
        per_rank_end_ns[key] = max(per_rank_end_ns.get(key, e), e)
        per_rank_k[key] += n_k[(step_meta, mod, tid)]
        per_rank_rms_k[key] += n_rms_k.get((step_meta, mod, tid), 0)

    rolled_ns = defaultdict(int)
    rolled_rms_ns = defaultdict(int)
    rolled_start_ns = {}
    rolled_end_ns = {}
    rolled_k = defaultdict(int)
    rolled_rms_k = defaultdict(int)
    if per_rank:
        rolled_ns.update(per_rank_ns)
        rolled_rms_ns.update(per_rank_rms_ns)
        rolled_start_ns.update(per_rank_start_ns)
        rolled_end_ns.update(per_rank_end_ns)
        rolled_k.update(per_rank_k)
        rolled_rms_k.update(per_rank_rms_k)
    else:
        for (step_meta, roll_key, rank), ns in per_rank_ns.items():
            out_key = (step_meta, roll_key, None)
            s = per_rank_start_ns[(step_meta, roll_key, rank)]
            e = per_rank_end_ns[(step_meta, roll_key, rank)]
            rolled_start_ns[out_key] = min(rolled_start_ns.get(out_key, s), s)
            rolled_end_ns[out_key] = max(rolled_end_ns.get(out_key, e), e)
            if rank_reduce == "sum":
                rolled_ns[out_key] += ns
                rolled_rms_ns[out_key] += per_rank_rms_ns[(step_meta, roll_key, rank)]
                rolled_k[out_key] += per_rank_k[(step_meta, roll_key, rank)]
                rolled_rms_k[out_key] += per_rank_rms_k[(step_meta, roll_key, rank)]
            elif ns > rolled_ns[out_key]:
                rolled_ns[out_key] = ns
                rolled_rms_ns[out_key] = per_rank_rms_ns[(step_meta, roll_key, rank)]
                rolled_k[out_key] = per_rank_k[(step_meta, roll_key, rank)]
                rolled_rms_k[out_key] = per_rank_rms_k[(step_meta, roll_key, rank)]

    rows = []
    for (step_meta, roll_key, rank), ns in rolled_ns.items():
        step_n, bs, past, measure_run = step_meta
        rows.append(
            {
                "step": step_n,
                "batch_size": bs,
                "past_kv": past,
                "measure_run": measure_run,
                "rank": rank,
                "rollup_key": "|".join(map(str, roll_key)),
                "rollup_parts": roll_key,
                "gpu_us": ns / 1000.0,
                "rms_us": rolled_rms_ns.get((step_meta, roll_key, rank), 0) / 1000.0,
                "span_us": (rolled_end_ns[(step_meta, roll_key, rank)] - rolled_start_ns[(step_meta, roll_key, rank)])
                / 1000.0,
                "start_ns": rolled_start_ns[(step_meta, roll_key, rank)],
                "end_ns": rolled_end_ns[(step_meta, roll_key, rank)],
                "kernel_count": rolled_k[(step_meta, roll_key, rank)],
                "rms_kernel_count": rolled_rms_k.get((step_meta, roll_key, rank), 0),
            }
        )
    rows.sort(key=lambda r: (r["step"], r["measure_run"], -1 if r["rank"] is None else r["rank"], r["rollup_key"]))
    return rows, len(tids)


def parse_step_sweep(
    sqlite_path,
    *,
    rollup=r"layers\.(\d+)\.(\w+)",
    layer=None,
    keep_comm=False,
    per_rank=False,
    rank_reduce="sum",
    force_nvtx_span=False,
):
    """Parse an nsys sqlite into structured per-step rollup rows.

    Returns (rows, metadata), where each row has:
      step, batch_size, past_kv, rank, rollup_key, rollup_parts, gpu_us,
      span_us, start_ns, end_ns, kernel_count.
    """
    kernel_drop_re = re.compile("^$") if keep_comm else _DEFAULT_KERNEL_DROP
    if not os.path.exists(sqlite_path):
        raise FileNotFoundError(
            f"nsys sqlite export not found: {sqlite_path}. "
            "Run `nsys export --type sqlite --output <path> <report.nsys-rep>`."
        )
    if os.path.getsize(sqlite_path) == 0:
        raise RuntimeError(
            f"nsys sqlite export is empty: {sqlite_path}. Re-export the report with an explicit `--output` path."
        )

    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    cur = con.cursor()
    nvtx_step_ranges, nvtx_module_ranges = _count_nvtx_ranges(cur)
    if (
        not force_nvtx_span
        and _has_table(cur, "CUPTI_ACTIVITY_KIND_KERNEL")
        and _has_table(cur, "CUPTI_ACTIVITY_KIND_RUNTIME")
    ):
        (
            gpu_ns,
            rms_ns,
            span_start_ns,
            span_end_ns,
            n_k,
            n_rms_k,
            (miss_step, miss_mod, dropped, step_pid_fallback, step_global_fallback),
        ) = _sum_kernels(cur, kernel_drop_re)
        attribution_source = "cupti"
    else:
        gpu_ns, rms_ns, span_start_ns, span_end_ns, n_k, n_rms_k = _sum_nvtx_ranges(cur)
        miss_step = miss_mod = dropped = step_pid_fallback = step_global_fallback = 0
        attribution_source = "nvtx_span"
    con.close()

    rows, n_ranks = _rollup_rows(
        gpu_ns, rms_ns, span_start_ns, span_end_ns, n_k, n_rms_k, rollup, layer, per_rank, rank_reduce
    )
    metadata = {
        "attributed_kernels": sum(n_k.values()),
        "outside_iteration_window": miss_step,
        "no_module_nvtx": miss_mod,
        "dropped_comm": dropped,
        "step_pid_fallback": step_pid_fallback,
        "step_global_fallback": step_global_fallback,
        "n_ranks": n_ranks,
        "rank_reduce": rank_reduce,
        "per_rank": per_rank,
        "nvtx_step_ranges": nvtx_step_ranges,
        "nvtx_module_ranges": nvtx_module_ranges,
        "attribution_source": attribution_source,
    }
    return rows, metadata


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sqlite_path")
    ap.add_argument(
        "--rollup",
        default=r"layers\.(\d+)\.(\w+)",
        help="Regex with capture groups; module rolled up to captures. "
        "Default groups by (layer_idx, first sub-submodule).",
    )
    ap.add_argument("--layer", type=int, default=None, help="Filter rolled-up rows whose first-capture == this layer.")
    ap.add_argument("--keep-comm", action="store_true", help="Include NCCL / all2all / deep_ep kernels in totals.")
    ap.add_argument(
        "--per-rank", action="store_true", help="Emit one row per (step, rank) instead of summing across ranks."
    )
    ap.add_argument(
        "--rank-reduce",
        choices=("sum", "max"),
        default="sum",
        help="How to reduce ranks when --per-rank is not set. 'max' is useful for EP/MoE wall-critical compute.",
    )
    args = ap.parse_args()

    print("[parse] running 2-step JOIN via originalGraphNodeId ...", file=sys.stderr)
    rows, meta = parse_step_sweep(
        args.sqlite_path,
        rollup=args.rollup,
        layer=args.layer,
        keep_comm=args.keep_comm,
        per_rank=args.per_rank,
        rank_reduce=args.rank_reduce,
    )
    print(
        f"[parse] kernels attributed: {meta['attributed_kernels']}, "
        f"outside iteration window: {meta['outside_iteration_window']}, "
        f"no Module NVTX: {meta['no_module_nvtx']}, "
        f"dropped (comm): {meta['dropped_comm']}, "
        f"step pid fallback: {meta['step_pid_fallback']}, "
        f"step global fallback: {meta['step_global_fallback']}, "
        f"bench_step ranges: {meta['nvtx_step_ranges']}, "
        f"Module ranges: {meta['nvtx_module_ranges']}",
        file=sys.stderr,
    )

    n_ranks = meta["n_ranks"]
    if n_ranks > 1:
        if args.per_rank:
            mode = "per-rank"
        elif args.rank_reduce == "max":
            mode = f"MAX across {n_ranks} ranks"
        else:
            mode = f"SUMMED across {n_ranks} ranks (divide for per-rank)"
        print(f"[multi-rank] {n_ranks} ranks detected — output mode: {mode}", file=sys.stderr)

    if not rows:
        print("No rows matched the rollup regex.", file=sys.stderr)
        sys.exit(1)

    step_seen = sorted(
        {(r["step"], r["batch_size"], r["past_kv"], r["measure_run"]) for r in rows},
        key=lambda s: (s[0], s[3]),
    )
    roll_keys = sorted({r["rollup_parts"] for r in rows})
    rank_seen = sorted(
        {r["rank"] for r in rows},
        key=lambda r: -1 if r is None else r,
    )
    by_cell = {
        (
            (r["step"], r["batch_size"], r["past_kv"], r["measure_run"]),
            r["rollup_parts"],
            r["rank"],
        ): r
        for r in rows
    }

    header_keys = ["|".join(map(str, k)) for k in roll_keys]
    col_w = max(14, max(len(h) for h in header_keys) + 1)

    print()
    rank_col = f"{'rank':>5}  " if args.per_rank else ""
    print(
        f"{'step':>8} {'bs':>5} {'past_kv':>8} {'run':>5}  {rank_col}" + "  ".join(f"{h:>{col_w}}" for h in header_keys)
    )
    for step_meta in step_seen:
        for rank in rank_seen:
            step_n, bs, past, measure_run = step_meta
            row_cells = []
            any_nonzero = False
            for k in roll_keys:
                row = by_cell.get((step_meta, k, rank))
                if row and row["gpu_us"] > 0:
                    any_nonzero = True
                    row_cells.append(f"{row['gpu_us']:>{col_w - 4}.1f} μs")
                else:
                    row_cells.append(f"{'-':>{col_w}}")
            if not any_nonzero:
                continue
            rank_prefix = f"{rank:>5}  " if args.per_rank and rank is not None else ""
            print(f"{step_n:>8} {bs:>5} {past:>8} {measure_run:>5}  {rank_prefix}" + "  ".join(row_cells))


if __name__ == "__main__":
    main()

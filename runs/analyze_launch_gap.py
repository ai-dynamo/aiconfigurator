"""Per-rank launch/graph/span/wall decomposition for the small-prefill mechanism
discrimination (1-GPU sharded vs real-tp).

Method (validated on 1-GPU baseline runs/nsys_ctx256, doc SMALL_PREFILL_ENVELOPE):
  Per bench_step NVTX window, per launching process (pid == rank):
    eager  = count of cudaLaunchKernel/cuLaunchKernel(Ex) runtime calls in window
    graph  = count of cudaGraphLaunch runtime calls in window  (= graph pieces)
    nccl   = count of NCCL/all-reduce/all-gather KERNELS with GPU start in window
    busy   = sum of kernel durations (K.start in window)            [GPU-busy]
    span   = last.end - first.start of in-window kernels            [active extent]
    wall   = NVTX window duration (host)                            [execute step]
    gap    = wall - span                                           [launch bubbles]
Cross-check: parser span_us/gpu_us from parse_nsys_step_sweep (module-attributed).

Steady state: report the LAST measured run for each (ntok, rank).
Usage: python runs/analyze_launch_gap.py <sqlite> [--label LBL]
"""
import argparse, re, sqlite3, sys
sys.path.insert(0, ".")
from collector.layerwise.common.parse_nsys_step_sweep import parse_step_sweep

MASK = -16777216
NCCL_RE = re.compile(r"nccl|all.?reduce|all.?gather|reduce.?scatter|all2all|multimem|ncclDevKernel", re.I)

def is_eager(nm): return nm.startswith(("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC"))
def is_graph(nm): return nm.startswith("cudaGraphLaunch")

def analyze(path, label):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True); cur = con.cursor()
    sid = dict(cur.execute("SELECT id,value FROM StringIds").fetchall())
    steps = cur.execute("SELECT text,start,end,globalTid FROM NVTX_EVENTS WHERE text LIKE 'bench_step::%' ORDER BY start").fetchall()
    pids = sorted({tid & MASK for *_, tid in steps})
    pid_rank = {p: i for i, p in enumerate(pids)}

    # parser cross-check (per-rank module-attributed span/busy)
    try:
        prows, _ = parse_step_sweep(path, rollup="CUDAGraphWrapper", per_rank=True)
        pspan = {(r["step"], r["rank"]): (r["span_us"], r["gpu_us"], r["kernel_count"]) for r in prows}
    except Exception as ex:
        pspan = {}; print(f"[warn] parser cross-check failed: {ex}", file=sys.stderr)

    # collect per (ntok, rank) -> list of run measurements (in temporal order)
    per = {}
    run_idx = {}
    for t, s, e, tid in steps:
        pid = tid & MASK; rank = pid_rank[pid]
        m = re.search(r"N(\d+)::bs(\d+)::past(\d+)", t)
        ntok, past = int(m.group(1)), int(m.group(3))
        rt = cur.execute("SELECT R.nameId,COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME R WHERE R.start>=? AND R.start<? AND (R.globalTid&?)=? GROUP BY R.nameId", (s, e, MASK, pid)).fetchall()
        eager = sum(c for n, c in rt if is_eager(sid.get(n, "")))
        graph = sum(c for n, c in rt if is_graph(sid.get(n, "")))
        ks = cur.execute("SELECT K.start,K.end,K.shortName FROM CUPTI_ACTIVITY_KIND_KERNEL K WHERE K.globalPid=? AND K.start>=? AND K.start<? ORDER BY K.start", (pid, s, e)).fetchall()
        busy = sum(b - a for a, b, _ in ks)
        nccl = sum(1 for a, b, nm in ks if NCCL_RE.search(sid.get(nm, "")))
        span = (ks[-1][1] - ks[0][0]) if ks else 0
        wall = e - s
        per.setdefault((ntok, rank), []).append(dict(past=past, eager=eager, graph=graph, nccl=nccl, busy=busy, span=span, wall=wall, nkern=len(ks)))
    con.close()

    print(f"\n===== {label} =====  ranks={len(pids)}")
    print(f"{'ntok':>5} {'rank':>4} {'eager':>6} {'graph':>6} {'nccl':>6} {'busy_us':>8} {'span_us':>8} {'wall_us':>8} {'gap_us':>8} {'kern':>6}  {'P:span':>7} {'P:busy':>7}")
    for (ntok, rank) in sorted(per):
        r = per[(ntok, rank)][-1]  # last measured run (steady state)
        gap = r["wall"] - r["span"]
        ps, pb, _ = pspan.get((ntok, rank), (0, 0, 0))
        print(f"{ntok:>5} {rank:>4} {r['eager']:>6} {r['graph']:>6} {r['nccl']:>6} {r['busy']/1e3:>8.0f} {r['span']/1e3:>8.0f} {r['wall']/1e3:>8.0f} {gap/1e3:>8.0f} {r['nkern']:>6}  {ps:>7.0f} {pb:>7.0f}")
    return per

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite"); ap.add_argument("--label", default="trace")
    a = ap.parse_args()
    analyze(a.sqlite, a.label)

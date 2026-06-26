"""Characterize WHERE the small-prefill GPU idle goes, per bench_step window.

For the gap-bound 256-tok prefill (busy << wall), split the host (CUDA-runtime)
timeline inside each bench_step NVTX window into:
  launch   = sum dur of cudaLaunchKernel / cuLaunchKernel(Ex) / cudaGraphLaunch
  sync     = sum dur of cudaStreamSynchronize / cudaDeviceSynchronize / cudaEventSynchronize
  d2h/h2d  = sum dur of cudaMemcpy* (host<->device copies, often D2H sampling syncs)
  other_api= sum dur of all remaining runtime API calls
  python   = window - union(all runtime-API intervals)  [host time in NO cuda call = Python/host compute]
Also: GPU-busy (sum kernel dur), span, wall, idle=wall-busy, and the 10 largest
inter-kernel GPU bubbles with the host API covering each bubble's start.

Discriminates: launch-rate (launch large) vs sync-lockstep (sync/d2h large) vs
host-compute (python large).

Usage: python runs/analyze_idle_cause.py <sqlite> [--label L]
"""
import argparse, re, sqlite3, sys

MASK = -16777216

def union_len(intervals):
    if not intervals: return 0
    intervals = sorted(intervals)
    tot = 0; cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s > ce: tot += ce - cs; cs, ce = s, e
        else: ce = max(ce, e)
    tot += ce - cs
    return tot

def classify(nm):
    if nm.startswith(("cudaLaunchKernel", "cuLaunchKernel", "cudaGraphLaunch")): return "launch"
    if "Synchronize" in nm: return "sync"
    if nm.startswith("cudaMemcpy") or nm.startswith("cuMemcpy"): return "memcpy"
    return "other_api"

def analyze(path, label):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True); cur = con.cursor()
    sid = dict(cur.execute("SELECT id,value FROM StringIds").fetchall())
    steps = cur.execute("SELECT text,start,end,globalTid FROM NVTX_EVENTS WHERE text LIKE 'bench_step::%' ORDER BY start").fetchall()
    pids = sorted({tid & MASK for *_, tid in steps})
    pid_rank = {p: i for i, p in enumerate(pids)}
    # one representative window per (ntok,rank): the LAST (steady state)
    chosen = {}
    for t, s, e, tid in steps:
        rank = pid_rank[tid & MASK]
        m = re.search(r"N(\d+)", t)
        chosen[(int(m.group(1)), rank)] = (s, e, tid & MASK)

    print(f"\n===== {label} =====")
    for (ntok, rank), (s, e, pid) in sorted(chosen.items()):
        wall = e - s
        # runtime API calls in window
        rt = cur.execute("SELECT R.start,R.end,R.nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME R "
                         "WHERE R.start>=? AND R.start<? AND (R.globalTid&?)=? ORDER BY R.start",
                         (s, e, MASK, pid)).fetchall()
        buckets = {"launch": 0, "sync": 0, "memcpy": 0, "other_api": 0}
        cnt = {"launch": 0, "sync": 0, "memcpy": 0, "other_api": 0}
        all_iv = []
        for a, b, nid in rt:
            c = classify(sid.get(nid, ""))
            buckets[c] += b - a; cnt[c] += 1
            all_iv.append((a, b))
        api_union = union_len(all_iv)
        python = wall - api_union
        # GPU kernels in window
        ks = cur.execute("SELECT K.start,K.end FROM CUPTI_ACTIVITY_KIND_KERNEL K "
                         "WHERE K.globalPid=? AND K.start>=? AND K.start<? ORDER BY K.start",
                         (pid, s, e)).fetchall()
        busy = sum(b - a for a, b in ks)
        span = (ks[-1][1] - ks[0][0]) if ks else 0
        idle = wall - busy
        u = 1e6
        print(f"\n-- ntok={ntok} rank={rank} --  (wall={wall/u:.2f}ms span={span/u:.2f}ms busy={busy/u:.2f}ms idle={idle/u:.2f}ms  nkern={len(ks)} nAPI={len(rt)})")
        print(f"  HOST time split inside window (wall={wall/u:.2f}ms):")
        for k in ("launch", "sync", "memcpy", "other_api"):
            print(f"    {k:10s}: {buckets[k]/u:7.2f} ms  ({100*buckets[k]/wall:4.1f}%)  [{cnt[k]} calls]")
        print(f"    {'python':10s}: {python/u:7.2f} ms  ({100*python/wall:4.1f}%)  [host time in NO cuda API call]")
        print(f"    {'(api_union)':10s}: {api_union/u:7.2f} ms  ({100*api_union/wall:4.1f}%)  [wall covered by >=1 cuda API]")
        # largest inter-kernel GPU bubbles
        bubbles = []
        for i in range(1, len(ks)):
            g = ks[i][0] - ks[i-1][1]
            if g > 0: bubbles.append((g, ks[i-1][1], ks[i][0]))
        bubbles.sort(reverse=True)
        gpu_bubble_tot = sum(g for g, *_ in bubbles)
        print(f"  inter-kernel GPU bubble total = {gpu_bubble_tot/u:.2f} ms over {len(bubbles)} gaps; top 8:")
        for g, b0, b1 in bubbles[:8]:
            # which API calls overlap this bubble?
            ov = [(a, bb, sid.get(nid, "")) for a, bb, nid in rt if a < b1 and bb > b0]
            names = {}
            for a, bb, nm in ov:
                names[classify(nm)] = names.get(classify(nm), 0) + 1
            tag = ",".join(f"{k}x{v}" for k, v in sorted(names.items(), key=lambda x: -x[1])) or "NONE(python)"
            print(f"    gap={g/u:6.3f}ms  host-APIs-overlapping: {tag}")
    con.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite"); ap.add_argument("--label", default="trace")
    a = ap.parse_args()
    analyze(a.sqlite, a.label)

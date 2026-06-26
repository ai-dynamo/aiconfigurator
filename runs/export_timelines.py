"""Export per-step nsys timeline DATA -> ONE CSV for the notebook Gantt cell
(collector/layerwise/fpm_ground_truth/layerwise-alignment.ipynb cell 14).

NO plotting. Emits collector/layerwise/fpm_ground_truth/timelines/timelines.csv
with the EXACT schema the cell reads:
    phase, source, lane, group, kclass, start_us, dur_us
  phase  in {prefill_captured, prefill_eager, decode, mixed}
  source in {golden, collector}
  lane   in {host, compute, memory, comm}
  group  in {backbone, moe, comm, idle}

ONE representative step per (phase, source). Times normalized to start_us=0.

GOLDEN (multi-proc serve, rank0): the cross_device_reduce_2stage all-reduce
busy-waits under nsys (the verdict's SPIN artifact: ~23 ms captured / ~25 ms
eager of cross-rank wait that does NOT exist in real serving). We DE-SPIN it:
every all-reduce instance is capped at the real transfer floor (the global
captured p25 per-call ~ tens of us; data volume is tiny & size-independent), then
the timeline is compacted and its idle scaled so the step span == golden's real
FPM latency (captured ~16 ms, eager ~40 ms). The remaining idle is the genuine
host-dispatch / launch / sync residual that serving pays.

COLLECTOR (isolated single GPU): shown at RAW nsys scale (gap_scale=1, no comm).
Its large inter-kernel idle IS the diagnostic (isolated host-dispatch starvation);
its true reported isolated latency is even higher than the raw nsys span here.

See timelines/README.md for the inflation factors actually used.
"""
import sqlite3, numpy as np, csv, os

RUNS = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(RUNS, "..", "collector", "layerwise", "fpm_ground_truth", "timelines")
os.makedirs(OUTDIR, exist_ok=True)
GAP_IDLE_US = 5.0

# ── kernel -> (lane, group, kclass) ───────────────────────────────────────────
def classify(nm):
    if (nm == "cross_device_reduce_2stage" or nm.startswith("ncclDevKernel")
            or nm.startswith("two_shot") or "all_reduce" in nm
            or "alltoall" in nm.lower() or "all2all" in nm.lower()
            or nm == "delayStreamKernel"):
        return ("comm", "comm", "all_reduce" if "reduce" in nm.lower() or nm.startswith("two_shot") or nm=="delayStreamKernel" else "all2all")
    if nm.startswith("bmm_Bfloat16"):            return ("compute", "moe", "moe_gemm")
    if nm.startswith("routing") or nm.startswith("finalizeKernel"):
        return ("compute", "moe", "router")
    if "moe_forward_shared" in nm or "mul_silu" in nm:
        return ("compute", "moe", "shared_expert")
    if "fmha" in nm or "flash" in nm or "attention" in nm.lower():
        return ("compute", "backbone", "attention")
    if any(x in nm for x in ("chunk_", "gated_delta", "recompute_w_u", "merge_16x16",
                             "causal_conv1d", "fused_post_conv", "reshape_and_cache",
                             "slot_mapping", "zero_kv", "conv1d", "_state")):
        return ("memory", "backbone", "gdn_mamba")
    if nm.startswith("nvjet") or nm == "splitKreduce_kernel":
        return ("compute", "backbone", "dense_gemm")
    return ("memory", "backbone", "elementwise")

# ── layout: de-spun kernels -> normalized rows (compact gaps, scale to target) ─
def layout(kernels, target_span_us=None):
    """kernels: list of (start_ns, end_ns, lane, group, kclass) with FINAL (de-spun)
    durations. Returns (rows, busy_us, span_us). Preserves intra-active overlap
    (kernels keep real dur & relative offset inside an active block); only the
    GPU-idle gaps between active blocks are scaled (to hit target_span_us, or
    kept raw if None)."""
    t0 = min(k[0] for k in kernels)
    K = sorted((s - t0, e - t0, ln, gp, kc) for s, e, ln, gp, kc in kernels)
    iv = sorted((s, e) for s, e, *_ in K)
    union = []
    for s, e in iv:
        if union and s <= union[-1][1]:
            union[-1] = (union[-1][0], max(union[-1][1], e))
        else:
            union.append((s, e))
    busy = sum(e - s for s, e in union)
    span_raw = union[-1][1] - union[0][0]
    idle_raw = span_raw - busy
    if target_span_us is not None and idle_raw > 0:
        gscale = max(0.0, (target_span_us * 1e3 - busy) / idle_raw)
    else:
        gscale = 1.0
    # piecewise map: active blocks scale 1, gaps scale gscale
    seg = []  # (orig_lo, orig_hi, new_lo, scale)
    npos = 0.0
    for i, (s, e) in enumerate(union):
        if i > 0:
            g0 = union[i - 1][1]
            seg.append((g0, s, npos, gscale)); npos += (s - g0) * gscale
        seg.append((s, e, npos, 1.0)); npos += (e - s)
    span_new = npos
    def remap(x):
        for lo, hi, nl, sc in seg:
            if lo <= x <= hi:
                return nl + (x - lo) * sc
        return span_new
    rows = []
    for s, e, ln, gp, kc in K:
        ns, ne = remap(s), remap(e)
        rows.append((ln, gp, kc, ns / 1e3, max(ne - ns, 0) / 1e3))
    # idle + host-backbone tiling on the host lane
    for i, (s, e) in enumerate(union):
        if i > 0:
            g0 = union[i - 1][1]
            sg = (s - g0) * gscale
            if sg / 1e3 >= GAP_IDLE_US:
                rows.append(("host", "idle", "gpu_idle", remap(g0) / 1e3, sg / 1e3))
        rows.append(("host", "backbone", "host_active",
                     remap(s) / 1e3, (e - s) / 1e3))
    return rows, busy / 1e3, span_new / 1e3

# ── golden serve: AR-gap step segmentation (rank0), with fixed-floor de-spin ───
def golden_steps(con):
    cur = con.cursor()
    sid = dict(cur.execute("SELECT id,value FROM StringIds").fetchall())
    CDR = next(i for i, v in sid.items() if v == "cross_device_reduce_2stage")
    P0 = 284221356638208
    t0 = cur.execute("SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    lo = t0 + 124e9
    ks = cur.execute("SELECT start,end,shortName,graphId FROM CUPTI_ACTIVITY_KIND_KERNEL "
                     "WHERE globalPid=? AND start>=? ORDER BY start", (P0, lo)).fetchall()
    ar = [(s, e) for s, e, sn, g in ks if sn == CDR]
    arr = [(s, e, sn, g) for s, e, sn, g in ks]  # all
    # segment by AR gaps
    segs = []; cs = [(ks[0])]
    arsteps = []; cur_ar = [ks_ for ks_ in [arr[0]]]
    # build AR-only list with graphId for regime + gap detection
    arg = [(s, e, g) for s, e, sn, g in arr if sn == CDR]
    steps = []; cstep = [arg[0]]
    for i in range(1, len(arg)):
        if arg[i][0] - arg[i - 1][1] > 1_500_000:
            steps.append(cstep); cstep = []
        cstep.append(arg[i])
    steps.append(cstep)
    # global captured per-call AR floor (real transfer); from captured steps
    cap_durs = []
    for st in steps:
        if len(st) >= 40 and all(g for *_, g in st):
            cap_durs += [e - s for s, e, g in st]
    floor_ns = float(np.percentile(cap_durs, 25))  # ~ real transfer per AR call
    caps = [st for st in steps if len(st) >= 40 and all(g for *_, g in st)]
    eags = [st for st in steps if len(st) >= 40 and not all(g for *_, g in st)]
    def step_kernels(st):
        s0, e1 = st[0][0], st[-1][1]
        out = []
        for s, e, sn, g in arr:
            if not (s0 <= s < e1):
                continue
            nm = sid.get(sn, str(sn))
            ln, gp, kc = classify(nm)
            if kc == "all_reduce" and nm == "cross_device_reduce_2stage":
                e = s + min(e - s, floor_ns)      # DE-SPIN to real transfer floor
            out.append((s, e, ln, gp, kc))
        return out
    return floor_ns, caps, eags, step_kernels

# ── collector: NVTX CUDAGraphWrapper window -> kernel cluster (via time bracket)─
def collector_step(con, pid, tok, occurrence=3, post_ms=12):
    cur = con.cursor()
    allc = cur.execute("SELECT start,end,text FROM NVTX_EVENTS WHERE text LIKE '%input_ids%' "
                       "ORDER BY start").fetchall()
    pat = "'input_ids': [[%d]]" % tok
    cand = [(s, e) for s, e, t in allc if t and pat in t and "layers" not in t]
    if not cand:
        return None
    s0, e1 = cand[min(occurrence, len(cand) - 1)]
    khi = e1 + post_ms * 1_000_000
    kk = cur.execute("SELECT start,end,shortName FROM CUPTI_ACTIVITY_KIND_KERNEL "
                     "WHERE globalPid=? AND start>=? AND start<? ORDER BY start",
                     (pid, s0, khi)).fetchall()
    if not kk:
        return None
    sid = dict(cur.execute("SELECT id,value FROM StringIds").fetchall())
    # trim to first contiguous cluster (split on >2ms gaps)
    cl = [kk[0]]; clusters = []
    for i in range(1, len(kk)):
        if kk[i][0] - kk[i - 1][1] > 2_000_000:
            clusters.append(cl); cl = []
        cl.append(kk[i])
    clusters.append(cl)
    clusters.sort(key=lambda c: -(c[-1][1] - c[0][0]))   # largest by span
    cluster = clusters[0]
    out = []
    for s, e, sn in cluster:
        ln, gp, kc = classify(sid.get(sn, str(sn)))
        out.append((s, e, ln, gp, kc))
    return out

# ── orchestrate ───────────────────────────────────────────────────────────────
# golden FPM latency (ms) per phase -> de-inflation target. collector = raw (None).
FPM_MS = {"prefill_captured": 16.0, "prefill_eager": 40.6}
allrows = []
info = []

def add(phase, source, kernels, target_ms):
    if not kernels:
        info.append((phase, source, "MISSING", 0, 0, 0)); return
    rows, busy_us, span_us = layout(kernels, None if target_ms is None else target_ms * 1e3)
    idle = sum(d for ln, gp, kc, st, d in rows if gp == "idle")
    for ln, gp, kc, st, d in rows:
        allrows.append((phase, source, ln, gp, kc, round(st, 3), round(d, 3)))
    info.append((phase, source, "ok n=%d" % len(kernels), round(busy_us / 1e3, 2),
                 round(span_us / 1e3, 2), round(100 * idle / max(span_us, 1e-9))))

# GOLDEN serve (rank0), de-spun + scaled to FPM
gc = sqlite3.connect("file:%s/serve_nsys_trace.sqlite?mode=ro" % RUNS, uri=True)
floor_ns, caps, eags, gk = golden_steps(gc)
print("golden AR de-spin floor = %.1f us/call (real transfer)" % (floor_ns / 1e3))
add("prefill_captured", "golden", gk(caps[len(caps) // 2]), FPM_MS["prefill_captured"])
add("prefill_eager",    "golden", gk(eags[len(eags) // 2]), FPM_MS["prefill_eager"])
gc.close()

# COLLECTOR isolated single-GPU REAL-MoE (partB_realmoe), raw nsys scale.
CB = sqlite3.connect("file:%s/partB_realmoe/profiles/nsys/wu_b7d45bf38bf6328c_ctx_a1.sqlite?mode=ro" % RUNS, uri=True)
PCB = 287514506035200
add("prefill_captured", "collector", collector_step(CB, PCB, 256, occurrence=10), None)
add("prefill_eager",    "collector", collector_step(CB, PCB, 1024, occurrence=10), None)
CB.close()
# decode / mixed: no REAL-MoE trace exists (nsys_dec16 is backbone-only); SKIPPED.

# write CSV
out = os.path.join(OUTDIR, "timelines.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["phase", "source", "lane", "group", "kclass", "start_us", "dur_us"])
    w.writerows(allrows)

print("\n%-18s %-10s %-12s %8s %8s %6s" % ("phase", "source", "status", "busy_ms", "span_ms", "idle%"))
for r in info:
    print("%-18s %-10s %-12s %8s %8s %5s%%" % r)
print("\nwrote %d rows -> %s" % (len(allrows), out))

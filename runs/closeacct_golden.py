"""Rigorous per-step structural decomposition of the golden tp4/ep4 serve trace.

Unlike partA (which SUMS kernel durations, double-counting concurrent streams), this
MERGES kernel intervals per rank to get true GPU occupancy, then closes:

    span (wall of one engine step) = merged_busy + GPU_idle
    merged_busy                    = compute_busy + allreduce_busy  (if no overlap)
    allreduce_busy                 = real_comm_floor + spin   (floor = p25*count heuristic)

Reports RAW numbers per step (median over steady steps), no fitting.
Usage: python closeacct_golden.py [sqlite]
"""
import sqlite3, sys
from collections import defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 else "serve_nsys_trace.sqlite"
con = sqlite3.connect(f"file:{PATH}?mode=ro", uri=True); cur = con.cursor()
sid = dict(cur.execute("SELECT id,value FROM StringIds").fetchall())

def pct(xs, p):
    if not xs: return 0.0
    s = sorted(xs); k = (len(s)-1)*p/100.0; lo=int(k)
    return s[lo] if lo+1>=len(s) else s[lo]+(s[lo+1]-s[lo])*(k-lo)
def med(xs): return pct(xs,50)

def merge_busy(intervals):
    """union length of [start,end) intervals (ns)."""
    if not intervals: return 0, 0, 0
    iv = sorted(intervals); tot=0; cs,ce = iv[0]
    span_lo, span_hi = iv[0][0], iv[0][1]
    for s,e in iv[1:]:
        span_hi = max(span_hi, e)
        if s > ce: tot += ce-cs; cs,ce = s,e
        else: ce = max(ce,e)
    tot += ce-cs
    return tot, span_lo, span_hi

def klass(nm):
    if nm == "cross_device_reduce_2stage": return "allreduce"
    if "all_reduce" in nm: return "ew_other"
    if any(x in nm for x in ("moe_forward_shared","mul_silu_slice")): return "shared_expert"
    if nm.startswith("bmm_Bfloat16"): return "moe_gemm"
    if nm.startswith("routing") or nm=="finalizeKernelVecLoad": return "router"
    if any(x in nm for x in ("fmha","chunk_","recompute_w_u","merge_16x16","causal_conv1d",
            "fused_post_conv","reshape_and_cache","slot_mapping","zero_kv")): return "attn_gdn"
    if nm.startswith("nvjet") or nm=="splitKreduce_kernel": return "dense_gemm"
    return "ew_other"

CDR = next(i for i,v in sid.items() if v=="cross_device_reduce_2stage")
ranks = [r[0] for r in cur.execute(
    "SELECT globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY globalPid ORDER BY count(*) DESC")]
P0 = min(ranks)  # rank0 = lowest globalPid (matches partA P0=284221356638208)
t0 = cur.execute("SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
lo = t0 + 124e9  # skip warmup (same as partA)

ks = cur.execute("SELECT start,end,shortName,graphId FROM CUPTI_ACTIVITY_KIND_KERNEL "
                 "WHERE globalPid=? AND start>=? ORDER BY start",(P0,lo)).fetchall()
print(f"trace={PATH} rank0_pid={P0} kernels_after_warmup={len(ks)}")

# segment into steps by AR-kernel gaps > 1.5ms
ar = [(s,e,g) for s,e,sn,g in ks if sn==CDR]
steps=[]; cs=[ar[0]]
for i in range(1,len(ar)):
    if ar[i][0]-ar[i-1][1] > 1_500_000: steps.append(cs); cs=[]
    cs.append(ar[i])
steps.append(cs)
def regime(st): return "CAPTURED" if all(g for *_,g in st) else "EAGER"

# index kernels for fast per-step slice
def step_kernels(s0,e1):
    return [(s,e,sn,g) for s,e,sn,g in ks if s0<=s<e1]

for reg in ("CAPTURED","EAGER"):
    grp=[st for st in steps if len(st)>=40 and regime(st)==reg]
    if not grp: continue
    rows=[]
    for st in grp:
        s0,e1 = st[0][0], st[-1][1]
        kk = step_kernels(s0,e1)
        all_iv=[(s,e) for s,e,sn,g in kk]
        non_iv=[(s,e) for s,e,sn,g in kk if sn!=CDR]
        ar_iv =[(s,e) for s,e,sn,g in kk if sn==CDR]
        ar_dur=[e-s for s,e in ar_iv]
        mb_all,_,_ = merge_busy(all_iv)
        mb_non,_,_ = merge_busy(non_iv)
        mb_ar ,_,_ = merge_busy(ar_iv)
        span = e1-s0
        sum_dur = sum(e-s for s,e in all_iv)
        # AR floor/spin
        floor = pct(ar_dur,25)*len(ar_dur) if ar_dur else 0
        # per-class summed (for compute breakdown)
        cls=defaultdict(float)
        for s,e,sn,g in kk: cls[klass(sid.get(sn,str(sn)))]+=e-s
        rows.append(dict(span=span, sum_dur=sum_dur, mb_all=mb_all, mb_non=mb_non,
                         mb_ar=mb_ar, ar_n=len(ar_dur), ar_tot=sum(ar_dur),
                         ar_floor=floor, ar_med=med(ar_dur), ar_p25=pct(ar_dur,25),
                         ar_max=max(ar_dur) if ar_dur else 0, idle=span-mb_all, cls=cls))
    rows = rows[3:] if len(rows)>6 else rows
    n=len(rows)
    g=lambda k: med([r[k] for r in rows])/1e6  # ns->ms
    print(f"\n===== {reg}  n_steady={n} =====")
    print(f"  step span (wall)        : {g('span'):7.2f} ms")
    print(f"  Sum(kernel dur)  [partA] : {g('sum_dur'):7.2f} ms   <- overcounts concurrent streams")
    print(f"  merged busy ALL          : {g('mb_all'):7.2f} ms   <- TRUE gpu occupancy")
    print(f"    merged busy non-AR     : {g('mb_non'):7.2f} ms")
    print(f"    merged busy AR-only    : {g('mb_ar'):7.2f} ms")
    print(f"  GPU-idle (span-busy)     : {g('idle'):7.2f} ms")
    print(f"  --- all-reduce calls ---")
    print(f"    count/step             : {med([r['ar_n'] for r in rows]):.0f}")
    print(f"    per-call p25/med/max   : {g('ar_p25')*1e3:.1f} / {g('ar_med')*1e3:.1f} / {g('ar_max')*1e3:.1f} us")
    print(f"    AR total/step          : {g('ar_tot'):7.2f} ms")
    print(f"    AR floor (p25*n)=comm  : {g('ar_floor'):7.2f} ms")
    print(f"    AR spin (tot-floor)    : {g('ar_tot')-g('ar_floor'):7.2f} ms")
    print(f"  --- compute classes (summed dur, ms) ---")
    for cc in ("attn_gdn","moe_gemm","shared_expert","router","dense_gemm","ew_other"):
        print(f"    {cc:<16}{med([r['cls'][cc] for r in rows])/1e6:7.2f}")
    # closure check
    print(f"  CLOSURE: busy_non-AR {g('mb_non'):.2f} + AR_busy {g('mb_ar'):.2f} + idle {g('idle'):.2f}"
          f" = {g('mb_non')+g('mb_ar')+g('idle'):.2f} vs span {g('span'):.2f}")
con.close()

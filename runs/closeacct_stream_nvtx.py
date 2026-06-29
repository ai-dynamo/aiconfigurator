"""Decompose the golden STREAM regime using an NVTX-bracketed window.

period_golden.py pushes an NVTX range `STREAMPHASE_<nt>` around the timed
LLM.generate([NREQ prompts]) call. This parser locks onto that exact window
(ground truth = the measured per-req wall), then on rank0 computes merged-interval
occupancy recognizing ALL all-reduce kernel variants (the saturated/packed stream
uses two_shot / fused-triton all-reduce, NOT cross_device_reduce_2stage):

    window span = compute_busy + AR_busy + GPU_idle
    AR_busy     = comm_floor(p25*count) + spin

Reports the whole window, a steady mid-sub-window (drop first/last 10%), and
per-request normalization (window / NREQ).

Usage: python closeacct_stream_nvtx.py <sqlite> [NREQ=128] [phase_substr=STREAMPHASE]
"""
import sqlite3, sys, numpy as np
from collections import defaultdict

PATH = sys.argv[1]
NREQ = int(sys.argv[2]) if len(sys.argv) > 2 else 128
TAG  = sys.argv[3] if len(sys.argv) > 3 else "STREAMPHASE"
con = sqlite3.connect(f"file:{PATH}?mode=ro", uri=True); cur = con.cursor()
sid = dict(cur.execute("SELECT id,value FROM StringIds")); inv = {v: k for k, v in sid.items()}

AR_NAMES = {"two_shot_all_reduce_kernel_inplace", "cross_device_reduce_2stage",
            "one_shot_all_reduce_kernel",
            "triton_poi_fused_add_all_reduce_4", "triton_poi_fused_add_all_reduce_2",
            "triton_poi_fused_add_all_reduce_0", "triton_poi_fused_add_all_reduce_6",
            "ncclDevKernel_AllReduce_Sum_f32_RING_LL"}
AR_IDS = {inv[n] for n in AR_NAMES if n in inv}

def merge(iv):
    if not iv: return 0
    iv = sorted(iv); tot = 0; cs, ce = iv[0]
    for s, e in iv[1:]:
        if s > ce: tot += ce - cs; cs, ce = s, e
        else: ce = max(ce, e)
    tot += ce - cs; return tot
def pct(xs, p):
    if not xs: return 0.0
    s = sorted(xs); k = (len(s)-1)*p/100; lo = int(k)
    return s[lo] if lo+1 >= len(s) else s[lo] + (s[lo+1]-s[lo])*(k-lo)
def med(xs): return pct(xs, 50)

# find NVTX window(s)
rows = cur.execute("SELECT text,start,end FROM NVTX_EVENTS WHERE text LIKE ? AND end IS NOT NULL ORDER BY start",
                   (f"%{TAG}%",)).fetchall()
if not rows:
    print(f"NO NVTX range matching '{TAG}' — markers absent?"); sys.exit(1)
ranks = [r[0] for r in cur.execute("SELECT globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY globalPid ORDER BY count(*) DESC")]
P0 = min(ranks)
ks = cur.execute("SELECT start,end,shortName FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid=? ORDER BY start", (P0,)).fetchall()
kstart = [k[0] for k in ks]
import bisect

def decomp(t0, t1, label, nreq=None):
    i = bisect.bisect_left(kstart, t0); j = bisect.bisect_right(kstart, t1)
    seg = ks[i:j]
    span = (t1 - t0) / 1e6
    allv = [(s, e) for s, e, sn in seg]
    arv  = [(s, e) for s, e, sn in seg if sn in AR_IDS]
    nonv = [(s, e) for s, e, sn in seg if sn not in AR_IDS]
    ard  = [e - s for s, e in arv]
    mb = merge(allv)/1e6; mc = merge(nonv)/1e6; mar = merge(arv)/1e6
    comm = pct(ard, 25)*len(ard)/1e6 if ard else 0.0
    idle = span - mb
    print(f"\n[{label}] span={span:.1f}ms  nkern={len(seg)}  nAR={len(arv)}")
    print(f"   merged_busy={mb:.2f}  compute(nonAR)={mc:.2f}  AR_busy={mar:.2f} (comm_floor={comm:.2f} spin={mar-comm:.2f})  idle={idle:.2f}")
    print(f"   occupancy: compute {100*mc/span:.0f}%  AR {100*mar/span:.0f}%  idle {100*idle/span:.0f}%")
    print(f"   CLOSURE: {mc:.2f}+{mar:.2f}+{idle:.2f} = {mc+mar+idle:.2f}  vs span {span:.2f}")
    if nreq:
        print(f"   PER-REQ (/{nreq}): wall={span/nreq:.2f}  compute={mc/nreq:.2f}  comm={comm/nreq:.3f}  spin={(mar-comm)/nreq:.2f}  idle={idle/nreq:.2f}")
    return dict(span=span, mc=mc, mar=mar, comm=comm, idle=idle)

print(f"trace={PATH} nGPU={len(ranks)} rank0={P0} AR_variants_found={len(AR_IDS)} nvtx_windows={len(rows)}")
for txt, t0, t1 in rows:
    print(f"\n===== NVTX '{txt}'  ({(t1-t0)/1e6:.0f}ms wall) =====")
    decomp(t0, t1, f"{txt} WHOLE", nreq=NREQ)
    # steady mid sub-window (drop first/last 10%)
    d = t1 - t0; decomp(int(t0 + 0.1*d), int(t1 - 0.1*d), f"{txt} STEADY-mid80%")
con.close()

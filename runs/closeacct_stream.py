"""Generalized per-step structural decomposition for ANY golden-style nsys trace
(serve OR offline LLM.generate; tp>1 with all-reduce, or 1-GPU). Merged-interval
occupancy (NOT summed durations). Segments steps by all-reduce gaps (tp>1) or by
kernel gaps (1-GPU). Drops warmup by fraction. Reports per-step:

    span(wall) = compute_busy + allreduce_busy + GPU_idle      (must close to span)
    allreduce_busy = real_comm_floor(p25*count) + spin

Usage: python closeacct_stream.py <sqlite> [warmup_frac=0.3] [gap_us=1500]
This is the parser to run on the FRESH golden-stream nsys trace (Run 3 in HANDOFF).
"""
import sqlite3, sys
from collections import defaultdict

PATH = sys.argv[1]
WARM = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
GAP  = int(sys.argv[3])*1000 if len(sys.argv) > 3 else 1_500_000
con = sqlite3.connect(f"file:{PATH}?mode=ro", uri=True); cur = con.cursor()
sid = dict(cur.execute("SELECT id,value FROM StringIds"))
def pct(xs,p):
    if not xs: return 0.0
    s=sorted(xs); k=(len(s)-1)*p/100; lo=int(k)
    return s[lo] if lo+1>=len(s) else s[lo]+(s[lo+1]-s[lo])*(k-lo)
def med(xs): return pct(xs,50)
def merge(iv):
    if not iv: return 0
    iv=sorted(iv); tot=0; cs,ce=iv[0]
    for s,e in iv[1:]:
        if s>ce: tot+=ce-cs; cs,ce=s,e
        else: ce=max(ce,e)
    tot+=ce-cs; return tot
def klass(nm):
    if nm=="cross_device_reduce_2stage": return "allreduce"
    if "all_reduce" in nm: return "ew_other"
    if any(x in nm for x in("moe_forward_shared","mul_silu_slice")): return "shared_expert"
    if nm.startswith("bmm_Bfloat16"): return "moe_gemm"
    if nm.startswith("routing") or nm=="finalizeKernelVecLoad": return "router"
    if any(x in nm for x in("fmha","chunk_","recompute_w_u","merge_16x16","causal_conv1d",
        "fused_post_conv","reshape_and_cache","slot_mapping","zero_kv")): return "attn_gdn"
    if nm.startswith("nvjet") or nm=="splitKreduce_kernel": return "dense_gemm"
    return "ew_other"

CDR = next((i for i,v in sid.items() if v=="cross_device_reduce_2stage"), None)
ranks=[r[0] for r in cur.execute("SELECT globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY globalPid ORDER BY count(*) DESC")]
P0=min(ranks)
ks=cur.execute("SELECT start,end,shortName,graphId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid=? ORDER BY start",(P0,)).fetchall()
print(f"trace={PATH} nGPU={len(ranks)} rank0={P0} AR={'yes' if CDR else 'no'} warmup_frac={WARM}")

# segment: by AR gaps if tp>1, else by kernel gaps
if CDR:
    anchors=[(s,e,g) for s,e,sn,g in ks if sn==CDR]
else:
    anchors=[(s,e,g) for s,e,sn,g in ks]
steps=[]; cs=[anchors[0]]
for i in range(1,len(anchors)):
    if anchors[i][0]-anchors[i-1][1] > GAP: steps.append((cs[0][0],cs[-1][1])); cs=[]
    cs.append(anchors[i])
steps.append((cs[0][0],cs[-1][1]))
# keep steady steps (drop warmup head + tail); require plausible step (has kernels)
n=len(steps); steps=steps[int(n*WARM):int(n*(1-0.05)) or None]
import bisect
kstart=[k[0] for k in ks]
rows=[]
for s0,e1 in steps:
    i=bisect.bisect_left(kstart,s0); j=bisect.bisect_right(kstart,e1)
    kk=ks[i:j]
    if len(kk)<20: continue
    all_iv=[(s,e) for s,e,sn,g in kk]; non=[(s,e) for s,e,sn,g in kk if sn!=CDR]
    ar=[(s,e) for s,e,sn,g in kk if sn==CDR]; ard=[e-s for s,e in ar]
    cls=defaultdict(float)
    for s,e,sn,g in kk: cls[klass(sid.get(sn,str(sn)))]+=e-s
    rows.append(dict(span=e1-s0, mb=merge(all_iv), non=merge(non), ar=merge(ar),
                     comm=pct(ard,25)*len(ard), arn=len(ard), cls=cls))
if not rows:
    print("NO STEADY STEPS — adjust gap_us / warmup_frac"); sys.exit(0)
g=lambda k: med([r[k] for r in rows])/1e6
print(f"n_steady_steps={len(rows)}  (median over them)")
print(f"  span (wall)       : {g('span'):7.2f} ms")
print(f"  merged busy ALL   : {g('mb'):7.2f} ms")
print(f"    compute (non-AR): {g('non'):7.2f} ms")
print(f"    all-reduce busy : {g('ar'):7.2f} ms   (comm_floor {g('comm'):.2f} + spin {g('ar')-g('comm'):.2f}, n_AR={med([r['arn'] for r in rows]):.0f})")
print(f"  GPU-idle          : {g('span')-g('mb'):7.2f} ms")
print(f"  compute classes (ms): " + "  ".join(f"{c}={med([r['cls'][c] for r in rows])/1e6:.2f}"
      for c in("attn_gdn","moe_gemm","shared_expert","router","dense_gemm","ew_other")))
print(f"  CLOSURE: compute {g('non'):.2f} + AR {g('ar'):.2f} + idle {g('span')-g('mb'):.2f} = "
      f"{g('non')+g('ar')+g('span')-g('mb'):.2f}  vs span {g('span'):.2f}")
con.close()

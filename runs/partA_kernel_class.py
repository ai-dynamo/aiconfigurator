"""Part A: kernel-class attribution of golden GPU-busy, per capture-regime.

Buckets per-step GPU-busy (Sum kernel durations) of the golden tp4/ep4 serve trace
into the task's classes, separating the all-reduce SPIN-WAIT (idle artifact of the
isolated probe) from the synced AR FLOOR (real comm). Qwen3.6-35B-A3B kernel map
(40 layers = 30 GDN + 10 flash-attn; every layer MoE-64-top8 + shared expert):

  all-reduce(real=floor) : cross_device_reduce_2stage (p25*count); spin = sum-floor
  attn/GDN/Mamba         : fmha*, chunk_*, recompute_w_u*, merge_16x16*, causal_conv1d*,
                           fused_post_conv*, reshape_and_cache*, slot_mapping*, zero_kv*
  MoE-expert GEMM        : bmm_Bfloat16_* (flashinfer trtllm_bf16_moe FC1/FC2 grouped GEMM)
  shared-expert          : *moe_forward_shared*, mul_silu_slice
  router                 : routing*, finalizeKernelVecLoad
  dense-proj GEMM        : nvjet_*, splitKreduce  (QKV/o_proj/gate/router-linear; backbone)
  elementwise/rmsnorm/oth: *elementwise*, *rms_norm*, DeviceScan*, residual triton, AR-fused

Usage: python runs/partA_kernel_class.py [sqlite] [--p0 PID]
"""
import sqlite3, sys, re, numpy as np
from collections import defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "runs/serve_nsys_trace.sqlite"
P0 = 284221356638208
con = sqlite3.connect(f"file:{PATH}?mode=ro", uri=True); cur = con.cursor()
sid = dict(cur.execute("SELECT id,value FROM StringIds").fetchall())
def has(s, *subs): return any(x in s for x in subs)

def klass(nm):
    if nm == "cross_device_reduce_2stage": return "allreduce"
    if "all_reduce" in nm: return "ew_other"          # fused residual+AR (small)
    if has(nm, "moe_forward_shared", "mul_silu_slice"): return "shared_expert"
    if nm.startswith("bmm_Bfloat16"): return "moe_gemm"
    if nm.startswith("routing") or nm == "finalizeKernelVecLoad": return "router"
    if has(nm, "fmha", "chunk_", "recompute_w_u", "merge_16x16", "causal_conv1d",
            "fused_post_conv", "reshape_and_cache", "slot_mapping", "zero_kv"): return "attn_gdn"
    if nm.startswith("nvjet") or nm == "splitKreduce_kernel": return "dense_gemm"
    return "ew_other"

CDR = next(i for i, v in sid.items() if v == "cross_device_reduce_2stage")
t0 = cur.execute("SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
lo = t0 + 124e9
ks = cur.execute("SELECT start,end,shortName,graphId,gridX FROM CUPTI_ACTIVITY_KIND_KERNEL "
                 "WHERE globalPid=? AND start>=? ORDER BY start", (P0, lo)).fetchall()
ar = [(s, e, g) for s, e, sn, g, gx in ks if sn == CDR]
steps = []; cs = [ar[0]]
for i in range(1, len(ar)):
    if ar[i][0] - ar[i-1][1] > 1_500_000: steps.append(cs); cs = []
    cs.append(ar[i])
steps.append(cs)

def regime(st):
    return "CAPTURED" if all(g for *_, g in st) else "EAGER"

def bucket(group_steps, name):
    # per-step class sums, then median across steady steps (drop first 3 cold)
    per = []
    for st in group_steps:
        s0, e1 = st[0][0], st[-1][1]
        cls = defaultdict(float)
        ar_durs = []
        for s, e, sn, g, gx in ks:
            if not (s0 <= s < e1): continue
            d = e - s
            nm = sid.get(sn, str(sn))
            c = klass(nm)
            cls[c] += d
            if c == "allreduce": ar_durs.append(d)
        ar_durs = np.array(ar_durs)
        floor = np.percentile(ar_durs, 25) * len(ar_durs) if len(ar_durs) else 0.0
        spin = cls["allreduce"] - floor
        # token size via triton AR-fused gridX
        per.append((cls, floor, spin, e1 - s0))
    per = per[3:] if len(per) > 6 else per
    def med(f): return np.median([f(x) for x in per])
    classes = ["attn_gdn", "moe_gemm", "shared_expert", "router", "dense_gemm", "ew_other"]
    comp = {c: med(lambda x: x[0][c]) / 1e3 for c in classes}
    floor = med(lambda x: x[1]) / 1e3
    spin = med(lambda x: x[2]) / 1e3
    span = med(lambda x: x[3]) / 1e3
    noncomm = sum(comp.values())
    print(f"\n===== GOLDEN {name}  (n_steady={len(per)}, span={span:.1f}ms) =====")
    print(f"  {'class':<22}{'ms':>9}  {'% of real-work':>14}")
    real = noncomm + floor   # real work = compute backbone + synced comm floor
    for c in classes:
        print(f"  {c:<22}{comp[c]:>9.2f}  {100*comp[c]/real:>13.1f}%")
    print(f"  {'all-reduce FLOOR(comm)':<22}{floor:>9.2f}  {100*floor/real:>13.1f}%")
    print(f"  {'-'*44}")
    print(f"  {'REAL WORK (busy+comm)':<22}{real:>9.2f}  {100:>13.1f}%   <- busy-metric prediction")
    print(f"  {'all-reduce SPIN (idle)':<22}{spin:>9.2f}     [probe artifact, NOT counted]")
    return real, comp, floor, spin

print(f"trace: {PATH}")
caps = [st for st in steps if len(st) >= 40 and regime(st) == "CAPTURED"]
eags = [st for st in steps if len(st) >= 40 and regime(st) == "EAGER"]
print(f"captured steps={len(caps)}  eager steps={len(eags)}")
if caps: bucket(caps, "CAPTURED (<=512 regime, ~256 tok)")
if eags: bucket(eags, "EAGER (>512 regime, ~512 tok)")
con.close()

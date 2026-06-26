"""Part B + D: collector (1-GPU sharded tp4/ep4, REAL MoE, full-depth) GPU-busy by
kernel class, per ctx size. Buckets kernels inside each bench_step (bs1, past0)
window using the SAME classifier as Part A, so the collector backbone is directly
comparable to golden per-class. moe_gemm here -> Part D shard-equivalence check.

Usage: python runs/partBD_collector_class.py <collector_sqlite>
"""
import sqlite3, sys, re, numpy as np
from collections import defaultdict

PATH = sys.argv[1]
MASK = -16777216
con = sqlite3.connect(f"file:{PATH}?mode=ro", uri=True); cur = con.cursor()
sid = dict(cur.execute("SELECT id,value FROM StringIds").fetchall())
def has(s, *subs): return any(x in s for x in subs)
def klass(nm):
    if nm == "cross_device_reduce_2stage": return "allreduce"
    if "all_reduce" in nm: return "ew_other"
    if has(nm, "moe_forward_shared", "mul_silu_slice"): return "shared_expert"
    if nm.startswith("bmm_Bfloat16"): return "moe_gemm"
    if nm.startswith("routing") or nm == "finalizeKernelVecLoad": return "router"
    if has(nm, "fmha", "chunk_", "recompute_w_u", "merge_16x16", "causal_conv1d",
            "fused_post_conv", "reshape_and_cache", "slot_mapping", "zero_kv"): return "attn_gdn"
    if nm.startswith("nvjet") or nm == "splitKreduce_kernel": return "dense_gemm"
    return "ew_other"

CLASSES = ["attn_gdn", "moe_gemm", "shared_expert", "router", "dense_gemm", "ew_other", "allreduce"]
steps = cur.execute("SELECT text,start,end,globalTid FROM NVTX_EVENTS WHERE text LIKE 'bench_step::%' ORDER BY start").fetchall()
# group windows by ntok for bs1 past0 only
per_tok = defaultdict(list)
for t, s, e, tid in steps:
    m = re.match(r"bench_step::N0*(\d+)::bs(\d+)::past0*(\d*)", t)
    if not m: continue
    ntok, bs, past = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    if bs != 1 or past != 0: continue
    per_tok[ntok].append((s, e, tid & MASK))

print(f"trace: {PATH}")
print(f"{'ntok':>6} | " + " ".join(f"{c:>10}" for c in CLASSES) + f" | {'NONCOMM':>9} {'span_ms':>8} {'nkern':>6}")
results = {}
for ntok in sorted(per_tok):
    wins = per_tok[ntok][3:] if len(per_tok[ntok]) > 6 else per_tok[ntok]  # drop cold
    rows = []
    for s, e, pid in wins:
        ks = cur.execute("SELECT K.start,K.end,K.shortName FROM CUPTI_ACTIVITY_KIND_KERNEL K "
                         "WHERE K.globalPid=? AND K.start>=? AND K.start<? ", (pid, s, e)).fetchall()
        cls = defaultdict(float); nk = 0
        for a, b, sn in ks:
            cls[klass(sid.get(sn, str(sn)))] += (b - a); nk += 1
        rows.append((cls, e - s, nk))
    def med(f): return np.median([f(r) for r in rows]) if rows else 0.0
    vals = {c: med(lambda r: r[0][c]) / 1e3 for c in CLASSES}
    noncomm = sum(vals[c] for c in CLASSES if c != "allreduce")
    span = med(lambda r: r[1]) / 1e3; nk = med(lambda r: r[2])
    results[ntok] = vals
    print(f"{ntok:>6} | " + " ".join(f"{vals[c]:>10.3f}" for c in CLASSES) + f" | {noncomm:>9.3f} {span:>8.2f} {nk:>6.0f}")
con.close()

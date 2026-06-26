"""Per-rank MoE-expert-GEMM busy on the golden tp4/ep4 serve trace.

Extends runs/partA_kernel_class.py: instead of bucketing one rank, it groups the
moe_gemm class (bmm_Bfloat16*) by rank (globalPid) and reports the REAL per-rank
routing skew of golden serving (no --enable-eplb -> skewed routing).

Per-rank busy = Sigma local kernel durations per captured step -> median across
steady captured steps. This is a purely local quantity (no cross-rank NVTX/step-
window alignment needed); the multi-process alignment issue only breaks cross-rank
step-window *timing*, not per-rank local kernel sums.

PART 1: per-rank moe_gemm busy (ms/step) for ALL 4 ranks; MAX/MIN/MEAN + MAX/MEAN.
PART 2: barrier check -- per-rank step SPAN ~ equal (all gated by collective
        barriers) while per-rank BUSY differs => step latency set by MAX-rank.

Usage: uv run --active python runs/partA_moe_perrank.py [sqlite]
"""
import sqlite3, sys, numpy as np
from collections import defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "runs/serve_nsys_trace.sqlite"
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

CDR = next(i for i, v in sid.items() if v == "cross_device_reduce_2stage")
pids = [p for (p,) in cur.execute(
    "SELECT DISTINCT globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY globalPid").fetchall()]
t0 = cur.execute("SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
lo = t0 + 124e9  # same warmup cutoff as partA

def analyze_rank(pid):
    ks = cur.execute("SELECT start,end,shortName,graphId FROM CUPTI_ACTIVITY_KIND_KERNEL "
                     "WHERE globalPid=? AND start>=? ORDER BY start", (pid, lo)).fetchall()
    ar = [(s, e, g) for s, e, sn, g in ks if sn == CDR]
    # segment into steps on >1.5ms gaps between consecutive all-reduce kernels
    steps = []; cs = [ar[0]]
    for i in range(1, len(ar)):
        if ar[i][0] - ar[i-1][1] > 1_500_000: steps.append(cs); cs = []
        cs.append(ar[i])
    steps.append(cs)
    rows = []
    for st in steps:
        if len(st) < 40: continue
        captured = all(g for *_, g in st)        # cudagraph replay -> graphId set
        if not captured: continue                 # PART 1/2 use CAPTURED <=512 steps
        s0, e1 = st[0][0], st[-1][1]
        cls = defaultdict(float)
        for s, e, sn, g in ks:
            if not (s0 <= s < e1): continue
            cls[klass(sid.get(sn, str(sn)))] += (e - s)
        busy = sum(v for k, v in cls.items() if k != "allreduce")  # local compute backbone
        rows.append((cls["moe_gemm"], busy, e1 - s0))             # ns
    rows = rows[3:] if len(rows) > 6 else rows                     # drop cold
    moe = np.median([r[0] for r in rows]) / 1e6                    # ns -> ms
    busy = np.median([r[1] for r in rows]) / 1e6
    span = np.median([r[2] for r in rows]) / 1e6
    return moe, busy, span, len(rows)

print(f"trace: {PATH}")
print(f"ranks (globalPid): {pids}\n")
res = {p: analyze_rank(p) for p in pids}

print("="*72)
print("PART 1 -- per-rank MoE-expert-GEMM busy (bmm_Bfloat16), CAPTURED <=512 steps")
print("="*72)
print(f"  {'rank':<6}{'globalPid':>20}{'moe_gemm ms':>14}{'n_steps':>9}")
moes = []
for i, p in enumerate(pids):
    moe, busy, span, n = res[p]
    moes.append(moe)
    print(f"  r{i:<5}{p:>20}{moe:>14.3f}{n:>9}")
moes = np.array(moes)
mx, mn, mean = moes.max(), moes.min(), moes.mean()
imax = int(moes.argmax())
print(f"\n  MAX  (rank{imax}) = {mx:.3f} ms/step   <-- golden BOTTLENECK rank")
print(f"  MIN            = {mn:.3f} ms/step")
print(f"  MEAN (4 ranks) = {mean:.3f} ms/step")
print(f"  MAX/MEAN       = {mx/mean:.3f}x   <-- golden REAL per-rank routing skew")
print(f"  MAX/MIN        = {mx/mn:.3f}x")

print("\n" + "="*72)
print("PART 2 -- barrier check: per-rank STEP SPAN vs per-rank BUSY")
print("="*72)
print(f"  {'rank':<6}{'moe_gemm':>11}{'busy(all)':>12}{'span':>10}{'slack=span-busy':>17}")
spans = []; busys = []
for i, p in enumerate(pids):
    moe, busy, span, n = res[p]
    spans.append(span); busys.append(busy)
    print(f"  r{i:<5}{moe:>11.3f}{busy:>12.3f}{span:>10.3f}{span-busy:>17.3f}")
spans = np.array(spans); busys = np.array(busys)
print(f"\n  span  MAX/MIN = {spans.max()/spans.min():.4f}x  (CV={spans.std()/spans.mean()*100:.1f}%)  <- ~equal => barrier-gated")
print(f"  busy  MAX/MIN = {busys.max()/busys.min():.4f}x  (CV={busys.std()/busys.mean()*100:.1f}%)  <- differs => routing skew")

print("\n" + "="*72)
print("PART 3 -- VERDICT: collector power_law_1.2 vs golden")
print("="*72)
COLLECTOR = 2.93   # collector rank0 power_law_1.2 moe_gemm @256 (verdict B/D)
GR0 = moes[0]      # golden rank0 (the verdict's 1.61 comparand)
print(f"  collector single-GPU power_law_1.2 moe_gemm   = {COLLECTOR:.2f} ms")
print(f"  golden rank0 (verdict comparand)              = {GR0:.3f} ms")
print(f"  golden BOTTLENECK rank (MAX)                  = {mx:.3f} ms")
print(f"  golden MEAN (4 ranks)                         = {mean:.3f} ms")
print()
print(f"  verdict ratio  collector / golden-rank0       = {COLLECTOR/GR0:.2f}x")
print(f"  vs BOTTLENECK  collector / golden-MAX         = {COLLECTOR/mx:.2f}x   <- still >> 1")
print(f"  rank-misalign  golden-MAX / golden-rank0      = {mx/GR0:.3f}x   <- only this is 'rank mis-compare'")
print()
print(f"  Decomposition of the {COLLECTOR/GR0:.2f}x:")
print(f"    rank-misalignment component  = {mx/GR0:.3f}x")
print(f"    genuine over-count component = {COLLECTOR/mx:.3f}x")
print(f"    product                      = {(mx/GR0)*(COLLECTOR/mx):.3f}x  (= collector/rank0)")
print()
print(f"  Routing concentration:")
print(f"    golden REAL skew (MAX/MEAN)            = {mx/mean:.3f}x  ({(mx/mean-1)*100:.1f}% over balanced)")
print(f"    power_law_1.2 implied (2.93/golden-MEAN) = {COLLECTOR/mean:.3f}x  ({(COLLECTOR/mean-1)*100:.0f}% over balanced)")
con.close()

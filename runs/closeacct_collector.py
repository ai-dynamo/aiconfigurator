"""Per-forward structural decomposition of a collector nsys trace (merged-interval).

Segments by the collector's bench_step::N<ntok> NVTX markers, but inside each marker
finds the actual forward KERNEL BURST (kernels separated by < GAP_NS) so the ~120ms
marker-control idle padding is excluded. Reports span / merged-busy / idle and AR
(if multi-GPU) per token size, median over runs. Raw numbers, no fitting.

Usage: python closeacct_collector.py <sqlite> [GAP_us]
"""
import sqlite3, sys
from collections import defaultdict

PATH = sys.argv[1]
GAP = int(sys.argv[2])*1000 if len(sys.argv)>2 else 2_000_000  # burst gap thresh (ns)
con = sqlite3.connect(f"file:{PATH}?mode=ro", uri=True); cur=con.cursor()
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

CDR = next((i for i,v in sid.items() if v=="cross_device_reduce_2stage"), None)
ranks=[r[0] for r in cur.execute("SELECT globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY globalPid ORDER BY count(*) DESC")]
P0=min(ranks)
print(f"trace={PATH}  nGPU={len(ranks)}  rank0={P0}  CDR={'yes' if CDR else 'no(1GPU)'}  burst_gap={GAP/1e6}ms")

bs=cur.execute("SELECT text,start,end FROM NVTX_EVENTS WHERE text LIKE 'bench_step%' AND end IS NOT NULL ORDER BY start").fetchall()
byN=defaultdict(list)
for t,s,e in bs: byN[t.split('::')[1]].append((s,e))

# pull rank0 kernels once
ks=cur.execute("SELECT start,end,shortName FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid=? ORDER BY start",(P0,)).fetchall()
import bisect
kstarts=[k[0] for k in ks]

def burst_in(s0,e1):
    """largest contiguous kernel burst within [s0,e1] on rank0."""
    i=bisect.bisect_left(kstarts,s0); j=bisect.bisect_right(kstarts,e1)
    sub=[(s,e,sn) for s,e,sn in ks[i:j]]
    if not sub: return None
    # split into bursts by GAP
    bursts=[]; cur_b=[sub[0]]
    for k in range(1,len(sub)):
        if sub[k][0]-cur_b[-1][1] > GAP: bursts.append(cur_b); cur_b=[]
        cur_b.append(sub[k])
    bursts.append(cur_b)
    # pick burst with max total kernel duration (the real forward)
    best=max(bursts,key=lambda b:sum(e-s for s,e,_ in b))
    return best

for N in sorted(byN, key=lambda x:int(x[1:])):
    nt=int(N[1:])
    if nt>512: continue
    rows=[]
    for s0,e1 in byN[N]:
        b=burst_in(s0,e1)
        if not b: continue
        span=b[-1][1]-b[0][0]
        all_iv=[(s,e) for s,e,_ in b]
        ar_iv=[(s,e) for s,e,sn in b if sn==CDR] if CDR else []
        non_iv=[(s,e) for s,e,sn in b if sn!=CDR]
        mb=merge(all_iv); mbnon=merge(non_iv); mbar=merge(ar_iv)
        rows.append((span,mb,mbnon,mbar,span-mb,len(b)))
    rows=rows[2:] if len(rows)>4 else rows   # drop warmup
    if not rows: continue
    g=lambda i: med([r[i] for r in rows])/1e6
    print(f"\n  N={nt} (n={len(rows)}): span={g(0):.2f}ms  busy_all={g(1):.2f}  busy_nonAR={g(2):.2f}  "
          f"AR={g(3):.2f}  idle={g(4):.2f}  nkern={med([r[5] for r in rows]):.0f}")
    print(f"     CLOSURE busy_nonAR+AR+idle = {g(2)+g(3)+g(4):.2f} vs span {g(0):.2f}")
con.close()

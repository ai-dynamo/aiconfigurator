"""Discriminate WHY golden decode escapes the TP all-reduce chain that dominates
golden prefill. Re-analyzes EXISTING serve_nsys_trace (real tp4, vLLM 0.20.1).

Segments forward steps by clustering all-reduce kernels (gap>1.5ms). Per step reports
launch API (eager cudaLaunchKernel vs in-graph cudaGraphLaunch), per-call us, count,
summed ms, and compute busy (token-size proxy).
"""
import sqlite3, numpy as np
from collections import Counter

PATH = "runs/serve_nsys_trace.sqlite"
P0 = 284221356638208; MASK = -16777216; PROBE_T = 124e9
con = sqlite3.connect(f"file:{PATH}?mode=ro", uri=True); cur = con.cursor()
sid = dict(cur.execute("SELECT id,value FROM StringIds").fetchall())
def ids(sub): return set(i for i, v in sid.items() if sub in v)
CDR = ids("cross_device_reduce_2stage"); TFA = ids("triton_poi_fused_add_all_reduce")
GLAUNCH = ids("cudaGraphLaunch"); ELAUNCH = ids("cudaLaunchKernel") | ids("cuLaunchKernel")
t0 = cur.execute("SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
lo = t0 + PROBE_T
rt = {}
for cid, nm in cur.execute("SELECT correlationId,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME "
                           "WHERE (globalTid&?)=? AND start>=?", (MASK, P0, lo)).fetchall():
    rt[cid] = "GRAPH" if nm in GLAUNCH else ("EAGER" if nm in ELAUNCH else "OTH")

# all kernels in probe region (for compute-busy per step span)
allk = cur.execute("SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid=? AND start>=? ORDER BY start",
                   (P0, lo)).fetchall()
allk = np.array(allk)

def cluster(kind_ids, label):
    rows = cur.execute(f"SELECT start,end,correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL "
                       f"WHERE globalPid=? AND start>=? AND shortName IN ({','.join(map(str,kind_ids))}) ORDER BY start",
                       (P0, lo)).fetchall()
    steps=[]; cs=[rows[0]]
    for i in range(1,len(rows)):
        if rows[i][0]-rows[i-1][1] > 1_500_000: steps.append(cs); cs=[]
        cs.append(rows[i])
    steps.append(cs)
    out=[]
    for st in steps:
        if len(st) < 20: continue   # ignore fragments; keep ~full chains
        d=np.array([e-s for s,e,_ in st])/1e3
        api=Counter(rt.get(c,"?") for *_,c in st)
        s0,e1=st[0][0],st[-1][1]
        # compute busy in span = all-kernel busy minus this AR busy
        m=(allk[:,0]>=s0)&(allk[:,0]<e1)
        busy=(allk[m,1]-allk[m,0]).sum()
        compute=busy-d.sum()*1e3
        out.append(dict(label=label,t=(s0-t0)/1e9,n=len(st),med=np.median(d),p25=np.percentile(d,25),
                        p75=np.percentile(d,75),summ=d.sum()/1e3,span=(e1-s0)/1e3,
                        compute=compute/1e3,api=dict(api)))
    return out

cdr_steps = cluster(CDR, "cross_device_2stage")
tfa_steps = cluster(TFA, "triton_fused_AR")
print(f"=== cross_device_2stage steps: {len(cdr_steps)} ===")
print(f"{'t_s':>7}{'nAR':>5}{'med_us':>8}{'p25':>7}{'p75':>8}{'sum_ms':>8}{'span_ms':>8}{'comp_ms':>8}  api")
for s in cdr_steps:
    print(f"{s['t']:>7.1f}{s['n']:>5}{s['med']:>8.1f}{s['p25']:>7.1f}{s['p75']:>8.1f}{s['summ']:>8.2f}{s['span']:>8.1f}{s['compute']:>8.1f}  {s['api']}")
print(f"\n=== triton_fused_AR steps: {len(tfa_steps)} ===")
print(f"{'t_s':>7}{'nAR':>5}{'med_us':>8}{'p25':>7}{'p75':>8}{'sum_ms':>8}{'span_ms':>8}{'comp_ms':>8}  api")
for s in tfa_steps:
    print(f"{s['t']:>7.1f}{s['n']:>5}{s['med']:>8.1f}{s['p25']:>7.1f}{s['p75']:>8.1f}{s['summ']:>8.2f}{s['span']:>8.1f}{s['compute']:>8.1f}  {s['api']}")

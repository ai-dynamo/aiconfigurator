"""Reconcile comm-bound vs compute-bound for golden <=512 prefill.
Splits serve_nsys_trace steps into CAPTURED (AR graphId set) vs EAGER (graphId null),
recovers token size via triton-fused-AR gridX, and decomposes each step's span into:
  AR-floor (synced, p25 x count) | AR-spin (summed - floor) | compute-busy | idle-gap.
"""
import sqlite3, numpy as np
PATH="runs/serve_nsys_trace.sqlite"; P0=284221356638208; MASK=-16777216
con=sqlite3.connect(f"file:{PATH}?mode=ro",uri=True); cur=con.cursor()
sid=dict(cur.execute("SELECT id,value FROM StringIds").fetchall())
def ids(s): return set(i for i,v in sid.items() if s in v)
CDR=ids("cross_device_reduce_2stage"); TFA=ids("triton_poi_fused_add_all_reduce")
t0=cur.execute("SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
lo=t0+124e9
# all kernels in probe region with graphId + shortName
ks=cur.execute("SELECT start,end,shortName,graphId,gridX FROM CUPTI_ACTIVITY_KIND_KERNEL "
               "WHERE globalPid=? AND start>=? ORDER BY start",(P0,lo)).fetchall()
allk=np.array([(s,e) for s,e,*_ in ks])
# segment steps by clustering cross_device AR (gap>1.5ms)
ar=[(s,e,g) for s,e,sn,g,gx in ks if sn in CDR]
steps=[]; cs=[ar[0]]
for i in range(1,len(ar)):
    if ar[i][0]-ar[i-1][1]>1_500_000: steps.append(cs); cs=[]
    cs.append(ar[i])
steps.append(cs)
rows=[]
for st in steps:
    if len(st)<40: continue
    s0,e1=st[0][0],st[-1][1]
    d=np.array([e-s for s,e,_ in st])/1e3
    cap = sum(1 for *_,g in st if g)/len(st)   # fraction in-graph
    # token size from triton AR gridX in this window
    gx=[gx for s,e,sn,g,gx in ks if sn in TFA and s0<=s<e1]
    tok=int(np.median(gx)) if gx else 0
    m=(allk[:,0]>=s0)&(allk[:,0]<e1)
    busy=(allk[m,1]-allk[m,0]).sum()/1e3
    ar_sum=d.sum(); compute=busy-ar_sum
    floor=np.percentile(d,25)*len(d); spin=ar_sum-floor
    rows.append(dict(t=(s0-t0)/1e9,cap=cap,tok=tok,n=len(st),
                     med=np.median(d),p25=np.percentile(d,25),mean=d.mean(),
                     ar_sum=ar_sum,floor=floor,spin=spin,compute=compute,
                     span=(e1-s0)/1e3,busy=busy,idle=(e1-s0)/1e3-busy))
# group by capture status, drop first 3 (cold) per group
cap=[r for r in rows if r["cap"]>0.5]; eag=[r for r in rows if r["cap"]<=0.5]
def med(g,k): return np.median([r[k] for r in g]) if g else 0
def show(name,g):
    g=g[3:] if len(g)>6 else g   # drop cold warmup
    sp=med(g,"span"); k=1e3      # us -> ms
    print(f"\n== {name}  (n_steps={len(g)}, ~{int(med(g,'tok'))//2} tokens (triton gridX={int(med(g,'tok'))}), AR in-graph={med(g,'cap')*100:.0f}%) ==")
    print(f"  span/wall (nsys)   : {sp/k:8.2f} ms")
    print(f"  all-reduce count   : {med(g,'n'):8.0f} /step")
    print(f"  AR per-call us     : median={med(g,'med'):.1f}  p25(floor)={med(g,'p25'):.1f}  mean={med(g,'mean'):.1f}")
    print(f"  AR summed          : {med(g,'ar_sum')/k:8.2f} ms  ({100*med(g,'ar_sum')/sp:4.0f}% of span)")
    print(f"    - AR floor(synced): {med(g,'floor')/k:8.2f} ms  ({100*med(g,'floor')/sp:4.0f}%)")
    print(f"    - AR spin-wait    : {med(g,'spin')/k:8.2f} ms  ({100*med(g,'spin')/sp:4.0f}%)")
    print(f"  compute busy       : {med(g,'compute')/k:8.2f} ms  ({100*med(g,'compute')/sp:4.0f}% of span)")
    print(f"  idle launch-gap    : {med(g,'idle')/k:8.2f} ms  ({100*med(g,'idle')/sp:4.0f}% of span)")
show("CAPTURED prefill (full cudagraph)",cap)
show("EAGER prefill (piecewise, AR eager)",eag)

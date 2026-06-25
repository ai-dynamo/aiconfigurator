import sqlite3,re,sys
from collections import defaultdict
P=sys.argv[1]
c=sqlite3.connect('file:%s?mode=ro'%P,uri=True);cur=c.cursor()
MASK=-16777216
sid=dict(cur.execute("SELECT id,value FROM StringIds").fetchall())
NCCL=re.compile(r'nccl|all.?reduce|all.?gather|reduce.?scatter|all2all|multimem',re.I)
def is_eager(n):return n.startswith(('cudaLaunchKernel','cuLaunchKernel','cudaLaunchKernelExC'))
def is_graph(n):return n.startswith('cudaGraphLaunch')
steps=cur.execute("SELECT text,start,end,globalTid FROM NVTX_EVENTS WHERE text LIKE 'bench_step::%' ORDER BY start").fetchall()
pids=sorted(set(tid&MASK for *_,tid in steps)); rank={p:i for i,p in enumerate(pids)}
# pick last bs1/past0 window per (ntok,pid)
want={}
for t,s,e,tid in steps:
    m=re.search(r'N(\d+)::bs(\d+)::past(\d+)',t)
    nt,bs,pa=int(m.group(1)),int(m.group(2)),int(m.group(3))
    if bs!=1 or pa!=0: continue
    want[(nt,tid&MASK)]=(s,e)   # last wins
print(f"{'ntok':>5} {'rank':>4} {'eager':>6} {'graph':>6} {'nccl':>6} {'busy_us':>8} {'span_us':>8} {'nsysWall_us':>11} {'kern':>7}")
for (nt,pid) in sorted(want):
    s,e=want[(nt,pid)]
    rt=cur.execute("SELECT R.nameId,COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME R WHERE R.start>=? AND R.start<? AND (R.globalTid&?)=? GROUP BY R.nameId",(s,e,MASK,pid)).fetchall()
    eager=sum(c for n,c in rt if is_eager(sid.get(n,'')))
    graph=sum(c for n,c in rt if is_graph(sid.get(n,'')))
    ks=cur.execute("SELECT K.start,K.end,K.shortName FROM CUPTI_ACTIVITY_KIND_KERNEL K WHERE K.globalPid=? AND K.start>=? AND K.start<? ORDER BY K.start",(pid,s,e)).fetchall()
    busy=sum(b-a for a,b,_ in ks); span=(ks[-1][1]-ks[0][0]) if ks else 0
    nccl=sum(1 for a,b,n in ks if NCCL.search(sid.get(n,'')))
    print(f"{nt:>5} {rank[pid]:>4} {eager:>6} {graph:>6} {nccl:>6} {busy/1e3:>8.0f} {span/1e3:>8.0f} {(e-s)/1e3:>11.0f} {len(ks):>7}")

import sys; sys.path.insert(0,"/workspace/repo/aiconfigurator")
import csv, statistics
from pathlib import Path
sys.argv=['x']
import tools.plot_fpm_vs_aic as P
from aiconfigurator.sdk.perf_database import PerfDatabase
from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend
import aiconfigurator.sdk.backends.vllm_backend as vb
from aiconfigurator.sdk.config import RuntimeConfig
from collector.layerwise.diagnostics.compare_aic_layerwise_fpm import _LayerwiseDatabase

# FIXED backbone: local clean gen collection (latency_source=execute_model_gpu, single
# source, monotonic in bs, includes_moe=False). The committed systems layerwise_perf.csv
# decode rows are the OLD 'span' representative-module timing the validator rejects, so the
# clean backbone is the local cleanctx4 run. Overlay (module-level fused MoE: moe_module_level
# =True) + comm tables come from the committed systems root (matches STEP 0 sanity).
SYS='src/aiconfigurator/systems'
LW=Path('runs/layerwise_qwen36_tp4ep4_cleanctx4/layerwise_native_tagged128.csv')
real=PerfDatabase('b300_sxm','vllm','0.20.1',systems_root=SYS)
db=_LayerwiseDatabase(LW, real)
backend=VLLMBackend(); vb._USE_LAYERWISE=True
rc=RuntimeConfig(vllm_max_num_batched_tokens=2048, vllm_max_num_seqs=128)
model=P._make_model('Qwen/Qwen3.6-35B-A3B',4,1,4)
print("moe_inter:", model._moe_inter_size, " num_layers:", model._num_layers)

GD='fpm_golden_runs/fpm_upfront_qwen36_moe_full_once_20260613_201336/tp4_ep4_past4096/fpm_metrics_phase.csv'
gd=[r for r in csv.DictReader(open(GD)) if r['phase']=='decode']
def gmed(bs):
    # kv~4096; widen the window only if the discrete golden kv grid for this bs has no
    # point near 4096 (e.g. bs=16, a sparsely-sampled batch in the golden trace).
    for lo,hi in [(3800,4300),(3500,4600),(3000,5100)]:
        v=[float(r['latency_ms']) for r in gd
           if int(r['decode_requests'])==bs and lo<=float(r['mean_decode_kv_tokens'])<=hi
           and float(r['latency_ms'])<200]
        if v:
            return statistics.median(v)
    return None

PAST=4096
rows=[]
print(f"\n{'bs':>3} | {'backbone':>9} | {'overlay':>8} | {'composed':>9} | {'golden':>7} | {'ratio':>5} | {'routed*40':>9}")
print("-"*72)
ratios=[]; gratios=[]
for bs in [1,2,4,8,16]:
    lat,_,_=backend._get_decode_step_latency(model, db, rc, batch_size=bs, past_kv=PAST)
    d=dict(lat)
    backbone = d.get('generation_layerwise',0.0)+d.get('generation_tp_allreduce',0.0)
    routed   = d.get('generation_moe',0.0)
    router   = d.get('generation_moe_router',0.0)
    dispatch = d.get('generation_moe_dispatch',0.0)
    shared   = d.get('generation_moe_shared_expert',0.0)
    ep_a2a   = d.get('generation_moe_ep_alltoall',0.0)
    moe_tp   = d.get('generation_moe_tp_allreduce',0.0)
    overlap  = d.get('generation_moe_shared_expert_overlap',0.0)
    overlay  = routed+router+dispatch+shared+ep_a2a+moe_tp+overlap
    composed = sum(d.values())
    # Guard removal: when moe_module_level=True the fused 'routed' term already contains
    # router/top-k/gather/scatter, so router+dispatch are double-counted on decode (the
    # is_context guard at vllm_backend.py:1076 blocks their deletion only on prefill).
    composed_g = composed - router - dispatch
    golden   = gmed(bs)
    ratio    = (composed/golden) if golden else None
    ratio_g  = (composed_g/golden) if golden else None
    if ratio: ratios.append(abs(composed-golden)/golden)
    if ratio_g: gratios.append(abs(composed_g-golden)/golden)
    rows.append(dict(bs=bs,backbone=backbone,routed=routed,router=router,dispatch=dispatch,
                     shared=shared,ep_a2a=ep_a2a,moe_tp=moe_tp,overlap=overlap,
                     overlay=overlay,composed=composed,composed_g=composed_g,
                     golden=golden,ratio=ratio,ratio_g=ratio_g))
    print(f"{bs:3d} | {backbone:9.3f} | {overlay:8.3f} | {composed:9.3f} | "
          f"{(golden if golden else float('nan')):7.3f} | {(ratio if ratio else float('nan')):5.2f} | {routed:9.4f}")

mape=100*statistics.mean(ratios) if ratios else float('nan')
mape_g=100*statistics.mean(gratios) if gratios else float('nan')
print(f"\nDecode MAPE (current, router+dispatch added): {mape:.1f}%")
print(f"\n--- guard-removed: router+dispatch deleted on decode (moe_module_level) ---")
print(f"{'bs':>3} | {'composed_g':>10} | {'golden':>7} | {'ratio_g':>7} | {'r+d saved':>9}")
for r in rows:
    rd=r['router']+r['dispatch']
    print(f"{r['bs']:3d} | {r['composed_g']:10.3f} | {(r['golden'] if r['golden'] else float('nan')):7.3f} | "
          f"{(r['ratio_g'] if r['ratio_g'] else float('nan')):7.2f} | {rd:9.3f}")
print(f"\nDecode MAPE (guard removed): {mape_g:.1f}%")
print("\n--- overlay component breakdown (ms/step, full-model x40) ---")
print(f"{'bs':>3} | {'routed':>7} | {'router':>7} | {'dispatch':>8} | {'shared':>7} | {'ep_a2a':>7} | {'moe_tp':>7} | {'overlap':>7}")
for r in rows:
    print(f"{r['bs']:3d} | {r['routed']:7.4f} | {r['router']:7.4f} | {r['dispatch']:8.4f} | "
          f"{r['shared']:7.4f} | {r['ep_a2a']:7.4f} | {r['moe_tp']:7.4f} | {r['overlap']:7.4f}")

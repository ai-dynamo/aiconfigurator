import sys; sys.path.insert(0, "/workspace/repo/aiconfigurator")
import sys, csv, statistics
from pathlib import Path
sys.argv=['x']
import tools.plot_fpm_vs_aic as P
from collector.layerwise.diagnostics.compare_aic_layerwise_fpm import _LayerwiseDatabase
from aiconfigurator.sdk.perf_database import PerfDatabase
from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend
from aiconfigurator.sdk import common
import aiconfigurator.sdk.backends.vllm_backend as vb
from aiconfigurator.sdk.config import RuntimeConfig

lw = Path('runs/layerwise_qwen36_tp4ep4_cleanctx4/layerwise_tagged128.csv')
model_name='Qwen/Qwen3.6-35B-A3B'; model=P._make_model(model_name,4,1,4)
# build overlay systems root (MoE)
out=Path('runs/cleanctx4_charts'); out.mkdir(parents=True,exist_ok=True)
sr=P._prepare_moe_overlay_systems_root(systems_root='src/aiconfigurator/systems',
     moe_perf_file=Path('collector/layerwise/wip/moe_perf.txt'), output=out/'overlay.csv')
real=PerfDatabase('b300_sxm','vllm','0.20.1',systems_root=sr)
db=_LayerwiseDatabase(lw, real, repair_decode_kv_above=None, repair_decode_models=(), repair_decode_anchor_kvs=(2048,4096))
backend=VLLMBackend()
rc=RuntimeConfig(vllm_max_num_batched_tokens=2048, vllm_max_num_seqs=128)
vb._USE_LAYERWISE=True
# AIC ctx (with MoE overlay) at past_kv=0
print("AIC ctx (clean, +MoE overlay), past_kv=0:")
for nt in [128,256,512,1024,2048]:
    v=P._aic_ctx(backend, model, db, rc, nt, 0)
    print(f"  new={nt:5d}  AIC={v:.2f}ms" if v else f"  new={nt}: None")
# golden context FPM medians
rows=list(csv.DictReader(open('fpm_golden_runs/fpm_upfront_qwen36_moe_full_once_20260613_201336/tp4_ep4_past4096/fpm_metrics_phase.csv')))
ctx=[r for r in rows if r['phase']=='context']
import collections
b=collections.defaultdict(list)
for r in ctx: b[int(r['ctx_tokens'])].append(float(r['latency_ms']))
print("golden context FPM medians:")
for c in sorted(b):
    med=statistics.median(b[c])
    if med<150: print(f"  ctx_tokens={c:5d}  n={len(b[c]):2d}  golden_med={med:.2f}ms")

import sys; sys.path.insert(0,"/workspace/repo/aiconfigurator")
import csv, statistics, collections
from pathlib import Path
sys.argv=['x']
import tools.plot_fpm_vs_aic as P
from collector.layerwise.diagnostics.compare_aic_layerwise_fpm import _LayerwiseDatabase
from aiconfigurator.sdk.perf_database import PerfDatabase
from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend
import aiconfigurator.sdk.backends.vllm_backend as vb
from aiconfigurator.sdk.config import RuntimeConfig

lw=Path('runs/layerwise_qwen36_tp4ep4_cleanctx4/layerwise_native_tagged128.csv')
out=Path('runs/dec_charts'); out.mkdir(parents=True,exist_ok=True)
sr=P._prepare_moe_overlay_systems_root(systems_root='src/aiconfigurator/systems',
     moe_perf_file=Path('collector/layerwise/wip/moe_perf.txt'), output=out/'overlay.csv')
real=PerfDatabase('b300_sxm','vllm','0.20.1',systems_root=sr)
db=_LayerwiseDatabase(lw, real, repair_decode_kv_above=8192, repair_decode_models=('Qwen/Qwen3.6-35B-A3B',), repair_decode_anchor_kvs=(2048,4096))
backend=VLLMBackend(); vb._USE_LAYERWISE=True
rc=RuntimeConfig(vllm_max_num_batched_tokens=2048, vllm_max_num_seqs=128)
model=P._make_model('Qwen/Qwen3.6-35B-A3B',4,1,4)

# golden decode medians by batch, kv~2048 and ~4096
GD='fpm_golden_runs/fpm_upfront_qwen36_moe_full_once_20260613_201336/tp4_ep4_past4096/fpm_metrics_phase.csv'
gd=[r for r in csv.DictReader(open(GD)) if r['phase']=='decode']
def gmed(bs,lo,hi):
    v=[float(r['latency_ms']) for r in gd if int(r['decode_requests'])==bs and lo<=float(r['mean_decode_kv_tokens'])<=hi and float(r['latency_ms'])<200]
    return statistics.median(v) if v else None

print("bs | past | AIC_decode(composed,+MoE) | golden_full | ratio")
for past,lo,hi in [(2048,1800,2300),(4096,3800,4300)]:
    for bs in [1,2,4,8,16]:
        aic=P._aic_gen(backend, model, db, rc, bs, past, comm=True)
        g=gmed(bs,lo,hi)
        r = (aic/g) if (aic and g) else None
        print(f"{bs:3d}|{past:5d}| {('%.3f'%aic) if aic else 'None':>8} | {('%.3f'%g) if g else 'NA':>8} | {('%.2f'%r) if r else 'NA'}")

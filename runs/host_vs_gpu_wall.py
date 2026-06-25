"""Decisive test for the small-prefill gap mechanism: on ONE warm engine, measure
the SAME <=512-tok prefill step two ways:
  HOST wall  = perf_counter around execute_model, NO cuda sync (what golden's FPM
               wall_time ~ per-iteration host loop time measures; pipelined-friendly)
  GPU wall   = cuda-event around execute_model WITH synchronize before+after (what the
               collector's execute_model_gpu measures; exposes inter-kernel launch gaps)
If HOST << GPU and HOST ~ golden's 16ms, the gap is a measurement/pipelining effect
(launch tax issued in ~HOST ms; GPU drains it over ~GPU ms incl. idle gaps), NOT
capture coverage.
"""
import os, statistics, time
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import torch
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.v1.worker import gpu_model_runner as gmr

CFG = "runs/layerwise_qwen36_tp4ep4_cleanctx4/config_cache/Qwen_Qwen3.6-35B-A3B_fd9719118c872c9fb3d947d5"
N = 20

_orig = gmr.GPUModelRunner.execute_model
mode = {"v": "off"}
host_ms, gpu_ms, ntok_seen = [], [], []
def timed(self, scheduler_output, intermediate_tensors=None):
    n = getattr(scheduler_output, "total_num_scheduled_tokens", None)
    if mode["v"] == "host":           # pure host wall, NO sync
        t0 = time.perf_counter()
        r = _orig(self, scheduler_output, intermediate_tensors)
        host_ms.append((n, (time.perf_counter() - t0) * 1000.0))
    elif mode["v"] == "gpu":          # GPU wall, sync before+after
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        r = _orig(self, scheduler_output, intermediate_tensors)
        torch.cuda.synchronize(); e.record(); e.synchronize()
        gpu_ms.append((n, s.elapsed_time(e)))
    else:
        r = _orig(self, scheduler_output, intermediate_tensors)
    return r
gmr.GPUModelRunner.execute_model = timed

llm = LLM(model=CFG, load_format="dummy", tensor_parallel_size=1,
          max_num_batched_tokens=2048, max_num_seqs=128, gpu_memory_utilization=0.9,
          enforce_eager=False, trust_remote_code=True,
          enable_prefix_caching=False, limit_mm_per_prompt={"image": 0, "video": 0})
sp = SamplingParams(max_tokens=1, temperature=0.0)
def run(ntok):
    llm.generate(prompts=[TokensPrompt(prompt_token_ids=list(range(ntok)))], sampling_params=sp, use_tqdm=False)

for ntok in (256, 512):
    mode["v"] = "off"
    for _ in range(3): run(ntok)          # warm (capture + replay)
    mode["v"] = "host"; host_ms.clear()
    for _ in range(N): run(ntok)
    mode["v"] = "gpu"; gpu_ms.clear()
    for _ in range(N): run(ntok)
    h = [m for n, m in host_ms if n and n >= ntok - 8]
    g = [m for n, m in gpu_ms if n and n >= ntok - 8]
    print(f"\n=== {ntok}-tok warm prefill (1-GPU sharded, no marker) ===")
    print(f"  HOST wall (no sync): median={statistics.median(h):.1f}ms  min={min(h):.1f}  [{', '.join(f'{x:.1f}' for x in h[:6])} ...]")
    print(f"  GPU  wall (sync):    median={statistics.median(g):.1f}ms  min={min(g):.1f}  [{', '.join(f'{x:.1f}' for x in g[:6])} ...]")
print("\n(collector execute_model_gpu=33ms; golden FPM wall_time <=512 ~16ms)")

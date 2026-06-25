"""STEP 1 cold-vs-warm: per-step execute_model GPU time for <=512-tok standalone
prefill on the collector's sharded config, plain vLLM (no marker, golden's own code).
Tests C1: does the warm/replayed step (step>=3) drop toward golden's ~16ms?
"""
import os, sys, statistics
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # run EngineCore in-process so patch applies
import torch
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.v1.worker import gpu_model_runner as gmr

CFG = "runs/layerwise_qwen36_tp4ep4_cleanctx4/config_cache/Qwen_Qwen3.6-35B-A3B_fd9719118c872c9fb3d947d5"
NSTEPS = int(os.environ.get("NSTEPS", "25"))

_orig = gmr.GPUModelRunner.execute_model
times = []
def timed(self, scheduler_output, intermediate_tensors=None):
    n = getattr(scheduler_output, "total_num_scheduled_tokens", None)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    r = _orig(self, scheduler_output, intermediate_tensors)
    torch.cuda.synchronize(); e.record(); e.synchronize()
    times.append((n, s.elapsed_time(e)))
    return r
gmr.GPUModelRunner.execute_model = timed

llm = LLM(model=CFG, load_format="dummy", tensor_parallel_size=1,
          max_num_batched_tokens=2048, max_num_seqs=128, gpu_memory_utilization=0.9,
          enforce_eager=False, trust_remote_code=True,
          enable_prefix_caching=False, limit_mm_per_prompt={"image": 0, "video": 0})
sp = SamplingParams(max_tokens=1, temperature=0.0)

for ntok in (256, 512):
    times.clear()
    for i in range(NSTEPS):
        llm.generate(prompts=[TokensPrompt(prompt_token_ids=list(range(ntok)))],
                     sampling_params=sp, use_tqdm=False)
    pre = [ms for n, ms in times if n and n >= ntok - 8]
    print(f"\n=== {ntok}-tok prefill, per-step execute_model GPU (ms), {len(pre)} steps ===")
    print("  ".join(f"{x:.1f}" for x in pre))
    if len(pre) >= 5:
        print(f"  step1(cold)={pre[0]:.1f}  step2={pre[1]:.1f}  warm(step>=3 median)={statistics.median(pre[2:]):.1f}  warm_min={min(pre[2:]):.1f}")
print("\n(collector cleanctx4 256-tok execute_model=33.1ms; golden warm <=512 ~16ms)")

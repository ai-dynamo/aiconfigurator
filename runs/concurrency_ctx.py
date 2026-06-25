"""Test pipelining hypothesis: golden serves a continuous stream (async scheduling +
continuous batching); the collector measures ISOLATED single prefills. Submit a STREAM
of 256-tok prefills (one prefill per scheduler iteration via small token budget) and
measure the amortized steady-state per-iteration wall vs the isolated step.

If amortized per-step ~16ms while isolated ~33ms -> pipelining/throughput is the gap
(launch tax overlaps across steps). If still ~33ms -> not pipelining.
"""
import os, statistics, time
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import torch
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.v1.worker import gpu_model_runner as gmr

CFG = "runs/layerwise_qwen36_tp4ep4_cleanctx4/config_cache/Qwen_Qwen3.6-35B-A3B_fd9719118c872c9fb3d947d5"

_orig = gmr.GPUModelRunner.execute_model
rec = {"on": False}
host_walls = []
def timed(self, scheduler_output, intermediate_tensors=None):
    n = getattr(scheduler_output, "total_num_scheduled_tokens", None)
    if rec["on"]:
        t0 = time.perf_counter()
        r = _orig(self, scheduler_output, intermediate_tensors)
        host_walls.append((n, (time.perf_counter() - t0) * 1000.0))
        return r
    return _orig(self, scheduler_output, intermediate_tensors)
gmr.GPUModelRunner.execute_model = timed

# token budget = 256 -> ~one 256-tok prefill per scheduler iteration (like golden's small-ctx rows)
llm = LLM(model=CFG, load_format="dummy", tensor_parallel_size=1,
          max_num_batched_tokens=256, max_num_seqs=128, gpu_memory_utilization=0.9,
          enforce_eager=False, trust_remote_code=True,
          enable_prefix_caching=False, limit_mm_per_prompt={"image": 0, "video": 0})
sp = SamplingParams(max_tokens=1, temperature=0.0)
mk = lambda: TokensPrompt(prompt_token_ids=list(range(256)))

# warm
llm.generate(prompts=[mk() for _ in range(8)], sampling_params=sp, use_tqdm=False)

# ISOLATED: one request at a time (collector-like)
rec["on"] = True; host_walls.clear()
t0 = time.perf_counter()
for _ in range(16):
    llm.generate(prompts=[mk()], sampling_params=sp, use_tqdm=False)
iso_wall = (time.perf_counter() - t0) * 1000.0 / 16
iso_steps = [m for n, m in host_walls if n and n >= 248]
print(f"ISOLATED (1 req/call x16): amortized wall/req={iso_wall:.1f}ms  per-step host median={statistics.median(iso_steps):.1f}ms  n={len(iso_steps)}")

# STREAM: many requests submitted together (continuous batching, async pipelining)
rec["on"] = True; host_walls.clear()
NREQ = 128
t0 = time.perf_counter()
llm.generate(prompts=[mk() for _ in range(NREQ)], sampling_params=sp, use_tqdm=False)
stream_wall = (time.perf_counter() - t0) * 1000.0 / NREQ
ctx_steps = [m for n, m in host_walls if n and n >= 248]
print(f"STREAM ({NREQ} reqs 1 call): amortized wall/req={stream_wall:.1f}ms")
if ctx_steps:
    print(f"  per-iteration host wall (ctx steps): median={statistics.median(ctx_steps):.1f}ms min={min(ctx_steps):.1f} n={len(ctx_steps)}")
    print(f"  [{', '.join(f'{x:.1f}' for x in ctx_steps[:10])} ...]")
print("\n(collector execute_model_gpu=33ms isolated; golden FPM <=512 ~16ms)")

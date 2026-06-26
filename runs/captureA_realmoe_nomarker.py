"""Trace A: 1-GPU sharded, REAL MoE, NO per-layer marker (plain-vLLM style, ~34.7ms).

Same engine config as runs/standalone_ctx.py (plain vLLM, no layerwise patch, no
moe-noop hook, full real MoE forward). The ONLY instrumentation is a single coarse
NVTX range around each execute_model call so analyze_launch_gap.py can slice the
256-tok prefill step out of the nsys trace. This is a STEP delimiter, not the
collector's per-layer bench marker -- the forward runs exactly as plain vLLM.

Run under: nsys profile --trace=cuda,nvtx --cuda-graph-trace=node ... python this
"""
import os, time, statistics
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
# Run engine core in-process (InprocClient) so this monkeypatch applies to the
# worker and nsys sees a single process. Required because nsys initializes CUDA
# early, which would otherwise force vLLM onto the spawn multiproc path.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import torch
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.v1.worker import gpu_model_runner as gmr

CFG = "runs/layerwise_qwen36_tp4ep4_cleanctx4/config_cache/Qwen_Qwen3.6-35B-A3B_fd9719118c872c9fb3d947d5"

_orig = gmr.GPUModelRunner.execute_model
times = []

def timed(self, scheduler_output, intermediate_tensors=None):
    n = getattr(scheduler_output, "total_num_scheduled_tokens", None)
    # coarse cuda-event wall (matches standalone_ctx.py methodology)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    # single step-level NVTX range (NOT per-layer) for nsys slicing
    if n and n >= 200:
        torch.cuda.nvtx.range_push(f"bench_step::N{n}::bs1::past0")
    s.record()
    r = _orig(self, scheduler_output, intermediate_tensors)
    torch.cuda.synchronize(); e.record(); e.synchronize()
    if n and n >= 200:
        torch.cuda.nvtx.range_pop()
    ms = s.elapsed_time(e)
    times.append((n, ms))
    return r

gmr.GPUModelRunner.execute_model = timed

llm = LLM(model=CFG, load_format="dummy", tensor_parallel_size=1,
          max_num_batched_tokens=2048, max_num_seqs=128, gpu_memory_utilization=0.9,
          enforce_eager=False, trust_remote_code=True,
          enable_prefix_caching=False, limit_mm_per_prompt={"image": 0, "video": 0})
sp = SamplingParams(max_tokens=1, temperature=0.0)

# warmup (markers suppressed since these are also prefills; we only keep the last steady runs)
for warm in range(2):
    llm.generate(prompts=[TokensPrompt(prompt_token_ids=list(range(256)))], sampling_params=sp, use_tqdm=False)
times.clear()
# steady-state measured prefills (each emits one bench_step NVTX range)
torch.cuda.profiler.start()
for i in range(6):
    llm.generate(prompts=[TokensPrompt(prompt_token_ids=list(range(256)))], sampling_params=sp, use_tqdm=False)
torch.cuda.profiler.stop()

pre = [ms for n, ms in times if n and n >= 200]
print("TRACE-A 256-tok prefill execute_model GPU times (ms):", [f"{x:.2f}" for x in pre])
if pre:
    print("median execute_model_gpu:", round(statistics.median(pre), 2))
print("(plain vLLM standalone = 34.7ms; collector cleanctx4 = 33.1ms; golden ~16ms)")

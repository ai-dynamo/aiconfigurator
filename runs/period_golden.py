"""Un-profiled (no nsys) isolated-vs-stream wall for golden tp4/ep4 real weights.
Resolves contradiction (C): is the ~16ms <=512 prefill an ISOLATED single-step wall,
or a SERVING/stream (occupancy-filled, pipelined) per-step throughput that an isolated
1-step measurement cannot reproduce?

Reports, per new-token size, on the SAME warm engine:
  ISOLATED per-step wall  = wall of generate([1 prompt]) (1 prefill = 1 engine step)
  STREAM   per-req wall    = wall of generate([NREQ prompts]) / NREQ
  STREAM   per-step wall   = same total / n_steps   (n_steps from token-budget packing)
All wall-clock (perf_counter), NO profiler, NO cuda sync injected.

env: TP(4) MNBT(2048) NREQ(128) MODEL(real snapshot) WEIGHTS(real|dummy)
"""
import os, time, statistics, math, sys
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
from vllm import LLM, SamplingParams, TokensPrompt
import torch  # for NVTX markers under nsys (no-op cost when not profiled)

TP   = int(os.environ.get("TP", "4"))
MNBT = int(os.environ.get("MNBT", "2048"))
NREQ = int(os.environ.get("NREQ", "128"))
WEIGHTS = os.environ.get("WEIGHTS", "real")
STREAM_ONLY = os.environ.get("STREAM_ONLY", "0") == "1"   # skip isolated phase -> clean nsys stream trace
ISO_ONLY    = os.environ.get("ISO_ONLY", "0") == "1"
MODEL = os.environ.get("MODEL",
    "/workspace/models/hf_home/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0")

kw = dict(model=MODEL, tensor_parallel_size=TP, enable_expert_parallel=(TP > 1),
          max_num_batched_tokens=MNBT, max_num_seqs=128, gpu_memory_utilization=0.9,
          enforce_eager=False, trust_remote_code=True, enable_prefix_caching=False,
          limit_mm_per_prompt={"image": 0, "video": 0})
if WEIGHTS == "dummy":
    kw["load_format"] = "dummy"

def main():
  print(f"== period_golden TP={TP} MNBT={MNBT} NREQ={NREQ} weights={WEIGHTS} ==", flush=True)
  llm = LLM(**kw)
  sp = SamplingParams(max_tokens=1, temperature=0.0)
  mk = lambda nt: TokensPrompt(prompt_token_ids=list(range(nt)))

  sizes = tuple(int(x) for x in os.environ.get("NT", "256,512").split(","))
  for nt in sizes:
    per_step_budget = max(1, MNBT // nt)          # prefills packed per step (token-bound)
    n_steps = math.ceil(NREQ / per_step_budget)
    # warm (capture + replay)
    llm.generate([mk(nt) for _ in range(per_step_budget * 2)], sp, use_tqdm=False)
    print(f"\n[{nt}-tok]  (packing {per_step_budget} prefill/step, n_steps~{n_steps})", flush=True)
    # ISOLATED: 1 prompt per generate -> 1 prefill step
    if not STREAM_ONLY:
        iso = []
        for _ in range(20):
            t0 = time.perf_counter(); llm.generate([mk(nt)], sp, use_tqdm=False)
            iso.append((time.perf_counter() - t0) * 1000.0)
        iso.sort()
        print(f"  ISOLATED per-step wall : median={iso[len(iso)//2]:.1f}ms  min={iso[0]:.1f}  "
              f"[{', '.join(f'{x:.1f}' for x in iso[:6])}]", flush=True)
    # STREAM: NREQ prompts in one call -> continuous batching, GPU stays filled
    if not ISO_ONLY:
        torch.cuda.nvtx.range_push(f"STREAMPHASE_{nt}")
        t0 = time.perf_counter(); llm.generate([mk(nt) for _ in range(NREQ)], sp, use_tqdm=False)
        tot = (time.perf_counter() - t0) * 1000.0
        torch.cuda.nvtx.range_pop()
        print(f"  STREAM   per-req  wall : {tot/NREQ:.2f}ms   (total {tot:.0f}ms / {NREQ} req)", flush=True)
        print(f"  STREAM   per-step wall : {tot/n_steps:.2f}ms   (total / {n_steps} steps)", flush=True)
  print("\n(golden published FPM <=512 ~16ms; collector isolated execute_model_gpu ~33-40ms)", flush=True)

if __name__ == "__main__":
    main()

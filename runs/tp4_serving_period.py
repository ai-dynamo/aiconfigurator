"""Measure the real-tp4 async-serving steady-state period for <=512-tok prefills
(golden's exact metric = inter-update period) on the REAL model + real weights,
multiprocess async engine. If period ~16ms -> golden's 16ms is the real-tp4 serving
throughput (and the collector's ~40ms isolated step is the artifact). If ~40ms ->
golden's number originates elsewhere.
"""
import os, time, sys
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
from vllm import LLM, SamplingParams, TokensPrompt

MODEL = "/workspace/models/hf_home/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"

def main():
    MNBT = int(os.environ.get("MNBT", "2048"))
    print(f"max_num_batched_tokens={MNBT}", flush=True)
    llm = LLM(model=MODEL, tensor_parallel_size=4, enable_expert_parallel=True,
              max_num_batched_tokens=MNBT, max_num_seqs=128, gpu_memory_utilization=0.9,
              enforce_eager=False, trust_remote_code=True, enable_prefix_caching=False,
              limit_mm_per_prompt={"image": 0, "video": 0})
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    mk = lambda nt: TokensPrompt(prompt_token_ids=list(range(nt)))

    for nt in (256, 512):
        llm.generate(prompts=[mk(nt) for _ in range(16)], sampling_params=sp, use_tqdm=False)  # warm
        # isolated
        t0 = time.perf_counter()
        for _ in range(16):
            llm.generate(prompts=[mk(nt)], sampling_params=sp, use_tqdm=False)
        iso = (time.perf_counter() - t0) * 1000.0 / 16
        # streamed (period)
        t0 = time.perf_counter()
        llm.generate(prompts=[mk(nt) for _ in range(128)], sampling_params=sp, use_tqdm=False)
        per = (time.perf_counter() - t0) * 1000.0 / 128
        print(f"[{nt}-tok] ISOLATED={iso:.1f} ms/req   STREAM period={per:.1f} ms/req", flush=True)
    print("(collector real-tp4 isolated ~40ms; golden FPM <=512 ~16ms)")

if __name__ == "__main__":
    main()

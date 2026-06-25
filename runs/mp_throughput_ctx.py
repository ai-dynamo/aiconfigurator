"""Test whether the REAL async multiprocess engine loop (golden's serving path,
NOT forced in-process) pipelines <=512-tok prefills to golden's ~16ms steady-state
period. Measures wall-clock throughput period = total_wall / N for a stream of
256-tok prefills with a 256-token budget (~one prefill per scheduler iteration).
No hook (engine runs in a subprocess); pure wall-clock.
"""
import os, time
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
# NOTE: do NOT set VLLM_ENABLE_V1_MULTIPROCESSING=0 -> use the real async engine core
from vllm import LLM, SamplingParams, TokensPrompt

CFG = "runs/layerwise_qwen36_tp4ep4_cleanctx4/config_cache/Qwen_Qwen3.6-35B-A3B_fd9719118c872c9fb3d947d5"

def main():
    llm = LLM(model=CFG, load_format="dummy", tensor_parallel_size=1,
              max_num_batched_tokens=256, max_num_seqs=128, gpu_memory_utilization=0.9,
              enforce_eager=False, trust_remote_code=True,
              enable_prefix_caching=False, limit_mm_per_prompt={"image": 0, "video": 0})
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    mk = lambda: TokensPrompt(prompt_token_ids=list(range(256)))

    llm.generate(prompts=[mk() for _ in range(16)], sampling_params=sp, use_tqdm=False)  # warm

    # isolated: one request per call
    t0 = time.perf_counter()
    for _ in range(16):
        llm.generate(prompts=[mk()], sampling_params=sp, use_tqdm=False)
    iso = (time.perf_counter() - t0) * 1000.0 / 16
    print(f"ISOLATED (1 req/call x16): {iso:.1f} ms/req")

    # streamed: N requests in one call -> continuous batching, async engine pipelines
    for NREQ in (64, 256):
        t0 = time.perf_counter()
        llm.generate(prompts=[mk() for _ in range(NREQ)], sampling_params=sp, use_tqdm=False)
        per = (time.perf_counter() - t0) * 1000.0 / NREQ
        print(f"STREAM ({NREQ} reqs/call): {per:.1f} ms/req (steady-state period)")
    print("\n(collector isolated execute_model_gpu=33ms; golden FPM <=512 steady-period ~16ms)")

if __name__ == "__main__":
    main()

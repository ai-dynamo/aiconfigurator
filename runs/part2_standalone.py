"""Part 2 (standalone): raw execute_model wall vs number of ACTIVE layers k.

Builds the qwen36 sharded-tp4 config with num_hidden_layers=k (layer_types[:k]) so
all k decoder layers physically execute (the "context-active-layers k" notion), then
times the 256-tok standalone prefill execute_model with cuda events. NO collector
representative-subset / identity-forward: every one of the k layers runs.

Same engine config as runs/standalone_ctx.py / captureA (plain vLLM, in-process,
enforce_eager=False, full FULL_AND_PIECEWISE cudagraph). Real MoE (load_format=dummy).

Usage: python runs/part2_standalone.py <k>   ->  prints "K=<k> WALL_MS=<median>"
"""
import os, sys, json, shutil, statistics, tempfile
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import torch
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.v1.worker import gpu_model_runner as gmr

k = int(sys.argv[1])
SRC = "runs/layerwise_qwen36_tp4ep4_cleanctx4/config_cache/Qwen_Qwen3.6-35B-A3B_fd9719118c872c9fb3d947d5"

# build a k-layer config dir (truncate to first k layers, same as collector override)
dst = tempfile.mkdtemp(prefix=f"qwen36_k{k}_")
for fn in os.listdir(SRC):
    if fn == "config.json":
        continue
    s = os.path.join(SRC, fn)
    if os.path.isfile(s):
        os.symlink(os.path.abspath(s), os.path.join(dst, fn))
cfg = json.load(open(os.path.join(SRC, "config.json")))
tc = cfg.get("text_config", cfg)
tc["num_hidden_layers"] = k
if isinstance(tc.get("layer_types"), list):
    tc["layer_types"] = tc["layer_types"][:k]
json.dump(cfg, open(os.path.join(dst, "config.json"), "w"))

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

llm = LLM(model=dst, load_format="dummy", tensor_parallel_size=1,
          max_num_batched_tokens=2048, max_num_seqs=128, gpu_memory_utilization=0.9,
          enforce_eager=False, trust_remote_code=True,
          enable_prefix_caching=False, limit_mm_per_prompt={"image": 0, "video": 0})
sp = SamplingParams(max_tokens=1, temperature=0.0)
for _ in range(2):
    llm.generate(prompts=[TokensPrompt(prompt_token_ids=list(range(256)))], sampling_params=sp, use_tqdm=False)
times.clear()
for _ in range(6):
    llm.generate(prompts=[TokensPrompt(prompt_token_ids=list(range(256)))], sampling_params=sp, use_tqdm=False)
pre = [ms for n, ms in times if n and n >= 200]
med = statistics.median(pre) if pre else float("nan")
print(f"K={k} WALL_MS={med:.4f} samples={['%.2f'%x for x in pre]}")
shutil.rmtree(dst, ignore_errors=True)

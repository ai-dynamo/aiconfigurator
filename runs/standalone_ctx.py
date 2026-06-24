import os, time, statistics
os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
import torch
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.v1.worker import gpu_model_runner as gmr

CFG="runs/layerwise_qwen36_tp4ep4_cleanctx4/config_cache/Qwen_Qwen3.6-35B-A3B_fd9719118c872c9fb3d947d5"

# minimal cuda-event timer around execute_model (NO layer-skip patch, NO marker)
_orig = gmr.GPUModelRunner.execute_model
times=[]
graphcalls=[]
def timed(self, scheduler_output, intermediate_tensors=None):
    n = getattr(scheduler_output,'total_num_scheduled_tokens',None)
    torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record()
    r=_orig(self, scheduler_output, intermediate_tensors)
    torch.cuda.synchronize(); e.record(); e.synchronize()
    ms=s.elapsed_time(e)
    times.append((n,ms))
    return r
gmr.GPUModelRunner.execute_model=timed

llm=LLM(model=CFG, load_format="dummy", tensor_parallel_size=1,
        max_num_batched_tokens=2048, max_num_seqs=128, gpu_memory_utilization=0.9,
        enforce_eager=False, trust_remote_code=True,
        enable_prefix_caching=False, limit_mm_per_prompt={"image":0,"video":0})
sp=SamplingParams(max_tokens=1, temperature=0.0)
import numpy as np
# 256-token prompt (token ids)
for warm in range(2):
    llm.generate(prompts=[TokensPrompt(prompt_token_ids=list(range(256)))], sampling_params=sp, use_tqdm=False)
times.clear()
for i in range(5):
    llm.generate(prompts=[TokensPrompt(prompt_token_ids=list(range(256)))], sampling_params=sp, use_tqdm=False)
# prefill steps = those with n>=256
pre=[ms for n,ms in times if n and n>=200]
print("STANDALONE 256-tok prefill execute_model GPU times (ms):", [f"{x:.2f}" for x in pre])
if pre: print("median:", round(statistics.median(pre),2))
print("(collector cleanctx4 256-tok = 33.1ms; golden context FPM ~16ms)")

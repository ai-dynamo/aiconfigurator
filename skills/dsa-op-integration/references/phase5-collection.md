# Phase 5: Performance Data Collection

## Objective

Collect DSA performance data matching MLA collector format for aiconfigurator integration.

## Collector Script

**File**: `collector/trtllm/collect_dsa.py`

```python
#!/usr/bin/env python3
"""DSA Performance Collector for TensorRT-LLM"""

import os
import torch
import tensorrt_llm
import argparse
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV32Attention
from tensorrt_llm._torch.attention_backend.sparse.dsa import (
    DSACacheManager,
    DSAtrtllmAttentionMetadata,
)

# Constants for DeepSeek V3.2
HIDDEN_SIZE = 7168
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 128
V_HEAD_DIM = 128
INDEX_N_HEADS = 64
INDEX_HEAD_DIM = 128
INDEX_TOPK = 2048


def run_dsa(
    input_len: int,
    batch_size: int,
    is_context_phase: bool,
    num_heads: int = 128,
    num_iterations: int = 6,
    device: str = "cuda:0",
    perf_filename: str = "dsa_perf.txt",
):
    """Run DSA benchmark."""
    
    device = torch.device(device)
    torch.cuda.set_device(device)
    
    # Validate
    assert num_heads % 64 == 0, f"num_heads must be multiple of 64, got {num_heads}"
    
    # Create DSA config
    sparse_attention_config = DeepSeekSparseAttentionConfig(
        index_n_heads=INDEX_N_HEADS,
        index_head_dim=INDEX_HEAD_DIM,
        index_topk=INDEX_TOPK,
    )
    
    # Create PretrainedConfig
    pretrained_config = PretrainedConfig()
    pretrained_config.hidden_size = HIDDEN_SIZE
    pretrained_config.num_attention_heads = num_heads
    pretrained_config.q_lora_rank = Q_LORA_RANK
    pretrained_config.kv_lora_rank = KV_LORA_RANK
    pretrained_config.qk_rope_head_dim = QK_ROPE_HEAD_DIM
    pretrained_config.qk_nope_head_dim = QK_NOPE_HEAD_DIM
    pretrained_config.v_head_dim = V_HEAD_DIM
    
    # Create ModelConfig
    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        mapping=Mapping(world_size=1, rank=0, tp_size=1),
        sparse_attention_config=sparse_attention_config,
    )
    
    # Create MLA module
    mla = DeepseekV32Attention(model_config=model_config, layer_idx=0)
    mla = mla.to(device)
    
    # Create KV cache
    max_seq_len = input_len + 10
    kv_cache_config = KvCacheConfig(
        max_tokens=batch_size * max_seq_len,
        enable_block_reuse=False,
    )
    
    kv_cache_manager = DSACacheManager(
        kv_cache_config,
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        tokens_per_block=64,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        mapping=Mapping(world_size=1, rank=0, tp_size=1),
        dtype=tensorrt_llm.bindings.DataType.BF16,
        sparse_attn_config=sparse_attention_config,
    )
    
    # Add requests
    total_seq_lens = [input_len + 1] * batch_size
    kv_cache_manager.add_dummy_requests(list(range(batch_size)), total_seq_lens)
    
    # Create metadata
    if is_context_phase:
        num_tokens = input_len * batch_size
        position_ids = torch.arange(input_len, device=device, dtype=torch.int32)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1).reshape(-1).contiguous()
    else:
        num_tokens = batch_size
        position_ids = torch.tensor([[input_len]] * batch_size, device=device, dtype=torch.int32)
    
    attn_metadata = DSAtrtllmAttentionMetadata(
        max_num_requests=batch_size,
        max_num_tokens=num_tokens,
        kv_cache_manager=kv_cache_manager,
        mapping=Mapping(world_size=1, rank=0, tp_size=1),
        enable_flash_mla=True,
        seq_lens=torch.tensor([input_len] * batch_size, dtype=torch.int32),
        position_ids=position_ids,
        num_contexts=batch_size if is_context_phase else 0,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[0] * batch_size if is_context_phase else [input_len] * batch_size,
        ),
        request_ids=list(range(batch_size)),
        prompt_lens=[input_len] * batch_size,
        runtime_features=AttentionRuntimeFeatures(chunked_prefill=False, cache_reuse=False),
        workspace=torch.tensor([], device=device, dtype=torch.int8),
        sparse_attention_config=sparse_attention_config,
    )
    
    attn_metadata.prepare()
    
    # Create input
    hidden_states = torch.randn([num_tokens, HIDDEN_SIZE], dtype=torch.bfloat16, device=device)
    
    # Warmup
    _ = mla.forward(position_ids, hidden_states, attn_metadata)
    torch.cuda.synchronize()
    
    # Benchmark
    def kernel_func():
        _ = mla.forward(position_ids, hidden_states, attn_metadata)
    
    with benchmark_with_power(
        kernel_func,
        num_iterations=num_iterations,
        warmup=10,
    ) as results:
        pass
    
    latency = results["median_ms"]
    
    # Log result
    if is_context_phase:
        isl = input_len
        step = 0
    else:
        isl = input_len
        step = 1
    
    log_perf(
        item_list=[{
            "dsa_dtype": "bfloat16",
            "num_heads": num_heads,
            "index_n_heads": INDEX_N_HEADS,
            "index_topk": INDEX_TOPK,
            "batch_size": batch_size,
            "isl": isl,
            "tp_size": 1,
            "step": step,
            "latency": latency,
        }],
        framework="TRTLLM",
        version=tensorrt_llm.__version__,
        device_name=torch.cuda.get_device_name(device),
        op_name=f"dsa_{'context' if is_context_phase else 'generation'}",
        kernel_source="default",
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )
    
    return latency
```

## Test Cases

### Context Phase

```python
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
seq_lens = [1, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384, 32768]

# Limit: batch_size * seq_len <= 65536
```

### Generation Phase

```python
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
seq_lens = [1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535, 131071]

# Limit: batch_size * seq_len <= 1024 * 4096 * 2 * 2 = 8388608
```

## Output Format

**File**: `dsa_perf_data/dsa_context_bf16_perf.txt`

```
framework,version,device,op_name,kernel_source,dsa_dtype,num_heads,index_n_heads,index_topk,batch_size,isl,tp_size,step,latency
TRTLLM,1.2.0rc5,NVIDIA H20-3e,dsa_context,default,bfloat16,128,64,2048,1,1,1,0,0.41
TRTLLM,1.2.0rc5,NVIDIA H20-3e,dsa_context,default,bfloat16,128,64,2048,1,16,1,0,0.43
...
TRTLLM,1.2.0rc5,NVIDIA H20-3e,dsa_context,default,bfloat16,128,64,2048,1,32768,1,0,346.29
```

**File**: `dsa_perf_data/dsa_generation_bf16_perf.txt`

```
framework,version,device,op_name,kernel_source,dsa_dtype,num_heads,index_n_heads,index_topk,batch_size,isl,tp_size,step,latency
TRTLLM,1.2.0rc5,NVIDIA H20-3e,dsa_generation,default,bfloat16,128,64,2048,1,1,1,1,0.41
TRTLLM,1.2.0rc5,NVIDIA H20-3e,dsa_generation,default,bfloat16,128,64,2048,1,131071,1,1,0.54
...
TRTLLM,1.2.0rc5,NVIDIA H20-3e,dsa_generation,default,bfloat16,128,64,2048,1024,16383,1,1,11.76
```

## Running the Collector

```bash
# Context phase
python3 collect_dsa.py --mode context \
  --batch_sizes 1,2,4,8,16,32,64,128,256 \
  --seq_lens 1,16,32,64,128,256,512,1024,1536,2048,3072,4096,6144,8192,10240,12288,16384,32768 \
  --num_iterations 3 \
  --output_dir ./dsa_perf_data

# Generation phase
python3 collect_dsa.py --mode generation \
  --batch_sizes 1,2,4,8,16,32,64,128,256,512,1024 \
  --seq_lens 1,3,7,15,31,63,127,255,511,1023,2047,4095,8191,16383,32767,65535,131071 \
  --num_iterations 3 \
  --output_dir ./dsa_perf_data
```

## Expected Results

### Context Phase Characteristics

- **Latency scales linearly** with `batch_size × seq_len`
- Typical: ~10 ms per 4096 tokens at batch=1
- Large batches: 100-600 ms for 256×256

### Generation Phase Characteristics

- **Nearly constant latency** for small batches (1-16)
- Small batch: ~0.4-0.5 ms
- Large batch (1024): ~10-11 ms
- Seq_len has minimal impact

## Known Limitations

### FP8 KV Cache

**Not supported in standalone collector.**

Reason: `flash_mla_sparse_fwd` only accepts BF16 input.

```python
# ❌ This will fail
kv_cache_dtype = tensorrt_llm.bindings.DataType.FP8
# Error: Expected kv.dtype() == torch::kBFloat16
```

**Solution**: Test FP8 with full model using `trtllm-serve`.

### GPU Requirements

DSA requires SM90+ (Hopper/Blackwell):
- ✅ H20, H100, H200, B100, B200
- ❌ A100, L40, L40S

## Validation

```python
# Verify data integrity
import csv

with open("dsa_perf_data/dsa_context_bf16_perf.txt") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    
print(f"Context data points: {len(rows)}")
assert len(rows) == 110, "Expected 110 context data points"

# Check data ranges
for row in rows:
    latency = float(row["latency"])
    assert 0 < latency < 1000, f"Invalid latency: {latency}"
```

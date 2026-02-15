# Phase 2: Build a Mock Layer

## Goal

Create a standalone runnable operation layer (mock layer) for:
1. Validating operation behavior
2. Subsequent nsys profile alignment
3. Serving as the foundation for Phase 5 data collection

## Option A: Use Framework-Provided Classes (Recommended)

Prefer existing layer classes from TensorRT-LLM / vLLM / SGLang whenever possible.

Reference pattern: `collector/trtllm/collect_mla.py`

```python
import tensorrt_llm
import torch
from tensorrt_llm._torch.attention_backend import AttentionInputType, TrtllmAttentionMetadata
from tensorrt_llm._torch.attention_backend.utils import create_attention
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.mapping import Mapping

# 1. Build pretrained config (derived from model config)
pretrained_config = ...

# 2. Build model config
model_config = ModelConfig(
    pretrained_config=pretrained_config,
    mapping=Mapping(world_size=1, rank=0, tp_size=1),
)

# 3. Instantiate operation/layer
layer = XxxAttention(model_config=model_config, layer_idx=0)
# or
layer = create_attention(...)
```

## Option B: Custom Implementation (Use Only If Necessary)

Use this only when framework classes cannot be reused directly (heavy dependencies, special runtime constraints, etc.).

```python
import torch
import torch.nn as nn

class XxxOp(nn.Module):
    """Mock implementation of the XXX operation."""
    
    def __init__(self, param1: int, param2: int, dtype=torch.bfloat16):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
        self.dtype = dtype
        # Initialize weights and submodules
        self.weight = nn.Parameter(
            torch.randn(param1, param2, dtype=dtype, device="cuda")
        )
    
    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        # Implement core operation logic
        output = ...
        return output
```

## Context vs Generation Differences

Some operations behave differently in context (prefill) vs generation (decode), especially attention-like ops:

| Dimension | Context Phase | Generation Phase |
|------|--------------|------------------|
| **Input tokens** | `batch_size * seq_len` (multi-token) | `batch_size * 1` (single token per request) |
| **KV Cache** | No read or small prefix read | Read full historical KV |
| **Compute pattern** | Usually compute-bound | Usually memory-bound |
| **Typical parameter range** | batch 1~256, seq 1~32768 | batch 1~1024, kv_len 1~131072 |

**Your mock layer must cover both modes.**

## Modeling Contract for Collector (Do This in Phase 2)

When a new op requires modeling, define the collector contract in this phase (not later), because
collector test-case schema depends on the modeling axes.

### SOL-first axis decision

1. Derive a first-pass SOL expression with symbolic variables (`b`, `s`, and op params).
2. Decide whether the op is:
   - **Token-equivalent**: model primarily by `x=b*s`
   - **Token-non-equivalent**: keep `b` and `s` as separate axes
3. Freeze collector input schema and downstream query signature from this decision.

### Practical rule of thumb

- GEMM-like paths are often close to token-equivalent.
- Attention-like paths are usually token-non-equivalent and should keep both `b` and `s`.

### Required output of this step

- Axis decision note (`x` vs `b,s`) with SOL justification
- Collector test-case layout matching that decision
- Planned query signature shape for `perf_database.py` and `operations.py`

## Key Validation Checks

| Check | Description | Validation Method |
|--------|------|----------|
| **Input format** | Verify shape, dtype, and device | `print(input.shape, input.dtype, input.device)` |
| **Output format** | Output shape should align with hidden size | `assert output.shape[-1] == hidden_size` |
| **KV Cache Manager** | Attention ops typically require cache manager | Confirm `KVCacheManager` initialization exists |
| **Boundary cases** | batch=1, seq=1, large batch, etc. | Test multiple parameter combinations |
| **Dtype support** | BF16/FP16/FP8 constraints | Validate under target dtype |
| **CUDA Graph compatibility** | Collection uses CUDA Graph acceleration | Verify graph capture succeeds |

## Standalone Test Script Template

```python
#!/usr/bin/env python3
"""Test script for Xxx mock layer."""

import torch

def create_xxx_layer(param1, param2, dtype=torch.bfloat16):
    """Create and return the mock layer."""
    # ... build logic ...
    return layer

def test_context_phase():
    """Test context (prefill) phase."""
    layer = create_xxx_layer(param1=64, param2=2048)
    
    batch_size, seq_len = 4, 1024
    hidden_size = 4096
    hidden = torch.randn(
        [batch_size * seq_len, hidden_size],
        dtype=torch.bfloat16, device="cuda"
    )
    
    # Forward pass
    output = layer(hidden)
    print(f"Context - Input: {hidden.shape}, Output: {output.shape}")
    assert output.shape == hidden.shape, f"Shape mismatch: {output.shape} != {hidden.shape}"
    
    # Verify no NaN
    assert not torch.isnan(output).any(), "Output contains NaN!"

def test_generation_phase():
    """Test generation (decode) phase."""
    layer = create_xxx_layer(param1=64, param2=2048)
    
    batch_size = 32
    hidden_size = 4096
    hidden = torch.randn(
        [batch_size, hidden_size],  # single token per request
        dtype=torch.bfloat16, device="cuda"
    )
    
    output = layer(hidden)
    print(f"Generation - Input: {hidden.shape}, Output: {output.shape}")
    assert output.shape == hidden.shape

def test_cuda_graph_compatibility():
    """Test that the layer works with CUDA Graph capture."""
    layer = create_xxx_layer(param1=64, param2=2048)
    hidden = torch.randn([32, 4096], dtype=torch.bfloat16, device="cuda")
    
    # Warmup
    for _ in range(3):
        layer(hidden)
    torch.cuda.synchronize()
    
    # Graph capture
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            output = layer(hidden)
        torch.cuda.synchronize()
        print("CUDA Graph capture: SUCCESS")
    except Exception as e:
        print(f"CUDA Graph capture: FAILED - {e}")
        print("Note: benchmark_with_power has allow_graph_fail fallback")

if __name__ == "__main__":
    test_context_phase()
    test_generation_phase()
    test_cuda_graph_compatibility()
    print("\nAll tests passed!")
```

## Deliverables

After completing Phase 2, you should have:
1. **Runnable mock-layer code** — either standalone `.py` or integrated into collector
2. **Test script** — validates both context and generation modes
3. **Parameter documentation** — records all required mock-layer parameters and meanings

## Common Issues

| Issue | Cause | Fix |
|------|------|------|
| `ImportError` for missing module | TRT-LLM version mismatch | Verify required TRT-LLM version (`frameworks/` as reference) |
| CUDA OOM | Input too large | Reduce `batch_size` or `seq_len` |
| CUDA Graph capture fails | Dynamic control flow in op | Use `allow_graph_fail=True` to fall back to eager path |
| Output is all zeros or NaN | Weights not initialized correctly | Check weight init and forward logic |
| KV cache not initialized | Attention ops need cache manager | Follow `KVCacheManager` usage in `collect_mla.py` |
| MPI initialization failure | TRT-LLM import requires MPI | Set `export OPAL_PREFIX=/opt/hpcx/ompi` |

## TP Emulation (Critical)

Follow `collect_mla.py`: **multi-GPU is not required** for collection. Compute `local_num_heads = global_heads // tp_size` on a single GPU, then instantiate the layer with `Mapping(world_size=1, tp_size=1)`. Store local heads in the `num_heads` column of perf data.

---
name: dsa-op-integration
description: |
  Integrate new DSA (DeepSeek Sparse Attention) operations into aiconfigurator. Use when adding support for DeepSeek V3.2 or similar models with sparse attention. Covers: (1) Model architecture analysis, (2) Mock model layer creation, (3) nsys profiling and alignment, (4) aiconfigurator integration, (5) Performance data collection.
---

# DSA Operation Integration for aiconfigurator

This skill provides a complete workflow for integrating DSA (DeepSeek Sparse Attention) operations into the aiconfigurator SDK.

## Overview

DSA is a sparse attention mechanism introduced in DeepSeek V3.2. Key characteristics:
- **O(n×k) complexity** vs MLA's O(n²)
- **Indexer**: Selects top-k relevant tokens using MQA
- **FlashMLA Sparse**: SM90+ optimized kernel (Hopper/Blackwell GPUs only)

## Integration Workflow

### Phase 1: Model Architecture Analysis

**Goal**: Understand the new model's DSA parameters and how they differ from existing implementations.

1. **Find config.json** in the model repository (e.g., HuggingFace)
2. **Extract DSA parameters**:
   ```python
   # Key DSA config fields:
   index_n_heads: int      # Number of heads for indexer (e.g., 64)
   index_head_dim: int     # Head dimension for indexer (e.g., 128)  
   index_topk: int         # Top-k selection (e.g., 2048)
   first_k_dense_replace: int  # Layers using dense/MLA (e.g., 3)
   ```

3. **Map architecture name**:
   ```python
   # In common.py
   ARCH_MAPPING = {
       "DeepseekV32ForCausalLM": ModelFamily.DEEPSEEK,
   }
   ```

4. **Create config dataclass**:
   ```python
   @dataclass
   class DeepSeekV32Config:
       index_n_heads: int = 64
       index_head_dim: int = 128
       index_topk: int = 2048
       first_k_dense_replace: int = 3
   ```

**Reference**: [references/phase1-analysis.md](references/phase1-analysis.md)

### Phase 2: Mock Model Layer

**Goal**: Create a standalone DSA operation for testing and profiling.

#### Option A: Use Official Class (Recommended)

```python
from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV32Attention
from tensorrt_llm._torch.attention_backend.sparse.dsa import DSACacheManager

# Create MLA module with DSA enabled
mla = DeepseekV32Attention(model_config=model_config, layer_idx=0)

# Use DSACacheManager (not regular KVCacheManager!)
kv_cache_manager = DSACacheManager(
    ...,
    sparse_attn_config=sparse_attention_config,
)
```

#### Option B: Custom DSAOp (For learning/debugging)

```python
class DSAOp(nn.Module):
    """Complete DSA operation: Indexer + FlashMLA Sparse"""
    
    def __init__(self, num_heads, index_n_heads, index_topk):
        self.indexer = Indexer(...)
        self.mqa = create_attention(...)  # MLA attention
        
    def forward(self, hidden_states, ...):
        # 1. Indexer computes topk_indices
        topk_indices = self.indexer(qr, hidden_states, ...)
        
        # 2. MLA RoPE + KV append
        self.mqa.mla_rope_append_paged_kv_assign_q(...)
        
        # 3. FlashMLA sparse attention
        attn_out = flash_mla_sparse_fwd(q, kv, topk_indices, ...)
        
        return attn_out
```

**Critical Requirements**:
- `num_heads` must be **multiple of 64** for FlashMLA sparse on SM90
- FlashMLA sparse **only accepts BF16 input** (no FP8)
- Use `DSACacheManager` for indexer K cache

**Reference**: [references/phase2-mock-layer.md](references/phase2-mock-layer.md)

### Phase 3: nsys Profiling and Alignment

**Goal**: Verify the mock layer produces correct kernel calls and matches E2E model timing.

#### 3.1 Profile Mock Layer

```bash
nsys profile -o dsa_op \
  -t cuda,nvtx \
  python3 test_dsa_op.py
```

#### 3.2 Profile E2E Model (Reference)

For full models using MPI spawn, use system-wide sampling:

```bash
# Start server first
trtllm-serve /path/to/model --tp_size 8 &

# Wait for warmup, then profile
nsys profile -o e2e_ref \
  -y 60 -d 20 \
  --sample=system-wide \
  --cpuctxsw=system-wide \
  -t cuda,nvtx,osrt \
  --cuda-graph-trace=node \
  trtllm-serve ...
```

#### 3.3 Compare Kernels

```bash
nsys stats --report cuda_gpu_kern_sum dsa_op.nsys-rep
nsys stats --report cuda_gpu_kern_sum e2e_ref.nsys-rep
```

**Key kernels to verify**:
- `sm90::sparse_attn_fwd_kernel` - Main sparse attention
- `deep_gemm::sm90_fp8_mqa_logits` - Indexer MQA computation
- `topKPerRowPrefill` / `topKPerRowDecode` - Top-k selection

**Alignment checklist**:
- [ ] Kernel names match
- [ ] Latency within 2x of E2E (overhead from standalone setup)
- [ ] No missing or extra kernels

**Reference**: [references/phase3-nsys-alignment.md](references/phase3-nsys-alignment.md)

### Phase 4: aiconfigurator Integration

**Goal**: Add DSA operations to the SDK.

#### 4.1 Add Operation Classes

```python
# operations.py
@dataclass
class ContextDSA(Operation):
    """DSA context phase operation"""
    num_heads: int
    index_n_heads: int
    index_topk: int
    
    def query(self, db, batch_size, s, **kwargs) -> float:
        # SOL estimation: O(n×k) complexity
        # Per-token compute: index_n_heads × index_head_dim × index_topk
        # ...
        return latency_ms
```

#### 4.2 Add Database Queries

```python
# perf_database.py
def query_context_dsa(self, b, s, num_heads, index_n_heads, index_topk):
    # Load from collected data or use SOL estimation
    pass
```

#### 4.3 Add Model Integration

```python
# models.py
class DeepSeekModel:
    def _setup_operations(self):
        if self._use_dsa:
            # Layer 0-2: MLA (first_k_dense_replace layers)
            # Layer 3-60: DSA
            for i in range(self.num_layers):
                if i < self._dsa_config.first_k_dense_replace:
                    self.context_ops.append(ContextMLA(...))
                else:
                    self.context_ops.append(ContextDSA(...))
```

**Reference**: [references/phase4-integration.md](references/phase4-integration.md)

### Phase 5: Performance Data Collection

**Goal**: Collect benchmark data matching MLA collector format.

#### 5.1 Create Collector Script

```python
# collect_dsa.py
def run_dsa(input_len, batch_size, is_context_phase, ...):
    # Create DeepseekV32Attention
    mla = DeepseekV32Attention(model_config, layer_idx=0)
    
    # Create DSACacheManager
    kv_cache_manager = DSACacheManager(..., sparse_attn_config=dsa_config)
    
    # Benchmark
    latency = benchmark_with_power(kernel_func, ...)
    
    # Log result
    log_perf([{
        "batch_size": batch_size,
        "isl": input_len if is_context_phase else input_len,
        "step": 0 if is_context_phase else 1,
        "latency": latency,
    }], ...)
```

#### 5.2 Standard Test Cases

**Context Phase**:
```python
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
seq_lens = [1, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384, 32768]
limit = batch_size * seq_len <= 65536
```

**Generation Phase**:
```python
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
seq_lens = [1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535, 131071]
limit = batch_size * seq_len <= 1024 * 4096 * 2 * 2
```

#### 5.3 Output Format

```
framework,version,device,op_name,kernel_source,dsa_dtype,num_heads,index_n_heads,index_topk,batch_size,isl,tp_size,step,latency
TRTLLM,1.2.0rc5,NVIDIA H20-3e,dsa_context,default,bfloat16,128,64,2048,1,4096,1,0,40.77
```

**Reference**: [references/phase5-collection.md](references/phase5-collection.md)

## Known Limitations

### FP8 KV Cache

**Standalone collector does NOT support FP8 KV cache.**

Root cause: `flash_mla_sparse_fwd` only accepts BF16 input:
```python
def flash_mla_sparse_fwd(
    q: torch.Tensor,  # bfloat16 only
    kv: torch.Tensor,  # bfloat16 only
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
):
```

For FP8 testing, use full model environment with `trtllm-serve`.

### GPU Requirements

DSA requires **SM90+ (Hopper or Blackwell)**:
- NVIDIA H20, H100, H200, B100, B200
- NOT supported on SM80 (A100) or SM89 (L40)

## Troubleshooting

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `num_heads must be multiple of 64` | FlashMLA sparse SM90 requirement | Use `num_heads` divisible by 64 |
| `kv.dtype() == torch::kBFloat16` | FP8 not supported | Use BF16 KV cache |
| `pos_ids must be contiguous` | Position IDs not contiguous | Call `.contiguous()` on position_ids |
| `max_input_uncached_seq_len > 0` | Generation metadata mismatch | Check `num_cached_tokens_per_seq` |

### Debug Tips

1. **Use nsys to verify kernel calls**:
   ```bash
   nsys stats --report cuda_gpu_kern_sum output.nsys-rep
   ```

2. **Compare with official implementation**:
   ```python
   # Check DeepseekV32Attention source
   from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV32Attention
   import inspect
   print(inspect.getsource(DeepseekV32Attention.forward_context_dsa))
   ```

3. **Test with dummy weights**:
   ```python
   llm = LLM(model=checkpoint, load_format='dummy', ...)
   ```

## References

- [Phase 1: Model Analysis](references/phase1-analysis.md)
- [Phase 2: Mock Layer](references/phase2-mock-layer.md)
- [Phase 3: nsys Alignment](references/phase3-nsys-alignment.md)
- [Phase 4: Integration](references/phase4-integration.md)
- [Phase 5: Collection](references/phase5-collection.md)

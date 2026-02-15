# Phase 3: nsys Profiling and Alignment

## Objective

Verify mock layer correctness and measure performance using nsys profiler.

## Profiling Setup

### 3.1 Profile Mock Layer

Simple profiling (no MPI):

```bash
nsys profile -o dsa_op \
  -t cuda,nvtx \
  python3 test_dsa_op.py --mode context --seq_len 4096
```

Expected kernel output:
```
sm90::sparse_attn_fwd_kernel    61.8%  25.20 ms
deep_gemm::sm90_fp8_mqa_logits   1.3%   0.53 ms
topKPerRowPrefill                0.4%   0.17 ms
```

### 3.2 Profile E2E Full Model

For models using MPI spawn, use system-wide sampling:

```bash
# Start server without nsys
trtllm-serve /path/to/model \
  --config config.yaml \
  --tp_size 8 &

# Wait for warmup (5-10 min)
sleep 600

# Profile with system-wide sampling
nsys profile -o e2e_full \
  -y 60 -d 20 \
  --sample=system-wide \
  --cpuctxsw=system-wide \
  -t cuda,nvtx,osrt \
  --cuda-graph-trace=node \
  trtllm-serve /path/to/model --config config.yaml --tp_size 8
```

**Why system-wide?**
- MPI spawn creates worker processes
- Regular nsys can't attach to workers
- `--sample=system-wide` captures all GPU activity

### 3.3 Profile Single-Layer Model

Create minimal test model for clean comparison:

```python
# Create single-layer V3.2 checkpoint
config = {
    "model_type": "deepseek_v32",
    "num_hidden_layers": 1,
    "first_k_dense_replace": 0,  # All DSA
    "n_routed_experts": 8,  # Minimal MoE
    "index_n_heads": 64,
    "index_topk": 2048,
}
```

## Kernel Comparison

### Extract Kernel Stats

```bash
nsys stats --report cuda_gpu_kern_sum dsa_op.nsys-rep
```

### Expected DSA Kernels

| Kernel | Function | Typical % |
|--------|----------|-----------|
| `sm90::sparse_attn_fwd_kernel` | FlashMLA sparse attention | 60-65% |
| `deep_gemm::sm90_fp8_mqa_logits` | Indexer MQA computation | 1-2% |
| `nvjet_sm90_tst_*` | Indexer MQA (various configs) | 15-25% |
| `topKPerRowPrefill` | Top-k selection (context) | 0.3-0.5% |
| `topKPerRowDecode` | Top-k selection (generation) | 2-4% |
| `applyMLARopeAppendPagedKVAssignQKernel` | MLA RoPE | 0.1% |

### Comparison Table Format

```
| Kernel | Mock Layer | E2E Model | Match |
|--------|-----------|-----------|-------|
| sparse_attn_fwd_kernel | 25.20 ms | 25.2 ms | ✅ |
| fp8_mqa_logits | 0.53 ms | 0.53 ms | ✅ |
| topKPerRowPrefill | 0.17 ms | 0.17 ms | ✅ |
```

## Alignment Checklist

Run through this checklist:

- [ ] **Kernel names match** - Same kernel functions called
- [ ] **Latency within 2x** - Mock layer should be close (some overhead expected)
- [ ] **No missing kernels** - All E2E kernels present in mock
- [ ] **No extra kernels** - Mock doesn't call unnecessary functions
- [ ] **Data flow correct** - Verify tensor shapes match

## Troubleshooting

### nsys Captures No GPU Data

**Symptom**: `.nsys-rep` file is 0 bytes or only has CPU data

**Solutions**:

1. **For MPI spawn**: Use `--sample=system-wide`
   ```bash
   nsys profile --sample=system-wide ...
   ```

2. **Check permissions**: Need root or low `perf_event_paranoid`
   ```bash
   cat /proc/sys/kernel/perf_event_paranoid
   # Should be <= 0, or run with sudo
   ```

3. **Verify CUDA tracing**:
   ```bash
   nsys profile -t cuda,nvtx ...
   ```

### Kernel Times Don't Match

**Possible causes**:

1. **Different num_heads**
   ```python
   # Mock: num_heads=128
   # E2E: num_heads=128 / TP_size
   ```

2. **Different batch_size or seq_len**
   ```bash
   # Check test parameters match
   python3 test.py --batch_size 1 --seq_len 4096
   ```

3. **Different dtype**
   ```python
   # Mock: BF16
   # E2E: BF16 or FP8 (affects latency)
   ```

### Missing Indexer Kernels

**Symptom**: Only `sparse_attn_fwd_kernel`, no MQA kernels

**Cause**: Indexer not called correctly

**Fix**:
```python
# Verify indexer is called
topk_indices = self.indexer(qr, hidden_states, ...)
print(f"topk_indices shape: {topk_indices.shape}")  # [num_tokens, index_n_heads, topk]
```

## Example: Full Profiling Session

```bash
# 1. Profile context phase
nsys profile -o context_bf16 \
  -t cuda,nvtx \
  python3 collect_dsa.py --mode context --batch_sizes 1 --seq_lens 4096

# 2. Profile generation phase
nsys profile -o generation_bf16 \
  -t cuda,nvtx \
  python3 collect_dsa.py --mode generation --batch_sizes 1 --seq_lens 4096

# 3. Analyze
nsys stats --report cuda_gpu_kern_sum context_bf16.nsys-rep > context_stats.txt
nsys stats --report cuda_gpu_kern_sum generation_bf16.nsys-rep > generation_stats.txt

# 4. Compare key kernels
grep "sparse_attn\|mqa_logits\|topK" context_stats.txt
grep "sparse_attn\|mqa_logits\|topK" generation_stats.txt
```

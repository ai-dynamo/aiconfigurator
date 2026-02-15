# Phase 3: nsys Profile Alignment

## Goal

Validate that the GPU-kernel behavior of the Phase 2 mock layer matches full-model E2E inference.
This is critical for data validity: if mock-layer kernels differ from real inference kernels, collected data will be unreliable.

## Core Principle

aiconfigurator models inference by decomposing it into Operations. Each operation (such as attention or MoE) corresponds to a set of GPU kernels.
Alignment target: **kernel set launched by mock layer ≈ kernel set for that operation in E2E inference**.

## Step 1: Profile Mock Layer

```bash
# Profile the Phase 2 test script
nsys profile -o mock_layer_report \
  -t cuda,nvtx \
  --force-overwrite true \
  python3 test_xxx_op.py
```

> **Note**: Ensure warmup steps exist in the test script so profiling captures steady-state behavior.

## Step 2: Profile E2E Model

Choose profiling strategy based on deployment mode:

### Option A: Single-process model

```bash
nsys profile -o e2e_report \
  -t cuda,nvtx \
  --force-overwrite true \
  python3 run_full_model.py
```

### Option B: MPI / Multi-process model (for example `trtllm-serve`)

Multi-process deployments require system-wide sampling because workers are spawned internally by the framework:

```bash
# 1. Start service first
trtllm-serve /path/to/model --tp_size 8 &

# 2. Wait for model load + warmup
sleep 300

# 3. Run system-wide profiling
nsys profile -o e2e_report \
  -y 60 -d 20 \
  --sample=system-wide \
  --cpuctxsw=system-wide \
  -t cuda,nvtx,osrt \
  --cuda-graph-trace=node \
  --force-overwrite true \
  trtllm-serve /path/to/model --tp_size 8
```

Parameter notes:
- `-y 60`: delay collection start by 60 seconds (to pass warmup)
- `-d 20`: collect for 20 seconds
- `--sample=system-wide`: capture all processes
- `--cuda-graph-trace=node`: expand CUDA Graph nodes

## Step 3: Extract Kernel Statistics

```bash
# Mock-layer kernel summary
nsys stats --report cuda_gpu_kern_sum mock_layer_report.nsys-rep > mock_kernels.txt

# E2E model kernel summary
nsys stats --report cuda_gpu_kern_sum e2e_report.nsys-rep > e2e_kernels.txt
```

You can also open `.nsys-rep` files in Nsight Systems GUI for visual comparison.

## Step 4: Alignment Analysis

### Alignment Checklist

| Check | Pass Criteria | If It Fails |
|--------|----------|----------|
| **Kernel name match** | Major kernel names appear in both traces | Verify mock layer uses the correct low-level implementation |
| **Kernel count** | Similar kernel count per forward | If mock has extra kernels, check unnecessary initialization |
| **Latency ratio** | Total per-forward latency gap < 2x | Check for extra overhead or missing compute |
| **No missing critical kernels** | Critical kernels in E2E also appear in mock | Submodule init may be missing (for example cache manager) |
| **No extra kernels** | Mock has no kernels absent in E2E | Remove unnecessary compute or debug code |

### How to identify kernels of the target operation

E2E profiles include kernels for all operations across all layers. Isolate target-op kernels via:

1. **NVTX ranges**: many frameworks mark op scopes with NVTX; filter by range in GUI
2. **Kernel-name patterns**: attention kernels often contain keywords like `fmha`, `flash`
3. **Timeline sequence matching**: compare one-layer kernel sequence in E2E vs mock

### Alignment metric example

```
Mock Layer:
  flash_fwd_kernel          : 1 call, 0.42ms
  void gemm_kernel<...>     : 2 calls, 0.15ms + 0.15ms
  
E2E Model (target op in one layer):
  flash_fwd_kernel          : 1 call, 0.40ms  <- match
  void gemm_kernel<...>     : 2 calls, 0.16ms + 0.14ms  <- match
  
Alignment result: PASS (kernel names and counts match, latency gap < 10%)
```

## Common Issues

| Issue | Cause | Fix |
|------|------|------|
| nsys has no GPU data | MPI worker processes were not captured | Use `--sample=system-wide` |
| Mock latency is much higher than E2E | Mock is missing optimizations (for example no CUDA Graph) | Ensure mock benchmark also uses graph replay |
| Missing kernels in mock | Dependency modules not initialized | Check `KVCacheManager`, quantization modules, etc. |
| Target op not identifiable in E2E | Missing NVTX or kernel fusion | Use kernel-name patterns; consider framework-level fusion |
| Large latency gap with matching names | Input mismatch | Ensure mock input shapes match E2E exactly |

## Correct Alignment Method (Practical Guidance)

**Not sufficient**: only comparing kernel-name lists.
**Correct approach**: build a full decoder layer, extract the target submodule (for example `decoder.self_attn`), and benchmark with the exact same inputs as collector mock layer:

```python
# Extract attention submodule from decoder layer
decoder = DecoderLayer(model_config=mc, layer_idx=0, ...)
decoder_attn = decoder.self_attn

# Standalone attention from collector
collector_attn = create_xxx_layer(tp_size=8)

# Same inputs, compare latency
for name, attn in [("collector", collector_attn), ("decoder", decoder_attn)]:
    # ... create identical metadata, hidden_states, position_ids ...
    with benchmark_with_power(...) as res:
        pass
    print(f"{name}: {res['latency_ms']:.4f}ms")
```

**Pass criteria**: latency gap < 10% and consistent CUDA Graph behavior.

## Deliverables

After Phase 3 you should have:
1. **Alignment report** — kernel matching status and latency comparison
2. **Known-difference notes** — reasons for any known deviations and modeling impact
3. **Mock-layer fixes** (if needed) — Phase 2 implementation updates based on alignment findings

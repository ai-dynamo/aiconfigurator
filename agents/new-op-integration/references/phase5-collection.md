# Phase 5: Performance Data Collection

## Goal

Collect latency data for the new operation across parameter combinations on target GPUs, and generate `*_perf.txt` files for aiconfigurator queries.

## Architecture Overview

```
collector/
├── collect.py                  # unified collection entrypoint (multi-process scheduling)
├── helper.py                   # benchmark_with_power(), log_perf()
├── common_test_cases.py        # standard test-case definitions
├── trtllm/
│   ├── collect_mla.py          # MLA collection (reference implementation)
│   ├── collect_dsa.py          # DSA collection (V3.2 practical example)
│   ├── collect_moe.py
│   └── collect_xxx.py          # <- add your script here
└── ...
```

**Key point**: New-op collection scripts must conform to `collect.py` orchestration. Do not create an unrelated standalone parallel pipeline.

## Step 1: Implement the collection script

**File**: `collector/trtllm/collect_new_op.py`

Export these three entrypoints:
1. `get_context_xxx_test_cases()` — returns a test-case list
2. `get_generation_xxx_test_cases()` — same for generation
3. `run_xxx()` — runs one test case

### Test-case format

Each test case is a **positional-argument list** and is expanded via `run_xxx(*test_case)`. Follow MLA conventions:

```python
# MLA: [input_len, batch_size, output_len, dtype, num_heads, world_size, tp_size, ...]
# DSA: [seq_len, batch_size, tp_size, is_context, perf_filename]
```

Core rules:
- List order must exactly match `run_xxx()` parameter order.
- Include `perf_filename` in the list (do not hardcode it in `run_xxx`).

### Choose collection axes from modeling decision

Collection axes must follow the axis decision made in Phase 2 mock-layer design:

- If op is token-equivalent (`x=b*s`), sweep `x` as the primary axis.
- If op is token-non-equivalent, sweep `b` and `s` as separate axes.

Do not use an `x`-only grid for attention-like ops.

### Reuse existing range baselines

For new ops, start from the nearest existing op range:

- GEMM-like: reuse GEMM/linear token ranges.
- Attention-like: reuse attention/MLA/DSA `b`/`s` ranges.
- Keep existing guardrails (`b*s` limits, TP list, context/generation split).

Only expand ranges when required by a clear model-specific reason.

```python
def get_context_xxx_test_cases():
    test_cases = []
    for tp_size in [1, 2, 4, 8, 16, 32, 64, 128]:
        for b in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            for s in [1, 16, 32, 64, 128, 256, 512, 1024, ...]:
                if b * s > 65536:
                    continue
                test_cases.append([s, b, tp_size, True, "xxx_context_perf.txt"])
    return test_cases
```

### `run_xxx` function

```python
def run_xxx(seq_len, batch_size, tp_size, is_context, perf_filename, device="cuda:0"):
    """Run a single benchmark. Args match test case list order."""
    # 1. Build mock layer (single-GPU TP emulation)
    layer = create_xxx_layer(tp_size, device)  # local_heads = global_heads / tp_size
    
    # 2. Prepare inputs + metadata
    # ...
    
    # 3. benchmark_with_power
    def kernel_func():
        layer.forward(...)
    with benchmark_with_power(device, kernel_func, num_warmups=5, num_runs=6,
                              repeat_n=1, allow_graph_fail=True) as res:
        pass
    
    # 4. log_perf
    log_perf(item_list=[{...}], framework="TRTLLM", version=...,
             device_name=..., op_name="xxx_context", kernel_source="default",
             perf_filename=perf_filename, power_stats=res["power_stats"])
```

### TP Emulation (Critical)

**Multi-GPU is not required**. Follow MLA/GQA:
```python
local_num_heads = GLOBAL_HEADS // tp_size
mapping = Mapping(world_size=1, rank=0, tp_size=1)  # always single-GPU
# pass local_num_heads into layer constructor
```

## Step 2: Register in `collect.py`

Add entries in the `collections` list inside `collect_trtllm()`:

```python
{
    "name": "trtllm",
    "type": "xxx_context",
    "module": "collector.trtllm.collect_xxx",
    "get_func": "get_context_xxx_test_cases",
    "run_func": "run_xxx",
},
{
    "name": "trtllm",
    "type": "xxx_generation",
    "module": "collector.trtllm.collect_xxx",
    "get_func": "get_generation_xxx_test_cases",
    "run_func": "run_xxx",
},
```

Also add `"xxx_context"` and `"xxx_generation"` into `--ops` choices.

## Step 3: Run collection

```bash
# Via unified collect.py entrypoint (recommended; handles multi-GPU scheduling)
cd collector/
python3 collect.py --backend trtllm --ops xxx_context xxx_generation

# Or run standalone (debug only)
cd collector/trtllm/
python3 collect_xxx.py
```

## Step 4: Validate

After collection, validate:
1. Check row counts are as expected (you need to calculate how many cases there should be)
2. Re-run several points with the same parameters; latency deviation should be < 10%
3. After installation, run `aiconfigurator cli default` to confirm E2E works

```python
# Quick verification: rerun a few points and compare
run_xxx(4096, 1, 8, True, "/tmp/verify.txt")
# Compare last row of /tmp/verify.txt against existing data
```

## Step 5: Place data files

```bash
cp xxx_context_perf.txt src/aiconfigurator/systems/data/{system}/{backend}/{version}/
cp xxx_generation_perf.txt src/aiconfigurator/systems/data/{system}/{backend}/{version}/
```

## Common Issues

| Issue | Cause | Fix |
|------|------|------|
| CUDA OOM | `batch * seq` too large | Add `if b*s > limit: continue` in test-case generation |
| High latency jitter | GPU throttling or background load | Check `results["throttled"]`; keep GPU idle |
| Corrupted `log_perf` writes | Multiple processes writing same file | `log_perf` uses `fcntl.flock`; note NFS is not safe |
| `collect.py` cannot import module | Import path issue | Add `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` at script top |
| MPI init failure | TRT-LLM requires MPI | `export OPAL_PREFIX=/opt/hpcx/ompi` |

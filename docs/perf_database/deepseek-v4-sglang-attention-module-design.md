# Design: DeepSeek-V4 SGLang Attention Module Data Collection

## Overview

This document describes the AIC changes for DeepSeek-V4 SGLang attention
performance data. The design replaces the default dependency on separately
collected sparse attention kernels with direct module-level collection for the
full DeepSeek-V4 attention path. It also defines the CSA top-k latency
correction applied inside the AIC performance database query path.

The implementation targets the SGLang backend and the DeepSeek-V4 Flash/Pro
model family:

```text
deepseek-ai/DeepSeek-V4-Flash
deepseek-ai/DeepSeek-V4-Pro
sgl-project/DeepSeek-V4-Flash-FP8
sgl-project/DeepSeek-V4-Pro-FP8
```

The default AIC path should query module-level CSA/HCA attention data. Sparse
kernel data remains available for experiments or old-model comparison, but it
is not required for the main DeepSeek-V4 module-level path.

## Motivation

The previous sparse-kernel correction flow had two practical problems:

1. It required maintaining separate collector interfaces for individual sparse
   kernels such as paged MQA logits and HCA attention.
2. The isolated sparse-kernel workload could diverge from the real module
   workload used by SGLang, making the correction fragile for long-context and
   mixed-prefix cases.

The new flow measures the full `self_attn` module for each DeepSeek-V4
attention kind. That module-level measurement includes the projection path,
norm/rope/cache work, CSA indexer/top-k work, compression-specific attention,
and final output path for the selected layer.

A small CSA top-k correction is still kept because the top-k portion can have a
systematic bandwidth-utilization mismatch between the collector workload and
real serving workload. The correction is applied in AIC after the module lookup.

## Data Files

The SGLang registry stages four module-level DeepSeek-V4 attention
CSV-formatted `*_perf.txt` files:

```text
dsv4_csa_context_module_perf.txt
dsv4_hca_context_module_perf.txt
dsv4_csa_generation_module_perf.txt
dsv4_hca_generation_module_perf.txt
```

`collect.py` finalizes those staging files as parquet. Packaged tables live
under
`aic-core/src/aiconfigurator_core/systems/data/<system>/sparse_attention/sglang/<version>/`
with the corresponding `*.parquet` names.

The ops and their canonical filenames are registered in:

```text
collector/sglang/registry.py
collector/registry_types.py
src/aiconfigurator/sdk/common.py
```

The four files are loaded and merged into the existing DeepSeek-V4 attention
module attributes in `PerfDatabase`:

```python
self._context_deepseek_v4_attention_module_data
self._raw_context_deepseek_v4_attention_module_data
self._generation_deepseek_v4_attention_module_data
```

This preserves the public query surface while changing the physical storage
from a legacy monolithic file to split CSA/HCA files.

## Collector Design

The collector entrypoint is:

```text
collector/sglang/collect_dsv4_attn.py
```

It builds an SGLang `ModelRunner` for one DeepSeek-V4 attention kind and times
CUDA graph replay of one layer's `self_attn(...)` call. The measured module
includes the attention kind's full module boundary, not only a single sparse
kernel.

The registry-facing worker is `run_dsv4_attn_worker`. A collect.py task is one
outer shape:

```text
(attn_kind, tp_size, gemm_type, batch_size)
```

The subprocess then sweeps valid sequence lengths internally. For context
collection, it also sweeps prefix lengths internally so the outer collect.py
case count stays small.

Manual examples:

```bash
python3 collector/sglang/collect_dsv4_attn.py \
  --model-path sgl-project/DeepSeek-V4-Flash-FP8 \
  --mode context \
  --attn-kind csa \
  --batch-sizes 1,4 \
  --seq-lens 128,1024 \
  --gemm-type fp8_block \
  --tp-sizes 8

python3 collector/sglang/collect_dsv4_attn.py \
  --model-path sgl-project/DeepSeek-V4-Flash-FP8 \
  --mode generation \
  --attn-kind hca \
  --batch-sizes 1,16 \
  --seq-lens 1024,8192 \
  --gemm-type fp8_block \
  --tp-sizes 8
```

Full collection for attention modules:

```bash
# From the repository root:
python3 collector/collect.py \
  --backend sglang \
  --model-path sgl-project/DeepSeek-V4-Flash-FP8 \
  --ops dsv4_csa_context_module dsv4_hca_context_module \
        dsv4_csa_generation_module dsv4_hca_generation_module
```

Collect the other required SGLang ops separately:

```bash
python3 collector/collect.py \
  --backend sglang \
  --model-path sgl-project/DeepSeek-V4-Flash-FP8 \
  --ops gemm moe mhc_module
```

Custom all-reduce collection, without NCCL collection:

```bash
bash collector/collect_comm.sh --all_reduce_backend sglang --skip-nccl
```

Smoke test before long collection:

```bash
python3 collector/collect.py \
  --backend sglang \
  --model-path sgl-project/DeepSeek-V4-Flash-FP8 \
  --ops gemm moe mhc_module dsv4_csa_context_module dsv4_hca_context_module \
        dsv4_csa_generation_module dsv4_hca_generation_module \
  --smoke
```

## Test-Case Grid

The shared DeepSeek-V4 test-case definitions live in:

```text
collector/common_test_cases.py
```

Supported attention models:

```text
deepseek-ai/DeepSeek-V4-Flash
deepseek-ai/DeepSeek-V4-Pro
sgl-project/DeepSeek-V4-Flash-FP8
sgl-project/DeepSeek-V4-Pro-FP8
```

Supported attention kinds:

```text
csa -> compress_ratio = 4
hca -> compress_ratio = 128
```

Module-level batch sizes:

```python
[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
```

Module-level sequence lengths:

```python
[
    1, 4, 8, 16, 32, 64, 128, 256, 512,
    1024, 1536, 2048, 3072, 4096, 6144, 8192,
    10240, 12288, 16384, 32768, 65536, 131072,
    262144, 524288, 1048575,
]
```

Context prefix lengths:

```python
[
    0, 1, 4, 8, 16, 32, 64, 128, 256, 512,
    1024, 1536, 2048, 3072, 4096, 6144, 8192,
    10000, 10240, 12288, 16384, 32768, 65536,
    131072, 262144, 524288, 1000000, 1048575,
]
```

TP sizes:

```python
[1, 2, 4, 8]
```

The shape filter enforces the SGLang/DeepSeek-V4 module limits:

```python
# context
bs * isl <= 8192
bs * (isl + past_kv) <= 1024 * 1024

# generation
bs * s <= 1024 * 1024
```

Generation additionally caps batch size for very long KV lengths to avoid
invalid or unrealistic memory pressure:

```python
s >= 524288 -> bs <= 1
s >= 262144 -> bs <= 2
s >= 131072 -> bs <= 4
s >= 65536  -> bs <= 8
s >= 32768  -> bs <= 16
s >= 8192   -> bs <= 64
```

The maximum generation sequence length is `1048575`, not `1048576`, because
the new token position equals the history length and `1048576` would index one
past the rotary embedding tables.

## Precision and Model Namespace Rules

The collector emits precision triples of:

```text
(compute_dtype, kv_cache_dtype, gemm_type)
```

DeepSeek-V4 rejects bf16 KV cache in this path, so KV cache collection uses
`fp8` only.

Current projection GEMM modes:

```text
bfloat16  -> cuBLASLt / nvjet-style projection path
fp8_block -> FP8 block-quantized projection path through DeepGEMM
```

Native `deepseek-ai/*` checkpoints contain FP4 routed experts. On pre-Blackwell
systems, the `fp8_block` sweep for these native checkpoints is skipped because
SGLang requires a native FP4 MoE backend before it can materialize those expert
weights. Converted `sgl-project/*-FP8` checkpoints can still collect
`fp8_block` on Hopper-class systems.

## TP and Head-Axis Convention

The AIC operation surface passes rank-local `num_heads`, computed by the model
as:

```python
local_heads = total_num_heads // tp_size
```

The collector stores the same rank-local `num_heads` value as the lookup axis.
SGLang may zero-pad Q back to the native 64-head shape inside the FlashMLA
backend, but that padding is an implementation detail of the module execution.
AIC lookup should continue to use the rank-local `num_heads` value because that
is what the operation query passes.

This keeps DeepSeek-V4 consistent with the other attention module data and
avoids making TP an interpolation axis.

## Loader Design

Context loader:

```python
load_context_dsv4_kind_module_data(file_path)
```

Context dictionary shape:

```python
data[fmha_quant][kv_quant][gemm_quant][architecture][compress_ratio]
    [num_heads][prefix][s][batch] = {
        "latency": ms,
        "power": W,
        "energy": W * ms,
    }
```

For context CSV files, the `step` column stores the prefix length.

Generation loader:

```python
load_generation_dsv4_kind_module_data(file_path)
```

Generation dictionary shape:

```python
data[kv_quant][gemm_quant][architecture][compress_ratio]
    [num_heads][batch][s_total] = {
        "latency": ms,
        "power": W,
        "energy": W * ms,
    }
```

For generation CSV files, `s_total = isl + step`. Decode is modeled as one new
query token with a past-KV length equal to `step`.

The `PerfDatabase` constructor loads the split files and deep-merges them:

```python
ctx_split = [
    _load_op_data(PerfDataFilename.dsv4_csa_context_module),
    _load_op_data(PerfDataFilename.dsv4_hca_context_module),
]
gen_split = [
    _load_op_data(PerfDataFilename.dsv4_csa_generation_module),
    _load_op_data(PerfDataFilename.dsv4_hca_generation_module),
]
```

The merged result is assigned to the existing DeepSeek-V4 attention module
attributes used by the SDK query methods.

## Query Design

### Context Query

Context uses prefix-aware module data. The lookup path is:

1. Select the requested attention kind by `compress_ratio`.
2. Select the rank-local `num_heads` axis.
3. Select or interpolate the prefix axis.
4. Inside a prefix slice, query the `(num_heads, s, batch)` grid.
5. Apply CSA top-k correction when `compress_ratio == 4`.

For exact prefix hits, AIC queries that prefix slice directly. If the exact
prefix is missing, AIC searches nearby prefix anchors that can actually answer
the requested `(s, batch)` shape, then interpolates or extrapolates along the
prefix axis.

For CSA, the context query first tries the raw top-k piecewise interpolation
around the compressed top-k boundary when raw data is available. If that does
not return a finite value, it falls back to robust module lookup.

### Generation Query

Generation data is keyed as:

```python
[num_heads][batch][s_total]
```

The generation query performs robust 3D lookup over:

```text
num_heads, batch, s_total
```

For DeepSeek-V4 Flash, `num_heads` is a fixed lookup key from the rank-local
operation input. It is not meant to model arbitrary TP interpolation.

CSA generation applies the same top-k correction as context, with:

```python
query_len = 1
prefix = s_total
```

HCA generation does not apply the CSA top-k correction.

## Robust Lookup and Extrapolation

The collected module grid is intentionally sparse at high batch sizes and long
sequence lengths. If cubic interpolation fails, AIC uses a robust lookup path
that can fall back to smaller-batch sampled data and extrapolate by batch ratio
when the larger requested batch is outside the available grid.

This behavior is required for shapes such as:

```text
bs = 3, isl = 2682, prefix = 0
```

where collected data may contain `bs=2` at longer sequence lengths and `bs=4`
only at shorter sequence lengths.

The extrapolation fallback must preserve all finite fields (`latency`, `power`,
`energy`) instead of silently zeroing missing energy fields.

## CSA Top-K Correction

The full CSA module includes top-k score processing. The collector and real
serving path can have different effective memory bandwidth utilization for the
same logical top-k score traffic. AIC corrects the module latency after the
module lookup.

Constants:

```python
_DSV4_CSA_TOPK_SCORE_BYTES = 4
_DSV4_CSA_TOPK_AIC_BW_UTIL = 0.0008  # 0.08 percent
_DSV4_CSA_TOPK_REAL_BW_UTIL = 0.03    # 3 percent
```

The score byte count is computed by:

```python
score_bytes = _dsv4_csa_topk_score_bytes(
    batch_size=b,
    query_len=query_len,
    prefix=prefix,
    ratio=compress_ratio,
    topk=index_topk,
)
```

For integer inputs, the exact compressed score element count is:

```python
start = max(prefix + 1, (topk + 1) * ratio)
end = prefix + query_len
score_elems = sum(floor(t / ratio) for t in range(start, end + 1))
score_bytes = batch_size * score_elems * 4
```

If the input is non-integer because the caller used averaged request shapes,
AIC uses the continuous approximation:

```python
score_elems = (end * end - start * start) / (2 * ratio)
score_bytes = batch_size * score_elems * 4
```

The latency delta is:

```python
collected_ms = score_bytes / (mem_bw * AIC_BW_UTIL) * 1000
real_ms = score_bytes / (mem_bw * REAL_BW_UTIL) * 1000
delta_ms = real_ms - collected_ms
```

The corrected latency is clamped at zero:

```python
corrected_latency = max(0.0, latency + delta_ms)
```

If energy exists, it is scaled by the same latency ratio:

```python
energy *= corrected_latency / latency
```

Context and generation use the same helper with different query lengths:

```python
# context / prefill
query_len = s
prefix = prefix

# generation / decode
query_len = 1
prefix = s_total
```

The correction applies only to CSA (`compress_ratio == 4`). HCA uses
`compress_ratio == 128` and has no CSA top-k path.

## Commands for the Modified Path

Collect all SGLang DeepSeek-V4 Flash FP8 module-level attention data:

```bash
# From the repository root:
python3 collector/collect.py \
  --backend sglang \
  --model-path sgl-project/DeepSeek-V4-Flash-FP8 \
  --ops dsv4_csa_context_module dsv4_hca_context_module \
        dsv4_csa_generation_module dsv4_hca_generation_module
```

Collect non-attention dependencies:

```bash
python3 collector/collect.py \
  --backend sglang \
  --model-path sgl-project/DeepSeek-V4-Flash-FP8 \
  --ops gemm moe mhc_module
```

Collect custom all-reduce only:

```bash
bash collector/collect_comm.sh --all_reduce_backend sglang --skip-nccl
```

Run focused unit tests for the DeepSeek-V4 module path:

```bash
PYTHONPATH=src \
python3 -m pytest tests/unit/sdk/database/test_deepseek_v4_module.py -q
```

Run only the CSA top-k correction tests:

```bash
PYTHONPATH=src \
python3 -m pytest tests/unit/sdk/database/test_deepseek_v4_module.py \
  -k "csa_topk_bandwidth_delta" -q
```

## Validation Expectations

The test suite should cover:

- Split CSA/HCA file loading and merging into existing SDK attributes.
- Context prefix-aware lookup with exact prefix and interpolated prefix cases.
- Generation lookup below the minimum sampled `s_total`.
- Missing-batch robust lookup and smaller-batch extrapolation.
- CSA top-k correction for both context and generation.
- Energy scaling after latency correction.
- DeepSeek-V4 Flash/Pro model namespace coverage for collector test cases.

The current focused test command is expected to pass:

```text
33 passed
```

for `tests/unit/sdk/database/test_deepseek_v4_module.py`.

## Compatibility and Migration

The SDK query surface remains unchanged:

```python
query_context_deepseek_v4_attention_module(...)
query_generation_deepseek_v4_attention_module(...)
```

The storage changes are internal to `PerfDatabase` loading and merging. Existing
callers continue to query by model config, runtime config, and database mode.

The legacy sparse-kernel files are still present as optional collector outputs:

```text
dsv4_flash_paged_mqa_logits_module_perf.txt
dsv4_flash_hca_attn_module_perf.txt
```

They are no longer required for the default DeepSeek-V4 SGLang module-level
attention path.

## Open Tuning Knobs

The CSA top-k bandwidth constants are empirical and should be revisited when:

- the top-k kernel implementation changes,
- the collector workload changes,
- the serving workload changes materially,
- the target GPU generation changes,
- or a new DeepSeek-V4 checkpoint changes the CSA indexer path.

The current constants are intentionally centralized in `perf_database.py` so
future tuning does not require changing loader or collector semantics.

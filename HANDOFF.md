# Handoff: Layerwise Collector TP2

Date: 2026-06-05
Branch: `dev-layerwise-collector`
Repo: `aiconfigurator`

## Current State

- FPM ground truth collection has been rewritten to collect context, decode, and mixed traffic in one Dynamo/vLLM deployment.
- Repo-local skills were added under `.agents/skills/`:
  - `.agents/skills/collect-fpm-ground-truth`
  - `.agents/skills/collect-vllm-layerwise`
- The skills use portable env vars (`AIC_REPO`, `AIC_LAYERWISE_ARTIFACTS`, `HF_HOME`, `HF_TOKEN` / `HF_TOKEN_FILE`, `NSYS_VERSION`) instead of machine-specific paths.
- Home-dir copies of those skills were removed. Codex should discover repo skills from `.agents/skills` when launched from this repo or a subdirectory; restart Codex if new skills do not appear.

## Important Gotcha

Use `NSYS_VERSION=2025.3.2` or newer on B300. The TP2 decode run with Nsight `2024.6.2` produced a SQLite export with NVTX events but no `CUPTI_ACTIVITY_KIND_KERNEL` table, so the parser emitted zero rows. Re-running with `2025.3.2` fixed it.

## TP2 FPM Ground Truth

Artifacts are under `${AIC_LAYERWISE_ARTIFACTS}`. Keep this artifact directory outside the repo unless the user explicitly wants raw traces checked in.

Completed TP2 FPM sweep:

- `qwen3_32b_tp2_fpm_sweep_b300_vllm0201.csv`
- `qwen3_32b_tp2_fpm_sweep_b300_vllm0201_detail.csv`
- `qwen3_32b_tp2_fpm_sweep_b300_vllm0201_phase.csv`
- `qwen3_32b_tp2_fpm_sweep_b300_vllm0201_workload.csv`
- `qwen3_32b_tp2_fpm_sweep_b300_vllm0201_warmup_workload.csv`

Phase rows:

- `context`: 63
- `decode`: 325
- `mixed`: 97
- total: 485

Trimmed FPM context latencies:

| ctx_tokens | ctx_kv_tokens | fpm_ms |
| ---: | ---: | ---: |
| 1 | 0 | 9.732 |
| 64 | 0 | 10.109 |
| 256 | 0 | 14.165 |
| 1024 | 0 | 31.420 |
| 2048 | 0 | 57.300 |
| 4096 | 0 | 116.534 |
| 8192 | 0 | 240.091 |
| 8192 | 8192 | 294.284 |
| 16384 chunked sum | 0 + 8192 | 534.375 |

Trimmed FPM decode latencies, filtering `mean_decode_kv_tokens` to roughly 1024-1056:

| decode_requests | fpm_ms |
| ---: | ---: |
| 1 | 8.089 |
| 2 | 8.216 |
| 4 | 8.247 |
| 8 | 8.307 |
| 16 | 8.573 |
| 32 | 9.109 |
| 64 | 9.977 |

## TP2 Layerwise Data Collected

Validated TP2 decode layerwise, one layer, span latency:

- `qwen3_32b_tp2_vllm_decode_b1_64_past1024_span_nsys2025.csv`
- profile dir: `profiles/vllm_decode_qwen32b_tp2_b1_64_past1024_span_nsys2025`

Rows:

| batch_size | per_layer_ms |
| ---: | ---: |
| 1 | 0.142687 |
| 2 | 0.143935 |
| 4 | 0.145087 |
| 8 | 0.147679 |
| 16 | 0.150943 |
| 32 | 0.160095 |
| 64 | 0.172191 |

Validated TP2 context layerwise, 16 layers, `gpu_capped`, warmup 2, measured 6, trimmed mean:

- `qwen3_32b_tp2_vllm_context_b300_gpu_capped_w2m6_trimmed_16layers_nsys2025.csv`
- `qwen3_32b_tp2_vllm_context_8192past8192_gpu_capped_w2m6_trimmed_16layers_nsys2025.csv`

The AIC table stores per-layer values, so these context rows were divided by 16 before insertion.

| new_tokens | past_kv | full_16_layer_ms | per_layer_ms |
| ---: | ---: | ---: | ---: |
| 1 | 0 | 2.24978425 | 0.140611515625 |
| 64 | 0 | 2.336568 | 0.1460355 |
| 256 | 0 | 2.789716 | 0.17435725 |
| 1024 | 0 | 6.52044075 | 0.407527546875 |
| 2048 | 0 | 12.294067 | 0.7683791875 |
| 4096 | 0 | 25.37665875 | 1.586041171875 |
| 8192 | 0 | 55.14070275 | 3.446293921875 |
| 8192 | 8192 | 67.16987325 | 4.198117078125 |

## AIC Data Updated

Updated:

- `src/aiconfigurator/systems/data/b300_sxm/vllm/0.20.1/layerwise_perf.csv`

Current row counts:

- `CTX,tp=1`: 8
- `GEN,tp=1`: 7
- `CTX,tp=2`: 8
- `GEN,tp=2`: 7
- total: 30

Focused tests passed:

```bash
uv run --extra dev pytest tests/test_layerwise_vllm_collect.py tests/unit/sdk/database/test_layerwise.py tests/unit/sdk/backends/test_vllm_layerwise_backend.py -q
```

Result: `29 passed`.

## AIC vs FPM Snapshot

Use `PerfDatabase(..., database_mode="HYBRID")` for comparison, because `vllm/0.20.1` currently only has layerwise data and needs inherited custom-allreduce data.

Context comparison is good:

- MAPE across collected context chunks: about 2.07%
- MAPE excluding tiny `ctx=1`: about 2.10%
- 16k chunked prefill: AIC `532.952 ms` vs FPM chunked sum `534.375 ms`, APE `0.27%`

Decode comparison is not good yet:

- Full AIC path overpredicts by about 24.43% MAPE.
- Layerwise-only decode (`per_layer_ms * 64`) is already 10-14% high.
- Custom allreduce adds another about 0.9-1.4 ms depending batch size, making total error about 24%.

Decode breakdown from today:

| batch | layer_only_ms | allreduce_ms | total_aic_ms | fpm_ms |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 9.132 | 0.914 | 10.046 | 8.089 |
| 2 | 9.212 | 0.956 | 10.167 | 8.216 |
| 4 | 9.286 | 0.973 | 10.259 | 8.247 |
| 8 | 9.451 | 0.994 | 10.445 | 8.307 |
| 16 | 9.660 | 1.007 | 10.667 | 8.573 |
| 32 | 10.246 | 1.064 | 11.310 | 9.109 |
| 64 | 11.020 | 1.384 | 12.404 | 9.977 |

Note: for this direct decode comparison, the helper used `isl=1023, osl=2` so `_run_generation_phase()` queries the collected `seq_len_kv_cache=1024` GEN table row.

## Interrupted Work

An exploratory TP2 16-layer decode run was started and then interrupted:

- `qwen3_32b_tp2_vllm_decode_b1_64_past1024_span_16layers_nsys2025.csv`
- `profiles/vllm_decode_qwen32b_tp2_b1_64_past1024_span_16layers_nsys2025`

The container was stopped. The CSV has zero rows, so treat this as partial/invalid. Use a fresh output/work-dir suffix if retrying.

## Recommended Next Steps

1. Rerun TP2 decode with more than one layer, preferably 16 layers, using a fresh suffix such as `_16layers_retry_nsys2025`.
2. Divide the 16-layer decode latencies by 16 and compare layer-only decode against FPM before touching the AIC table.
3. If 16-layer decode removes the 10-14% layer-only overprediction, replace the TP2 `GEN` rows in `layerwise_perf.csv`.
4. If decode is still high after 16-layer collection, investigate whether decode should use `gpu_capped` instead of `span`, and separately validate the TP custom-allreduce term.
5. After decode is fixed, run the AIC comparison again for context, decode, and mixed FPM rows.

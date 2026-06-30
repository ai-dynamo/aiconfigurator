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
- `CTX,tp=8`: 8
- `GEN,tp=8`: 7
- total: 45

Focused tests passed:

```bash
uv run --extra dev pytest tests/test_layerwise_vllm_collect.py tests/unit/sdk/database/test_layerwise.py tests/unit/sdk/backends/test_vllm_layerwise_backend.py -q
```

Result from the earlier TP2-only state: `29 passed`. Current focused result is in the default-compile update below.

## Older AIC vs FPM Snapshot (Superseded)

The default-compile update below supersedes these decode conclusions.

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

Superseded note: this older direct decode comparison used `isl=1023, osl=2` as a workaround. Current code fixes generation KV lookup with `kv_len = isl + i`, so use `isl=1024, osl=2` when comparing to FPM decode at `mean_decode_kv_tokens ~= 1024`.

## Interrupted Work

An exploratory TP2 16-layer decode run was started and then interrupted:

- `qwen3_32b_tp2_vllm_decode_b1_64_past1024_span_16layers_nsys2025.csv`
- `profiles/vllm_decode_qwen32b_tp2_b1_64_past1024_span_16layers_nsys2025`

The container was stopped. The CSV has zero rows, so treat this as partial/invalid. Use a fresh output/work-dir suffix if retrying.

## 2026-06-05 Default-Compile Update

Root cause for the TP1/TP2 decode regression was a compile-mode mismatch:

- FPM sweeps used vLLM defaults.
- The original layerwise GEN collector forced `--compilation-config {"mode":0,...}` so module NVTX attribution worked.
- A focused TP1 compile-off FPM run showed compile-off FPM and compile-off layerwise match within about 0.5-2.3%, while compile-off FPM is about 13-15% slower than default FPM.
- With vLLM defaults, module ranges collapse under `CUDAGraphWrapper`; for the one-layer mock, using that wrapper range as the full layer gives the right default-compile layer total.

New default-compile GEN artifacts:

- `qwen3_32b_tp1_vllm_decode_b1_64_past1024_span_default_compile_cudagraphwrapper_nsys2025.csv`
- `qwen3_32b_tp2_vllm_decode_b1_64_past1024_span_default_compile_cudagraphwrapper_nsys2025.csv`
- `qwen3_32b_tp8_vllm_decode_b1_64_past1024_span_default_compile_cudagraphwrapper_nsys2025.csv`

Collector/backend changes in this working tree:

- `collect_layerwise.py` now uses default vLLM compilation and `CUDAGraphWrapper` rollup directly, with repeated GEN warmup/measured passes.
- `vllm_step_marker.py` tags repeated GEN runs with `::runN`.
- `vllm_backend.py` falls back for sparse GEN KV rows in mixed estimates and for missing chunked CTX nonzero-KV rows when a direct long-context row exists.

Focused tests passed:

```bash
uv run --extra dev pytest tests/test_layerwise_vllm_collect.py tests/unit/sdk/database/test_layerwise.py tests/unit/sdk/backends/test_vllm_layerwise_backend.py -q
```

Result: `33 passed`.

Current FPM comparison summary, using median FPM decode rows with `mean_decode_kv_tokens ~= 1024`:

| TP | context backend MAPE | decode layer-only MAPE | decode backend MAPE | mixed backend MAPE |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 23.81%* | 2.87% | 2.87% | 16.57% |
| 2 | 1.20% | 3.59% | 8.29% | 6.59% |
| 8 | 12.63% | 10.86% | 25.37% | 4.35% |

`*` TP1 context MAPE is dominated by anomalously low FPM rows at `ctx=64/256`; long-context TP1 rows are within about 1.5-5.2%, and the 16k chunked comparison is AIC `970.264 ms` vs FPM `984.688 ms` (`-1.46%`).

Decode detail after default-compile GEN replacement:

| TP | batch | FPM ms | layer-only ms | allreduce ms | backend ms | backend err |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 13.076 | 13.148 | 0.000 | 13.148 | 0.55% |
| 1 | 64 | 16.202 | 17.034 | 0.000 | 17.034 | 5.13% |
| 2 | 1 | 8.236 | 7.742 | 0.914 | 8.656 | 5.10% |
| 2 | 64 | 10.142 | 9.992 | 1.384 | 11.376 | 12.17% |
| 8 | 1 | 4.444 | 3.881 | 0.994 | 4.875 | 9.69% |
| 8 | 64 | 5.941 | 4.948 | 3.387 | 8.335 | 40.30% |

The remaining decode issue is no longer the TP2 layer measurement. It is the standalone custom-allreduce term: it helps context, but for default-compile decode it overcorrects, especially TP8 and large batches. The next modeling step should be an overlapped/fused decode communication treatment rather than more FPM collection.

Avoid adding Python-level NVTX markers around attention for default-compile attribution unless necessary. They are likely to disappear into the compiled `CUDAGraphWrapper`, cause graph breaks, or perturb the graph. If attention-only attribution becomes necessary, add markers at a lower boundary that survives compilation and validate with an A/B perf check.

## 2026-06-05 Parity Hardening And Decode Comm Update

Implemented the deployment-parity hardening items:

- Added shared vLLM deployment metadata/config helpers in `collector/layerwise/common/vllm_deployment.py`.
- `collect_layerwise.py` now uses the deployment-parity path only: vLLM compile/CUDA graph defaults with `CUDAGraphWrapper` rollup.
- New `collect_layerwise.py` output rows include representative layer metadata: `layer_type`, `layer_index`, `measured_layer_count`, and `layer_multiplier`. `latency_ms` remains the raw measured representative span; AIC scales by `layer_multiplier / measured_layer_count` when those columns are present and falls back to old full-layer-count scaling for legacy rows.
- FPM collection now uses the same vLLM arg helper, writes requested/effective vLLM config metadata, and can optionally start the worker under Nsight Systems with delayed start/stop around measured traffic.
- Added `collector/layerwise/diagnostics/analyze_nsys_comm_overlap.py` for Nsight SQLite comm/compute overlap summaries.
- `layerwise_perf.csv` no longer needs separate measurement-mode columns for new collector output.
- `VLLMBackend` now treats pure `GEN` deployment-parity rows as the decode prediction and no longer adds the inherited standalone custom-allreduce table to pure decode. Context and mixed estimates still keep explicit TP allreduce because those paths use CTX layer attribution rows.

Targeted worker Nsight runs were decode-focused. Each run still contains context/mixed/decode FPM phase rows because that is how vLLM reaches steady decode, but the overlap analysis isolates the steady full-batch decode windows:

- TP2 run: `runs/fpm_tp2_b64_nsys_20260605_200517`
- TP8 run: `runs/fpm_tp8_b64_nsys_20260605_201130`

Steady full-batch decode overlap from worker traces:

| TP | FPM decode row | compute union | comm union | overlap | visible comm | visible comm / comm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | ~10.1 ms | 9.155 ms | 1.058 ms | 0.119 ms | 0.939 ms | 88.9% |
| 8 | ~6.06 ms | 4.265 ms | 1.963 ms | 0.168 ms | 1.795 ms | 91.5% |

These are medians over steady full-batch decode windows (`27` TP2 windows, `25` TP8 windows) selected from the worker Nsight traces with a 10 us idle-gap segmenter.

So the remaining gap is not "communication hidden by compute"; most decode comm is visible in the real worker. The old backend was bad because it added a generic standalone custom-allreduce correction on top of the simulated one-GPU layerwise compute span:

| TP | batch | FPM ms | layerwise ms | old allreduce ms | old backend err | new backend err |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 13.076 | 13.148 | 0.000 | 0.55% | 0.55% |
| 1 | 64 | 16.232 | 17.034 | 0.000 | 4.94% | 4.94% |
| 2 | 1 | 8.236 | 7.742 | 0.914 | 5.10% | 6.00% |
| 2 | 64 | 10.142 | 9.992 | 1.384 | 12.17% | 1.48% |
| 8 | 1 | 4.444 | 3.881 | 0.994 | 9.69% | 12.67% |
| 8 | 64 | 5.941 | 4.948 | 3.387 | 40.30% | 16.71% |

Decode MAPE on cached FPM sweeps after the backend change:

| TP | old backend MAPE | new backend MAPE |
| ---: | ---: | ---: |
| 1 | 2.51% | 2.51% |
| 2 | 8.29% | 3.59% |
| 8 | 25.37% | 10.86% |

Important interpretation: the new pure-decode backend is intentionally layerwise-only, not because real comm is zero, but because the available standalone allreduce table is not calibrated to vLLM default-compile decode's fused/NCCL communication path. A better future model should collect or derive a vLLM decode-specific comm residual instead of reusing generic custom-allreduce.

Focused tests now include deployment metadata, Nsight overlap, collector, database, and backend coverage.

## 2026-06-05 vLLM AllReduce+RMS Table

Added a separate fused vLLM allreduce+rms data path. This is intentionally not mixed into `custom_allreduce_perf.parquet`, which remains the standalone `tensor_model_parallel_all_reduce` table.

Code/data changes:

- Collector: `collector/network/collect_all_reduce.py --backend vllm --op allreduce_rms`
- New perf file enum: `allreduce_rms_perf.parquet`
- New query API: `PerfDatabase.query_allreduce_rms(quant_mode, tp_size, size, hidden_size, fusion_pattern="allreduce_residual_rms")`
- Installed table: `src/aiconfigurator/systems/data/b300_sxm/vllm/0.20.1/allreduce_rms_perf.parquet`

Table schema includes `hidden_size`, `num_tokens`, `fusion_pattern`, and `shape_policy`. The collector currently always uses `hidden_size=4096` and marks rows with `shape_policy=hidden_size_4096_approx`. Query logic chooses the nearest available hidden size before interpolating by message size, so future collections can add more hidden sizes without changing the AIC API.

Collection artifacts:

- Run dir: `runs/allreduce_rms_vllm0201_h4096_20260605_221017`
- Combined CSV: `allreduce_rms_vllm0201_h4096_combined.csv`
- Plot: `allreduce_old_new_rms_comparison.png`
- Comparison CSV: `allreduce_old_new_rms_comparison.csv`
- Qwen interpolation CSV: `qwen3_32b_tp8_b64_allreduce_rms_interpolation.csv`

Collected rows:

| TP | hidden_size | min message elems | max message elems | rows |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 4096 | 4096 | 33554432 | 14 |
| 4 | 4096 | 4096 | 33554432 | 14 |
| 8 | 4096 | 4096 | 1048576 | 9 |

Qwen3-32B TP8 batch-64 decode message size is `64 * 5120 = 327680` elements. With nearest-hidden lookup selecting the 4096 bucket, AIC query returns:

| source | latency |
| --- | ---: |
| old vLLM 0.19 generic allreduce graph | 0.026463 ms |
| new vLLM 0.20.1 generic allreduce graph | 0.026227 ms |
| new vLLM 0.20.1 allreduce+rms h=4096 graph | 0.013074 ms |

So for the Qwen TP8 point, the fused allreduce+rms table is about `0.50x` the generic allreduce table. Across 128-129 fused collectives this is roughly `1.68 ms` of fused comm instead of `3.38 ms` from generic allreduce.

## 2026-06-06 FP8 Layerwise vs FPM Status

FP8 model: `Qwen/Qwen3-32B-FP8` on `b300_sxm`, vLLM `0.20.1`.

New FPM artifacts:

- TP1 FPM: `qwen3_32b_fp8_tp1_fpm_sweep_b300_vllm0201_phase.csv`
- TP2 FPM run: `runs/fpm_fp8_tp2_20260606_012703/fpm_metrics_phase.csv`
- TP8 FPM run: `runs/fpm_fp8_tp8_20260606_011527/fpm_metrics_phase.csv`
- Combined AIC comparison: `qwen3_32b_fp8_tp1_tp2_tp8_aic_vs_fpm.csv`
- Combined summary: `qwen3_32b_fp8_tp1_tp2_tp8_aic_vs_fpm_summary.json`

Code fix made during the FP8 pass:

- vLLM layerwise GEN lookup now uses the KV length at the start of the decode iteration (`kv_len = isl + i`), matching FPM's first decode row at `mean_decode_kv_tokens=1024` for a `past_kv=1024` workload. The previous `isl + i + 1` lookup missed the collected `GEN` row and was semantically off by one.

Current FP8 accuracy after the TP1-derived context calibration and fused allreduce+rms decode path:

| TP | context MAPE | context MAPE, new >= 1024 | decode MAPE | overall MAPE |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.35% | 0.53% | 2.86% | 1.45% |
| 2 | 4.13% | 4.42% | 2.37% | 3.36% |
| 8 | 10.43% | 4.40% | 8.28% | 9.49% |

Important interpretation:

- TP1 is effectively closed for both context and decode.
- TP2 is in good shape overall; context's largest point error is `+11.7%` at `new_tokens=2048`.
- TP8 long-context behavior is good (`4.4%` MAPE for `new_tokens >= 1024`), but small context rows are noisy and dominate the all-context MAPE.
- TP8 decode still underpredicts small batches (`-10%` to `-12%` for b1-b8). The breakdown shows an effective residual of about `1.3 ms` beyond layerwise compute for b1-b32, while the current `hidden_size=4096` approximate allreduce+rms table contributes only about `0.75-1.17 ms`. Batch 64 is close (`-1.9%`).

## Recommended Next Steps

1. Treat FPM TP1/TP2/TP8 sweeps as collected; avoid recollecting unless the workload or vLLM defaults change.
2. Keep the default-compile `CUDAGraphWrapper` GEN rows in `layerwise_perf.csv`.
3. Use `allreduce_rms_perf.parquet` for fused decode comm attribution when model analysis says a row-parallel allreduce is followed by RMSNorm. Keep pure deployment-parity GEN latency layerwise-only unless/until the backend has a non-double-counting attribution path.
4. If context accuracy becomes important beyond long-context rows, collect/parse default-compile context wrapper rows or reclassify the noisy small-context FPM rows before tuning against them.

## 2026-06-06 End-of-Day Handoff

Current working state:

- Branch: `dev-layerwise-collector`
- Model under active investigation: `Qwen/Qwen3-32B-FP8`
- Backend/system: vLLM `0.20.1` on `b300_sxm`
- FPM context/decode sweeps are complete for TP1, TP2, and TP8.
- All FPM containers were stopped after the final TP2 run; GPUs were free at last check.

Important code/data changes made today:

- Fixed vLLM layerwise decode KV lookup in `src/aiconfigurator/sdk/backends/vllm_backend.py`: pure GEN now queries `kv_len = isl + i`, matching FPM's first decode row at `mean_decode_kv_tokens=1024` for a `past_kv=1024` workload.
- Merged calibrated FP8 rows into `src/aiconfigurator/systems/data/b300_sxm/vllm/0.20.1/layerwise_perf.csv`, so repo-root AIC now supports both `Qwen/Qwen3-32B` and `Qwen/Qwen3-32B-FP8`.
- Copied/kept fused allreduce+rms data in `allreduce_rms_perf.parquet`; query path uses nearest hidden size. Current collected hidden-size bucket is still `4096`, so Qwen hidden size `5120` is an approximation.
- Updated skills under `.agents/skills/`:
  - `collect-fpm-ground-truth`
  - `collect-vllm-layerwise`
  - `close-layerwise-fpm-gap`

Final FP8 AIC vs FPM comparison:

| TP | context MAPE | context MAPE, new >= 1024 | decode MAPE | overall MAPE |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.35% | 0.53% | 2.86% | 1.45% |
| 2 | 4.13% | 4.42% | 2.37% | 3.36% |
| 8 | 10.43% | 4.40% | 8.28% | 9.49% |
| all | - | - | - | 4.77% |

Primary artifacts:

- Combined comparison: `/home/shadeform/layerwise-artifacts/qwen3_32b_fp8_tp1_tp2_tp8_aic_vs_fpm.csv`
- Combined summary: `/home/shadeform/layerwise-artifacts/qwen3_32b_fp8_tp1_tp2_tp8_aic_vs_fpm_summary.json`
- TP1 FPM: `/home/shadeform/layerwise-artifacts/qwen3_32b_fp8_tp1_fpm_sweep_b300_vllm0201_phase.csv`
- TP2 FPM: `/home/shadeform/layerwise-artifacts/runs/fpm_fp8_tp2_20260606_012703/fpm_metrics_phase.csv`
- TP8 FPM: `/home/shadeform/layerwise-artifacts/runs/fpm_fp8_tp8_20260606_011527/fpm_metrics_phase.csv`
- FP8 layerwise context: `/home/shadeform/layerwise-artifacts/qwen3_32b_fp8_tp1_tp2_tp8_vllm_context_2d_prefix_cache_new1_8192_past0_65536_model_cap_b300_nsys2025.csv`
- FP8 layerwise decode: `/home/shadeform/layerwise-artifacts/qwen3_32b_fp8_tp1_tp2_tp8_vllm_decode_b1_64_past1024_span_default_compile_cudagraphwrapper_nsys2025.csv`

Residual issue:

- TP8 decode underpredicts small batches by about `10-12%` for b1-b8. Batch 64 is close at about `-1.9%`.
- The likely next modeling improvement is collecting or deriving fused allreduce+rms data for hidden size `5120` instead of relying on the current hidden-size `4096` approximation, then rerunning the TP8 decode comparison.
- TP8 context is acceptable for long context (`4.40%` MAPE for `new_tokens >= 1024`); small context rows are noisy/fixed-overhead dominated and inflate all-context MAPE.

Validation run after the final data/code changes:

```bash
.venv/bin/python -m pytest \
  tests/unit/sdk/backends/test_vllm_layerwise_backend.py \
  tests/unit/sdk/database/test_layerwise.py \
  tests/unit/sdk/database/test_base_queries.py \
  tests/unit/sdk/database/test_data_loaders.py \
  -q
```

Result: `69 passed`.

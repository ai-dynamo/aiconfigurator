# Today Learnings: Layerwise vs FPM Gap Closure

These notes capture the B300/vLLM 0.20.1 Qwen3-32B/Qwen3-32B-FP8 lessons from the 2026-06-06 session.

## Artifact Discipline

- Prefer existing artifacts; do not recollect FPM by default.
- FPM can exit nonzero after preserving useful rows. Inspect `*_phase.csv` and `*_workload.csv` before rerunning.
- Confirm no accidental containers are running after interruptions: `docker ps`, `docker ps -a`, `nvidia-smi`.
- A failed TP2 FP8 FPM start using `--gpus device=0,1` did not collect measurements; Docker rejected the GPU spec before the worker ran. Use `--gpus '"device=0,1"'`.

## FP8 Model Setup

- FP8 model ID: `Qwen/Qwen3-32B-FP8`.
- Downloaded snapshot: `/home/shadeform/.cache/huggingface/hub/models--Qwen--Qwen3-32B-FP8/snapshots/aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df`.
- Default FP8 checkpoint behavior should not force KV FP8 unless explicitly measuring that mode. In the layerwise collector, `--kv-quant fp8` forces `--kv-cache-dtype fp8`; for default vLLM FP8 checkpoint behavior, label the row with `--kv-quant bf16` so KV dtype stays `auto`.
- FP8 effective config enabled `+quant_fp8` custom ops and `norm_quant, act_quant` fusions.

## Collected FP8 Layerwise Data

Context:

- File: `/home/shadeform/layerwise-artifacts/qwen3_32b_fp8_tp1_tp2_tp8_vllm_context_2d_prefix_cache_new1_8192_past0_65536_model_cap_b300_nsys2025.csv`
- TP sizes: 1, 2, 8
- Rows: 711 total, 237 per TP
- Measurement: 16-layer context, `gpu_capped`, `ctx_driver=prefix_cache`, `ctx_warmup_runs=2`, `ctx_measured_runs=6`, trimmed mean
- Runtime: about 9.44 minutes wall-clock for TP1/TP2/TP8 in parallel
- AIC normalization: divide `latency_ms` and `rms_latency_ms` by 16 before loading

Decode:

- File: `/home/shadeform/layerwise-artifacts/qwen3_32b_fp8_tp1_tp2_tp8_vllm_decode_b1_64_past1024_span_default_compile_cudagraphwrapper_nsys2025.csv`
- TP sizes: 1, 2, 8
- Rows: 21 total, 7 per TP
- Measurement: one-layer decode, `span`, default compile, CUDAGraphWrapper attribution, batch sizes 1..64, past KV 1024
- Runtime: about 1.42 minutes wall-clock
- AIC normalization: do not divide one-layer decode rows

## FP8 FPM Status

- TP1 FPM produced usable context and decode rows.
- Files:
  - `/home/shadeform/layerwise-artifacts/qwen3_32b_fp8_tp1_fpm_sweep_b300_vllm0201_phase.csv`
  - `/home/shadeform/layerwise-artifacts/qwen3_32b_fp8_tp1_fpm_sweep_b300_vllm0201_detail.csv`
  - `/home/shadeform/layerwise-artifacts/qwen3_32b_fp8_tp1_fpm_sweep_b300_vllm0201_workload.csv`
  - `/home/shadeform/layerwise-artifacts/qwen3_32b_fp8_tp1_fpm_sweep_b300_vllm0201_effective_vllm_config.json`
- Phase rows: context 82, decode 276, mixed 39.
- Context and decode request workloads completed.
- Mixed had one vLLM internal Unicode/char-boundary error. Treat mixed as partial; context/decode remain usable.
- TP2 FPM run: `/home/shadeform/layerwise-artifacts/runs/fpm_fp8_tp2_20260606_012703/fpm_metrics_phase.csv`
- TP8 FPM run: `/home/shadeform/layerwise-artifacts/runs/fpm_fp8_tp8_20260606_011527/fpm_metrics_phase.csv`
- TP2 and TP8 context/decode completed. Mixed was intentionally not recollected for the final FP8 comparison.

## Current FP8 AIC Comparison

The final comparison uses the normal repo systems root. FP8 rows have been merged into `src/aiconfigurator/systems/data/b300_sxm/vllm/0.20.1/layerwise_perf.csv`.

Artifacts:

- Combined comparison: `/home/shadeform/layerwise-artifacts/qwen3_32b_fp8_tp1_tp2_tp8_aic_vs_fpm.csv`
- Summary: `/home/shadeform/layerwise-artifacts/qwen3_32b_fp8_tp1_tp2_tp8_aic_vs_fpm_summary.json`

MAPE:

| TP | context | context new>=1024 | decode | overall |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.35% | 0.53% | 2.86% | 1.45% |
| 2 | 4.13% | 4.42% | 2.37% | 3.36% |
| 8 | 10.43% | 4.40% | 8.28% | 9.49% |
| all | - | - | - | 4.77% |

Interpretation:

- TP1 is effectively closed for context and decode.
- TP2 is in good shape overall. The largest context point error is about `+11.7%` at `new_tokens=2048`.
- TP8 long-context behavior is good once small-token overhead rows are excluded: `4.40%` context MAPE for `new_tokens >= 1024`.
- TP8 decode still underpredicts small batches by about `10-12%` for b1-b8. Batch 64 is close at about `-1.9%`.

## What The FP8 Context Gap Meant

- TP1 initially exposed a pure compute/runtime gap because no TP communication exists. Real FPM/Nsight showed the dummy FP8 layerwise kernels were faster than the real FP8 checkpoint kernels across GEMM, attention, and quantization.
- A TP1-derived context calibration was applied to FP8 layerwise rows, which closed TP1 context and improved TP2/TP8 long-context accuracy.
- Do not use the old uncalibrated TP1 FP8 context numbers as current status; they were intermediate debugging evidence.

## Deployment Parity Lessons

- Default compile behavior matters. Earlier decode gaps were caused by compile-off layerwise attribution being compared to default-compile FPM.
- In default-compile decode, module ranges can collapse into `CUDAGraphWrapper`; use wrapper range for deployment-parity measurements.
- Avoid Python-level NVTX markers around attention unless measuring an A/B perf impact; they can disappear into compiled graphs or perturb graph capture.
- The first FP8 context layerwise run differed from TP1 FPM metadata:
  - FPM: `scheduler_config.max_num_batched_tokens=2048`, `scheduler_config.max_num_seqs=64`
  - Layerwise context: `scheduler_config.max_num_batched_tokens=8192`, `scheduler_config.max_num_seqs=1024`
- Later targeted checks showed max-num-seqs was not the full explanation; the real issue was real FP8 checkpoint kernels being slower than dummy layerwise kernels. Keep metadata comparisons anyway.

## Decode Communication Lessons

- TP1 decode is layerwise-only: no TP comm.
- TP>1 decode should not blindly add generic `tensor_model_parallel_all_reduce` latency.
- vLLM can use fused allreduce+RMS paths such as `flashinfer_trtllm_fused_allreduce_norm`; generic allreduce tables can overpredict.
- The collector should keep allreduce RMS data separate from generic custom-allreduce data. Query by dtype, TP size, message size, and nearest hidden size; current allreduce RMS collector uses hidden_size=4096 as an approximation and records that as a column.
- For AIC decode, subtract measured RMS latency from layerwise rows and add fused allreduce RMS for TP>1 when reliable data exists.
- vLLM layerwise GEN lookup now uses the KV length at the start of the decode iteration (`kv_len = isl + i`). For a `past_kv=1024` FPM decode sweep, compare with AIC `isl=1024, osl=2`.

## Context Axis Lessons

- Context should be collected on two axes: `past_kv` and `new_tokens`.
- Current useful grid target: `new_tokens=1..8192`, `past_kv=0..65536`, respecting model max length.
- vLLM chunked context must model later chunks with nonzero `past_kv`; do not treat a direct 16k row as equivalent to two 8192 chunks.

## AIC Comparison Implementation Notes

- FP8 rows are now in `src/aiconfigurator/systems/data/b300_sxm/vllm/0.20.1/layerwise_perf.csv`. Use a temporary systems root only for experimental calibration.
- Use `PerfDatabase('b300_sxm', 'vllm', '0.20.1', systems_root='/home/shadeform/aiconfigurator/src/aiconfigurator/systems')` for normal repo-root comparisons.
- Set `aiconfigurator.sdk.backends.vllm_backend._USE_LAYERWISE = True`.
- Build the model with `models.get_model('Qwen/Qwen3-32B-FP8', ModelConfig(tp_size=TP, pp_size=1), 'vllm')`.
- For direct decode comparison against collected `past_kv=1024`, call the generation phase with `isl=1024, osl=2`.
- Always state which TP sizes have FPM. Current FP8 context/decode has TP1, TP2, and TP8 FPM; mixed is still partial/secondary.

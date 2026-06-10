---
name: collect-vllm-layerwise
description: Collect vLLM layerwise latency data with Nsight Systems for AIC layerwise performance modeling. Use when the user asks to collect layerwise data, run the vLLM layerwise collector, gather TP=1/TP=2/TP=8 context or decode layer rows, convert layerwise CSVs into AIC data, or compare AIC/layerwise predictions against FPM ground truth.
---

# Collect vLLM Layerwise

## Core Workflow

Work from the `aiconfigurator` repo. Use:

- Collector: `collector/layerwise/vllm/collect_layerwise.py`
- Parser/common code: `collector/layerwise/common/`
- AIC data target: `src/aiconfigurator/systems/data/b300_sxm/vllm/0.20.1/layerwise_perf.csv`
- Artifact directory: set `AIC_LAYERWISE_ARTIFACTS`; default to `$PWD/.tmp/layerwise-artifacts` when running from the repo.
- HF token: export `HF_TOKEN` directly, or set `HF_TOKEN_FILE` and export `HF_TOKEN="$(tr -d '\n' < "$HF_TOKEN_FILE")"`.
- Model cache: set `HF_HOME`; default to `$HOME/.cache/huggingface` when unspecified.

The local Python env may not have vLLM installed. If so, run the collector inside `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0` and bind-mount host Nsight Systems from `/opt/nvidia/nsight-systems`.
Prefer `NSYS_VERSION=2025.3.2` or newer on B300. If host `/opt` has an older Nsight, check for a newer local install such as `/home/shadeform/tools/nsight-systems-2025.6.3/opt/nvidia/nsight-systems`. Older Nsight builds can export NVTX ranges without CUDA kernel tables, producing zero parsed layerwise rows.

Read [references/layerwise-commands.md](references/layerwise-commands.md) for the canonical Docker wrapper, TP=2 decode/context commands, conversion rules, and AIC comparison notes.

## Collection Rules

- The vLLM collector simulates TP on one physical GPU by patching model dimensions. It is not a real multi-rank TP deployment, and AIC adds TP communication analytically.
- For TP>1, keep `--rank-reduce max`. Context rows are per simulated rank; AIC adds generic TP allreduce. Decode rows should keep `rms_latency_ms` so AIC can subtract RMS and add fused allreduce+rms when data exists.
- Decode collection is fast and should use one transformer layer, `--latency-source span`, default compile/deployment parity, and batch sizes `1,2,4,8,16,32,64` at KV 1024.
- Context collection used for AIC should use the established 16-layer path unless the user explicitly prioritizes speed over accuracy. Use `--latency-source gpu_capped`, `--ctx-warmup-runs 2`, `--ctx-measured-runs 6`, and `--ctx-repeat-aggregation trimmed_mean`.
- Context should be a 2D grid: `new_tokens` up to 8192 and `past_kv` up to 65536, filtered by model max length. Include `new_tokens=8192,past_kv=8192` for vLLM chunked 16k behavior.
- For FP8 checkpoints, do not force KV FP8 unless that is the experiment. `--kv-quant fp8` forces `--kv-cache-dtype fp8`; default vLLM FP8 checkpoint behavior is measured with `--kv-quant bf16` so KV dtype remains `auto`.
- GPT-OSS layerwise specs automatically mirror vLLM's GPT-OSS runtime defaults for FP8 KV cache, CUDA graph capture size, and stream interval. Do not pass real `--tensor-parallel-size` or `--enable-expert-parallel` through layerwise; TP/EP are simulated/added analytically. Context collection always uses prefix caching to create arbitrary `past_kv` points; decode-only `--gen-driver prefill` may disable prefix caching when no context points are present.

## Convert To AIC Data

Map collector CSV rows into AIC `layerwise_perf.csv`:

- `ctx` -> `CTX`, `gen` -> `GEN`
- `attn_tp` -> `tp_size`
- `batch_size` unchanged
- `new_tokens` -> `seq_len_q`
- `past_kv` -> `seq_len_kv_cache`
- Preserve the actual model path in AIC data, e.g. keep `Qwen/Qwen3-32B-FP8` distinct from `Qwen/Qwen3-32B`.
- Divide `latency_ms` by `target_layer_count` for multi-layer context collection, exactly once. One-layer decode rows are already per layer.
- Divide `rms_latency_ms` by `target_layer_count` with the same rule as `latency_ms`.
- If a full-depth sanity point still leaves a repeatable FPM-vs-layerwise gap, identify a model-general missing component or collector mismatch. Do not add target-model FPM deltas as layerwise prediction inputs.

Run focused tests after code changes:

```bash
uv run --extra dev pytest tests/test_layerwise_vllm_collect.py tests/unit/sdk/database/test_layerwise.py tests/unit/sdk/database/test_data_loaders.py tests/unit/sdk/backends/test_vllm_layerwise_backend.py -q
```

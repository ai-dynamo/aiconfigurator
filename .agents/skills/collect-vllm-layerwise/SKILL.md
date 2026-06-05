---
name: collect-vllm-layerwise
description: Collect vLLM layerwise latency data with Nsight Systems for AIC layerwise performance modeling. Use when the user asks to collect layerwise data, run the vLLM layerwise collector, gather TP=1 or TP=2 context/decode layer rows, convert layerwise CSVs into AIC data, or compare AIC/layerwise predictions against FPM ground truth.
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
Prefer `NSYS_VERSION=2025.3.2` or newer on B300. Older Nsight builds can export NVTX ranges without CUDA kernel tables, producing zero parsed layerwise rows.

Read [references/layerwise-commands.md](references/layerwise-commands.md) for the canonical Docker wrapper, TP=2 decode/context commands, conversion rules, and AIC comparison notes.

## Collection Rules

- The vLLM collector simulates TP on one physical GPU by patching model dimensions. It is not a real multi-rank TP deployment.
- For TP>1, keep `--rank-reduce max` and make AIC add TP communication separately. The per-rank layerwise row itself does not include real allreduce.
- Decode collection is fast and should use one transformer layer, `--latency-source span`, and batch sizes `1,2,4,8,16,32,64` at KV 1024.
- Context collection used for AIC should use the established 16-layer path unless the user explicitly prioritizes speed over accuracy. Use `--latency-source gpu_capped`, `--ctx-warmup-runs 2`, `--ctx-measured-runs 6`, and `--ctx-repeat-aggregation trimmed_mean`.
- For 16k context, collect chunked prefill explicitly as `new_tokens=8192,past_kv=8192`; do not assume a direct 16k row represents real vLLM scheduler behavior.

## Convert To AIC Data

Map collector CSV rows into AIC `layerwise_perf.csv`:

- `ctx` -> `CTX`, `gen` -> `GEN`
- `attn_tp` -> `tp_size`
- `batch_size` unchanged
- `new_tokens` -> `seq_len_q`
- `past_kv` -> `seq_len_kv_cache`
- Normalize model path to `Qwen/Qwen3-32B` when using the patched config cache path.
- Divide `latency_ms` by `target_layer_count` for multi-layer context collection. One-layer decode rows are already per layer.

Run focused tests after code changes:

```bash
uv run --extra dev pytest tests/test_layerwise_vllm_collect.py tests/unit/sdk/database/test_layerwise.py tests/unit/sdk/backends/test_vllm_layerwise_backend.py -q
```

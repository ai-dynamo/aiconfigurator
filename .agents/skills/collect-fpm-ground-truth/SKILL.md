---
name: collect-fpm-ground-truth
description: Collect Dynamo/vLLM ForwardPassMetrics ground-truth latency data for AIC/layerwise validation. Use when the user asks to run or update FPM ground truth, collect context/decode/mixed vLLM metrics, compare layerwise/AIC against real vLLM behavior, debug Dynamo FPM collection, or preserve mixed-step FPM latency rows.
---

# Collect FPM Ground Truth

## Core Workflow

Work from the `aiconfigurator` repo unless the user points elsewhere. Use:

- Collector: `python -m collector.layerwise.fpm.collect --model <model>`
- Internal shell wrapper: `collector/layerwise/fpm_ground_truth/collect_fpm_metrics.sh`
- Dynamo image: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0`
- Expected vLLM: `0.20.1`; do not allow version mismatch unless explicitly accepted.
- HF token: export `HF_TOKEN` directly, or set `HF_TOKEN_FILE` and export `HF_TOKEN="$(tr -d '\n' < "$HF_TOKEN_FILE")"`.
- Model cache: set `HF_HOME`; default to `$HOME/.cache/huggingface` when unspecified.
- vLLM compile/cache directory: set `VLLM_CACHE_HOST`; default to `$HOME/.cache/aic-vllm`. The wrapper mounts it to both `/home/dynamo/.cache/vllm` and `/root/.cache/vllm` so DeepGEMM, FlashInfer, TileLang, and torch compile artifacts survive worker restarts in Dynamo and root-run vLLM containers. For DeepSeek/TileLang paths, export `TILELANG_CACHE_DIR=/home/dynamo/.cache/vllm/tilelang` and `TILELANG_TMP_DIR=/home/dynamo/.cache/vllm/tilelang/tmp`.
- Artifact directory: set `AIC_LAYERWISE_ARTIFACTS`; default to `$PWD/.tmp/layerwise-artifacts` when running from the repo.

Prefer one deployment per sweep. For layerwise gap closure, collect `context,decode` first and add `mixed` only after those phases are understood. Always preserve `*_phase.csv`, `*_detail.csv`, and `*_workload.csv`; partial runs can still be valid for phases that finished.

Read [references/fpm-commands.md](references/fpm-commands.md) for the smoke command, canonical TP=2 command, output interpretation, and failure handling.

The normal CLI should be compact: provide `--model`, optional `--run-dir`, and `--tp-sizes`/`--ep-sizes` when needed. The wrapper infers Docker GPUs from TP/EP unless `--gpus` is explicitly supplied. Standard output files are created under `--run-dir`; use explicit output-path flags only for unusual plumbing.

## Important Defaults

- Use random prompt token IDs through the completions API. Do not use constant token IDs.
- Keep `ignore_eos=true` unless the user asks for OSL-as-cap behavior.
- Use local file-discovery heartbeat. `--file-discovery-touch-seconds 2` fixed prior discovery expiry/503 failures.
- Omit `--gpus` for the common case. If overriding devices, quote Docker device selectors as `--gpus '"device=0,1"'`; unquoted `--gpus device=0,1` is rejected by Docker.
- GPT-OSS FPM synthetic sweeps use vLLM's recommended benchmark defaults automatically: FP8 KV cache when unset, prefix caching disabled unless explicitly overridden, `max-cudagraph-capture-size=2048`, and `stream-interval=20`. Treat prefix-cache disablement as a measurement consistency default, not a generic GPT-OSS correctness rule.
- For context repeats, use medians or trimmed means and inspect first-pass cold outliers before tuning against them.
- For decode, compare pure decode-only FPM rows by `decode_requests` and `mean_decode_kv_tokens`. Prefer the longest consecutive full-batch block where `decode_tokens == decode_requests`; exclude tail rows after requests finish.
- For 16k context with vLLM chunked prefill, FPM appears as `8192 @ ctx_kv=0` plus `8192 @ ctx_kv=8192`, not a single `ctx_tokens=16384` row.

## Smoke Validation

Use the reference smoke command after FPM wrapper changes. A passing Qwen3-32B TP1 smoke writes `fpm_metrics.csv`, `fpm_metrics_detail.csv`, `fpm_metrics_phase.csv`, request/warmup workload CSVs, and vLLM metadata/config JSON files. Expect both context and decode rows in `fpm_metrics_phase.csv`; cold first context rows can be much slower than subsequent rows.

## After Collection

Summarize `*_phase.csv` before comparing:

- `context`: group by `(ctx_tokens, ctx_kv_tokens)` for chunked-prefill correctness.
- `decode`: group by `decode_requests`, usually with mean KV near 1024 for the b1..64 decode sweep.
- `mixed`: keep rows as observed scheduler iterations; shapes are `(ctx_tokens, decode_requests, mean_decode_kv_tokens)`.

If collection fails after some traffic, still inspect copied partial outputs. The script should preserve collector CSVs before exiting.

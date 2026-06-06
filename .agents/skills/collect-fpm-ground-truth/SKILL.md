---
name: collect-fpm-ground-truth
description: Collect Dynamo/vLLM ForwardPassMetrics ground-truth latency data for AIC/layerwise validation. Use when the user asks to run or update FPM ground truth, collect context/decode/mixed vLLM metrics, compare layerwise/AIC against real vLLM behavior, debug Dynamo FPM collection, or preserve mixed-step FPM latency rows.
---

# Collect FPM Ground Truth

## Core Workflow

Work from the `aiconfigurator` repo unless the user points elsewhere. Use:

- Collector: `collector/layerwise/fpm_ground_truth/collect_fpm_metrics.sh`
- Dynamo image: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0`
- Expected vLLM: `0.20.1`; do not allow version mismatch unless explicitly accepted.
- HF token: export `HF_TOKEN` directly, or set `HF_TOKEN_FILE` and export `HF_TOKEN="$(tr -d '\n' < "$HF_TOKEN_FILE")"`.
- Model cache: set `HF_HOME`; default to `$HOME/.cache/huggingface` when unspecified.
- Artifact directory: set `AIC_LAYERWISE_ARTIFACTS`; default to `$PWD/.tmp/layerwise-artifacts` when running from the repo.

Prefer one deployment per sweep. For layerwise gap closure, collect `context,decode` first and add `mixed` only after those phases are understood. Always preserve `*_phase.csv`, `*_detail.csv`, and `*_workload.csv`; partial runs can still be valid for phases that finished.

Read [references/fpm-commands.md](references/fpm-commands.md) for the canonical TP=2 command, output interpretation, and failure handling.

## Important Defaults

- Use random prompt token IDs through the completions API. Do not use constant token IDs.
- Keep `ignore_eos=true` unless the user asks for OSL-as-cap behavior.
- Use local file-discovery heartbeat. `--file-discovery-touch-seconds 2` fixed prior discovery expiry/503 failures.
- Quote Docker device selectors as `--gpus '"device=0,1"'`; unquoted `--gpus device=0,1` is rejected by Docker.
- For context repeats, use medians or trimmed means and inspect first-pass cold outliers before tuning against them.
- For decode, compare pure decode-only FPM rows by `decode_requests` and `mean_decode_kv_tokens`. Prefer the longest consecutive full-batch block where `decode_tokens == decode_requests`; exclude tail rows after requests finish.
- For 16k context with vLLM chunked prefill, FPM appears as `8192 @ ctx_kv=0` plus `8192 @ ctx_kv=8192`, not a single `ctx_tokens=16384` row.

## After Collection

Summarize `*_phase.csv` before comparing:

- `context`: group by `(ctx_tokens, ctx_kv_tokens)` for chunked-prefill correctness.
- `decode`: group by `decode_requests`, usually with mean KV near 1024 for the b1..64 decode sweep.
- `mixed`: keep rows as observed scheduler iterations; shapes are `(ctx_tokens, decode_requests, mean_decode_kv_tokens)`.

If collection fails after some traffic, still inspect copied partial outputs. The script should preserve collector CSVs before exiting.

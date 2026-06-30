---
name: close-layerwise-fpm-gap
description: Diagnose and close AIC/vLLM layerwise prediction gaps against Dynamo/vLLM ForwardPassMetrics ground truth. Use when the user asks why layerwise differs from FPM, asks to compare AIC layerwise vs FPM for context/decode/mixed phases, asks whether to recollect FPM/layerwise data, asks about compile/CUDA graph/deployment parity mismatches, or asks to debug TP1/TP2/TP8/Qwen3 latency accuracy.
---

# Close Layerwise-FPM Gap

## Core Rule

Treat FPM as the ground truth for complete vLLM forward-pass iterations. Treat layerwise as a decomposed approximation that is only valid after proving the collector and FPM deployment configs match.

Do not recollect FPM until existing artifacts have been checked. FPM is slow and partial runs can still contain useful `*_phase.csv`, `*_detail.csv`, and `*_workload.csv` rows.

Use raw AIC-vs-FPM error as the accuracy metric. Post-hoc scaled/multiplier columns are diagnostics only; do not report them as predictive accuracy, hide calibration inside a global multiplier, or add target-model FPM deltas as prediction inputs.

Read [references/today-learnings.md](references/today-learnings.md) when working on Qwen3-32B/B300/vLLM 0.20.1, FP8, TP1/TP2/TP8, fused allreduce RMS, or context/decode mismatch analysis.

## Workflow

1. **Inventory artifacts before running anything.**
   - Check active Docker/GPU state with `docker ps` and `nvidia-smi`.
   - Find FPM files: `*_fpm_*_phase.csv`, `*_detail.csv`, `*_workload.csv`, `*_effective_vllm_config.json`, `*_metadata.json`.
   - Find layerwise files: `*_vllm_context_*.csv`, `*_vllm_decode_*.csv`, profile `status.jsonl`, metadata JSON, and Nsight SQLite reports.
   - Report which TP sizes have both FPM and layerwise data. Do not imply a TP comparison exists when only layerwise was collected.

2. **Validate deployment parity before interpreting error.**
   Compare FPM and layerwise effective config metadata:
   - `vllm_version`
   - `parallel_config.tensor_parallel_size`
   - `model_config.dtype`
   - `cache_config.cache_dtype`
   - `scheduler_config.max_num_batched_tokens`
   - `scheduler_config.max_num_seqs`
   - `compilation_config.mode`
   - `compilation_config.cudagraph_mode`
   - `compilation_config.custom_ops`
   - key `compilation_config.pass_config` fusions

   Any mismatch can dominate the gap. Common culprits are compile mode, CUDA graph mode, max sequence/batch-token settings, default-vs-forced KV dtype, and FP8 fusion settings.

3. **Normalize layerwise rows exactly once.**
   - Map `ctx` to `CTX`, `gen` to `GEN`.
   - Map `attn_tp` to `tp_size`.
   - Map `new_tokens` to `seq_len_q`.
   - Map `past_kv` to `seq_len_kv_cache`.
   - Divide multi-layer context rows by the collected `target_layer_count` before loading into AIC, exactly once.
   - Do not divide one-layer decode rows.
   - Keep `rms_latency_ms` with the same division as `latency_ms`.

4. **Compare through AIC, not a hand formula, unless debugging a single component.**
   - Set `AIC_VLLM_USE_LAYERWISE=1` before importing/running AIC comparison code.
   - If the repo AIC CSV does not yet contain the target rows, create a temporary systems root under the artifact directory and load `PerfDatabase(..., systems_root=temp_root)`.
   - Use AIC’s vLLM layerwise backend so context chunking, per-layer scaling, and fused RMS logic follow the code under test.
   - For TP1, comm should be zero; any gap is compute/runtime/collector/deployment parity.
   - For TP>1 decode, do not add generic allreduce blindly. vLLM default-compile decode may use fused or overlapped paths; current AIC pure-decode modeling intentionally avoids generic allreduce unless calibrated data supports it.

5. **Summarize FPM by scheduled shape.**
   - Context: group by `(ctx_tokens, ctx_kv_tokens, ctx_requests)` and use medians or trimmed means. Exclude mixed/decode rows when comparing pure context.
   - vLLM chunked 16k context appears as `8192,past=0` plus `8192,past=8192`.
   - Decode: group by `decode_requests`; filter `mean_decode_kv_tokens` to the intended KV window, usually around 1024 for the b1..64 sweep. Prefer the longest consecutive full-batch block where `decode_tokens == decode_requests`, and exclude tail rows after requests finish.
   - Mixed: compare only after context/decode are understood; shape is scheduler-dependent.

6. **Break down the error before proposing collection.**
   - Context TP1: compare FPM vs `layerwise_context * num_layers`; no comm exists.
   - Decode TP1: compare FPM vs `layerwise_decode * num_layers`; no comm exists.
   - TP>1 context: show compute and generic TP allreduce separately.
   - TP>1 decode: show layerwise compute after subtracting measured RMS, fused allreduce RMS estimate, and total. Do not use the old `isl=1023` workaround; AIC GEN lookup should query the KV length at the start of the decode step.
   - For Nsight-backed checks, compare wrapper span vs kernel/GPU time before blaming `gpu_capped`.

7. **Choose the next action conservatively.**
   - If metadata differs, first run a small targeted layerwise sanity pass matching FPM defaults.
   - If multi-layer extrapolation is suspicious, run a small full-depth or larger-slice point before broad recollection.
   - If FPM failed after preserving output, use complete phases that finished; rerun only the missing phase if needed.
   - If FPM and layerwise configs match and the gap persists, update AIC modeling or collector normalization before collecting a large grid. Do not "fix" a target model by feeding its own FPM residual back into the prediction.

## Reporting

Give the user a compact table with:

- phase and shape
- FPM latency
- AIC/layerwise predicted latency
- signed error and MAPE
- compute vs comm breakdown when TP>1
- exact caveats, especially missing TP FPM rows or failed mixed runs

State whether the evidence points to FPM noise, deployment parity mismatch, layerwise collector error, AIC normalization error, or a real modeling gap.

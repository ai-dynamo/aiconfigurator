# Task: Fix corrupt MoE decode layerwise timing (collector)

## Goal
The vLLM **layerwise** decode (generation) timings for **MoE models**
(Qwen3.6-35B-A3B, DeepSeek-V4-Flash) are non-physical, which makes AIC decode and
mixed predictions for those models wrong. The numbers were partially fixed by a
recent collector change but are still corrupt. Your job is to make the collected
decode **backbone** latency a clean, monotonic function of batch and KV length, by
using **one consistent GPU timing method across all batch sizes and shapes**.

This is a **data-collection** problem, not an AIC modeling problem. Do not "fix" it
by adding correction factors in the modeling layer.

---

## The single most important instruction

**Do NOT switch the timing method / `latency_source` based on batch size or shape.**

The current data mixes two timing methods within one sweep:
- small batches (1, 2, 4) -> `latency_source = schedule_to_update`
- batches >= 8           -> `latency_source = live_step_wall`

The two methods do not agree, so the sweep has a **step-change discontinuity**
exactly at the boundary (batch 4 -> 8). Switching method by batch/shape is the
direct cause of the non-monotonic "jump then flat" artifact. Pick **one** timing
method and use it for **every** batch size, KV length, model, and phase. A latency
curve must be produced by a single, uniform measurement procedure end to end.

---

## What "corrupt" looks like (evidence)

Collected decode `latency_ms` must be **non-decreasing in batch** and should rise
with KV length (decode attention reads `batch x kv` of KV cache). It does neither.

Qwen3.6-35B-A3B, gen, tp1, ep1, past_kv=4096, by batch (newest data):
```
batch:  1     2     4     8     16    32    64    128   256
ms:    2.68  2.73  3.99  7.58  7.26  7.92  8.10  12.30 14.17
src:   sched sched sched  live  live  live  live  live  live   <- method switches at b8
```
Two defects:
1. **Jump at b8** (3.99 -> 7.58) — coincides exactly with the `schedule_to_update`
   -> `live_step_wall` switch.
2. **Flat plateau b8..b64** (~7.3-8.1 ms) that does not rise with batch, and is
   also flat across KV (at b16: past_kv 1/4096/16384/32768 -> 7.69/7.26/8.32/8.42).

`live_step_wall` here is ~7-8 ms **regardless of batch or KV**, and it even exceeds
the real full-step latency (see below). That is the signature of a **fixed per-step
host/wall overhead** (scheduling, sampling, CPU<->GPU sync, Python/launch latency)
being captured instead of the GPU step latency.

Ground truth (real serving, `fpm_metrics_phase.csv`) is smooth and correct, e.g.
Qwen3.6 tp1 decode: 3.64 (b1) -> 5.31 (b8) -> 7.14 (b16) -> 8.75 (b32). The
collected backbone (no-op MoE) at b8 is 7.58 ms — **larger than the full real step
(5.31 ms)**, which is impossible if it is measuring only GPU compute.

DeepSeek-V4-Flash is worse (b1 -> b2 jumps ~7 -> ~57 ms). Dense Qwen3-32B decode is
clean (it does not hit the same mixed-source/overhead pattern), so a correct,
uniform method should make MoE look like the dense case.

---

## Root cause (two compounding issues)
1. **Mixed timing methods in one sweep** (the `schedule_to_update` vs
   `live_step_wall` switch at batch 8) -> the discontinuity / jump.
2. **`live_step_wall` measures full live-iteration wall time**, including host
   overhead, not the isolated GPU step -> a large fixed floor (~7-8 ms) that
   swamps the real decode compute and erases batch/KV scaling (and exceeds the
   real full step).

The earlier `schedule_to_update` method had spiky scheduler-envelope noise; the
`live_step_wall` "fix" replaced it with a fixed host-overhead floor. Neither is the
isolated GPU step time.

## The fix
Measure the **GPU decode step in isolation**, the **same way for all batch sizes /
KV / models / phases**:
- Time only the model forward on the GPU (e.g. CUDA events around the forward
  pass), excluding scheduler, sampling/detokenization, and host-side overhead.
- Use enough warmup + averaging to remove run-to-run variance (the old
  `schedule_to_update` spikes).
- Do **not** branch the timing method on batch/shape. One `latency_source` per
  run.

Success criteria:
- Collected decode `latency_ms` is **monotonic non-decreasing in batch** and rises
  with KV length, for qwen36 and dsv4, at all (tp, ep).
- A single `latency_source` value across the whole gen sweep.
- The decode chart's AIC line tracks FPM comparably to dense Qwen3-32B
  (decode MAPE ~5%). At batch=1 the current data is already correct, so the fixed
  method should reproduce that and extend it cleanly to larger batches.

Where to implement (files touched by the prior timing change):
- `collector/layerwise/vllm/scheduler.py`
- `collector/layerwise/vllm/worker.py`
- `collector/layerwise/vllm/collect.py`
- latency-source handling: `src/aiconfigurator/sdk/operations/layerwise.py`
  (`SCHEDULER_ENVELOPE_LATENCY_SOURCES`).

---

## What is NOT broken — do not change these
- **The MoE op overlay is correct.** AIC decode = collected backbone
  (`generation_layerwise`, collected with `includes_moe=False`) + MoE-expert
  overlay (`generation_moe` etc.) queried from `moe_perf` at `num_tokens=batch`,
  scaled per layer. The `moe_perf` values are **real measured kernel timings** and
  the query parameters (tokens, hidden, inter, topk, experts, distribution) are
  correct. At batch=1, where the backbone is clean, `backbone + moe ~= FPM`. The
  apparent MoE over-prediction at large batch is entirely the corrupt backbone
  (the backbone is inflated up to ~the full step, so adding the legitimate MoE
  overshoots). Fix the backbone, not the overlay.
- **The decode compute batch calibration is correct and dense-only.**
  `_DECODE_COMPUTE_BATCH_CAL = 0.0066` in `vllm_backend.py` (gated behind
  `is_moe_model`) is validated on dense Qwen3-32B against the new data (decode
  MAPE 10.6% -> 3.4%). It is intentionally not applied to MoE.

## Other modeling fixes already in place (context; do not redo)
- Fused all-reduce for decode comm (`_LAYERWISE_USE_FUSED_ALLREDUCE_RMS`, decode
  only — fused is only cheaper at small/decode message sizes).
- Generation comm un-suppression for single-GPU-collected data
  (`_LAYERWISE_GEN_SINGLE_GPU_COMM`, gated on `physical_gpus >= tp_size`).
- High-KV dense decode repair (`_repair_decode_high_kv`) — extrapolates Qwen3-32B
  past_kv>=8192 from 2048/4096 (those rows were also envelope-noise corrupt).
- Mixed model = context_total + decode attention (`_get_mix_step_latency`).

---

## Where things live
- **Layerwise data:**
  - latest: `runs/layerwise_full_vllm0201_20260616_174245_checkpoint_20260616_182815/layerwise.csv`
  - prior:  `runs/layerwise_full_vllm0201_20260615_045248/layerwise.csv`
  - One row per (model, attn_tp, moe_tp, ep, phase, batch_size, new_tokens,
    past_kv, layer_type). Key cols: `latency_ms`, `latency_source`,
    `measured_layer_count`, `layer_multiplier`, `physical_gpus`, `includes_moe`.
- **Ground truth (FPM):** `fpm_golden_runs/fpm_*/.../fpm_metrics_phase.csv`
  (phase = context/decode/mixed). Consistent (within-config CV ~0.3%).
- **AIC decode model:** `src/aiconfigurator/sdk/backends/vllm_backend.py`,
  `_get_decode_step_latency()`.
- **Diagnostic / charting tool:** `tools/plot_fpm_vs_aic.py` — FPM-vs-AIC log-log
  charts (ctx/gen/mixed) per parallelism x past_kv, plus an all-reduce comparison.

## How to verify a fix
1. Re-collect MoE decode with the uniform GPU-isolated timing.
2. Sanity-check raw monotonicity directly:
   group `layerwise.csv` by (model, attn_tp, ep, past_kv); `latency_ms` must be
   non-decreasing in `batch_size`, and there must be a single `latency_source`.
3. Regenerate charts:
   ```
   .venv/bin/python tools/plot_fpm_vs_aic.py \
     --layerwise <NEW_layerwise.csv> \
     --model "Qwen/Qwen3.6-35B-A3B" --moe-perf-file moe_perf.txt \
     --out-dir /tmp/check_qwen36
   ```
   The gen chart's AIC line should track the FPM points; the blue "layerwise
   collected" backbone dots should rise smoothly with batch (no jump at 8, no flat
   plateau). Repeat for `deepseek-ai/DeepSeek-V4-Flash`.

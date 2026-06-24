# Clean ctx-envelope recollection results (qwen36 tp4_ep4)

Follow-up to `MIX_STEP_FIX_VALIDATION.md`, which recommended recollecting the
context phase with a **compute-only latency source** (`execute_model_gpu`, the
same CUDA-event source the gen rows already use) to test whether the ~2.2x
mixed-step over-prediction ("README#1 MoE mixed really bad") is a ctx
**collection artifact** rather than a composition bug.

In-image on the SLURM B300 node, Dynamo vLLM 0.20.1, `uv run --active`,
`HF_HOME=/workspace/models/hf_home`, nsys `2026.2.1`. Model
`Qwen/Qwen3.6-35B-A3B`, `tp=4 ep=4` -> `attn_tp=4 moe_tp=1 moe_ep=4`,
`--moe-noop` (MoE added back via `collector/layerwise/wip/moe_perf.txt` overlay).

## Collector change required (the ctx timing was never wired to execute_model_gpu)

The June accuracy campaign established `execute_model_gpu` (GPU-isolated CUDA
events around `execute_model`) as the correct source, and gen already used it,
but **ctx silently never did**: the marker recorded `execute_model_gpu_time_ms`
only when `measure_execute_model_gpu_time` was set in the marker-control JSON,
and the prefix-cached ctx path
(`worker.py:_run_prefix_cached_ctx_iteration`) re-wrote marker control for the
measured forward **without** that flag — clobbering it. Result: with
`--latency-source execute_model_gpu` the scheduler's ctx lookup demanded
`execute_model_gpu_time_ms` events that were never emitted -> **0 ctx rows**
(observed in the earlier `cleanctx` attempt).

Fix (this branch):
- `worker.py`: thread `measure_execute_model_gpu_time` through
  `_run_prefix_cached_ctx_iteration` into its internal `_set_marker_state`, so
  the measured ctx forward keeps the flag.
- `scheduler.py`: `_lookup_scheduler_timing_aggs` already had the
  `prefer_execute_model_gpu_for_ctx` branch (reads `execute_model_gpu_time_ms`
  for ctx, mirroring gen); added per-run summation of chunked-prefill
  execute_model calls (defensive — see "Chunking" below).

Verified: clean ctx events now carry `execute_model_gpu_time_ms`
(e.g. new=2048/past=128 -> 46.1 ms GPU, ~= its 46.1 ms wall: execute_model is
GPU-bound, no host-gap leakage).

### Chunking caveat (matters for interpreting large new_tokens)
The marker only instruments scheduler iteration n=1 (active_iterations={1}), so
for a single-request prefill exactly **one** `execute_model` call is measured:
the first chunk of <= `max_num_batched_tokens` (2048) new tokens. Therefore:
- **new <= 2048 -> single chunk -> full per-step forward (correct).** This is
  the regime every mixed step and the ctx sanity actually use.
- **new = 4096/8192 -> only the first 2048-token chunk is captured** (labelled
  as 8192). This is fine here: vLLM caps a real step at 2048 new tokens, so the
  golden deployment's per-step prefill is also a single <=2048-token chunk
  (golden `max_num_batched_tokens=2048`). The per-step (first-chunk) value is
  exactly what mixed-step composition needs; the full-prefill sum is not.

## Headline numbers (golden `tp4_ep4_past4096`, tagged max_num_seqs=128 / mnbt=2048)

Apples-to-apples = identical gen rows + identical ctx grid shape, **only the ctx
latency source differs.**

### Matched subset (15-gen grid = the `MIX_STEP_FIX_VALIDATION.md` point set, n=92/300)
| ctx source | median(AIC/FPM) | MAPE |
|---|---|---|
| OLD `schedule_to_update`/`worker_wall` | 2.193 | 167.3% |
| CLEAN `execute_model_gpu` | **1.465** | 78.7% |

### Full evaluable set (56-gen grid -> 299/300 golden mixed steps evaluable)
| ctx source | median(AIC/FPM) | MAPE |
|---|---|---|
| OLD `schedule_to_update`/`worker_wall` | 3.151 | 290.3% |
| CLEAN `execute_model_gpu` | **2.499** | 141.1% |

(The n=92 subset is gated by the sparse 15-row gen grid and is biased toward
well-matched large-ctx/decode_kv~4096 steps; the 56-row gen grid is the more
complete picture. Both show the same direction and similar relative drop.)

## Answers to the three questions

**1. Does median(AIC/FPM) drop from ~2.2 toward ~1 with clean ctx?**
It DROPS substantially but NOT to ~1: 2.193 -> 1.465 (matched subset) or
3.151 -> 2.499 (full set). So the ctx **collection method was a major
contributor** to the over-prediction — confirming part of the hypothesis — but
a large residual (~1.5-2.5x) remains. README#1 is *partly* a collection
artifact, *not entirely*.

The improvement comes almost entirely from **large new_tokens (>2048) steps**:
old `worker_wall` timed the *full multi-chunk* `generate()` (4096 -> 107 ms,
8192 -> 206 ms), whereas a real mixed step processes only one <=2048-token chunk.
Golden mixed per-step latency is flat ~44 ms for ctx_tokens 1k..8k (chunked at
2048); clean `execute_model_gpu` first-chunk = ~46 ms, which **matches** golden.

**2. Is the per-prefill-bucket over-prediction monotonically shrinking as
ctx_tokens grows (fixed-floor signature)?**
NOT on the full set. Clean per-bucket MAPE is **hump-shaped**:
`<=2k:118%  2-4k:172%  4-8k:230%  8-16k:113%  >16k:58%` — it peaks at 4-8k and is
*lowest* at >16k. A fixed host floor would amortize monotonically; the mid-prefill
peak means **the mechanism is not just a fixed floor.** (On the narrow 92-pt
subset it does look monotone — `185->108->53->52%` — but that is a coverage
artifact of the sparse gen grid.)

**3. ctx-only sanity: does the clean ctx envelope now match golden context FPM
(~17 ms at 256 tok)?**
NO — the "~17 ms at 256 tok" expectation is **refuted**. Clean AIC ctx (MoE-incl):

| new tok | CLEAN AIC ctx | golden ctx FPM | ratio |
|---|---|---|---|
| 128 | 38.5 ms | 15.9 ms | 2.42x |
| 256 | 38.9 ms | ~16.9 ms | ~2.3x |
| 512 | 41.2 ms | 16.2 ms | 2.54x |
| 1024 | 55.7 ms | 53.5 ms | 1.04x |
| 2048 | 58.8 ms | 50.6 ms | 1.16x |

The clean envelope **matches golden at >= 1024 new tokens but is ~2.4x too high
at <= 512.** Crucially, for new <= 2048 (single chunk) clean `execute_model_gpu`
~= the old `schedule_to_update` (256 tok: 33 ms raw clean vs 35.6 ms old — only
~2 ms host overhead removed). The dramatic old/clean difference was confined to
new > 2048 (`worker_wall` full-prefill rows). So the small-prefill over-prediction
is a **genuine envelope discrepancy, not a timing artifact**: golden shows a sharp
CUDA-graph step (~16 ms for <=512 tok, jumping to ~40 ms at ~528 tok — a fast
small-batch capture regime), while the AIC layerwise envelope sits at a ~38 ms
floor even at 128 tokens and never models that fast regime.

## Verdict
- The ctx timing source **was** a dominant chunk of the mixed over-prediction
  (worker_wall full-prefill on new>2048). Switching ctx to `execute_model_gpu`
  cuts median AIC/FPM by ~25-33% and roughly halves MAPE.
- It does **not** close the gap: a ~1.5-2.5x residual remains, dominated by a
  real ~2.4x over-prediction of the **small-prefill (<=512 tok) ctx envelope**
  (golden's fast small-batch CUDA-graph regime, unmodelled by AIC), peaking in
  the mid-prefill (4-8k total ctx) buckets — i.e. not a fixed host floor.
- Per the task caveat, the decode-MoE term is a rounding error on these
  decode-starved (decode 1-15) golden rows and was not chased here.

## Artifacts
- Clean ctx+gen: `runs/layerwise_qwen36_tp4ep4_cleanctx4/layerwise.csv`
  (99 ctx `execute_model_gpu` + 56 gen `execute_model_gpu`),
  tagged `.../layerwise_native_tagged128.csv`.
- Old-source ctx on identical 99-grid:
  `runs/layerwise_qwen36_tp4ep4_oldctx99/layerwise.csv`
  (+ `oldctx_56gen_tagged128.csv` = old ctx + clean 56 gen for the full-set baseline).
- Charts: `runs/cleanctx4_native_charts/`, `runs/oldctx99_charts/`.
- Helpers: `runs/tag128.py`, `runs/ctx_sanity.py`.

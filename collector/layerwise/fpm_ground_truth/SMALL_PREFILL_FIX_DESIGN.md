# Small-prefill backbone over-prediction — diagnosis + fix shape (DEFERRED)

**Status: fully diagnosed; fix shape known; implementation DEFERRED.** This is direction #4
in MOE_INVESTIGATION_SUMMARY (the backbone layer, MoE-unrelated). The root cause is settled
and the fix shape is clear, but the fix needs collector instrumentation for a payoff that is
bounded and does not move config recommendations — so it is parked, not abandoned. Anyone
picking this up should NOT re-run the investigation below; it converged.

## Symptom
qwen36 (Qwen3.6-35B-A3B), b300, vLLM 0.20.1. AIC's context (prefill) layerwise envelope
over-predicts golden by ~2.4–3× at small new-token counts (<=512): AIC ~33–40 ms vs golden
~16 ms. Matches golden at >=1024 tok. On the BATCHED target (golden mixed rows) this is the
single largest residual: mixed MAPE 38.8%, with the <=512-new-tok bucket at 171% MAPE
(10% of steps but 43% of error mass; 6/7 are the terminal chunk of a chunked 4096 prefill,
~400 new + 3696 prefix).

## Root cause (settled): 1-GPU-collection launch-gap artifact — NOT comm / topology / capture
golden's real per-step is busy-bound (the GPU is kept full, so the ~470 eager kernel launches
hide behind the work). AIC's collector models tp4 as a **1-GPU sharded** config: no TP
all-reduce, tiny moe-noop kernels (~3.4 ms busy), so the GPU idles ~60 µs between ~510 eager
launches → ~30 ms of launch-gap → a launch-bound ~33 ms floor that is *higher* than golden's
real busy-bound step.

### The wrong turns (do not re-run — each was empirically refuted)
| we suspected | refuted by |
| --- | --- |
| topology (1-GPU vs real tp4) | real-tp4 also ~40 ms; f(tp) rises 40.6→43 ms tp1→8 (wrong sign) |
| golden captures more graph | launch counts identical: golden 470 vs collector 510 eager, 41 graph pieces each |
| comm-bound (all-reduce ~75% of busy) | that "75%" is rank-desync SPIN-WAIT misread as communication; the synced all-reduce floor is only ~1.85 ms (4%) |
| cost is message-scaled (so decode escapes via small messages) | same `cross_device_reduce_2stage` kernel = 33 µs/call in-graph vs 1036 µs/call eager; message term is sub-µs. It is capture/sync-bound, not message-scaled |

### Why decode escapes (and why it's NOT always-add comm)
decode runs FULL-graph captured → its all-reduces replay in-graph, ranks stay synced, per-call
~33 µs (cheap). Prefill >512 runs eager → ranks desync → each all-reduce spins ~1036 µs
(exposed). So comm is not "always added, message-scaled"; the cost is the in-graph(synced) vs
eager(desynced) execution mode. decode's tiny synced comm is why the earlier "decode comm
overlap≈0" held.

### The stable physical floor
Across every re-slice, only two quantities held still: **compute-busy ≈ 7 ms (incl MoE)** and
**synced-comm floor ≈ 1.85 ms**. Everything else (spin 25 ms, launch-gap 8.5–30 ms) moved with
execution mode (isolation / 1-GPU-vs-tp / capture). So the real per-step floor ≈ **busy ≈ 9 ms**;
the 33 ms over-prediction is the 1-GPU isolated launch-gap/spin artifact. (Caveat: golden's own
isolated step still shows ~8.5 ms launch-gap, but production batching fills it — isolated 40 ms
vs batched-stream 14 ms, a ~3× swing larger than any sliced term. A single "correct 16 ms"
independent of occupancy may not exist.)

## Fix shape (known): backbone WALL → busy, no threshold
```
ctx step ≈ busy_compute + synced-comm floor   (drop the launch-gap term entirely)
```
No `<=512` conditional: busy scales with compute (tokens), so small-new-token → low busy,
large-prefill → high busy; the crossover is physical, not an empirical token threshold. This is
the `max(compute, memory, comm)` roofline with the launch-gap (a 1-GPU/isolation artifact)
removed. For the BATCHED target this is correct because batching fills the gap → the step is
busy-bound.

## Why DEFERRED (the ROI call)
- **payoff is bounded**: shape-narrow (~1 shape class: terminal chunks of chunked prefills),
  ~17pp of mixed MAPE (38.8% → ~22%), and it does NOT move config recommendations (that's the
  decode/throughput regime — already fixed, MAPE 51→21%, committed).
- **the fix needs real engineering, not a flag**: the launch-gap-free "busy" is NOT obtainable
  by re-collecting existing metrics — every collector `latency_source` (wall / span / gpu /
  execute_model_gpu) includes the launch gap. Getting busy needs NEW per-kernel-duration
  (CUPTI-level) instrumentation in the collector. op-wise SOL roofline is not a clean
  substitute: op-wise SILICON has no `context_attention_perf.parquet` for this system, and the
  SOL roofline does not model qwen36's GDN/Mamba attention (it queries standard ContextAttention)
  — which is exactly why layerwise exists.
- **curve-fit risk**: ~1 shape class is too narrow to validate a model change; needs a broader
  hybrid workload first.

## When/how to revisit (future implementer)
1. Add a launch-gap-free busy metric to the collector (sum of per-kernel durations, CUPTI), OR
   collect the small-new-token regime under realistic batching so the measured wall is busy-bound.
2. Backbone: use busy (not wall) for the launch-bound small-new-token regime; let ctx + mixed
   FPM-vs-AIC MAPE arbitrate (no more nsys). If it lands near golden → done; if it undershoots
   ~8 ms → the launch-gap is a real production per-step cost and needs one term.
3. Validate on a BROADER hybrid workload (not just the terminal-chunk shape) before shipping.
4. Guard: must be provably inert for compute-bound (large prefill) and memory-bound (decode) —
   those already match golden ~1.0×.

## Artifacts (GPU-local, runs/ gitignored; analyzers committed)
- `runs/analyze_decode_escape.py`, `runs/analyze_prefill_floor.py` — the capture/spin decomposition.
- Verdict docs: SMALL_PREFILL_MECHANISM_DISCRIMINATION, SMALL_PREFILL_WHY_GOLDEN_16MS,
  SMALL_PREFILL_BATCHED_TARGET, SMALL_PREFILL_ENVELOPE_ROOTCAUSE (the full trail).

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

## Root cause: 1-GPU-collection launch-gap artifact — NOT comm / topology / capture
**Phenomenon (confirmed):** collector's 1-GPU-sharded step is launch-bound (~33 ms, GPU-busy only
~3.4 ms → ~30 ms GPU idle), while golden's real-tp4 step is busy-bound (~16 ms). golden's GPU is
kept full (TP all-reduce + real MoE), so its ~470 eager launches hide behind the work; the
1-GPU-sharded collection has no all-reduce + tiny moe-noop kernels, so the GPU idles between
launches and the gap is exposed.

**⚠️ Micro-causality NOT fully nailed (strong inference, not side-by-side nsys-confirmed):**
*why* the 1-GPU idle is larger than tp4's is the open part. Under a symmetric per-launch-cost model
it doesn't close: if CPU launch were identical both ways, golden (busy 16 ms) couldn't hide a 30 ms
launch chain either → it would also be ~30 ms. So the 1-GPU gap must be genuinely larger than tp4's,
and we have NOT nsys-confirmed why. What IS confirmed: it's the **1-GPU-sharded-vs-real-tp4 topology**
(control: plain vLLM, no marker, on the 1-GPU sharded config also ~34.7 ms; real tp4 ~16 ms) — not
the marker/moe-noop and not capture mode. The micro-mechanism (CPU launch pacing / scheduling /
inter-kernel sync that makes 1-GPU idle more) is unverified. See Open Questions.

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

### Confirmation: the gap is exactly ≤512 new tokens (big prefill is fine)
A filter-off check (`--no-filter-pathological-context`) exposes the big non-terminal chunks the
ctx-phase metric normally hides. golden's own latency has a STEP at the 512 capture boundary, and
AIC matches above it:

| ctx_tokens | ctx_kv | golden | AIC | error |
|---|---|---|---|---|
| 128 | 0 | 15.9 | 37.0 | +133% |
| 400 | 3696 | 16.4 | 51.5 | +214% |
| 496 | 528 | 16.2 | 51.8 | +220% |
| 528 | 0 | 40.6 | 39.3 | **−3%** |
| 3696 | 0 | 43.7 | 57.4 | **+31%** |

- ≤512 new tokens → golden ~16 ms (captured fast path); AIC ~37–51 ms (eager floor) → over 2.3–3.2×.
- >512 new tokens → golden ~40 ms (eager); AIC ~39–57 ms → matches (big-chunk MAPE 17%, ratio 1.14).
So the over-prediction is cleanly isolated to **≤512 new tokens**. Big prefill is accurate — no hidden
large-prefill bug (an earlier plot_fpm_vs_aic "4–8k=208%" was a different-grid artifact, refuted here).

### ctx-phase metric caveat (don't misread the "189%")
The summary tool's ctx-phase evaluates only CLEAN standalone/terminal-chunk targets: it filters
non-terminal prefill chunks (the big first chunks like 3696@0, 528@0) because their chunk boundary is
the scheduler's choice and AIC chunks differently → not apples-to-apples per-step. What's left is 3
small terminal/standalone points (128/400/496), all ≤512 → so ctx-phase MAPE (120–189%) is a
**small-prefill-only probe over 3 points, NOT a general ctx-accuracy number**. Product accuracy is the
mixed phase (38.8%), where ≤512 is only 7/71 steps (the >512 bulk is ~1.2×).

## How to FIX — VERDICT IN (`runs/BUSY_METRIC_VERDICT.md`): pure busy-metric is DEAD; viable = busy + comm + regime host-residual + shard-correct MoE
GPU-validated across all sizes (attribute, not fit). **golden sits BETWEEN the collector's two
estimates, and neither matches:**
```
busy+comm   <   GOLDEN   <   isolated WALL
~10 ms          15.5 ms       ~45 ms     (256 tok)
```
- **WALL over-predicts** (counts the full ~30 ms host idle serving overlaps away).
- **busy+comm UNDER-predicts at every size** — by a **non-kernel, regime-constant host residual**:
  ~5 ms in CAPTURED (≤512), ~30 ms in EAGER (>512), jumping at the 512 capture boundary. It is **not
  any kernel class** (all 6 compute classes are fully measured and sum to GPU-busy) and **does NOT
  scale with tokens** (golden is flat across sizes; the gap is constant, not growing) → it's the
  per-step host-dispatch/launch/cross-rank-sync idle that Σ-kernel-duration is blind to by construction.
- **Shard inequivalence** (advisor trap, confirmed): 1-GPU-sharded MoE-expert GEMM = **1.82×** golden's
  per-GPU value (ep4-on-1-GPU carries ~1.8× the per-expert token load). Non-MoE classes match ±5–15%.
  So the collector backbone over-counts MoE compute → the under-prediction gap is a *lower bound*.

So promoting `latency_source=gpu`+comm just **swaps WALL's over-prediction for a comparable
under-prediction** — it fixes neither small-prefill nor the 22%.

**Viable model (per the verdict):**
`golden ≈ GPU-busy + comm + regime-dependent host-residual (one const for captured ≤512, one for
eager >512, calibrated vs golden FPM) + a shard-correct MoE-expert term (ep-correct per-GPU, not
1-GPU-collapsed).`
- This is **calibration-flavored** (2 fitted constants per model), not pure physics.
- Equivalent alternative: **serving full-step collection** — golden's own path already contains the
  residual; but it abandons the per-layer decomposition.

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

## RESOLVED (nsys-confirmed): the 30ms idle is per-layer host-dispatch starvation
GPU experiment nailed it — see `SMALL_PREFILL_GAP_MICROCAUSALITY.md` (commit 5492156d):

- **A (1-GPU, real MoE, no marker) is gap-bound**: GPU-busy 8.7 ms ≪ wall 45.5 ms (**81% idle**).
  Same 510 eager + 41 graph launches as B → B's gap is **not** a marker/moe-noop artifact.
- **Idle host-time split**: launch-API (`cudaLaunchKernel`) 13%, sync 0.5% (ruled out), D2H 1.5%
  (ruled out), **host-dispatch between kernels (Python/PyTorch eager) 84.6% — dominant**.
- **Layer-count scan**: `wall_k = 1.61 + 0.854·k` (R²=0.996). Per-step fixed overhead (intercept)
  only **~1.6 ms**; the cost is **per-layer** (~0.85 ms/layer, ~0.65 ms of it idle launch-gap).
- **The "single-layer fixed overhead, amortizable" hypothesis is REFUTED** (intercept 1.6 ms, not
  ~30 ms). It's per-layer, repeated every layer (40 × ~0.65 ms ≈ 26 ms idle), **NOT** amortizable
  by collecting more layers — each layer brings its own dispatch bubble.

**Mechanism**: in the eager GDN path (30/40 hybrid layers) the GPU sits idle between kernels
waiting for the host (Python/PyTorch eager dispatch) to issue the next op → GPU starved ~80% of
the step. tp4's all-reduce (GPU kernel + cross-rank sync) overlaps/fills that gap; 1-GPU has no
all-reduce → the host-dispatch idle is exposed. **Fix lever = capture coverage** (full cudagraph
replay removes per-kernel host dispatch) — consistent with the original root cause.

**Deeper framing — it's ISOLATED-vs-SERVING, not really topology (resolves "30 ms idle > 16 ms golden"):**
the 30 ms idle is an *isolated-single-step* artifact. real-tp4 **ISOLATED is also ~40 ms** (host-dispatch
exposed), but real-tp4 **STREAM/serving is ~14 ms** — batching 3×. golden's 16 ms is the SERVING steady
state: the host pipeline is full, so dispatch hides under the continuous multi-request work stream. So
the "30 idle vs 16 wall" paradox dissolves — they're *different execution modes*, not the same step.
The dominant factor is **isolated-vs-serving (host overlap), and topology (1-GPU vs tp4) is secondary**
(tp4 isolated is slow too). The collector measures an isolated single step (for per-layer attribution)
→ inherently host-dispatch-exposed → it can NOT see the serving busy-bound value by measuring wall.
(Strong inference across experiments — 34.7 / 40 / 14 / 16 from different runs — not one side-by-side.)

## When/how to FIX (if revisited) — verdict settled
Pure GPU-busy+comm is **dead** (proven, above). Two viable paths, both calibration-flavored:
1. **busy + comm + regime host-residual + shard-correct MoE** (the verdict's model): measure
   GPU-busy (CUPTI) + comm; add a host-residual constant per regime (captured ≤512 / eager >512)
   calibrated vs golden FPM; use an **ep-correct** per-GPU MoE-expert term (not the 1.82×-inflated
   1-GPU-collapsed one). 2 fitted constants/model.
2. **serving full-step collection** (golden's own path) — contains the residual already, but abandons
   per-layer decomposition.
- Either way: validate on a BROADER hybrid workload (not the 1 terminal-chunk shape); guard inert for
  decode (matches golden ~1.0×). The shard-correct MoE term is needed regardless.

## The OTHER ~22% of mixed (after small-prefill) is the SAME family — not composition
Checked by reading `_get_mix_step_latency`: it is already `context_total + decode_delta` (decode
KV-attention increment), with `max(context,decode)` **explicitly rejected** (max drops decode
attention when prefill dominates). So the mixed residual is **NOT** a sum-vs-max composition bug —
that idea is dead. The remaining ~22% lives in the **ctx envelope**: golden ctx has the 512 capture
jump (≤512 ~16 ms → 528 ~40 ms → 1056 ~53 ms → 3696 ~50 ms), and AIC is over/under by size
(528 matches −3%; 3696 AIC 57 vs golden 50, **+14%**; mixed buckets: 512–1k **under** 0.93, 2–4k/>4k
**over** 1.22–1.27 — mixed directions, not a single bias).

So the full mixed 38.8 % (small-prefill ~17 pp + this ~22 %) is **ONE root**: layerwise *isolated wall*
vs golden *serving busy-bound / capture coverage*. It's a serving-fidelity problem, **not several quick
SDK fixes**. The fix family is the same for both: **busy-metric (GPU-busy + comm)** / **serving
full-step collection** / **calibration** — all need the same GPU validation (below).

## Decision-complete conclusion (the DEFERRED call stands)
Measuring the **wall of an isolated 1-GPU step** can't reproduce golden's serving busy-bound tp4 step
(isolated exposes ~30 ms host-dispatch idle that serving overlaps away). The clean direction — measure
**GPU-busy + comm** instead of wall (regime-independent) — is **promising but unverified** (real-MoE
busy 8.7 vs moe-noop 3.4 makes it far less dead, but the 8.7→16 completion + the contested comm term
need CUPTI instrumentation + cross-shape GPU verification). For a **bounded, deferred,
non-recommendation-moving** regime that's not worth opening now → DEFERRED. The 30 ms idle
micro-causality is settled (per-layer host-dispatch); isolated-vs-serving is the framing.

## Artifacts
- `runs/` (analyzers committed; heavy `.sqlite`/`.nsys-rep` are GPU-local):
  `analyze_decode_escape.py`, `analyze_prefill_floor.py` — capture/spin decomposition.
- Verdict docs: SMALL_PREFILL_MECHANISM_DISCRIMINATION, SMALL_PREFILL_WHY_GOLDEN_16MS,
  SMALL_PREFILL_BATCHED_TARGET, SMALL_PREFILL_ENVELOPE_ROOTCAUSE (the full trail).

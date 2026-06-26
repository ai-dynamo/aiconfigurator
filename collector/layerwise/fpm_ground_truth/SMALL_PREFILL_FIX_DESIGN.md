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

## How collector SHOULD collect: measure GPU-busy (launch-gap-free) + comm — NOT wall
Root of the error: collector measures the **wall of an ISOLATED step**, which includes ~30 ms of
host-dispatch idle that serving overlaps away. Fix = measure the **launch-gap-free GPU-busy**
(Σ per-kernel durations) + comm, because in serving steady-state the step IS busy-bound:
- GPU-busy is a *physical* quantity (kernel durations) — independent of isolated-vs-serving and of
  host dispatch → it equals the serving busy-bound latency.
- **Much LESS dead than it looked, but NOT reconciled** (NEW, from Part 1): A (1-GPU, **real MoE**)
  GPU-busy = **8.7 ms** — far above the **3.4 ms moe-noop** busy the prior "dead" call used. So the
  busy-metric starts close to golden 16, not at 3.4. BUT completing 8.7 → 16 needs the comm term, and
  that number is **contested**: golden-busy − A-busy ≈ **7 ms** by subtraction, vs the synced-AR-floor
  **~1.85 ms** from `SMALL_PREFILL_GAP_MICROCAUSALITY`/prefill_floor — they don't cleanly add
  (8.7 + 1.85 = 10.5 ≠ 16; these come from different nsys runs w/ inflation). So: **promising, NOT
  proven.** Needs GPU verification that backbone-busy + MoE-overlay + comm ≈ golden across shapes.

**So collector should:**
1. measure **GPU-busy = Σ per-kernel durations** (CUPTI), NOT execute_model wall → drops the isolated
   host-dispatch idle uniformly (and it's regime-independent, so isolated collection is fine);
2. keep the **comm (all-reduce) model** (a 1-GPU collection physically can't measure it);
3. assemble **backbone-busy + MoE-overlay + comm**.
   ⚠️ Needs GPU verification that this sum ≈ golden across shapes (one shape reconciling ≠ proof).

**Alternatives:** real-tp full-step serving collection (most faithful, but abandons per-layer
decomposition) · calibration of the ≤512 envelope (cheap, per-model). The GPU-busy metric (above)
is the cleanest IF it verifies across shapes.

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

## When/how to FIX (if revisited)
- **Leading candidate: GPU-busy + comm** (see "How collector SHOULD collect"). Promising (real-MoE busy
  8.7 ≫ moe-noop 3.4, so close to golden 16) but **NOT proven** — the comm term is contested (~7 by
  subtraction vs ~1.85 synced-floor) and 8.7+1.85≠16. Build = measure Σ-per-kernel-durations (CUPTI)
  instead of wall; assemble backbone-busy + MoE-overlay + comm. **Gate: GPU-verify sum ≈ golden across
  shapes** AND resolve the comm number first.
- Alternatives: real-tp full-step serving collection (most faithful, abandons per-layer decomposition)
  · calibration of the ≤512 envelope. All: validate on a BROADER hybrid workload (not the 1 terminal-
  chunk shape); guard inert for large-prefill + decode (those match golden ~1.0×).

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

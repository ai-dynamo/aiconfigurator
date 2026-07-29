<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Queueing (Pass-Calendar) Correction

| | |
|---|---|
| **Status** | Implemented (`aic-core/src/aiconfigurator_core/sdk/queueing/`, compat alias `aiconfigurator.sdk.queueing`) |
| **Replaces** | the empirical `_ttft_queuing_factor` heuristic and, for reporting, the blended-mean-only TTFT/TPOT columns |
| **Validation** | development-time DES oracle anchored clause-by-clause to the vLLM v1 scheduler source; recorded results in §5 |

## 1. What it is

An **algorithm-derived** (not data-fitted) correction for continuous-batching
queueing dynamics. It produces **distributions** of TTFT and ITL — not just
means — by evaluating the scheduler's own pass calendar: every request's
TTFT is (the pass in flight at arrival) + (its own prefill chunk passes),
and every ITL gap is a pass duration. Workload inputs follow the fidelity
contract of §3.1: from fixed shapes + closed-loop concurrency (W0) through
open-loop rates (W1), shape marginals/joints (W2/W3), verbatim trace replay
(W3), to session lanes with endogenous arrivals (W4).

One model, two precision tiers:

| Tier | Entry point | Cost | Accuracy role |
|---|---|---|---|
| **screening** (closed form) | `closed_form.operating_point_columns` | µs — pure arithmetic on quantities `run_agg` already computed, zero extra DB queries | populates the summary columns on the sweep hot path. Within one sweep the workload is fixed and candidates differ only in engine/parallel config, so its per-workload bias is shared across candidates and **ranking is preserved**; cross-workload quantitative use should go through the evaluator |
| **quantitative** (limit-cycle evaluator) | `evaluate_closed_loop` | ~ms–10ms | the same model evaluated numerically: a deterministic pass-level recursion (no RNG, no event heap, no per-token events) that captures the cohort effects the closed form approximates. Validated ≤10–15% (mostly 0.0%) across 9 config families — recorded results in §5 |

## 2. Term provenance (no fitted constants)

| Term | Source |
|---|---|
| `B_eff = B − b` (decode spends budget first) | vLLM v1 scheduler semantics (running set scheduled before waiting) |
| prefill staircase `ceil(k·isl_eff/B_eff)` | chunked-prefill scheduling loop |
| residual wait `E[T²]/(2E[T])` | renewal-theory residual life (inspection paradox) |
| transient window = initial concurrency burst | closed-loop dispatch semantics (all C arrive at t=0) |
| ITL gap weights `(c−1)/c` for mix passes | a mix pass stalls only the requests not being prefilled in it |
| `isl_eff = isl − prefix` in chunk counts | the scheduler budgets only tokens that still need compute, so cached tokens never consume the token budget — also what keeps the funnel bracket a valid bound on the evaluator's distribution for prefix rows |
| `WorkloadSpec.turnaround_ms` arrival-visibility delay | client/frontend per-request turnaround (HTTP → tokenize → IPC → waiting queue). ε = 0 (default) reproduces the DES oracle's idealized same-instant client; any ε > 0 selects the causal regime (a replacement can never catch the pass built at the completion instant). CAUTION — do not tune this to match measured loaded TTFT: within the deterministic recursion, ε beyond 0⁺ only moves the cycle along a phase staircase (tread ≈ decode-pass, riser ≈ mix-pass) that REDISTRIBUTES time between TTFT and TPOT under the Little's-law identity; the real engine is phase-mixed and measured steady TTFT is ε-invariant (§6.16). Use 0⁺ for the causal shape; use `ttft_anchor="identity"` for location |
| `WorkloadSpec.ingest_us_per_token` arrival-plane slope | the size-dependent part of the same turnaround path (client serialize → HTTP → frontend tokenize): a request dispatched at t reaches the scheduler's FCFS queue at t + turnaround_ms + isl·c/1000, so near-simultaneous dispatches are served shortest-first. Measured (h20e_sxm trtllm 1.3.0rc20, live probes): a 23k-token and a 1k-token prompt dispatched together ALWAYS serve small-first, flipping only once the small lags 60–100 ms ⇒ c ≈ 3.6 µs/token; in a Mooncake replay 96% of same-bucket big-then-small pairs emit first token small-first while client dispatch order is 0% inverted. Predictions are ordering-sensitive but magnitude-insensitive (any c > 0 gives the same same-instant ordering; ±2× shifts TTFT < 1%) — except for prefix-dominated workloads where TTFT ≈ the overhead itself. The slope is mildly superlinear in isl: 3.6 µs/tok at ≤23k vs 6.4/7.5 µs/tok at 50k/100k (fully-cached solo TTFT floor, InferenceX cc-traces replay: c-at-scale moved the ~66k-prefix small-work band from −49% to −9%) — measure c at the isl scale that dominates the trace. Placement verified: a fully-cached 105k request arriving during a prefill backlog gains ZERO post-queue delay (first token co-schedules with the backlog's last-chunk leftover), so the overhead is entirely pre-queue and absorbed by queue wait as this term assumes — none is in-service block-match/KV-attach. Traces with bucketed/batched timestamps make this correction load-bearing; c = 0 (default) keeps the dispatch plane |
| mixed-pass hook regime split (whole-prompt vs mid-prompt chunks) | run_agg's `t_mix` prices chunk attention via `prefix·floor(ctx_tokens/isl)` — the workload-average shape, exact only when every prompt prefills in a single pass (`floor = 1`). A mid-prompt chunk resumes at a per-pass past the signature cannot express (8k chunk at 16k past ≈ 1.45× a fresh one, h20e/trtllm), so the calendars route passes carrying a resumed or budget-clipped chunk to `prefill_ms(computed_before as prefix) + decode_ms`, which carries that state. Measured (Mooncake replay): hook-everywhere compounded per-pass optimism into TPOT −38% and in-flight N 7.8 vs 15; the split lands TPOT −5% with the whole-prompt regime (fixed-8192: hook off moved TTFT p99 only +0.4%) unchanged |
| `EngineSpec.async_scheduling` one-pass lookahead | vLLM `AsyncScheduler` (default ON since vLLM 0.24): pass k+1 is built while pass k executes, so an arrival during pass k joins pass k+2 at the earliest. Measured A/B @C=32 (b300, Qwen3-32B TP4): +176 ms TTFT p50, and sync-mode TPOT matches the calendar exactly (14.48 vs 14.48) while async runs faster (13.06 — the timing layer's per-step CPU component is hidden by the overlap, a perf-database property, not a calendar term). The flag encodes ADMISSION visibility, not engine overlap in general: TRT-LLM's overlap scheduler (default ON) preps execution for the already-admitted batch while admission still sees fresh arrivals, so TRT-LLM maps to False — measured (h20e_sxm, trtllm 1.3.0rc20, Qwen3-32B TP4, isl4096/osl256, C≥32): True overpredicts steady TTFT 25-30%, False lands within ~4% |
| `ttft_anchor="identity"` Little's-law location | saturated closed loop: each of C slots cycles in exactly C/X, so E[TTFT] = C/X − (osl−1)·E[TPOT] − turnaround, an accounting identity independent of arrival phase (verified: +15 ms injected client delay on a live server moved measured steady TTFT p50 by 0.1 ms). The calendar provides the distribution shape; the identity pins its location. Corollary: steady TTFT is the small residual of two large terms — its precision is bounded by osl × (TPOT error), an ill-conditioning of the QUANTITY, not of the model |
| additive TTFT stages: encoder, per-request dispatch overhead | the same additive terms the legacy `ttft` carries (run_agg's own composition); omitting them would make percentile screens permissive for multimodal rows |
| static degenerate mapping | static batching has no admission queue and no phase interference, by construction |

Policy: residuals against the oracle must be traced to a *nameable
mechanism* and then (a) modeled structurally, (b) exposed as a labeled
knob, or (c) documented as a scope boundary — never absorbed into a fitted
coefficient. Fitting is legitimate only in the timing layer (perf-database
measurement/interpolation), which this package consumes as a black box.

## 3. Backend calendars

| Backend | Calendar | Status |
|---|---|---|
| vLLM | fused pass (unified budget, chunked prefill shares remainder) | **validated** across 9 config families (§5) + live b300 vllm 0.24 (§6.16) |
| TRT-LLM | fused like vLLM + `GUARANTEED_NO_EVICT` admission cap | **validated** live on h20e_sxm trtllm 1.3.0rc20 (chunked on/off ITL signatures, KV-capped deep queueing ±3%, TPOT ≤2% at C≥32; see `TrtllmCalendar` flag comment) |
| SGLang | mixed-chunk pass (per-iteration extend budget is `chunked_prefill_size` SHARED across the batch, and mixed decode rows debit it — verified against sglang 0.5.14 `PrefillAdder`); with `EngineSpec.enable_mixed_chunk=False` (SGLang's own server default), alternating passes (dedicated prefill batches pause decode → ITL spikes are whole cohort prefill waves) | **validated** live on h20e_sxm sglang 0.5.14, both branches (mixed budget spike ±1.7%; alternating TPOT <1%, X exact at C=32) |

KV-pressure scope: the only modeled admission-under-KV-pressure semantics is
the GUARANTEED_NO_EVICT gate (TRT-LLM row above, validated in the KV-capped
deep-queueing arm). vLLM preempts-and-recomputes and SGLang retracts under KV
pressure — different dynamics that no calendar reproduces, so those calendars
REJECT `kv_capacity_tokens`/`guaranteed_no_evict` inputs loudly (and the
trtllm calendar rejects `kv_capacity_tokens` without the no-evict gate, i.e.
MAX_UTILIZATION) instead of silently returning optimistic numbers.

## 3.1 Workload-fidelity contract (input tiers × evaluator tiers × error bars)

Two orthogonal axes grade every prediction. The INPUT axis (W-tiers) is
"what you tell us about the traffic"; each tier buys a named capability and
degrades gracefully — every tier's degenerate input reproduces the tier
below bit-for-bit (parity-tested), so callers pass whatever they have.

| Tier | Input | What it buys | Validation record |
|---|---|---|---|
| W0 | `(isl, osl)` + `concurrency` | structural signatures, saturated throughput, SLA ordering | h20e trtllm/sglang + b300 vllm, ±1–6% |
| W1 | + `request_rate` (open loop) | queue-wait TTFT under real arrivals (closed loop cannot express it) | h20e trtllm tp4, TTFT p50 ±3–8% at ρ≤0.7; p99 −16~−24% with the correlation-free arrival stream (was −49~−54% under the anti-clustering rotation) |
| W2 | + `isl_quantiles` / `osl_quantiles` (marginals) | frees TTFT from the homogeneous-convoy artifact (measured 2× pessimism at cv 0.25) | h20e trtllm tp4, TTFT +2%/+31% at C=8/32 |
| W3 | + joint shape/prefix streams, empirical inter-arrival quantiles; `arrival_trace` exact replay | prefix-hit economics, isl↔osl correlation, super-Poisson burst first-order; verbatim (arrival, isl, prefix, osl) pairing | Mooncake conversation trace on h20e trtllm tp4: X exact, TTFT p50 −11%/p99 −7%, TPOT −5% (with ingest slope + hook regime split) |
| W4 | + session structure (`evaluate_sessions`: lanes of turns with think gaps — arrivals become an OUTPUT of the recursion) | a-priori prediction of agentic workloads whose arrival process depends on serving speed (no replay needed) | cc-traces 10 sessions/315 turns, zero arrival feed-in: TTFT bands −11~−35%, p90/p99 +8/+13%; emergent dispatch instants drift |Δ| p50 81 s over 6–40 min sessions, bias −105 s — the frozen decode-corner timing bias advances the timeline, desynchronizing prefill storms (the W4 error-propagation path; windowed non-stationarity remains future work) |

W3 trace timestamps must declare their reference plane: recorded traces
capture CLIENT DISPATCH, but the engine's FCFS queue orders by SCHEDULER
arrival, and the ingest path in between has a size-dependent slope
(`WorkloadSpec.ingest_us_per_token`, §2) that re-orders near-simultaneous
dispatches shortest-first. For traces with bucketed/batched timestamps
(Mooncake's 3 s buckets) this mapping dominates small-request TTFT —
consuming dispatch times as scheduler arrivals mispredicted the small-work
band's TTFT p50 by +146% while the same calendar with the plane mapping
lands −20%.

The EVALUATOR axis is the existing cost ladder: E0 closed-form static (µs,
whole sweep), E1 pass-calendar recursion (ms, `--sla-refine`; sole consumer
of W1+ inputs — E0 degrades them to means), E2 DES oracle (validation gate
only), E3 live benchmark. Input upgrades do not change evaluator cost —
W2/W3 run in the same pass loop.

Error-bar clauses that travel with every tier (measured, see §5/§6):
steady-TTFT precision is bounded by osl × (timing-layer TPOT error) in
closed loop (§6.16) and amplified by 1/(1−ρ) near capacity in open loop —
treat ρ ≳ 0.8 open-loop predictions as optimistic bounds; deterministic
synthetic arrival streams use a correlation-free per-period shuffle (count dispersion 0.96 vs Poisson 1.0); the residual finite-period truncation leaves open-loop p99 ~16–24% light at ρ≥0.8, previously ~2× under
W4. `QueueingReport.workload_fidelity` (set by every evaluator; see
`workload_fidelity()`) declares the consumed tier so downstream consumers
can gate on prediction quality without re-deriving it; evaluators that do
not consume a tier REJECT its inputs loudly (e.g. the fixed-shape disagg
tandem raises on shape quantiles) instead of silently downgrading.

## 4. New summary columns (additive; legacy `ttft`/`tpot` untouched)

`ttft_steady_{mean,p50,p90,p99}`, `ttft_transient_{mean,max}`,
`itl_{mean,p50,p99}` in `ColumnsAgg` / `ColumnsStatic` / `ColumnsDisagg`.

- **agg**: screening tier at the `run_agg` operating point.
- **static**: degenerate — `ttft_steady_* == ttft`, `itl_* == tpot`.
- **disagg**: composed tier by default — TTFT side follows the prefill
  stage (static batch semantics on the prefill worker); ITL side is a
  single mass at decode `tpot` — decode workers have no prefill
  interference, which is the measurable signature of disagg vs agg (agg
  `itl_p99` spikes to the mix-pass duration, e.g. 190ms vs `itl_p50` 9.3ms
  for Llama-3.1-8B @ h200). With `--sla-refine`, the report boundary
  upgrades disagg rows to the TANDEM-RECURSION quantitative tier
  (`sdk/queueing/disagg.py`): both pools rebuilt from the row's (p)/(d)
  metadata with their own timing models (heterogeneous P/D supported),
  KV handoff computed on a max-min-fair per-NIC transfer fabric wired from
  the system spec, first token emitted prefill-side (the handoff lands in
  the first ITL gap), and phase-mixed output (see §6.15).

Additionally: `ttft_steady_p99_{lo,hi}` (the cohort bracket — structural
bounds on the steady distribution's support, used by the sweep funnel) and
`queueing_tier` ("screening" | "quantitative" | "static" | "composed"),
which makes the precision class of every row's numbers visible.

`ttft_mean(N)` (benchmark-length-blended mean) is deliberately NOT a column:
N is a property of the measurement, not the deployment. It is available as
`QueueingReport.ttft_mean_n`.

## 4.1 Percentile SLA semantics and the sweep funnel

SLA targets are (value, percentile) pairs on the steady-state
distributions. `RuntimeConfig` gains `ttft_percentile` / `tpot_percentile`
/ `itl_percentile` / `request_latency_percentile` (supported: 0.5, 0.75,
0.9, 0.95, 0.99, 0.999) plus an optional `itl` target (streaming-smoothness
SLA — the metric where the calendar is most accurate and where agg vs
disagg differ most). Defaults: **p50 for ttft/tpot/request_latency**
("typical request under sustained load"), **p99 for itl** (the tail is the
point of a smoothness SLA). p999 is the *calendar* p999 (deterministic
tail); real-deployment p999 adds stochastic effects outside the model.

The legacy `ttft` scalar has no stable equivalent in these semantics: it
behaves like `ttft_mean(N)` at an N implicitly baked into its fitted
constants (~420 on the reference workload) and drifting per family. It
remains emitted for reference; feasibility no longer uses it on the sweep
path.

**Screening-tier scope**: without `--sla-refine`, percentile enforcement
is available only for the metrics with stored screening distributions —
**TTFT** (full quantile set) and **ITL** (two-mass anchors). TPOT and
request-latency percentiles require the evaluator (`--sla-refine`); until
then their legacy mean screens apply unchanged. Open-loop (request-rate)
queueing is future work and not exposed.

Feasibility is resolved by a two-stage funnel (`sdk/queueing/refine.py`):

1. **wide-keep**: reject a candidate only when the bracket lower bound
   `ttft_steady_p99_lo` violates the target — screening bias can never
   falsely reject (the bracket bounds the distribution support, so one
   bracket serves every percentile);
2. **lazy quantitative resolution**: candidates whose bracket straddles
   the target are re-scored with the evaluator in throughput order until
   `top_k` feasible rows are confirmed; every refined row is checked
   against ALL requested constraints at their percentiles. Unresolved
   straddlers are kept (conservative) with `queueing_tier == "screening"`
   visible; skips are logged, never silent.

Measured cost on the reference case: loose SLA (no straddlers) ~2x sweep
time; boundary-tight SLA ~7x. Measured benefit: with `--ttft 210` the
funnel rescued 16 configs — including the top-throughput one — that the
legacy scalar (283ms) would have falsely rejected; their quantitative p50
is 123ms.

## 5. Validation (2026-07-18, recorded)

Methodology: a development-time DES oracle — a discrete-event simulation of
vLLM v1 iteration-level scheduling — was driven with timing functions
IDENTICAL to the model's, so residuals isolate scheduling semantics. The
oracle's own semantics were audited clause-by-clause against the vLLM v1
scheduler source (v0.24.0, the version the perf database ships, at audit
time). The clauses are semantic, not line references — upstream scheduling
changes surface as validation-gate drift (§6):

| Clause | vLLM v1 scheduler behavior |
|---|---|
| unified token budget | one per-step budget (`max_num_scheduled_tokens`) is shared by decode tokens and prefill chunks |
| running set scheduled first | the running set is served before the waiting queue, in admission order; each decode consumes one budget token |
| chunked prefill budget cap | a prefill chunk is `min(remaining_prompt, remaining_budget)` |
| chunked-off whole-prompt gate | with chunked prefill disabled, admission stops once a whole prompt no longer fits the remaining budget |
| admission concurrency cap | waiting-queue admission stops at `max_num_seqs` running requests |
| waiting-admission alloc failure | a waiting request whose KV cannot be allocated stays queued (no preemption); admission stops for the step |
| running-path alloc failure | KV pressure on the running path preempts the newest request (LIFO) with full recompute |
| prefix-cached tokens skip the budget | the scheduler budgets only tokens that still need compute; cached tokens are subtracted first |
| fused execution (one pass = one forward) | decode tokens + prefill chunks form a single varlen batch; a completing prefill's first token is sampled off the final chunk's logits in the same pass |

Battery: 9 agg config families (isl 512–8192, osl 16–512, C 1–128, budget
2048–8192, chunked on/off, prefix). Recorded results:

- **Evaluator**: within 10% on TTFT steady/transient mean/p50/p99 and
  blended mean — most metrics 0.0%; ITL within 15%. One documented
  exemption: prefix workloads' `itl_p99` (the constant-hit assumption locks
  a different cohort phase than a cold-start cache; the mix-pass mass point
  shifts by one cohort step; TTFT unaffected).
- **Closed-form screening tier**: family-dependent bias, small on
  long-prompt families and up to ~2× on cohort-dominated short-prompt
  families — the mechanism (cohort locking is initial-condition dependent)
  is exactly what the evaluator exists to capture. Screening/ranking role
  per §1.
- **End-to-end** (real perf DB, Llama-3.1-8B / h200_sxm / vLLM 0.24.0,
  isl4096/osl256/C32, measured against a development-time replay baseline):
  legacy `ttft` −30%; closed form −6.9% (blended); evaluator −2.5%,
  `itl_p50` exact, `itl_p99` 0.1%.

Re-validated 2026-07-19 after two semantics corrections (prefill completers
are not decode rows — the fused-execution clause above; effective-isl chunk
counts). The oracle shared the completer-as-decode-row convention with the
evaluator — both sides wrong together, invisible to the gate, exactly the
shared-convention blind spot §6.13 describes — and was corrected against
the engine first. All 9 families then re-gate within tolerance on the
evaluator tier; the C1 family is now exact (0.0% on every metric — it
previously carried a spurious decode-row term on both sides). The prefix
family keeps its documented `itl_p99` exemption; TTFT residuals unchanged.

Timing-layer note: the evaluator's mixed passes now delegate to the same
mix-step runner run_agg uses (`DatabaseTimingModel.mixed_pass_ms` →
`_get_mix_step_latency`, batching efficiency included) instead of a
prefill+decode sum that double-counts the shared non-attention cost. The
scheduling-semantics gate is timing-independent and unaffected (it drives
the evaluator with plain prefill/decode callbacks); the recorded
end-to-end figure above predates this delegation — re-measure at the next
replay session.

End-to-end (real perf DB, Llama-3.1-8B / h200_sxm / vLLM 0.24.0, disagg
report upgrade): the tandem tier's first measurement of composed-tier
optimism — a rate-matched 5P3D deployment at C=252 shows composed
TTFT 183.8ms (single static prefill batch) vs tandem p50 1185.7ms
(prefill-pool queueing dominates at deep closed-loop concurrency), while
ITL stays a tight single mass (13.6ms, no prefill interference; the
handoff carries 1/(osl-1) of the gap mass and is invisible at p99 for
osl=256). Elimination semantics unchanged — the row's tier makes the
precision class visible.

Disagg (P/D tandem) families added 2026-07-19: the tandem recursion vs
the DES `DisaggSimulator`, identical timing and transfer fabric on both
sides, same-phase initial conditions. Four gated families (1P1D, fan-in
2P1D, bandwidth-tight 2P1D, 2P2D): TTFT steady mean/p50/p99 and ITL
p50/p99 essentially exact (mostly 0.0%; itl_mean <= 2.6%), including the
fan-in transfer spike in `itl_p99` reproduced to 0.0%.

The oracle and its gate live in `tools/queueing_oracle/` (stdlib-only DES
of vLLM v1 iteration-level scheduling + the 9-family gate), and the gate
runs in CI as a marked test (`tests/unit/sdk/queueing/test_oracle_gate.py`,
~1s) — so scheduler-semantics drift (§6.7) has a standing executable
detector, and future extensions (disagg gate families, variable-length
workloads, a real-`Scheduler` driver as semantic authority) build on the
same harness.

## 6. Failure modes

Loud (guarded, fails fast):

1. **Non-stationary input** — the API only accepts fixed isl/osl + C or λ;
   traces are structurally impossible to pass in.
2. **Pathological configs** — budget smaller than concurrency, prompt
   larger than budget with chunked prefill off: evaluator raises.
3. **OOM configs** — inherited from `run_agg`'s existing memory check; the
   queueing columns of an OOM row are as invalid as its legacy columns.

Silent (each with a designated detector):

4. **Wide isl/osl distributions.** Assumes fixed lengths; feeding means of
   a heavy-tailed workload smears the staircase and the ITL bimodality in
   reality but not in the model. Detector: the DES supports per-request
   lengths; extend validation before trusting.
5. **KV-pressure / preemption regime.** Out of domain: the model carries no
   KV state. AIC's existing static KV-capacity check gates configs before
   the model runs; configs near dynamic saturation (concurrency ×
   (isl+osl) approaching KV capacity) should not be trusted and belong to
   simulation-level tooling.
6. **Prefix-cache dynamics.** `prefix` is a constant steady-state hit
   assumption; under cache-capacity pressure real hit rates are
   history-dependent and lower. Measured secondary effect: cohort-phase
   lock shifts `itl_p99` by one cohort step (~30%) while TTFT and
   throughput stay aligned.
7. **Scheduler-semantics drift.** If a backend changes scheduling upstream,
   the calendar silently diverges. Detector: the validation gate — drift
   shows up as growing residuals with a nameable cause.
8. **Router-layer effects.** Multi-worker deployments assume balanced
   round-robin dispatch; affinity/queue-depth routing policies are out of
   scope for the analytical model.
9. **Multimodal rows** are refinable when the runtime image context is
   available (vision tokens join the prefill length; encoder latency
   shifts the TTFT/e2e distributions additively, matching run_agg's own
   composition). Where it is not (e.g. rows re-scored purely from
   metadata), they stay at screening tier — visibly.
10. **Metric-definition mismatch.** `ttft_steady_*` must be compared against
   warmup-excluded benchmarks; blended means against full-run benchmarks
   with matching N.
11. **Speculative decoding (`nextn > 0`).** The engine emits up to nextn+1
   tokens per verified step (vLLM: `num_tokens_with_spec`), breaking the
   1-token-per-pass invariant. Since PR #1405 aic-core prices the FULL
   verification iteration (nextn shapes the op graph) and acceptance is a
   summary-level projection above core (`SpeculativeDecodingProfile`:
   tpot ÷ (1 + nextn_accepted)). The tandem tier divides decode-pass
   pricing by the same progress factor so `itl_p50` stays on the sweep's
   projected-tpot basis (amortized cadence, not the real lumpy gap
   distribution — a full step gap followed by zero-gaps). Screening and
   agg-evaluator `itl_*` for nextn rows are NOT projected: they describe
   the full verification-step cadence, which is the physical gap length
   but sits `(1 + nextn_accepted)×` above the row's projected tpot.
   Detector: compare against a token-timestamped benchmark of an MTP model.
12. **Attention-DP prefill cadence (`dp > 1`).** Under data parallelism
   vLLM defers new prefills on non-cadence steps (prefill throttling, so
   ranks batch prefills on cadence-aligned steps); the per-rank calendar
   assumes every pass may mix. TTFT distributions for dp>1 rows can shift
   by the cadence interval.
13. **Closed-loop admission boundary convention.** The evaluator admits a
   replacement request into the pass that starts at its arrival instant
   (arrival = completion pass boundary). A real closed-loop client's next
   request typically arrives after the engine has already scheduled the
   following iteration and waits ~one extra pass. The DES oracle shares the
   evaluator's convention, so §5 residuals cannot see this term; its
   direction matches the recorded −2.5% end-to-end underestimate. Nameable
   mechanism, kept as a documented boundary until the real-scheduler oracle
   (which owns the true `schedule()` boundary) can measure it.
14. **Async scheduling / overlapped batches.** `max_concurrent_batches > 1`
   and async scheduling overlap step N+1's schedule with step N's execute —
   the "one pass = one gap" mapping blurs. Not active in the validated
   configuration; watch item for newer engine defaults.
15. **Disagg cohort-phase multi-stability.** The P/D tandem system has
   multiple steady limit cycles selected by the initial cohort phase — a
   simultaneous burst locks a large-batch prefill cycle, spread arrivals
   lock a small-batch pipeline cycle, with TTFTs differing by multiples.
   Which cycle a real deployment lands in is set by arrival jitter outside
   the model. Structural handling: `evaluate_disagg_mixed` reports an
   equal-weight mixture over a deterministic set of initial-arrival
   staggers (no RNG; each component is a valid limit cycle); the gate
   compares single phases with matched initial conditions.
   Measured instance of the boundary (h20e_sxm, trtllm 1.3.0rc20 native
   disagg 2P1D, Qwen3-32B tp1, isl4096/osl256): at the saturation knee
   (C=8 ≈ 2 prefill workers × knee) every stagger phase and any
   kappa/turnaround locks the zero-wait pipeline attractor (X→capacity,
   TTFT≈solo), while the real deployment's service-time jitter selects a
   queued attractor (X 0.814 vs 1.006 predicted; TTFT quantized at
   integer multiples of the solo prefill — measured 3.6s mean vs 2.0
   predicted). Below the knee (C≤6: within 3-9%) and beyond it (C≥2×knee:
   within 2-4%) the recursion is accurate; the knee itself is
   jitter-selected and out of deterministic scope. Do not compensate by
   fitting eps or kappa — that is phase compensation, same rejection as
   item 16(c).
16. **Closed-loop steady TTFT is ill-conditioned at saturation — measured
   resolution of items 13/14.** Validated against live vLLM 0.24.0 on
   b300_sxm (Qwen3-32B bf16 TP4, dummy weights, isl4096/osl256, closed
   loop C∈{1,8,32,128}; `vllm bench serve`). Throughput/TPOT/ITL/e2e match
   the evaluator within ≤12% (mostly ≤5%), including the agg mix-pass
   `itl_p99` spike (137 measured vs 127 predicted @C=32). Steady TTFT does
   not: measured p50 680 ms vs 178 predicted @C=32. Decomposition:
   (a) item 13's boundary convention: any client turnaround ε > 0 leaves
   the same-instant knife edge — the causal 0⁺ regime costs ~+1 mix pass;
   (b) item 14's async lookahead (vLLM ≥ 0.24 default ON): +176 ms measured
   A/B, now a calendar term (`EngineSpec.async_scheduling`);
   (c) the remainder is NOT a mechanism to model: within the deterministic
   recursion ε traces a staircase (tread ≈ decode-pass, riser ≈ mix-pass,
   evaluator: 381/508/634/885 ms at ε=1/8/16/32 @C=32) but the staircase
   only redistributes time between TTFT and TPOT under the Little's-law
   cycle identity E[TTFT] = C/X − (osl−1)·E[TPOT] − ε_client; the real
   phase-mixed engine sits at ONE split and is ε-invariant — injecting
   +15 ms of client delay through a request-path proxy moved measured
   steady TTFT p50 by 0.1 ms (679.9 → 679.8). Fitting ε to loaded TTFT is
   therefore phase-compensation, i.e. overfitting, and is rejected.
   Consequence: report closed-loop steady TTFT via `ttft_anchor="identity"`
   (shape from the causal 0⁺ calendar, location from the identity) and
   carry its honest error bar, osl × (timing-layer TPOT error) — at
   osl=256, a 3% TPOT bias is ~±100 ms of TTFT. Improving saturated-TTFT
   precision is a TIMING-layer problem (per-step CPU cost under async
   overlap), not a calendar problem.

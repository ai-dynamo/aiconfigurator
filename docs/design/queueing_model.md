<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Queueing (Pass-Calendar) Correction

| | |
|---|---|
| **Status** | Implemented (`src/aiconfigurator/sdk/queueing/`) |
| **Replaces** | the empirical `_ttft_queuing_factor` heuristic and, for reporting, the blended-mean-only TTFT/TPOT columns |
| **Validation** | development-time DES oracle anchored clause-by-clause to the vLLM v1 scheduler source; recorded results in §5 |

## 1. What it is

An **algorithm-derived** (not data-fitted) correction for continuous-batching
queueing dynamics. For stationary workloads (fixed isl/osl/prefix,
closed-loop concurrency or open-loop Poisson rate) it produces
**distributions** of TTFT and ITL — not just means — by evaluating the
scheduler's own pass calendar: every request's TTFT is (the pass in flight
at arrival) + (its own prefill chunk passes), and every ITL gap is a pass
duration.

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
| static degenerate mapping | static batching has no admission queue and no phase interference, by construction |

Policy: residuals against the oracle must be traced to a *nameable
mechanism* and then (a) modeled structurally, (b) exposed as a labeled
knob, or (c) documented as a scope boundary — never absorbed into a fitted
coefficient. Fitting is legitimate only in the timing layer (perf-database
measurement/interpolation), which this package consumes as a black box.

## 3. Backend calendars

| Backend | Calendar | Status |
|---|---|---|
| vLLM | fused pass (unified budget, chunked prefill shares remainder) | **validated** across 9 config families (§5) |
| TRT-LLM | fused like vLLM + `GUARANTEED_NO_EVICT` admission cap | structural, **not validated** against a TRT-LLM reference |
| SGLang | alternating passes (dedicated prefill batches pause decode → ITL spikes are whole prefill batches) | structural, **not validated** against an SGLang reference |

## 4. New summary columns (additive; legacy `ttft`/`tpot` untouched)

`ttft_steady_{mean,p50,p90,p99}`, `ttft_transient_{mean,max}`,
`itl_{mean,p50,p99}` in `ColumnsAgg` / `ColumnsStatic` / `ColumnsDisagg`.

- **agg**: screening tier at the `run_agg` operating point.
- **static**: degenerate — `ttft_steady_* == ttft`, `itl_* == tpot`.
- **disagg**: TTFT side follows the prefill stage (static batch semantics on
  the prefill worker); ITL side is a single mass at decode `tpot` — decode
  workers have no prefill interference, which is the measurable signature
  of disagg vs agg (agg `itl_p99` spikes to the mix-pass duration, e.g.
  190ms vs `itl_p50` 9.3ms for Llama-3.1-8B @ h200).

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
scheduler source:

| Clause | vllm/v1/core/sched/scheduler.py |
|---|---|
| unified token budget | `token_budget = max_num_scheduled_tokens` (:334) |
| running set scheduled first | `while req_index < len(self.running) and token_budget > 0` (:346) |
| chunked prefill budget cap | `num_new_tokens = min(num_new_tokens, token_budget)` (:372) |
| chunked-off whole-prompt gate | break when `num_new_tokens > token_budget` (:652) |
| admission concurrency cap | `len(self.running) == max_num_running_reqs` (:534) |
| waiting-admission alloc failure | put back + break — no preemption (:716-723) |
| running-path alloc failure | preempt (LIFO pop) with recompute (:437-471) |

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

The oracle and its gate script are development tooling and are not part of
this change; they are preserved on a development branch for future
re-validation (e.g. after upstream scheduler changes).

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

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Queueing (Pass-Calendar) Model

| | |
|---|---|
| **Status** | Implemented (`src/aiconfigurator/sdk/queueing/`) |
| **Replaces** | the empirical `_ttft_queuing_factor` heuristic and, for reporting, the blended-mean-only TTFT/TPOT columns |
| **Oracle** | `tools/queueing_oracle/` — Python DES verified to 0.0% against dynamo mocker on every reported metric |

## 1. What it is

An **algorithm-derived** (not data-fitted) model of continuous-batching
queueing dynamics. For stationary workloads (fixed isl/osl/prefix, closed-loop
concurrency or open-loop Poisson rate) it produces full **distributions** of
TTFT and ITL — not just means — by evaluating the scheduler's own pass
calendar: every request's TTFT is (residual of the pass in flight at arrival)
+ (its own prefill chunk passes), and every ITL gap is a pass duration.

Three consumption tiers, cheapest first:

| Tier | Entry point | Cost | Used by |
|---|---|---|---|
| operating-point closed form | `closed_form.operating_point_columns` | µs (pure arithmetic on quantities `run_agg` already computed — zero extra DB queries) | every `run_agg` row (sweep hot path) |
| limit-cycle evaluator | `evaluate_closed_loop` / `evaluate_disagg` | ~ms–10ms | reports, top-K refinement, validation |
| DES oracle | `tools/queueing_oracle/` | 10–100ms/1k requests | arbitrary timestamped traces; CI parity gate |

## 2. Term provenance (no fitted constants)

| Term | Source |
|---|---|
| `B_eff = B − b` (decode spends budget first) | mocker `scheduler/vllm/core.rs` running-set-first loop (vLLM v1 semantics) |
| prefill staircase `ceil(k·isl_eff/B_eff)` | chunked-prefill scheduling loop |
| residual wait `E[T²]/(2E[T])` | renewal-theory residual life (inspection paradox) |
| transient window = initial concurrency burst | closed-loop dispatch semantics (all C arrive at t=0) |
| open-loop `W_q` | M/D/1 Pollaczek–Khinchine |
| disagg TTFT = W_P + S_P + handoff + first decode pass | verified against dynamo offline disagg replay: the user-visible first token is emitted by the **decode** stage; handoff defers decode enqueue |
| KV handoff `isl × bytes/token ÷ bandwidth` | mocker `common/utils.rs::compute_prefill_handoff_delay_ms` |
| static degenerate mapping | static batching has no admission queue and no phase interference, by construction |

Policy: residuals against the oracle must be traced to a *nameable mechanism*
and then (a) modeled structurally, (b) exposed as a labeled knob, or (c)
documented as a scope boundary. Never absorbed into a fitted coefficient.

## 3. Backend calendars

| Backend | Calendar | Status |
|---|---|---|
| vLLM | fused pass (unified budget, chunked prefill shares remainder) | **validated**: 0.0% on TTFT p50/p99/transient/ITL-quantiles vs DES/mocker across 5 config families |
| TRT-LLM | fused like vLLM + `GUARANTEED_NO_EVICT` admission cap | structural, **not validated** against a TRT-LLM oracle |
| SGLang | alternating passes (dedicated prefill batches pause decode → ITL spikes are whole prefill batches; `max_prefill_tokens`/`chunked_prefill_size` budgets) | structural, **not validated**; the mocker has an SGLang variant — validation path exists |

## 4. New summary columns (additive; legacy `ttft`/`tpot` untouched)

`ttft_steady_{mean,p50,p90,p99}`, `ttft_transient_{mean,max}`,
`itl_{mean,p50,p99}` in `ColumnsAgg` / `ColumnsStatic` / `ColumnsDisagg`.

- **agg**: from `operating_point_columns` at the `run_agg` operating point.
- **static**: degenerate — `ttft_steady_* == ttft`, `itl_* == tpot`.
- **disagg**: TTFT side follows the prefill stage; ITL side is a single mass
  at decode `tpot` (no prefill interference on decode workers — the
  measurable signature of disagg: in agg, `itl_p99` spikes to the mix-pass
  duration, e.g. 190ms vs `itl_p50` 9.3ms for Llama-3.1-8B @ h200).
  Full tandem distributions via `sdk.queueing.evaluate_disagg`.

`ttft_mean(N)` (benchmark-length-blended mean) is deliberately NOT a column:
N is a property of the measurement, not the deployment. It is available as
`QueueingReport.ttft_mean_n`.

**SLA semantics recommendation**: constrain on `ttft_steady_p99` (industry
norm), report `ttft_transient_max` as the cold-start / synchronized-burst
envelope. The legacy blended `ttft` heuristic underestimated dynamic TTFT by
~30% on the reference workload while its N-dependence made recommendations a
function of benchmark length.

## 5. Validation results (2026-07-18)

Formula vs DES (identical timing functions on both sides; residual =
scheduling-semantics error only). Five agg config families
(isl 512–8192, osl 64–512, C 16–128, budget 4096–8192, chunked on/off):

- `ttft_steady_p50/p99`, `ttft_transient_mean`: **0.0%** everywhere
- `ttft_steady_mean`: ≤1.4% | throughput: ≤1.2% | `itl_p50`: 0.0%
- `itl_mean`: ≤8.6%; `itl_p99`: 0.1% (one outlier 12.6% at isl1024/C64)
- disagg (1P1D tandem): TTFT 0.1%, ITL 0.0%, throughput 7.3%

End-to-end cross-check against dynamo mocker with real AIC timing
(Llama-3.1-8B, h200_sxm, vLLM 0.24.0, isl4096/osl256/C32, N=200):
mocker TTFT mean 500.3ms; legacy `ttft` 350.6 (−30%); this model's
N-blended mean 466 (−6.9%); `itl_p50` exact, `itl_p99` −4%.

Run: `PYTHONPATH=src:tools/queueing_oracle python3 tools/queueing_oracle/validate_formula.py`

## 6. Failure modes

Loud (guarded, fails fast):

1. **Non-stationary input** — the API only accepts fixed isl/osl + C or λ;
   traces are structurally impossible to pass in. Escalate to the DES.
2. **Pathological configs** — budget smaller than concurrency, prompt larger
   than budget with chunked prefill off: evaluator raises `RuntimeError`.
3. **OOM configs** — inherited from `run_agg`'s existing memory check; the
   queueing columns of an OOM row are as invalid as its legacy columns.

Silent (the dangerous ones — each has a designated detector):

4. **Wide isl/osl distributions.** v1 assumes fixed lengths. Feeding means of
   a heavy-tailed workload smears the staircase and the ITL bimodality in
   reality but not in the model. Detector: DES supports per-request lengths;
   extend validation with distributional workloads before trusting.
5. **KV-pressure / preemption regime.** The calendar carries NO KV state: no
   preemption, no eviction. Near saturation (C·(isl+osl) approaching KV
   capacity) the real engine enters preemption storms (DES measured 3000+
   preemptions and throughput collapse) while the formula predicts health.
   Guard: treat `C·(isl+osl) > ~0.9 × kv_capacity_tokens` as out of domain;
   AIC's static KV check catches most, the DES catches the dynamic band.
6. **Prefix-cache dynamics.** `prefix` is a constant steady-state hit
   assumption; under cache-capacity pressure real hit rates are
   history-dependent and lower. The DES models block-level caching.
7. **Calendar drift.** If a backend changes scheduler semantics upstream
   (vLLM async scheduling, SGLang policy changes), the calendar silently
   diverges. Detector: the CI parity gate (validate_formula vs DES, DES vs
   mocker) — drift shows up as growing residuals with a nameable cause.
8. **Router-layer effects.** Multi-worker agg assumes balanced round-robin;
   disagg approximates dispatch policy with the `prefill_inflight_cap` knob
   (κ=None engine-batched vs κ=1 serialized ≈ kv_router; measured spread
   ~20% on TTFT mean). KV-affinity routing policies are out of scope for
   the analytical model; use router-level simulation (e.g. the Dynamo
   mocker/replay stack) when those effects matter.
9. **Metric-definition mismatch.** `ttft_steady_*` must be compared against
   warmup-excluded benchmarks; blended means against full-run benchmarks
   with matching N. Comparing across definitions is user error the docs
   must prevent.

# Closed-loop latency refinement (post-processing)

Sweep result rows carry `ttft`/`tpot` computed with per-operating-point
heuristics (agg: a clamped queuing factor; disagg: a constant prefill
pre-correction). This feature makes result rows **self-contained** and adds
a post-processing step that replays the scheduler's own arithmetic to
produce quantitatively refined latencies — without touching the prediction
pipeline.

## Recorded columns (additive)

Agg rows (`ColumnsAgg`): `mix_step_ms`, `genonly_step_ms`,
`prefill_step_ms`, `num_mix_steps`, `num_genonly_steps` — the
operating-point step timings `run_agg` already computes (through the
compiled engine step when routed there; the values are the FFI-returned
scalars, identical under either engine, so no Rust-side change is needed
and no parity-governed path is touched).

Disagg rows (`ColumnsDisagg`): `(p)prefill_step_ms` — the raw solo context
latency of one prefill worker, captured *before* the autoscale TTFT
pre-correction is applied.

## The estimators (`sdk/closed_loop_ttft.py`)

`estimate_closed_loop_latency` replays the fused continuous-batching pass
calendar (budget arithmetic per pass, admission-ordered chunking, closed
loop with visibility delay) — a deterministic recursion, no RNG, no fitted
constants, milliseconds per operating point. `estimate_disagg_closed_loop_latency`
replays the P/D tandem (whole-prompt prefill batches behind a round-robin
router, decode-attach handoff, per-iteration decode). Timing enters as the
recorded row scalars.

`refine_closed_loop_latency(df)` dispatches per row shape and returns a
copy with additive `ttft_refined` / `tpot_refined` / `throughput_refined`
columns. The three are one consistent timeline — they satisfy the
closed-loop identity `C/X = TTFT + (osl-1) * TPOT` — so consume them as a
set; mixing a refined column with a legacy one breaks that conservation.
Rows that cannot be priced keep NaN.

## Why a recursion instead of a formula

Closed-loop TTFT under continuous batching is not monotone in concurrency
(replacement prefill chunks stretch everyone's passes; past the knee,
larger decode batches eat the cycle and TTFT recedes), and any closed-form
multiplier misses that shape. The recursion reproduces it because both
effects emerge from iterating the budget arithmetic.

## Validation

The figures below are one-off measurements (August 2026, Qwen3-32B,
H20 trtllm perf data, isl4096/osl256), recorded with their configuration
rather than reproduced in CI; the evaluator comparisons ran against the
gated evaluator on the queueing-model branch. The in-repo unit tests
(`tests/unit/sdk/test_closed_loop_ttft.py`) lock the estimator semantics
these results rely on (closed-loop identity, P/D imbalance signatures,
admission rules, DataFrame refinement).

Against the full queueing-model evaluator on identical timing (Qwen3-32B,
H20, isl4096/osl256):

- agg estimator, exact timing callables: TTFT/TPOT/throughput within 0.2%
  over C=2..64 (bit-identical up to C=8);
- agg estimator fed only the recorded row scalars: within ~3% over the
  same sweep (the linear-in-tokens chunk reconstruction);
- disagg tandem estimator vs the disagg evaluator (1P1D/2P1D/2P2D, serial
  and batched prefill, shallow to deep queueing): bit-identical TTFT and
  throughput at all nine validation points.

Against real serving (trtllm-serve 1.3.0rc20, H20 tp4, chunked prefill,
9 agg closed-loop points, isl 1024-8192, osl 128-256, C 4-48, predictions
generated before the measurements): throughput within 5.3% on 7 of 9
points and TTFT p99 within a few percent at low/moderate concurrency; the
residual concentrates in deep-churn small-prompt corners where real mixed
passes cost up to 2x the perf-DB pricing — a timing-data gap, not an
estimator-structure one (refined TTFT errs on the conservative side
there).

Known error modes (measured, understood, not corrected for):

- saturation-knee amplification: within a few percent of the knee, a
  5-10% decode-step pricing error is leveraged into a 20-40% TTFT queue
  error (the same sensitivity that makes any rho->1 queueing formula
  fragile); away from the knee the same pricing error costs 1-3%. Read
  near-knee refined TTFT as a lower-bound estimate.
- sub-saturated tails: the deterministic recursion settles into a
  collision-free phase, so its p50/p99 understate the occasional
  prefill collisions a jittered system shows (means unaffected);
- deep-concurrency small-prompt agg: admission order (immediate vs
  bunched prefill waves) can shift TTFT conservatively upward while ITL
  and throughput still match.

Scope: fixed-shape closed loop with TRT-LLM scheduling semantics (fused
chunked passes; disagg decode-attach not budget-gated). Engines that
account a disagg KV attach against the decode prefill budget (vLLM-style)
serialize the tandem when isl reaches that budget — a real behavior this
estimator does not model. Open-loop arrivals, shape distributions,
prefix-cache dynamics and latency *distributions* (p50/p99 structure,
transients) belong to the full queueing-model evaluator work.

## Relation to SLA filtering

The pipeline is untouched: sweep, SLA-target filtering, pareto and picking
all still read the legacy `ttft`/`tpot` columns, so default results are
unchanged by this PR. For consumers who want the SLA verdict on the
values a fixed-concurrency benchmark would actually measure,
`filter_closed_loop_sla(df, ttft_ms=..., tpot_ms=...)` is an approximate
post-filter: it refines the rows and drops those whose refined values
exceed the targets (rows it cannot price are kept — it only ever tightens
the legacy verdict where it has evidence). Because the pipeline picks
each deployment's operating point before any post-processing runs,
post-filtering a picked table can return empty even when a
lower-concurrency point of the same deployment complies;
`pick_under_closed_loop_sla(df, ...)` closes that gap — feed it the full
per-operating-point summary (sweep with a loose TTFT target) and it
re-picks each deployment's best surviving row.

For full in-pipeline enforcement, `Task(refined_sla=True)` (default off)
makes the sweep itself compare the targets against the refined values:
agg candidate points gate on the solo chunked-prefill lower bound (the
static TTFT is NOT a bound — deep saturation moves cycle time into TPOT,
so gating on it would discard compliant points) and are then priced;
disagg drops the 1.8 prefill gate to the solo bound and keeps, per
decode-parallel category, the best rate-matched combination whose
refined values comply. Pricing is memoized on the operating-point
inputs. Two effects observed on an 8-GPU Qwen3-32B sweep
(isl4096/osl256, ttft 2000ms): points the fixed factor wrongly killed
come back (disagg: a 3-worker tp2 prefill pool, solo x1.8 = 2070ms >
target but refined 1145ms, +57% tokens/s/gpu over the legacy pick; agg:
deep-saturation rows with static TTFT above target but refined below,
+6% on the top pick), and rate-matched combinations whose refined TTFT
explodes under queueing are dropped. AFD frames pass through unpriced
(refined NaN, legacy verdict kept): their closed-loop mapping — the mini
tandem over `prefill_t_step`/`decode_t_step` — needs its own validation
pass first.

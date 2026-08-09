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

Against the full queueing-model evaluator on identical timing (Qwen3-32B,
H20, isl4096/osl256):

- agg estimator, exact timing callables: TTFT/TPOT/throughput within 0.2%
  over C=2..64 (bit-identical up to C=8);
- agg estimator fed only the recorded row scalars: within ~3% over the
  same sweep (the linear-in-tokens chunk reconstruction);
- disagg tandem estimator vs the disagg evaluator (1P1D/2P1D/2P2D, serial
  and batched prefill, shallow to deep queueing): bit-identical TTFT and
  throughput at all nine validation points.

Scope: fixed-shape closed loop. Open-loop arrivals, shape distributions,
prefix-cache dynamics and latency *distributions* (p50/p99 structure,
transients) belong to the full queueing-model evaluator work.

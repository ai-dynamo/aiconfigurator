# Forward Pass Metric Work-Local Modeling

This document summarizes a lightweight modeling direction for Forward Pass
Metric (FPM) latency prediction. The goal is to use a small set of measured
prefill or decode forward-pass shapes and predict nearby shapes with bounded
error, without introducing heavyweight ML models or module-level latency
decomposition.

## Scope

The model is forward-pass level. It treats the measured FPM latency as the
target and does not require attention/MoE/module split measurements. Kernel
or module-specific formulas are intentionally out of scope for the default
path.

The intended inputs are shape-level FPM fields:

- `batch_size = B`
- `new_tokens = S` for prefill
- `past_kv_tokens = P`

For decode, each sequence contributes one new token, so `S = 1`.

## Motivation

Direct shape-space interpolation can work when the training grid is dense, but
it treats `(B, S, P)` mostly as coordinates. That makes point compression and
long-prefix generalization harder, because actual forward-pass cost is driven
by simple work terms such as:

- request or batch overhead
- per-token work, roughly proportional to `B * S`
- prefix context work, roughly proportional to `B * S * P`
- prefill self-context work, roughly proportional to `B * S^2`

The proposed model first maps raw shapes into a small forward-pass work feature
space, then fits a local affine latency model around the query.

## Work Features

For prefill:

```text
N = B * S
W_prefix = B * S * P
W_self = B * S * (S + 1) / 2

T ~= a + b * B + c * N + d * W_prefix + e * W_self
```

`N` captures shape terms whose cost scales with scheduled tokens. `W_prefix`
and `W_self` capture context-dependent work without depending on any specific
attention kernel implementation.

For decode:

```text
N = B
W_context = B * (P + 1)

T ~= a + b * B + c * W_context
```

These equations are not expected to be globally accurate over the whole shape
space. They are used as local basis functions.

## Predictor

The recommended prediction order is:

1. Exact FPM database hit.
   Return the measured latency when the exact shape exists.

2. Work-local affine model.
   Normalize a small local key space, find nearby training points, and solve a
   small weighted ridge regression in the work feature space. This keeps the
   model simple and makes the fitted slopes local to the current regime.

3. Optional SOL efficiency cap.
   If a speed-of-light latency estimate is available, compute training
   efficiencies:

   ```text
   efficiency = T_sol / T_actual
   ```

   For extrapolation, efficiency should not grow beyond the observed saturated
   region. The latency prediction can be capped from below:

   ```text
   T_pred >= T_sol / efficiency_cap
   ```

   `efficiency_cap` can be a high quantile such as p95 or p99 of the training
   efficiencies in the corresponding regime. If no SOL provider is available,
   the model falls back to the work-local prediction.

4. Debug fallback.
   Shape-space IDW or nearest-neighbor interpolation may be kept as a baseline
   or fallback, but it is not the preferred main model.

## Why Not Global Affine

A single global affine fit is too coarse. Experiments showed that global
work-space regression can have reasonable average error in some models but is
not stable enough across regimes.

The robust version is local:

```text
shape -> work features -> local neighborhood -> local affine fit
```

The model remains cheap. For a few hundred to a few thousand training points,
query cost is dominated by a nearest-neighbor scan plus a tiny least-squares
solve.

## Prefill Sampling Candidate

A safe compressed prefill train grid used in experiments contains 773 points.
It is generated from:

```text
B in {1, 2, 4, 8, 32}

S in {
  1, 2, 4, 8, 16,
  64, 256, 1024, 4096, 16384, 65536
}
```

For each `(B, S)`, prefix candidates are generated from:

```text
0
absolute anchors: 1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 131072
ratio anchors: S, 3S, 7S, 15S, 31S
boundary anchor: 262144 - S
```

Then invalid shapes are filtered by:

```text
S + P <= 262144
B * S <= 262144
B * (S + P) <= 1048576
```

This plan intentionally keeps prefix coverage while removing many intermediate
batch and sequence anchors. More aggressive plans around 649 points also worked
in the current synthetic/static experiment, but the 773-point plan has a wider
tail-error margin.

## Experimental Summary

The current study reused static AIC latency as ground truth and compared the
work-local model with direct shape-space IDW on the same train/validation/test
splits. Tested prefill datasets included:

- Qwen3-235B
- Qwen3-32B
- DeepSeek-V3
- Llama3.1-8B

Across these datasets, work-local reduced test MAPE and tail error versus the
shape-space baseline. With full prefill train grids, test MAPE improved from
roughly `4.8%-6.2%` to roughly `2.2%-3.8%` depending on the model.

The compressed 773-point plan kept worst validation/test errors within the
target range in the current experiment:

```text
worst MAPE: 5.21%
worst P95: 12.18%
worst P99: 23.05%
worst Max: 29.63%
```

Pure collection latency, computed as the sum of training-point `latency_ms`, was
also reduced. Compared with previous compressed shape-space plans:

```text
new 773 vs old 2051: about 72%-73% collection-latency reduction
new 773 vs old 1809: about 68%-70% collection-latency reduction
new 773 vs old 1908 prefix-sparse plan: about 59%-60% reduction
```

These numbers exclude model load, collector startup, warmup, and repeated
measurements. Those overheads should be accounted for separately in a real FPM
collector, but the ratios are useful for comparing train shape plans.

## Open Items

- Validate against real Dynamo/vLLM FPM measurements instead of static AIC
  latency.
- Add an optional SOL provider and an extrapolation-only efficiency cap.
- Tune local-neighborhood selection by regime. Very short prefill and decode
  batch/KV boundary regions are the most sensitive.
- Extend the same work-local framework to mixed-shape batches by summing the
  corresponding work features across scheduled requests.
- Keep module-level decomposition out of the default model unless a separate
  collection path explicitly needs it.

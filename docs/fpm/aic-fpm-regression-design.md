# Worker-type-bound FPM regression design

> Status: implementation contract for the next minor `aiconfigurator-core`
> release. This document describes the online regression over
> `ForwardPassMetrics` telemetry. It is distinct from the separately collected
> whole-model `fpm_forward_perf` database.

## Goal and boundary

The regression backend binds each `ForwardPassPerfModel` instance to one
immutable engine role: `prefill`, `decode`, or `aggregated`. Every attention-DP
rank in an engine iteration therefore uses the same feature rule. The model
owns one two-dimensional dynamic sample store and one fit; it does not infer a
different regression kind from each rank's scheduled work.

This change is regression-only. Native AIC continues to infer prefill, decode,
or mixed work per iteration, use its existing representative-rank features,
and maintain its existing three fixed-grid correction stores. Native feature
coordinates, correction bounds, ratios, readiness, and prediction behavior do
not use the logarithm, standardization, or regression weights described here.

## Symbols and features

For attention-DP rank \(d\), define:

| Symbol | `ForwardPassMetrics` quantity | Meaning |
|---|---|---|
| \(P_d\) | `sum_prefill_tokens` | newly computed Prefill tokens |
| \(H_d\) | `sum_prefill_kv_tokens` | previously cached Prefill KV tokens |
| \(N_d\) | `num_prefill_requests` | scheduled Prefill requests |
| \(B_d\) | `num_decode_requests` | scheduled Decode requests |
| \(K_d\) | `sum_decode_kv_tokens` | Decode KV tokens read |

The telemetry schema permits fully cached metadata with \(P_d=0\) and
\(H_d>0\). Cached Prefill KV creates attention work only when fresh Prefill
work exists:

\[
\widetilde H_d=H_d\,\mathbf 1[P_d>0].
\]

Per-request token lengths are not available, so Prefill attention pairs use a
balanced-request approximation:

\[
Q_d=
\begin{cases}
\dfrac{H_dP_d}{N_d}+\dfrac{P_d^2}{2N_d}+\dfrac{P_d}{2},
  &P_d>0,\ N_d>0,\\[6pt]
0, &P_d=0.
\end{cases}
\]

All roles share the axis order

\[
x=[\text{critical attention},\ \text{global FFN/MoE}].
\]

With \(\alpha\) the KV-attention weight, \(\beta\) the Prefill
attention-pair weight, and \(\gamma\) the tokenwise FFN/MoE weight, the exact
features are:

\[
x_P=
\left[
\max_d\left(\alpha\widetilde H_d+\beta Q_d\right),
\ \gamma\sum_d P_d
\right],
\]

\[
x_D=
\left[
\alpha\max_d K_d,
\ \gamma\sum_d B_d
\right],
\]

\[
x_A=
\left[
\max_d\left(\alpha(\widetilde H_d+K_d)+\beta Q_d\right),
\ \gamma\sum_d(P_d+B_d)
\right].
\]

The Aggregated maximum is taken after composing the entire rank-local
attention score; the implementation must not combine independent maxima from
different ranks. Counters are converted to `f64` before multiplication or
cross-rank summation, and derived features must be finite and nonnegative.

The options are construction-time knobs and all default to `1.0`:

| Formula | `ForwardPassPerfOptions` field |
|---|---|
| \(\alpha\) | `regression_attention_kv_weight` |
| \(\beta\) | `regression_prefill_attention_pair_weight` |
| \(\gamma\) | `regression_ffn_token_weight` |

They must be finite and strictly positive when regression is constructed. They
are ignored by a successful native-only model. Changing a weight changes the
feature and bucket space and therefore requires a new model or replay of the
original per-rank FPM observations.

The ergonomic Python facade accepts ordinary Python floats and marshals the
three weight fields on a shallow copy of the caller's options dictionary.
Finite values remain JSON numbers; nonfinite values use the exact valid-JSON
string sentinels `"NaN"`, `"Infinity"`, and `"-Infinity"`. JSON-oriented raw
PyO3 callers may send those same sentinels. Rust deserialization maps only
those exact strings back to their corresponding `f64` values; unknown strings
and other value types are invalid. This transport does not relax validation:
`from_native` and a successful native `best_available` ignore all three
weights, whereas `from_regression` and a fallback `best_available` reject a
decoded nonfinite value with the corresponding field-specific error.

## Role and observation contract

`ForwardPassWorkerType::{Prefill, Decode, Aggregated}` is the Rust API. Python
uses exactly `"prefill"`, `"decode"`, and `"aggregated"`; aliases such as
`"agg"` are rejected.

- A Prefill model rejects an iteration containing any scheduled Decode
  request.
- A Decode model rejects an iteration containing any fresh Prefill token.
  \(P_d=0,H_d>0\) is cached metadata, not a role mismatch.
- An Aggregated model accepts Prefill-only, Decode-only, mixed-rank, and
  phase-separated-rank iterations.
- An iteration is idle when \(P_d=B_d=0\) for every rank, even if \(H_d>0\).
  Estimation returns zero and tuning does not retain it.

The latency target remains the maximum finite, positive rank `wall_time`,
converted to milliseconds. Callers must supply a complete, unique, and
iteration-aligned attention-DP rank set. Expected DP size is deliberately not
added to the core constructor; Dynamo remains responsible for rank-count and
duplicate-rank validation.

The constructors become:

```text
Rust:   from_regression(worker_type, options)
        best_available(config, worker_type, options)
        best_available_with_roots(config, worker_type, options, systems_root)
Python: from_regression(worker_type, options=None)
        best_available(config, worker_type, options=None)
```

`from_native` and `from_native_with_roots` remain role-free. Requiring
`worker_type` in the regression-only and fallback-capable constructors is a
documented next-minor API break. Adding the three public regression-weight
fields is also a Rust source break for exhaustive `ForwardPassPerfOptions`
struct literals; downstream callers should use
`ForwardPassPerfOptions { bucket_count: 8, ..Default::default() }`. The raw
PyO3 constructors use the same required string argument as the ergonomic
Python facade. This feature change does not itself bump package versions or
any FPM, `EngineConfig`, or `EngineSpec` schema version.

## Retention and fit pipeline

Each accepted observation keeps the raw two-dimensional feature vector and
observed milliseconds. Retention uses separate bucket coordinates:

\[
b_j=\log(1+x_j).
\]

The existing dynamic two-dimensional grid consumes these continuous `f64`
coordinates unchanged. Its bounds expand and trigger rebucketing, never
shrink, and its fattest-cell eviction policy still enforces the global sample
cap. With the default `bucket_count=16`, the grid is \(4\times4\). Buckets
choose which observations survive; they are not local predictors and are not
queried during estimation.

After insertion and eviction, the fit is rebuilt from the retained **raw**
features. For each axis, population mean and standard deviation are computed
with stable Welford accumulation:

\[
\mu_j=\frac1n\sum_i x_{ij},\qquad
\sigma_j=\sqrt{\frac1n\sum_i(x_{ij}-\mu_j)^2},\qquad
z_{ij}=\frac{x_{ij}-\mu_j}{\sigma_j}.
\]

An axis is inactive when

\[
\sigma_j\le 10^{-12}\max(1,|\mu_j|).
\]

Its standardized value and coefficient are zero. If both axes are inactive,
the regression remains unready. Otherwise, the existing nonnegative
active-set linear regression is fitted on standardized features with a free
intercept and its existing slope-only regularized fallback. A fit stores its
own means, scales, active-axis flags, standardized coefficients, and intercept;
prediction transforms raw features with that same snapshot. Inputs must be
exactly two finite, nonnegative values. Extrapolation and the positive nonidle
prediction floor remain supported.

The standard defaults remain `bucket_count=16`, `min_observations=5`, and
`max_observations=64`. The observation cap applies once to the role-bound model,
not once to each inferred workload kind. Diagnostics retain their existing
wire shape, but readiness and retained count now describe that one regression
store.

## Dynamo integration and deferred work

After a matching next-minor core release, Dynamo should pass its existing
engine `worker_type` through both `from_regression` and `best_available`, expose
the three weights as positive finite Planner configuration knobs with unity
defaults, and include them in the engine-model reconstruction key. Weight
changes require reconstruction followed by replay of Dynamo's retained raw FPM
iterations. The role remains a constructor argument rather than an options-map
entry, and Dynamo must update its exact core dependency pins together. This
worktree changes only AIC: forwarding the worker role from Dynamo remains
pending downstream integration.

The following are intentionally documented for later work rather than added to
this implementation:

- calibrating or reference-scaling \(\alpha,\beta,\gamma\);
- retaining the original per-rank raw FPM observations in the core alongside
  the derived two-dimensional samples, for replay and a future
  Gaussian-process feature space;
- estimating prefix reuse for synthetic or queued Prefill requests, which
  currently behave as cold-prefix inputs with \(H=0\);
- topology-aware FFN/MoE work and explicit speculative-token accounting;
- freshness or decay, robust shrinking bucket bounds, and deterministic
  eviction ties;
- optional core enforcement of complete expected DP-rank membership.

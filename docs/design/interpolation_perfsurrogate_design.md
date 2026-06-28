# Sparse Performance Surrogate Design

Status: implemented in draft PR #1255. The Python integration and structural
tests are complete; predictive-accuracy validation is not yet accepted.

## Decision summary

Performance data is modeled as a set of real, potentially ragged samples rather
than as a complete Cartesian grid. A query declares one semantically meaningful
`varying` axis, such as GEMM `m`, attention sequence length, or MoE token count.
The estimator first evaluates real one-dimensional curves along that axis and
then combines nearby curves with a line or triangle in the remaining axes.

This is a deliberate middle ground:

- it preserves operation knowledge about which dimension should be evaluated
  first;
- it fills holes without synthesizing a dense grid;
- it supports physically anchored extrapolation through an operation-provided
  baseline; and
- it keeps one small numerical implementation shared by many operations.

The design is not an arbitrary-dimensional or learned surrogate. It supports
one varying axis plus at most two other continuous axes. Categorical routing,
discontinuous regimes, and operation-specific physical formulas remain in the
operation adapter.

The curve-plus-simplex decomposition structurally accommodates the dimensionality
of the measured tables in scope. Predictive suitability is not yet established.
The main unresolved risks are the absence of a support-distance limit and the
lack of data-backed acceptance results for interpolation and extrapolation. The
implementation should not be expanded to more operation families until those
risks are evaluated.

## Problem

AIConfigurator performance tables are collected at selected workload shapes.
They are not guaranteed to contain every point in a Cartesian product:

- expensive or invalid shape combinations may be omitted;
- different fixed configurations may contain different sequence or token
  samples;
- collectors may stop at different frontiers; and
- categorical regimes such as backend, kernel, quantization, or tensor
  parallelism must never be interpolated together.

The legacy approach often expected a dense nested grid or filled missing values
axis by axis. That creates three problems:

1. generated points are treated like measurements by later interpolation;
2. the result can depend on the order in which axes are filled; and
3. a Cartesian product cannot represent a genuinely ragged frontier cleanly.

At the same time, a completely generic scattered interpolator would discard
useful prior knowledge. For example, GEMM should form performance curves along
`m` before combining neighboring `(n, k)` configurations. The common layer must
therefore share numerical mechanics without erasing operation semantics.

## Goals

- Query real ragged samples without load-time Cartesian expansion.
- Return an exact measurement unchanged when the core receives an exact point.
- Preserve one operation-declared semantic varying axis.
- Interpolate between neighboring real curves in up to two fixed axes.
- Authorize exterior estimates per axis and per direction.
- Allow the operation to provide a physical or empirical baseline for exterior
  scaling.
- Keep latency and energy internally consistent by interpolating average power
  for modeled points.
- Compile and cache table geometry once per unique query key, axis declaration,
  varying axis, and table identity.
- Keep operation adapters small enough that future measured operations can use
  the same query path.

## Non-goals

- Arbitrary-dimensional interpolation.
- Inferring categorical compatibility from numeric proximity.
- Replacing analytical or formula-only operation paths.
- Defining a universal SOL formula or a universal latency floor.
- Providing uncertainty, confidence, provenance, or support-distance metadata.
- Training or serving an ML model.
- Rust parity in this change.
- Changing collector schemas except where a loader must preserve dimensions
  already present in collected data.

## Public API and adapter contract

The reusable surface in `sdk/perf_surrogate.py` consists of `Axis` and
`estimate_sparse`:

```python
latency, energy = estimate_sparse(
    database,
    key=("gemm", table_quant_mode, quant_mode),
    table=gemm_data,
    query={"m": m, "n": n, "k": k},
    axes=(
        Axis("m", extrapolate="both"),
        Axis("n", log=True, extrapolate="both"),
        Axis("k", log=True, extrapolate="both"),
    ),
    varying="m",
    curve="raw",
    mesh="baseline_ratio",
    exterior="baseline_ratio",
    baseline=lambda point: get_sol(point["m"], point["n"], point["k"], quant_mode)[0],
)
```

An adapter must provide:

- numeric axes in exactly the same order as the nested table;
- one unique `varying` axis;
- optional `log=True` coordinate transforms;
- one of `never`, `lower`, `upper`, or `both` as each axis's exterior policy;
- `raw`, `sqrt`, or `baseline_ratio` responses for curve, mesh, and exterior
  blending;
- a finite positive baseline whenever `baseline_ratio` is selected; and
- a cache key that uniquely identifies the already-selected categorical table.

Before calling the common estimator, the adapter must select exact categories
such as backend, kernel source, dtype, quantization mode, phase, TP/EP/CP size,
head layout, or compression ratio. A discontinuity such as DSA's top-k regime
must also be split before interpolation.

After the call, operation-specific behavior remains outside the estimator. This
includes prefix correction, decode smoothing, top-k deltas, topology scaling,
and composite-operation assembly. The common estimator does not attempt to
understand these semantics.

## Data contract

The table is a nested mapping in axis order:

```text
table[axis_0][axis_1]...[axis_n] -> metric leaf
```

A leaf may be a latency scalar or a mapping with `latency` and optional
`energy` or `power`. If only power is present, the core derives
`energy = power * latency`; explicit energy takes precedence when both fields
exist. Scalar leaves and mappings without either energy or power default to
zero energy. Coordinates and metrics must be finite; log axes must be positive;
latency and energy must be non-negative.

The estimator flattens and sorts every leaf supplied by the adapter. It has no
provenance field that can distinguish a measurement from a previously generated
value. Adapters must therefore pass a measurement-only table or view. The
estimator itself never inserts a synthetic sample into the source table.

## Mathematical decomposition

Write a query as `q = (z, t)`, where `t` is the semantic varying axis and `z`
contains zero, one, or two fixed axes. In the support equations, coordinates
mean their declared encoded values, such as `log2(n)` rather than raw `n`.
Samples sharing the same raw fixed key form a real curve `C_z(t)`.

For an interior fixed-axis query, the line or triangle support provides weights
such that:

```text
z = sum_i w_i z_i
sum_i w_i = 1
w_i >= 0
```

Each supporting curve is first evaluated at `t`. Those component estimates are
then combined with the same fixed-axis weights. This preserves the semantic
axis ordering without an `X`, then `Y`, then `Z` chain and without requiring a
rectangular grid.

The three response functions are:

```text
raw:
    L_hat(q) = sum_i w_i L_i

sqrt:
    L_hat(q) = (sum_i w_i sqrt(L_i))^2

baseline_ratio:
    L_hat(q) = B(q) * sum_i w_i * L_i / B(x_i)
```

`B` is supplied by the operation. It may be an analytical SOL, a measured
boundary model, or a simpler workload proxy when that is the only justified
policy. `baseline_ratio` is an arithmetic ratio interpolation; it is not a log
residual model and does not by itself guarantee `L_hat >= B`.

For modeled points, energy is reconstructed from interpolated average power:

```text
P_i = E_i / L_i                  # zero when L_i is zero
P_hat = sum_i w_i P_i
E_hat = P_hat * L_hat
```

An exact core hit returns the normalized exact latency and energy directly; a
power-only leaf is normalized to energy while the table is flattened. An
operation may still apply a correction after the core returns, so any such
correction must separately preserve the desired power/energy semantics.

## Query state machine

After a cache hit or model compilation, the current implementation resolves a
query in this order:

1. Validate that query names exactly match the declared axes.
2. Return an exact measured point when present.
3. If the fixed key exactly matches a real curve, bracket the varying axis on
   that curve.
4. Otherwise locate the fixed key inside a real line segment or Delaunay
   triangle and evaluate the supporting curves.
5. If the fixed key is outside the convex hull, choose an allowed boundary
   candidate from a one-dimensional line segment/endpoint or a two-dimensional
   hull edge.
6. If the varying coordinate lies outside a supporting curve, use its permitted
   endpoint.
7. Use the requested curve or mesh response for an interior result. If either
   the fixed support or a component curve is exterior, use the exterior
   response.
8. Reject non-finite or negative modeled metrics.

If no authorized support exists, the core raises
`InterpolationDataNotAvailableError`. It does not silently fall back to a
global nearest neighbor.

## Geometry

The fixed-key mesh supports at most two axes:

- zero active fixed axes: use the only curve;
- one active fixed axis: sort curve keys and perform a line bracket; and
- two active fixed axes: standardize coordinates and build a SciPy Delaunay
  triangulation when at least three non-collinear points exist.

Fixed axes whose encoded span is within numerical tolerance are treated as
inactive. A query must match an inactive axis within tolerance unless that axis
explicitly authorizes the requested exterior direction. `lower` and `upper`
describe the query relative to the selected boundary candidate, not merely its
position relative to the table's global minimum or maximum.

For a two-dimensional exterior query, the implementation scans convex-hull
edges. Candidate boundary points include both endpoints, the orthogonal
projection onto an edge, and intersections needed to keep a `never` axis equal
to the query. It uses the closest admissible point from this finite candidate
set. For `lower` and `upper` constraints it filters candidates but does not add
the constraint-boundary/edge intersections, so this is an approximate
projection and is not guaranteed to find the globally nearest feasible hull
point. There is no SLSQP optimization, k-nearest-neighbor fallback, or learned
boundary model.

This design has no maximum interpolation diameter or extrapolation distance.
An axis marked `both` can therefore authorize a query far beyond the measured
frontier. Whether to add a small trust-region policy is an open decision that
must be driven by holdout results rather than an arbitrary constant.

## Operation integration

| Family | Numeric model | Adapter behavior retained outside the core |
| --- | --- | --- |
| GEMM | `m` curves over log2 `(n, k)`; raw curve and baseline-ratio mesh/exterior | Exact fast path, load/query SOL correction, quantization routing, `m = 0` behavior |
| GEMM scale overheads | `m` curves over log2 `k` | Separate compute-scale and scale-matrix tables and SOL baselines |
| Context/encoder attention | sequence curves over heads and batch; sqrt curve response | Prefix correction, FMHA/KV categories, RoPE and memory additions |
| Generation attention | sequence curves over heads and batch; raw curve response | Five-point sequence smoothing and SOL correction |
| Communication | one-dimensional message-size curves | GPU-count selection, cross-node topology/bandwidth scaling; P2P remains separate |
| Mamba/GDN | context sequence curves over batch; generation batch curves | Existing same-`d_model` structural fallback, GDN head-distance selection and kernel alias, then SOL fallback on a miss |
| MLA | token curves over heads and batch; MLABmm uses a one-dimensional token curve | Backend/kernel routing, TP-to-head mapping, prefix correction, composite module formulas |
| MoE | token curves; selected DeepEP tables also include SM count | Exact categorical routing, low-token launch policy, high-token utilization policy, dispatch/combine composition |
| DSA | context `s` and generation `s_total` curves over heads and batch | Pre/post-top-k table split, outer prefix handling, backend selection, CP assembly |
| DeepSeek-V4 attention | context sequence curves over prefix and batch; generation sequence curves over batch | Exact head/TP/compression/quantization selection and CSA top-k correction |
| DeepSeek-V4 MHC/MegaMoE | one-dimensional token curves | Exact module/category selection; MHC uses its SOL baseline, while MegaMoE holds below its first token sample and scales proportionally above its frontier |

These are the targeted measured-table paths, not every operation class in the
repository. Intentionally unchanged paths include Rust, P2P, and formula-only
operations. DSA context prefix blending and CP composition also remain layered
adapters rather than one monolithic surrogate call.

All 24 concrete Python operation classes that own measured performance tables
now route their main SILICON numeric lookup through `estimate_sparse`. Measured
correction surfaces such as the DSV4 top-k delta remain adapter logic rather
than independent operation classes.

## Cache and mutation contract

Compiled geometry is stored on the database instance in
`_sparse_surrogate_cache`. The logical key contains the caller key, axis
declaration, and varying axis. The cache entry also retains the source table
identity; replacing the table object rebuilds the model even if the logical key
is reused.

The flattened metrics are a snapshot. Mutating the same table object in place
does not automatically invalidate that snapshot. Tests or SDK callers that
intentionally mutate a table must call `PerfDatabase.clear_runtime_caches()`.
Normal loaded performance tables are treated as immutable.

## Complexity

Let `N` be the number of measured leaves, `C` the number of real curves, and
`H` the number of two-dimensional convex-hull edges.

- The first query for a unique cache entry flattens and sorts `N` samples,
  groups them into `C` curves, and builds at most a two-dimensional
  triangulation. A cold exact query therefore pays this compilation cost.
- A warm exact query is a dictionary lookup.
- A curve query currently rebuilds and encodes that curve's `K` varying-axis
  coordinates before applying a binary bracket, so it is `O(K)` rather than a
  pure `O(log K)` lookup.
- An interior mesh query evaluates at most three curves and performs a small
  weighted blend.
- A two-dimensional exterior query additionally scans `H` hull edges.

There is no training, model weight loading, accelerator execution, or large
matrix inference. The work is a small deterministic CPU geometry query and does
not carry the training or serving machinery of an ML surrogate. Compilation is
cached, but query cost should still be benchmarked if this path becomes a
configuration-search hot spot.

## Failure semantics

The core distinguishes unsupported numerical support from invalid
configuration:

- unavailable authorized support raises `InterpolationDataNotAvailableError`;
- malformed axes, tables, queries, responses, or baselines encountered by the
  evaluated path raise `TypeError`/`ValueError`; and
- valid estimates must have finite, non-negative latency and energy.

Each operation retains its existing SILICON/HYBRID policy around the call. The
common layer does not reinterpret a missing category as a nearby numeric point.

## Known limitations and risks

1. **At most three continuous axes.** There is one varying axis and at most two
   fixed axes. A higher-dimensional table must first be partitioned by the
   operation or use a different backend.
2. **No trust region.** Large holes and distant authorized exterior queries are
   not rejected by distance.
3. **No uncertainty or provenance.** Callers receive only latency and energy;
   they cannot distinguish exact, interior, or exterior results from the return
   type alone.
4. **No global physical floor.** Baseline quality and any final SOL clamp belong
   to the adapter. A baseline is not necessarily a lower bound.
5. **No monotonicity constraint.** Measured kernel teeth and simplex layout can
   produce locally non-monotone estimates.
6. **Rank-deficient correlated fixed axes.** The engine does not infer an
   arbitrary lower-dimensional manifold when two nominally active axes are
   collinear.
7. **Delaunay topology.** Sparse layouts can admit more than one reasonable
   diagonal; the selected local linear surface is not a tensor-product surface.
8. **Directional hull projection is approximate.** The finite candidate set
   can miss the true nearest feasible point where a one-way `lower` or `upper`
   constraint intersects a hull edge.
9. **Fixed support is selected without varying-axis coverage.** After choosing
   a line or triangle in fixed space, the engine does not search an alternate
   support if one selected curve cannot serve the varying coordinate. It can
   also extrapolate a selected curve even when a different nearby support might
   interpolate that coordinate.
10. **Some declaration validation is lazy.** Exact hits bypass response and
    baseline evaluation, and invalid exterior-policy strings are not rejected
    eagerly. A bad declaration may therefore surface only on a modeled path.
11. **Adapter postprocessing can break energy consistency.** Any latency clamp
    after estimation must rescale energy if average power is intended to remain
    constant. The current rollout audit identified GEMM and
    generation-attention SOL corrections, plus DSA's independent prefix
    interpolation of latency, power, and energy, as paths requiring review.
12. **Prediction quality is not yet established.** Structural correctness and
    finite outputs do not prove that held-out measurements are estimated well.

## Future four-dimensional attention

Future attention tables may retain prefix as an independent numeric dimension,
for example:

```text
prefix x heads x sequence x batch -> metrics
```

With `sequence` as the semantic varying axis, this leaves three fixed axes and
therefore exceeds the current mesh limit. This change deliberately does not
generalize `_FixedMesh` to three dimensions before representative four-
dimensional data and holdouts exist.

Collectors and loaders for this shape must retain prefix as a real numeric axis
rather than pre-filling prefix combinations or collapsing them into synthetic
samples. Existing full-sequence reductions may remain operation-specific only
where the measured data demonstrates that prefix has no independent effect.

The preferred first extension is an explicit sliced axis rather than an
unconditional three-dimensional Delaunay mesh:

1. Select categorical regimes before numerical interpolation as today.
2. Partition the table into exact prefix slices.
3. Evaluate each usable prefix slice with the existing
   `sequence + (heads, batch)` surrogate.
4. For an off-grid prefix, use only bracketing prefix slices that can both
   answer the complete inner query.
5. Blend latency in the declared response space and blend average power before
   reconstructing energy. Do not independently interpolate latency, power, and
   energy.
6. Apply baseline-ratio exterior behavior on prefix only when the adapter
   explicitly authorizes it.
7. Split any regime defined by `prefix + sequence`, such as a top-k boundary,
   before selecting prefix support.

The future API should declare the sliced axis explicitly, for example through
an `outer="prefix"` policy, rather than infer it from axis order. Exact API
naming is deferred until the first caller is implemented.

A full tetrahedral mesh over `(prefix, heads, batch)` remains an alternative,
but it would add three-dimensional Delaunay memory, sliver-tetrahedron behavior,
triangular hull-facet projection, and more complex directional constraints. It
should be adopted only if holdout data shows a material accuracy advantage over
the sliced design.

Four-dimensional validation must remove complete prefix slices as well as
individual sequence points, exercise prefix/sequence regime boundaries, and
report error versus prefix distance. Until that validation exists, the current
DSA outer-prefix adapter is a compatibility layer, not the final generic 4-D
implementation.

## Alternatives considered

### Dense Cartesian expansion

Rejected as the default because it invents intermediate samples, makes later
queries depend on generated values, increases load cost, and cannot represent a
ragged frontier without arbitrary fill rules.

### Per-operation hierarchical interpolation

The semantic varying-axis idea is retained, but duplicating the numerical state
machine in every operation makes bug fixes inconsistent. The fixed-axis
line/simplex removes the need for a bespoke `X -> Y -> Z` implementation while
preserving the operation's first-axis prior.

### Global three-dimensional scattered interpolation

Not selected because it discards the repeated-curve structure, uses more
geometry than needed, and gives no natural priority to the physically meaningful
axis.

### kNN, RBF, Gaussian process, or learned model

These remain possible research backends, but they add neighbor policy,
hyperparameters, training or fitting cost, and difficult exterior behavior. A
small deterministic estimator with an explicit operation baseline is easier to
review and deploy. A learned model should be introduced only after a measured
accuracy advantage justifies a second backend and its interface.

## Verification status

Local validation of the current implementation has:

- focused surrogate and migrated-operation tests: 291 passed;
- the complete unit selection: 1,524 passed and 39 skipped, with only the four
  documented non-TTY `test_plain_output.py` failures;
- the complete real-database sanity suite: 27 passed and 1 expected failure;
- passing Ruff, formatting, and `git diff --check`; and
- real-table smoke coverage for the DeepSeek-V4 head/TP loader variants.

The pushed head is DCO-signed; hosted checks rerun for each updated head and
remain separate from these local results.

The end-to-end accuracy job is not an acceptance signal yet. Its GitHub job is
green because the regression step is allowed to fail, but the regression test
for PR #1255 reported `1 failed, 1 passed`:

| Metric | Base | PR | Change |
| --- | ---: | ---: | ---: |
| TTFT MAPE | 55.2365% | 55.3236% | +0.0871 percentage points |
| TPOT MAPE | 37.7661% | 39.1102% | +1.3441 percentage points |

Four Qwen threshold checks exceeded the current 10-point limit: one model-level
aggregate and three configuration slices. In addition, Qwen3.5-397B-A17B
regressed by about 6.9 percentage points across 300 samples. The suite added no
newly predictable end-to-end samples, so it does not demonstrate hole-filling
coverage.

These results do not identify whether the error comes from GEMM, attention,
MoE, or a downstream composition. They show that implementation tests pass but
the numerical policy has not yet earned a quality sign-off.

## Required validation before merge

Build a deterministic holdout study from raw performance tables and report each
operation family separately:

1. Remove interior points from existing semantic curves.
2. Remove complete interior fixed-key curves to create genuine ragged holes.
3. Hold out a measured boundary layer and query it as exterior support.
4. Compare the new estimator with the legacy path, nearest sample, and the pure
   baseline.
5. Report coverage plus median, P95, and maximum absolute percentage error.
6. Stratify error by support diameter and extrapolation distance, and include
   ragged cases where candidate fixed supports have different varying-axis
   coverage.
7. Verify finite/non-negative output, exact-point preservation, boundary
   continuity, and category isolation.
8. Decompose the Qwen regressions by operation and shape.

The existing end-to-end threshold test must pass without relying on
`continue-on-error`. If holdout error grows materially with support distance,
add the smallest data-backed trust-region rule and return a structured miss
beyond it. Do not add a distance cap merely to silence individual examples.

## Rollout

1. **Current draft:** keep the common core and migrated Python adapters in one
   reviewable PR.
2. **Accuracy hardening:** run the holdout study, localize the Qwen regressions,
   and fix adapter-level energy handling.
3. **Policy decision:** retain the simple design if the data supports it;
   otherwise add a minimal trust-region or revise only the failing operation's
   response/baseline policy.
4. **Ready for review:** rerun all functional and accuracy gates, then move the
   PR out of Draft.
5. **Later expansion:** use this path by default for new measured Python
   operations only after the acceptance gates pass. Rust parity and richer
   estimate metadata remain separate decisions.

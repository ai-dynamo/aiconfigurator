# DSV4 Sparse Work Interpolation: Mathematical Notes

This document describes the mathematical basis of the sparse attention latency
model used for DSV4-style MQA logits, FlashMLA/HCA, CSA, and the planned DSA
extension.

In short:

```text
This is not a new numerical method.

It is an engineering composition of:
  empirical roofline modeling
  effective-work feature construction
  local affine interpolation
  prefix-delta decomposition
  dense sampling around discontinuities
```

Good names for the approach are:

```text
work-space local affine interpolation
empirical roofline interpolation
sparse-work latency predictor
```

It should not be described as a new interpolation theory. The useful part is
the choice of sparse-attention work variables and the way those variables are
integrated into the empirical database query path.

## Problem Definition

The raw query shape is:

```text
q = (b, isl, prefix)
```

where:

- `b` is the batch size.
- `isl` is the number of newly computed input tokens.
- `prefix` is the number of cached past-KV tokens.

The direct latency function is:

```text
T = f(b, isl, prefix)
```

Plain 3D interpolation in `(b, isl, prefix)` is fragile because `prefix` does
not change runtime linearly. The actual kernel work depends on compression
ratio, window size, page size, top-k caps, and block scheduling. The same
change in `prefix` can represent very different amounts of real work in
different regions.

The first step is to map the raw query into a more physical work space:

```text
M = b * isl
W = effective sparse work
kbar = W / max(M, 1)
```

The model then predicts:

```text
T ~= g(M, W)
```

or, for modules with multiple prefix-dependent paths:

```text
T ~= g(M, W_1, W_2, ...)
```

This is a feature transformation. It moves interpolation from raw shape space
to work space, where latency is closer to locally linear.

## Effective Work

Effective work is not exact FLOPs and not exact memory bytes. It is an
equivalent work quantity that better explains a specific kernel's latency.

Define:

```text
G_d(n) = sum_{i=0}^{n} floor(i / d)
```

### MQA Logits

MQA logits computes sparse logits without softmax. For DSV4-style compression
with ratio 4:

```text
W_mqa(b, isl, prefix)
  = b * (G_4(prefix + isl) - G_4(prefix))
```

This represents the accumulated compressed-KV length seen by the new tokens.

### FlashMLA / HCA

HCA/FlashMLA-style sparse attention can be approximated as a local-window term
plus a compressed-cache term:

```text
W_hca(b, isl, prefix)
  = b * sum_{t=1}^{isl} [
      min(prefix + t, 128)
      + floor((prefix + t) / 128)
    ]
```

The constant `128` is model/kernel specific and should be checked when the
backend or implementation changes.

Intuitively:

- At low prefix, the local window has not saturated and work grows quickly.
- At long prefix, the local window is saturated and most incremental work comes
  from compressed cache.
- This explains why long-prefix agentic inference can leave the initial
  compute-bound region.

### CSA

CSA prefix growth does not only come from MQA logits. The top-k/indexed path
also changes until it saturates. A useful decomposition is:

```text
W_mqa(b, isl, prefix)
  = b * (G_4(prefix + isl) - G_4(prefix))

W_topk(b, isl, prefix)
  = b * sum_{t=1}^{isl} min(floor((prefix + t) / 4), topk)
```

For current DSV4 CSA, `topk` is typically 1024.

The CSA module latency can then be modeled as:

```text
T_csa ~= a
       + beta_M    * M
       + beta_mqa  * W_mqa
       + beta_topk * W_topk
```

Using only `W_mqa` systematically under-predicts the CSA prefix delta. Adding
`W_topk` better matches the module-level behavior.

### DSA

DSA should fit the same structure, but it currently needs prefix-aware
collection data. Existing prefix-0 prefill data ties `isl` and `full_s`
together, so it cannot separate:

```text
current-token work
past-KV delta work
```

A candidate decomposition is:

```text
M = b * isl
full_s = isl + prefix

W_attn(b, isl, prefix)
  = b * sum_{t=1}^{isl} min(prefix + t, topk)

W_mqa(b, isl, prefix)
  = 0,                  if full_s <= topk
  = M * full_s,         otherwise
```

For DSA, `topk` is typically 2048. This formula should be validated with
prefix-aware collection.

## Roofline View

The classical roofline lower bound is:

```text
T >= max(FLOPs / P_peak, Bytes / BW_peak) + overhead
```

Equivalently:

```text
T >= max(C_compute * FLOPs, C_mem * Bytes) + overhead
```

This predictor does not use theoretical peak throughput directly. Instead, it
uses collector data to fit empirical coefficients:

```text
T ~= a + beta_M * M + beta_W * W
```

or with multiple work terms:

```text
T ~= a
   + beta_M * M
   + beta_1 * W_1
   + beta_2 * W_2
   + ...
```

The interpretation is:

- `W_j` acts as effective compute work or effective memory work.
- `beta_j` is the measured cost per unit work.
- `a` absorbs launch overhead, fixed module overhead, and backend constants.
- `M` captures current-token-dependent cost not explained by prefix work alone.

If a region is cleanly compute-bound or memory-bound, a max model is natural:

```text
T ~= max(beta_compute * W_compute,
         beta_mem     * W_mem) + a
```

In practice, module measurements often mix several kernels, fixed overheads,
and backend details. A small affine model is usually more robust.

## Local Affine Interpolation

Away from scheduling discontinuities, latency is assumed to be locally smooth
in work space.

For a query point:

```text
x_q = [M_q, W_q]
```

or:

```text
x_q = [M_q, W_1q, W_2q, ...]
```

nearby samples are:

```text
(x_i, T_i)
```

A first-order local model is:

```text
T_i ~= theta_0 + beta^T (x_i - x_q)
```

The least-squares problem is:

```text
min_{theta_0, beta}
  sum_i weight_i * (T_i - theta_0 - beta^T (x_i - x_q))^2
```

The prediction is:

```text
T_pred(x_q) = theta_0
```

This is local affine regression. It is also equivalent to first-order LOESS.

The current implementation uses a cheaper cell-based variant:

```text
1. Build power-of-two buckets over M and kbar = W / M.
2. Fit a small affine model in each bucket when enough samples exist.
3. Fall back to a global affine model when the bucket is sparse.
4. At query time, compute M, W, kbar, choose a bucket, and evaluate the fit.
```

The affine form is:

```text
T ~= c_0
   + c_1 * normalize(W)
   + c_2 * normalize(M)
```

Normalization only improves numerical conditioning. It does not change the
model class.

## Prefix-Delta Decomposition

For a full attention module, much of the latency is independent of `prefix`:
projection, reshape, fixed launch overhead, and some memory movement.

A useful decomposition is:

```text
T_module(b, isl, prefix)
  ~= T_module(b, isl, 0)
    + Delta_sparse(b, isl, prefix)
```

where:

```text
Delta_sparse
  ~= beta_mqa  * (W_mqa(prefix)  - W_mqa(0))
   + beta_topk * (W_topk(prefix) - W_topk(0))
   + ...
```

This is similar to a control-variate or residual-correction model:

- `T_module(b, isl, 0)` captures opaque module base cost.
- Prefix-related work deltas explain the extra past-KV cost.
- Full-module prefix anchors can directly estimate module-level delta slopes.

For example, if a reference prefix `p_ref` is collected:

```text
beta_ref =
  (T_module(b, isl, p_ref) - T_module(b, isl, 0))
  /
  (W(b, isl, p_ref) - W(b, isl, 0))
```

Then:

```text
T_module(b, isl, prefix)
  ~= T_module(b, isl, 0)
    + beta_ref * (W(b, isl, prefix) - W(b, isl, 0))
```

With multiple work terms, `beta_ref` becomes a small multi-feature affine fit.

## Why Raw Latency Interpolation Is Weaker

Raw interpolation estimates:

```text
T = f(b, isl, prefix)
```

directly. The problem is that distance in `(b, isl, prefix)` does not represent
equal hardware work.

Two points may be close in raw shape space but far apart in real work,
especially near:

- compression boundaries,
- page or block boundaries,
- top-k saturation,
- small-M launch-overhead-dominated regions,
- backend tiling or scheduling changes.

Work-space interpolation estimates:

```text
T = g(M, W)
```

Here, movement in the input variables is more directly related to the work
performed by the kernel. This reduces curvature and makes a first-order affine
model useful over a larger region.

## Discontinuities

No smooth interpolation method can reconstruct an unsampled discontinuity.

GPU kernel latency often jumps because of:

- tile size changes,
- block scheduling changes,
- top-k thresholds,
- page-table or slot-count changes,
- launch overhead dominating small shapes,
- backend implementation changes.

The real function can be viewed as:

```text
T(x) = smooth_work_part(x) + jump_part(x)
```

Work-affine interpolation handles the smooth part. The jump part needs:

```text
dense sampling near the boundary
regime-specific buckets
nearest/raw fallback in small or unstable regions
```

Recommended policy:

```text
small M or known jump region:
  exact/raw/nearest or dense local interpolation

regular region:
  work-space affine interpolation
```

## Runtime Cost

The online predictor is intentionally cheap.

At database load time:

```text
1. Convert raw samples to (M, W, latency).
2. Build power-of-two bucket edges.
3. Fit global and local affine coefficients.
```

At query time:

```text
1. Compute M = b * isl.
2. Compute one or more closed-form work values W.
3. Compute kbar = W / M.
4. Choose a bucket.
5. Evaluate the affine formula.
```

The online complexity is:

```text
O(number_of_work_terms)
```

This is a few integer sums, bucket comparisons, and floating-point
multiply-adds. There is no neural network inference and no online optimizer.

## Backend Or Hardware Changes

If the kernel semantics are unchanged, changing backend or hardware usually
requires:

```text
same work formula
new raw samples
new affine coefficients
same query-time algorithm
```

If the backend changes the kernel algorithm, page size, top-k behavior, or
compression ratio, the effective-work formula must be reviewed.

## Recommended Query Policy

A practical empirical database query order is:

```text
1. Exact raw table hit:
     return measured latency

2. Small M or known unstable region:
     use raw nearest / dense local interpolation

3. Regular sparse region:
     compute effective work and use local affine fit

4. Module prefix query:
     predict prefix-0 base latency,
     then add work-based prefix delta
```

For current DSV4 sparse attention:

```text
MQA logits:
  use M + W_mqa

FlashMLA / HCA:
  use M + W_hca

CSA:
  use M + W_mqa + W_topk

DSA:
  likely use M + W_attn + W_mqa,
  but prefix-aware collection is still required
```

## Learning References

Recommended learning order:

```text
1. Roofline model
2. Least-squares linear regression
3. Local polynomial regression / LOESS
4. Piecewise affine interpolation
5. Empirical performance modeling
```

References:

- Williams, Waterman, Patterson, "Roofline: An Insightful Visual Performance
  Model for Multicore Architectures"
  https://cacm.acm.org/research/roofline-an-insightful-visual-performance-model-for-multicore-architectures/
- LBNL Roofline overview
  https://crd.lbl.gov/divisions/amcr/computer-science-amcr/par/research/roofline/
- "An Empirical Roofline Methodology for Quantitatively Assessing Performance
  Portability"
  https://www.osti.gov/biblio/1567493
- Cleveland and Devlin, "Locally Weighted Regression: An Approach to Regression
  Analysis by Local Fitting"
  https://sites.stat.washington.edu/courses/stat527/s14/readings/Cleveland_Delvin_JASA_1988.pdf
- Hastie, Tibshirani, Friedman, "The Elements of Statistical Learning"
  https://link.springer.com/book/10.1007/978-0-387-21606-5

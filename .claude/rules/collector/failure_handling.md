# Collector Failure Handling

Core doctrine: **observe, don't predict.** A failing case costs seconds and is
recorded automatically with a classification. A prediction rule costs
maintenance forever, rots when the framework fixes the bug, and depends on
fragile matching contracts. The system therefore has NO declarative
expected-failure layer; it has a failure log and a circuit breaker.

## What happens automatically (no action needed)

- Every worker failure is recorded in `errors_<module>.json` and
  `collection_summary_<backend>.json` with `classification: "unexpected"`,
  the case parameters, exception type/message, and its breaker group.
- After `BREAKER_THRESHOLD` (5) consecutive failures of one
  `(model, dtype)` group within an op, the breaker skips the rest of that
  group: skips land in the resume checkpoint as `skipped` and in the summary
  as one `classification: "breaker_skipped"` entry. Skipped work is never
  silently dropped and never looks collected.
- CUDA-fatal errors reset the worker process; the task is already recorded
  before the reset.
- Missing data points are tolerated downstream: the SDK interpolates,
  extrapolates, reuses sibling-version rows, and can fall back to HYBRID
  empirical estimates. A handful of failed cases does NOT invalidate a
  (system, backend, version) dataset.

## Decision tree for a failing case

```text
case failed
├─ Is it recorded & classified in the failure log?  → yes → DONE (default path)
├─ Does it HANG or kill the node?                   → denylist.yaml entry (dated + reason)
├─ Whole (op × backend) never debugged?             → registry OpEntry unverified=True
├─ Physically impossible on this hardware?          → capabilities.yaml positive floor
│    (dtype doesn't exist below SM X — NOT "this framework version lacks a kernel")
└─ Proven collector-code bug?                       → fix the code; never fix via skip;
                                                      re-check the dispatch/skip rule
```

Framework-version gaps (an rc that crashes on some shape, a backend that has
no SM120 recipe yet) deliberately have NO home in YAML. They fail, the breaker
contains them, the log explains them, and the next version bump re-tests them
for free.

## Triage thresholds (from collection-campaign experience)

- Isolated failures — even a few dozen across a large plan — are acceptable
  when explained and unclustered.
- Investigate around 10% unexpected failures, or earlier when failures cluster
  by op, backend, dtype, model, shape family, or SM.
- Roughly one third failing, or an entire family failing, is a systemic
  collector problem — stop collecting, fix the collector.
- Treat an OOM as unclassified until the same case fails on a clean GPU.

Never hide failures with broad skips, retries, generic OOM labels, reduced
coverage, synthetic rows, or weakened benchmarks. Record a failed result
before resetting a worker after a fatal CUDA error.

## Consuming failure records

- Compare against the previous run of the same (backend, version): only NEW
  failure groups deserve attention.
- Failure records are observations. Do not machine-translate them into
  denylist/capability entries — each escalation above requires the specific
  evidence named in the tree.

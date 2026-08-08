# Autoscale TTFT queueing pre-correction

The disagg autoscale/sweep paths filter prefill-worker candidates against
the TTFT SLA using the **static** per-worker `ttft` (one prompt, empty
queue). A worker serving a sustained arrival stream additionally queues
prompts, so the static value is multiplied by a pre-correction factor
before filtering. The default is `1.8` (unchanged); this document gives the
constant its provenance and a formula for other operating points.

## The formula

For a prefill worker at utilization `rho`, the G/G/1 mean waiting time
(Kingman approximation) gives the expected TTFT as a multiple of the solo
prefill time:

```text
factor(rho, ca^2, cs^2) = 1 + rho * (ca^2 + cs^2) / (2 * (1 - rho))
```

`cs^2` is the squared coefficient of variation of the prefill service time
(0 for a fixed request shape). `ca^2` is that of the **arrivals the worker
actually sees** — the load-bearing input:

- `ca^2 = 1` (Poisson): a single worker fed by open traffic. At
  `rho = 0.9` this gives 5.5x.
- `ca^2 << 1` (regularized): a router splitting traffic across `x` prefill
  workers hands each an Erlang-like stream (`ca^2 ~= 1/x`), and upstream
  concurrency caps clip bursts further. At `rho = 0.9` with
  `ca^2 ~= 0.2` the factor is ~2 — matching what high-utilization
  disagg fleets typically observe.

`prefill_queueing_ttft_factor` in `sdk/picking.py` implements this.

## Why the default stays 1.8

The legacy constant sits inside the regularized-arrival regime at high
utilization (`factor(0.9, ca^2~=0.18) ~= 1.8`; equivalently a Poisson-fed
worker at `rho ~= 0.62`), consistent with fleet observations of ~2x — so it
is kept as the default and gains a knob instead of being replaced.

Two deliberate non-couplings:

- The rate-matching prefill degradation factor (0.9) is a **throughput
  capacity derate** used by rate matching; it is *not* reused as the
  queueing utilization here. A capacity derate says how much of nominal
  throughput to count on when sizing, not how hot workers run sustained —
  plugging it into the Poisson branch would predict 5.5x, contradicted by
  observation.
- TPOT carries no queueing correction: in steady state the queueing time
  lives in TTFT (conservation across the request cycle).

## Overriding

`Task.autoscale_ttft_correction_factor` pins any fixed multiplier;
`prefill_queueing_ttft_factor(rho, cs^2, ca^2)` computes one for a specific
operating point (e.g. single-prefill-worker deployments fed by open
traffic should consider the Poisson branch, which is materially larger).

Scope note: this is a single-stage steady-state correction for the
candidate filter. Full queueing structure (distributions, transients,
multi-stage disagg interaction) is out of scope here and belongs to the
queueing-model evaluator work.

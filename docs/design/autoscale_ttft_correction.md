# Autoscale TTFT queueing pre-correction

The disagg autoscale/sweep paths filter prefill-worker candidates against
the TTFT SLA using the **static** per-worker `ttft` (one prompt, empty
queue). A worker serving a sustained arrival stream additionally queues
prompts, so the static value is multiplied by a pre-correction factor
before filtering. This used to be a hand-tuned constant (`1.8`).

## The derived factor

For a prefill worker at utilization `rho`, the M/G/1 mean waiting time
(Pollaczek-Khinchine) gives the expected TTFT as a multiple of the solo
prefill time:

```text
factor(rho, cv^2) = 1 + rho * (1 + cv^2) / (2 * (1 - rho))
```

`cv^2` is the squared coefficient of variation of the prefill service time
(0 for a fixed request shape). `prefill_queueing_ttft_factor` in
`sdk/picking.py` implements this; the module default evaluates it at the
utilization the sizing itself targets — the rate-matching prefill
degradation factor (0.9) — giving **5.5x**.

## Why replace 1.8

The constant corresponds to `rho ~= 0.62`, well below the design
utilization the autoscale sizing drives workers to, so the TTFT filter was
systematically optimistic exactly where autoscale operates. The formula
carries no fitted constants and moves with the design point if the
rate-matching degradation ever changes.

## Overriding

`Task.autoscale_ttft_correction_factor` (default `None` = derived) pins any
fixed multiplier, e.g. `1.8` to restore the previous behavior, or a value
computed from a different target utilization / service-time variance.

Scope note: this is a single-stage steady-state correction for the
candidate filter. Full queueing structure (distributions, transients,
multi-stage disagg interaction) is out of scope here and belongs to the
queueing-model evaluator work.

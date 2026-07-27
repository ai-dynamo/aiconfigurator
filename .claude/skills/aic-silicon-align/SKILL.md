---
name: aic-silicon-align
description: Use when validating AIC predictions against real GPU serving measurements and root-causing fidelity gaps (prediction-vs-measured issues, memory/concurrency mismatches, per-op latency divergence). Covers dummy-weight methodology, measurement parity traps, component-isolation ladders, and ledger reconciliation.
---

# AIC Silicon Alignment

Root-cause a prediction-vs-measurement gap by isolating layers and comparing
adjacent ones only. Never attribute a residual across more than one rung.

## The ladder

1. **Reproduce the prediction** (CPU-only): same yaml/version/flags. If the
   reported numbers don't reproduce, stop — it's a config diff, not fidelity.
2. **Consistency arithmetic before GPUs**: `throughput = concurrency x
   per-user speed`. Derive the *effective* concurrency from the report; a gap
   vs configured concurrency localizes the problem to admission/memory, not
   kernel speed.
3. **Engine memory ledger** (dummy weights suffice): serve with
   `--load-format dummy` and read the engine's own lines — weights GiB, KV
   tokens, max-concurrency. Compare against AIC's per-component memory dict,
   component by component, never totals only (errors cancel: one model showed
   -8/+2.3/-1.4 GiB netting to a small total delta).
4. **Standalone per-role benchmarks** (prefill worker, decode worker) at
   KV-feasible batch sizes; compare against AIC's per-role rows.
5. **Deployment-faithful e2e**: exact images, generated scripts, patches,
   and flags the reporter used. Only this rung may be compared to the
   reported end-to-end numbers.

## What dummy weights do and don't preserve

- Usually preserve: memory footprint and data-independent kernel timing —
  but this is loader-specific, NOT a guarantee. Before relying on it, verify
  against the real checkpoint: parameter dtypes/shapes/layouts and quant
  scales after loading, post-load weight transforms, and that the framework
  dispatches the same kernels (some dummy loaders allocate a different dtype
  than the checkpoint ships, or skip scale tensors that change dispatch).
  If any of these differ, validate the affected measurement with real
  weights before using it.
- Do NOT preserve: anything routed by data — e.g. MoE expert distributions
  collapse (near-identical hidden states -> few unique experts -> weight
  reads shrink several-fold). Mark such measurements as biased and state the
  direction. To still use them: sweep the hidden axis synthetically in the
  collector (controlled router logits, controlled unique-expert count) to
  build a calibration curve, then invert the engine measurement onto it.

## Measurement parity checklist (each one has flipped a conclusion)

- **CUDA-graph capture sizes must match the deployment** — coverage swung a
  small-batch decode step 2.4x in one case. AIC-generated deploys pass an
  explicit list; your benchmark must too.
- Standalone decode ITL is polluted by chunked-prefill mixing when
  `max-num-batched-tokens` is small; a disagg decode worker never prefills.
  Prefer deployment-faithful setups or report medians with the caveat.
- torch-profiler distorts wall time and large-kernel durations (a single
  stream showing >100% busy is the tell). Anchor on unprofiled ITL; trust
  only small-kernel durations from traces.
- EP/TP ranks run in lockstep: stragglers' wait is absorbed into other
  ranks' NCCL kernel durations. Account per-step across all ranks, never one
  rank's kernels in isolation.
- Discard first-run numbers (JIT/autotune warmup skews percentiles).
- Ops that model compute+comm jointly (overlap ops) must be compared against
  the same joint quantity on the engine side.

## Perf-DB cross-checks

- Re-collect the suspect datapoint on the same silicon: separates
  methodology error from machine/version skew.
- Kernel-diff the collector's timed region against an engine trace: same
  kernel families and algos? Extra eager glue ops are a harness bug (see the
  fused-ops fix); missing ops are an accounting gap.
- Collector data is only as true as its op dispatch: verify against
  framework source (file:line@version), per `.claude/rules/collector/`.

## Model/checkpoint facts are per-checkpoint, not per-family

Read quantization `ignore`/`exclude_modules` from the checkpoint (e.g. NVFP4
releases keep attention BF16; native FP8 ones quantize it). Framework
defaults (memory fractions, capture sizes) come from framework source with
citations, never from memory.

## Discipline

- Keep a running ledger (prediction | measured | delta | attribution) and
  update it every experiment; record retracted claims with the reason.
- Fix coupled accounting items together, or verify that a partial fix does
  not regress feasibility (a correct-but-lone weight increase can push a
  model into "infeasible" because other components over-count).
- A fidelity fix lands with: the measured ledger it was validated against,
  framework citations, and the residuals it deliberately leaves.

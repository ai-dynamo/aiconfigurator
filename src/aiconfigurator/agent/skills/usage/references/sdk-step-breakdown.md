# SDK Step Breakdown

Use the CLI first for per-operation latency. Drop to SDK-style analysis only
when the user needs deeper step-level diagnosis than the CLI output provides.

## CLI First

For a known aggregate config:

```bash
aiconfigurator cli estimate \
  --model Qwen/Qwen3-32B \
  --system h200_sxm \
  --backend trtllm \
  --database-mode SILICON \
  --isl 4000 \
  --osl 1000 \
  --batch-size 128 \
  --ctx-tokens 4000 \
  --tp-size 4 \
  --pp-size 1 \
  --print-per-ops-latency
```

For a known disaggregated config, pass explicit prefill/decode batch sizes,
worker counts, and parallelism:

```bash
aiconfigurator cli estimate \
  --estimate-mode disagg \
  --model Qwen/Qwen3-32B \
  --system h200_sxm \
  --backend trtllm \
  --database-mode SILICON \
  --isl 4000 \
  --osl 1000 \
  --prefill-batch-size 1 \
  --prefill-num-workers 2 \
  --prefill-tp-size 4 \
  --decode-batch-size 128 \
  --decode-num-workers 2 \
  --decode-tp-size 4 \
  --print-per-ops-latency
```

## What the Breakdown Means

- Aggregate mode reports `mix_step`, `genonly_step`, and scheduling counts.
- Disaggregated mode reports `prefill` from `static_ctx` and `decode` from
  `static_gen`.
- `static_ctx` models the prompt/context phase.
- `static_gen` models generation-only decoding steps.
- `static` combines context and generation.
- The `stride` parameter trades generation-step precision for speed by sampling
  every Nth decode step.

## When to Use SDK Objects

Use SDK objects when you need to compare hand-picked runtime configs, alter
`stride`, isolate `static_ctx` versus `static_gen`, or inspect
`InferenceSummary` fields directly.

The relevant objects are:

- `aiconfigurator.sdk.inference_session.InferenceSession`
- `aiconfigurator.sdk.config.RuntimeConfig`
- backend-specific `run_static` implementations through `InferenceSession`
- `InferenceSummary.get_context_latency_dict()`
- `InferenceSummary.get_generation_latency_dict()`
- `InferenceSummary.get_per_ops_data()`

Prefer reusing `aiconfigurator.cli.api.cli_estimate` if the desired inputs map
cleanly to CLI estimate arguments. It returns structured fields plus
`per_ops_data` when available.

## Reporting Guidance

- Separate prefill latency from decode latency.
- Name the slowest operations and their percentage of that step.
- Include batch size, TP/PP/DP, MoE TP/EP, ISL, OSL, database mode, and backend
  version.
- Do not claim a step breakdown is live profiling; it is still AIC estimation.
- Keep final diagnosis grounded in `SILICON`; use `SOL` only for an upper-bound
  cross-check.

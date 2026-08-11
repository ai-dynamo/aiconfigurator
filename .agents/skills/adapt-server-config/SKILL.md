---
name: adapt-server-config
description: Convert InferenceX DB config/benchmark pairs, Dynamo recipe YAML, or confirmed custom serving configs into validated aiconfigurator estimate requests. Use when adapting server configuration, checking whether a recipe can be estimated, generating canonical request JSON, or explicitly running estimates from adapted requests.
---

# Adapt Server Config

Create versioned AIC estimate requests without silently dropping operating points. Keep adaptation separate from estimate execution.

## Choose the workflow

- InferenceX DB record plus benchmark row: use the helper with `--format inferencex`.
- DynamoGraphDeployment with optional `perf.yaml`, or a concrete `dynamo-ci` SGLang benchmark recipe: use the helper with `--format dynamo`.
- Existing canonical request: use the helper with `--format request` to validate it.
- Any other format: follow [custom-mapping.md](references/custom-mapping.md). Do not pass it to a known-format adapter.

## Adapt a known format

Run from the aiconfigurator repository root:

```bash
uv run python .agents/skills/adapt-server-config/scripts/adapt_config.py \
  --format inferencex --config configs-record.json --benchmark benchmark-row.json \
  --overrides '{"backend_version":"0.19.0"}'
```

```bash
uv run python .agents/skills/adapt-server-config/scripts/adapt_config.py \
  --format dynamo --deploy deploy.yaml --perf perf.yaml \
  --overrides '{"system_name":"h200_sxm"}'
```

The helper prints an ordered adaptation report. Preserve every outcome, including rejected points and warnings. Treat any rejection as unresolved; do not repair it with guesses.

For benchmark cookbooks or Slurm command templates, require a rendered concrete recipe or DynamoGraphDeployment. For Helm inputs, require a rendered DynamoGraphDeployment; the matching unrendered `benchmark-values.yaml` can supply literal `toolPipeline` workload points through `--perf`. Do not evaluate templates or infer runtime parameters.

The helper validates adapted requests against both the Python model and packaged JSON Schema. Use `--output PATH` to save the report.

## Handle unknown formats

Read [custom-mapping.md](references/custom-mapping.md). Inspect the source, then show the user:

1. Explicit source fields.
2. Inferred canonical fields and the evidence for each inference.
3. Missing fields and assumptions.
4. Every discovered operating point in source order.

Obtain confirmation before creating canonical request JSON. After confirmation, create `aic-estimate-request/1.0.0` JSON and validate it:

```bash
uv run python .agents/skills/adapt-server-config/scripts/adapt_config.py \
  --format request --request request.json
```

Do not add heuristic behavior to the SDK. Do not infer missing model identity, system, workload, concurrency, or speculative-token acceptance without confirmation.

## Run estimates only on explicit request

Adaptation never runs an estimate. Add `--run-estimate` only when the user explicitly asks to execute estimates:

```bash
uv run python .agents/skills/adapt-server-config/scripts/adapt_config.py \
  --format dynamo --deploy deploy.yaml --perf perf.yaml \
  --overrides '{"system_name":"h200_sxm"}' --run-estimate
```

Never execute recipe shell commands. The SDK only parses literal configuration values.

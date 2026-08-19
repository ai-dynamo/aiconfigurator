# Versioned Config Adapter

The config adapter converts external serving configurations into validated AIC
estimate requests. Adaptation performs no estimate and executes no source shell
commands.

## Public API

Import the API from `aiconfigurator.sdk.config_adapter`:

```python
from pathlib import Path

from aiconfigurator.cli.api import cli_estimate
from aiconfigurator.sdk.config_adapter import (
    AdapterOverrides,
    DynamoRecipeSource,
    InferenceXSource,
    adapt_config,
    to_cli_estimate_kwargs,
)

report = adapt_config(
    DynamoRecipeSource(Path("deploy.yaml"), Path("perf.yaml")),
    AdapterOverrides(system_name="h200_sxm", backend_version="0.19.0"),
)

for outcome in report.outcomes:
    if outcome.status == "adapted":
        # Run only when the application explicitly requests an estimate.
        result = cli_estimate(**to_cli_estimate_kwargs(outcome.request))
```

`EstimateRequestV1` uses schema version
`aic-estimate-request/1.0.0`. The packaged snapshot is available through
`EstimateRequestV1.schema_path()`. Unknown versions and unknown canonical fields
are rejected.

The generated JSON Schema is the language-neutral structural contract for field
types and bounds. `EstimateRequestV1` remains authoritative for cross-field
rules, including prefix versus ISL, topology and MoE width, and the
`nextn`/`nextn_accepted` pairing.

## Tutorial: adapt a Dynamo recipe

The repository helper turns one concrete Dynamo recipe into canonical AIC
requests. Adaptation and estimation are separate operations: the first command
below only parses and validates configuration; it does not run AIC and does not
execute anything from the recipe.

### 1. Choose concrete inputs

Start with a `deploy.yaml` containing one `DynamoGraphDeployment`. If the
operating point is stored separately, also pass the adjacent `perf.yaml`.
The performance file is optional only when the deployment itself contains a
literal workload and concurrency.

For Helm recipes, render `recipe-values.yaml` into a concrete
`DynamoGraphDeployment` before using the adapter. The adapter does not render
Helm, expand templates, or execute recipe shell. An unrendered
`benchmark-values.yaml` can still be used as `--perf` when its
`toolPipeline[].config` contains literal ISL, OSL, and concurrency values.

### 2. Generate and inspect the adaptation report

Run this command from the AIC repository root:

```bash
uv run python .agents/skills/adapt-server-config/scripts/adapt_config.py \
  --format dynamo \
  --deploy /path/to/recipe/deploy.yaml \
  --perf /path/to/recipe/perf.yaml \
  --source-reference https://github.com/ai-dynamo/dynamo/blob/<sha>/recipes/<recipe>/deploy.yaml \
  --output /tmp/adaptation-report.json
```

Omit `--perf` when there is no separate performance file. The report preserves
every discovered operating point in source order. Each outcome has one of two
statuses:

- `adapted`: `request` contains a validated `aic-estimate-request/1.0.0`
  object. Review `provenance.assumptions` before using it.
- `rejected`: no request was created. Read `diagnostics[].message`, `path`, and
  `hint` to find the missing, conflicting, or unsafe source value.

A rejection is an adaptation failure, not an AIC performance result. No
estimate has been attempted for that point. The helper exits with status 1 if
any point is rejected, while still writing the complete report.

### 3. Resolve missing values with explicit overrides

Do not guess values that the recipe does not declare. Verify them from the
recipe owner or benchmark record, then pass only those confirmed values through
`--overrides`. For example:

```bash
uv run python .agents/skills/adapt-server-config/scripts/adapt_config.py \
  --format dynamo \
  --deploy /path/to/recipe/deploy.yaml \
  --perf /path/to/recipe/perf.yaml \
  --overrides '{
    "system_name": "h200_sxm",
    "backend_version": "0.19.0",
    "nextn_accepted": 1.5,
    "decode_batch_size": 16
  }' \
  --output /tmp/adaptation-report.json
```

Common overrides include `system_name`, `backend_version`, `isl`, `osl`,
`concurrency`, `batch_size`, `prefill_batch_size`, `decode_batch_size`,
`kvcache_quant_mode`, and the paired `nextn`/`nextn_accepted` values. Overrides
take precedence over source values and are recorded as provenance assumptions
where appropriate.

### 4. Run AIC only after reviewing the mapping

Once every intended point is adapted and its canonical request is correct, add
`--run-estimate` to the same command:

```bash
uv run python .agents/skills/adapt-server-config/scripts/adapt_config.py \
  --format dynamo \
  --deploy /path/to/recipe/deploy.yaml \
  --perf /path/to/recipe/perf.yaml \
  --overrides '{"system_name":"h200_sxm","backend_version":"0.19.0"}' \
  --run-estimate \
  --output /tmp/estimate-report.json
```

The helper validates each canonical request against the packaged JSON Schema,
lowers adapted requests through `to_cli_estimate_kwargs`, and writes returned
estimates under `estimates`. It never executes commands embedded in the Dynamo
files.

## Request groups

| Group | Content |
| --- | --- |
| `model` | Hugging Face path and speculative decoding settings |
| `quantization` | GEMM, KV cache, FMHA, MoE, and communication modes |
| `backend` | Framework, optional database version, database mode, transfer policy |
| `systems` | Prefill/agg system and optional decode system |
| `workload` | ISL, OSL, source concurrency, prefix, and image shape |
| `topology` | Agg worker or prefill/decode workers with replicas, batch, and parallelism |
| `runtime` | Memory fraction, maximum sequence length, system paths, engine backend |
| `provenance` | Source identity, adapter version, identifiers, and assumptions |

The topology records source replicas and GPUs per replica. Lowering converts
source concurrency into each worker's active batch while preserving the original
concurrency in the request. `cli_estimate` has no aggregated worker-count
argument, so aggregated replicas are omitted during lowering and that limitation
is recorded in source provenance.

## Precedence and failure behavior

Values resolve in this order:

1. Explicit `AdapterOverrides`.
2. One unambiguous source value.
3. A documented source-specific default.

Programmatic adaptation fails closed. Missing model, system, workload,
concurrency, or speculative-token acceptance creates a rejected outcome.
Conflicting command and ConfigMap values are rejected. An unpinned backend
version is accepted with a warning because AIC will choose its latest compatible
database version.

When global concurrency is not evenly divisible by source replicas and
attention-DP ranks, callers must provide an explicit aggregated or decode batch
override. The canonical request keeps the original global concurrency and
records the batch override as a provenance assumption.

Every discovered operating point creates one ordered outcome. Partial success is
allowed; invalid points are never omitted.

## InferenceX records

`InferenceXSource` accepts one `configs.json` record and one matching benchmark
row. It maps NVIDIA hardware names, folded framework names, known model and
precision aliases, workload, concurrency, and worker topology.

InferenceX does not expose pipeline parallelism, so PP defaults to 1. A
disaggregated prefill batch defaults to 1 unless overridden. Both assumptions are
recorded in provenance. Aggregated exports use `decode_num_workers=0` as an
irrelevant sentinel; it is normalized to one replica before worker validation.
All actual worker counts must be positive. MTP rows require explicit `nextn` and
`nextn_accepted` overrides.

## Dynamo recipes

`DynamoRecipeSource` safely parses multi-document YAML containing ConfigMaps,
one DynamoGraphDeployment, and optional performance Jobs. It supports standard
agg and P/D-disaggregated vLLM, SGLang, and TRT-LLM workers in both the legacy
`spec.services` schema and the current `spec.components` schema.

Pass a `Path` to load a YAML file. A plain `str` is always parsed as YAML text
and is never resolved as a filesystem path.

It also accepts concrete `dynamo-ci` SGLang benchmark recipes with top-level
`model`, `resources`, `backend.sglang_config`, and `benchmark` sections. The
adapter expands literal `benchmark.concurrencies` in source order and derives
prefill/decode worker size from nodes, workers, GPUs per node, and engine
parallelism. Role-specific memory fractions cannot be represented by the
current flat estimate API, so they require an explicit shared
`free_gpu_memory_fraction` override.

Worker sizing uses replicas, `multinode.nodeCount`, GPU limits, and literal engine
parallelism flags. Literal environment substitutions are supported. Shell
comments are discarded while parsing worker commands, and no command is
executed. A literal space-separated `CONCURRENCIES` environment value expands
to ordered operating points. A known numeric
`CONCURRENCY_PER_GPU * DEPLOYMENT_GPU_COUNT` performance point is expanded
without executing shell. Multiple explicit points create multiple outcomes.

TRT-LLM speculative settings are read from worker flags and engine ConfigMaps
whose volume mount resolves the exact `--extra-engine-args` path. Unmounted
files, conflicting active depths across roles, and ambiguous mounts are
rejected. A zero depth remains disabled. Active speculation requires an
explicit `nextn_accepted` override; the adapter never guesses an acceptance
rate. Uneven concurrency distribution and conflicting role-specific memory
fractions remain rejected with stable diagnostics.

For Helm-based benchmark infrastructure, render `recipe-values.yaml` into a
DynamoGraphDeployment before adaptation. The matching unrendered
`benchmark-values.yaml` may be passed as `performance`; literal
`toolPipeline[].config` ISL, OSL, and concurrency lists are expanded directly.
This keeps Helm execution outside the SDK while avoiding shell parsing for the
workload.

Adapter v1 rejects EPD/encode, AFD, heterogeneous hardware,
`componentType: main`, unsupported backends, ambiguous values, arbitrary
shell-derived values, and topologies other than agg or P/D disaggregation.
For a worker launched through a shell wrapper, the adapter may extract a
literal trailing `python -m dynamo.<backend>` invocation; it never evaluates
the wrapper or accepts shell control operators in the extracted invocation.
Parameterized benchmark cookbooks, Slurm command templates, and Helm values
must be rendered into a concrete recipe or DynamoGraphDeployment first. The
programmatic adapter never evaluates template expressions or executes commands.

## Agent workflow

The repo-local `.agents/skills/adapt-server-config` skill wraps the SDK for known
formats. For an unknown format it must show inferred fields and obtain user
confirmation before constructing canonical JSON. The skill validates every
request against the packaged schema and runs estimates only when explicitly
requested.

The skill, helper, fixtures, reports, datasets, and future gap-analysis pipelines
are infrastructure. They are excluded from both wheels and the Rust crate.

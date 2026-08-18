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
agg and P/D-disaggregated vLLM, SGLang, and TRT-LLM workers.

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

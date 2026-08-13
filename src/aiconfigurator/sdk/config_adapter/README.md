# Config Adapter

This package converts supported server configurations into validated,
versioned AIC estimate requests. Adaptation is deterministic and does not run an
estimate or execute commands from source recipes.

For the full contract and mapping details, see the
[config adapter guide](../../../../docs/config_adapter.md).

## Public API

Import public types and functions from `aiconfigurator.sdk.config_adapter`:

```python
from pathlib import Path

from aiconfigurator.cli.api import cli_estimate
from aiconfigurator.sdk.config_adapter import (
    AdapterOverrides,
    DynamoRecipeSource,
    adapt_config,
    to_cli_estimate_kwargs,
)

report = adapt_config(
    DynamoRecipeSource(
        deployment=Path("deploy.yaml"),
        performance=Path("perf.yaml"),
    ),
    AdapterOverrides(backend_version="0.19.0"),
)

for outcome in report.outcomes:
    if outcome.status == "adapted":
        result = cli_estimate(**to_cli_estimate_kwargs(outcome.request))
```

Call `cli_estimate` only when estimation is explicitly requested. Adaptation
itself never invokes it.

## Supported sources

- `InferenceXSource`: one DB-export config record and one benchmark operating
  point.
- `DynamoRecipeSource`: standard aggregate or prefill/decode-disaggregated
  DynamoGraphDeployment YAML, optional performance YAML, and concrete
  `dynamo-ci` benchmark recipes.

The adapters support vLLM, SGLang, and TRT-LLM. Unsupported or ambiguous
topologies are returned as rejected outcomes with structured diagnostics.
For Dynamo performance Jobs, literal `CONCURRENCIES` values are expanded in
order and shell comments are ignored during command parsing. Mounted TRT-LLM
engine ConfigMaps are inspected for speculative decoding; an active setting is
rejected unless the caller supplies `nextn_accepted`.

## Contract

Every adapted request uses schema version `aic-estimate-request/1.0.0`.
`EstimateRequestV1.schema_path()` locates the packaged JSON Schema snapshot.
The Python model remains authoritative for cross-field validation.

Adaptation follows these rules:

1. Explicit overrides take precedence.
2. Unambiguous source values are used next.
3. Documented source-specific defaults are used last.
4. Every discovered operating point produces one ordered outcome.
5. Missing or conflicting required values are rejected; no point is silently
   dropped.

## Package layout

| Module | Responsibility |
| --- | --- |
| `schema.py` | Canonical request, override, report, and diagnostic models |
| `api.py` | Public dispatch and lowering to `cli_estimate` keyword arguments |
| `inferencex.py` | InferenceX record adaptation |
| `dynamo.py` | DynamoGraphDeployment and performance YAML adaptation |
| `dynamo_ci.py` | Concrete `dynamo-ci` recipe adaptation |
| `schemas/` | Language-neutral JSON Schema snapshot |

Only the Python package and canonical schema belong in the upper wheel. Agent
skills, fixtures, datasets, reports, and gap-analysis infrastructure remain
repository-only.

## Tests

Run the focused suite from the repository root:

```bash
uv run --extra dev pytest -q \
  tests/unit/sdk/config_adapter \
  tests/integration/test_config_adapter_estimate.py
```

# Cross-Module Impact Map

When modifying the generator, check this map to identify affected modules outside
`src/aiconfigurator/generator/`.

## Dependency Directions

```text
CLI (cli/)
  |-- calls --> generator/api.py (generate_config_from_input_dict, parse_cli_params)
  |-- calls --> generator/main.py (render-config, render-artifacts subcommands)
  |-- reads --> generator/config/deployment_config.yaml (for --generator-help)

SDK/Profiler Bridge (generator/module_bridge.py)
  |-- reads <-- sdk/task.py (TaskConfig)
  |-- reads <-- sdk/perf_database.py (result DataFrame)
  |-- calls --> generator/api.py (generate_config_from_input_dict)

Dynamo Profiler (external: ai-dynamo/dynamo/components/src/dynamo/profiler)
  |-- produces --> performance CSV data consumed by sdk/perf_database.py
  |-- field names must match --> model_configs/*.json + systems/*.yaml

Generator Validator (tools/generator_validator/)
  |-- reads <-- generated artifacts (cli_args, engine configs, k8s manifests)
  |-- validates against --> actual backend engine APIs (vLLM, SGLang, TRT-LLM)

Support Matrix (tools/support_matrix/)
  |-- reads <-- model_configs/*.json, systems/*.yaml
  |-- reads <-- generator/config/backend_version_matrix.yaml

Collector (collector/)
  |-- produces --> performance data CSVs
  |-- references --> generator parameter names for benchmark configs
```

## Impact by Change Type

### Adding a New Parameter

| Step | File(s) | Why |
|---|---|---|
| 1. Schema | `src/aiconfigurator/generator/config/deployment_config.yaml` | Define the param, default, backend support |
| 2. Mapping | `src/aiconfigurator/generator/config/backend_config_mapping.yaml` | Map to backend CLI flag names |
| 3. CLI | `src/aiconfigurator/cli/main.py` or CLI arg group | Expose via `--generator-set` (auto if in schema) |
| 4. Bridge | `src/aiconfigurator/generator/module_bridge.py` | Pass from SDK search results if profiler-sourced |
| 5. Validator | `tools/generator_validator/` | Add to expected flags if new CLI flag |
| 6. Docs | `docs/cli_user_guide.md` | Document the new parameter |

### Renaming a Parameter

| Step | File(s) | Why |
|---|---|---|
| 1. Schema | `deployment_config.yaml` | Update key name |
| 2. Mapping | `backend_config_mapping.yaml` | Update param_key |
| 3. Bridge | `module_bridge.py` | Update field extraction from TaskConfig/DataFrame |
| 4. Rules | `rule_plugin/*.rule` | Update all references in rule expressions |
| 5. Templates | `backend_templates/**/*.j2` | Update variable references |
| 6. Aggregators | `aggregators.py` | Update if param is aggregated there |
| 7. Tests | `tests/unit/generator/` | Update test fixtures and assertions |
| 8. Backward compat | `aggregators.py` or `api.py` | Add alias if old name was user-facing |

### Modifying Rule Logic

| Check | File(s) | Why |
|---|---|---|
| 1. All backends | `rule_plugin/*.rule` | Same rule may need update in all backends |
| 2. Benchmark rules | `rule_plugin/benchmark/*.rule` | Benchmark may need different logic |
| 3. Bridge output | `module_bridge.py` | If rule uses fields from bridge output |
| 4. Template consumers | `backend_templates/**/*.j2` | Templates may assume rule output format |
| 5. Validator | `tools/generator_validator/` | New computed values may need validation |

### Adding a New Backend Version

| Step | File(s) | Why |
|---|---|---|
| 1. Version matrix | `src/aiconfigurator/generator/config/backend_version_matrix.yaml` | Map Dynamo -> backend version |
| 2. Templates | `src/aiconfigurator/generator/config/backend_templates/<backend>/` | Version-specific templates if CLI changed |
| 3. Image defaults | `deployment_config.yaml` | Container image tag expressions |
| 4. Support matrix | `tools/support_matrix/` | Update supported combinations |
| 5. Validator | `tools/generator_validator/` | May need new backend image for validation |

### Modifying Template Output Format

| Check | File(s) | Why |
|---|---|---|
| 1. K8s manifests | Verify YAML is valid K8s | `kubectl apply --dry-run` |
| 2. Run scripts | Verify bash syntax | `bash -n generated_run.sh` |
| 3. Engine configs | Verify against backend schema | `tools/generator_validator/` |
| 4. Collector | `collector/` | If collector parses generated configs |

## External Dependencies

### Dynamo Profiler (ai-dynamo/dynamo)

The Dynamo profiler runs benchmarks and produces performance data that the SDK
consumes. Generator changes that affect:

- **Worker topology** (TP, PP, DP dimensions) -- profiler must benchmark matching configs
- **Batch size computation** -- profiler's benchmark configs must align
- **K8s deployment format** -- Dynamo's DynamoGraphDeployment CRD must accept generated manifests
- **Entry points** (e.g., `python3 -m dynamo.vllm`) -- must match Dynamo runtime

When changing these areas, verify compatibility with the Dynamo profiler or flag
the change to the developer as potentially requiring upstream Dynamo updates.

### Backend Engines (vLLM, SGLang, TRT-LLM)

The generator produces CLI arguments and config files consumed by these engines.
Changes to generated flags must be validated against the actual engine version's API:

- **vLLM**: `vllm.engine.arg_utils.EngineArgs`
- **SGLang**: `sglang.srt.server_args.ServerArgs`
- **TRT-LLM**: `tensorrt_llm` Python API / `trtllm-serve` CLI

Use `tools/generator_validator/` when backend images are available.

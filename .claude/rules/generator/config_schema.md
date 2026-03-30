# Config Schema Modification Reference

The generator's behavior is driven by two declarative YAML files. This reference
covers safe practices for modifying them.

## When to Use This Reference

- Adding a new user-facing configuration parameter
- Modifying defaults (global or backend-specific)
- Adding a parameter mapping for a backend
- Changing parameter constraints or required status
- Adding a new config section (namespace)

## Key Files

| File | Purpose |
|---|---|
| `src/aiconfigurator/generator/config/deployment_config.yaml` | Input schema: ~54 params, defaults, constraints |
| `src/aiconfigurator/generator/config/backend_config_mapping.yaml` | Unified param -> backend CLI flag mapping |
| `src/aiconfigurator/generator/rendering/schemas.py` | Schema validation, default application |

## Schema File Structures

**deployment_config.yaml entry:**
```yaml
- key: <Section>.<param_name>        # e.g., K8sConfig.k8s_namespace
  required: true|false
  default: <value or Jinja2 expr>    # e.g., '"default-ns"' or 'model_path.split("/")[-1]'
  backend_defaults:                   # Optional: override default per backend
    trtllm: <value or expr>
    vllm: <value or expr>
    sglang: <value or expr>
  backends:                            # Optional: restrict to specific backends
    - trtllm
```

**backend_config_mapping.yaml entry:**
```yaml
- param_key: <unified_name>          # e.g., tensor_parallel_size
  vllm: <cli_flag_name>             # e.g., tensor-parallel-size (or null if unsupported)
  sglang: <cli_flag_name>
  trtllm: <cli_flag_name>
  # OR with value transformation:
  sglang:
    key: disable-cuda-graph-padding
    value: "not cuda_graph_enable_padding"  # Jinja2 expression
```

## Workflow

### 1. Define the Parameter

- What section does it belong to? (ServiceConfig, K8sConfig, ModelConfig, etc.)
- Is it required or optional?
- What is the default value? Is it static or computed?
- Does the default differ by backend?
- Which backends support this parameter?

### 2. Add to deployment_config.yaml

- Place in the correct section (entries are grouped by section)
- If default uses an expression, test it:
  1. What variables does the expression reference?
  2. Are those variables available at default-evaluation time?
  3. Test with missing/None values of referenced variables
- If backend-specific: add `backend_defaults` block
- If backend-restricted: add `backend_constraint` list

### 3. Add to backend_config_mapping.yaml

- Add entry with unified `param_key`
- For each backend:
  - If supported: add the CLI flag name
  - If unsupported: set to null
  - If value needs transformation: use `{key:, value:}` format
- VERIFY the CLI flag names:
  - TRT-LLM: check `tensorrt_llm` Python API or docs
  - vLLM: check `vllm.engine.arg_utils.EngineArgs`
  - SGLang: check `sglang.srt.server_args.ServerArgs`

### 4. Update Rule Files (if needed)

If the parameter requires computation, add rules.
See `.claude/rules/generator/rule_authoring.md` for rule authoring details.

### 5. Update Templates (if needed)

If the parameter needs special handling in artifacts.
See `.claude/rules/generator/template_authoring.md` for template authoring details.

### 6. Validate

- Run generator with the new parameter set to a test value
- Run generator with the parameter omitted (verify default works)
- Run generator validator to check output against backend API schemas
- Verify the parameter appears in `--generator-help` output

## Common Mistakes

1. **Jinja2 expression quoting**: String defaults must be double-quoted inside single
   quotes: `default: '"my-string"'`. Without inner quotes, YAML interprets it as a
   bare string and Jinja2 evaluation breaks.

2. **Missing null handling in expressions**: If `default: 'model_path.split("/")[-1]'`
   and `model_path` is None, the expression throws. Guard with:
   `'model_path.split("/")[-1] if model_path else ""'`.

3. **Backend flag name format mismatch**: TRT-LLM uses underscores (`kv_cache_dtype`),
   vLLM/SGLang use dashes (`kv-cache-dtype`). Always check the actual backend CLI.

4. **Mapping value transformation gotchas**: SGLang's `disable-cuda-graph-padding` is
   the INVERSE of the unified `cuda_graph_enable_padding`. The mapping
   `value: "not cuda_graph_enable_padding"` handles this. Forgetting the inversion
   produces a silent semantic bug.

5. **Circular default references**: If param A defaults to an expression using param B,
   and B defaults to an expression using A, the schema loader will silently produce
   None for both. Verify the dependency chain is acyclic.

## Checklist

```text
[ ] Define parameter: section, required/optional, default, backend support
[ ] Add entry to deployment_config.yaml with correct syntax
[ ] Test default expression with edge cases (None, empty string)
[ ] Add entry to backend_config_mapping.yaml
[ ] Verify CLI flag names against actual backend APIs
[ ] Add rules if parameter needs computation
[ ] Add template changes if needed
[ ] Run generator with parameter set and omitted
[ ] Run generator validator
[ ] Verify --generator-help output includes new parameter
```

# Template Authoring Reference

Backend templates are the most error-prone part of the generator. This reference
covers safe practices for adding and modifying Jinja2 templates.

## When to Use This Reference

- Adding a new backend version template
- Modifying existing templates (bug fix, feature addition)
- Adding a new parameter to template rendering context
- Adding a new artifact type (e.g., sflow templates)

## Key Files

| File | Purpose |
|---|---|
| `src/aiconfigurator/generator/config/backend_templates/<backend>/` | Jinja2 templates |
| `src/aiconfigurator/generator/config/backend_config_mapping.yaml` | Param name mapping |
| `src/aiconfigurator/generator/config/deployment_config.yaml` | Input schema + defaults |
| `src/aiconfigurator/generator/config/backend_version_matrix.yaml` | Version compatibility |
| `src/aiconfigurator/generator/rendering/engine.py` | Template rendering + context building |
| `src/aiconfigurator/generator/rule_plugin/*.rule` | Rule DSL files |
| `tools/generator_validator/` | Post-generation validation |

## Template Types by Backend

```text
backend_templates/
  vllm/       cli_args[.version].j2, k8s_deploy.yaml.j2, run.sh.j2
  sglang/     cli_args[.version].j2, k8s_deploy.yaml.j2, run.sh.j2, sflow_deploy.yaml.j2
  trtllm/     cli_args.j2, extra_engine_args[.version].yaml.j2, k8s_deploy.yaml.j2, run.sh.j2
  benchmark/  bench_run.sh.j2, k8s_bench.yaml.j2
```

## Workflow

### 1. Identify Scope

- Which backends are affected? (trtllm, vllm, sglang, benchmark, sflow)
- Which artifact types? (cli_args, engine_config, k8s_deploy, run.sh, bench)
- Which versions need the change? (check `backend_version_matrix.yaml`)

### 2. Read Current State

- Read the target template(s)
- Read `backend_config_mapping.yaml` for the parameter mapping
- Read `rendering/engine.py:make_worker_context()` to understand context shape
- Read the corresponding `.rule` file to see if rules compute the variable

### 3. Cross-Backend Parity Check

For each affected parameter:

1. Grep all backend templates for the parameter name
2. Compare handling across backends
3. Flag any backend that is missing the parameter or handles it differently
4. Document intentional differences vs. accidental omissions

### 4. Implement

- **New version templates**: Copy the closest prior version, modify. NEVER edit prior versions.
- **Fixes**: Apply to ALL affected version templates (list them explicitly).
- **New parameters**:
  1. Add to `backend_config_mapping.yaml` if it's a mapped param
  2. Add to `deployment_config.yaml` if it's a new input
  3. Add to `.rule` file if it requires computation
  4. Add to template(s)

### 5. Verify Template Variables

List all variables referenced in the modified template(s). For each variable,
confirm its source:

- [ ] Direct from `deployment_config.yaml` input
- [ ] Computed by rule plugin
- [ ] Mapped via `backend_config_mapping.yaml`
- [ ] Injected by `rendering/engine.py` context

Flag any variable that cannot be traced to a source.

### 6. Test

- Run existing generator unit tests
- If adding a new parameter: add a test in `test_<backend>_cli_args.py`
- Use generator validator (`tools/generator_validator/`) to check output validity
- Compare output diff against golden artifacts if available

## Anti-Patterns

1. **Don't extract shared templates** -- PR #314 attempted shared macros; PR #340 reverted
   it because backend-specific variations broke override mechanisms. Keep templates
   standalone per backend.

2. **Don't fix one version and forget the rest** -- Always enumerate ALL version-specific
   templates. Use `glob backend_templates/<backend>/<artifact>*.j2` to list them.

3. **Don't assume parameter names are the same across backends** -- Check
   `backend_config_mapping.yaml`. SGLang uses `disable-cuda-graph-padding` (inverted!)
   while TRT-LLM uses `cuda_graph_enable_padding`.

4. **Don't add a template variable without a source** -- Every variable must be traceable
   through the pipeline: input -> default -> rule -> mapping -> context -> template.

## Checklist

```text
[ ] Identify all affected backends and version templates
[ ] Read current template + mapping + rule file
[ ] Cross-backend parity check for affected parameters
[ ] Implement changes in ALL affected templates
[ ] Verify all template variables have traced sources
[ ] Update backend_config_mapping.yaml if needed
[ ] Update deployment_config.yaml if needed
[ ] Update .rule files if needed
[ ] Run unit tests
[ ] Run generator validator on output
```

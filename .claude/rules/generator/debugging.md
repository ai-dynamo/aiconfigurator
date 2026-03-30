# Generator Debugging Reference

Generator bugs have distinct patterns: silent output corruption, mode confusion,
version template drift, and MoE edge cases. This reference provides a systematic
backwards-tracing diagnostic approach.

## When to Use This Reference

- Generator produces wrong output (invalid CLI args, missing config blocks, wrong values)
- Engine startup fails with generated config
- Generator validator reports mismatches
- User reports deployment failure with generated artifacts

## Diagnostic Pipeline

The generator has 6 stages. A bug can originate at any stage but manifests at the output.
Trace backwards from the output:

```text
Stage 6: TEMPLATE RENDERING (most visible)
  Symptom: Wrong CLI flag name, missing block, wrong YAML structure
  Check:   Read the template (.j2 file), verify variable names match context
  Tool:    Diff the template output against a known-good output

Stage 5: RULE EVALUATION (most common source)
  Symptom: Wrong computed value (batch size, TP, token budget)
  Check:   Read the .rule file, trace expression evaluation
  Tool:    Add debug logging in rule_engine.py:apply_rule_plugins()

Stage 4: PARAMETER MAPPING (subtle bugs)
  Symptom: Correct value, wrong flag name or inverted boolean
  Check:   Read backend_config_mapping.yaml for the parameter
  Tool:    Grep for the param_key across all mapping entries

Stage 3: DEFAULT APPLICATION (silent failures)
  Symptom: Parameter is None/missing when it should have a value
  Check:   Read deployment_config.yaml for the entry's default expression
  Tool:    Evaluate the default expression manually with test inputs

Stage 2: INPUT PARSING (user-facing)
  Symptom: Override not taking effect, wrong section targeting
  Check:   Read api.py:parse_cli_params(), check dotted-path resolution
  Tool:    Print parsed input dict before processing

Stage 1: SCHEMA LOADING (rare)
  Symptom: Schema validation error or missing field
  Check:   Read rendering/schemas.py:apply_defaults()
```

## Workflow

### 1. Reproduce

- Get the exact input config (YAML + CLI overrides)
- Get the backend and version
- Run the generator and capture output
- Identify the specific wrong output (CLI flag, YAML block, value)

### 2. Trace Backwards from Output

Start at Stage 6 and work backwards:

**a. TEMPLATE**: Find the template that produced the wrong output
- Glob: `src/aiconfigurator/generator/config/backend_templates/<backend>/<artifact>*.j2`
- Find the exact version template used (check `backend_version_matrix.yaml`)
- Read the template, identify which variable is wrong

**b. CONTEXT**: Check what value the variable had in the rendering context
- Read `src/aiconfigurator/generator/rendering/engine.py:make_worker_context()`
- Trace how the variable gets into the context dict

**c. RULE**: Check if a rule computed the wrong value
- Read `src/aiconfigurator/generator/rule_plugin/<backend>.rule`
- Find the rule for this variable
- Check scope prefix (agg vs prefill vs decode)
- Evaluate the expression manually with known inputs

**d. MAPPING**: Check if the mapping is correct
- Read `src/aiconfigurator/generator/config/backend_config_mapping.yaml`
- Verify the param_key -> backend flag mapping
- Check for value transformations

**e. DEFAULT**: Check if the default was applied correctly
- Read `src/aiconfigurator/generator/config/deployment_config.yaml` for the parameter
- Evaluate the default expression
- Check `backend_defaults` if applicable

### 3. Identify Root Cause

Common root causes:
- Wrong scope prefix in `.rule` file (e.g., `agg` instead of `agg_prefill_decode`)
- Stale variable reference in template after context restructuring
- Missing null guard in expression `(value or 0)`
- Parameter only fixed in one backend's `.rule`, not all
- Version template not updated (fix in v1.0.0 but not v1.1.0)

### 4. Fix

- Apply fix at the correct stage (don't patch templates for rule bugs)
- Check ALL backends and ALL version templates for the same issue
- Use `.claude/rules/generator/template_authoring.md` if fixing templates
- Use `.claude/rules/generator/rule_authoring.md` if fixing rules

### 5. Verify

- Re-run with the original failing input
- Run generator validator
- Check other backends for the same bug pattern
- Add regression test

## Bug Pattern Catalog

| Pattern | Example PR | Root Cause | Fix Location |
|---|---|---|---|
| Wrong value for one mode | #609 | Rule scope wrong | `.rule` file scope prefix |
| Missing parameter in one backend | #579 | Rule not added | `.rule` file for that backend |
| Invalid CLI flag | #613 | Backend doesn't support it | mapping yaml + template |
| Stale template variable | #540 | Context restructured | ALL version templates |
| RFC-1123 violation | #490 | No sanitization | `naive.py` or `aggregators.py` |
| Disagg config in agg mode | #609 | Missing mode guard | Template `{% if %}` block |
| Version mismatch | #519 | Deprecated flag | Version-specific template |

## Checklist

```text
[ ] Reproduce the bug with exact inputs
[ ] Identify the wrong output (specific flag/value/block)
[ ] Trace backwards through the 6-stage pipeline
[ ] Identify root cause stage and specific file
[ ] Check all backends for the same bug pattern
[ ] Check all version templates for the same bug pattern
[ ] Apply fix at the correct stage
[ ] Add regression test
[ ] Run generator validator
```

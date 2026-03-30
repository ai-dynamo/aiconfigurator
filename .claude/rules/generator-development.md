---
description: >
  Universal rules for developing the aiconfigurator generator module.
  Covers cross-module awareness, template/rule safety, and documentation
  discipline. Applies to all generator work including templates, rules,
  config schemas, and rendering logic.
paths:
  - "src/aiconfigurator/generator/**"
---

Rules and workflows for safely modifying the aiconfigurator generator module.
Read the relevant reference file in `.claude/rules/generator/` before starting task-specific work.

## Generator Pipeline Overview

The generator transforms deployment intent into backend-specific artifacts through 6 stages:

```text
Raw Input (YAML + CLI --set)
  -> [1] Input Parsing        (api.py)
  -> [2] Default Application   (rendering/schemas.py, deployment_config.yaml)
  -> [3] Rule Evaluation       (rendering/rule_engine.py, rule_plugin/*.rule)
  -> [4] Parameter Mapping     (rendering/engine.py, backend_config_mapping.yaml)
  -> [5] Template Rendering    (rendering/engine.py, backend_templates/**/*.j2)
  -> [6] Artifact Emission     (artifacts.py)
  -> Generated Artifacts: k8s_deploy.yaml, run.sh, cli_args, engine configs
```

Bugs can originate at any stage but manifest at stage 5/6. Always trace backwards.

## High-Risk Files

Modifying these files affects the entire pipeline. Understand the full data flow
before changing them:

| File | Risk | Why |
|---|---|---|
| `src/aiconfigurator/generator/rendering/engine.py` | High | Context building + template rendering; changes affect ALL backends and versions |
| `src/aiconfigurator/generator/rendering/rule_engine.py` | High | Rule DSL evaluator; bugs here corrupt ALL computed parameters silently |
| `src/aiconfigurator/generator/rendering/schemas.py` | High | Default application; wrong defaults propagate through every downstream stage |
| `src/aiconfigurator/generator/module_bridge.py` | Medium | SDK/profiler bridge; field name mismatches break the profiler -> generator flow |

## Universal Rules

These rules apply to ALL generator work regardless of task type.

### Rule 1: Respect Template and Rule Comments

Before modifying any `.j2` template or `.rule` file, read ALL comments in the file.
Comments with `# Guard:` or `# Why:` encode hard-won lessons from 60+ production
bug fixes. If the developer's requested change contradicts a guarded comment:

1. Quote the guard comment to the developer
2. Explain why the guard exists
3. Ask the developer to confirm they want to override it
4. Do NOT silently remove or bypass guard comments

When in doubt, the `# Guard:` comments in the source files are authoritative over
`.claude/rules/generator/guard_rails.md`. The reference is a convenient index but
may lag behind code changes.

See `.claude/rules/generator/guard_rails.md` for the full catalog of constraints by backend.

### Rule 2: Assess Cross-Module Impact

Before implementing a generator change, check whether it affects:

- **CLI** (`src/aiconfigurator/cli/`) -- new params need CLI arg registration
- **SDK/profiler bridge** (`src/aiconfigurator/generator/module_bridge.py`) -- field name changes break the bridge
- **Dynamo profiler** (https://github.com/ai-dynamo/dynamo/tree/main/components/src/dynamo/profiler) -- profiler feeds data that generator consumes; schema changes propagate
- **Generator validator** (`tools/generator_validator/`) -- new flags need validator updates
- **Support matrix** (`tools/support_matrix/`) -- new backends/models may need matrix updates
- **Collector** (`collector/`) -- performance data collection may reference generator params

See `.claude/rules/generator/cross_module_impact.md` for the detailed dependency map.

### Rule 3: Update Documentation

After completing a feature or adding a new user-facing parameter, update the
relevant docs under `docs/`. Skip doc updates for small bug fixes or internal
refactors that don't change user-visible behavior.

| Change Type | Update |
|---|---|
| Pipeline/rendering logic | `docs/generator_overview.md` |
| New CLI parameter | `docs/cli_user_guide.md` |
| Model config changes | `docs/add_a_new_model.md` |
| K8s/deployment changes | `docs/dynamo_deployment_guide.md` |
| New backend version | `docs/support-matrix/` |
| Tuning parameters | `docs/advanced_tuning.md` |

If no existing doc covers the change, note this to the developer and suggest where
documentation should be added.

## Task Routing

Read the appropriate reference file BEFORE starting work:

| Task Type | Reference File |
|---|---|
| Adding/modifying Jinja2 templates | `.claude/rules/generator/template_authoring.md` |
| Writing/modifying .rule files | `.claude/rules/generator/rule_authoring.md` |
| Modifying deployment_config.yaml or backend_config_mapping.yaml | `.claude/rules/generator/config_schema.md` |
| Debugging generator output bugs | `.claude/rules/generator/debugging.md` |
| Writing tests for generator changes | `.claude/rules/generator/testing.md` |
| Adding a new backend engine version | `.claude/rules/generator/new_backend_version.md` |
| Understanding backend-specific constraints | `.claude/rules/generator/guard_rails.md` |
| Checking cross-module dependencies | `.claude/rules/generator/cross_module_impact.md` |

## Cross-Backend Consistency

The generator supports 3 backends: **TRT-LLM**, **vLLM**, **SGLang**.
Each has its own templates, rules, and parameter mappings.

When modifying any backend-specific file, ALWAYS check the other two backends:

1. Does the same parameter/rule/template block exist in the other backends?
2. If yes, does it need the same change?
3. If the change is intentionally different across backends, document WHY.

The most common generator bug pattern is fixing something in one backend and
forgetting the others (PR #579: MoE TP rule missing from SGLang).

## Versioned Template Discipline

- Each backend has versioned templates (e.g., `cli_args.0.16.0.j2`)
- The renderer picks the HIGHEST version <= requested version (floor match)
- When fixing or updating versioned templates, check
  `src/aiconfigurator/generator/config/backend_version_matrix.yaml` to identify
  active versions. Generally only modify templates for the **latest 5 Dynamo versions**
  and their associated backend versions. If the change requires touching more than 5
  versions, confirm with the developer before proceeding.
- To support a new version: copy the closest prior template, modify the copy
- Only create a new version template when the backend CLI interface actually changed

## Quick Validation

After any generator change:

```bash
# Run existing unit tests
pytest tests/unit/generator/ -v

# Generate a test config and inspect output
python -m aiconfigurator generate --model <model> --system <system> --backend <backend> -o /tmp/test_output

# Run generator validator (if backend images available)
python tools/generator_validator/validator.py --backend <backend> --path /tmp/test_output/
```

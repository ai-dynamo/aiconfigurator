---
name: aiconfigurator-development
description: Use when an agent is modifying AIConfigurator source code, tests, model support, performance database logic, CLI behavior, or generator behavior in the repository.
---

# AIConfigurator Development

Use this skill when changing the AIC repository. Keep changes narrow and follow
local patterns before adding new abstractions.

## Default Workflow

1. Inspect the target area and nearby tests.
2. Make the smallest code change that satisfies the request.
3. Add or update focused tests when behavior changes.
4. Run focused pytest and ruff before finalizing.
5. Keep generated data, collector artifacts, and unrelated branch work out of the
   commit unless explicitly requested.

## Guardrails

- Check `git status --short --branch` before edits and before committing.
- Do not revert unrelated user changes.
- Do not broaden validation or fallback behavior globally unless the request
  explicitly needs it.
- For model support, keep parser, model, ops, perf database, task validation, and
  tests aligned.
- Before any edit under `src/aiconfigurator/generator/**`, read
  `.claude/rules/generator-development.md`.

## Load References Only When Needed

- Repo structure: `aiconfigurator agent development --ref repo-layout`
- Test/lint commands: `aiconfigurator agent development --ref testing`
- Model support checklist: `aiconfigurator agent development --ref model-support`
- Generator-specific rules: `aiconfigurator agent development --ref generator`
- PR readiness: `aiconfigurator agent development --ref pr-checklist`

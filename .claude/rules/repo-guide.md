# Repo Guide (always loaded — everything else loads by path)

This is the only always-injected rule file. Detailed rules live in
`.claude/rules/<module>/` and auto-load via `paths:` frontmatter when a task
touches that module. Do not add new always-on rules here without human
approval; new rule files MUST carry `paths:` frontmatter.

## Module map — read this before assuming what a word means

| Path | What it is |
|---|---|
| `aic-core/src/aiconfigurator_core/sdk/` | performance modeling core (perf DB, interpolation, models) |
| `src/aiconfigurator/sdk/` | upper-layer orchestration and legacy core import compatibility |
| `src/aiconfigurator/generator/` | **the "generator"**: renders deployment configs (cli_args, k8s manifests, engine YAML) from task results |
| `collector/` | GPU perf data collection (standalone; NOT part of the wheel runtime) |
| `tools/support_matrix/` | daily end-to-end support matrix generation/compare |
| `aic-core/rust/aiconfigurator-core/` | Rust port of modeling operators |

Disambiguation: `collector/case_generator.py` expands collection test cases —
it has NOTHING to do with `src/aiconfigurator/generator/`. Do not apply
generator-module rules to it, and do not drift into deployment-config topics
unless the task actually targets `src/aiconfigurator/generator/`.

## Governed areas

- Editing `src/aiconfigurator/generator/**` → generator rules auto-load; entry
  point `.claude/rules/generator-development.md`.
- Editing `collector/**` → collector rules auto-load; read
  `.claude/rules/collector/layer_permissions.md` AND `failure_handling.md`;
  for case YAML work also `case_authoring.md`.

## Reviews are governed too

These rules bind reviewers, not just authors. When reviewing a diff (or a
PR) that touches a governed area, READ that area's rule files first — the
path-based auto-loading may not fire for read-only review sessions — and
review the change against them. A change that violates a rule is a review
finding even when the code works.

## Cross-cutting hard rules (apply to every task)

1. **A task stays in its module.** Collector tasks touch `collector/` (+ its
   tests) only; generator tasks touch `src/aiconfigurator/generator/` (+ its
   tests) only; SDK tasks do not reach into either. Cross-module contract
   changes (e.g. a new perf-data column and its SDK consumer) require explicit
   human approval and are never done "while you're at it".
2. **Rule files are human-owned policy.** Propose changes; do not edit them as
   a side effect of a task.
3. **Policy lives once, in `.claude/rules/`.** `.claude/skills/` and
   `AGENTS.md` are procedural runbooks and pointers for agent runtimes that do
   not auto-load these rules; they reference rule files, never restate policy.
   When a skill and a rule conflict, the rule wins.

---
name: aic-version-slots
description: Use when adding perf data for a new framework version, deciding which backend_version to query or pin in tests, bumping the maintained (current/previous) versions, retiring old version directories, or hitting "is not a queryable version" errors from the version-slot gate.
---

# AIC Queryable-Version Slots

## The model in one paragraph

Each (system, framework) exposes at most **three queryable versions**,
resolved through `aic-core/src/aiconfigurator_core/systems/query_versions.yaml`:
**current** (newest maintainer-completed full upgrade; authored),
**previous** (the current before it; authored, may be empty), and
**next** (derived automatically: the highest DATA-BACKED version newer than
current anywhere in the fleet — single-op development drops land there).
The literal aliases `current` / `previous` / `next` are accepted anywhere a
version is requested. Every other version fails loudly. Data directories
outside the slots are **fill sources, not versions**: they keep serving
queries through backward fill and the cross-backend whitelist, but have no
query entry of their own.

## As a data contributor (supporting a new model)

1. Collect your op's data under its real version directory
   (`data/<system>/<family>/<backend>/<version>/`) with full provenance —
   nothing else needed. If your version is newer than current, it becomes
   the fleet-wide `next` automatically; ops your version lacks are served
   by backward fill from current (never forward).
2. Do NOT create reuse.yaml markers to make old versions queryable, and do
   not add per-system version exceptions — frozen baselines (a100, b60)
   are the only authored overrides.
3. Deleting a retired version is a plain directory deletion, but check the
   keep-list first: comm (multi-node curves), wideep, sole-source tables
   (a table no slot version holds — e.g. kda, raw MLA), dsa small-heads
   holders, and rust-test fixture coordinates stay.

## As a maintainer (systematic upgrade)

Bump `current` in query_versions.yaml (move the old current to `previous`)
in the same PR that lands the recollected data. The PR's data diff is the
review surface: families with new directories were recollected; families
without ride backward fill (an accepted approximation). When the previous
generation's data is retired later, delete the directories — no markers.

## In tests

- Product-surface tests (CLI, task, webapp, support matrix) pin slot
  versions or the aliases. If a slot bump breaks such a test, re-anchor or
  delete the test — never bypass the gate.
- Loader-level tests that intentionally address a DATA COORDINATE (strict
  provenance sidecars, sole-source tables like wideep 0.5.6.post2, the
  power-measured 0.22 tables, deliberate data-gap coordinates) pass
  `allow_unlisted_version=True` with a comment saying which coordinate and
  why. That flag is a declaration, not an escape.
- `AIC_ALLOW_UNLISTED_VERSIONS=1` exists only as a transition escape for
  legacy suites and is being retired; do not add new uses.

## Error you will meet

`ValueError: <backend>/<version> is not a queryable version on <system>;
available: {...}` — you asked for a non-slot version. Fix the request to a
slot/alias; only add `allow_unlisted_version=True` when you are genuinely
addressing raw data coordinates.

Policy authority: the version-slot section of
`docs/perf_database/collector-v3-op-centric-design.md` and
`.claude/rules/` own the rules; this skill is the procedural summary.

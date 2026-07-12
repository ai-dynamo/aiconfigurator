---
name: aic-codeowners
description: Use when working with aiconfigurator's generated CODEOWNERS - finding out who reviews a change, fixing a failing codeowners CI check, changing review routing, or granting an external contributor area-scoped ownership. Trigger when the codeowners check fails on a PR, a new directory is unclaimed, someone asks who reviews a path or PR, or review routing needs to change.
---

# AIC CODEOWNERS Operations

The root `CODEOWNERS` is a build artifact generated from
`.github/codeowners/areas.yaml` (one entry per area mapping path globs to a
GitHub team). Never hand-edit `CODEOWNERS` - CI regenerates it and fails on
any drift. Every change goes through `areas.yaml` (or
`external_contributors.yaml`) followed by regeneration.

## Flow 1: Who Reviews This Change?

```bash
# owners of your working tree's changed files (union, as GitHub will request)
python .github/codeowners/who_owns.py --codeowners CODEOWNERS --changed

# owners of specific paths
python .github/codeowners/who_owns.py --codeowners CODEOWNERS <path> [<path> ...]
```

Add `--people` to expand each team to its member logins (org members with an
authenticated `gh` only; GitHub hides team membership from non-members).

## Flow 2: The `codeowners` CI Check Failed

Read the failing step before changing ownership:

1. **Unit tests (matcher + minimal_cover):** fix the matcher, generator, or test
   failure and rerun the exact test command from the workflow. Do not change
   `areas.yaml` to bypass a unit-test failure.
2. **Validate 100% coverage (strict):** the report prints the total uncovered
   count and at most 15 paths under `catch-all-only sample`. Add the narrowest
   claim or claims under the owning areas' `path_globs` (directory claims end
   with `/`). If the paths form a new subsystem, add an area or an appropriate
   `classify` rule instead. Rerun the strict gate and repeat until
   `catch-all only` is zero.
3. **Regenerate and check for drift:** regenerate from the source files. Commit
   every changed generated artifact with its source: `CODEOWNERS`,
   and `CONTRIBUTORS.md`.

For coverage, routing, removal, or source-file changes, regenerate and verify:

```bash
python .github/codeowners/build_codeowners.py \
    --areas .github/codeowners/areas.yaml --repo . --strict
python .github/codeowners/emit_codeowners.py \
    --areas .github/codeowners/areas.yaml --repo . --out CODEOWNERS
```

Rerun the failing workflow step until it passes. Commit the changed source and
all changed generated artifacts together, signed (`git commit -s`).

Removals fail the DRIFT step instead (deleting a directory never fails
coverage): prune the now-dead glob, run the regeneration commands above, and
commit the deletion, `areas.yaml`, and every generated artifact that changed.
The coverage report lists globs that no longer match any file.

## Flow 3: Change Review Routing

Edit `.github/codeowners/areas.yaml` - move a glob between areas, add a
`shared:` entry (multi-team; any one team's approval satisfies the gate), or
adjust `classify` rules - then regenerate as in Flow 2. Changes to the
policy itself route to aiconfigurator-infra + maintainers (the `CODEOWNERS`
and `.github/` shared lines).

## Flow 4: Grant an External Contributor Area-Scoped Ownership

Add an entry to `.github/codeowners/external_contributors.yaml` (name,
github, level, affiliation, `areas: [<label>]`); regeneration appends the
handle as a co-owner on every line the area's team owns and rebuilds
`CONTRIBUTORS.md`. Commit all three files together.

## Reference

Schema and the last-match-wins model: `.github/codeowners/README.md`. The
gate and drift check run in `.github/workflows/codeowners.yml` on every PR.

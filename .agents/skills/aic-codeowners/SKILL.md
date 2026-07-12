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

## Flow 1: Who reviews this change?

```bash
# owners of your working tree's changed files (union, as GitHub will request)
python .github/codeowners/who_owns.py --codeowners CODEOWNERS --changed

# owners of specific paths
python .github/codeowners/who_owns.py --codeowners CODEOWNERS <path> [<path> ...]
```

Add `--people` to expand each team to its member logins (org members with an
authenticated `gh` only; GitHub hides team membership from non-members).

## Flow 2: The `codeowners` CI check failed

The coverage gate is doing its job: the PR adds at least one file that no
area claims.

1. Read the failing job log; the gate prints the exact uncovered files under
   `catch-all-only sample`.
2. Add ONE line to `.github/codeowners/areas.yaml` under the owning area's
   `path_globs` (directory claims end with `/`).
3. Regenerate and verify:
   ```bash
   python .github/codeowners/build_codeowners.py \
       --areas .github/codeowners/areas.yaml --repo . --strict
   python .github/codeowners/emit_codeowners.py \
       --areas .github/codeowners/areas.yaml --repo . --out CODEOWNERS
   ```
4. Commit `areas.yaml` and `CODEOWNERS` together, signed (`git commit -s`).

Removals fail the DRIFT step instead (deleting a directory never fails
coverage): run step 3 and commit both files, and prune the now-dead glob -
the coverage report lists globs that no longer match any file.

## Flow 3: Change review routing

Edit `.github/codeowners/areas.yaml` - move a glob between areas, add a
`shared:` entry (multi-team; any one team's approval satisfies the gate), or
adjust `classify` rules - then regenerate as in Flow 2. Changes to the
policy itself route to aiconfigurator-infra + maintainers (the `CODEOWNERS`
shared line).

## Flow 4: Grant an external contributor area-scoped ownership

Add an entry to `.github/codeowners/external_contributors.yaml` (name,
github, level, affiliation, `areas: [<label>]`); regeneration appends the
handle as a co-owner on every line the area's team owns and rebuilds
`CONTRIBUTORS.md`. Commit all three files together.

## Reference

Schema and the last-match-wins model: `.github/codeowners/README.md`. The
gate and drift check run in `.github/workflows/codeowners.yml` on every PR.

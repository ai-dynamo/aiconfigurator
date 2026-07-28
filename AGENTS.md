# AGENTS

This file adds an explicit guard for generator rule edits.

## Required First Step

Before making any change under:
- `src/aiconfigurator/generator/**`

MUST read:
- `.claude/rules/generator-development.md`

## Required Collector First Step

Before making any change under `collector/**` MUST read:
- `.claude/rules/collector/layer_permissions.md` (layer permission table,
  module boundary, dispatch-vs-skip rule)
- `.claude/rules/collector/failure_handling.md` (observe-don't-predict
  doctrine, escalation decision tree)
- For case YAML work: `.claude/rules/collector/case_authoring.md`

For adding a new Collector operation, additionally follow
`.claude/skills/aic-collector-op-development/SKILL.md` (consumer-contract,
case-identity, deduplication, and validation gates). Skills are procedural
runbooks; if a skill and a `.claude/rules/` file ever disagree, the rule
file wins.

## Cursor Cloud specific instructions

### Project overview

AIConfigurator is a Python CLI/SDK tool for optimizing LLM inference deployment configurations. See `README.md` for full details.

### Environment setup

Dependencies are managed via `uv` with a `uv.lock` lockfile. The virtual environment lives at `.venv/`. All commands below assume `.venv/bin/` is on PATH or you prefix with `.venv/bin/`.

- **Install/refresh deps:** `python3 -m uv sync --extra dev`
- **Performance data:** Current op profiles are parquet files under
  `aic-core/src/aiconfigurator_core/systems/data/<system>/<family>/<backend>/<version>/`
  and are checked in directly. Legacy `*.txt` perf files, when present, use
  Git LFS; run `git lfs pull` only when working with those legacy assets.

### Lint / Test / Run

- **Lint:** `ruff check .` and `ruff format --check .` (see `DEVELOPMENT.md`)
- **Unit tests:** `pytest -m unit` (868+ tests; no external deps or LFS data needed)
- **Build tests (PR subset):** `pytest -m "unit or build"` (requires LFS data for the `build`-marked tests)
- **CLI:** `aiconfigurator cli generate --model-path Qwen/Qwen3-32B-FP8 --total-gpus 8 --system h200_sxm` (works without LFS data)
### Known environment caveats

1. **Legacy LFS data:** `github-cloud.githubusercontent.com` may be blocked by
   network egress restrictions. If `git lfs pull` fails, tests that explicitly
   exercise legacy text assets may fail; current parquet-backed workflows do
   not depend on those legacy files.
2. **TTY tests:** 4 tests in `tests/unit/cli/test_plain_output.py` may fail because the agent runs in a non-TTY environment.
3. **Rust tests:** `tests/unit/sdk/test_rust_engine_step.py` requires `cargo` with network access to `crates.io`. It will fail if that domain is blocked.
4. **macOS pytest-timeout crash dialogs:** The `timeout = 120` setting in `pytest.ini` uses SIGALRM by default, which triggers "Python unexpectedly quit" crash reporter popups on macOS. Pass `-p no:timeout` to disable it locally: `.venv/bin/pytest -m unit -p no:timeout`
5. **torch-dependent tests:** `tests/unit/sdk/database/test_moe_dispatch.py` requires `torch` (not installed in the default dev venv). Ignore it with `--ignore=tests/unit/sdk/database/test_moe_dispatch.py`.

## CODEOWNERS

The root `CODEOWNERS` is generated from `.github/codeowners/areas.yaml` - never
hand-edit it; CI fails on drift. Repository rules must require the `codeowners`
check to make failures merge-blocking. If the check fails on a new directory,
claim it in `areas.yaml`, regenerate with
`emit_codeowners.py`, and commit every changed source and generated artifact
together. The `aic-codeowners` skill covers all flows (who reviews a change,
gate failures, routing changes, external contributor grants).

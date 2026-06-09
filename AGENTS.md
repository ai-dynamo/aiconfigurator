# AGENTS

This file adds an explicit guard for generator rule edits.

## Required First Step

Before making any change under:
- `src/aiconfigurator/generator/**`

MUST read:
- `.claude/rules/generator-development.md`

## Cursor Cloud specific instructions

### Project overview

AIConfigurator is a Python CLI/SDK/webapp tool for optimizing LLM inference deployment configurations. See `README.md` for full details.

### Environment setup

Dependencies are managed via `uv` with a `uv.lock` lockfile. The virtual environment lives at `.venv/`. All commands below assume `.venv/bin/` is on PATH or you prefix with `.venv/bin/`.

- **Install/refresh deps:** `python3 -m uv sync --extra dev --extra webapp`
- **Git LFS:** The performance database files under `src/aiconfigurator/systems/data/**/*.txt` are tracked with Git LFS. Run `git lfs pull` after cloning. If LFS pull fails (e.g., `github-cloud.githubusercontent.com` is blocked), the CLI `generate` and `support` modes still work. The `default` mode and the webapp require real LFS data.

### Lint / Test / Run

- **Lint:** `ruff check .` and `ruff format --check .` (see `DEVELOPMENT.md`)
- **Unit tests:** `pytest -m unit` (868+ tests; no external deps or LFS data needed)
- **Build tests (PR subset):** `pytest -m "unit or build"` (requires LFS data for the `build`-marked tests)
- **CLI:** `aiconfigurator cli generate --model-path Qwen/Qwen3-32B-FP8 --total-gpus 8 --system h200_sxm` (works without LFS data)
- **Webapp:** `aiconfigurator webapp` (requires LFS data; serves on port 7860)

### Known Cloud Agent environment caveats

1. **LFS data:** `github-cloud.githubusercontent.com` may be blocked by network egress restrictions. If `git lfs pull` fails, unit tests and CLI `generate`/`support` modes still work. The `default` mode, webapp, and `build`-marked tests will fail.
2. **TTY tests:** 4 tests in `tests/unit/cli/test_plain_output.py` may fail because the agent runs in a non-TTY environment.
3. **Rust tests:** `tests/unit/sdk/test_rust_engine_step.py` runs its mocked cases without the Rust core; the real-extension cases are skipped unless the `aiconfigurator-core` PyO3 wheel is installed (`pip install -e ".[rust]"` or `maturin develop --release`).

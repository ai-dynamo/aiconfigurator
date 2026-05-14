# AGENTS

This file adds an explicit guard for generator rule edits.

## Required First Step

Before making any change under:
- `aiconfigurator/src/aiconfigurator/generator/**`

MUST read:
- `aiconfigurator/.claude/rules/generator-development.md`

## Cursor Cloud specific instructions

### Project overview
`aiconfigurator` is a single Python package (no microservices, no databases). See `README.md` and `DEVELOPMENT.md` for full setup and usage docs.

### Environment activation
The dev virtualenv lives at `/workspace/.venv`. Activate it before any command:
```bash
source /workspace/.venv/bin/activate
```

### Lint / Test / Build / Run
- **Lint:** `ruff check .` and `ruff format --check .` (config in `pyproject.toml`)
- **Unit tests:** `pytest -m unit` (~870 tests, ~45s)
- **PR-level tests:** `pytest -m "unit or build"` (includes a small e2e subset)
- **CLI generate (no LFS needed):** `aiconfigurator cli generate --model-path Qwen/Qwen3-32B-FP8 --total-gpus 8 --system h200_sxm`
- **CLI default (needs LFS data):** `aiconfigurator cli default --model Qwen/Qwen3-32B-FP8 --total-gpus 32 --system h200_sxm`

### Git LFS data
~553 performance database files under `src/aiconfigurator/systems/data/` are stored via Git LFS. Without `git lfs pull`, the CLI `default`/`exp` modes (SILICON database) and many `build`-marked e2e tests will fail. The `generate` mode and all `unit`-marked tests work without LFS data. In Cloud Agent VMs, the `github-cloud.githubusercontent.com` domain used by LFS may be blocked by egress restrictions.

### Known test-environment quirks
- 4 tests in `tests/unit/cli/test_plain_output.py` fail because the CI terminal lacks a TTY (color detection). This is not a code bug.
- `tests/unit/sdk/test_rust_engine_step.py::test_ctypes_wrapper_calls_real_rust_core` requires the Rust toolchain (`cargo`). Skip if Rust is not installed.

### Rust core (optional)
The Rust native latency estimator at `rust/aiconfigurator-core/` is optional. Build with `cargo build --manifest-path rust/aiconfigurator-core/Cargo.toml` and set `AICONFIGURATOR_RUST_CORE_LIB` to the resulting `.so` path.
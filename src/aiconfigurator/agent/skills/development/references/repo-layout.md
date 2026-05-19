# Repo Layout

- `src/aiconfigurator/main.py`: top-level `aiconfigurator` entry point.
- `src/aiconfigurator/cli/`: CLI modes, reporting, and Python CLI API.
- `src/aiconfigurator/sdk/`: model parsing, task config, perf database, ops, and backends.
- `src/aiconfigurator/generator/`: deployment config generation. Read the generator rule before editing.
- `src/aiconfigurator/systems/`: built-in system specs, support matrix, and perf data paths.
- `src/aiconfigurator/model_configs/`: bundled Hugging Face config cache.
- `collector/`: data collection scripts; keep collector-only work out of model-support PRs unless requested.
- `tests/unit/`: focused unit coverage for CLI, SDK, generator, and collectors.
- `tests/e2e/`: CLI and integration workflows.

Use `rg` and `rg --files` for navigation. Avoid loading large perf data files unless needed.

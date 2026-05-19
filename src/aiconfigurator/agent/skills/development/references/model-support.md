# Model Support Checklist

When adding or changing model support, inspect the existing family closest to
the target model and keep contracts aligned across these areas:

- `sdk/common.py`: architecture-to-family mapping, default models, dataclasses, enums, perf filenames.
- `sdk/utils.py`: HF config parsing, local config handling, quant/default inference.
- `sdk/models.py`: model construction, op graph, memory and KV cache accounting.
- `sdk/operations.py`: operation metadata, weights, query arguments, and scale factors.
- `sdk/perf_database.py`: loaders, support extraction, silicon/SOL/HYBRID query behavior.
- `sdk/task.py`: task config defaults, validation, backend/system/quant support rules.
- `webapp/events/`: UI option exposure when model/backend choices change.
- `tests/unit/sdk/`: parser, model config, task validation, perf lookup, and end-to-end focused tests.

Prefer explicit rejection over silent fallback when a backend cannot model the
required operation. Keep `SILICON`, `HYBRID`, `EMPIRICAL`, and `SOL` semantics
clear in tests and user-facing errors.

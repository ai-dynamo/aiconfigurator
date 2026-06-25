<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## CLI E2E tests

These tests run the installed `aiconfigurator` CLI via `subprocess`.

### Run

```bash
python3 -m pytest tests/e2e/cli
```

### Useful subsets

```bash
# GitHub build workflow subset (fast/stable; also a good quick sanity subset)
python3 -m pytest tests/e2e/cli -m build

# Compatibility matrix (sweep)
python3 -m pytest tests/e2e/cli -m sweep
```

### Real Spica Trace Sweep

`test_cli_default_trace_path_real.py` is skipped by default because it requires
Spica plus compatible Dynamo replay bindings on `PYTHONPATH`, and it runs the
real default Spica smart sweep (`max_rounds=3`, `parallel_evals=16`).

```bash
AIC_RUN_SPICA_TRACE_E2E=true \
AIC_SPICA_TRACE_PATH=/path/to/mooncake_tiny.jsonl \
PYTHONPATH=/path/to/spica/src:/path/to/dynamo/components/src:/path/to/dynamo/lib/bindings/python/src \
python3 -m pytest tests/e2e/cli/test_cli_default_trace_path_real.py -q
```


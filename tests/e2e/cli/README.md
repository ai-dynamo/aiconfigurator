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



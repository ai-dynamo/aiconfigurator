# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the (out-of-repo) baseline data root.

The frozen + pre-cutover baselines are developer-only migration fixtures for the
generator refactor's byte/semantic-equivalence gates. They are intentionally NOT
committed to the repo (large, dev-only, of no interest to users). By default they
live under ``aiconfigurator-dev/refactor_baselines`` (a sibling of the repo);
override with the ``AIC_BASELINE_DIR`` env var.

Tests that consume them SKIP when the data is absent (CI, a fresh checkout, or any
machine without the fixtures) -- they only run where a developer has the data.
"""

from __future__ import annotations

import os
import pathlib

BASELINE_ROOT = pathlib.Path(
    os.environ.get("AIC_BASELINE_DIR") or (pathlib.Path(__file__).resolve().parents[3] / "refactor_baselines")
)
FROZEN_DIR = BASELINE_ROOT / "frozen"
REF_DIR = BASELINE_ROOT / "precutover_ref"

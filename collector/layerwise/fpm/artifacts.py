# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run-directory layout helpers for FPM collection."""

from __future__ import annotations

from pathlib import Path

from collector.layerwise.common.paths import default_run_dir
from collector.layerwise.fpm.datapoint_generator import FpmCase


def resolve_run_dir(raw_run_dir: Path | None, model: str) -> Path:
    """Return the user-supplied or default FPM run directory."""
    return (raw_run_dir or default_run_dir("fpm_vllm", model=model)).expanduser().resolve()


def case_run_dir(root: Path, case: FpmCase, *, multiple_cases: bool) -> Path:
    """Return the concrete run directory for a single FPM case."""
    return root / case.label if multiple_cases else root

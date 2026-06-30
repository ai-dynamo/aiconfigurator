# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run-directory helpers shared by layerwise and FPM collectors."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ARTIFACT_ROOT = Path(".tmp/layerwise-artifacts/runs")


def slugify(value: str) -> str:
    """Return a filesystem-safe slug for model names and run labels."""
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("_")


def timestamp_utc() -> str:
    """Return a compact UTC timestamp for default run directories."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def default_run_dir(prefix: str, *, model: str | None = None, root: Path | None = None) -> Path:
    """Build the default timestamped run directory path."""
    base = root or DEFAULT_ARTIFACT_ROOT
    parts = [prefix]
    if model:
        parts.append(slugify(model))
    parts.append(timestamp_utc())
    return base / "_".join(parts)

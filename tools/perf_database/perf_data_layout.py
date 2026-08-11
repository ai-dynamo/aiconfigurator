# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared traversal rules for the perf-database data tree.

This module is intentionally side-effect free so perf-database tools can share
layout knowledge without importing an executable report or manifest generator.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

# Columns that never participate in a perf-table shape key.
META_COLUMNS = frozenset({"framework", "version", "device", "op_name", "kernel_source"})

# Files to skip entirely (markers, already-shared layers, irregular formats).
# reuse.yaml/collection_meta.yaml never match the *.parquet/*.txt glob below;
# listing them is defensive and documents the intended boundary.
SKIP_FILE_BASENAMES = frozenset({"INCOMPLETE.txt", "reuse.yaml", "collection_meta.yaml"})

# Framework-agnostic communication backends are not cross-backend reuse inputs.
SKIP_BACKEND_DIRS = frozenset({"nccl", "oneccl"})

# First-level backend dirs in the legacy <system>/<backend>/<version> layout.
# Keep this set textually identical to the canonical _KNOWN_BACKEND_DIRS in
# aic-core/src/aiconfigurator_core/sdk/operations/base.py minus
# SKIP_BACKEND_DIRS (consumer backends only; no comm pseudo-backends).
LEGACY_BACKEND_DIRS = frozenset({"trtllm", "sglang", "vllm"})


def iter_backend_dirs(system_dir: Path) -> Iterable[tuple[str, Path]]:
    """Yield ``(backend, path)`` pairs from legacy and family-first layouts."""
    for entry in sorted(system_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith(".") or entry.name in SKIP_BACKEND_DIRS:
            continue
        if entry.name in LEGACY_BACKEND_DIRS:
            yield entry.name, entry
            continue
        for backend_dir in sorted(entry.iterdir()):
            if not backend_dir.is_dir() or backend_dir.name in SKIP_BACKEND_DIRS:
                continue
            yield backend_dir.name, backend_dir


def iter_data_files(
    data_root: Path,
    op_files: frozenset[str] | None = None,
) -> Iterable[tuple[str, str, str, Path]]:
    """Yield ``(system, backend, version, path)`` for every perf data table.

    ``op_files``, when given, restricts the walk to those table basenames.
    """
    for system_dir in sorted(data_root.iterdir()):
        if not system_dir.is_dir():
            continue
        for backend, backend_dir in iter_backend_dirs(system_dir):
            for version_dir in sorted(backend_dir.iterdir()):
                if not version_dir.is_dir():
                    continue
                paths = sorted([*version_dir.glob("*.parquet"), *version_dir.glob("*.txt")])
                for path in paths:
                    if path.name in SKIP_FILE_BASENAMES:
                        continue
                    if op_files is not None and path.name not in op_files:
                        continue
                    yield system_dir.name, backend, version_dir.name, path

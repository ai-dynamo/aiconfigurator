# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared data types for collector registries."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class VersionRoute:
    """A (min_version, module_path) pair for version-based module routing.

    ``min_version`` is a PEP 440 version string. The resolver picks the first
    ``VersionRoute`` whose ``min_version`` is <= the runtime version (entries
    must be listed in descending order).
    """

    min_version: str
    module: str


@dataclass(frozen=True, slots=True)
class OpEntry:
    """One operation in a collector registry.

    Exactly one of ``module`` (unversioned) or ``versions`` (versioned) must be
    provided.  This invariant is validated at construction time.
    """

    op: str
    get_func: str
    run_func: str
    module: str | None = None
    versions: tuple[VersionRoute, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.module and not self.versions:
            raise ValueError(f"OpEntry '{self.op}': must specify 'module' or 'versions'")
        if self.module and self.versions:
            raise ValueError(f"OpEntry '{self.op}': cannot specify both 'module' and 'versions'")

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Resolve version-specific collector modules from registry entries.

Given a registry entry and a runtime framework version, returns the
appropriate module path. For unversioned entries, returns the module
directly. For versioned entries, picks the highest min_version that
does not exceed the runtime version.
"""

from __future__ import annotations

import re

from packaging.specifiers import InvalidSpecifier, Specifier
from packaging.version import InvalidVersion, Version

from collector.registry_types import OpEntry

_FRAMEWORK_PREFIX_RE = re.compile(r"^[a-zA-Z_]+")


def _strip_local_metadata(v: str) -> str:
    """Drop local build metadata suffix to preserve legacy collector behavior."""
    return v.split("+", 1)[0].strip()


def _normalize_version(v: str) -> Version:
    """Parse and normalize a version with ``packaging.version.Version``.

    Local metadata suffixes (for example ``+cu124``) are ignored to keep the
    historical collector behavior where build tags do not affect routing.
    Invalid version strings are treated as ``0``.
    """
    normalized = _strip_local_metadata(v)
    if not normalized:
        return Version("0")
    try:
        return Version(normalized)
    except InvalidVersion:
        return Version("0")


def _parse_compat_specifier(compat_str: str) -> list[Specifier]:
    """Parse ``<framework><constraints>`` into validated specifier clauses."""
    spec = _FRAMEWORK_PREFIX_RE.sub("", compat_str, count=1).strip()
    if not spec:
        raise ValueError(f"Invalid __compat__ {compat_str!r}: missing version constraints")

    clauses = [clause.strip() for clause in spec.split(",") if clause.strip()]
    if not clauses:
        raise ValueError(f"Invalid __compat__ {compat_str!r}: no valid constraint clauses found")

    parsed: list[Specifier] = []
    for clause in clauses:
        try:
            parsed.append(Specifier(clause))
        except InvalidSpecifier as e:
            raise ValueError(f"Invalid __compat__ {compat_str!r}: {e}") from e
    return parsed


def _check_compat(compat_str: str, runtime_version: str) -> bool:
    """Check if runtime_version satisfies a __compat__ specifier.

    The specifier format is: ``<framework><constraints>``
    where constraints are comma-separated comparisons.
    Examples: "trtllm>=1.1.0", "trtllm>=0.21.0,<1.1.0"

    The framework prefix is stripped; only version constraints are evaluated.

    Raises:
        ValueError: If compat_str does not contain valid constraint clauses.
    """
    specifiers = _parse_compat_specifier(compat_str)
    rv = _normalize_version(runtime_version)

    for spec in specifiers:
        op = spec.operator
        ver = _normalize_version(spec.version)
        if op == ">=":
            ok = rv >= ver
        elif op == "<=":
            ok = rv <= ver
        elif op == ">":
            ok = rv > ver
        elif op == "<":
            ok = rv < ver
        elif op == "==":
            ok = rv == ver
        elif op == "!=":
            ok = rv != ver
        else:
            raise ValueError(
                f"Invalid __compat__ {compat_str!r}: unsupported operator {op!r}. Use one of >=, <=, >, <, ==, !="
            )
        if not ok:
            return False
    return True


def resolve_module(entry: OpEntry, runtime_version: str) -> str | None:
    """Return the collector module path for a registry entry.

    Args:
        entry: An :class:`OpEntry`. Uses ``module`` directly for unversioned
               entries, or walks ``versions`` (descending by min_version) for
               versioned entries.
        runtime_version: The framework version string detected at runtime.

    Returns:
        Module path string, or None if no version matches.
    """
    if not entry.versions:
        return entry.module

    rv = _normalize_version(runtime_version)
    for route in entry.versions:
        if rv >= _normalize_version(route.min_version):
            return route.module
    return None


def build_collections(
    registry: list[OpEntry],
    backend_name: str,
    runtime_version: str,
    ops: list[str] | None = None,
    *,
    logger=None,
) -> list[dict]:
    """Build the collections list from a registry.

    Args:
        registry: The backend's REGISTRY list of :class:`OpEntry`.
        backend_name: e.g. "trtllm", "vllm", "sglang".
        runtime_version: Framework version string.
        ops: Optional filter — only include these op types.
        logger: Optional logger for warnings.

    Returns:
        List of collection dicts ready for collect_ops().
    """
    collections = []
    for entry in registry:
        if ops and entry.op not in ops:
            continue

        module = resolve_module(entry, runtime_version)
        if module is None:
            if logger:
                logger.warning(f"Skipping {backend_name}.{entry.op} — no collector for v{runtime_version}")
            continue

        collections.append(
            {
                "name": backend_name,
                "type": entry.op,
                "module": module,
                "get_func": entry.get_func,
                "run_func": entry.run_func,
            }
        )

    return collections

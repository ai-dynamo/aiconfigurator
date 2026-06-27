# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protected legacy coverage checks for fully populated collector plans."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet

from collector.planner.coverage import CoverageManifest, assert_legacy_subset, load
from collector.planner.models import PlanContext, PopulationResult
from collector.planner.physical_keys import PhysicalRowKey
from collector.planner.registry import get_schema

DEFAULT_MANIFEST_ROOT = Path(__file__).resolve().parent / "manifests" / "collector_v1"


def _framework_scope_matches(actual: str | None, declared: str) -> bool:
    """Match an exact framework version or an explicit PEP 440 range."""

    if actual is None:
        return False
    if not declared.startswith(("<", ">", "=", "!", "~")):
        return actual == declared
    try:
        return actual in SpecifierSet(declared)
    except InvalidSpecifier as exc:
        raise ValueError(f"invalid framework version scope in coverage manifest: {declared!r}") from exc


def protected_manifest_path(
    backend: str,
    op: str,
    *,
    manifest_root: str | Path = DEFAULT_MANIFEST_ROOT,
) -> Path:
    return Path(manifest_root) / backend / f"{op}.jsonl.gz"


def check_protected_coverage(
    population: PopulationResult,
    context: PlanContext,
    *,
    manifest_root: str | Path = DEFAULT_MANIFEST_ROOT,
) -> dict[str, Any]:
    """Assert and summarize a scoped legacy physical-key subset.

    A manifest only applies when backend, framework version, and hardware
    capability match.  Targeted plans must not call this function: their goal
    is model-exact healing, while the protected contract applies to full/raw
    population.
    """

    path = protected_manifest_path(context.backend, context.op, manifest_root=manifest_root)
    if not path.exists():
        return {"status": "no_manifest"}

    legacy = load(path)
    scope_mismatches: dict[str, dict[str, object | None]] = {}
    for field_name, actual, expected, matches in (
        ("backend", context.backend, legacy.header.backend_variant, context.backend == legacy.header.backend_variant),
        (
            "framework_version",
            context.framework_version,
            legacy.header.framework_version,
            _framework_scope_matches(context.framework_version, legacy.header.framework_version),
        ),
        ("gpu_type", context.gpu_type, legacy.header.gpu_type, context.gpu_type == legacy.header.gpu_type),
        ("sm_version", context.sm_version, legacy.header.sm_version, context.sm_version == legacy.header.sm_version),
    ):
        if not matches:
            scope_mismatches[field_name] = {"actual": actual, "manifest": expected}
    if scope_mismatches:
        return {
            "status": "out_of_scope",
            "source_git_ref": legacy.header.source_git_ref,
            "scope_mismatches": scope_mismatches,
        }

    schema = get_schema(context.op, backend=context.backend)
    keys: set[PhysicalRowKey] = set()
    for planned_case in population.cases:
        key = schema.invocation_key(planned_case.candidate, context)
        if not isinstance(key, PhysicalRowKey):
            return {"status": "schema_has_no_physical_projection"}
        keys.add(key)

    generated = CoverageManifest(
        header=replace(legacy.header, source_git_ref="working-tree-current"),
        keys=frozenset(keys),
    )
    result = assert_legacy_subset(legacy, generated)
    return {
        "status": "preserved",
        "source_git_ref": legacy.header.source_git_ref,
        "legacy_physical_keys": result.legacy_count,
        "retained_legacy_keys": result.retained_count,
        "added_physical_keys": len(result.added),
        "removed_protected_keys": len(result.missing),
        "generated_physical_keys": result.generated_count,
    }


__all__ = ["DEFAULT_MANIFEST_ROOT", "check_protected_coverage", "protected_manifest_path"]

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collector provenance: authored hash closures + ``collection_meta.yaml`` writer.

Collector V3 makes the collector finalize step the provenance authority
(design: docs/perf_database/collector-v3-op-centric-design.md §5). This module
provides the building blocks:

- ``collector_hash``: a content hash over an AUTHORED closure of files (the
  op's ``collect_*.py`` module, a fixed shared core, and per-module extras
  declared in ``hash_closures.yaml``). Content-based, so it survives rebases.
- ``case_plan_hash``: a hash of the resolved/attested case-id set collected
  at run time (GPU needed to produce it — never recomputed in CI).
- ``write_collection_meta``: renders the design-§5 YAML deterministically.

``load_closures`` fails closed: every registered module and every explicitly
declared standalone collector module MUST have a ``hash_closures.yaml`` entry,
or loading raises ``KeyError``.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from typing import Any

import yaml

# Implicit in every collector_hash closure: files whose content affects every
# op's collected data regardless of which module runs it.
SHARED_CORE: tuple[str, ...] = (
    "collector/helper.py",
    "collector/case_generator.py",
    "collector/model_cases.py",
    "collector/capabilities.py",
    "collector/version_resolver.py",
)

# Sentinel closure-extra token: expands to every collector/cases/models/*.yaml
# file (the shared model-shapes group), sorted, at hash time.
MODEL_CASES_GROUP = "__model_cases__"
_MODEL_CASES_DIR = "collector/cases/models"

# Standalone distributed collectors do not fit collect.py's single-host
# OpEntry executor/checkpoint contract, but their published tables still need
# authored collector_hash closures.
STANDALONE_COLLECTOR_MODULES: frozenset[str] = frozenset(
    {
        "collector.sglang.collect_dsv4_megamoe",
        "collector.wideep.sglang.collect_moe_a2a",
        "collector.network.slurm.collect_trtllm_alltoall",
    }
)

STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"
COLLECTION_META_SCHEMA_VERSION = 1
MULTI_EVENT_COLLECTION_META_SCHEMA_VERSION = 2

_RUNTIME_FIELD_ORDER = ("framework", "version", "image", "image_variant", "image_digest")
_TABLE_FIELD_ORDER = ("collector_ref", "collector_hash", "case_plan_hash", "collected_at", "rows", "status")
_COLLECTION_OPTIONAL_FIELD_ORDER = ("source_campaign_rows", "source_campaign_status", "runtime")


def _registry_lists(registry_module) -> tuple[list, ...]:
    registries = [registry_module.REGISTRY]
    registry_xpu = getattr(registry_module, "REGISTRY_XPU", None)
    if registry_xpu is not None:
        registries.append(registry_xpu)
    return tuple(registries)


def enumerate_registry_modules() -> set[str]:
    """Return every collector module referenced by an OpEntry in any of the five
    registries enumerated by ``framework_manifest._REGISTRY_MODULES`` (sglang,
    trtllm, vllm, wideep_sglang, wideep_trtllm), plus active sibling registries
    such as vLLM's ``REGISTRY_XPU``.
    """
    import importlib

    from collector.framework_manifest import _REGISTRY_MODULES

    modules: set[str] = set()
    for registry_module_path in _REGISTRY_MODULES.values():
        registry_module = importlib.import_module(registry_module_path)
        for registry in _registry_lists(registry_module):
            for entry in registry:
                if entry.module:
                    modules.add(entry.module)
                for route in entry.versions:
                    modules.add(route.module)
    return modules


def enumerate_provenance_modules() -> set[str]:
    """Return every registered or explicitly standalone provenance producer."""
    return enumerate_registry_modules() | set(STANDALONE_COLLECTOR_MODULES)


def load_closures(path: str | Path) -> dict[str, list[str]]:
    """Load ``hash_closures.yaml`` and fail closed on incomplete coverage.

    Every module returned by :func:`enumerate_provenance_modules` MUST appear
    as a key; a module missing its closure entry is a KeyError, not a silent
    empty closure (fail-closed — see collector/hash_closures.yaml header).
    """
    closures_path = Path(path)
    with closures_path.open(encoding="utf-8") as closures_file:
        data = yaml.safe_load(closures_file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{closures_path}: hash closures file must be a mapping at the top level")  # noqa: TRY004

    closures: dict[str, list[str]] = {}
    for module, extras in data.items():
        if not isinstance(module, str):
            raise ValueError(f"{closures_path}: closure keys must be module dotted paths, got {module!r}")  # noqa: TRY004
        extras = extras or []
        if not isinstance(extras, list) or not all(isinstance(item, str) for item in extras):
            raise ValueError(f"{closures_path}: {module} closure must be a list of strings")
        closures[module] = list(extras)

    missing = enumerate_provenance_modules() - closures.keys()
    if missing:
        raise KeyError(
            f"{closures_path}: missing hash_closures.yaml entries for provenance module(s) "
            f"{sorted(missing)} (fail-closed — every provenance producer must declare its "
            "collector_hash closure)"
        )
    return closures


def _expand_closure_files(repo_root: Path, extras: list[str]) -> set[str]:
    files: set[str] = set()
    for extra in extras:
        if extra == MODEL_CASES_GROUP:
            files.update(
                str(model_cases_file.relative_to(repo_root))
                for model_cases_file in sorted((repo_root / _MODEL_CASES_DIR).glob("*.yaml"))
            )
        else:
            files.add(extra)
    return files


def collector_hash(module: str, repo_root: str | Path, closures: dict[str, list[str]]) -> str:
    """Content hash over module file + SHARED_CORE + the module's closure extras.

    Sha256 over sorted (relpath, file-bytes) pairs, "sha256:<hex>" formatted.
    Content-based (never absolute paths or a commit SHA), so it is stable
    across rebases and repo relocation.
    """
    if module not in closures:
        raise KeyError(
            f"{module}: no hash_closures.yaml entry (fail-closed — every collected module must "
            "declare its hash closure in collector/hash_closures.yaml)"
        )
    root = Path(repo_root)
    module_file = module.replace(".", "/") + ".py"
    relpaths = {module_file, *SHARED_CORE, *_expand_closure_files(root, closures[module])}

    digest = hashlib.sha256()
    for relpath in sorted(relpaths):
        content = (root / relpath).read_bytes()
        digest.update(relpath.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def case_plan_hash(case_ids: list[str]) -> str:
    """Hash of the sorted, deduplicated case-id set (the attested expanded plan)."""
    unique_sorted = sorted(set(case_ids))
    digest = hashlib.sha256("\n".join(unique_sorted).encode("utf-8"))
    return f"sha256:{digest.hexdigest()}"


def derive_table_status(*, unresolved_failed_count: int, had_module_failure: bool) -> str:
    """complete unless a ModuleCollectionFailure was recorded for one of the
    table's producing ops.

    Owner decision (tianhaox, 2026-08-08, PR #1486): recorded per-case
    failures do NOT demote a table. failure_handling.md's core doctrine is
    that a classified failure is DATA — deterministic framework limits (OOM
    at sweep extremes, kernel grid caps) land in the failure log by design,
    and demoting the table for them made every honest campaign
    unpublishable. The anti-false-success guarantees are unaffected and live
    elsewhere: a run that dies mid-way never finalizes (parquet without a
    matching ``tables`` entry fails the coverage gate / strict loader), and
    an op that produced zero rows still demotes via ``had_module_failure``.
    ``unresolved_failed_count`` is retained for observability at call sites
    (logged, recorded in error summaries) but no longer affects status.
    """
    del unresolved_failed_count  # observability-only; see docstring
    if had_module_failure:
        return STATUS_PARTIAL
    return STATUS_COMPLETE


def spdx_header() -> str:
    """The repo-standard copyright header, dated to the year of emission
    (the copyright CI check requires the year to cover the file's last commit).
    """
    return (
        f"# SPDX-FileCopyrightText: Copyright (c) {date.today().year} NVIDIA CORPORATION & AFFILIATES."
        " All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n"
        "\n"
    )


def _ordered_runtime(runtime: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(runtime, dict):
        raise ValueError(f"{field} must be a mapping")  # noqa: TRY004
    unknown = sorted(set(runtime) - set(_RUNTIME_FIELD_ORDER))
    if unknown:
        raise ValueError(f"{field} has unsupported field(s): {', '.join(unknown)}")
    for key in ("framework", "version"):
        if not isinstance(runtime.get(key), str) or not runtime[key].strip():
            raise ValueError(f"{field}.{key} must be a non-empty string")
    for key in _RUNTIME_FIELD_ORDER[2:]:
        if key in runtime and (not isinstance(runtime[key], str) or not runtime[key].strip()):
            raise ValueError(f"{field}.{key} must be a non-empty string when provided")
    return {key: runtime[key] for key in _RUNTIME_FIELD_ORDER if key in runtime}


def _ordered_collection_event(event: Any, *, table: str, index: int) -> dict[str, Any]:
    if not isinstance(event, dict):
        raise ValueError(f"{table}.collections[{index}] must be a mapping")  # noqa: TRY004

    missing = [field for field in _TABLE_FIELD_ORDER if field not in event]
    if missing:
        raise ValueError(f"{table}.collections[{index}] missing required field(s): {', '.join(missing)}")

    allowed = {*_TABLE_FIELD_ORDER, *_COLLECTION_OPTIONAL_FIELD_ORDER}
    unknown = sorted(set(event) - allowed)
    if unknown:
        raise ValueError(f"{table}.collections[{index}] has unsupported field(s): {', '.join(unknown)}")

    for field in ("collector_ref", "collector_hash", "case_plan_hash", "collected_at"):
        if not isinstance(event[field], str) or not event[field].strip():
            raise ValueError(f"{table}.collections[{index}].{field} must be a non-empty string")
    if event["status"] not in (STATUS_COMPLETE, STATUS_PARTIAL):
        raise ValueError(f"{table}.collections[{index}].status must be '{STATUS_COMPLETE}' or '{STATUS_PARTIAL}'")
    if not isinstance(event["rows"], int) or isinstance(event["rows"], bool) or event["rows"] < 0:
        raise ValueError(f"{table}.collections[{index}].rows must be a non-negative integer")
    source_rows = event.get("source_campaign_rows")
    if source_rows is not None and (
        not isinstance(source_rows, int) or isinstance(source_rows, bool) or source_rows < event["rows"]
    ):
        raise ValueError(
            f"{table}.collections[{index}].source_campaign_rows must be an integer at least as large as rows"
        )
    source_status = event.get("source_campaign_status")
    if source_status is not None and source_status not in (STATUS_COMPLETE, STATUS_PARTIAL):
        raise ValueError(
            f"{table}.collections[{index}].source_campaign_status must be '{STATUS_COMPLETE}' or '{STATUS_PARTIAL}'"
        )

    ordered = {
        field: event[field]
        for field in (*_TABLE_FIELD_ORDER, "source_campaign_rows", "source_campaign_status")
        if field in event
    }
    if "runtime" in event:
        ordered["runtime"] = _ordered_runtime(event["runtime"], field=f"{table}.collections[{index}].runtime")
    return ordered


def _ordered_multi_event_table(table: str, entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError(f"{table} must be a mapping")  # noqa: TRY004
    if "rows" not in entry or "status" not in entry:
        raise ValueError(f"{table} missing required merged-table rows/status")
    if not isinstance(entry["rows"], int) or isinstance(entry["rows"], bool) or entry["rows"] < 0:
        raise ValueError(f"{table}.rows must be a non-negative integer")
    if entry["status"] not in (STATUS_COMPLETE, STATUS_PARTIAL):
        raise ValueError(f"{table}.status must be '{STATUS_COMPLETE}' or '{STATUS_PARTIAL}'")

    collections = entry.get("collections")
    if collections is None:
        collections = [entry]
    if not isinstance(collections, list) or not collections:
        raise ValueError(f"{table}.collections must be a non-empty list")

    return {
        "rows": entry["rows"],
        "status": entry["status"],
        "collections": [
            _ordered_collection_event(event, table=table, index=index) for index, event in enumerate(collections)
        ],
    }


def validate_collection_meta_for_update(
    document: Any,
    *,
    tables_to_update: set[str] | frozenset[str] = frozenset(),
) -> None:
    """Validate an existing sidecar before a collector mutates table data.

    Historical ``local``/``collected`` v1 sidecars may contain reduced table
    summaries. They can be preserved beside a new table, but cannot take part
    in a v2 history promotion when any existing table is updated.
    """
    if not isinstance(document, dict):
        raise ValueError("collection_meta.yaml must be a mapping")  # noqa: TRY004
    unknown_document_fields = sorted(set(document) - {"schema_version", "provenance", "runtime", "tables"})
    if unknown_document_fields:
        raise ValueError(f"collection_meta.yaml has unsupported field(s): {', '.join(unknown_document_fields)}")
    schema_version = document.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version not in (COLLECTION_META_SCHEMA_VERSION, MULTI_EVENT_COLLECTION_META_SCHEMA_VERSION)
    ):
        raise ValueError(f"unsupported collection_meta.yaml schema_version {schema_version!r}")
    provenance_tier = document.get("provenance")
    if provenance_tier is not None and (not isinstance(provenance_tier, str) or not provenance_tier.strip()):
        raise ValueError("collection_meta.yaml provenance must be a non-empty string when provided")
    if provenance_tier == "legacy":
        raise ValueError("legacy-tier collection_meta.yaml cannot be updated by a fresh collection")
    _ordered_runtime(document.get("runtime"), field="runtime")

    tables = document.get("tables")
    if not isinstance(tables, dict):
        raise ValueError("collection_meta.yaml tables must be a mapping")  # noqa: TRY004
    if any(not isinstance(table, str) or not table for table in tables):
        raise ValueError("collection_meta.yaml table names must be non-empty strings")

    if schema_version == MULTI_EVENT_COLLECTION_META_SCHEMA_VERSION:
        for table, entry in tables.items():
            if not isinstance(entry, dict) or set(entry) != {"rows", "status", "collections"}:
                raise ValueError(f"{table} must contain exactly rows, status, and collections")
            _ordered_multi_event_table(table, entry)
        return

    reduced_tables: set[str] = set()
    for table, entry in tables.items():
        if isinstance(entry, dict) and set(entry) == {"status"} and provenance_tier in {"local", "collected"}:
            if entry["status"] not in (STATUS_COMPLETE, STATUS_PARTIAL):
                raise ValueError(f"{table}.status must be '{STATUS_COMPLETE}' or '{STATUS_PARTIAL}'")
            reduced_tables.add(table)
            continue
        if (
            isinstance(entry, dict)
            and set(entry) == {"collected_at", "rows", "status"}
            and provenance_tier in {"local", "collected"}
        ):
            if not isinstance(entry["collected_at"], str) or not entry["collected_at"].strip():
                raise ValueError(f"{table}.collected_at must be a non-empty string")
            if not isinstance(entry["rows"], int) or isinstance(entry["rows"], bool) or entry["rows"] < 0:
                raise ValueError(f"{table}.rows must be a non-negative integer")
            if entry["status"] not in (STATUS_COMPLETE, STATUS_PARTIAL):
                raise ValueError(f"{table}.status must be '{STATUS_COMPLETE}' or '{STATUS_PARTIAL}'")
            reduced_tables.add(table)
            continue
        if not isinstance(entry, dict) or set(entry) != set(_TABLE_FIELD_ORDER):
            raise ValueError(f"{table} must contain exactly {', '.join(_TABLE_FIELD_ORDER)}")
        _ordered_collection_event(entry, table=table, index=0)

    if reduced_tables and set(tables_to_update) & set(tables):
        raise ValueError(
            "reduced historical table metadata cannot be promoted while updating an existing sidecar table"
        )


def append_collection_event(
    existing: dict[str, Any], current: dict[str, Any], *, table: str, merged_rows: int
) -> dict[str, Any]:
    """Append a fresh collection event while retaining an existing valid history.

    A full v1 entry is promoted to the first v2 event. Incomplete v1/local
    entries cannot be promoted honestly and fail instead of being discarded.
    ``current.rows`` describes this event; ``merged_rows`` describes the
    accumulated parquet table as shipped.
    """
    if "collections" in existing:
        existing_history = _ordered_multi_event_table(table, existing)["collections"]
    else:
        existing_history = [_ordered_collection_event(existing, table=table, index=0)]
    current_event = _ordered_collection_event(current, table=table, index=len(existing_history))
    return {
        "rows": merged_rows,
        "status": current["status"],
        "collections": [*existing_history, current_event],
    }


def write_collection_meta(
    out_dir: str | Path,
    runtime_meta: dict[str, Any],
    tables: dict[str, dict[str, Any]],
    *,
    provenance_tier: str | None = None,
) -> Path:
    """Render ``collection_meta.yaml`` per design §5, with deterministic key order."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    schema_version = (
        MULTI_EVENT_COLLECTION_META_SCHEMA_VERSION
        if any("collections" in entry for entry in tables.values())
        else COLLECTION_META_SCHEMA_VERSION
    )
    if schema_version == MULTI_EVENT_COLLECTION_META_SCHEMA_VERSION:
        rendered_tables = {table: _ordered_multi_event_table(table, tables[table]) for table in sorted(tables)}
    else:
        rendered_tables = {
            table: {key: tables[table][key] for key in _TABLE_FIELD_ORDER if key in tables[table]}
            for table in sorted(tables)
        }

    doc = {"schema_version": schema_version}
    if provenance_tier is not None:
        if not isinstance(provenance_tier, str) or not provenance_tier.strip():
            raise ValueError("provenance_tier must be a non-empty string when provided")
        doc["provenance"] = provenance_tier
    doc["runtime"] = (
        _ordered_runtime(runtime_meta, field="runtime")
        if schema_version == MULTI_EVENT_COLLECTION_META_SCHEMA_VERSION
        else {key: runtime_meta[key] for key in _RUNTIME_FIELD_ORDER if key in runtime_meta}
    )
    doc["tables"] = rendered_tables

    meta_path = out_path / "collection_meta.yaml"
    with meta_path.open("w", encoding="utf-8") as meta_file:
        meta_file.write(spdx_header())
        yaml.safe_dump(doc, meta_file, sort_keys=False, default_flow_style=False)
    return meta_path

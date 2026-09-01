# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import functools
import importlib.resources as pkg_resources
import json
import logging
import os
import traceback
from collections import UserDict, defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import ClassVar, Optional

import yaml

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.common import PerfDataFilename
from aiconfigurator_core.sdk.errors import PerfDataNotAvailableError
from aiconfigurator_core.sdk.system_spec import SystemSpec

databases_cache = defaultdict(lambda: defaultdict(lambda: defaultdict()))
logger = logging.getLogger(__name__)

_SYSTEMS_PATHS: list[str] = [os.fspath(pkg_resources.files("aiconfigurator_core") / "systems")]

# Ad-hoc probe shapes must fit the op-list FFI's u32 token count; comm shims
# factor an element count into ``x * per_token_elements`` with both in range
# (power-of-two splits always exist for the legacy chart sweeps; odd counts
# stay whole in the per-token field).
_MAX_PROBE_X = 2**31


def _factor_message_size(size: int) -> tuple[int, int]:
    per_token = 1
    x = int(size)
    while x > _MAX_PROBE_X:
        if x % 2:
            raise ValueError(f"message size {size} cannot be factored into u32 range")
        x //= 2
        per_token *= 2
    return x, per_token


SHARED_LAYER_REUSE_MARKER = "SHARED_LAYER_REUSE.txt"
INCOMPLETE_MARKER = "INCOMPLETE.txt"
# Structured provenance markers (Collector V3 design §5/§6.3), yaml-first with the
# two legacy .txt markers above kept as a one-transition-window fallback.
REUSE_YAML_MARKER = "reuse.yaml"
COLLECTION_META_MARKER = "collection_meta.yaml"
_DATABASE_VERSION_METADATA_FILES = {
    SHARED_LAYER_REUSE_MARKER,
    INCOMPLETE_MARKER,
    REUSE_YAML_MARKER,
    COLLECTION_META_MARKER,
}


def _normalize_systems_paths(raw_paths: str | Iterable[str] | None) -> list[str]:
    default_path = os.fspath(pkg_resources.files("aiconfigurator_core") / "systems")
    if raw_paths is None:
        return [default_path]
    if isinstance(raw_paths, str):
        entries = [part.strip() for part in raw_paths.split(",") if part.strip()]
    else:
        entries = [os.fspath(entry) for entry in raw_paths if entry is not None]
    if not entries:
        return [default_path]
    resolved: list[str] = []
    for entry in entries:
        if str(entry).lower() == "default":
            resolved.append(default_path)
        else:
            resolved.append(os.fspath(entry))
    return resolved


def set_systems_paths(raw_paths: str | Iterable[str] | None) -> None:
    """
    Override the system search paths for the current process.

    Also evicts every Operation subclass's class-level CSV cache via
    ``clear_all_op_caches()`` — those caches are keyed by ``systems_root``
    among other things, so changing the path set could otherwise serve
    stale rows on a subsequent ``PerfDatabase`` construction that aliases
    a previously-loaded key tuple.
    """
    global _SYSTEMS_PATHS
    resolved_paths = _normalize_systems_paths(raw_paths)
    invalid_paths = [path for path in resolved_paths if not os.path.isdir(path)]
    if invalid_paths:
        raise ValueError(
            "Invalid --systems-paths: each entry must be an existing directory. "
            f"Invalid entries: {', '.join(invalid_paths)}"
        )
    _SYSTEMS_PATHS = resolved_paths
    _load_system_spec_from_paths.cache_clear()
    _cached_configured_database_view.cache_clear()
    from aiconfigurator_core.sdk.operations.base import clear_all_op_caches

    clear_all_op_caches()


def get_systems_paths() -> list[str]:
    return list(_SYSTEMS_PATHS)


def _version_sort_tuple(version_str: str) -> tuple:
    """Comparable tuple for version strings like '1.3.0rc10' / '0.5.6.post2' / 'v0.20_fix'."""
    import re

    version_str = str(version_str).lower()

    def suffix_number(start: int) -> int:
        suffix_match = re.search(r"(\d+)(?!.*\d)", version_str[start:])
        return int(suffix_match.group(1)) if suffix_match else 0

    def prerelease_parts() -> list[int]:
        rc_match = re.search(r"rc(\d+)", version_str)
        if rc_match:
            return [0, int(rc_match.group(1))]
        if "rc" in version_str:
            return [0, 0]
        return [1, 0]

    m = re.search(r"(\d+)\.(\d+)\.(\d+)", version_str)
    if m:
        parts = [int(m.group(1)), int(m.group(2)), int(m.group(3))]
        parts.extend(prerelease_parts())
        parts.append(suffix_number(m.end()))
        return tuple(parts)
    m = re.search(r"v?(\d+)\.(\d+)", version_str)
    if m:
        parts = [int(m.group(1)), int(m.group(2)), 0]
        parts.extend(prerelease_parts())
        parts.append(suffix_number(m.end()))
        return tuple(parts)
    return (0, 0, 0, 0, 0, 0)


# ---------------------------------------------------------------------------
# Queryable-version slots.
#
# Each (system, framework) exposes at most three queryable versions:
#   current  — the newest maintainer-completed full upgrade (authored in
#              systems/query_versions.yaml; per-system overrides win)
#   previous — the current before it (authored alongside)
#   next     — DERIVED: the highest declared version newer than current
#              (single-op development drops for new models land here; their
#              lower-version op data serves next queries via backward fill)
#
# The literal aliases "current" / "previous" / "next" are accepted wherever a
# version is requested. Any other version is rejected loudly unless the
# caller passes allow_unlisted_version=True (test fixtures pinning data
# coordinates). Data directories outside the slots are NOT versions — they
# are backward-fill / cross-backend sources only. When the slots file is
# absent from the systems root (external/synthetic trees), the gate is off
# and behavior is unchanged.
# ---------------------------------------------------------------------------

_QUERY_VERSIONS_BASENAME = "query_versions.yaml"
_SLOT_ALIASES = ("current", "previous", "next")


@functools.cache
def _load_query_slots_doc(systems_paths: tuple[str, ...]) -> dict | None:
    for systems_root in systems_paths:
        path = os.path.join(systems_root, _QUERY_VERSIONS_BASENAME)
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"failed to parse {path}: {e}; version slots disabled")
                return None
    return None


def get_version_slots(system: str, backend: str, systems_paths: str | list[str] | None = None) -> dict[str, str] | None:
    """Resolved slots for (system, backend): {'current': v[, 'previous': v][, 'next': v]}.

    None when the slots file is absent (gate off) or the combination has no
    authored entry (e.g. a framework never maintained on that system).
    """
    if systems_paths is None:
        systems_paths = get_systems_paths()
    elif isinstance(systems_paths, str):
        systems_paths = [systems_paths]
    doc = _load_query_slots_doc(tuple(systems_paths))
    if not doc:
        return None
    override_entry = (doc.get("overrides") or {}).get(system, {}).get(backend)
    entry = override_entry if override_entry is not None else (doc.get("defaults") or {}).get(backend)
    if not entry or not entry.get("current"):
        return None
    # Slots only govern systems that actually hold data for this backend:
    # estimate-only spec yamls (no data tree) and synthetic test systems keep
    # the ungated behavior even though the framework defaults exist.
    has_any_version_dir = False
    for systems_root in systems_paths:
        system_yaml_path = os.path.join(systems_root, f"{system}.yaml")
        if not os.path.isfile(system_yaml_path):
            continue
        try:
            with open(system_yaml_path) as f:
                system_spec = yaml.safe_load(f) or {}
            data_dir = os.path.join(systems_root, system_spec.get("data_dir", ""))
            if any(True for _ in _iter_backend_version_dirs(data_dir, backend)):
                has_any_version_dir = True
                break
        except Exception:
            continue
    if not has_any_version_dir:
        return None
    slots = {"current": str(entry["current"])}
    if entry.get("previous"):
        slots["previous"] = str(entry["previous"])
    # next is derived FLEET-WIDE per framework: systematic upgrades and
    # development drops land on typical hardware together, so the next version
    # is one label across every defaults-governed system — a system without
    # its own next-version data serves next queries through backward fill (exactly the
    # rows the retired reuse markers used to graft). Override systems
    # (frozen baselines like a100/b60) expose no next.
    if override_entry is None:
        # Override-governed systems (frozen baselines like a100/b60) are a
        # separate compatibility domain: their data drops must not advertise
        # a fleet-wide next that defaults-governed systems cannot load.
        override_systems = frozenset(
            name for name, per_backend in (doc.get("overrides") or {}).items() if (per_backend or {}).get(backend)
        )
        nxt = _derive_fleet_next(tuple(systems_paths), backend, slots["current"], override_systems)
        if nxt is not None:
            slots["next"] = nxt
    return slots


@functools.cache
def _derive_fleet_next(
    systems_paths: tuple[str, ...], backend: str, current: str, exclude_systems: frozenset[str] = frozenset()
) -> str | None:
    """Highest DATA-BACKED version strictly newer than `current`, across all
    defaults-governed systems in the tree (override-governed systems are
    excluded — their frozen baselines are not part of the fleet upgrade
    cadence). Marker-only directories do not qualify — next means a developer
    actually dropped measurements somewhere."""
    current_key = _version_sort_tuple(current)
    candidates: set[str] = set()
    for systems_root in systems_paths:
        try:
            entries = os.listdir(systems_root)
        except Exception:
            continue
        for entry in entries:
            if not entry.endswith(".yaml"):
                continue
            if entry[: -len(".yaml")] in exclude_systems:
                continue
            try:
                with open(os.path.join(systems_root, entry)) as f:
                    system_spec = yaml.safe_load(f) or {}
                data_dir = os.path.join(systems_root, system_spec.get("data_dir", ""))
                if not os.path.isdir(data_dir):
                    continue
                for version, version_path in _iter_backend_version_dirs(data_dir, backend):
                    if _version_sort_tuple(version) > current_key and _database_version_dir_has_perf_files(
                        version_path
                    ):
                        candidates.add(version)
            except Exception as e:
                logger.warning("could not derive next slot from %s: %s", entry, e)
    return max(candidates, key=_version_sort_tuple) if candidates else None


def resolve_query_version(
    system: str,
    backend: str,
    version: str,
    systems_paths: str | list[str] | None = None,
    allow_unlisted: bool = False,
) -> str:
    """Map a requested version (or slot alias) onto the queryable slots.

    Raises ValueError for versions outside the slots unless allow_unlisted.
    """
    slots = get_version_slots(system, backend, systems_paths=systems_paths)
    if slots is None:
        if version in _SLOT_ALIASES:
            raise ValueError(
                f"no queryable versions defined for {backend} on {system}; cannot resolve alias {version!r}"
            )
        return version
    if version in _SLOT_ALIASES:
        resolved = slots.get(version)
        if resolved is None:
            raise ValueError(f"{backend} on {system} has no {version!r} version; available slots: {slots}")
        return resolved
    if version in slots.values() or allow_unlisted:
        return version
    # Transition escape for legacy test fixtures that pin data coordinates
    # through user-level surfaces (Task/CLI) with no allow parameter. Not for
    # production use; the fixture-discipline follow-up retires it.
    if os.environ.get("AIC_ALLOW_UNLISTED_VERSIONS", "").lower() in ("1", "true", "yes"):
        return version
    raise ValueError(
        f"{backend}/{version!r} looks like an old-style raw version query; "
        f"{system} now resolves versions through queryable slots. "
        f"New way: use an alias ('current'/'previous'/'next') or one of the "
        f"slot versions {slots}. "
        f"Old way (raw data-coordinate access, data outside these slots is "
        f"not maintained to the queryable bar): re-run with the environment "
        f"variable AIC_ALLOW_UNLISTED_VERSIONS=1, or pass "
        f"allow_unlisted_version=True in SDK code."
    )


@functools.cache
def _load_system_spec_from_paths(systems_paths: tuple[str, ...], system_name: str) -> dict:
    for systems_root in systems_paths:
        spec_path = os.path.join(systems_root, f"{system_name}.yaml")
        if os.path.exists(spec_path):
            with open(spec_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    return {}


def load_system_spec(
    system_name: str | None,
    systems_paths: str | os.PathLike | Iterable[str] | None = None,
) -> dict:
    if not system_name:
        return {}
    resolved_paths = _normalize_systems_paths(systems_paths if systems_paths is not None else get_systems_paths())
    return _load_system_spec_from_paths(tuple(resolved_paths), system_name)


def is_blackwell_system(system_name: str | None) -> bool:
    """True for Blackwell-class systems (SM >= 100, e.g. b200_sxm / gb200 / b300 / gb300)."""
    if not system_name:
        return False
    spec = load_system_spec(system_name)
    return int(spec.get("gpu", {}).get("sm_version", -1)) >= 100


def is_hopper_system(system_name: str | None) -> bool:
    """True for Hopper-class systems (SM 90, e.g. h100 / h200 / gh200)."""
    if not system_name:
        return False
    spec = load_system_spec(system_name)
    return int(spec.get("gpu", {}).get("sm_version", -1)) == 90


def build_no_databases_message() -> str:
    """Build a concise error message for systems path/db validation failures."""
    resolved_paths = get_systems_paths()
    resolved_display = ", ".join(resolved_paths) if resolved_paths else "<none>"
    default_path = os.fspath(pkg_resources.files("aiconfigurator_core") / "systems")
    has_default = default_path in resolved_paths

    lines = [
        "No loadable performance databases found under --systems-paths.",
        f"Configured systems paths: {resolved_display}",
    ]
    if has_default:
        lines.append(
            "Built-in `default` systems path is already included, and no databases "
            "could be loaded from either default or extra paths."
        )
    else:
        lines.append("Tip: try adding `default` to --systems-paths and run again.")
    return "\n".join(lines)


def has_perf_data_not_available_cause(error: BaseException) -> bool:
    """Return True when an exception's effective chain has a structured perf-data miss."""
    seen: set[int] = set()
    stack: list[BaseException] = [error]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        if isinstance(current, PerfDataNotAvailableError):
            return True
        seen.add(id(current))
        if current.__cause__ is not None:
            stack.append(current.__cause__)
        elif not current.__suppress_context__ and current.__context__ is not None:
            stack.append(current.__context__)
    return False


# Instance attribute(s) holding the raw table(s) behind each fmha-keyed context
# op. Every listed table is keyed [fmha][kv_cache]... at its top two levels, so
# joint (fmha, kv) slice presence can be checked uniformly. Ops absent from
# this map (e.g. wideep_context_mla: [kernel_source][quant], no kv axis) fall
# back to the flat supported list.
_CONTEXT_FMHA_OP_TABLES: dict[str, tuple[str, ...]] = {
    "context_attention": ("_context_attention_data",),
    "context_mla": ("_context_mla_data", "_context_mla_module_data"),
    "context_mla_granular": ("_context_mla_data",),
    "dsa_context_module": ("_context_dsa_module_data",),
    "deepseek_v4_context_module": ("_context_deepseek_v4_attention_module_data",),
}


def context_fmha_supported_modes(database, ctx_op: str, kv_cache_mode) -> list[str]:
    """FMHA mode names with perf data for ``ctx_op``, restricted to slices that
    exist JOINTLY with ``kv_cache_mode``.

    The flat ``supported_quant_mode[ctx_op]`` list unions fmha keys across kv
    slices (and across granular+module tables for ``context_mla``), so an fmha
    mode collected only under a different kv dtype — e.g. the fp8 fmha slice
    that exists solely under kv=fp8 — would look available for a bf16-kv role
    and then miss at query time.  Returns ``[]`` when there is no information
    (missing op/table); falls back to the flat list when the op has no kv axis
    or the database exposes no raw tables (test stubs).
    """
    supported = getattr(database, "supported_quant_mode", {}) or {}
    flat = supported.get(ctx_op, []) or []  # triggers the lazy load of the op's table(s)
    if not flat:
        return []
    table_attrs = _CONTEXT_FMHA_OP_TABLES.get(ctx_op)
    if table_attrs is None or kv_cache_mode is None:
        return list(flat)
    modes: set[str] = set()
    saw_table = False
    for attr in table_attrs:
        data = getattr(database, attr, None)
        if not data:
            continue
        saw_table = True
        for fmha_key in data:
            if kv_cache_mode in data[fmha_key]:
                modes.add(fmha_key.name if hasattr(fmha_key, "name") else str(fmha_key))
    if not saw_table:
        return list(flat)
    return sorted(modes)


# Path-resolution helpers live in ``operations.base`` so the per-op-module
# loaders can import them without a circular dependency on ``perf_database``
# at module load time; re-exported here for external callers. (The row
# readers ``_read_perf_rows`` / ``_read_filtered_rows`` retired with the
# last Python parser — the engine table view serves every table.)
from aiconfigurator_core.sdk.operations.base import (  # noqa: F401
    _KNOWN_BACKEND_DIRS,
    _resolve_perf_data_path,
    resolve_op_data_path,
)


def get_supported_databases(
    systems_paths: str | list[str] | None = None,
) -> dict[str, dict[str, list[str]]]:
    """
    Get all supported databases for all systems, backends and versions without loading them.
    """
    supported_sets: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    if systems_paths is None:
        systems_paths = get_systems_paths()
    elif isinstance(systems_paths, str):
        systems_paths = [systems_paths]

    for systems_root in systems_paths:
        try:
            entries = os.listdir(systems_root)
        except Exception as e:
            logger.warning("Could not list systems dir %s: %s", systems_root, e)
            continue
        for entry in entries:
            if not entry.endswith(".yaml"):
                continue
            system = entry[:-5]
            system_yaml_path = os.path.join(systems_root, entry)
            try:
                with open(system_yaml_path) as f:
                    system_spec = yaml.safe_load(f)

                data_dir = os.path.join(systems_root, system_spec.get("data_dir", ""))
                if not os.path.isdir(data_dir):
                    continue

                for backend in common.BackendName:
                    versions = _declared_versions(data_dir, backend.value)
                    if versions:
                        supported_sets[system][backend.value].update(versions)
            except Exception as e:
                logger.warning(f"Could not process system config {os.path.basename(system_yaml_path)}: {e}")

    supported_dict = defaultdict(lambda: defaultdict(list))
    for system, backend_versions in supported_sets.items():
        for backend, versions in backend_versions.items():
            # With version slots active, the queryable surface is the slot set
            # (current / previous / derived next), not every declared directory —
            # directories outside the slots are fill sources, not versions.
            slots = get_version_slots(system, backend, systems_paths=systems_paths)
            if slots is not None:
                supported_dict[system][backend] = sorted(set(slots.values()), key=_version_sort_tuple)
            else:
                supported_dict[system][backend] = sorted(versions)

    return supported_dict


def _iter_database_version_paths(
    system: str,
    backend: str,
    version: str,
    systems_paths: str | list[str] | None = None,
):
    """Yield (version_path, data_dir) pairs for a (system, backend, version)."""
    if systems_paths is None:
        systems_paths = get_systems_paths()
    elif isinstance(systems_paths, str):
        systems_paths = [systems_paths]

    for systems_root in systems_paths:
        system_yaml_path = os.path.join(systems_root, f"{system}.yaml")
        if not os.path.isfile(system_yaml_path):
            continue
        try:
            with open(system_yaml_path) as f:
                system_spec = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("Could not process system config %s: %s", os.path.basename(system_yaml_path), e)
            continue
        data_dir = os.path.join(systems_root, system_spec.get("data_dir", ""))
        for v, version_path in _iter_backend_version_dirs(data_dir, backend):
            if v == version:
                yield version_path, data_dir


def _database_version_dir_has_perf_files(version_path: str) -> bool:
    try:
        entries = os.listdir(version_path)
    except Exception:
        return False
    for entry in entries:
        if entry.startswith(".") or entry in _DATABASE_VERSION_METADATA_FILES:
            continue
        if os.path.isfile(os.path.join(version_path, entry)):
            return True
    return False


_REUSE_ENTRY_REQUIRED_KEYS = ("table", "from_version", "reason", "approved_by")
_LEGACY_MARKER_WARNED: set[tuple[str, str]] = set()


def _warn_legacy_marker_once(scope: str, marker_name: str, replacement: str) -> None:
    """One-time-per-(scope, marker) deprecation warning, mirroring _LEGACY_LAYOUT_WARNED."""
    key = (scope, marker_name)
    if key in _LEGACY_MARKER_WARNED:
        return
    _LEGACY_MARKER_WARNED.add(key)
    logger.warning(
        "Legacy marker file %s honored under %s; migrate to %s (Collector V3 design §5/§6.3)",
        marker_name,
        scope,
        replacement,
    )


def _parse_reuse_yaml(path: str) -> dict:
    """Parse+validate a ``reuse.yaml`` sidecar (design §6.3): top-level ``reuse:``
    list of {table, from_version, reason, approved_by}. Raises ValueError on any
    structural or type mismatch (fail loudly on malformed authored data).

    A present-but-empty ``reuse: []`` is a valid "nothing declared" document.
    A MISSING (or misspelled, e.g. ``reuses:``) top-level ``reuse`` key is a
    schema error, not silently treated as empty — it almost always means the
    author typo'd the key and the file's declarations are being dropped.
    """
    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"{path}: failed to parse reuse.yaml: {e}") from e
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level, got {type(raw).__name__}")  # noqa: TRY004
    if "reuse" not in raw:
        raise ValueError(f"{path}: missing required top-level 'reuse' key")
    entries = raw["reuse"]
    if not isinstance(entries, list):
        raise ValueError(f"{path}: 'reuse' must be a list, got {type(entries).__name__}")  # noqa: TRY004

    validated: list[dict[str, str]] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: reuse[{i}] must be a mapping, got {type(entry).__name__}")  # noqa: TRY004
        missing = [key for key in _REUSE_ENTRY_REQUIRED_KEYS if key not in entry]
        if missing:
            raise ValueError(f"{path}: reuse[{i}] missing required key(s): {', '.join(missing)}")
        for key in _REUSE_ENTRY_REQUIRED_KEYS:
            if not isinstance(entry[key], str) or not entry[key].strip():
                raise ValueError(f"{path}: reuse[{i}].{key} must be a non-empty string")
        validated.append({key: entry[key] for key in _REUSE_ENTRY_REQUIRED_KEYS})
    return {"entries": validated}


def _load_collection_meta_yaml(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"{path}: failed to parse collection_meta.yaml: {e}") from e
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level, got {type(raw).__name__}")  # noqa: TRY004
    schema_version = raw.get("schema_version", 1)
    if schema_version == 2:
        _validate_collection_meta_v2(raw, path)
    elif schema_version == 1:
        tables = raw.get("tables")
        if isinstance(tables, dict):
            for table, entry in tables.items():
                if isinstance(entry, dict) and "collections" in entry:
                    raise ValueError(f"{path}: tables.{table}.collections requires schema_version 2")
    else:
        raise ValueError(f"{path}: unsupported collection_meta.yaml schema_version {schema_version!r}")
    return raw


_COLLECTION_EVENT_REQUIRED_KEYS = (
    "collector_ref",
    "collector_hash",
    "case_plan_hash",
    "collected_at",
    "rows",
    "status",
)
_COLLECTION_EVENT_OPTIONAL_KEYS = ("source_campaign_rows", "source_campaign_status", "runtime")
_COLLECTION_STATUSES = frozenset({"complete", "partial"})
_COLLECTION_RUNTIME_KEYS = ("framework", "version", "image", "image_variant", "image_digest")


def _validate_non_negative_row_count(value: object, *, field: str, path: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{path}: {field} must be a non-negative integer")


def _validate_collection_runtime(runtime: object, *, field: str, path: str) -> None:
    if not isinstance(runtime, dict):
        raise ValueError(f"{path}: {field} must be a mapping")  # noqa: TRY004
    unknown = sorted(set(runtime) - set(_COLLECTION_RUNTIME_KEYS))
    if unknown:
        raise ValueError(f"{path}: {field} has unsupported key(s): {', '.join(unknown)}")
    for key in ("framework", "version"):
        if not isinstance(runtime.get(key), str) or not runtime[key].strip():
            raise ValueError(f"{path}: {field}.{key} must be a non-empty string")
    for key in _COLLECTION_RUNTIME_KEYS[2:]:
        if key in runtime and (not isinstance(runtime[key], str) or not runtime[key].strip()):
            raise ValueError(f"{path}: {field}.{key} must be a non-empty string when provided")


def _validate_collection_event(event: object, *, table: str, index: int, path: str) -> None:
    field_path = f"tables.{table}.collections[{index}]"
    if not isinstance(event, dict):
        raise ValueError(f"{path}: {field_path} must be a mapping")  # noqa: TRY004

    missing = [key for key in _COLLECTION_EVENT_REQUIRED_KEYS if key not in event]
    if missing:
        raise ValueError(f"{path}: {field_path} missing required key(s): {', '.join(missing)}")
    allowed = {*_COLLECTION_EVENT_REQUIRED_KEYS, *_COLLECTION_EVENT_OPTIONAL_KEYS}
    unknown = sorted(set(event) - allowed)
    if unknown:
        raise ValueError(f"{path}: {field_path} has unsupported key(s): {', '.join(unknown)}")

    for key in ("collector_ref", "collector_hash", "case_plan_hash", "collected_at"):
        if not isinstance(event[key], str) or not event[key].strip():
            raise ValueError(f"{path}: {field_path}.{key} must be a non-empty string")
    _validate_non_negative_row_count(event["rows"], field=f"{field_path}.rows", path=path)
    if event["status"] not in _COLLECTION_STATUSES:
        raise ValueError(f"{path}: {field_path}.status must be 'complete' or 'partial'")

    if "source_campaign_rows" in event:
        _validate_non_negative_row_count(
            event["source_campaign_rows"], field=f"{field_path}.source_campaign_rows", path=path
        )
        if event["source_campaign_rows"] < event["rows"]:
            raise ValueError(f"{path}: {field_path}.source_campaign_rows must be at least as large as rows")
    if "source_campaign_status" in event and event["source_campaign_status"] not in _COLLECTION_STATUSES:
        raise ValueError(f"{path}: {field_path}.source_campaign_status must be 'complete' or 'partial'")
    if "runtime" in event:
        _validate_collection_runtime(event["runtime"], field=f"{field_path}.runtime", path=path)


def _validate_collection_meta_v2(raw: dict, path: str) -> None:
    _validate_collection_runtime(raw.get("runtime"), field="runtime", path=path)

    tables = raw.get("tables")
    if not isinstance(tables, dict):
        raise ValueError(f"{path}: schema v2 'tables' must be a mapping")  # noqa: TRY004
    for table, entry in tables.items():
        if not isinstance(table, str) or not table:
            raise ValueError(f"{path}: schema v2 table names must be non-empty strings")
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: tables.{table} must be a mapping")  # noqa: TRY004
        unknown = sorted(set(entry) - {"rows", "status", "collections"})
        if unknown:
            raise ValueError(f"{path}: tables.{table} has unsupported key(s): {', '.join(unknown)}")
        if "rows" not in entry or "status" not in entry or "collections" not in entry:
            raise ValueError(f"{path}: tables.{table} requires rows, status, and collections")
        _validate_non_negative_row_count(entry["rows"], field=f"tables.{table}.rows", path=path)
        if entry["status"] not in _COLLECTION_STATUSES:
            raise ValueError(f"{path}: tables.{table}.status must be 'complete' or 'partial'")
        collections = entry["collections"]
        if not isinstance(collections, list) or not collections:
            raise ValueError(f"{path}: tables.{table}.collections must be a non-empty list")
        for index, event in enumerate(collections):
            _validate_collection_event(event, table=table, index=index, path=path)


def _collection_meta_partial_tables(meta: dict) -> frozenset[str]:
    """Return tables whose collection coverage is incomplete.

    ``status: partial`` is table/shape coverage metadata, not a statement that
    the successfully collected rows are invalid.  Those rows remain eligible
    as the primary source; older versions only fill coordinates they do not
    contain (design §6.1/§6.2 first-wins semantics).
    """
    tables = meta.get("tables")
    if not isinstance(tables, dict):
        return frozenset()
    return frozenset(
        name
        for name, table in tables.items()
        if isinstance(name, str) and isinstance(table, dict) and table.get("status") == "partial"
    )


def _collection_meta_has_partial_table(meta: dict) -> bool:
    """Compatibility predicate for metadata/reporting callers."""
    return bool(_collection_meta_partial_tables(meta))


def _version_dir_state(version_path: str, *, data_dir: str | None = None) -> dict[str, object]:
    """Consolidated per-version-dir marker state, yaml-first with legacy .txt fallback.

    - ``declared_reuse``: parsed ``reuse.yaml`` contents (design §6.3) when it has
      >=1 entry; the legacy marker-only sentinel when only ``SHARED_LAYER_REUSE.txt``
      is present; ``None`` when neither declares reuse.
    - ``partial`` / ``partial_tables``: informational collection-coverage
      state from ``collection_meta.yaml``. Partial tables still contribute all
      successfully collected rows; older versions fill only missing shapes.
    - ``unusable``: whole-directory exclusion. This is reserved for the legacy
      ``INCOMPLETE.txt`` marker, which has no table/shape granularity. A present
      ``collection_meta.yaml`` supersedes that legacy marker.
    - ``has_perf``: whether the dir holds real perf output files.

    ``data_dir`` scopes the one-time-per-tree legacy-marker deprecation warning
    (mirrors ``_LEGACY_LAYOUT_WARNED``); callers that have it in scope should pass
    it so honoring many legacy marker dirs under one tree warns once, not per dir.
    """
    warn_scope = data_dir if data_dir is not None else version_path

    declared_reuse: dict | None = None
    reuse_yaml_path = os.path.join(version_path, REUSE_YAML_MARKER)
    legacy_reuse_path = os.path.join(version_path, SHARED_LAYER_REUSE_MARKER)
    if os.path.isfile(reuse_yaml_path):
        parsed = _parse_reuse_yaml(reuse_yaml_path)
        declared_reuse = parsed if parsed["entries"] else None
    elif os.path.isfile(legacy_reuse_path):
        _warn_legacy_marker_once(warn_scope, SHARED_LAYER_REUSE_MARKER, REUSE_YAML_MARKER)
        declared_reuse = {"entries": [], "legacy": True}

    partial_tables: frozenset[str] = frozenset()
    unusable = False
    meta_yaml_path = os.path.join(version_path, COLLECTION_META_MARKER)
    legacy_incomplete_path = os.path.join(version_path, INCOMPLETE_MARKER)
    if os.path.isfile(meta_yaml_path):
        partial_tables = _collection_meta_partial_tables(_load_collection_meta_yaml(meta_yaml_path))
    elif os.path.isfile(legacy_incomplete_path):
        _warn_legacy_marker_once(warn_scope, INCOMPLETE_MARKER, COLLECTION_META_MARKER)
        # INCOMPLETE.txt predates per-table provenance, so there is no safe way
        # to identify which rows/tables succeeded. Keep the legacy fail-closed
        # whole-directory behavior only for this unstructured marker.
        unusable = True

    return {
        "declared_reuse": declared_reuse,
        "partial": bool(partial_tables) or unusable,
        "partial_tables": partial_tables,
        "unusable": unusable,
        "has_perf": _database_version_dir_has_perf_files(version_path),
    }


def _database_version_dir_is_declared(version_path: str, *, data_dir: str | None = None) -> bool:
    if not os.path.isdir(version_path):
        return False
    state = _version_dir_state(version_path, data_dir=data_dir)
    if state["unusable"]:
        return False
    return bool(state["has_perf"]) or state["declared_reuse"] is not None


# Alias of the canonical set in operations/base.py (which lists every
# standalone copy that must stay in sync).
KNOWN_BACKEND_DIRS = _KNOWN_BACKEND_DIRS
_LEGACY_LAYOUT_WARNED: set[str] = set()


def _iter_backend_version_dirs(data_dir: str, backend: str):
    """Yield (version, version_path) for a backend across BOTH tree layouts.

    Family layout: <data_dir>/<family>/<backend>/<version> — any first-level
    directory whose name is not a known backend dir is a family dir. Legacy
    layout: <data_dir>/<backend>/<version> (deprecated; warns once per tree).
    A (backend, version) may yield several paths — one per family dir holding
    it — and callers aggregate.
    """
    try:
        entries = os.listdir(data_dir)
    except Exception:
        return
    for entry in entries:
        entry_path = os.path.join(data_dir, entry)
        if entry.startswith(".") or not os.path.isdir(entry_path):
            continue
        if entry == backend:  # legacy layout
            if data_dir not in _LEGACY_LAYOUT_WARNED:
                _LEGACY_LAYOUT_WARNED.add(data_dir)
                logger.warning(
                    "Legacy perf-data layout (<backend>/<version>) found under %s; "
                    "migrate to <family>/<backend>/<version> (Collector V3 design §3)",
                    data_dir,
                )
            yield from _iter_version_subdirs(entry_path)
        elif entry not in KNOWN_BACKEND_DIRS:  # family dir
            backend_path = os.path.join(entry_path, backend)
            if os.path.isdir(backend_path):
                yield from _iter_version_subdirs(backend_path)


def _iter_version_subdirs(backend_path: str):
    try:
        versions = os.listdir(backend_path)
    except Exception:
        return
    for version in versions:
        version_path = os.path.join(backend_path, version)
        if not version.startswith((".", "_")) and os.path.isdir(version_path):
            yield version, version_path


def _declared_versions(data_dir: str, backend: str) -> set[str]:
    """Versions declared for a backend: >=1 path passes the per-dir check."""
    declared: set[str] = set()
    for version, version_path in _iter_backend_version_dirs(data_dir, backend):
        if version not in declared and _database_version_dir_is_declared(version_path, data_dir=data_dir):
            declared.add(version)
    return declared


def is_shared_layer_marker_only_version(
    system: str,
    backend: str,
    version: str,
    systems_paths: str | list[str] | None = None,
) -> bool:
    """True when a declared version has only the shared-layer marker and no measured files."""
    saw_marker = False
    for version_path, data_dir in _iter_database_version_paths(system, backend, version, systems_paths=systems_paths):
        state = _version_dir_state(version_path, data_dir=data_dir)
        if state["unusable"]:
            continue
        if state["has_perf"]:
            return False
        saw_marker = saw_marker or state["declared_reuse"] is not None
    return saw_marker


def get_latest_database_version(
    system: str,
    backend: str,
    systems_paths: str | list[str] | None = None,
    include_shared_layer_marker_versions: bool = False,
) -> str | None:
    """
    Get the latest database version for a given system and backend
    """
    import re

    # Under version slots, "latest" means the maintained default (current) —
    # next is opt-in via its alias, never the implicit default. The shortcut
    # only fires when current is visible in the supported set, so tests that
    # monkeypatch get_supported_databases keep their synthetic behavior.
    slots = get_version_slots(system, backend, systems_paths=systems_paths)

    if systems_paths is None:
        supported_databases = get_supported_databases()
    else:
        supported_databases = get_supported_databases(systems_paths=systems_paths)
    database_versions = supported_databases.get(system, {}).get(backend, [])
    if slots is not None and slots["current"] in database_versions:
        return slots["current"]
    if not include_shared_layer_marker_versions:
        database_versions = [
            version
            for version in database_versions
            if not is_shared_layer_marker_only_version(system, backend, version, systems_paths=systems_paths)
        ]
    if not database_versions:
        logger.info("database not found for %s, %s", system, backend)
        return None

    def parse_version(version_str):
        """Parse version string into comparable tuple"""
        # Handle different version formats
        version_str = version_str.lower()

        def suffix_number(start: int) -> int:
            suffix = version_str[start:]
            suffix_match = re.search(r"(\d+)(?!.*\d)", suffix)
            return int(suffix_match.group(1)) if suffix_match else 0

        def prerelease_parts() -> list[int]:
            rc_match = re.search(r"rc(\d+)", version_str)
            if rc_match:
                return [0, int(rc_match.group(1))]
            if "rc" in version_str:
                return [0, 0]
            return [1, 0]

        # Extract numeric version pattern (e.g., "1.2.3" from "v1.2.3rc4" or "1.2.3_suffix")
        version_match = re.search(r"(\d+)\.(\d+)\.(\d+)", version_str)
        if version_match:
            major, minor, patch = map(int, version_match.groups())
            version_parts = [major, minor, patch]
            version_parts.extend(prerelease_parts())
            version_parts.append(suffix_number(version_match.end()))
            return tuple(version_parts)

        # Try to extract version from other patterns (e.g., "v0.20_fix0719")
        version_match = re.search(r"v?(\d+)\.(\d+)", version_str)
        if version_match:
            major, minor = map(int, version_match.groups())
            version_parts = [major, minor, 0]
            version_parts.extend(prerelease_parts())
            version_parts.append(suffix_number(version_match.end()))
            return tuple(version_parts)

        # For completely non-standard versions, try to extract any numbers
        numbers = re.findall(r"\d+", version_str)
        if numbers:
            # Use first few numbers found, pad with zeros
            version_parts = [int(x) for x in numbers[:3]]
            while len(version_parts) < 3:
                version_parts.append(0)
            version_parts.extend([0, 0, 0])  # Add RC and suffix indicators
            return tuple(version_parts)

        # If no numbers found, return a very low priority tuple
        return (0, 0, 0, -1, 0, 0)

    # Convert version strings to comparable tuples
    versions_ids = []
    for version in database_versions:
        try:
            version_parts = parse_version(version)
            versions_ids.append((version_parts, version))
            logger.debug(f"Parsed version {version} as {version_parts}")
        except Exception as e:
            logger.warning(f"Failed to parse version {version}: {e}")
            continue

    if not versions_ids:
        logger.info("no valid versions parsed for %s, %s", system, backend)
        return None

    # Find the latest version by comparing version tuples.
    # The tuple format (major, minor, patch, is_stable, rc_num, suffix_num)
    # ensures correct sorting across stable, RC, and suffixed releases.
    latest_version = max(versions_ids, key=lambda x: x[0])

    logger.debug(f"Latest version for {system}/{backend}: {latest_version[1]} (parsed as {latest_version[0]})")
    return latest_version[1]


def _shared_layer_enabled(database_mode: str | None) -> bool:
    """Whether the shared layer (sibling/cross-version row inheritance) loads.

    Enabled for the default database mode, SILICON, and HYBRID: all consult the
    silicon tables, so they benefit from reusing older collected data points when
    the active backend/version lacks a shape. Explicit formula-only modes
    compute without sibling silicon rows.
    """
    return database_mode is None or database_mode.upper() in ("SILICON", "HYBRID")


# ─────────────────────────────────────────────────────────────────────────
# Strict provenance mode (Collector V3 design §5/§7.4, AIC-1503 PR4 task 2).
#
# Fail-closed load-time validation, scoped to the dirs a ``get_database()``
# request actually touches: the REQUESTED (system, backend, version)'s own
# family dirs (primary), plus any donor dir their own ``reuse.yaml`` declares
# (channel 2, design §6.3). Nearest-earlier/cross-backend fallback siblings
# are NOT walked here -- their count is unbounded per request, and auditing
# them is the CI audit's job (design §8), not this per-request hook's.
#
# Off by default; CI turns it on (env var, see build-test.yml). When on, a
# violation raises instead of the warn-and-continue this module otherwise
# uses for questionable-but-recoverable data.
# ─────────────────────────────────────────────────────────────────────────

_STRICT_PROVENANCE_WARNED: set[tuple[str, str]] = set()

# (sorted primary paths, backend) pairs that already passed
# ``_check_strict_provenance_for_request`` with ``strict=True`` this process.
# ``get_database()`` consults it on strict cache HITS: ``databases_cache`` can
# hold worker-imported instances that were never request-validated (see
# ``_store_loaded_database``), so a strict hit must validate before returning —
# this memo keeps repeated strict hits cheap. Cleared wherever
# ``databases_cache`` entries are invalidated (``unload_database``).
_STRICT_VALIDATED_REQUESTS: set[tuple[tuple[str, ...], str]] = set()


def _strict_provenance_enabled(strict_provenance: bool | None) -> bool:
    """Resolve the effective strict-provenance flag. An explicit bool wins;
    ``None`` falls back to the ``AIC_STRICT_PROVENANCE`` env var (truthy:
    ``"1"`` or ``"true"``, case-insensitive)."""
    if strict_provenance is not None:
        return bool(strict_provenance)
    return os.environ.get("AIC_STRICT_PROVENANCE", "").strip().lower() in ("1", "true")


def _warn_strict_provenance_once(kind: str, key: str, message: str) -> None:
    """One-time-per-(kind, key) warning, mirroring this module's other
    warn-once helpers (e.g. ``_warn_legacy_marker_once``)."""
    dedupe_key = (kind, key)
    if dedupe_key in _STRICT_PROVENANCE_WARNED:
        return
    _STRICT_PROVENANCE_WARNED.add(dedupe_key)
    logger.warning(message)


def _version_dir_data_filenames(version_path: str) -> list[str]:
    """Real perf-data filenames physically present in a version dir (excludes
    dotfiles and the structured/legacy provenance marker files themselves).

    A nonexistent dir is a legitimate "no files" answer; any other ``OSError``
    (e.g. permission denied) propagates so ``_check_strict_provenance_coverage``
    can fail closed instead of silently treating an unreadable dir as empty."""
    try:
        entries = os.listdir(version_path)
    except (FileNotFoundError, NotADirectoryError):
        return []
    names = []
    for entry in entries:
        if entry.startswith(".") or entry in _DATABASE_VERSION_METADATA_FILES:
            continue
        if os.path.isfile(os.path.join(version_path, entry)):
            names.append(entry)
    return names


def _check_strict_provenance_coverage(version_path: str, *, strict: bool, only_table: str | None = None) -> None:
    """Design §5/§7.4 sidecar-coverage check for one version dir.

    ``only_table`` narrows the check to a single declared-reuse table
    (channel 2 admits one table at a time, design §6.3); ``None`` checks
    every real data file physically present (used for primary version
    dirs). A missing ``collection_meta.yaml`` sidecar, or one that does not
    list the table(s) in question, raises ``ValueError`` naming the dir and
    table(s) when ``strict`` is True; otherwise it logs a warning once per
    (dir, condition) and returns. A malformed sidecar surfaces the existing
    ``ValueError`` from ``_load_collection_meta_yaml`` in strict mode;
    non-strict warns instead. A ``provenance: legacy`` sidecar is graced --
    warns once per dir, never raises -- in BOTH modes (one-release grace,
    design §5): the AIC-1502 backfill covers the whole tree, but the tier is
    honest about not having per-table hashes, so a coverage gap there is
    treated as a data-quality note, not a hard failure.
    """
    if only_table is not None:
        stems = {only_table}
    else:
        try:
            data_filenames = _version_dir_data_filenames(version_path)
        except OSError as e:
            message = (
                f"{version_path}: cannot inspect perf-data files ({e}); strict provenance "
                "cannot verify sidecar coverage (Collector V3 design §5/§7.4)"
            )
            if strict:
                raise ValueError(message) from e
            _warn_strict_provenance_once("unreadable-version-dir", version_path, message)
            return
        if not data_filenames:
            return
        stems = {os.path.splitext(name)[0] for name in data_filenames}

    meta_path = os.path.join(version_path, COLLECTION_META_MARKER)
    if not os.path.isfile(meta_path):
        message = (
            f"{version_path}: holds table(s) {sorted(stems)} with no collection_meta.yaml "
            "sidecar (Collector V3 design §5/§7.4)"
        )
        if strict:
            raise ValueError(message)
        _warn_strict_provenance_once("missing-sidecar", version_path, message)
        return

    try:
        meta = _load_collection_meta_yaml(meta_path)
    except ValueError as e:
        if strict:
            raise
        _warn_strict_provenance_once("malformed-sidecar", meta_path, str(e))
        return

    tables = meta.get("tables") if isinstance(meta, dict) else None
    covered = tables if isinstance(tables, dict) else {}
    uncovered = sorted(stems - set(covered))
    if not uncovered:
        return

    if isinstance(meta, dict) and meta.get("provenance") == "legacy":
        _warn_strict_provenance_once(
            "legacy-uncovered",
            meta_path,
            f"{meta_path}: provenance: legacy sidecar does not list table(s) {uncovered}; "
            "graced for one release (Collector V3 design §5)",
        )
        return

    message = (
        f"{meta_path}: table(s) {uncovered} not covered by collection_meta.yaml 'tables' entries "
        "(Collector V3 design §5/§7.4)"
    )
    if strict:
        raise ValueError(message)
    _warn_strict_provenance_once("uncovered-table", meta_path, message)


def _is_family_layout_version_dir(version_path: str, data_dir: str) -> bool:
    """True when ``version_path`` is ``<data_dir>/<family>/<backend>/<version>``
    (3 path components under ``data_dir``), false for the legacy
    ``<data_dir>/<backend>/<version>`` layout (2 components) or an
    otherwise-unresolved path. ``collection_meta.yaml``/``reuse.yaml`` are a
    family-layout-paired concept (design §7 item 1: "family tree first,
    legacy layout as deprecated fallback for one transition window") -- a
    legacy-shaped dir predates the V3 metadata regime entirely, so strict
    provenance does not apply to it, mirroring how the engine resolver's
    ``op_file_namespace_from_path`` (``perf_database/source_resolution.rs``)
    treats legacy-shaped paths as structurally outside the comm-family
    exclusion (Task 1's FIX-2).
    """
    try:
        rel = os.path.relpath(version_path, data_dir)
    except ValueError:
        return False
    parts = rel.split(os.sep)
    return len(parts) == 3 and parts[0] not in KNOWN_BACKEND_DIRS


def _check_strict_provenance_for_request(paths: list[str], backend: str, data_dir_abs: str, *, strict: bool) -> None:
    """Run ``_check_strict_provenance_coverage`` over every dir one
    ``get_database()`` request touches: each primary ``path`` (a family dir
    holding the requested (backend, version)), plus every donor dir that
    path's own ``reuse.yaml`` declares (design §6.3). A malformed
    ``reuse.yaml`` surfaces its existing ``ValueError`` in strict mode;
    non-strict warns instead (same contract as the sidecar-coverage check).
    Legacy-layout primary dirs (see ``_is_family_layout_version_dir``) are
    skipped entirely -- out of the V3 metadata regime's scope.
    """
    for version_path in paths:
        if not _is_family_layout_version_dir(version_path, data_dir_abs):
            continue
        _check_strict_provenance_coverage(version_path, strict=strict)
        reuse_path = os.path.join(version_path, REUSE_YAML_MARKER)
        if not os.path.isfile(reuse_path):
            continue
        try:
            declared_entries = _parse_reuse_yaml(reuse_path)["entries"]
        except ValueError as e:
            if strict:
                raise
            _warn_strict_provenance_once("malformed-reuse", reuse_path, str(e))
            continue
        for entry in declared_entries:
            donor_path = resolve_op_data_path(data_dir_abs, backend, entry["from_version"], f"{entry['table']}.parquet")
            if not os.path.isfile(donor_path):
                continue  # not admitted -- mirrors _build_op_sources' own existence check
            _check_strict_provenance_coverage(os.path.dirname(donor_path), strict=strict, only_table=entry["table"])


def _version_dir_unusable_for_request(version_path: str, data_dir: str, *, strict: bool) -> bool:
    """Return whether a request must reject the whole version directory.

    Structured ``collection_meta.yaml`` partial status is intentionally *not*
    grounds for rejection: valid primary rows are loaded and sibling versions
    fill only missing shapes. Whole-directory rejection remains only for the
    unstructured legacy ``INCOMPLETE.txt`` marker.

    A malformed sidecar raises in strict mode; non-strict mode warns and lets
    normal loading continue, matching the existing request-scoped behavior.

    LOAD-GATE ONLY since the deprecation-cleanup PR: per-op source admission
    (the four-channel walk) moved into the engine
    (``perf_database/source_resolution.rs``), which applies this same
    predicate to every candidate dir. This Python copy remains the
    ``get_database`` request gate's incomplete-check.
    """
    try:
        return bool(_version_dir_state(version_path, data_dir=data_dir)["unusable"])
    except ValueError as e:
        if strict:
            raise
        _warn_strict_provenance_once("malformed-sidecar", version_path, str(e))
        return False


def get_database(
    system: str,
    backend: str,
    version: str,
    systems_paths: str | list[str] | None = None,
    allow_missing_data: bool = False,
    database_mode: str | None = None,
    shared_layer: bool | None = None,
    strict_provenance: bool | None = None,
    allow_unlisted_version: bool = False,
) -> PerfDatabase | None:
    """
    Get the database for a given system, backend and version.

    Args:
        system: the system name
        backend: the backend name
        version: the version name
        systems_paths: the systems search paths
        allow_missing_data: instantiate a database from system specs even when
            backend/version data files are absent. This is intended for SOL/EMPIRICAL
            formula-only modes. Silicon shared-layer reuse still requires
            an explicit backend/version directory; marker-only directories can
            declare new framework versions whose rows come from siblings.
        database_mode: the mode the caller will query under (`SILICON` / `HYBRID` /
            `EMPIRICAL` / `SOL`). The default mode, SILICON, and HYBRID enable
            the shared layer (sibling-row inheritance, including
            `kernel_source=default` fallback rows) so missing shapes are filled
            from older collected data; explicit formula-only modes keep it off.
        shared_layer: explicit shared-layer override. ``None`` (default) derives
            the flag from ``database_mode``; ``False`` restricts loading to the
            active backend/version's own rows even under SILICON; ``True``
            forces sibling inheritance on. Overridden templates are cached
            separately from derived ones.
        strict_provenance: fail-closed provenance mode (Collector V3 design
            §5/§7.4). ``None`` (default) resolves from the ``AIC_STRICT_PROVENANCE``
            env var. When on, a missing/malformed/uncovering ``collection_meta.yaml``
            or ``reuse.yaml`` sidecar under the requested (system, backend, version)
            or its declared donors raises instead of the module's usual
            warn-and-continue; `provenance: legacy` sidecars are always graced
            (warn, never raise).

    Returns:
        PerfDatabase for the given system, backend, version.
    """
    if systems_paths is None:
        systems_paths = get_systems_paths()
    elif isinstance(systems_paths, str):
        systems_paths = [systems_paths]

    if not version:
        logger.error(f"No database version available for {system=}, {backend=}")
        return None

    version = resolve_query_version(
        system, backend, version, systems_paths=systems_paths, allow_unlisted=allow_unlisted_version
    )

    shared_flag = _shared_layer_enabled(database_mode) if shared_layer is None else bool(shared_layer)
    # Only pass the override kwarg when explicitly set: PerfDatabase derives the
    # same flag from database_mode otherwise, and tests monkeypatch PerfDatabase
    # with fakes that predate the kwarg. Same rule for strict_provenance below --
    # when unset, PerfDatabase resolves the identical env-derived default itself.
    extra_database_kwargs = {} if shared_layer is None else {"shared_layer": shared_flag}
    if strict_provenance is not None:
        extra_database_kwargs["strict_provenance"] = strict_provenance
    effective_strict = _strict_provenance_enabled(strict_provenance)
    missing_data_candidate = None
    dirless_next_candidate = None
    is_advertised_next = None
    for systems_root in systems_paths:
        system_yaml_path = os.path.join(systems_root, f"{system}.yaml")
        if not os.path.isfile(system_yaml_path):
            continue
        cache_key = (systems_root, system, shared_flag, effective_strict)
        try:
            with open(system_yaml_path) as f:
                system_spec = yaml.load(f, Loader=yaml.SafeLoader)
            data_dir = system_spec["data_dir"]
        except Exception:
            logger.warning(f"failed to read system spec at {system_yaml_path}, continuing searching")
            continue

        data_dir_abs = os.path.join(systems_root, data_dir)
        paths = [p for v, p in _iter_backend_version_dirs(data_dir_abs, backend) if v == version]
        is_incomplete = bool(paths) and all(
            _version_dir_unusable_for_request(p, data_dir_abs, strict=effective_strict) for p in paths
        )
        if paths and not is_incomplete:
            request_key = (tuple(sorted(paths)), backend)
            try:
                database = databases_cache[cache_key][backend][version]
            except KeyError:
                logger.info(f"Loading database for {system=}, {backend=}, {version=}")
                try:
                    _check_strict_provenance_for_request(paths, backend, data_dir_abs, strict=effective_strict)
                    if effective_strict:
                        _STRICT_VALIDATED_REQUESTS.add(request_key)
                    database = PerfDatabase(
                        system,
                        backend,
                        version,
                        systems_root,
                        database_mode=database_mode,
                        **extra_database_kwargs,
                    )
                    databases_cache[cache_key][backend][version] = database
                    return database
                except Exception:
                    if effective_strict:
                        raise
                    logger.warning(
                        f"failed to load {system=}, {backend=}, {version=}, continuing searching",
                        exc_info=True,
                    )
            else:
                # Cache HIT: the entry may be a worker-imported instance that
                # was never request-validated (``_store_loaded_database``
                # inserts under a strict key without validating). Strict mode
                # must fail closed here too, not just on the miss path above.
                if effective_strict and request_key not in _STRICT_VALIDATED_REQUESTS:
                    _check_strict_provenance_for_request(paths, backend, data_dir_abs, strict=True)
                    _STRICT_VALIDATED_REQUESTS.add(request_key)
                return database
        else:
            # Directory-less NEXT candidate (design §14): the fleet-derived
            # next may be data-backed on other default-governed systems only;
            # a system without its own directory serves every op through the
            # channel-1 backward fill PerfDatabase already performs. DEFERRED,
            # never returned mid-loop: an exact next directory in a LATER
            # configured root must win over a directory-less fallback from an
            # earlier root (review blocker 2026-08-28). The relaxation covers
            # exactly the advertised next slot — raw versions and the
            # authored current/previous keep the loud missing-directory gate,
            # and provenance labeling is untouched.
            if is_advertised_next is None:
                slots = get_version_slots(system, backend, systems_paths=list(systems_paths))
                is_advertised_next = slots is not None and version == slots.get("next")
            if is_advertised_next:
                if dirless_next_candidate is None:
                    dirless_next_candidate = (systems_root, cache_key)
            elif allow_missing_data:
                if missing_data_candidate is None:
                    missing_data_candidate = (systems_root, cache_key)
                continue
            if is_incomplete:
                logger.warning(
                    f"data for {system=}, {backend=}, {version=} is marked incomplete in either layout, "
                    "continuing searching"
                )
            else:
                logger.warning(
                    f"no data found for {system=}, {backend=}, {version=} in either layout, continuing searching"
                )

    if dirless_next_candidate is not None:
        systems_root, cache_key = dirless_next_candidate
        try:
            database = databases_cache[cache_key][backend][version]
        except KeyError:
            pass
        else:
            database.dirless_next_load = True
            return database
        logger.info(
            f"Loading directory-less next database for {system=}, {backend=}, {version=} "
            "(no exact directory in any configured root; all ops ride backward fill)"
        )
        try:
            database = PerfDatabase(
                system, backend, version, systems_root, database_mode=database_mode, **extra_database_kwargs
            )
            # The engine reload (Rust) keeps its own loud missing-directory
            # gate; the spec builders read this marker to carry the
            # dir-less-next tolerance on the wire.
            database.dirless_next_load = True
            databases_cache[cache_key][backend][version] = database
            return database
        except Exception:
            if effective_strict:
                raise
            logger.warning(
                f"failed directory-less next load for {system=}, {backend=}, {version=}",
                exc_info=True,
            )

    if missing_data_candidate is not None:
        systems_root, cache_key = missing_data_candidate
        try:
            database = databases_cache[cache_key][backend][version]
            return database
        except KeyError:
            logger.info(f"Loading estimate-only database for {system=}, {backend=}, {version=}")
            try:
                database = PerfDatabase(
                    system, backend, version, systems_root, database_mode=database_mode, **extra_database_kwargs
                )
                databases_cache[cache_key][backend][version] = database
                return database
            except Exception:
                if effective_strict:
                    raise
                logger.warning(
                    f"failed to load estimate-only {system=}, {backend=}, {version=}",
                    exc_info=True,
                )

    logger.error(f"failed to get {system=}, {backend=}, {version=}")
    return None


def _normalize_database_mode(database_mode: str | common.DatabaseMode | None) -> common.DatabaseMode:
    if database_mode is None:
        return common.DatabaseMode.SILICON
    if isinstance(database_mode, common.DatabaseMode):
        _reject_retired_database_mode(database_mode)
        return database_mode
    mode = common.DatabaseMode[database_mode.upper()]
    _reject_retired_database_mode(mode)
    return mode


def _reject_retired_database_mode(mode: common.DatabaseMode) -> None:
    """SOL_FULL is a Python-side PER-CALL diagnostic, never a default mode.

    Per-call ``query_*(..., database_mode=SOL_FULL)`` returns the raw
    ``(sol_time, sol_math, sol_mem)`` tuple the sanity-check notebook
    (``tools/sanity_check/validate_database.ipynb``) plots — that surface
    stays, permanently Python-side per the freeze plan (#1357). As a
    database's ACTIVE mode it has never worked (the phase runners cannot
    consume a bare tuple), so mode entry rejects it with a clear error."""
    if mode == common.DatabaseMode.SOL_FULL:
        raise ValueError(
            "DatabaseMode.SOL_FULL cannot be a database's default mode; it is "
            "a per-call diagnostic (query_*(..., database_mode=SOL_FULL) "
            "returns the raw (sol_time, sol_math, sol_mem) tuple). Use "
            "DatabaseMode.SOL for engine-step speed-of-light estimates."
        )


@functools.cache
def _cached_configured_database_view(
    root_template: PerfDatabase,
    mode: common.DatabaseMode,
    policy: frozenset[common.TransferKind],
) -> PerfDatabase:
    """Build one lightweight immutable query view per normalized configuration."""
    view = copy.copy(root_template)
    view._root_database_template = root_template
    view._default_database_mode = mode
    view._transfer_policy = policy
    view._is_query_view = True

    # Lazy support resolution binds loaded op tables onto its database. Rebind
    # it to the configured copy while preserving already-resolved values; the
    # loaded table objects themselves remain shared and read-only.
    supported = getattr(root_template, "supported_quant_mode", None)
    if isinstance(supported, _LazySupportMatrix):
        lazy_support = _LazySupportMatrix(view)
        lazy_support._resolved = {key: list(value) for key, value in supported._resolved.items()}
        view.supported_quant_mode = lazy_support
    elif isinstance(supported, dict):
        view.supported_quant_mode = copy.deepcopy(supported)

    return view


def _get_configured_database_view(
    database: PerfDatabase,
    mode: str | common.DatabaseMode | None,
    transfer_policy=None,
    shared_layer: bool | None = None,
) -> PerfDatabase:
    """Return a cached configured copy rooted at the original data template."""
    normalized_mode = _normalize_database_mode(mode)
    policy = common.resolve_transfer_policy(transfer_policy)
    root_template = getattr(database, "_root_database_template", database)

    expected_shared_layer = _shared_layer_enabled(normalized_mode.name) if shared_layer is None else bool(shared_layer)
    if root_template.enable_shared_layer != expected_shared_layer:
        raise ValueError(
            f"Cannot create a {normalized_mode.name} query view from a database template with "
            f"enable_shared_layer={root_template.enable_shared_layer}; use get_database_view() "
            "so the correct data template is selected."
        )

    return _cached_configured_database_view(root_template, normalized_mode, policy)


def get_database_view(
    system: str,
    backend: str,
    version: str,
    systems_paths: str | list[str] | None = None,
    allow_missing_data: bool = False,
    database_mode: str | common.DatabaseMode | None = None,
    transfer_policy=None,
    shared_layer: bool | None = None,
    strict_provenance: bool | None = None,
    allow_unlisted_version: bool = False,
) -> PerfDatabase | None:
    """Return an isolated, lightweight query view over a cached database.

    The cached :class:`PerfDatabase` is a data template. Query mode and transfer
    policy are immutable, configuration-scoped state: callers requesting the
    same normalized configuration reuse a cached copy. The copy shares loaded,
    read-only perf tables while owning its interpolation cache and lazy
    support-matrix binding. ``database_mode`` is also forwarded to
    :func:`get_database` so EMPIRICAL/SOL views do not accidentally inherit the
    shared SILICON data layer. ``shared_layer`` explicitly overrides the
    mode-derived shared-layer flag (see :func:`get_database`); regression
    harnesses pass ``False`` to pin SILICON queries to per-version data.
    ``strict_provenance`` is forwarded to :func:`get_database` unchanged (see
    its docstring); ``None`` resolves from the ``AIC_STRICT_PROVENANCE`` env var.
    """
    mode = _normalize_database_mode(database_mode)
    database_kwargs = {
        "system": system,
        "backend": backend,
        "version": version,
        "allow_missing_data": allow_missing_data,
        "database_mode": mode.name,
        "shared_layer": shared_layer,
        "strict_provenance": strict_provenance,
        "allow_unlisted_version": allow_unlisted_version,
    }
    if systems_paths is not None:
        database_kwargs["systems_paths"] = systems_paths
    database = get_database(**database_kwargs)
    if database is None:
        return None
    return _get_configured_database_view(database, mode, transfer_policy, shared_layer=shared_layer)


DatabaseRef = tuple[str, str, str, str]
LoadedDatabaseResult = tuple[DatabaseRef, object | None, str | None]


def _as_systems_path_list(systems_paths: str | os.PathLike | Iterable[str] | None) -> list[str]:
    if systems_paths is None:
        return get_systems_paths()
    if isinstance(systems_paths, str | os.PathLike):
        return [os.fspath(systems_paths)]
    return [os.fspath(path) for path in systems_paths]


def _iter_system_yaml_files(systems_paths: list[str]):
    for systems_root in systems_paths:
        try:
            entries = sorted(os.listdir(systems_root))
        except Exception as e:
            logger.warning("Could not list systems dir %s: %s", systems_root, e)
            continue

        for entry in entries:
            if entry.endswith(".yaml"):
                yield systems_root, entry[:-5], os.path.join(systems_root, entry)


def _load_system_spec(system_yaml_path: str) -> dict | None:
    try:
        with open(system_yaml_path) as f:
            system_spec = yaml.load(f, Loader=yaml.SafeLoader)
    except Exception as e:
        logger.warning("Could not process system config %s: %s", os.path.basename(system_yaml_path), e)
        return None
    if not isinstance(system_spec, dict) or "data_dir" not in system_spec:
        logger.warning("Could not process system config %s: missing data_dir", os.path.basename(system_yaml_path))
        return None
    return system_spec


def _iter_database_refs_for_system(systems_root: str, system: str, system_spec: dict):
    data_dir = os.path.join(systems_root, system_spec["data_dir"])
    if not os.path.isdir(data_dir):
        return

    for backend in common.BackendName:
        backend_name = backend.value
        version_paths: dict[str, list[str]] = defaultdict(list)
        for version, version_path in _iter_backend_version_dirs(data_dir, backend_name):
            version_paths[version].append(version_path)

        for version in sorted(version_paths):
            paths = version_paths[version]
            if all(_version_dir_state(p, data_dir=data_dir)["unusable"] for p in paths):
                continue
            yield system, backend_name, version, systems_root


def _discover_database_refs(systems_paths: list[str]) -> list[DatabaseRef]:
    refs: list[DatabaseRef] = []
    seen_systems: dict[str, str] = {}
    seen_databases: dict[tuple[str, str, str], str] = {}

    for systems_root, system, system_yaml_path in _iter_system_yaml_files(systems_paths):
        if system in seen_systems:
            logger.warning(
                "System config '%s' already loaded from %s; also found in %s",
                system,
                seen_systems[system],
                systems_root,
            )
        else:
            seen_systems[system] = systems_root

        system_spec = _load_system_spec(system_yaml_path)
        if system_spec is None:
            continue

        for ref in _iter_database_refs_for_system(systems_root, system, system_spec):
            db_key = ref[:3]
            existing_root = seen_databases.get(db_key)
            if existing_root is not None:
                logger.warning(
                    "Database '%s/%s/%s' already loaded from %s; ignoring %s",
                    db_key[0],
                    db_key[1],
                    db_key[2],
                    existing_root,
                    systems_root,
                )
                continue
            seen_databases[db_key] = systems_root
            refs.append(ref)

    return refs


def _finalize_loaded_value(value):
    if isinstance(value, SystemSpec):
        return value
    if isinstance(value, LoadedOpData):
        value.data = _finalize_loaded_value(value.data)
        return value
    if isinstance(value, defaultdict):
        return {_finalize_loaded_value(key): _finalize_loaded_value(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {_finalize_loaded_value(key): _finalize_loaded_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_finalize_loaded_value(item) for item in value)
    if isinstance(value, list):
        return [_finalize_loaded_value(item) for item in value]
    return value


def _load_database_ref(ref: DatabaseRef) -> LoadedDatabaseResult:
    system, backend, version, systems_root = ref
    try:
        database = get_database(system, backend, version, systems_root)
        if database is None:
            return ref, None, "get_database returned None"
        return ref, database, None
    except Exception:
        return ref, None, traceback.format_exc()


def _new_database_dict() -> dict[str, dict[str, dict[str, PerfDatabase]]]:
    return defaultdict(lambda: defaultdict(lambda: defaultdict()))


def _store_loaded_database(
    database_dict: dict[str, dict[str, dict[str, PerfDatabase]]],
    ref: DatabaseRef,
    database: PerfDatabase,
) -> None:
    system, backend, version, systems_root = ref
    # A worker result may replace an existing root object for this data key.
    # Drop configured copies keyed by the old root so they cannot accumulate.
    _cached_configured_database_view.cache_clear()
    database_dict[system][backend][version] = database
    # get_all_databases() constructs the default (shared-enabled) view. Preserve
    # that identity when importing a database from a worker; putting it in the
    # formula-only slot would make a later EMPIRICAL lookup reuse shared rows.
    shared_flag = database.enable_shared_layer
    databases_cache[(systems_root, system, shared_flag, database.strict_provenance)][backend][version] = database


def clear_database_runtime_caches(system: str, backend: str, version: str) -> None:
    """Clear per-query/interpolation caches for one loaded database.

    Also evicts every Operation subclass's class-level CSV cache via
    ``clear_all_op_caches()`` so a subsequent reload reads fresh rows
    from disk — the per-class caches survive the per-instance
    ``clear_runtime_caches()`` and would otherwise serve the prior data.
    """
    seen_database_ids: set[int] = set()
    for cache_key, systems_cache in databases_cache.items():
        _, cached_system, _, _ = cache_key
        if cached_system != system:
            continue

        backend_cache = systems_cache.get(backend)
        if not backend_cache or version not in backend_cache:
            continue

        database = backend_cache[version]
        database_id = id(database)
        if database_id in seen_database_ids:
            continue
        seen_database_ids.add(database_id)
        clear_runtime_caches = getattr(database, "clear_runtime_caches", None)
        if callable(clear_runtime_caches):
            clear_runtime_caches()

    from aiconfigurator_core.sdk.operations.base import clear_all_op_caches

    clear_all_op_caches()
    _cached_configured_database_view.cache_clear()


def unload_database(system: str, backend: str, version: str) -> None:
    """Remove one loaded database from every systems-root/shared-mode cache.

    Also evicts every Operation subclass's class-level CSV cache via
    ``clear_all_op_caches()`` so a future ``get_database(...)`` for the
    same ``(system, backend, version)`` rebuilds the op-level caches from
    disk instead of aliasing the stale tables that survived the database
    pop.
    """
    for cache_key in list(databases_cache.keys()):
        _, cached_system, _, _ = cache_key
        if cached_system != system:
            continue

        systems_cache = databases_cache[cache_key]
        backend_cache = systems_cache.get(backend)
        if not backend_cache or version not in backend_cache:
            continue

        database = backend_cache.pop(version)
        clear_runtime_caches = getattr(database, "clear_runtime_caches", None)
        if callable(clear_runtime_caches):
            clear_runtime_caches()
        if not backend_cache:
            systems_cache.pop(backend, None)
        if not systems_cache:
            databases_cache.pop(cache_key, None)

    from aiconfigurator_core.sdk.operations.base import clear_all_op_caches

    clear_all_op_caches()
    _cached_configured_database_view.cache_clear()
    # A future get_database() for this triple must re-validate under strict
    # mode; the memo is coarse (keyed by primary paths), so drop it wholesale.
    _STRICT_VALIDATED_REQUESTS.clear()


def _load_database_ref_in_parent(ref: DatabaseRef) -> PerfDatabase | None:
    system, backend, version, systems_root = ref
    return get_database(system, backend, version, systems_root)


def get_all_databases(
    systems_paths: str | os.PathLike | Iterable[str] | None = None,
    max_workers: int | None = None,
) -> dict[str, dict[str, dict[str, PerfDatabase]]]:
    """
    Get all databases for all systems, backends, and versions.

    Discovery stays in-process so path precedence and duplicate warnings are
    deterministic. Database construction runs in a process pool because loading
    the CSV-backed op tables is the expensive part.
    """
    database_dict = _new_database_dict()
    refs = _discover_database_refs(_as_systems_path_list(systems_paths))
    if not refs:
        return database_dict

    if max_workers is None:
        max_workers = min(len(refs), max(1, (os.cpu_count() or 1) - 1))
    else:
        max_workers = max(1, min(max_workers, len(refs)))

    if max_workers == 1:
        for ref in refs:
            database = _load_database_ref_in_parent(ref)
            if database is not None:
                _store_loaded_database(database_dict, ref, database)
        return database_dict

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_load_database_ref, ref): ref for ref in refs}
        for future in as_completed(futures):
            ref = futures[future]
            system, backend, version, systems_root = ref
            try:
                loaded_ref, database, error = future.result()
            except Exception:
                logger.warning(
                    "Parallel load failed for %s/%s/%s from %s; retrying in parent",
                    system,
                    backend,
                    version,
                    systems_root,
                    exc_info=True,
                )
                database = _load_database_ref_in_parent(ref)
                if database is not None:
                    _store_loaded_database(database_dict, ref, database)
                continue

            if error is not None:
                logger.warning(
                    "Could not load database %s/%s/%s from %s: %s",
                    system,
                    backend,
                    version,
                    systems_root,
                    error,
                )
                continue
            if database is not None:
                _store_loaded_database(database_dict, loaded_ref, database)

    return database_dict


# ─────────────────────────────────────────────────────────────────────────
# Loader disposition after the deprecation-cleanup PR: the compiled engine
# owns the data plane — every ``PerfDatabase._<family>_data`` attribute is
# served by the engine table view (``sdk/engine_table_view.py``) in the
# retired parsers' exact nested shape, and NO path parses perf files in
# Python anymore, tests included (the collector-format handshake asserts
# against the engine view / frozen schema literals instead),
# tests included; the LAST survivor — the dsv4 csa-topk-calib parser —
# retired when the ``_dsv4_csa_topk_calib_data`` view attribute landed
# (``test_dsv4_cp_top_last_reuse.py`` now pins the engine view). The DSA
# model-dims constants keep their legacy import path.
# ─────────────────────────────────────────────────────────────────────────
from aiconfigurator_core.sdk.operations.dsa import (  # noqa: F401
    DEFAULT_DSA_ARCHITECTURE,
    DSA_MODEL_DIMS,
)


class LoadedOpData(UserDict):
    """
    A dictionary-like object which also keeps track of which file the data was loaded from.
    """

    def __init__(self, dict_data: Optional[dict], op_name_enum: PerfDataFilename, filepath: str):
        self.op_name_enum = op_name_enum
        self.filepath = filepath
        self.loaded = dict_data is not None

        super().__init__()
        if dict_data:
            # Freeze any defaultdicts so missing-key access at query time
            # raises ``KeyError`` instead of silently creating empty
            # branches. Previously this was handled by a one-shot
            # ``_finalize_loaded_data()`` walk at the end of
            # ``PerfDatabase.__init__``; the lazy contract means each
            # load_data may bind data long after construction, so freezing
            # at wrap time covers every entry point uniformly.
            super().update(_finalize_loaded_value(dict_data))

    def raise_if_not_loaded(self):
        if self.loaded:
            return

        error_suffix = (
            "This combination of model, system, backend, and backend version is not supported by AIC in SILICON mode."
        )

        if not os.path.exists(self.filepath):
            raise PerfDataNotAvailableError(
                f"Error loading silicon data for op {self.op_name_enum}: "
                f"File does not exist at {self.filepath}. "
                f"{error_suffix}"
            )
        raise PerfDataNotAvailableError(
            f"Unknown error loading {self.op_name_enum} data from {self.filepath}. {error_suffix}"
        )

    def __getitem__(self, key):
        self.raise_if_not_loaded()
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        self.raise_if_not_loaded()
        return super().__setitem__(key, value)

    def __contains__(self, key):
        self.raise_if_not_loaded()
        return super().__contains__(key)


class _LazySupportMatrix:
    """Dict-like ``database.supported_quant_mode`` that resolves each key
    on first read.

    Reading a key triggers ``OpClass.load_data(database)`` on the op class
    that owns the relevant table, then extracts the supported modes from
    the freshly-bound instance attribute. Subsequent reads of the same
    key return the memoized list. ``load_data`` itself is idempotent and
    early-exits on cache hit, so repeated access is O(1).

    The catalog of valid keys is fixed per backend at construction time
    and mirrors the four branches of the previous ``_update_support_matrix``.
    Reading a key that doesn't apply to the active backend raises
    ``KeyError``, matching the previous dict semantics — callers that
    expect ``key in db.supported_quant_mode`` checks (e.g.
    ``supported.get(context_attn_key, [])`` in ``task.py``) work
    unchanged because ``get()`` returns the default for both unknown keys
    and resolved-to-empty keys.

    Instance assignment (``db.supported_quant_mode = {...}``) replaces
    the matrix entirely; both per-key reads and the lazy contract no
    longer apply on the overwritten value. The pre-refactor
    ``_update_support_matrix`` method continues to work and produces a
    plain dict snapshot that overwrites the lazy matrix in place.
    """

    # Catalog mirrors the four branches of the previous
    # ``_update_support_matrix``. Backends absent from the map produce an
    # empty matrix.
    _BACKEND_KEYS: ClassVar[dict[str, tuple[str, ...]]] = {
        "sglang": (
            "gemm",
            "context_attention",
            "generation_attention",
            "context_mla",
            "context_mla_granular",
            "generation_mla",
            "dsa_context_module",
            "dsa_generation_module",
            "deepseek_v4_context_module",
            "deepseek_v4_generation_module",
            "mla_bmm",
            "nccl",
            "moe",
            "wideep_context_moe",
            "wideep_generation_moe",
            "wideep_context_mla",
            "wideep_generation_mla",
            "dsv4_megamoe_module",
        ),
        "trtllm": (
            "gemm",
            "context_attention",
            "generation_attention",
            "context_mla",
            "context_mla_granular",
            "generation_mla",
            "dsa_context_module",
            "dsa_generation_module",
            "deepseek_v4_context_module",
            "deepseek_v4_generation_module",
            "mla_bmm",
            "nccl",
            "moe",
        ),
        "vllm": (
            "gemm",
            "context_attention",
            "generation_attention",
            "context_mla",
            "context_mla_granular",
            "generation_mla",
            "dsa_context_module",
            "dsa_generation_module",
            "deepseek_v4_context_module",
            "deepseek_v4_generation_module",
            "mla_bmm",
            "moe",
            "nccl",
        ),
    }

    def __init__(self, database: PerfDatabase):
        self._database = database
        self._resolved: dict[str, list[str]] = {}
        self._keys: tuple[str, ...] = self._BACKEND_KEYS.get(database.backend, ())

    def __getitem__(self, key: str) -> list[str]:
        if key in self._resolved:
            return self._resolved[key]
        if key not in self._keys:
            raise KeyError(key)
        value = self._resolve(key)
        self._resolved[key] = value
        return value

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        return key in self._keys

    def __iter__(self):
        return iter(self._keys)

    def keys(self):
        return list(self._keys)

    def values(self):
        return [self[k] for k in self._keys]

    def items(self):
        return [(k, self[k]) for k in self._keys]

    def __len__(self) -> int:
        return len(self._keys)

    def __repr__(self) -> str:
        return f"_LazySupportMatrix(backend={self._database.backend!r}, resolved={list(self._resolved)})"

    # --- Per-key resolvers ----------------------------------------------------
    #
    # Each resolver triggers ``load_data`` on the relevant op class(es), then
    # extracts the modes from the instance attributes those ``load_data`` calls
    # bind. Local imports keep the lazy contract: building a database doesn't
    # import op modules until the matrix is first read.

    def _resolve(self, key: str) -> list[str]:
        db = self._database
        if key == "gemm":
            from aiconfigurator_core.sdk.operations.gemm import GEMM

            GEMM.load_data(db)
            return _gemm_key_names(db)

        if key == "context_attention":
            from aiconfigurator_core.sdk.operations.attention import ContextAttention

            ContextAttention.load_data(db)
            return _enum_key_names(getattr(db, "_context_attention_data", None))

        if key == "generation_attention":
            from aiconfigurator_core.sdk.operations.attention import GenerationAttention

            GenerationAttention.load_data(db)
            return _enum_key_names(getattr(db, "_generation_attention_data", None))

        if key == "context_mla":
            from aiconfigurator_core.sdk.operations.mla import ContextMLA, MLAModule

            ContextMLA.load_data(db)
            MLAModule.load_data(db)
            return _merge_key_names(
                getattr(db, "_context_mla_data", None),
                getattr(db, "_context_mla_module_data", None),
            )

        if key == "context_mla_granular":
            # Granular-table-only capability: the trtllm wideep context path
            # queries the granular context_mla table directly (no module
            # primary), so module-only slices must not count for it.
            from aiconfigurator_core.sdk.operations.mla import ContextMLA

            ContextMLA.load_data(db)
            return _enum_key_names(getattr(db, "_context_mla_data", None))

        if key == "generation_mla":
            from aiconfigurator_core.sdk.operations.mla import GenerationMLA, MLAModule

            GenerationMLA.load_data(db)
            MLAModule.load_data(db)
            # Both granular and module data key on kv_cache_dtype at the top
            # level (generation MLA has no fmha axis).
            return _merge_key_names(
                getattr(db, "_generation_mla_data", None),
                getattr(db, "_generation_mla_module_data", None),
            )

        if key == "dsa_context_module":
            from aiconfigurator_core.sdk.operations.dsa import ContextDSAModule

            ContextDSAModule.load_data(db)
            return _enum_key_names(getattr(db, "_context_dsa_module_data", None))

        if key == "dsa_generation_module":
            from aiconfigurator_core.sdk.operations.dsa import GenerationDSAModule

            GenerationDSAModule.load_data(db)
            return _enum_key_names(getattr(db, "_generation_dsa_module_data", None))

        if key == "deepseek_v4_context_module":
            from aiconfigurator_core.sdk.operations.dsv4 import ContextDeepSeekV4AttentionModule

            ContextDeepSeekV4AttentionModule.load_data(db)
            return _enum_key_names(getattr(db, "_context_deepseek_v4_attention_module_data", None))

        if key == "deepseek_v4_generation_module":
            from aiconfigurator_core.sdk.operations.dsv4 import GenerationDeepSeekV4AttentionModule

            GenerationDeepSeekV4AttentionModule.load_data(db)
            return _enum_key_names(getattr(db, "_generation_deepseek_v4_attention_module_data", None))

        if key == "mla_bmm":
            from aiconfigurator_core.sdk.operations.mla import MLABmm

            MLABmm.load_data(db)
            return _enum_key_names(getattr(db, "_mla_bmm_data", None))

        if key == "nccl":
            from aiconfigurator_core.sdk.operations.communication import NCCL

            NCCL.load_data(db)
            # vllm matrix prefers ``_nccl_data`` but falls back to ``_oneccl_data``
            # because the original code used ``... or getattr(self, "_oneccl_data", None)``.
            primary = getattr(db, "_nccl_data", None)
            if db.backend == "vllm" and not primary:
                primary = getattr(db, "_oneccl_data", None)
            return _enum_key_names(primary)

        if key == "moe":
            from aiconfigurator_core.sdk.operations.moe import MoE

            MoE.load_data(db)
            return _enum_key_names(getattr(db, "_moe_data", None))

        if key == "wideep_context_moe":
            from aiconfigurator_core.sdk.operations.moe import MoE

            MoE.load_data(db)
            return _enum_key_names(getattr(db, "_wideep_context_moe_data", None))

        if key == "wideep_generation_moe":
            from aiconfigurator_core.sdk.operations.moe import MoE

            MoE.load_data(db)
            return _enum_key_names(getattr(db, "_wideep_generation_moe_data", None))

        if key == "wideep_context_mla":
            from aiconfigurator_core.sdk.operations.mla import WideEPContextMLA

            WideEPContextMLA.load_data(db)
            modes: set[str] = set()
            data = getattr(db, "_wideep_context_mla_data", None) or {}
            for kernel_source in data:
                for quant_mode in data[kernel_source]:
                    modes.add(quant_mode.name if hasattr(quant_mode, "name") else str(quant_mode))
            return sorted(modes)

        if key == "wideep_generation_mla":
            from aiconfigurator_core.sdk.operations.mla import WideEPGenerationMLA

            WideEPGenerationMLA.load_data(db)
            modes = set()
            data = getattr(db, "_wideep_generation_mla_data", None) or {}
            for kernel_source in data:
                for kv_cache_dtype in data[kernel_source]:
                    modes.add(kv_cache_dtype.name if hasattr(kv_cache_dtype, "name") else str(kv_cache_dtype))
            return sorted(modes)

        if key == "dsv4_megamoe_module":
            from aiconfigurator_core.sdk.operations.dsv4 import DeepSeekV4MegaMoEModule

            DeepSeekV4MegaMoEModule.load_data(db)
            modes: set[str] = set()
            data = getattr(db, "_dsv4_megamoe_module_data", None) or {}
            for phase in data:
                for kernel_source in data[phase]:
                    for kernel_dtype in data[phase][kernel_source]:
                        for quant_mode in data[phase][kernel_source][kernel_dtype]:
                            modes.add(quant_mode.name if hasattr(quant_mode, "name") else str(quant_mode))
            return sorted(modes)

        # Unreachable given the _keys gate in __getitem__, but stay defensive.
        raise KeyError(key)


def _enum_key_names(data) -> list[str]:
    """Safely extract Enum key names from a mapping.

    Many perf tables are optional and loaders return ``None`` when data
    files are missing. Treat missing/empty tables as supporting no modes."""
    if not data:
        return []
    names: list[str] = []
    for key in data:
        names.append(key.name if hasattr(key, "name") else str(key))
    return names


def _merge_key_names(*sources) -> list[str]:
    """Merge top-level Enum key names from multiple data sources."""
    merged: set[str] = set()
    for source in sources:
        merged.update(_enum_key_names(source))
    return sorted(merged)


def _contains_quant_mode(data, quant_mode: common.GEMMQuantMode) -> bool:
    if not data:
        return False
    try:
        return quant_mode in data
    except PerfDataNotAvailableError:
        return False


def _gemm_key_names(database) -> list[str]:
    """Return GEMM modes, deriving static FP8 from dynamic FP8 plus overheads."""
    names = set(_enum_key_names(getattr(database, "_gemm_data", None)))
    fp8_static_name = common.GEMMQuantMode.fp8_static.name
    names.discard(fp8_static_name)
    if (
        _contains_quant_mode(getattr(database, "_gemm_data", None), common.GEMMQuantMode.fp8)
        and _contains_quant_mode(getattr(database, "_compute_scale_data", None), common.GEMMQuantMode.fp8)
        and _contains_quant_mode(getattr(database, "_scale_matrix_data", None), common.GEMMQuantMode.fp8)
    ):
        names.add(fp8_static_name)
    return sorted(names)


# ``comm`` is the one family design §6.5 rule 5 hard-excludes from every reuse
# channel (NCCL curves are topology-bound; shape-filling across versions is
# wrong there). Detected structurally off the op's resolved primary path.
_UNPARSEABLE_SIBLING_VERSION_WARNED: set[tuple[str, str, str]] = set()


def _warn_unparseable_sibling_version_once(system_data_root: str, backend: str, version: str) -> None:
    """One-time-per-(tree, backend, version) warning for a sibling version string
    that fails PEP 440 parsing — it cannot be ordered against the requested
    version, so it's excluded from implicit nearest-earlier fallback entirely
    (design §6.2). An explicit ``reuse.yaml`` declaration still works for it."""
    key = (system_data_root, backend, version)
    if key in _UNPARSEABLE_SIBLING_VERSION_WARNED:
        return
    _UNPARSEABLE_SIBLING_VERSION_WARNED.add(key)
    logger.warning(
        "Sibling version %r of backend %s is not PEP 440-parseable; excluded from "
        "implicit nearest-earlier fallback (Collector V3 design §6.2). Declare an "
        "explicit reuse.yaml entry if this version's data should be reused.",
        version,
        backend,
    )


class PerfDatabase:
    """
    The perf database for a given system, backend and version

    Attributes:
        system (str): the system name
        backend (str): the backend name
        version (str): the version name
        system_spec (dict): the system spec
        _default_database_mode (common.DatabaseMode): the default mode of the database
        _gemm_data (dict): the gemm data
        _context_attention_data (dict): the context attention data
        _generation_attention_data (dict): the generation attention data
        _custom_allreduce_data (dict): the custom allreduce data
        _moe_data (dict): the moe data
        _context_mla_data (dict): the context mla data
        _generation_mla_data (dict): the generation mla data
        _nccl_data (dict): the nccl data
        _mla_bmm_data (dict): the mla bmm data
        SGLang wideep:
        _wideep_context_moe_data (dict): the wideep context moe data
        _wideep_generation_moe_data (dict): the wideep generation moe data
        _wideep_context_mla_data (dict): the wideep context mla data
        _wideep_generation_mla_data (dict): the wideep generation mla data
        _wideep_deepep_normal_data (dict): the wideep deepep normal data
        _wideep_deepep_ll_data (dict): the wideep deepep ll data
        TensorRT-LLM wideep:
        _wideep_moe_compute_data (dict): the wideep moe compute data (pure computation, no all2all)
        _trtllm_alltoall_data (dict): the wideep all2all data (prepare, dispatch, combine)

    Methods:
        query_gemm: query the gemm data
        query_context_attention: query the context attention data
        query_generation_attention: query the generation attention data
        query_context_mla: query the context mla data
        query_generation_mla: query the generation mla data
        query_nccl: query the nccl data
        query_mla_bmm: query the mla bmm data
        query_mem_op: query the mem op data
        query_p2p: query the p2p data
        query_custom_allreduce: query the custom allreduce data
        query_moe: query the moe data
    """

    def __init__(
        self,
        system: str,
        backend: str,
        version: str,
        systems_root: str = "./systems",
        database_mode: str | None = None,
        shared_layer: bool | None = None,
        strict_provenance: bool | None = None,
    ) -> None:
        """
        Initialize the perf database.

        Args:
            database_mode: drives the shared-layer load behavior. The default
                mode, `"SILICON"`, and `"HYBRID"` enable sibling-row inheritance
                (including `kernel_source=default` fallback rows); explicit
                formula-only modes keep it off. Doesn't change which rows are
                interpolated at query time; that's controlled by
                `set_default_database_mode`.
            shared_layer: explicit shared-layer override. ``None`` (default)
                derives the flag from ``database_mode`` as described above;
                ``False`` loads only the active backend/version's own rows even
                under SILICON (used by regression harnesses to pin per-version
                behavior); ``True`` forces sibling inheritance on.
            strict_provenance: fail-closed provenance mode (Collector V3 design
                §5/§7.4). ``None`` (default) resolves from the
                ``AIC_STRICT_PROVENANCE`` env var (truthy: ``"1"``/``"true"``).
                Stored as ``self.strict_provenance``; the load-time validation
                itself runs in :func:`get_database` (the loader entry point),
                not here -- constructing a ``PerfDatabase`` directly (as tests
                commonly do against synthetic trees) performs no provenance
                validation of its own.
        """
        self.system = system
        self.backend = backend
        self.version = version
        self.systems_root = systems_root
        self.strict_provenance: bool = _strict_provenance_enabled(strict_provenance)
        self._shared_layer_mode = _shared_layer_enabled(database_mode) if shared_layer is None else bool(shared_layer)
        # Which empirical transfer kinds are permitted (HYBRID/EMPIRICAL only). All on by
        # default = current behaviour; set_transfer_policy() narrows it for fine-grained
        # HYBRID control. Read at query time by op get_empirical, so it can be retuned on
        # the (cached, shared) instance like the default database mode.
        self._transfer_policy: frozenset[common.TransferKind] = common.ALL_TRANSFERS
        with open(os.path.join(systems_root, system + ".yaml")) as f:
            self.system_spec = SystemSpec(yaml.load(f, Loader=yaml.SafeLoader))
        self._default_database_mode = common.DatabaseMode.SILICON  # default mode is SILICON

        # Per-op-file source diagnostics (design §6.5 guardrail), lazily populated
        # by ``_build_op_sources`` as each op loads: op_file basename -> list of
        # {version, path, channel, exists} for every ADMITTED source, in priority
        # order. Granularity is admitted sources, not per-row/per-shape
        # attribution — two different tables in the same source file share one
        # entry even if only one of them actually contributed rows.
        self.data_provenance: dict[str, list[dict[str, object]]] = {}
        # (op_file_basename, sibling_path) pairs already warned about as
        # low-fidelity cross-backend fallbacks — see _build_op_sources.
        self._fallback_warned: set[tuple[str, str]] = set()

        # lazy per-op data ownership: every op class owns its CSV data and loads it on first query
        # via ``OpClass.load_data(database)``. No eager warm-up here — each op
        # opens its data file the first time a query (or the lazy support
        # matrix below) needs it. ``PerfDatabase()`` opens zero CSVs.
        self.supported_quant_mode = _LazySupportMatrix(self)
        self._finalize_loaded_data()
        self._is_query_view = False

    def _finalize_loaded_data(self) -> None:
        """Stop loader-time defaultdicts from mutating database state after construction."""
        for attr, value in list(vars(self).items()):
            setattr(self, attr, _finalize_loaded_value(value))

    def _update_support_matrix(self):
        """
        Update the support matrix
        """

        def _enum_key_names(data: dict | None) -> list[str]:
            """
            Safely extract Enum key names from a mapping.

            Many perf tables are optional and loaders return None when data files
            are missing. Treat missing/empty tables as supporting no modes.
            """
            if not data:
                return []
            names: list[str] = []
            for key in data:
                names.append(key.name if hasattr(key, "name") else str(key))
            return names

        def _merge_key_names(*sources: dict | None) -> list[str]:
            """Merge top-level Enum key names from multiple data sources."""
            merged: set[str] = set()
            for data in sources:
                merged.update(_enum_key_names(data))
            return sorted(merged)

        def _generation_mla_kv_modes() -> list[str]:
            """Collect kv_cache_dtype names for generation MLA from both sources.

            Both granular and module data key on kv_cache_dtype at the top
            level (generation MLA has no fmha axis).
            """
            return _merge_key_names(
                getattr(self, "_generation_mla_data", None),
                getattr(self, "_generation_mla_module_data", None),
            )

        def _dsv4_megamoe_modes(data: dict | None) -> list[str]:
            """Collect MoE quant-mode names from DSv4 MegaMoE data.

            The table is keyed ``phase -> kernel_source -> kernel_dtype -> quant_mode -> ...``.
            """
            if not data:
                return []
            modes: set[str] = set()
            for phase in data:
                for kernel_source in data[phase]:
                    for kernel_dtype in data[phase][kernel_source]:
                        for quant_mode in data[phase][kernel_source][kernel_dtype]:
                            modes.add(quant_mode.name if hasattr(quant_mode, "name") else str(quant_mode))
            return sorted(modes)

        # For sglang backend, context_mla_data and generation_mla_data have kernel_source as first
        # level
        # We need to collect quant_modes from the nested structure
        if self.backend == "sglang":
            wideep_context_mla_modes = set()
            wideep_context_mla_data = getattr(self, "_wideep_context_mla_data", None) or {}
            for kernel_source in wideep_context_mla_data:
                for quant_mode in wideep_context_mla_data[kernel_source]:
                    wideep_context_mla_modes.add(quant_mode.name)

            wideep_generation_mla_modes = set()
            wideep_generation_mla_data = getattr(self, "_wideep_generation_mla_data", None) or {}
            for kernel_source in wideep_generation_mla_data:
                for kv_cache_dtype in wideep_generation_mla_data[kernel_source]:
                    wideep_generation_mla_modes.add(kv_cache_dtype.name)

            self.supported_quant_mode = {
                "gemm": _gemm_key_names(self),
                "context_attention": _enum_key_names(getattr(self, "_context_attention_data", None)),
                "generation_attention": _enum_key_names(getattr(self, "_generation_attention_data", None)),
                "context_mla": _merge_key_names(
                    getattr(self, "_context_mla_data", None),
                    getattr(self, "_context_mla_module_data", None),
                ),
                "context_mla_granular": _enum_key_names(getattr(self, "_context_mla_data", None)),
                "generation_mla": _generation_mla_kv_modes(),
                "dsa_context_module": _enum_key_names(getattr(self, "_context_dsa_module_data", None)),
                "dsa_generation_module": _enum_key_names(getattr(self, "_generation_dsa_module_data", None)),
                "deepseek_v4_context_module": _enum_key_names(
                    getattr(self, "_context_deepseek_v4_attention_module_data", None)
                ),
                "deepseek_v4_generation_module": _enum_key_names(
                    getattr(self, "_generation_deepseek_v4_attention_module_data", None)
                ),
                "mla_bmm": _enum_key_names(getattr(self, "_mla_bmm_data", None)),
                "nccl": _enum_key_names(getattr(self, "_nccl_data", None)),
                "moe": _enum_key_names(getattr(self, "_moe_data", None)),
                "wideep_context_moe": _enum_key_names(getattr(self, "_wideep_context_moe_data", None)),
                "wideep_generation_moe": _enum_key_names(getattr(self, "_wideep_generation_moe_data", None)),
                "wideep_context_mla": list(wideep_context_mla_modes),
                "wideep_generation_mla": list(wideep_generation_mla_modes),
                "dsv4_megamoe_module": _dsv4_megamoe_modes(getattr(self, "_dsv4_megamoe_module_data", None)),
            }
        elif self.backend == "trtllm":
            self.supported_quant_mode = {
                "gemm": _gemm_key_names(self),
                "context_attention": _enum_key_names(getattr(self, "_context_attention_data", None)),
                "generation_attention": _enum_key_names(getattr(self, "_generation_attention_data", None)),
                "context_mla": _merge_key_names(
                    getattr(self, "_context_mla_data", None),
                    getattr(self, "_context_mla_module_data", None),
                ),
                "context_mla_granular": _enum_key_names(getattr(self, "_context_mla_data", None)),
                "generation_mla": _generation_mla_kv_modes(),
                "dsa_context_module": _enum_key_names(getattr(self, "_context_dsa_module_data", None)),
                "dsa_generation_module": _enum_key_names(getattr(self, "_generation_dsa_module_data", None)),
                "deepseek_v4_context_module": _enum_key_names(
                    getattr(self, "_context_deepseek_v4_attention_module_data", None)
                ),
                "deepseek_v4_generation_module": _enum_key_names(
                    getattr(self, "_generation_deepseek_v4_attention_module_data", None)
                ),
                "mla_bmm": _enum_key_names(getattr(self, "_mla_bmm_data", None)),
                "nccl": _enum_key_names(getattr(self, "_nccl_data", None)),
                "moe": _enum_key_names(getattr(self, "_moe_data", None)),
            }
        elif self.backend == "vllm":
            self.supported_quant_mode = {
                "gemm": _gemm_key_names(self),
                "context_attention": _enum_key_names(getattr(self, "_context_attention_data", None)),
                "generation_attention": _enum_key_names(getattr(self, "_generation_attention_data", None)),
                "context_mla": _merge_key_names(
                    getattr(self, "_context_mla_data", None),
                    getattr(self, "_context_mla_module_data", None),
                ),
                "context_mla_granular": _enum_key_names(getattr(self, "_context_mla_data", None)),
                "generation_mla": _generation_mla_kv_modes(),
                "dsa_context_module": _enum_key_names(getattr(self, "_context_dsa_module_data", None)),
                "dsa_generation_module": _enum_key_names(getattr(self, "_generation_dsa_module_data", None)),
                "deepseek_v4_context_module": _enum_key_names(
                    getattr(self, "_context_deepseek_v4_attention_module_data", None)
                ),
                "deepseek_v4_generation_module": _enum_key_names(
                    getattr(self, "_generation_deepseek_v4_attention_module_data", None)
                ),
                "mla_bmm": _enum_key_names(getattr(self, "_mla_bmm_data", None)),
                "moe": _enum_key_names(getattr(self, "_moe_data", None)),
                "nccl": _enum_key_names(getattr(self, "_nccl_data", None) or getattr(self, "_oneccl_data", None)),
            }
        else:
            self.supported_quant_mode = {}

    def _build_op_sources(
        self,
        op_filename_enum: PerfDataFilename,
        primary_path: str,
        system_data_root: str,
    ) -> list[tuple[str, Optional[set[str]]]]:
        """Priority-ordered source files for one op, resolved BY THE ENGINE.

        The four-channel resolution (primary / declared_reuse / fallback /
        cross_backend — Collector V3 design §6) lives in the compiled engine
        (`perf_database/source_resolution.rs`) since the deprecation-cleanup
        PR; the engine's own table loads run the same code, so this report
        always matches what the engine loaded. This wrapper keeps the
        established Python surface: it returns the `(file_path,
        kernel_source_filter)` list, records every ADMITTED source into
        ``self.data_provenance[op_file_basename]`` (entry shape
        ``{version, path, channel, exists}``), and re-emits the resolver's
        structured warnings through the module's established registries
        (legacy-marker/layout deprecations, unparseable siblings, strict
        sidecar problems, low-fidelity cross-backend fills). Strict-mode
        failures (malformed sidecar metadata with ``strict_provenance``)
        raise ``ValueError`` exactly like the retired in-Python resolver.
        """
        import aiconfigurator_core as _core

        report = json.loads(
            _core.resolve_op_sources_report_json(
                self.systems_root,
                system_data_root,
                self.backend,
                self.version,
                op_filename_enum.value,
                primary_path,
                enable_shared_layer=self.enable_shared_layer,
                strict=self.strict_provenance,
            )
        )
        self._emit_resolver_warnings(report["warnings"])
        self.data_provenance[op_filename_enum.value] = [
            {key: record[key] for key in ("version", "path", "channel", "exists")} for record in report["records"]
        ]
        return [
            (record["path"], set(record["ks_filter"]) if record["ks_filter"] is not None else None)
            for record in report["records"]
        ]

    def _emit_resolver_warnings(self, events: list[dict]) -> None:
        """Re-emit the engine resolver's structured warning events through the
        module's own loggers and warn-once registries, preserving the retired
        in-Python resolver's exact messages and dedupe granularity (global
        per-tree for layout/marker/unparseable warnings; per-database for the
        low-fidelity cross-backend warning)."""
        for event in events:
            kind, args = event["kind"], event["args"]
            if kind == "primary_veto":
                logger.warning(
                    "Not admitting primary source %s for %s: version dir %s carries legacy %s "
                    "without table/shape-level coverage metadata; sibling sources may still fill.",
                    args[0],
                    args[1],
                    args[2],
                    INCOMPLETE_MARKER,
                )
            elif kind == "legacy_layout":
                if args[0] not in _LEGACY_LAYOUT_WARNED:
                    _LEGACY_LAYOUT_WARNED.add(args[0])
                    logger.warning(
                        "Legacy perf-data layout (<backend>/<version>) found under %s; "
                        "migrate to <family>/<backend>/<version> (Collector V3 design §3)",
                        args[0],
                    )
            elif kind == "legacy_marker":
                _warn_legacy_marker_once(args[0], args[1], args[2])
            elif kind == "malformed_sidecar":
                _warn_strict_provenance_once("malformed-sidecar", args[0], args[1])
            elif kind == "malformed_reuse":
                _warn_strict_provenance_once("malformed-reuse", args[0], args[1])
            elif kind == "strict_provenance":
                _warn_strict_provenance_once(args[0], args[1], args[2])
            elif kind == "duplicate_declared":
                logger.debug(
                    "Duplicate declared-reuse entry for table %s from_version %s under %s; first occurrence wins.",
                    args[0],
                    args[1],
                    args[2],
                )
            elif kind == "unparseable_sibling":
                _warn_unparseable_sibling_version_once(args[0], args[1], args[2])
            elif kind == "low_fidelity":
                if (args[0], args[1]) not in self._fallback_warned:
                    self._fallback_warned.add((args[0], args[1]))
                    logger.warning(
                        "Loading low-fidelity fallback rows for %s from %s. Queries "
                        "returning these rows are framework-implicit and may differ "
                        "from real backend behavior.",
                        args[0],
                        args[1],
                    )
            else:  # pragma: no cover - future resolver kinds surface loudly
                logger.warning("Unknown source-resolver warning %s: %s", kind, args)

    def _materialize_source_reports(self) -> None:
        """Resolve every op file once (provenance + warnings), mirroring the
        retired eager wire sweep (`_compute_perf_db_sources` resolved all op
        files while building a probe spec). The engine resolves its OWN
        sources at load; this Python-side sweep exists so ``data_provenance``
        fills and resolver warnings reach the module logger at the same
        moment they used to — the first engine-backed table fetch. Idempotent
        per database instance."""
        if self.__dict__.get("_source_reports_materialized"):
            return
        system_data_root = os.path.join(self.systems_root, self.system_spec["data_dir"])
        for filename_enum in PerfDataFilename:
            primary_path = resolve_op_data_path(system_data_root, self.backend, self.version, filename_enum.value)
            self._build_op_sources(filename_enum, primary_path, system_data_root)
        # Set only AFTER the sweep completes: a strict-provenance failure
        # mid-sweep must leave the next call free to retry (and fail closed
        # again) instead of short-circuiting with data_provenance half-filled.
        self._source_reports_materialized = True

    def is_inter_node(self, num_gpus: int) -> bool:
        """
        Check if the number of GPUs is an inter node
        """
        return num_gpus > self.system_spec["node"]["num_gpus_per_node"]

    def _get_p2p_bandwidth(self, num_gpus: int) -> float:
        """Thin wrapper — delegates to ``SystemSpec.get_p2p_bandwidth``."""
        return self.system_spec.get_p2p_bandwidth(num_gpus)

    def set_default_database_mode(self, mode: common.DatabaseMode) -> None:
        """
        Set the default database mode
        """
        _reject_retired_database_mode(mode)
        if getattr(self, "_is_query_view", False) and mode != self._default_database_mode:
            raise RuntimeError(
                "A cached query view has immutable mode/policy state; request a different view with "
                "get_database_view()."
            )
        if mode != self._default_database_mode:
            self.clear_runtime_caches()
            from aiconfigurator_core.sdk.operations import util_empirical

            util_empirical.clear_grid_cache()  # mode change alters which data/transfers feed grids
            self._default_database_mode = mode

    def get_default_database_mode(self) -> common.DatabaseMode:
        """
        Get the default database mode
        """
        return self._default_database_mode

    def set_transfer_policy(self, spec) -> None:
        """Set which empirical transfer kinds are permitted (fine-grained HYBRID control).

        ``spec`` is anything :func:`common.resolve_transfer_policy` accepts: ``None``
        (all), a preset name, a :class:`common.TransferKind`, or an iterable of those.
        Clears runtime caches so already-cached query results don't mask the new policy.
        """
        policy = common.resolve_transfer_policy(spec)
        if getattr(self, "_is_query_view", False) and policy != self._transfer_policy:
            raise RuntimeError(
                "A cached query view has immutable mode/policy state; request a different view with "
                "get_database_view()."
            )
        if policy != self._transfer_policy:
            self.clear_runtime_caches()
            from aiconfigurator_core.sdk.operations import util_empirical

            # The util grid cache key doesn't encode the policy (xshape/xquant share a
            # key), so a stale grid would mask the new policy -- drop it.
            util_empirical.clear_grid_cache()
            self._transfer_policy = policy

    @property
    def transfer_policy(self) -> frozenset[common.TransferKind]:
        """Empirical transfer kinds currently permitted (see :class:`common.TransferKind`).

        Defaults to all kinds when unset (e.g. a bare instance), so attribute
        introspection (``dir``/``clear_runtime_caches``) never trips on it."""
        return getattr(self, "_transfer_policy", common.ALL_TRANSFERS)

    @property
    def enable_shared_layer(self) -> bool:
        """Whether sibling-version shared-layer sourcing is active (read at op load time
        and in op cache keys). Shared rows are collected silicon data, so the default,
        SILICON, and HYBRID modes enable them independently of empirical transfer policy;
        EMPIRICAL and SOL modes keep them disabled."""
        return getattr(self, "_shared_layer_mode", False)

    def clear_runtime_caches(self) -> None:
        """Clear cached query state while preserving loaded op data."""
        _cached_configured_database_view.cache_clear()
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            cache_clear = getattr(attr, "cache_clear", None)
            if callable(cache_clear):
                cache_clear()

    def moe_a2a_coverage(
        self,
        hidden_size: int,
        topk: int,
        num_experts: int,
        model_quantization=None,
        inference_phase: str | None = None,
    ) -> dict[str, set[tuple[int, int]]]:
        """Probe moe_a2a coverage for one model shape (PR 2's enumerator contract).

        Returns ``comm_backend -> {(ep_size, node_num)}`` where BOTH the
        dispatch AND combine phases carry a non-empty token curve for the
        shape. When model quantization and inference phase are supplied, each
        phase must exist under the exact serving communication dtype; omitted
        arguments preserve the legacy ANY-dtype introspection behavior. SMS
        remains an ANY axis and prepare is not required. Read-only key walk over the table bound by
        ``MoEAllToAll.load_data`` — no query execution, and non-vivifying
        (``.get`` only: injected stores may be auto-vivifying defaultdicts).
        An absent or unloaded table — including a ``LoadedOpData`` wrapping
        ``None``, which raises on item access but is falsy — yields ``{}``.
        Deliberately not lru_cached: the returned sets are mutable.
        """
        from aiconfigurator_core.sdk.operations.moe_comm import (
            MOE_A2A_BACKENDS,
            MoEAllToAll,
            communication_dtype_for,
        )

        MoEAllToAll.load_data(self)
        table = self._moe_a2a_data
        if not table:
            return {}

        def pairs_for(by_phase, phase: str, required_dtype: str | None) -> set[tuple[int, int]]:
            pairs: set[tuple[int, int]] = set()
            by_dtype = by_phase.get(phase) or {}
            dtype_slices = by_dtype.values() if required_dtype is None else (by_dtype.get(required_dtype) or {},)
            for by_ep in dtype_slices:
                for ep_size, by_node in by_ep.items():
                    for node_num, by_hidden in by_node.items():
                        by_sms = ((by_hidden.get(hidden_size) or {}).get(topk) or {}).get(num_experts) or {}
                        if any(by_sms.values()):  # ANY sms with a non-empty token curve
                            pairs.add((ep_size, node_num))
            return pairs

        coverage: dict[str, set[tuple[int, int]]] = {}
        for comm_backend, by_phase in table.items():
            required = {"dispatch": None, "combine": None}
            if model_quantization is not None and inference_phase is not None:
                backend_spec = MOE_A2A_BACKENDS.get(comm_backend)
                if backend_spec is None or inference_phase not in backend_spec.inference_phases:
                    continue
                try:
                    required = {
                        phase: communication_dtype_for(
                            system=self.system,
                            comm_backend=comm_backend,
                            model_quantization=model_quantization,
                            communication_phase=phase,
                            inference_phase=inference_phase,
                        )
                        for phase in ("dispatch", "combine")
                    }
                except ValueError:
                    continue
            covered = pairs_for(by_phase, "dispatch", required["dispatch"]) & pairs_for(
                by_phase, "combine", required["combine"]
            )
            if covered:
                coverage[comm_backend] = covered
        return coverage

    def moe_expert_compute_coverage(
        self,
        hidden_size: int,
        inter_size: int,
        topk: int,
        num_experts: int,
        quant_mode: common.MoEQuantMode,
        inference_phase: str,
    ) -> set[int]:
        """Probe moe_ep compute coverage for one model shape (PR 2's enumerator contract).

        Returns the ``{moe_ep_size}`` set with a non-empty token curve for
        the shape, unioned across ANY ``kernel_source``, ANY workload
        distribution, and ANY ``num_slots``, with ``moe_tp_size`` fixed to 1
        (the large-EP family is EP-only). Same read-only, non-vivifying,
        never-raising contract as :meth:`moe_a2a_coverage`; an absent or
        unloaded table yields an empty set. Deliberately not lru_cached: the
        returned set is mutable.
        """
        from aiconfigurator_core.sdk.operations.moe_comm import MoEExpertCompute

        MoEExpertCompute.load_data(self)
        table = self._moe_ep_data
        if not table:
            return set()

        covered: set[int] = set()
        for by_quant in table.values():  # ANY kernel_source
            for by_phase in (by_quant.get(quant_mode) or {}).values():  # ANY distribution
                by_slots = ((by_phase.get(inference_phase) or {}).get(topk) or {}).get(num_experts) or {}
                for by_hidden in by_slots.values():  # ANY num_slots
                    by_ep = ((by_hidden.get(hidden_size) or {}).get(inter_size) or {}).get(1) or {}  # moe_tp == 1
                    covered.update(ep_size for ep_size, tokens in by_ep.items() if tokens)
        return covered

    def legacy_moe_compute_coverage(
        self,
        hidden_size: int,
        inter_size: int,
        topk: int,
        num_experts: int,
        quant_mode: common.MoEQuantMode,
    ) -> set[int]:
        """Return EP sizes covered by the regular expert-kernel MoE table.

        vLLM and TensorRT-LLM publish expert compute in ``moe_perf.parquet``
        rather than the WideEP-specific compute tables.  This probe is used
        only to gate the compute leg of a graph whose communication is already
        owned by :class:`MoEAllToAll`; it does not enable the legacy
        ``MoEDispatch``/NCCL graph.
        """
        from aiconfigurator_core.sdk.operations.moe import MoE

        MoE.load_data(self)
        table = self._moe_data
        if not table:
            return set()

        covered: set[int] = set()
        for by_topk in (table.get(quant_mode) or {}).values():  # ANY distribution
            by_hidden = ((by_topk.get(topk) or {}).get(num_experts) or {}).get(hidden_size) or {}
            by_ep = (by_hidden.get(inter_size) or {}).get(1) or {}  # moe_tp == 1
            covered.update(ep_size for ep_size, tokens in by_ep.items() if tokens)
        return covered

    # ═══════════════════════════════════════════════════════════════════
    # DSA (DeepSeek Sparse Attention) Queries
    # ═══════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    database_dict = get_all_databases()

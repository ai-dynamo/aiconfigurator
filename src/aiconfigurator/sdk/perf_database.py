# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import importlib.resources as pkg_resources
import logging
import os
import traceback
from collections import UserDict, defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import ClassVar, Optional

import yaml

from aiconfigurator.sdk import common
from aiconfigurator.sdk.common import PerfDataFilename, parse_support_matrix_version
from aiconfigurator.sdk.performance_result import PerformanceResult
from aiconfigurator.sdk.system_spec import SystemSpec

databases_cache = defaultdict(lambda: defaultdict(lambda: defaultdict()))
logger = logging.getLogger(__name__)

_SYSTEMS_PATHS: list[str] = [os.fspath(pkg_resources.files("aiconfigurator") / "systems")]
_MISSING_SILICON_DATA_EXCEPTIONS = (KeyError, IndexError)


def _normalize_systems_paths(raw_paths: str | Iterable[str] | None) -> list[str]:
    default_path = os.fspath(pkg_resources.files("aiconfigurator") / "systems")
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
    from aiconfigurator.sdk.operations.base import clear_all_op_caches

    clear_all_op_caches()


def get_systems_paths() -> list[str]:
    return list(_SYSTEMS_PATHS)


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
    default_path = os.fspath(pkg_resources.files("aiconfigurator") / "systems")
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


class PerfDataNotAvailableError(RuntimeError):
    """Raised when required performance data is missing or unsupported for a requested mode.

    This is a *per-op* data-miss raised deep inside a single config's evaluation
    (and in non-sweep paths like validate / single-point estimate), so it is
    deliberately NOT a :class:`~aiconfigurator.sdk.errors.NoResultsError`: a miss
    on one config can be skipped while other configs still produce results.
    Recognize it via :func:`has_perf_data_not_available_cause`.
    """


def has_perf_data_not_available_cause(error: BaseException) -> bool:
    """Return True when an exception or chained cause is a structured perf-data miss."""
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
        if current.__context__ is not None:
            stack.append(current.__context__)
    return False


@functools.cache
def _load_op_kernel_source_manifest_entries(systems_root: str) -> dict[str, tuple[dict, ...]]:
    """Load `<systems_root>/op_kernel_source_manifest.yaml` and group entries by op_file.

    Returns `op_file -> tuple of entries` (each entry has tier, kernel_source, frameworks).
    Used by PerfDatabase to discover which sibling backend/version dirs hold rows that the
    active backend can inherit. Returns an empty dict if the manifest is absent or empty.

    The manifest is generated by `tools/perf_database/audit_kernel_source.py`.
    """
    manifest_path = os.path.join(systems_root, "op_kernel_source_manifest.yaml")
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path) as f:
        data = yaml.safe_load(f) or {}
    accum: dict[str, list[dict]] = defaultdict(list)
    for entry in data.get("groups", []) or []:
        op_file = entry.get("op_file")
        if not op_file:
            continue
        if op_file.endswith(".txt"):
            op_file = f"{os.path.splitext(op_file)[0]}.parquet"
        accum[op_file].append(entry)
    return {key: tuple(value) for key, value in accum.items()}


# ``_read_filtered_rows`` lives in ``operations.base`` so the per-op-module
# loaders can import it without a circular dependency on ``perf_database``
# at module load time. Re-exported here for any external callers that may
# still import it via ``aiconfigurator.sdk.perf_database._read_filtered_rows``.
from aiconfigurator.sdk.operations.base import (  # noqa: F401
    _read_filtered_rows,
    _read_perf_rows,
    _resolve_perf_data_path,
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
                    backend_path = os.path.join(data_dir, backend.value)
                    if not os.path.isdir(backend_path):
                        continue

                    versions = [
                        v
                        for v in os.listdir(backend_path)
                        if not v.startswith(".")
                        and os.path.isdir(os.path.join(backend_path, v))
                        and not os.path.isfile(os.path.join(backend_path, v, "INCOMPLETE.txt"))
                    ]
                    if versions:
                        supported_sets[system][backend.value].update(versions)
            except Exception as e:
                logger.warning(f"Could not process system config {os.path.basename(system_yaml_path)}: {e}")

    supported_dict = defaultdict(lambda: defaultdict(list))
    for system, backend_versions in supported_sets.items():
        for backend, versions in backend_versions.items():
            supported_dict[system][backend] = sorted(versions)

    return supported_dict


def get_latest_database_version(
    system: str,
    backend: str,
    systems_paths: str | list[str] | None = None,
) -> str | None:
    """
    Get the latest database version for a given system and backend
    """
    import re

    if systems_paths is None:
        supported_databases = get_supported_databases()
    else:
        supported_databases = get_supported_databases(systems_paths=systems_paths)
    database_versions = supported_databases.get(system, {}).get(backend, [])
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


def get_database(
    system: str,
    backend: str,
    version: str,
    systems_paths: str | list[str] | None = None,
    allow_missing_data: bool = False,
    database_mode: str | None = None,
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
            estimate-only modes, not SILICON mode.
        database_mode: the mode the caller will query under (`SILICON` / `HYBRID` /
            `EMPIRICAL` / `SOL`). HYBRID auto-enables the shared layer (sibling-row
            inheritance, including `kernel_source=default` fallback rows); other
            modes keep it off so predictions stay bit-identical to main.

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

    shared_flag = (database_mode or "").upper() == "HYBRID"
    missing_data_candidate = None
    for systems_root in systems_paths:
        system_yaml_path = os.path.join(systems_root, f"{system}.yaml")
        if not os.path.isfile(system_yaml_path):
            continue
        cache_key = (systems_root, system, shared_flag)
        try:
            with open(system_yaml_path) as f:
                system_spec = yaml.load(f, Loader=yaml.SafeLoader)
            data_dir = system_spec["data_dir"]
        except Exception:
            logger.warning(f"failed to read system spec at {system_yaml_path}, continuing searching")
            continue

        data_path = os.path.join(systems_root, data_dir, backend, version)
        is_incomplete = os.path.isfile(os.path.join(data_path, "INCOMPLETE.txt"))
        if os.path.exists(data_path) and not is_incomplete:
            try:
                database = databases_cache[cache_key][backend][version]
                return database
            except KeyError:
                logger.info(f"Loading database for {system=}, {backend=}, {version=}")
                try:
                    database = PerfDatabase(
                        system,
                        backend,
                        version,
                        systems_root,
                        database_mode=database_mode,
                    )
                    databases_cache[cache_key][backend][version] = database
                    return database
                except Exception:
                    logger.warning(
                        f"failed to load {system=}, {backend=}, {version=}, continuing searching",
                        exc_info=True,
                    )
        elif allow_missing_data:
            if missing_data_candidate is None:
                missing_data_candidate = (systems_root, cache_key)
        else:
            if is_incomplete:
                logger.warning(f"data path {data_path} is marked incomplete, continuing searching")
            else:
                logger.warning(f"data path {data_path} not found, continuing searching")

    if missing_data_candidate is not None:
        systems_root, cache_key = missing_data_candidate
        try:
            database = databases_cache[cache_key][backend][version]
            return database
        except KeyError:
            logger.info(f"Loading estimate-only database for {system=}, {backend=}, {version=}")
            try:
                database = PerfDatabase(system, backend, version, systems_root, database_mode=database_mode)
                databases_cache[cache_key][backend][version] = database
                return database
            except Exception:
                logger.warning(
                    f"failed to load estimate-only {system=}, {backend=}, {version=}",
                    exc_info=True,
                )

    logger.error(f"failed to get {system=}, {backend=}, {version=}")
    return None


DatabaseRef = tuple[str, str, str, str]
LoadedDatabaseResult = tuple[DatabaseRef, object | None, str | None]


def _as_systems_path_list(systems_paths: str | os.PathLike | Iterable[str] | None) -> list[str]:
    if systems_paths is None:
        return get_systems_paths()
    if isinstance(systems_paths, (str, os.PathLike)):
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
        backend_path = os.path.join(data_dir, backend_name)
        if not os.path.isdir(backend_path):
            continue

        for version in sorted(os.listdir(backend_path)):
            version_path = os.path.join(backend_path, version)
            if version.startswith(".") or not os.path.isdir(version_path):
                continue
            if os.path.isfile(os.path.join(version_path, "INCOMPLETE.txt")):
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
    database_dict[system][backend][version] = database
    databases_cache[(systems_root, system, False)][backend][version] = database


def clear_database_runtime_caches(system: str, backend: str, version: str) -> None:
    """Clear per-query/interpolation caches for one loaded database.

    Also evicts every Operation subclass's class-level CSV cache via
    ``clear_all_op_caches()`` so a subsequent reload reads fresh rows
    from disk — the per-class caches survive the per-instance
    ``clear_runtime_caches()`` and would otherwise serve the prior data.
    """
    seen_database_ids: set[int] = set()
    for cache_key, systems_cache in databases_cache.items():
        _, cached_system, _ = cache_key
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

    from aiconfigurator.sdk.operations.base import clear_all_op_caches

    clear_all_op_caches()


def unload_database(system: str, backend: str, version: str) -> None:
    """Remove one loaded database from every systems-root/shared-mode cache.

    Also evicts every Operation subclass's class-level CSV cache via
    ``clear_all_op_caches()`` so a future ``get_database(...)`` for the
    same ``(system, backend, version)`` rebuilds the op-level caches from
    disk instead of aliasing the stale tables that survived the database
    pop.
    """
    for cache_key in list(databases_cache.keys()):
        _, cached_system, _ = cache_key
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

    from aiconfigurator.sdk.operations.base import clear_all_op_caches

    clear_all_op_caches()


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
# CSV loader re-exports.
#
# Every ``load_*_data`` function lives in the op module that owns the
# data it parses (lazy per-op data ownership). The re-exports below keep the previous
# import paths working for external callers and for legacy
# ``aiconfigurator.sdk.perf_database.<loader>`` patch sites in test
# fixtures (the conftest now patches the new locations directly; these
# survive for code outside this repo).
# ─────────────────────────────────────────────────────────────────────────
from aiconfigurator.sdk.operations.attention import (  # noqa: F401
    load_context_attention_data,
    load_encoder_attention_data,
    load_generation_attention_data,
)
from aiconfigurator.sdk.operations.communication import (  # noqa: F401
    load_custom_allreduce_data,
    load_nccl_data,
)
from aiconfigurator.sdk.operations.dsa import (  # noqa: F401
    DEFAULT_DSA_ARCHITECTURE,
    DSA_MODEL_DIMS,
    load_context_dsa_module_data,
    load_generation_dsa_module_data,
)
from aiconfigurator.sdk.operations.dsv4 import (  # noqa: F401
    _dsv4_normalize_dtype,
    load_context_dsv4_kind_module_data,
    load_dsv4_megamoe_module_data,
    load_dsv4_sparse_kernel_data,
    load_generation_dsv4_kind_module_data,
    load_mhc_module_data,
)
from aiconfigurator.sdk.operations.gemm import (  # noqa: F401
    load_compute_scale_data,
    load_gemm_data,
    load_scale_matrix_data,
)
from aiconfigurator.sdk.operations.mamba import (  # noqa: F401
    load_gdn_data,
    load_mamba2_data,
)
from aiconfigurator.sdk.operations.mla import (  # noqa: F401
    load_context_mla_data,
    load_context_mla_module_data,
    load_generation_mla_data,
    load_generation_mla_module_data,
    load_mla_bmm_data,
    load_wideep_context_mla_data,
    load_wideep_generation_mla_data,
)
from aiconfigurator.sdk.operations.moe import (  # noqa: F401
    load_moe_data,
    load_trtllm_alltoall_data,
    load_wideep_context_moe_data,
    load_wideep_deepep_ll_data,
    load_wideep_deepep_normal_data,
    load_wideep_generation_moe_data,
    load_wideep_moe_compute_data,
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
            from aiconfigurator.sdk.operations.gemm import GEMM

            GEMM.load_data(db)
            return _gemm_key_names(db)

        if key == "context_attention":
            from aiconfigurator.sdk.operations.attention import ContextAttention

            ContextAttention.load_data(db)
            return _enum_key_names(getattr(db, "_context_attention_data", None))

        if key == "generation_attention":
            from aiconfigurator.sdk.operations.attention import GenerationAttention

            GenerationAttention.load_data(db)
            return _enum_key_names(getattr(db, "_generation_attention_data", None))

        if key == "context_mla":
            from aiconfigurator.sdk.operations.mla import ContextMLA, MLAModule

            ContextMLA.load_data(db)
            MLAModule.load_data(db)
            return _merge_key_names(
                getattr(db, "_context_mla_data", None),
                getattr(db, "_context_mla_module_data", None),
            )

        if key == "generation_mla":
            from aiconfigurator.sdk.operations.mla import GenerationMLA, MLAModule

            GenerationMLA.load_data(db)
            MLAModule.load_data(db)
            # Granular data is keyed [kv_cache]...; module data is keyed
            # [fmha][kv_cache][gemm]... so kv modes are at the second level there.
            kv_modes: set[str] = set()
            granular = getattr(db, "_generation_mla_data", None)
            if granular:
                kv_modes.update(_enum_key_names(granular))
            module = getattr(db, "_generation_mla_module_data", None)
            if module:
                for fmha_mode in module:
                    for kv_mode in module[fmha_mode]:
                        kv_modes.add(kv_mode.name if hasattr(kv_mode, "name") else str(kv_mode))
            return sorted(kv_modes)

        if key == "dsa_context_module":
            from aiconfigurator.sdk.operations.dsa import ContextDSAModule

            ContextDSAModule.load_data(db)
            return _enum_key_names(getattr(db, "_context_dsa_module_data", None))

        if key == "dsa_generation_module":
            from aiconfigurator.sdk.operations.dsa import GenerationDSAModule

            GenerationDSAModule.load_data(db)
            return _enum_key_names(getattr(db, "_generation_dsa_module_data", None))

        if key == "deepseek_v4_context_module":
            from aiconfigurator.sdk.operations.dsv4 import ContextDeepSeekV4AttentionModule

            ContextDeepSeekV4AttentionModule.load_data(db)
            return _enum_key_names(getattr(db, "_context_deepseek_v4_attention_module_data", None))

        if key == "deepseek_v4_generation_module":
            from aiconfigurator.sdk.operations.dsv4 import GenerationDeepSeekV4AttentionModule

            GenerationDeepSeekV4AttentionModule.load_data(db)
            return _enum_key_names(getattr(db, "_generation_deepseek_v4_attention_module_data", None))

        if key == "mla_bmm":
            from aiconfigurator.sdk.operations.mla import MLABmm

            MLABmm.load_data(db)
            return _enum_key_names(getattr(db, "_mla_bmm_data", None))

        if key == "nccl":
            from aiconfigurator.sdk.operations.communication import NCCL

            NCCL.load_data(db)
            # vllm matrix prefers ``_nccl_data`` but falls back to ``_oneccl_data``
            # because the original code used ``... or getattr(self, "_oneccl_data", None)``.
            primary = getattr(db, "_nccl_data", None)
            if db.backend == "vllm" and not primary:
                primary = getattr(db, "_oneccl_data", None)
            return _enum_key_names(primary)

        if key == "moe":
            from aiconfigurator.sdk.operations.moe import MoE

            MoE.load_data(db)
            return _enum_key_names(getattr(db, "_moe_data", None))

        if key == "wideep_context_moe":
            from aiconfigurator.sdk.operations.moe import MoE

            MoE.load_data(db)
            return _enum_key_names(getattr(db, "_wideep_context_moe_data", None))

        if key == "wideep_generation_moe":
            from aiconfigurator.sdk.operations.moe import MoE

            MoE.load_data(db)
            return _enum_key_names(getattr(db, "_wideep_generation_moe_data", None))

        if key == "wideep_context_mla":
            from aiconfigurator.sdk.operations.mla import WideEPContextMLA

            WideEPContextMLA.load_data(db)
            modes: set[str] = set()
            data = getattr(db, "_wideep_context_mla_data", None) or {}
            for kernel_source in data:
                for quant_mode in data[kernel_source]:
                    modes.add(quant_mode.name if hasattr(quant_mode, "name") else str(quant_mode))
            return sorted(modes)

        if key == "wideep_generation_mla":
            from aiconfigurator.sdk.operations.mla import WideEPGenerationMLA

            WideEPGenerationMLA.load_data(db)
            modes = set()
            data = getattr(db, "_wideep_generation_mla_data", None) or {}
            for kernel_source in data:
                for kv_cache_dtype in data[kernel_source]:
                    modes.add(kv_cache_dtype.name if hasattr(kv_cache_dtype, "name") else str(kv_cache_dtype))
            return sorted(modes)

        if key == "dsv4_megamoe_module":
            from aiconfigurator.sdk.operations.dsv4 import DeepSeekV4MegaMoEModule

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
    ) -> None:
        """
        Initialize the perf database.

        Args:
            database_mode: drives the shared-layer load behavior. `"HYBRID"` enables
                sibling-row inheritance (including `kernel_source=default` fallback
                rows); other modes keep it off so predictions stay bit-identical to
                main. Doesn't change which rows are interpolated at query time;
                that's controlled by `set_default_database_mode`.
        """
        self.system = system
        self.backend = backend
        self.version = version
        self.systems_root = systems_root
        self.enable_shared_layer = (database_mode or "").upper() == "HYBRID"
        with open(os.path.join(systems_root, system + ".yaml")) as f:
            self.system_spec = SystemSpec(yaml.load(f, Loader=yaml.SafeLoader))
        self._default_database_mode = common.DatabaseMode.SILICON  # default mode is SILICON

        # Cache for extracted metric data to avoid repeated extraction in _interp_3d
        self._extracted_metrics_cache = {}

        # Manifest entries grouped by op_file. Used by ``_build_op_sources``
        # (lazy-load path inside each op class) to discover which sibling
        # backend/version dirs hold rows the active backend can inherit.
        self._op_kernel_source_manifest_entries = _load_op_kernel_source_manifest_entries(systems_root)

        # lazy per-op data ownership: every op class owns its CSV data and loads it on first query
        # via ``OpClass.load_data(database)``. No eager warm-up here — each op
        # opens its data file the first time a query (or the lazy support
        # matrix below) needs it. ``PerfDatabase()`` opens zero CSVs.
        self.supported_quant_mode = _LazySupportMatrix(self)
        self._finalize_loaded_data()

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

            Granular data is keyed [kv_cache]... so top-level keys are kv modes.
            Module data is keyed [fmha][kv_cache][gemm]... so kv modes are
            at the second level.
            """
            kv_modes: set[str] = set()
            # Granular: top-level keys are kv_cache_dtype
            granular = getattr(self, "_generation_mla_data", None)
            if granular:
                kv_modes.update(_enum_key_names(granular))
            # Module: kv_cache_dtype is at the second level
            module = getattr(self, "_generation_mla_module_data", None)
            if module:
                for fmha_mode in module:
                    for kv_mode in module[fmha_mode]:
                        kv_modes.add(kv_mode.name if hasattr(kv_mode, "name") else str(kv_mode))
            return sorted(kv_modes)

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
        """Build the priority-ordered list of source files for one op.

        Returns a list of `(file_path, kernel_source_filter)` tuples to be
        loaded in order. The first source whose file actually contains rows
        for a shape becomes the source of truth for that shape — later sources
        only fill in shapes the earlier ones lacked. Ordering, in priority:

          1. Active backend/version (primary). Filter is `None` — load every row.
          2. Other versions of the *same* framework, newest first. A different
             version of the same backend is closer to the active measurement
             than any other framework, so it wins on shape conflicts.
          3. Other frameworks alphabetically; within each, newest version first.

        Returns just the primary tuple when the shared layer is disabled, when
        the op file is framework-agnostic (nccl / oneccl), or when no manifest
        entry whitelists the active backend for this op. The kernel-source
        filter on sibling sources is essential — `load_*` functions strip
        `kernel_source` from dict keys, so unfiltered sibling rows would
        silently clobber active-backend rows on key conflict.
        """
        sources: list[tuple[str, Optional[set[str]]]] = [(primary_path, None)]
        if not self.enable_shared_layer:
            return sources
        if op_filename_enum in (PerfDataFilename.nccl, PerfDataFilename.oneccl):
            return sources

        op_file_basename = op_filename_enum.value
        backend_lower = self.backend.lower()

        # `framework -> set of kernel_sources` that the active backend may inherit
        # from sibling dirs of that framework. Both `shared` and `shared_fallback`
        # rows are admitted in HYBRID mode; the fallback set is tracked separately
        # only so we can emit a single warning per fallback source.
        per_framework_filter: dict[str, set[str]] = defaultdict(set)
        per_framework_fallback: dict[str, set[str]] = defaultdict(set)
        for entry in self._op_kernel_source_manifest_entries.get(op_file_basename, ()):
            frameworks_lower = {fw.lower() for fw in entry.get("frameworks") or []}
            if backend_lower not in frameworks_lower:
                continue  # Active backend isn't listed as a consumer of this kernel_source.
            ks = entry.get("kernel_source")
            if not ks:
                continue
            tier = entry.get("tier")
            if tier in ("shared", "shared_fallback"):
                for fw in frameworks_lower:
                    per_framework_filter[fw].add(ks)
                if tier == "shared_fallback":
                    for fw in frameworks_lower:
                        per_framework_fallback[fw].add(ks)

        if not per_framework_filter:
            return sources

        # Iterate the active framework first (newest sibling versions first), then
        # other frameworks alphabetically. Putting the active framework first means
        # cross-version siblings outrank cross-framework on shape conflicts.
        ordered_frameworks: list[str] = []
        if backend_lower in per_framework_filter:
            ordered_frameworks.append(backend_lower)
        ordered_frameworks.extend(sorted(set(per_framework_filter) - {backend_lower}))

        # Sort key for newest-first ordering. Parseable PEP 440 versions form one
        # group and always rank above unparseable strings — guarantees `1.10.0`
        # beats `1.2.0` regardless of the lexicographic accident.
        def _newest_first(version: str) -> tuple:
            parsed = parse_support_matrix_version(version)
            return (1, parsed) if parsed is not None else (0, version)

        for framework in ordered_frameworks:
            fw_dir = os.path.join(system_data_root, framework)
            if not os.path.isdir(fw_dir):
                continue
            ks_filter = per_framework_filter[framework]
            fallback_only = per_framework_fallback.get(framework, set())
            fw_versions = sorted(
                (v for v in os.listdir(fw_dir) if not v.startswith("_")),
                key=_newest_first,
                reverse=True,
            )
            for sibling_version in fw_versions:
                if framework == backend_lower and sibling_version == self.version:
                    continue  # Active source already added as the primary.
                sibling_path = _resolve_perf_data_path(os.path.join(fw_dir, sibling_version, op_file_basename))
                if not os.path.isfile(sibling_path):
                    continue
                sources.append((sibling_path, ks_filter))
                if fallback_only & ks_filter:
                    logger.warning(
                        "Loading low-fidelity fallback rows for %s from %s. Queries "
                        "returning these rows are framework-implicit and may differ "
                        "from real backend behavior.",
                        op_file_basename,
                        sibling_path,
                    )
        return sources

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
        if mode != self._default_database_mode:
            self.clear_runtime_caches()
            self._default_database_mode = mode

    def get_default_database_mode(self) -> common.DatabaseMode:
        """
        Get the default database mode
        """
        return self._default_database_mode

    def clear_runtime_caches(self) -> None:
        """Clear cached query/interpolation state while preserving loaded op data."""
        self._extracted_metrics_cache.clear()
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            cache_clear = getattr(attr, "cache_clear", None)
            if callable(cache_clear):
                cache_clear()

    @staticmethod
    def _interp_pr(latency: float, energy: float = 0.0) -> PerformanceResult:
        """Build a PerformanceResult derived from silicon table data.

        Silicon-table interpolation/extrapolation still uses silicon data; only
        explicit formula fallbacks should be tagged as ``"empirical"``.
        """
        return PerformanceResult(latency, energy=energy, source="silicon")

    def _query_silicon_or_hybrid(
        self,
        get_silicon: Callable[[], PerformanceResult],
        get_empirical: Callable[[], float],
        database_mode: common.DatabaseMode,
        error_msg: str,
    ) -> PerformanceResult:
        """
        Helper method to query database (SILICON mode) with optional fallback to empirical mode.

        Args:
            get_silicon: Callable that performs the database query and returns PerformanceResult
            get_empirical: Callable that returns empirical latency (float) - should be a lambda or function
                          that captures the necessary arguments
            database_mode: Database mode (SILICON or HYBRID) - HYBRID mode will fall back to empirical on exception
            error_msg: Error message for logging when query fails

        Returns:
            PerformanceResult from database query or empirical fallback (if database_mode is HYBRID)
        """
        if not error_msg.endswith("."):
            error_msg += "."

        try:
            return get_silicon()

        except Exception as e:
            if database_mode == common.DatabaseMode.HYBRID:
                debug_msg = error_msg + " Will try empirical mode."
                logger.debug(debug_msg)
                return PerformanceResult(get_empirical(), energy=0.0, source="empirical")

            exception_msg = error_msg + " Consider using HYBRID mode."
            # PerfDataNotAvailableError is a structured signal that callers (e.g.
            # FallbackOp, Pareto search) are expected to handle. Log it without a
            # stack trace so successful searches that merely skip a candidate do
            # not spam internal tracebacks. Genuinely unexpected exceptions still
            # get the full traceback via logger.exception.
            if isinstance(e, PerfDataNotAvailableError):
                logger.warning(exception_msg)
            elif isinstance(e, _MISSING_SILICON_DATA_EXCEPTIONS):
                missing_data_error = PerfDataNotAvailableError(
                    f"{exception_msg} Missing silicon data for the requested lookup."
                )
                logger.warning(str(missing_data_error))
                raise missing_data_error from e
            else:
                logger.exception(exception_msg)
            # Modify the original exception message
            if e.args:
                e.args = (str(e.args[0]) + " " + exception_msg,) + e.args[1:]
            else:
                e.args = (exception_msg,)
            raise

    @functools.lru_cache(maxsize=32768)
    def query_gemm(
        self,
        m: int,
        n: int,
        k: int,
        quant_mode: common.GEMMQuantMode,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """
        Query GEMM operation latency and energy. Delegates to ``GEMM``;
        see ``aiconfigurator.sdk.operations.gemm.GEMM._query_gemm_table``.

        Returns:
            PerformanceResult: Acts as float (latency in ms).
                              Energy accessible via .energy attribute (W·ms).
                              Power can be computed as energy/latency (W).

        Example:
            >>> result = db.query_gemm(4096, 4096, 4096, GEMMQuantMode.nvfp4)
            >>> latency_ms = float(result)  # Use as float
            >>> energy_wms = result.energy
            >>> power_w = result.power  # or result.energy / float(result)
        """
        from aiconfigurator.sdk.operations.gemm import GEMM

        return GEMM._query_gemm_table(self, m, n, k, quant_mode, database_mode)

    @functools.lru_cache(maxsize=32768)
    def query_compute_scale(
        self,
        m: int,
        k: int,
        quant_mode: common.GEMMQuantMode,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Query compute scale latency. Delegates to
        ``GEMM._query_compute_scale_table``."""
        from aiconfigurator.sdk.operations.gemm import GEMM

        return GEMM._query_compute_scale_table(self, m, k, quant_mode, database_mode)

    @functools.lru_cache(maxsize=32768)
    def query_scale_matrix(
        self,
        m: int,
        k: int,
        quant_mode: common.GEMMQuantMode,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Query scale matrix latency. Delegates to
        ``GEMM._query_scale_matrix_table``."""
        from aiconfigurator.sdk.operations.gemm import GEMM

        return GEMM._query_scale_matrix_table(self, m, k, quant_mode, database_mode)

    @functools.lru_cache(maxsize=32768)
    def query_context_attention(
        self,
        b: int,
        s: int,
        prefix: int,
        n: int,
        n_kv: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        database_mode: Optional[common.DatabaseMode] = None,
        window_size: int = 0,
        head_size: int = 128,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Query context attention latency. Delegates to
        ``ContextAttention._query_context_attention_table``."""
        from aiconfigurator.sdk.operations.attention import ContextAttention

        return ContextAttention._query_context_attention_table(
            self,
            b,
            s,
            prefix,
            n,
            n_kv,
            kvcache_quant_mode,
            fmha_quant_mode,
            database_mode,
            window_size,
            head_size,
        )

    @functools.lru_cache(maxsize=32768)
    def query_encoder_attention(
        self,
        b: int,
        s: int,
        n: int,
        head_size: int,
        fmha_quant_mode: common.FMHAQuantMode,
        database_mode: Optional[common.DatabaseMode] = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Query non-causal encoder attention latency. Delegates to
        ``EncoderAttention._query_encoder_attention_table``."""
        from aiconfigurator.sdk.operations.attention import EncoderAttention

        return EncoderAttention._query_encoder_attention_table(
            self,
            b,
            s,
            n,
            head_size,
            fmha_quant_mode,
            database_mode,
        )

    @functools.lru_cache(maxsize=32768)
    def query_generation_attention(
        self,
        b: int,
        s: int,
        n: int,
        n_kv: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        database_mode: Optional[common.DatabaseMode] = None,
        window_size: int = 0,
        head_size: int = 128,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Query generation attention latency. Delegates to
        ``GenerationAttention._query_generation_attention_table``."""
        from aiconfigurator.sdk.operations.attention import GenerationAttention

        return GenerationAttention._query_generation_attention_table(
            self,
            b,
            s,
            n,
            n_kv,
            kvcache_quant_mode,
            database_mode,
            window_size,
            head_size,
        )

    @functools.lru_cache(maxsize=32768)
    def query_context_mla(
        self,
        b: int,
        s: int,
        prefix: int,
        num_heads: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Query context MLA latency. Delegates to ``ContextMLA._query_context_mla_table``."""
        from aiconfigurator.sdk.operations.mla import ContextMLA

        return ContextMLA._query_context_mla_table(
            self,
            b,
            s,
            prefix,
            num_heads,
            kvcache_quant_mode,
            fmha_quant_mode,
            database_mode,
        )

    @functools.lru_cache(maxsize=32768)
    def query_generation_mla(
        self,
        b: int,
        s: int,
        num_heads: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Query generation MLA latency. Delegates to ``GenerationMLA._query_generation_mla_table``."""
        from aiconfigurator.sdk.operations.mla import GenerationMLA

        return GenerationMLA._query_generation_mla_table(
            self,
            b,
            s,
            num_heads,
            kvcache_quant_mode,
            database_mode,
        )

    @functools.lru_cache(maxsize=32768)
    def query_context_mla_module(
        self,
        b: int,
        s: int,
        prefix: int,
        num_heads: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        gemm_quant_mode: common.GEMMQuantMode = common.GEMMQuantMode.bfloat16,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Query context MLA module latency. Delegates to ``MLAModule._query_context_mla_module_table``."""
        from aiconfigurator.sdk.operations.mla import MLAModule

        return MLAModule._query_context_mla_module_table(
            self,
            b,
            s,
            prefix,
            num_heads,
            kvcache_quant_mode,
            fmha_quant_mode,
            gemm_quant_mode,
            database_mode,
        )

    @functools.lru_cache(maxsize=32768)
    def query_generation_mla_module(
        self,
        b: int,
        s: int,
        num_heads: int,
        kv_cache_dtype: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode = common.FMHAQuantMode.bfloat16,
        gemm_quant_mode: common.GEMMQuantMode = common.GEMMQuantMode.bfloat16,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Query generation MLA module latency. Delegates to ``MLAModule._query_generation_mla_module_table``."""
        from aiconfigurator.sdk.operations.mla import MLAModule

        return MLAModule._query_generation_mla_module_table(
            self,
            b,
            s,
            num_heads,
            kv_cache_dtype,
            fmha_quant_mode,
            gemm_quant_mode,
            database_mode,
        )

    @functools.lru_cache(maxsize=32768)
    def query_wideep_generation_mla(
        self,
        b: int,
        s: int,
        tp_size: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        attention_backend: str | None = None,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Query WideEP generation MLA latency.

        Delegates to ``WideEPGenerationMLA._query_wideep_generation_mla_table``.
        """
        from aiconfigurator.sdk.operations.mla import WideEPGenerationMLA

        return WideEPGenerationMLA._query_wideep_generation_mla_table(
            self,
            b,
            s,
            tp_size,
            kvcache_quant_mode,
            fmha_quant_mode,
            attention_backend,
            database_mode,
        )

    @functools.lru_cache(maxsize=32768)
    def query_wideep_context_mla(
        self,
        b: int,
        s: int,
        prefix: int,
        tp_size: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        attention_backend: str | None = None,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Query WideEP context MLA latency. Delegates to ``WideEPContextMLA._query_wideep_context_mla_table``."""
        from aiconfigurator.sdk.operations.mla import WideEPContextMLA

        return WideEPContextMLA._query_wideep_context_mla_table(
            self,
            b,
            s,
            prefix,
            tp_size,
            kvcache_quant_mode,
            fmha_quant_mode,
            attention_backend,
            database_mode,
        )

    # to simplify, we no longer support allreduce_strategy
    @functools.lru_cache(maxsize=32768)
    def query_custom_allreduce(
        self,
        quant_mode: common.CommQuantMode,
        tp_size: int,
        size: int,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Query custom AllReduce latency. Delegates to
        ``CustomAllReduce._query_custom_allreduce_table``."""
        from aiconfigurator.sdk.operations.communication import CustomAllReduce

        return CustomAllReduce._query_custom_allreduce_table(self, quant_mode, tp_size, size, database_mode)

    @functools.lru_cache(maxsize=32768)
    def query_nccl(
        self,
        dtype: common.CommQuantMode,
        num_gpus: int,
        operation: str,
        message_size: int,  # element number
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Query NCCL collective communication latency. Delegates to
        ``NCCL._query_nccl_table``."""
        from aiconfigurator.sdk.operations.communication import NCCL

        return NCCL._query_nccl_table(self, dtype, num_gpus, operation, message_size, database_mode)

    @functools.lru_cache(maxsize=32768)
    def query_moe(
        self,
        num_tokens: int,
        hidden_size: int,
        inter_size: int,
        topk: int,
        num_experts: int,
        moe_tp_size: int,
        moe_ep_size: int,
        quant_mode: common.MoEQuantMode,
        workload_distribution: str,
        is_context: bool = True,
        moe_backend: str | None = None,
        database_mode: common.DatabaseMode | None = None,
        is_gated: bool = True,
        enable_eplb: bool = False,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Delegates to ``MoE``; see ``operations.moe.MoE._query_moe_table``."""
        from aiconfigurator.sdk.operations.moe import MoE

        return MoE._query_moe_table(
            self,
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            inter_size=inter_size,
            topk=topk,
            num_experts=num_experts,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
            quant_mode=quant_mode,
            workload_distribution=workload_distribution,
            is_context=is_context,
            moe_backend=moe_backend,
            database_mode=database_mode,
            is_gated=is_gated,
            enable_eplb=enable_eplb,
        )

    @functools.lru_cache(maxsize=32768)
    def query_mla_bmm(
        self,
        num_tokens: int,
        num_heads: int,
        quant_mode: common.GEMMQuantMode,
        if_pre: bool = True,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Query MLA BMM latency. Delegates to ``MLABmm._query_mla_bmm_table``."""
        from aiconfigurator.sdk.operations.mla import MLABmm

        return MLABmm._query_mla_bmm_table(
            self,
            num_tokens,
            num_heads,
            quant_mode,
            if_pre,
            database_mode,
        )

    @functools.lru_cache(maxsize=32768)
    def query_mem_op(
        self, mem_bytes: int, database_mode: common.DatabaseMode | None = None
    ) -> PerformanceResult | tuple[float, float, float]:
        """Query memory-operation latency analytically (no CSV data).

        Returns:
            PerformanceResult acting as float (latency in ms); energy via ``.energy``.
            For SOL_FULL, returns a ``(sol_time, 0, sol_time)`` tuple.
        """
        gpu_spec = self.system_spec["gpu"]

        def get_sol() -> tuple[float, float, float]:
            sol_time = mem_bytes / gpu_spec["mem_bw"] * 1000
            return sol_time, 0, sol_time

        def get_empirical() -> float:
            return (
                mem_bytes / (gpu_spec["mem_bw"] * gpu_spec["mem_bw_empirical_scaling_factor"])
                + gpu_spec["mem_empirical_constant_latency"]
            ) * 1000

        if database_mode is None:
            database_mode = self._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            return PerformanceResult(get_sol()[0], energy=0.0, source="sol")
        if database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol()
        # EMPIRICAL / SILICON / HYBRID share the same empirical formula. There is
        # no silicon table for raw memory ops, so always tag as ``empirical``.
        return PerformanceResult(get_empirical(), energy=0.0, source="empirical")

    def query_mamba2(
        self,
        phase: str,
        kernel_source: str,
        batch_size: int,
        seq_len: int | None,
        d_model: int,
        d_state: int,
        d_conv: int,
        nheads: int,
        head_dim: int,
        n_groups: int,
        chunk_size: int,
    ) -> PerformanceResult:
        """Query Mamba2 kernel latency. Delegates to ``Mamba2Kernel._query_mamba2_table``."""
        from aiconfigurator.sdk.operations.mamba import Mamba2Kernel

        return Mamba2Kernel._query_mamba2_table(
            self,
            phase,
            kernel_source,
            batch_size,
            seq_len,
            d_model,
            d_state,
            d_conv,
            nheads,
            head_dim,
            n_groups,
            chunk_size,
        )

    def query_gdn(
        self,
        phase: str,
        kernel_source: str,
        batch_size: int,
        seq_len: int | None,
        d_model: int,
        num_k_heads: int,
        head_k_dim: int,
        num_v_heads: int,
        head_v_dim: int,
        d_conv: int,
    ) -> PerformanceResult:
        """Query GDN kernel latency. Delegates to ``GDNKernel._query_gdn_table``."""
        from aiconfigurator.sdk.operations.mamba import GDNKernel

        return GDNKernel._query_gdn_table(
            self,
            phase,
            kernel_source,
            batch_size,
            seq_len,
            d_model,
            num_k_heads,
            head_k_dim,
            num_v_heads,
            head_v_dim,
            d_conv,
        )

    @functools.lru_cache(maxsize=32768)
    def query_p2p(
        self, message_bytes: int, database_mode: common.DatabaseMode | None = None
    ) -> PerformanceResult | tuple[float, float, float]:
        """Query P2P latency. Delegates to ``P2P._query_p2p_table``."""
        from aiconfigurator.sdk.operations.communication import P2P

        return P2P._query_p2p_table(self, message_bytes, database_mode)

    @functools.lru_cache(maxsize=32768)
    def query_wideep_deepep_ll(
        self,
        node_num: int,
        num_tokens: int,
        num_experts: int,
        topk: int,
        hidden_size: int,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Delegates to ``MoEDispatch``; see
        ``operations.moe.MoEDispatch._query_wideep_deepep_ll_table``."""
        from aiconfigurator.sdk.operations.moe import MoEDispatch

        return MoEDispatch._query_wideep_deepep_ll_table(
            self,
            node_num=node_num,
            num_tokens=num_tokens,
            num_experts=num_experts,
            topk=topk,
            hidden_size=hidden_size,
            database_mode=database_mode,
        )

    @functools.lru_cache(maxsize=32768)
    def query_wideep_deepep_normal(
        self,
        node_num: int,
        num_tokens: int,
        num_experts: int,
        topk: int,
        hidden_size: int,
        sms: int,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Delegates to ``MoEDispatch``; see
        ``operations.moe.MoEDispatch._query_wideep_deepep_normal_table``."""
        from aiconfigurator.sdk.operations.moe import MoEDispatch

        return MoEDispatch._query_wideep_deepep_normal_table(
            self,
            node_num=node_num,
            num_tokens=num_tokens,
            num_experts=num_experts,
            topk=topk,
            hidden_size=hidden_size,
            sms=sms,
            database_mode=database_mode,
        )

    @functools.lru_cache(maxsize=32768)
    def query_wideep_moe_compute(
        self,
        num_tokens: int,
        hidden_size: int,
        inter_size: int,
        topk: int,
        num_experts: int,
        num_slots: int,
        moe_tp_size: int,
        moe_ep_size: int,
        quant_mode: common.MoEQuantMode,
        workload_distribution: str,
        database_mode: common.DatabaseMode | None = None,
        is_gated: bool = True,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Delegates to ``TrtLLMWideEPMoE``; see
        ``operations.moe.TrtLLMWideEPMoE._query_compute_table``."""
        from aiconfigurator.sdk.operations.moe import TrtLLMWideEPMoE

        return TrtLLMWideEPMoE._query_compute_table(
            self,
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            inter_size=inter_size,
            topk=topk,
            num_experts=num_experts,
            num_slots=num_slots,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
            quant_mode=quant_mode,
            workload_distribution=workload_distribution,
            database_mode=database_mode,
            is_gated=is_gated,
        )

    @functools.lru_cache(maxsize=32768)
    def query_trtllm_alltoall(
        self,
        op_name: str,
        num_tokens: int,
        hidden_size: int,
        topk: int,
        num_experts: int,
        moe_ep_size: int,
        quant_mode: common.MoEQuantMode,
        node_num: int | None = None,
        database_mode: common.DatabaseMode | None = None,
        moe_backend: str | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Delegates to ``TrtLLMWideEPMoEDispatch``; see
        ``operations.moe.TrtLLMWideEPMoEDispatch._query_alltoall_table``."""
        from aiconfigurator.sdk.operations.moe import TrtLLMWideEPMoEDispatch

        return TrtLLMWideEPMoEDispatch._query_alltoall_table(
            self,
            op_name=op_name,
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            topk=topk,
            num_experts=num_experts,
            moe_ep_size=moe_ep_size,
            quant_mode=quant_mode,
            node_num=node_num,
            database_mode=database_mode,
            moe_backend=moe_backend,
        )

    # ═══════════════════════════════════════════════════════════════════
    # DSA (DeepSeek Sparse Attention) Queries
    # ═══════════════════════════════════════════════════════════════════

    @functools.lru_cache(maxsize=32768)
    def query_context_dsa_module(
        self,
        b: int,
        s: int,
        num_heads: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        gemm_quant_mode: common.GEMMQuantMode = common.GEMMQuantMode.bfloat16,
        database_mode: common.DatabaseMode | None = None,
        *,
        prefix: int = 0,
        architecture: str = DEFAULT_DSA_ARCHITECTURE,
        index_n_heads: int | None = None,
        index_head_dim: int | None = None,
        index_topk: int | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Query context DSA module latency. Delegates to
        ``ContextDSAModule._query_context_dsa_module_table``."""
        from aiconfigurator.sdk.operations.dsa import ContextDSAModule

        return ContextDSAModule._query_context_dsa_module_table(
            self,
            b,
            s,
            num_heads,
            kvcache_quant_mode,
            fmha_quant_mode,
            gemm_quant_mode,
            database_mode,
            prefix=prefix,
            architecture=architecture,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
        )

    @functools.lru_cache(maxsize=32768)
    def query_generation_dsa_module(
        self,
        b: int,
        s: int,
        num_heads: int,
        kv_cache_dtype: common.KVCacheQuantMode,
        gemm_quant_mode: common.GEMMQuantMode = common.GEMMQuantMode.bfloat16,
        database_mode: common.DatabaseMode | None = None,
        *,
        architecture: str = DEFAULT_DSA_ARCHITECTURE,
        index_n_heads: int | None = None,
        index_head_dim: int | None = None,
        index_topk: int | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Query generation DSA module latency. Delegates to
        GenerationDSAModule._query_generation_dsa_module_table."""
        from aiconfigurator.sdk.operations.dsa import GenerationDSAModule

        return GenerationDSAModule._query_generation_dsa_module_table(
            self,
            b,
            s,
            num_heads,
            kv_cache_dtype,
            gemm_quant_mode,
            database_mode,
            architecture=architecture,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
        )

    @staticmethod
    def _causal_limited_pairs(batch_size: int, query_len: int, prefix: int, limit: int) -> int:
        """Return sum over queries of min(prefix + query_index + 1, limit), times batch."""
        if limit <= 0 or query_len <= 0:
            return 0
        full_s = prefix + query_len
        if prefix >= limit:
            return batch_size * query_len * limit
        if full_s <= limit:
            return batch_size * (full_s * (full_s + 1) - prefix * (prefix + 1)) // 2
        ramp = batch_size * (limit * (limit + 1) - prefix * (prefix + 1)) // 2
        saturated = batch_size * (full_s - limit) * limit
        return ramp + saturated

    @staticmethod
    def _sum_floor_upto(n: int, divisor: int) -> int:
        """Return sum_{i=0..n} floor(i / divisor)."""
        if n < 0:
            return 0
        q, r = divmod(n, divisor)
        return divisor * q * (q - 1) // 2 + q * (r + 1)

    @classmethod
    def _compressed_context_pairs(cls, batch_size: int, query_len: int, prefix: int, ratio: int, limit: int) -> int:
        if ratio <= 0 or query_len <= 0 or limit <= 0:
            return 0
        start = prefix + 1
        end = prefix + query_len
        saturation_start = limit * ratio
        if end < saturation_start:
            total = cls._sum_floor_upto(end, ratio) - cls._sum_floor_upto(start - 1, ratio)
        elif start >= saturation_start:
            total = query_len * limit
        else:
            ramp = cls._sum_floor_upto(saturation_start - 1, ratio) - cls._sum_floor_upto(start - 1, ratio)
            total = ramp + (end - saturation_start + 1) * limit
        return batch_size * total

    @functools.lru_cache(maxsize=32768)
    def query_mhc_module(
        self,
        num_tokens: int,
        hidden_size: int,
        hc_mult: int,
        sinkhorn_iters: int,
        op: str,
        quant_mode: common.GEMMQuantMode = common.GEMMQuantMode.bfloat16,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Delegates to ``DeepSeekV4MHCModule``; see
        ``aiconfigurator.sdk.operations.dsv4.DeepSeekV4MHCModule._query_mhc_table``.
        """
        from aiconfigurator.sdk.operations.dsv4 import DeepSeekV4MHCModule

        return DeepSeekV4MHCModule._query_mhc_table(
            self,
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            sinkhorn_iters=sinkhorn_iters,
            op=op,
            quant_mode=quant_mode,
            database_mode=database_mode,
        )

    @functools.lru_cache(maxsize=32768)
    def query_context_deepseek_v4_attention_module(
        self,
        b: int,
        s: int,
        num_heads: int,
        native_heads: int,
        tp_size: int,
        hidden_size: int,
        q_lora_rank: int,
        o_lora_rank: int,
        head_dim: int,
        rope_head_dim: int,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        window_size: int,
        compress_ratio: int,
        o_groups: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        gemm_quant_mode: common.GEMMQuantMode = common.GEMMQuantMode.bfloat16,
        database_mode: common.DatabaseMode | None = None,
        *,
        prefix: int = 0,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Delegates to ``ContextDeepSeekV4AttentionModule``; see
        ``operations.dsv4.ContextDeepSeekV4AttentionModule._query_context_attn_table``.
        """
        from aiconfigurator.sdk.operations.dsv4 import ContextDeepSeekV4AttentionModule

        return ContextDeepSeekV4AttentionModule._query_context_attn_table(
            self,
            b=b,
            s=s,
            num_heads=num_heads,
            native_heads=native_heads,
            tp_size=tp_size,
            hidden_size=hidden_size,
            q_lora_rank=q_lora_rank,
            o_lora_rank=o_lora_rank,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            window_size=window_size,
            compress_ratio=compress_ratio,
            o_groups=o_groups,
            kvcache_quant_mode=kvcache_quant_mode,
            fmha_quant_mode=fmha_quant_mode,
            gemm_quant_mode=gemm_quant_mode,
            database_mode=database_mode,
            prefix=prefix,
        )

    @functools.lru_cache(maxsize=32768)
    def query_generation_deepseek_v4_attention_module(
        self,
        b: int,
        s: int,
        num_heads: int,
        native_heads: int,
        tp_size: int,
        hidden_size: int,
        q_lora_rank: int,
        o_lora_rank: int,
        head_dim: int,
        rope_head_dim: int,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        window_size: int,
        compress_ratio: int,
        o_groups: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        gemm_quant_mode: common.GEMMQuantMode = common.GEMMQuantMode.bfloat16,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Delegates to ``GenerationDeepSeekV4AttentionModule``; see
        ``operations.dsv4.GenerationDeepSeekV4AttentionModule._query_generation_attn_table``.
        """
        from aiconfigurator.sdk.operations.dsv4 import GenerationDeepSeekV4AttentionModule

        return GenerationDeepSeekV4AttentionModule._query_generation_attn_table(
            self,
            b=b,
            s=s,
            num_heads=num_heads,
            native_heads=native_heads,
            tp_size=tp_size,
            hidden_size=hidden_size,
            q_lora_rank=q_lora_rank,
            o_lora_rank=o_lora_rank,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            window_size=window_size,
            compress_ratio=compress_ratio,
            o_groups=o_groups,
            kvcache_quant_mode=kvcache_quant_mode,
            fmha_quant_mode=fmha_quant_mode,
            gemm_quant_mode=gemm_quant_mode,
            database_mode=database_mode,
        )

    @functools.lru_cache(maxsize=32768)
    def query_dsv4_megamoe_module(
        self,
        num_tokens: int,
        hidden_size: int,
        inter_size: int,
        topk: int,
        num_experts: int,
        moe_tp_size: int,
        moe_ep_size: int,
        quant_mode: common.MoEQuantMode,
        workload_distribution: str,
        is_context: bool = True,
        source_policy: str = "random",
        pre_dispatch: str = "sglang_jit",
        num_fused_shared_experts: int = 0,
        kernel_source: str = "deepgemm_megamoe",
        kernel_dtype: str = "fp8_fp4",
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult:
        """Delegates to ``DeepSeekV4MegaMoEModule``; see
        ``operations.dsv4.DeepSeekV4MegaMoEModule._query_megamoe_table``.
        """
        from aiconfigurator.sdk.operations.dsv4 import DeepSeekV4MegaMoEModule

        return DeepSeekV4MegaMoEModule._query_megamoe_table(
            self,
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            inter_size=inter_size,
            topk=topk,
            num_experts=num_experts,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
            quant_mode=quant_mode,
            workload_distribution=workload_distribution,
            is_context=is_context,
            source_policy=source_policy,
            pre_dispatch=pre_dispatch,
            num_fused_shared_experts=num_fused_shared_experts,
            kernel_source=kernel_source,
            kernel_dtype=kernel_dtype,
            database_mode=database_mode,
        )


if __name__ == "__main__":
    database_dict = get_all_databases()

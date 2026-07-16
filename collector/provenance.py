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

``load_closures`` fails closed: every module referenced by any of the five
collector registries (``collector.framework_manifest._REGISTRY_MODULES``)
MUST have a ``hash_closures.yaml`` entry, or loading raises ``KeyError``.
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

STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"

_RUNTIME_FIELD_ORDER = ("framework", "version", "image", "image_digest")
_TABLE_FIELD_ORDER = ("collector_ref", "collector_hash", "case_plan_hash", "collected_at", "rows", "status")


def enumerate_registry_modules() -> set[str]:
    """Return every collector module referenced by an OpEntry in any of the five
    registries enumerated by ``framework_manifest._REGISTRY_MODULES`` (sglang,
    trtllm, vllm, wideep_sglang, wideep_trtllm).
    """
    import importlib

    from collector.framework_manifest import _REGISTRY_MODULES

    modules: set[str] = set()
    for registry_module_path in _REGISTRY_MODULES.values():
        registry = importlib.import_module(registry_module_path).REGISTRY
        for entry in registry:
            if entry.module:
                modules.add(entry.module)
            for route in entry.versions:
                modules.add(route.module)
    return modules


def load_closures(path: str | Path) -> dict[str, list[str]]:
    """Load ``hash_closures.yaml`` and fail closed on incomplete coverage.

    Every module returned by :func:`enumerate_registry_modules` MUST appear as
    a key; a module missing its closure entry is a KeyError, not a silent
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

    missing = enumerate_registry_modules() - closures.keys()
    if missing:
        raise KeyError(
            f"{closures_path}: missing hash_closures.yaml entries for registry module(s) "
            f"{sorted(missing)} (fail-closed — every registry module must declare its "
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
    """complete unless the table's checkpoint holds unresolved failures or a
    ModuleCollectionFailure was recorded for one of its producing ops.
    """
    if unresolved_failed_count > 0 or had_module_failure:
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


def write_collection_meta(out_dir: str | Path, runtime_meta: dict[str, Any], tables: dict[str, dict[str, Any]]) -> Path:
    """Render ``collection_meta.yaml`` per design §5, with deterministic key order."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    doc = {
        "schema_version": 1,
        "runtime": {key: runtime_meta[key] for key in _RUNTIME_FIELD_ORDER if key in runtime_meta},
        "tables": {
            table: {key: tables[table][key] for key in _TABLE_FIELD_ORDER if key in tables[table]}
            for table in sorted(tables)
        },
    }

    meta_path = out_path / "collection_meta.yaml"
    with meta_path.open("w", encoding="utf-8") as meta_file:
        meta_file.write(spdx_header())
        yaml.safe_dump(doc, meta_file, sort_keys=False, default_flow_style=False)
    return meta_path

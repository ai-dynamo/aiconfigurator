# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed CI audit of the collected perf-database tree (Collector V3 §8).

The primary enforcement surface of the whole Collector V3 design: run on every
PR touching `data/`, `collector/`, or the manifest (sibling of
`audit_kernel_source.py`). Exits non-zero on ANY failure and prints every
failure it finds — never stops at the first one.

Six rules, each named after the design section it enforces:

- **R1 sidecar coverage** (design §5, §8): every family version dir
  (`<system>/<family>/<backend>/<version>/`) holding >=1 parquet table must
  have a `collection_meta.yaml` whose `tables` key set covers every parquet
  stem present. The legacy tier (`provenance: legacy`, migrated pre-V3 data)
  satisfies coverage with status-only entries — the check is key presence,
  not field completeness, so a legacy entry with only `status` is enough.
- **R2 reuse validity** (design §6.3, §6.5 rules 1-3): every `reuse.yaml`
  must parse via the real loader parser
  (`aiconfigurator_core.sdk.perf_database._parse_reuse_yaml`); each entry's
  donor (same family, same backend, `from_version`, `table`) must exist as
  real parquet data, not merely another declared-reuse dir. "Same backend
  only" is a structural guarantee here: a `reuse.yaml` entry carries no
  backend field, so its donor is always resolved under the same
  `<family>/<backend>/` subtree it lives in. `from_version` may be newer or
  older than the declaring dir's own version — that asymmetry is exactly what
  a declaration is for (§6.3); this audit does not re-derive direction.
- **R3 comm exclusion** (design §6.5 rule 5): no `reuse.yaml` may exist
  anywhere under a `comm` family dir (NCCL/oneCCL curves are topology-bound).
- **R4 family placement** (design §2, catalog-driven): every parquet table's
  stem must map, via the op catalog (`collector/op_backend_catalog.yaml`), to
  the family directory it is actually filed under.
- **R5 identity** (design §4): `collector.framework_manifest.validate_resolution()`
  must return no errors — every registry op resolves to exactly one pinned
  runtime.
- **R6 no legacy markers**: zero `SHARED_LAYER_REUSE.txt` / `INCOMPLETE.txt`
  files remain anywhere under the data root (both are fully superseded by
  `reuse.yaml` / `collection_meta.yaml`).

Reuses (imports, does not duplicate): `collector.op_catalog.load_family_map`,
`collector.framework_manifest.validate_resolution`, and the loader's real
parsers `aiconfigurator_core.sdk.perf_database._parse_reuse_yaml` /
`_load_collection_meta_yaml` (via the `aiconfigurator.sdk` compatibility
alias, which is the same module object — see
`src/aiconfigurator/sdk/_compat.py`). The tree-walking shape mirrors
`audit_kernel_source.py`'s `_iter_data_files`, specialized to the
post-migration family-first layout (`<system>/<family>/<backend>/<version>/`)
that is now the only layout on disk.

Usage:
    python3 tools/perf_database/audit_collector_data.py
    python3 tools/perf_database/audit_collector_data.py \\
        --data-root aic-core/src/aiconfigurator_core/systems/data \\
        --catalog collector/op_backend_catalog.yaml

Exit codes: 0 = every rule OK. 1 = at least one failure (printed, grouped by
rule).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# `collector/` is a standalone top-level package (not part of the installed
# wheel — see .claude/rules/repo-guide.md), so it is only importable when the
# repo root is on sys.path. Running this file directly (`python3
# tools/perf_database/audit_collector_data.py`) puts the script's own
# directory on sys.path[0], not the repo root, so the insert below is
# required (mirrors collect.py:1808 / changed_ops.py's REPO_ROOT). Harmless
# under pytest, where pytest.ini's `pythonpath = .` already covers it.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# aiconfigurator_core is pip-installed editable (pyproject.toml
# [tool.uv.sources]), so it needs no sys.path help. Imported via the
# `aiconfigurator.sdk` compatibility alias, as the SDK-layer test suite does
# (e.g. tests/unit/sdk/database/test_dual_layout_discovery.py) — it is the
# same module object as aiconfigurator_core.sdk.perf_database
# (src/aiconfigurator/sdk/_compat.py:alias_module).
from aiconfigurator.sdk.perf_database import (
    _load_collection_meta_yaml,
    _parse_reuse_yaml,
)
from collector.framework_manifest import validate_resolution
from collector.op_catalog import (
    CATALOG_PATH,
    load_family_map,
)

DATA_ROOT_DEFAULT = Path("aic-core/src/aiconfigurator_core/systems/data")

REUSE_YAML = "reuse.yaml"
COLLECTION_META_YAML = "collection_meta.yaml"
LEGACY_MARKERS = ("SHARED_LAYER_REUSE.txt", "INCOMPLETE.txt")
COMM_FAMILY = "comm"

RULES = ("R1", "R2", "R3", "R4", "R5", "R6")
RULE_TITLES = {
    "R1": "sidecar coverage",
    "R2": "reuse validity",
    "R3": "comm exclusion",
    "R4": "family placement",
    "R5": "identity (manifest v2 resolution)",
    "R6": "no legacy markers",
}


# --------------------------------------------------------------------------
# tree walking: <data_root>/<system>/<family>/<backend>/<version>/
# --------------------------------------------------------------------------


def _subdirs(path: Path) -> list[Path]:
    """Sorted, non-dot-prefixed subdirectories (dot-dirs are never perf-data
    dirs; guards against `.git` when a test points data_root at a repo root).
    """
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.is_dir() and not p.name.startswith("."))


def _parquet_stems(version_dir: Path) -> set[str]:
    return {p.stem for p in version_dir.iterdir() if p.is_file() and p.suffix == ".parquet"}


def iter_version_dirs(data_root: Path) -> list[tuple[str, str, str, str, Path]]:
    """Every `(system, family, backend, version, version_dir)` in the
    family-first layout (Collector V3 §3) — the only layout left on disk
    after the PR2 migration.
    """
    version_dirs: list[tuple[str, str, str, str, Path]] = []
    for system_dir in _subdirs(data_root):
        for family_dir in _subdirs(system_dir):
            for backend_dir in _subdirs(family_dir):
                for version_dir in _subdirs(backend_dir):
                    version_dirs.append(
                        (system_dir.name, family_dir.name, backend_dir.name, version_dir.name, version_dir)
                    )
    return version_dirs


# --------------------------------------------------------------------------
# R1: sidecar coverage
# --------------------------------------------------------------------------


def check_r1_sidecar_coverage(data_root: Path, version_dirs: list[tuple[str, str, str, str, Path]]) -> list[str]:
    failures: list[str] = []
    for _system, _family, _backend, _version, version_dir in version_dirs:
        parquet_stems = _parquet_stems(version_dir)
        if not parquet_stems:
            continue
        rel_dir = version_dir.relative_to(data_root)
        meta_path = version_dir / COLLECTION_META_YAML
        if not meta_path.is_file():
            failures.append(f"{rel_dir}: missing {COLLECTION_META_YAML} for parquet table(s) {sorted(parquet_stems)}")
            continue
        try:
            meta = _load_collection_meta_yaml(str(meta_path))
        except ValueError as exc:
            failures.append(f"{rel_dir}/{COLLECTION_META_YAML}: {exc}")
            continue
        tables = meta.get("tables")
        covered = set(tables) if isinstance(tables, dict) else set()
        for table in sorted(parquet_stems - covered):
            failures.append(f"{rel_dir}: parquet table '{table}' has no entry in {COLLECTION_META_YAML} tables")
    return failures


# --------------------------------------------------------------------------
# R2: reuse validity
# --------------------------------------------------------------------------


def check_r2_reuse_validity(data_root: Path, version_dirs: list[tuple[str, str, str, str, Path]]) -> list[str]:
    failures: list[str] = []
    # (system, family, backend) -> {version: version_dir}, for donor lookup.
    # A reuse.yaml entry carries no backend field, so donors are always
    # resolved within the SAME (system, family, backend) subtree the
    # declaring dir lives in -- "same backend only" is structural, not a
    # separate check.
    by_line: dict[tuple[str, str, str], dict[str, Path]] = defaultdict(dict)
    for system, family, backend, version, version_dir in version_dirs:
        by_line[(system, family, backend)][version] = version_dir

    for system, family, backend, _version, version_dir in version_dirs:
        reuse_path = version_dir / REUSE_YAML
        if not reuse_path.is_file():
            continue
        rel_reuse = reuse_path.relative_to(data_root)
        try:
            parsed = _parse_reuse_yaml(str(reuse_path))
        except ValueError as exc:
            failures.append(f"{rel_reuse}: {exc}")
            continue

        siblings = by_line[(system, family, backend)]
        for entry in parsed["entries"]:
            table = entry["table"]
            from_version = entry["from_version"]
            donor_dir = siblings.get(from_version)
            if donor_dir is None:
                failures.append(
                    f"{rel_reuse}: table '{table}' declares from_version '{from_version}', which does not "
                    f"exist under {system}/{family}/{backend}"
                )
                continue
            donor_table = donor_dir / f"{table}.parquet"
            if not donor_table.is_file():
                failures.append(
                    f"{rel_reuse}: table '{table}' from_version '{from_version}' has no real parquet data at "
                    f"{donor_table.relative_to(data_root)} (donor must hold actual data, not itself be a "
                    "declared-reuse-only dir)"
                )
    return failures


# --------------------------------------------------------------------------
# R3: comm exclusion (design §6.5 rule 5)
# --------------------------------------------------------------------------


def check_r3_comm_exclusion(data_root: Path) -> list[str]:
    failures: list[str] = []
    for system_dir in _subdirs(data_root):
        comm_dir = system_dir / COMM_FAMILY
        if not comm_dir.is_dir():
            continue
        for reuse_path in sorted(comm_dir.rglob(REUSE_YAML)):
            failures.append(
                f"{reuse_path.relative_to(data_root)}: reuse.yaml is not allowed under the comm family "
                "(design §6.5 rule 5 -- NCCL/oneCCL curves are topology-bound)"
            )
    return failures


# --------------------------------------------------------------------------
# R4: family placement
# --------------------------------------------------------------------------


def check_r4_family_placement(
    data_root: Path,
    version_dirs: list[tuple[str, str, str, str, Path]],
    family_map: dict[str, str] | None,
) -> list[str]:
    if family_map is None:
        return [f"op catalog not found; cannot validate family placement (fail-closed, expected at {CATALOG_PATH})"]

    failures: list[str] = []
    for _system, family, _backend, _version, version_dir in version_dirs:
        for parquet_path in sorted(p for p in version_dir.iterdir() if p.is_file() and p.suffix == ".parquet"):
            stem = parquet_path.stem
            expected_family = family_map.get(stem)
            rel = parquet_path.relative_to(data_root)
            if expected_family is None:
                failures.append(f"{rel}: table '{stem}' is not mapped to any family in the op catalog")
            elif expected_family != family:
                failures.append(
                    f"{rel}: table '{stem}' belongs to family '{expected_family}' per the op catalog, "
                    f"but is filed under family '{family}'"
                )
    return failures


# --------------------------------------------------------------------------
# R5: identity (manifest v2 resolution)
# --------------------------------------------------------------------------


def check_r5_identity(catalog_path: Path) -> list[str]:
    return list(validate_resolution(catalog_path=catalog_path))


# --------------------------------------------------------------------------
# R6: no legacy markers
# --------------------------------------------------------------------------


def check_r6_no_legacy_markers(data_root: Path) -> list[str]:
    failures: list[str] = []
    for marker_name in LEGACY_MARKERS:
        for marker_path in sorted(data_root.rglob(marker_name)):
            failures.append(
                f"{marker_path.relative_to(data_root)}: legacy marker file must be migrated and removed "
                "(superseded by reuse.yaml / collection_meta.yaml)"
            )
    return failures


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


def run_audit(data_root: Path, catalog_path: Path) -> dict[str, list[str]]:
    """Run all six rules and return {rule: [failure messages]} (empty list = pass).

    Pure function of (data_root, catalog_path); never raises on data
    problems -- every problem becomes a failure message so the caller can
    print all of them, never stopping at the first.
    """
    if not data_root.is_dir():
        no_root = [f"data root {data_root} does not exist"]
        return {rule: (no_root if rule != "R5" else check_r5_identity(catalog_path)) for rule in RULES}

    version_dirs = iter_version_dirs(data_root)
    family_map = load_family_map(catalog_path)

    return {
        "R1": check_r1_sidecar_coverage(data_root, version_dirs),
        "R2": check_r2_reuse_validity(data_root, version_dirs),
        "R3": check_r3_comm_exclusion(data_root),
        "R4": check_r4_family_placement(data_root, version_dirs, family_map),
        "R5": check_r5_identity(catalog_path),
        "R6": check_r6_no_legacy_markers(data_root),
    }


def render_report(results: dict[str, list[str]]) -> tuple[str, bool]:
    lines: list[str] = []
    any_failures = False
    for rule in RULES:
        failures = results[rule]
        title = RULE_TITLES[rule]
        if failures:
            any_failures = True
            lines.append(f"[FAIL] {rule} {title}: {len(failures)} failure(s)")
            lines.extend(f"    - {msg}" for msg in failures)
        else:
            lines.append(f"[OK]   {rule} {title}")

    total = sum(len(v) for v in results.values())
    if any_failures:
        failed_rules = [rule for rule in RULES if results[rule]]
        lines.append(f"\ncollector data audit FAILED: {total} failure(s) across rule(s) {', '.join(failed_rules)}")
    else:
        lines.append("\ncollector data audit OK: all rules passed")
    return "\n".join(lines), any_failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT, help="Root of the systems/data tree.")
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH, help="Path to the op backend catalog yaml.")
    args = parser.parse_args(argv)

    results = run_audit(args.data_root, args.catalog)
    report, failed = render_report(results)
    print(report)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Marker-to-structured-metadata migration (Collector V3 design §5, §6.3).

Converts the two legacy presence-only marker files into the structured yaml
siblings the loader now reads yaml-first (`_version_dir_state` in
`aic-core/src/aiconfigurator_core/sdk/perf_database.py`), one conversion per
version dir, in place — no cross-family replication (the family-layout
migration, PR 2, already made every marker dir single-family):

- `SHARED_LAYER_REUSE.txt` -> `reuse.yaml`: one entry per table that ANY
  sibling version of the same `<system>/<family>/<backend>` actually holds
  (a real data file, not another marker/sidecar), with `from_version` set to
  the NEWEST such sibling per `packaging.version.Version` ordering (textually
  duplicated from `parse_support_matrix_version`,
  `aic-core/src/aiconfigurator_core/sdk/common.py:15` — this is today's
  effective newest-first donor, so behavior is preserved). A marker whose
  siblings hold no table at all (nothing to inherit) is dead data: fail
  closed and list the offender rather than emit an empty declaration.
  EXCEPTION — design §6.5 rule 5 excludes the `comm` family from
  sibling-version reuse ENTIRELY (NCCL/oneCCL curves are topology-bound, so
  cross-version shape-filling is wrong there). A `SHARED_LAYER_REUSE.txt`
  marker inside a `<system>/comm/<backend>/<version>` dir is deleted with NO
  `reuse.yaml` emitted — a comm `reuse.yaml` would be a standing
  contradiction of that rule, and PR 4's loader/CI audit must never see one.
- `INCOMPLETE.txt` -> `collection_meta.yaml`: a synthesized `provenance:
  legacy` sidecar (T6 amendment to design §5 — no hashes, since legacy data
  predates the collector's provenance writer) marking every table the
  version dir itself holds as `status: partial`. A marker-only INCOMPLETE dir
  (no data files) is unexpected on the current tree (PR 2 already dropped
  marker-only dirs at the family-layout migration) and fails closed rather
  than silently guessing what to synthesize.
- Backfill (AIC-1502, extends the legacy-provenance tier to ALL pre-V3
  data): any family version dir holding >=1 parquet table but carrying NO
  `collection_meta.yaml` at all — i.e. it never had an `INCOMPLETE.txt`
  marker either, because it was always served as complete — gets a
  synthesized `provenance: legacy` sidecar with every held table marked
  `status: complete` (not `partial`; unlike the `INCOMPLETE.txt` conversion
  above, nothing here was ever incomplete). A dir that already carries a
  sidecar (including the `INCOMPLETE.txt`-derived ones, which keep their
  `status: partial` entries) is left untouched — backfill never overwrites
  an existing `collection_meta.yaml`. A dir with no parquet table of its own
  (e.g. a declared-reuse-only dir holding only `reuse.yaml`) has nothing to
  describe and is skipped.

Both marker files are deleted as part of their conversion. The backfill has
no source marker to delete — it only adds a sidecar where none exists.

Modeled on `migrate_family_layout.py`: plan/execute/verify CLI modes, a `git`
subprocess wrapper for execute, fail-closed aborts that list every offender
before raising, deterministic sorted output, and idempotency (a migrated
tree's plan is empty).

    migrate_markers.py --data-root DATA            # print the plan
    migrate_markers.py --data-root DATA --execute  # perform the conversions
    migrate_markers.py --data-root DATA --verify   # structural + round-trip checks

`--verify` also round-trips every generated file through the real loader
parser (`aiconfigurator_core.sdk.perf_database._parse_reuse_yaml` /
`_load_collection_meta_yaml`) — this is a one-shot migration tool, not a
hot-path predicate, so taking the aic-core dependency for real-parser
verification is worth it (unlike e.g. `prediction_regression_gate/grid.py`'s
duplicated predicate).

See docs/perf_database/collector-v3-op-centric-design.md §5, §6.3.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml
from packaging.version import InvalidVersion, Version

logger = logging.getLogger("migrate_markers")

SHARED_LAYER_REUSE = "SHARED_LAYER_REUSE.txt"
INCOMPLETE = "INCOMPLETE.txt"
MARKER_NAMES = frozenset({SHARED_LAYER_REUSE, INCOMPLETE})

REUSE_YAML = "reuse.yaml"
COLLECTION_META_YAML = "collection_meta.yaml"
SIDECAR_NAMES = frozenset({REUSE_YAML, COLLECTION_META_YAML})

# Never treated as a "table" file when scanning a version dir for data/donors.
METADATA_NAMES = MARKER_NAMES | SIDECAR_NAMES

APPROVED_BY = "yimingl"
SHARED_REUSE_REASON = "migrated from SHARED_LAYER_REUSE.txt (legacy newest-first effective donor)"


def _spdx_header() -> str:
    """Repo-standard copyright header for emitted sidecars, dated to the year of
    emission (the copyright CI check requires the year to cover the last commit).
    """
    return (
        f"# SPDX-FileCopyrightText: Copyright (c) {date.today().year} NVIDIA CORPORATION & AFFILIATES."
        " All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n"
        "\n"
    )


# Design §6.5 rule 5: comm is excluded from sibling-version reuse entirely.
COMM_FAMILY = "comm"
COMM_EXCLUSION_LOG = "comm family excluded from sibling reuse (design §6.5 rule 5); marker dropped without declaration"


class MigrationError(Exception):
    """Fail-closed error: abort, never guess, never skip silently."""


@dataclass(frozen=True)
class ReuseEntry:
    table: str
    from_version: str


@dataclass(frozen=True)
class ReuseAction:
    src: Path  # relative SHARED_LAYER_REUSE.txt path
    dst: Path  # relative reuse.yaml path (same dir)
    entries: tuple[ReuseEntry, ...]  # sorted by table name


@dataclass(frozen=True)
class SidecarAction:
    src: Path  # relative INCOMPLETE.txt path
    dst: Path  # relative collection_meta.yaml path (same dir)
    framework: str  # backend dir name (version_dir.parent.name)
    version: str  # version dir name
    tables: tuple[str, ...]  # sorted table stems, all synthesized as status: partial


@dataclass(frozen=True)
class BackfillAction:
    """A parquet-holding version dir with NO `collection_meta.yaml` at all — it
    never had an `INCOMPLETE.txt` marker either, because it was always served as
    complete. Unlike `SidecarAction`, there is no source marker file to delete;
    this only adds a sidecar where none exists.
    """

    dst: Path  # relative collection_meta.yaml path
    framework: str  # backend dir name (version_dir.parent.name)
    version: str  # version dir name
    tables: tuple[str, ...]  # sorted parquet stems, all synthesized as status: complete


@dataclass(frozen=True)
class CommExclusionAction:
    """A `SHARED_LAYER_REUSE.txt` marker found inside a `comm`-family version dir.

    Design §6.5 rule 5 excludes `comm` from sibling-version reuse entirely, so this
    is deleted outright — no `reuse.yaml` is emitted for it.
    """

    src: Path  # relative SHARED_LAYER_REUSE.txt path


@dataclass
class ScanResult:
    reuse_actions: list[ReuseAction]
    sidecar_actions: list[SidecarAction]
    comm_exclusion_deletions: list[CommExclusionAction]
    backfill_actions: list[BackfillAction]


# --- version ordering (textually duplicated from parse_support_matrix_version, ---
# --- aic-core/src/aiconfigurator_core/sdk/common.py:15) --------------------------


def _parse_version(version: str) -> Version | None:
    try:
        return Version(version)
    except InvalidVersion:
        return None


def _version_sort_key(version: str) -> tuple[bool, Version, str]:
    """Newest-first sort key. Unparseable versions sort as oldest (none exist in
    the real tree today; the string tie-break keeps this deterministic anyway).
    """
    parsed = _parse_version(version)
    return (parsed is not None, parsed if parsed is not None else Version("0"), version)


# --- tree walking ------------------------------------------------------------------


def _iter_version_dirs(data_root: Path) -> list[Path]:
    """Every leaf directory (holds >=1 file) under the family-layout tree —
    i.e. every `<system>/<family>/<backend>/<version>/` dir. Depth-agnostic by
    design: siblings for donor lookup are simply `version_dir.parent`'s other
    subdirectories, so this does not need to know what a "family" or "backend"
    name looks like.
    """
    leaf_dirs = {path.parent for path in data_root.rglob("*") if path.is_file()}
    return sorted(leaf_dirs)


def _table_files(version_dir: Path) -> list[Path]:
    return sorted(f for f in version_dir.iterdir() if f.is_file() and f.name not in METADATA_NAMES)


def _parquet_table_stems(version_dir: Path) -> tuple[str, ...]:
    """Sorted parquet-file stems directly in a version dir — design §8's
    definition of a "table" for R1 sidecar-coverage purposes
    (`audit_collector_data.py`'s `_parquet_stems` mirrors this exactly). Narrower
    than `_table_files`: only real `.parquet` data counts as a table to backfill
    or verify coverage for.
    """
    return tuple(sorted(f.stem for f in version_dir.iterdir() if f.is_file() and f.suffix == ".parquet"))


def _is_comm_family_version_dir(version_dir: Path) -> bool:
    """True for `<system>/comm/<backend>/<version>` dirs. Relies on the family-layout
    tree PR 2 already produced (same assumption `SidecarAction.framework` makes via
    `version_dir.parent.name`): the family dir is `version_dir.parent.parent`.
    """
    return version_dir.parent.parent.name == COMM_FAMILY


# --- SHARED_LAYER_REUSE.txt -> reuse.yaml -------------------------------------------


def donor_table_map(version_dir: Path) -> dict[str, str]:
    """table stem -> newest OTHER sibling version dir (same parent) holding a real
    data file for that table. Empty when no sibling holds any table at all.
    """
    backend_dir = version_dir.parent
    this_version = version_dir.name
    table_versions: dict[str, list[str]] = defaultdict(list)
    for sibling in sorted(p for p in backend_dir.iterdir() if p.is_dir() and p.name != this_version):
        for f in _table_files(sibling):
            table_versions[f.stem].append(sibling.name)
    return {table: max(versions, key=_version_sort_key) for table, versions in table_versions.items()}


def render_reuse_yaml(entries: tuple[ReuseEntry, ...]) -> str:
    doc = {
        "schema_version": 1,
        "reuse": [
            {
                "table": entry.table,
                "from_version": entry.from_version,
                "reason": SHARED_REUSE_REASON,
                "approved_by": APPROVED_BY,
            }
            for entry in entries
        ],
    }
    return _spdx_header() + yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)


# --- INCOMPLETE.txt -> collection_meta.yaml -----------------------------------------


def _render_legacy_collection_meta_yaml(framework: str, version: str, tables: tuple[str, ...], status: str) -> str:
    doc = {
        "schema_version": 1,
        "provenance": "legacy",
        "runtime": {
            "framework": framework,
            "version": version,
        },
        "tables": {table: {"status": status} for table in tables},
    }
    return _spdx_header() + yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)


def render_collection_meta_yaml(action: SidecarAction) -> str:
    return _render_legacy_collection_meta_yaml(action.framework, action.version, action.tables, "partial")


def render_backfill_collection_meta_yaml(action: BackfillAction) -> str:
    """Backfill status is always `complete`, never `partial` — these dirs were
    always served as complete data; unlike `INCOMPLETE.txt` conversions, nothing
    here was ever flagged incomplete.
    """
    return _render_legacy_collection_meta_yaml(action.framework, action.version, action.tables, "complete")


# --- scan: walk the tree, apply both rules, fail closed -----------------------------


def scan_tree(data_root: Path) -> ScanResult:
    reuse_actions: list[ReuseAction] = []
    sidecar_actions: list[SidecarAction] = []
    comm_exclusion_deletions: list[CommExclusionAction] = []
    backfill_actions: list[BackfillAction] = []
    no_donor_offenders: list[str] = []
    marker_only_incomplete_offenders: list[str] = []

    for version_dir in _iter_version_dirs(data_root):
        shared_marker = version_dir / SHARED_LAYER_REUSE
        if shared_marker.is_file():
            rel_marker = shared_marker.relative_to(data_root)
            if _is_comm_family_version_dir(version_dir):
                logger.info("%s: %s", COMM_EXCLUSION_LOG, rel_marker)
                comm_exclusion_deletions.append(CommExclusionAction(src=rel_marker))
            else:
                donor_map = donor_table_map(version_dir)
                if not donor_map:
                    no_donor_offenders.append(str(rel_marker))
                else:
                    entries = tuple(ReuseEntry(table=t, from_version=v) for t, v in sorted(donor_map.items()))
                    reuse_actions.append(
                        ReuseAction(
                            src=rel_marker,
                            dst=(version_dir / REUSE_YAML).relative_to(data_root),
                            entries=entries,
                        )
                    )

        incomplete_marker = version_dir / INCOMPLETE
        has_incomplete_marker = incomplete_marker.is_file()
        if has_incomplete_marker:
            own_tables = tuple(sorted(f.stem for f in _table_files(version_dir)))
            if not own_tables:
                marker_only_incomplete_offenders.append(str(incomplete_marker.relative_to(data_root)))
            else:
                sidecar_actions.append(
                    SidecarAction(
                        src=incomplete_marker.relative_to(data_root),
                        dst=(version_dir / COLLECTION_META_YAML).relative_to(data_root),
                        framework=version_dir.parent.name,
                        version=version_dir.name,
                        tables=own_tables,
                    )
                )

        # Backfill (AIC-1502): a parquet-holding dir with no sidecar at all --
        # never had an INCOMPLETE.txt marker either (always served as complete).
        # Skips dirs that already have a collection_meta.yaml (including the
        # INCOMPLETE.txt-derived ones handled above) and dirs with no parquet of
        # their own (e.g. declared-reuse-only dirs -- nothing to describe).
        meta_path = version_dir / COLLECTION_META_YAML
        if not has_incomplete_marker and not meta_path.is_file():
            parquet_stems = _parquet_table_stems(version_dir)
            if parquet_stems:
                backfill_actions.append(
                    BackfillAction(
                        dst=meta_path.relative_to(data_root),
                        framework=version_dir.parent.name,
                        version=version_dir.name,
                        tables=parquet_stems,
                    )
                )

    if no_donor_offenders or marker_only_incomplete_offenders:
        parts = []
        if no_donor_offenders:
            parts.append(
                "SHARED_LAYER_REUSE.txt with no donor table in any sibling version (fail-closed): "
                + ", ".join(sorted(no_donor_offenders))
            )
        if marker_only_incomplete_offenders:
            parts.append(
                "INCOMPLETE.txt with no data files of its own, marker-only (fail-closed, unexpected on the "
                "current tree — PR 2 already dropped marker-only dirs): "
                + ", ".join(sorted(marker_only_incomplete_offenders))
            )
        raise MigrationError("; ".join(parts))

    reuse_actions.sort(key=lambda a: str(a.src))
    sidecar_actions.sort(key=lambda a: str(a.src))
    comm_exclusion_deletions.sort(key=lambda a: str(a.src))
    backfill_actions.sort(key=lambda a: str(a.dst))
    return ScanResult(
        reuse_actions=reuse_actions,
        sidecar_actions=sidecar_actions,
        comm_exclusion_deletions=comm_exclusion_deletions,
        backfill_actions=backfill_actions,
    )


# --- plan rendering ------------------------------------------------------------------


def render_plan(scan: ScanResult) -> list[str]:
    lines: list[str] = []
    for action in scan.reuse_actions:
        lines.append(f"git add {action.dst}  # reuse.yaml ({len(action.entries)} table(s)) replacing {action.src}")
        lines.append(f"git rm {action.src}")
    for action in scan.sidecar_actions:
        lines.append(
            f"git add {action.dst}  # collection_meta.yaml legacy sidecar "
            f"({len(action.tables)} table(s)) replacing {action.src}"
        )
        lines.append(f"git rm {action.src}")
    for action in scan.comm_exclusion_deletions:
        lines.append(f"git rm {action.src}  # {COMM_EXCLUSION_LOG}")
    for action in scan.backfill_actions:
        lines.append(
            f"git add {action.dst}  # collection_meta.yaml legacy backfill sidecar "
            f"({len(action.tables)} table(s), status: complete)"
        )
    lines.sort()
    total_marker_deletions = len(scan.reuse_actions) + len(scan.sidecar_actions) + len(scan.comm_exclusion_deletions)
    lines.append(
        f"# summary: shared_reuse_conversions={len(scan.reuse_actions)} "
        f"incomplete_sidecars={len(scan.sidecar_actions)} "
        f"marker_deletions={total_marker_deletions} "
        f"comm_exclusions={len(scan.comm_exclusion_deletions)} "
        f"legacy_backfills={len(scan.backfill_actions)}"
    )
    return lines


# --- git wrapper + execute ------------------------------------------------------------


def _git(args: list[str], cwd: Path) -> None:
    result = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise MigrationError(f"git {' '.join(args)} failed: {result.stderr.strip()}")


def execute_plan(data_root: Path, scan: ScanResult) -> None:
    for action in scan.reuse_actions:
        dst = data_root / action.dst
        dst.write_text(render_reuse_yaml(action.entries), encoding="utf-8")
        _git(["add", str(dst)], cwd=data_root)
        _git(["rm", "-f", "-q", str(data_root / action.src)], cwd=data_root)

    for action in scan.sidecar_actions:
        dst = data_root / action.dst
        dst.write_text(render_collection_meta_yaml(action), encoding="utf-8")
        _git(["add", str(dst)], cwd=data_root)
        _git(["rm", "-f", "-q", str(data_root / action.src)], cwd=data_root)

    for action in scan.comm_exclusion_deletions:
        _git(["rm", "-f", "-q", str(data_root / action.src)], cwd=data_root)

    for action in scan.backfill_actions:
        dst = data_root / action.dst
        dst.write_text(render_backfill_collection_meta_yaml(action), encoding="utf-8")
        _git(["add", str(dst)], cwd=data_root)

    errors: list[str] = []
    for action in (*scan.reuse_actions, *scan.sidecar_actions):
        if (data_root / action.src).exists():
            errors.append(f"still present at legacy path: {action.src}")
        if not (data_root / action.dst).exists():
            errors.append(f"missing at destination: {action.dst}")
    for action in scan.comm_exclusion_deletions:
        if (data_root / action.src).exists():
            errors.append(f"still present at legacy path: {action.src}")
    for action in scan.backfill_actions:
        if not (data_root / action.dst).exists():
            errors.append(f"missing at destination: {action.dst}")
    if errors:
        raise MigrationError("execute post-check failed: " + "; ".join(errors))


# --- verify ----------------------------------------------------------------------------


def verify_tree(data_root: Path) -> list[str]:
    errors: list[str] = []

    stray_markers = sorted(
        str(p.relative_to(data_root)) for p in data_root.rglob("*") if p.is_file() and p.name in MARKER_NAMES
    )
    if stray_markers:
        errors.append("legacy marker .txt file(s) remain: " + ", ".join(stray_markers))

    try:
        from aiconfigurator_core.sdk.perf_database import (
            _load_collection_meta_yaml,
            _parse_reuse_yaml,
        )
    except ImportError as exc:  # pragma: no cover - aic-core is an editable dependency of this repo
        errors.append(f"could not import aic-core loader parser for round-trip verification: {exc}")
        return errors

    for reuse_path in sorted(data_root.rglob(REUSE_YAML)):
        try:
            parsed = _parse_reuse_yaml(str(reuse_path))
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not parsed["entries"]:
            errors.append(f"{reuse_path.relative_to(data_root)}: reuse.yaml has no entries (not a valid conversion)")
            continue
        backend_dir = reuse_path.parent.parent
        for entry in parsed["entries"]:
            donor_dir = backend_dir / entry["from_version"]
            if not donor_dir.is_dir():
                errors.append(
                    f"{reuse_path.relative_to(data_root)}: from_version target does not exist: "
                    f"{donor_dir.relative_to(data_root)}"
                )
                continue
            if not any(f.stem == entry["table"] for f in _table_files(donor_dir)):
                errors.append(
                    f"{reuse_path.relative_to(data_root)}: donor {entry['from_version']} has no data file "
                    f"for table {entry['table']}"
                )

    # Coverage completeness (AIC-1502 backfill): every parquet-holding version
    # dir must have a collection_meta.yaml whose `tables` key set covers every
    # parquet stem present -- mirrors audit_collector_data.py's R1 rule, applied
    # here as a self-check that plan/execute actually closed the gap.
    for version_dir in _iter_version_dirs(data_root):
        parquet_stems = set(_parquet_table_stems(version_dir))
        if not parquet_stems:
            continue
        rel_dir = version_dir.relative_to(data_root)
        meta_path = version_dir / COLLECTION_META_YAML
        if not meta_path.is_file():
            errors.append(f"{rel_dir}: missing {COLLECTION_META_YAML} for parquet table(s) {sorted(parquet_stems)}")
            continue
        try:
            meta = _load_collection_meta_yaml(str(meta_path))
        except ValueError as exc:
            errors.append(str(exc))
            continue
        tables = meta.get("tables")
        covered = set(tables) if isinstance(tables, dict) else set()
        missing = sorted(parquet_stems - covered)
        if missing:
            errors.append(f"{rel_dir}/{COLLECTION_META_YAML}: sidecar does not cover parquet table(s) {missing}")

    return errors


# --- CLI ---------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="perform the conversions (default: print the plan)")
    mode.add_argument("--verify", action="store_true", help="check the tree has zero legacy markers left")
    args = parser.parse_args(argv)

    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        print(f"ABORT: data root not found: {data_root}", file=sys.stderr)
        return 1

    if args.verify:
        errors = verify_tree(data_root)
        if errors:
            for err in errors:
                print(f"VERIFY FAIL: {err}", file=sys.stderr)
            return 1
        print("VERIFY OK")
        return 0

    try:
        scan = scan_tree(data_root)
    except MigrationError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 1

    if args.execute:
        try:
            execute_plan(data_root, scan)
        except MigrationError as exc:
            print(f"ABORT: {exc}", file=sys.stderr)
            return 1
        total_marker_deletions = (
            len(scan.reuse_actions) + len(scan.sidecar_actions) + len(scan.comm_exclusion_deletions)
        )
        print(
            f"executed {len(scan.reuse_actions)} reuse.yaml conversion(s), "
            f"{len(scan.sidecar_actions)} collection_meta.yaml sidecar(s), "
            f"{len(scan.backfill_actions)} legacy backfill sidecar(s), "
            f"{len(scan.comm_exclusion_deletions)} comm-family exclusion(s) (no declaration emitted), "
            f"{total_marker_deletions} marker deletion(s)"
        )
        return 0

    for line in render_plan(scan):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

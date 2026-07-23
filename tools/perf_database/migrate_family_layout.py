# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Catalog-driven migration of the perf-data tree to the family-first layout.

Legacy layout:  <data_root>/<system>/<backend>/<version>/<table>
Family layout:  <data_root>/<system>/<family>/<backend>/<version>/<table>

`<family>` comes from `collector/op_backend_catalog.yaml` (families[].op_files);
this is the only catalog consumer in this script — it is parsed directly with
`yaml.safe_load`, collector modules are never imported.

Fail-closed: a parquet/txt table whose stem is not in the catalog, or a
first-level directory under a system dir that is neither a known legacy
backend dir nor an existing catalog family dir, aborts the whole run with a
clear error. So does a legacy marker whose system+backend already has
family-layout data (a partially migrated tree, e.g. an interrupted
--execute): rescanning such a tree cannot derive the marker's scope from the
legacy layout alone, so the run aborts instead of dropping the marker. The
script never guesses a family and never skips silently.

Count-verified workflow (recommended, three calls):
    migrate_family_layout.py --data-root DATA --manifest M
        # plan mode: prints the plan and writes the PRE-migration per-table
        # count manifest (table filename -> count) to M.
    migrate_family_layout.py --data-root DATA --execute
        # performs the moves (has its own internal pre/post count self-check).
    migrate_family_layout.py --data-root DATA --verify --manifest M
        # structural checks plus: recomputes POST-migration per-table counts
        # across the family dirs and compares against M, erroring on any
        # table whose count differs or is missing.
`--verify` without `--manifest` still runs the structural checks but prints
a one-line notice that count verification was skipped — it never silently
implies completeness.

See docs/perf_database/collector-v3-op-centric-design.md §3.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger("migrate_family_layout")

# Textually identical to the CANONICAL _KNOWN_BACKEND_DIRS in
# aic-core/src/aiconfigurator_core/sdk/operations/base.py, which lists every
# copy that must stay in sync (this standalone tool cannot import aic-core).
KNOWN_BACKEND_DIRS = frozenset({"trtllm", "sglang", "vllm", "nccl", "oneccl"})

SHARED_LAYER_REUSE = "SHARED_LAYER_REUSE.txt"
INCOMPLETE = "INCOMPLETE.txt"
MARKER_NAMES = frozenset({SHARED_LAYER_REUSE, INCOMPLETE})

# Structured per-version sidecars (provenance + declared reuse) that live beside
# the perf tables in FAMILY-layout version dirs only: verify_tree must not treat
# them as perf tables ("table not in catalog") and they never count toward the
# per-table manifest. A legacy-layout version dir never legitimately contains
# them, so scan_legacy_tree keeps failing closed on any occurrence there.
STRUCTURED_SIDECAR_NAMES = frozenset({"collection_meta.yaml", "reuse.yaml"})

DEFAULT_CATALOG = Path(__file__).resolve().parents[2] / "collector" / "op_backend_catalog.yaml"


class MigrationError(Exception):
    """Fail-closed error: abort, never guess, never skip silently."""


@dataclass(frozen=True)
class TableMove:
    src: Path  # relative to data_root
    dst: Path  # relative to data_root


@dataclass(frozen=True)
class MarkerAction:
    marker: str  # SHARED_LAYER_REUSE.txt or INCOMPLETE.txt
    src: Path  # relative to data_root
    targets: tuple[Path, ...]  # relative dest paths the marker is copied into; empty -> dropped
    reason: str


@dataclass
class ScanResult:
    moves: list[TableMove]
    marker_actions: list[MarkerAction]
    manifest: dict[str, int]  # table filename (with ext) -> count of legacy occurrences
    legacy_dirs: list[Path]  # version dirs (relative) touched, for post-move rmdir cleanup


# --- Rule 1: catalog lookup -------------------------------------------------


def load_family_map(catalog_path: Path) -> dict[str, str]:
    """table basename (no extension) -> family name."""
    data = yaml.safe_load(catalog_path.read_text())
    mapping: dict[str, str] = {}
    for fam in data["families"]:
        family = fam["family"]
        for op_file in fam.get("op_files", []):
            mapping[op_file] = family
    return mapping


def family_for_table(basename: str, family_map: dict[str, str]) -> str:
    try:
        return family_map[basename]
    except KeyError:
        raise MigrationError(f"table '{basename}' has no family in the catalog") from None


# --- Rule 2: table move ------------------------------------------------------


def plan_table_move(system: str, backend: str, version: str, filename: str, family_map: dict[str, str]) -> TableMove:
    """<sys>/<backend>/<ver>/<table> -> <sys>/<family(table)>/<backend>/<ver>/<table>.

    Applies unchanged to pseudo-backends (nccl/oneccl land under family "comm").
    """
    stem = Path(filename).stem
    family = family_for_table(stem, family_map)
    src = Path(system, backend, version, filename)
    dst = Path(system, family, backend, version, filename)
    return TableMove(src=src, dst=dst)


# --- Rule 3: SHARED_LAYER_REUSE.txt replication scope -----------------------


def shared_layer_reuse_scope(version_family_index: dict[str, set[str]], this_version: str) -> set[str]:
    """Families with data in any OTHER version of the same system+backend."""
    scope: set[str] = set()
    for version, families in version_family_index.items():
        if version == this_version:
            continue
        scope |= families
    return scope


# --- Rule 4: INCOMPLETE.txt targets ------------------------------------------


def incomplete_targets(data_file_stems: Iterable[str], family_map: dict[str, str]) -> set[str]:
    """Families this version's own data files move to (empty -> marker-only)."""
    return {family_for_table(stem, family_map) for stem in data_file_stems}


# --- Rule 5: empty legacy dir cleanup ----------------------------------------


def cleanup_empty_dirs(version_dirs: Iterable[Path]) -> list[Path]:
    """rmdir each now-empty version dir, then its backend dir if also empty."""
    removed: list[Path] = []
    backend_dirs: set[Path] = set()
    for vdir in version_dirs:
        backend_dirs.add(vdir.parent)
        if vdir.is_dir() and not any(vdir.iterdir()):
            vdir.rmdir()
            removed.append(vdir)
    for bdir in sorted(backend_dirs, key=lambda p: -len(p.parts)):
        if bdir.is_dir() and not any(bdir.iterdir()):
            bdir.rmdir()
            removed.append(bdir)
    return removed


# --- Scan: walk the legacy layout, apply rules 1-4, fail closed -------------


def _system_dirs(data_root: Path) -> list[Path]:
    # Dot-dirs (e.g. .git, when data_root happens to be a repo root as in tests)
    # are never perf-data systems.
    return sorted(p for p in data_root.iterdir() if p.is_dir() and not p.name.startswith("."))


def scan_legacy_tree(data_root: Path, family_map: dict[str, str], family_set: set[str]) -> ScanResult:
    moves: list[TableMove] = []
    marker_actions: list[MarkerAction] = []
    manifest: Counter[str] = Counter()
    legacy_dirs: list[Path] = []
    unexpected_dirs: list[str] = []
    unknown_tables: list[str] = []
    mixed_marker_dirs: list[str] = []

    for system_dir in _system_dirs(data_root):
        system = system_dir.name
        # Backends that already hold family-layout data under this system. A
        # legacy marker for such a backend means the tree is partially migrated
        # (e.g. an interrupted --execute): the marker's replication scope can no
        # longer be derived from the legacy layout alone, so treating it as
        # marker-only would silently drop it. Fail closed instead.
        migrated_backends: set[str] = set()
        for entry in sorted(system_dir.iterdir()):
            if entry.is_dir() and entry.name in family_set:
                migrated_backends.update(b.name for b in entry.iterdir() if b.is_dir())
        for entry in sorted(system_dir.iterdir()):
            if not entry.is_dir():
                continue  # e.g. stray .gitkeep directly under a system dir: leave alone
            backend = entry.name
            if backend not in KNOWN_BACKEND_DIRS:
                if backend in family_set:
                    continue  # already-migrated family dir from a prior partial run
                unexpected_dirs.append(str(entry.relative_to(data_root)))
                continue

            version_dirs = sorted(p for p in entry.iterdir() if p.is_dir())

            # Pass 1: per-version data-file families (needed for rule 3's "other version" scope).
            version_files: dict[str, list[Path]] = {}
            version_family_index: dict[str, set[str]] = {}
            for vdir in version_dirs:
                files = sorted(p for p in vdir.iterdir() if p.is_file())
                data_files = [f for f in files if f.name not in MARKER_NAMES]
                fams: set[str] = set()
                for f in data_files:
                    if f.stem not in family_map:
                        unknown_tables.append(str(f.relative_to(data_root)))
                        continue
                    fams.add(family_map[f.stem])
                version_files[vdir.name] = files
                version_family_index[vdir.name] = fams

            # Pass 2: build moves + marker actions.
            for vdir in version_dirs:
                legacy_dirs.append(vdir.relative_to(data_root))
                files = version_files[vdir.name]
                data_files = [f for f in files if f.name not in MARKER_NAMES]

                for f in data_files:
                    if f.stem not in family_map:
                        continue  # already recorded in unknown_tables above
                    mv = plan_table_move(system, backend, vdir.name, f.name, family_map)
                    moves.append(mv)
                    manifest[f.name] += 1

                has_marker = (vdir / SHARED_LAYER_REUSE).is_file() or (vdir / INCOMPLETE).is_file()
                if has_marker and backend in migrated_backends:
                    mixed_marker_dirs.append(str(vdir.relative_to(data_root)))
                    continue

                if (vdir / SHARED_LAYER_REUSE).is_file():
                    scope = shared_layer_reuse_scope(version_family_index, vdir.name)
                    marker_actions.append(_plan_marker(system, backend, vdir.name, SHARED_LAYER_REUSE, scope))

                if (vdir / INCOMPLETE).is_file():
                    own_stems = (f.stem for f in data_files if f.stem in family_map)
                    targets = incomplete_targets(own_stems, family_map)
                    marker_actions.append(_plan_marker(system, backend, vdir.name, INCOMPLETE, targets))

    if unexpected_dirs or unknown_tables or mixed_marker_dirs:
        parts = []
        if unexpected_dirs:
            parts.append(
                "unexpected directory under a system dir (not a known backend dir or catalog family dir): "
                + ", ".join(sorted(set(unexpected_dirs)))
            )
        if unknown_tables:
            parts.append("tables with no catalog family (fail-closed): " + ", ".join(sorted(set(unknown_tables))))
        if mixed_marker_dirs:
            parts.append(
                "legacy marker in a partially migrated tree (family-layout data already exists for the same "
                "system+backend, so the marker's scope cannot be derived from the legacy layout alone; restore "
                "the tree with git before re-running): " + ", ".join(sorted(set(mixed_marker_dirs)))
            )
        raise MigrationError("; ".join(parts))

    moves.sort(key=lambda m: str(m.src))
    marker_actions.sort(key=lambda a: str(a.src))
    return ScanResult(moves=moves, marker_actions=marker_actions, manifest=dict(manifest), legacy_dirs=legacy_dirs)


def _plan_marker(system: str, backend: str, version: str, marker_name: str, families: set[str]) -> MarkerAction:
    src = Path(system, backend, version, marker_name)
    if families:
        targets = tuple(sorted(Path(system, fam, backend, version, marker_name) for fam in families))
        reason = f"replicate into {len(targets)} family dir(s): {', '.join(sorted(families))}"
    else:
        targets = ()
        reason = "marker-only, no target families: dropped"
        logger.warning("dropping marker-only %s (%s)", src, marker_name)
    return MarkerAction(marker=marker_name, src=src, targets=targets, reason=reason)


# --- Plan rendering -----------------------------------------------------------


def render_plan(scan: ScanResult) -> list[str]:
    action_lines: list[str] = [f"git mv {mv.src} {mv.dst}" for mv in scan.moves]
    for action in scan.marker_actions:
        if action.targets:
            for target in action.targets:
                action_lines.append(f"git add {target}  # replicate {action.marker} from {action.src}")
            action_lines.append(f"git rm {action.src}")
        else:
            action_lines.append(f"git rm {action.src}  # {action.reason}")
    action_lines.sort()
    manifest_lines = [f"# manifest: {name}={count}" for name, count in sorted(scan.manifest.items())]
    return action_lines + manifest_lines


# --- git wrapper + execute ----------------------------------------------------


def _git(args: list[str], cwd: Path) -> None:
    result = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise MigrationError(f"git {' '.join(args)} failed: {result.stderr.strip()}")


def execute_plan(data_root: Path, scan: ScanResult) -> None:
    for mv in scan.moves:
        src = data_root / mv.src
        dst = data_root / mv.dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        _git(["mv", str(src), str(dst)], cwd=data_root)

    for action in scan.marker_actions:
        src = data_root / action.src
        for target in action.targets:
            dst = data_root / target
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            _git(["add", str(dst)], cwd=data_root)
        # -f: the marker is deliberately being replaced/dropped; it may not yet be committed
        # (e.g. the first migration run happens against an uncommitted working tree).
        _git(["rm", "-f", "-q", str(src)], cwd=data_root)

    cleanup_empty_dirs([data_root / rel for rel in scan.legacy_dirs])

    # Re-derive the manifest from the moved-to destinations and compare against
    # the pre-move manifest plan mode already printed: no table may be lost.
    errors: list[str] = []
    manifest_after: Counter[str] = Counter()
    for mv in scan.moves:
        src, dst = data_root / mv.src, data_root / mv.dst
        if src.exists():
            errors.append(f"still present at legacy path: {mv.src}")
        if not dst.exists():
            errors.append(f"missing at destination: {mv.dst}")
        else:
            manifest_after[mv.dst.name] += 1
    if dict(manifest_after) != scan.manifest:
        errors.append(f"manifest mismatch: before={scan.manifest} after={dict(manifest_after)}")
    if errors:
        raise MigrationError("execute post-check failed: " + "; ".join(errors))


# --- Manifest (pre-migration per-table counts, for --verify's count check) ----


def write_manifest(path: Path, manifest: dict[str, int]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def load_manifest(path: Path) -> dict[str, int]:
    return json.loads(path.read_text())


# --- Verify --------------------------------------------------------------------


def verify_tree(
    data_root: Path,
    family_map: dict[str, str],
    family_set: set[str],
    manifest: dict[str, int] | None = None,
) -> list[str]:
    errors: list[str] = []
    try:
        scan = scan_legacy_tree(data_root, family_map, family_set)
    except MigrationError as exc:
        return [str(exc)]

    if scan.moves:
        errors.append("legacy-shaped tables remain: " + ", ".join(sorted(str(m.src) for m in scan.moves)))
    if scan.marker_actions:
        errors.append("legacy-shaped markers remain: " + ", ".join(sorted(str(a.src) for a in scan.marker_actions)))

    post_counts: Counter[str] = Counter()
    for system_dir in _system_dirs(data_root):
        for family_dir in sorted(p for p in system_dir.iterdir() if p.is_dir()):
            family = family_dir.name
            if family not in family_set:
                errors.append(f"not a catalog family: {family_dir.relative_to(data_root)}")
                continue
            for backend_dir in sorted(p for p in family_dir.iterdir() if p.is_dir()):
                for version_dir in sorted(p for p in backend_dir.iterdir() if p.is_dir()):
                    for f in sorted(p for p in version_dir.iterdir() if p.is_file()):
                        if f.name in MARKER_NAMES or f.name in STRUCTURED_SIDECAR_NAMES:
                            continue
                        if f.stem not in family_map:
                            errors.append(f"table not in catalog: {f.relative_to(data_root)}")
                            continue
                        post_counts[f.name] += 1
                        if family_map[f.stem] != family:
                            errors.append(
                                f"table under wrong family (expected {family_map[f.stem]}): {f.relative_to(data_root)}"
                            )

    if manifest is not None:
        for name, pre_count in sorted(manifest.items()):
            post_count = post_counts.get(name)
            if post_count is None:
                errors.append(f"count check: table missing after migration: {name} (pre-migration count={pre_count})")
            elif post_count != pre_count:
                errors.append(f"count check: table count mismatch: {name} (manifest={pre_count}, found={post_count})")

    return errors


# --- CLI -----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="perform the moves (default: print the plan)")
    mode.add_argument("--verify", action="store_true", help="check the tree is fully migrated")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "path to a JSON per-table count manifest. In plan mode (and --execute) the "
            "PRE-migration manifest is written here. With --verify, the manifest is read back "
            "and POST-migration per-table counts are checked against it; --verify without "
            "--manifest skips this count check (prints a notice, does not fail)."
        ),
    )
    args = parser.parse_args(argv)

    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        print(f"ABORT: data root not found: {data_root}", file=sys.stderr)
        return 1

    family_map = load_family_map(DEFAULT_CATALOG)
    family_set = set(family_map.values())

    if args.verify:
        manifest = None
        if args.manifest:
            manifest = load_manifest(args.manifest)
        else:
            print("NOTICE: --verify without --manifest: per-table count check skipped", file=sys.stderr)
        errors = verify_tree(data_root, family_map, family_set, manifest=manifest)
        if errors:
            for err in errors:
                print(f"VERIFY FAIL: {err}", file=sys.stderr)
            return 1
        print("VERIFY OK")
        return 0

    try:
        scan = scan_legacy_tree(data_root, family_map, family_set)
    except MigrationError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 1

    if args.manifest:
        write_manifest(args.manifest, scan.manifest)

    if args.execute:
        try:
            execute_plan(data_root, scan)
        except MigrationError as exc:
            print(f"ABORT: {exc}", file=sys.stderr)
            return 1
        print(f"executed {len(scan.moves)} table move(s), {len(scan.marker_actions)} marker action(s)")
        return 0

    for line in render_plan(scan):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

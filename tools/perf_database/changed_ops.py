# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Changed-operation manifest: diff two git revisions per (framework, family).

Collector V3 design §8 (docs/perf_database/collector-v3-op-centric-design.md):
this is the single input consumed by the evidence resolver (§9), the CI gate
(AIC-1214), and the out-of-repo support-matrix healer — its output schema is
a LOCKED contract.

For each (framework, family) pair the tool compares three GPU-free signals
between `--base` and `--head`:

1. ``pin_version`` — manifest v2 resolution (`families[family] or default`)
   differs: framework version OR image (which carries the digest).
2. ``collector_code`` — any collector module mapped to the family has a
   differing revision-aware ``collector_hash`` (module file + the shared core
   + its hash_closures.yaml extras, all read from that revision).
3. ``case_plan`` — the family's case-INPUT files differ: `case_generator.py`
   + `model_cases.py` plus every hash-closure extra under `collector/cases/`
   for the family's modules. This is a GPU-free content hash over case
   *inputs*, deterministic at any revision — distinct from
   `collection_meta.yaml`'s `case_plan_hash`, which is the GPU-attested hash
   of the actually-expanded case-id set at collection time (T6 amendment).

Everything is read via ``git show <rev>:<path>`` / ``git ls-tree`` — never by
checking out a revision or executing collector code. Registries are Python
source; they are parsed with ``ast``, never imported/exec'd, so this tool
never needs the frameworks it diffs to be installed.

Standalone by design (mirrors, does not import, `parquet_diff.py`'s git
plumbing: `_git`/`_read_git_file`/`_git_file_exists`/`_parse_diff` at
parquet_diff.py:113-154). Revision-purity extends to `collector/provenance.py`
itself: `SHARED_CORE` is a file LIST (it changes what gets hashed), so it is
parsed via `ast` from `git show <rev>:collector/provenance.py` independently
at BASE and HEAD (`_shared_core_at_rev`) rather than imported from the live
checkout — a rename or reshaping of `SHARED_CORE` on one side of the diff
must be visible to that side's hash. `MODEL_CASES_GROUP` (a sentinel token
used inside `hash_closures.yaml`) and its directory are naming/path
CONVENTIONS, not a file list — like `MANIFEST_PATH`/`CATALOG_PATH`/
`DATA_PREFIX` below, they are hardcoded literals here rather than re-parsed
per revision.

Registry module paths are derived from each revision's OWN
`framework_manifest.yaml` content (`collector_dir` for wideep frameworks,
else the `collector/<framework>/registry.py` convention) rather than the
currently-checked-out `collector.framework_manifest._REGISTRY_MODULES` dict:
this keeps the tool revision-safe (a framework's registry path is exactly
what that revision's manifest says it is) and independently testable with
small fixture trees that do not need to replicate all five real registries.

Known gaps
----------
- ``REGISTRY_XPU`` (`collector/vllm/registry.py`, selected by `collect_vllm`
  on Intel XPU — collect.py:1165-1170) is a second, XPU-only registry list
  that neither this tool nor `provenance.enumerate_registry_modules` ever
  reads (both resolve only the CUDA-path `.REGISTRY`). XPU-only collector
  edits are therefore invisible to change tracking today. Tracked as a
  follow-up (Linear issue to be filed: "Collector V3 follow-up: XPU registry
  tracking").
- VersionRoute-versioned registry entries (`OpEntry(..., versions=(...))`,
  no literal `module=`) are not parsed: `_parse_registry_entries` raises
  `NotImplementedError` rather than silently dropping them (fail-closed, not
  a silent gap like the XPU one above). No registry uses this shape today.

Exit codes
----------
0   success — report emitted to `--out` or stdout.
3   `--base` (or `--head`) predates Collector V3 provenance metadata
    (missing `collector/provenance.py`, `collector/hash_closures.yaml`, or
    `collector/op_backend_catalog.yaml` at that revision): the
    changed-operation manifest cannot be computed against it. The CI
    workflow maps this exit code to a neutral skip, not a failure.

    changed_ops.py --base origin/main --head HEAD [--out FILE]
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

MANIFEST_PATH = "collector/framework_manifest.yaml"
CATALOG_PATH = "collector/op_backend_catalog.yaml"
HASH_CLOSURES_PATH = "collector/hash_closures.yaml"
REGISTRY_TYPES_PATH = "collector/registry_types.py"
PROVENANCE_PATH = "collector/provenance.py"
DATA_PREFIX = "aic-core/src/aiconfigurator_core/systems/data"

# Naming/path CONVENTIONS mirrored from collector/provenance.py, not a file
# list — see module docstring. Kept as hardcoded literals (like the PATH
# constants above) rather than re-parsed per revision.
MODEL_CASES_GROUP = "__model_cases__"
MODEL_CASES_DIR = "collector/cases/models"

# collector/provenance.py, collector/hash_closures.yaml, and
# collector/op_backend_catalog.yaml together are the Collector V3 provenance
# baseline. If `--base` or `--head` predates all three being introduced
# together, there is nothing for this tool to diff against (see module
# docstring "Exit codes").
V3_BASELINE_PATHS = (PROVENANCE_PATH, HASH_CLOSURES_PATH, CATALOG_PATH)

# case_plan's shared file set (design §8 T6 amendment): case_generator.py and
# model_cases.py are already part of provenance.SHARED_CORE for collector_hash,
# but case_plan is spelled out explicitly and independently per the task brief.
CASE_PLAN_SHARED_FILES = ("collector/case_generator.py", "collector/model_cases.py")

REASON_PIN_VERSION = "pin_version"
REASON_COLLECTOR_CODE = "collector_code"
REASON_CASE_PLAN = "case_plan"

ACTION_RECOLLECT = "recollect"


# --------------------------------------------------------------------------
# git plumbing (imitates parquet_diff.py:113-154; kept standalone/local so
# this tool has no import dependency on parquet_diff.py)
# --------------------------------------------------------------------------


def _git(repo_root: Path, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo_root, capture_output=True, check=check)


def _git_file_exists(repo_root: Path, rev: str, path: str) -> bool:
    return _git(repo_root, ["cat-file", "-e", f"{rev}:{path}"], check=False).returncode == 0


def _resolve_rev(repo_root: Path, rev: str) -> str:
    """Pin a possibly-symbolic rev (branch, HEAD, origin/main) to a concrete
    commit sha up front, so every `git show <rev>:<path>` call below reads a
    fixed, cacheable snapshot regardless of what `rev` points to later.
    """
    proc = _git(repo_root, ["rev-parse", rev])
    return proc.stdout.decode("utf-8").strip()


@cache
def _read_git_file(repo_root: Path, rev: str, path: str) -> bytes:
    return _git(repo_root, ["show", f"{rev}:{path}"]).stdout


def _read_git_text(repo_root: Path, rev: str, path: str) -> str:
    return _read_git_file(repo_root, rev, path).decode("utf-8")


@cache
def _git_ls_tree(repo_root: Path, rev: str, path_prefix: str) -> tuple[str, ...]:
    proc = _git(repo_root, ["ls-tree", "-r", "--name-only", rev, "--", path_prefix])
    return tuple(line for line in proc.stdout.decode("utf-8").splitlines() if line)


def _load_yaml_at_rev(repo_root: Path, rev: str, path: str) -> dict[str, Any]:
    data = yaml.safe_load(_read_git_text(repo_root, rev, path))
    return data if isinstance(data, dict) else {}


# --------------------------------------------------------------------------
# manifest v2 pin resolution (reimplemented minimally for revision-diffing —
# framework_manifest.resolve_op_runtime only reads the checked-out tree)
# --------------------------------------------------------------------------


def _manifest_at_rev(repo_root: Path, rev: str) -> dict[str, Any]:
    return _load_yaml_at_rev(repo_root, rev, MANIFEST_PATH)


PinKey = tuple[str, tuple[tuple[str, str], ...]] | None


def _resolve_pin(manifest: dict[str, Any], framework: str, family: str) -> PinKey:
    """(version, sorted images) for (framework, family) — `families[family] or
    default`, spec §4. None when the framework has no manifest entry at this
    revision (e.g. a framework added only on one side of the diff).
    """
    spec = manifest.get("frameworks", {}).get(framework)
    if spec is None:
        return None
    families = spec.get("families") or {}
    runtime_spec = families.get(family) or spec["default"]
    images = tuple(sorted((runtime_spec.get("images") or {}).items()))
    return (runtime_spec["version"], images)


def _data_backend(manifest: dict[str, Any], framework: str) -> str | None:
    spec = manifest.get("frameworks", {}).get(framework)
    if spec is None:
        return None
    return spec.get("data_backend") or framework


def _registry_path_for_framework(spec: dict[str, Any], framework: str) -> str:
    collector_dir = spec.get("collector_dir")
    if collector_dir:
        return f"{collector_dir}/registry.py"
    return f"collector/{framework}/registry.py"


# --------------------------------------------------------------------------
# op catalog: table stem -> family (reimplemented minimally; see op_catalog.py)
# --------------------------------------------------------------------------


def _family_map_at_rev(repo_root: Path, rev: str) -> dict[str, str]:
    data = _load_yaml_at_rev(repo_root, rev, CATALOG_PATH)
    family_map: dict[str, str] = {}
    for entry in data.get("families", []):
        family = entry["family"]
        for op_file in entry["op_files"]:
            family_map[op_file] = family
    return family_map


# --------------------------------------------------------------------------
# hash_closures.yaml at a revision
# --------------------------------------------------------------------------


def _closures_at_rev(repo_root: Path, rev: str) -> dict[str, list[str]]:
    data = _load_yaml_at_rev(repo_root, rev, HASH_CLOSURES_PATH)
    return {module: list(extras or []) for module, extras in data.items()}


def _expand_closure_extras_at_rev(repo_root: Path, rev: str, extras: list[str]) -> set[str]:
    files: set[str] = set()
    for extra in extras:
        if extra == MODEL_CASES_GROUP:
            files.update(p for p in _git_ls_tree(repo_root, rev, MODEL_CASES_DIR) if p.endswith(".yaml"))
        else:
            files.add(extra)
    return files


# --------------------------------------------------------------------------
# registry parsing: PerfFile enum + OpEntry(...) calls, via ast — never exec'd
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RegistryEntry:
    op: str
    module: str
    table_stem: str


@cache
def _perf_file_map_at_rev(repo_root: Path, rev: str) -> dict[str, str]:
    """{enum member NAME -> filename}, e.g. {"GEMM": "gemm_perf.txt"}."""
    tree = ast.parse(_read_git_text(repo_root, rev, REGISTRY_TYPES_PATH))
    mapping: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "PerfFile":
            for item in node.body:
                is_single_name_assign = (
                    isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name)
                )
                if not is_single_name_assign:
                    continue
                value = item.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    mapping[item.targets[0].id] = value.value
            break
    return mapping


def _ast_str(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _resolve_perf_filename(node: ast.expr | None, perf_file_map: dict[str, str]) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "PerfFile":
        return perf_file_map.get(node.attr)
    return _ast_str(node)


def _assigns_name(node: ast.stmt, name: str) -> bool:
    """Whether `node` is a module-level `<name> = ...` / `<name>: T = ...`."""
    if isinstance(node, ast.Assign):
        return any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    if isinstance(node, ast.AnnAssign):
        return isinstance(node.target, ast.Name) and node.target.id == name
    return False


def _assigns_registry(node: ast.stmt) -> bool:
    return _assigns_name(node, "REGISTRY")


def _find_registry_list(tree: ast.Module) -> ast.List | ast.Tuple | None:
    """The module-level `REGISTRY: list[OpEntry] = [...]` (or unannotated
    `REGISTRY = [...]`) assignment's list, module-top-level statements only.

    Deliberately does NOT walk the whole file: some registries also define an
    inactive sibling list (e.g. `REGISTRY_XPU` in collector/vllm/registry.py)
    that the real loader (`importlib.import_module(...).REGISTRY`, both
    `framework_manifest._registry_entries` and
    `provenance.enumerate_registry_modules`) never reads either.
    """
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and _assigns_registry(node):
            value = node.value
            return value if isinstance(value, (ast.List, ast.Tuple)) else None
    return None


def _parse_registry_entries(repo_root: Path, rev: str, manifest: dict[str, Any], framework: str) -> list[RegistryEntry]:
    spec = manifest.get("frameworks", {}).get(framework)
    if spec is None:
        return []  # framework does not exist at this revision
    registry_path = _registry_path_for_framework(spec, framework)
    if not _git_file_exists(repo_root, rev, registry_path):
        return []

    perf_file_map = _perf_file_map_at_rev(repo_root, rev)
    tree = ast.parse(_read_git_text(repo_root, rev, registry_path))
    registry_list = _find_registry_list(tree)
    if registry_list is None:
        return []

    entries: list[RegistryEntry] = []
    for element in registry_list.elts:
        if not (isinstance(element, ast.Call) and isinstance(element.func, ast.Name) and element.func.id == "OpEntry"):
            continue
        kwargs = {kw.arg: kw.value for kw in element.keywords if kw.arg}
        op = _ast_str(kwargs.get("op"))
        module = _ast_str(kwargs.get("module"))
        if module is None and "versions" in kwargs:
            # Versioned entries (module=None, versions=(VersionRoute(...), ...))
            # are not parsed — fail closed rather than silently omitting the
            # op from its family's module set (see "Known gaps" above).
            raise NotImplementedError(
                f"changed_ops does not support VersionRoute-versioned registry entries; extend the parser (op: {op})"
            )
        perf_filename = _resolve_perf_filename(kwargs.get("perf_filename"), perf_file_map)
        if not (op and module and perf_filename):
            continue
        entries.append(RegistryEntry(op=op, module=module, table_stem=Path(perf_filename).stem))
    return entries


def _group_by_family(entries: list[RegistryEntry], family_map: dict[str, str], key: str) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        family = family_map.get(entry.table_stem)
        if family:
            result[family].add(entry.module if key == "module" else entry.table_stem)
    return result


# --------------------------------------------------------------------------
# collector/provenance.py's SHARED_CORE at a revision — ast-parsed from
# `git show <rev>:collector/provenance.py`, never imported from the live
# checkout (see module docstring: SHARED_CORE is a file LIST, so it must be
# read from the same revision whose collector_hash it feeds).
# --------------------------------------------------------------------------


@cache
def _shared_core_at_rev(repo_root: Path, rev: str) -> tuple[str, ...]:
    """SHARED_CORE (a module-level tuple-of-string-literals) at this revision."""
    tree = ast.parse(_read_git_text(repo_root, rev, PROVENANCE_PATH))
    for node in tree.body:
        if not (isinstance(node, (ast.Assign, ast.AnnAssign)) and _assigns_name(node, "SHARED_CORE")):
            continue
        value = node.value
        if isinstance(value, (ast.Tuple, ast.List)):
            items = [_ast_str(elt) for elt in value.elts]
            if all(isinstance(item, str) for item in items):
                return tuple(items)
    raise ValueError(
        f"{PROVENANCE_PATH}@{rev}: SHARED_CORE not found as a module-level tuple-of-string-literals assignment"
    )


# --------------------------------------------------------------------------
# revision-aware hashing (mirrors collector.provenance.collector_hash's
# sorted-(relpath, bytes) construction; reads via `git show` instead of disk)
# --------------------------------------------------------------------------


def _hash_files_at_rev(repo_root: Path, rev: str, relpaths: set[str]) -> str:
    digest = hashlib.sha256()
    for relpath in sorted(relpaths):
        content = _read_git_file(repo_root, rev, relpath)
        digest.update(relpath.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _collector_hash_at_rev(repo_root: Path, rev: str, module: str, closures: dict[str, list[str]]) -> str:
    if module not in closures:
        raise KeyError(f"{module}: no hash_closures.yaml entry at {rev} (fail-closed)")
    module_file = module.replace(".", "/") + ".py"
    shared_core = _shared_core_at_rev(repo_root, rev)
    relpaths = {module_file, *shared_core, *_expand_closure_extras_at_rev(repo_root, rev, closures[module])}
    return _hash_files_at_rev(repo_root, rev, relpaths)


def _case_plan_hash_at_rev(repo_root: Path, rev: str, modules: set[str], closures: dict[str, list[str]]) -> str | None:
    if not modules:
        return None
    relpaths = set(CASE_PLAN_SHARED_FILES)
    for module in modules:
        if module not in closures:
            raise KeyError(f"{module}: no hash_closures.yaml entry at {rev} (fail-closed)")
        relpaths.update(
            extra
            for extra in _expand_closure_extras_at_rev(repo_root, rev, closures[module])
            if extra.startswith("collector/cases/")
        )
    return _hash_files_at_rev(repo_root, rev, relpaths)


# --------------------------------------------------------------------------
# systems holding a family's tables for a framework, at one revision
# --------------------------------------------------------------------------


def _systems_holding(
    repo_root: Path, rev: str, family: str, backend_dir: str | None, table_stems: set[str]
) -> set[str]:
    if not backend_dir or not table_stems:
        return set()
    prefix_depth = len(Path(DATA_PREFIX).parts)
    systems: set[str] = set()
    for entry_path in _git_ls_tree(repo_root, rev, DATA_PREFIX):
        rel_parts = Path(entry_path).parts[prefix_depth:]
        if len(rel_parts) != 5:
            continue
        system, entry_family, entry_backend, _version, filename = rel_parts
        if entry_family == family and entry_backend == backend_dir and Path(filename).stem in table_stems:
            systems.add(system)
    return systems


# --------------------------------------------------------------------------
# per-(framework, family) diff + top-level orchestration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _RevisionContext:
    rev: str
    manifest: dict[str, Any]
    family_map: dict[str, str]
    closures: dict[str, list[str]]


def _load_revision_context(repo_root: Path, rev: str) -> _RevisionContext:
    return _RevisionContext(
        rev=rev,
        manifest=_manifest_at_rev(repo_root, rev),
        family_map=_family_map_at_rev(repo_root, rev),
        closures=_closures_at_rev(repo_root, rev),
    )


@dataclass(frozen=True)
class FamilyDiff:
    framework: str
    family: str
    reasons: tuple[str, ...]
    tables: tuple[str, ...]
    systems: tuple[str, ...]


def _collector_code_differs(
    repo_root: Path,
    base_ctx: _RevisionContext,
    head_ctx: _RevisionContext,
    base_modules: set[str],
    head_modules: set[str],
) -> bool:
    for module in sorted(base_modules | head_modules):
        base_hash = None
        if module in base_modules:
            base_hash = _collector_hash_at_rev(repo_root, base_ctx.rev, module, base_ctx.closures)
        head_hash = None
        if module in head_modules:
            head_hash = _collector_hash_at_rev(repo_root, head_ctx.rev, module, head_ctx.closures)
        if base_hash != head_hash:
            return True
    return False


def _diff_framework(
    repo_root: Path, framework: str, base_ctx: _RevisionContext, head_ctx: _RevisionContext
) -> list[FamilyDiff]:
    base_entries = _parse_registry_entries(repo_root, base_ctx.rev, base_ctx.manifest, framework)
    head_entries = _parse_registry_entries(repo_root, head_ctx.rev, head_ctx.manifest, framework)

    base_modules_by_family = _group_by_family(base_entries, base_ctx.family_map, "module")
    head_modules_by_family = _group_by_family(head_entries, head_ctx.family_map, "module")
    base_tables_by_family = _group_by_family(base_entries, base_ctx.family_map, "table")
    head_tables_by_family = _group_by_family(head_entries, head_ctx.family_map, "table")

    families = sorted(set(base_modules_by_family) | set(head_modules_by_family))
    diffs: list[FamilyDiff] = []
    for family in families:
        base_modules = base_modules_by_family.get(family, set())
        head_modules = head_modules_by_family.get(family, set())

        reasons: list[str] = []
        if _resolve_pin(base_ctx.manifest, framework, family) != _resolve_pin(head_ctx.manifest, framework, family):
            reasons.append(REASON_PIN_VERSION)
        if _collector_code_differs(repo_root, base_ctx, head_ctx, base_modules, head_modules):
            reasons.append(REASON_COLLECTOR_CODE)
        base_case_hash = _case_plan_hash_at_rev(repo_root, base_ctx.rev, base_modules, base_ctx.closures)
        head_case_hash = _case_plan_hash_at_rev(repo_root, head_ctx.rev, head_modules, head_ctx.closures)
        if base_case_hash != head_case_hash:
            reasons.append(REASON_CASE_PLAN)

        tables = sorted(base_tables_by_family.get(family, set()) | head_tables_by_family.get(family, set()))
        backend_dir = _data_backend(base_ctx.manifest, framework)
        systems = sorted(_systems_holding(repo_root, base_ctx.rev, family, backend_dir, set(tables)))

        diffs.append(
            FamilyDiff(
                framework=framework,
                family=family,
                reasons=tuple(reasons),
                tables=tuple(tables),
                systems=tuple(systems),
            )
        )
    return diffs


class PreCollectorV3BaselineError(RuntimeError):
    """`--base` (or `--head`) predates Collector V3 provenance metadata (see
    module docstring "Exit codes") — there is nothing for this tool to diff.
    """


PRE_V3_BASELINE_MESSAGE = (
    "{role} revision predates Collector V3 provenance metadata; "
    "changed-operation manifest cannot be computed against it"
)


def _check_v3_baseline(repo_root: Path, rev: str, *, role: str = "base") -> None:
    if any(not _git_file_exists(repo_root, rev, path) for path in V3_BASELINE_PATHS):
        raise PreCollectorV3BaselineError(PRE_V3_BASELINE_MESSAGE.format(role=role))


def compute_changed_ops(repo_root: Path, base_rev: str, head_rev: str) -> tuple[list[FamilyDiff], list[FamilyDiff]]:
    """Pure function of (repo, base, head): returns (changed, unchanged), both
    sorted deterministically by (framework, family).

    Raises `PreCollectorV3BaselineError` if `base_rev` or `head_rev` predates
    Collector V3 provenance metadata (see module docstring "Exit codes") — the
    CLI (`main`) maps this to exit code 3.
    """
    base_sha = _resolve_rev(repo_root, base_rev)
    head_sha = _resolve_rev(repo_root, head_rev)
    _check_v3_baseline(repo_root, base_sha)
    _check_v3_baseline(repo_root, head_sha, role="head")

    base_ctx = _load_revision_context(repo_root, base_sha)
    head_ctx = _load_revision_context(repo_root, head_sha)
    frameworks = sorted(set(base_ctx.manifest.get("frameworks", {})) | set(head_ctx.manifest.get("frameworks", {})))

    changed: list[FamilyDiff] = []
    unchanged: list[FamilyDiff] = []
    for framework in frameworks:
        for diff in _diff_framework(repo_root, framework, base_ctx, head_ctx):
            (changed if diff.reasons else unchanged).append(diff)

    changed.sort(key=lambda d: (d.framework, d.family))
    unchanged.sort(key=lambda d: (d.framework, d.family))
    return changed, unchanged


# --------------------------------------------------------------------------
# rendering: design §8's locked schema
# --------------------------------------------------------------------------


def _render_changed(diff: FamilyDiff) -> dict[str, Any]:
    return {
        "framework": diff.framework,
        "family": diff.family,
        "reasons": list(diff.reasons),
        "tables": list(diff.tables),
        "systems": list(diff.systems),
        "action": ACTION_RECOLLECT,
    }


def _render_unchanged(diff: FamilyDiff) -> dict[str, Any]:
    return {
        "framework": diff.framework,
        "family": diff.family,
        "tables": list(diff.tables),
        "systems": list(diff.systems),
    }


def render_report(changed: list[FamilyDiff], unchanged: list[FamilyDiff]) -> str:
    doc = {
        "changed": [_render_changed(diff) for diff in changed],
        "unchanged": [_render_unchanged(diff) for diff in unchanged],
    }
    return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

EXIT_OK = 0
EXIT_PRE_V3_BASELINE = 3  # see module docstring "Exit codes"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        changed, unchanged = compute_changed_ops(REPO_ROOT, args.base, args.head)
    except PreCollectorV3BaselineError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_PRE_V3_BASELINE

    report = render_report(changed, unchanged)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
    else:
        sys.stdout.write(report)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())

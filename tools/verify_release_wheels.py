# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the release wheel boundary and ensure every payload has one owner."""

from __future__ import annotations

import argparse
import re
import zipfile
from email import message_from_bytes
from email.message import Message
from pathlib import Path

PAYLOAD_SUFFIXES = {
    ".css",
    ".csv",
    ".j2",
    ".js",
    ".json",
    ".md",
    ".parquet",
    ".py",
    ".pyi",
    ".rule",
    ".txt",
    ".typed",
    ".yaml",
}


def _wheel_files(wheel: Path) -> tuple[set[str], Message]:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_paths = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise RuntimeError(f"{wheel.name}: expected one METADATA file, found {metadata_paths}")
        metadata = message_from_bytes(archive.read(metadata_paths[0]))
    return names, metadata


def _payload_files(names: set[str]) -> set[str]:
    return {name for name in names if ".dist-info/" not in name and not name.endswith("/")}


def _spica_entries(names: set[str]) -> list[str]:
    """Return every stale Spica archive member, regardless of type or suffix."""
    return sorted(name for name in names if name.startswith("spica/"))


def _one_wheel(dist_dir: Path, pattern: str) -> Path:
    matches = sorted(dist_dir.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one wheel matching {pattern!r}, found {[path.name for path in matches]}")
    return matches[0]


def _add_source_tree(expected: set[str], source_root: Path, package_root: str) -> None:
    for path in source_root.rglob("*"):
        if path.is_file() and path.suffix in PAYLOAD_SUFFIXES:
            expected.add((Path(package_root) / path.relative_to(source_root)).as_posix())


def _source_payloads() -> tuple[set[str], set[str]]:
    """Return ``(upper, core)`` payload paths expected from the source tree."""
    repo_root = Path(__file__).resolve().parents[1]
    upper_source = repo_root / "src"
    core_source = repo_root / "aic-core" / "src"
    upper: set[str] = set()
    core: set[str] = set()

    _add_source_tree(upper, upper_source / "aiconfigurator", "aiconfigurator")
    _add_source_tree(core, core_source / "aiconfigurator_core", "aiconfigurator_core")
    return upper, core


def _requirement_name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9_.-]+", requirement)
    if match is None:
        return ""
    return re.sub(r"[-_.]+", "-", match.group(0)).lower()


def _verify_main_wheel(wheel: Path, expected_payload: set[str]) -> tuple[str, set[str]]:
    names, metadata = _wheel_files(wheel)
    payload = _payload_files(names)
    required = {
        "aiconfigurator/__init__.py",
        "aiconfigurator/cli/main.py",
        "aiconfigurator/generator/api.py",
        "aiconfigurator/logging_utils.py",
        "aiconfigurator/sdk/_compat.py",
        "aiconfigurator/sdk/engine.py",
        "aiconfigurator/sdk/task_v2.py",
    }
    missing = sorted(required - payload)
    if missing:
        raise RuntimeError(f"{wheel.name}: missing upper-layer payload: {missing}")

    missing_source = sorted(expected_payload - payload)
    if missing_source:
        raise RuntimeError(f"{wheel.name}: missing upper source-tree payload: {missing_source}")

    misplaced = sorted(name for name in payload if name.startswith("aiconfigurator_core/"))
    if misplaced:
        raise RuntimeError(f"{wheel.name}: upper wheel must not own core payload: {misplaced}")

    # Scan every archive member, not only recognized source payload suffixes: a
    # stale Spica binary, data file, or directory entry must also fail the boundary.
    removed = _spica_entries(names)
    if removed:
        raise RuntimeError(f"{wheel.name}: removed Spica payload is still present: {removed}")

    if "spica" in metadata.get_all("Provides-Extra", []):
        raise RuntimeError(f"{wheel.name}: removed Spica extra is still present in metadata")

    with zipfile.ZipFile(wheel) as archive:
        entry_point_paths = [name for name in names if name.endswith(".dist-info/entry_points.txt")]
        if len(entry_point_paths) != 1:
            raise RuntimeError(f"{wheel.name}: expected one entry_points.txt, found {entry_point_paths}")
        entry_points = archive.read(entry_point_paths[0]).decode()
    if re.search(r"(?m)^spica\s*=", entry_points):
        raise RuntimeError(f"{wheel.name}: removed Spica console script is still present")

    version = metadata.get("Version")
    if not version:
        raise RuntimeError(f"{wheel.name}: missing distribution version")
    expected_requirement = f"aiconfigurator-core=={version}"
    requirements = metadata.get_all("Requires-Dist", [])
    if expected_requirement not in requirements:
        raise RuntimeError(f"{wheel.name}: expected Requires-Dist {expected_requirement!r}, found {requirements}")
    return version, payload


def _verify_core_wheel(wheel: Path, aic_version: str, expected_payload: set[str]) -> set[str]:
    names, metadata = _wheel_files(wheel)
    payload = _payload_files(names)
    required = {
        "aiconfigurator_core/__init__.py",
        "aiconfigurator_core/_aiconfigurator_core.pyi",
        "aiconfigurator_core/model_configs/meta-llama--Meta-Llama-3.1-8B_config.json",
        "aiconfigurator_core/py.typed",
        "aiconfigurator_core/sdk/__init__.py",
        "aiconfigurator_core/sdk/common.py",
        "aiconfigurator_core/sdk/engine.py",
        "aiconfigurator_core/sdk/memory.py",
        "aiconfigurator_core/systems/h100_sxm.yaml",
    }
    missing = sorted(required - payload)
    if missing:
        raise RuntimeError(f"{wheel.name}: missing standalone core payload: {missing}")

    missing_source = sorted(expected_payload - payload)
    if missing_source:
        raise RuntimeError(f"{wheel.name}: missing core source-tree payload: {missing_source}")

    misplaced = sorted(name for name in payload if name.startswith("aiconfigurator/") or name.startswith("spica/"))
    if misplaced:
        raise RuntimeError(f"{wheel.name}: core wheel must not own upper-layer payload: {misplaced}")

    checks = {
        "native core extension": any(
            name.startswith("aiconfigurator_core/_aiconfigurator_core.") and name.endswith((".so", ".pyd"))
            for name in payload
        ),
        "nested performance data": any(
            name.startswith("aiconfigurator_core/systems/data/") and name.endswith(".parquet") for name in payload
        ),
        "Rust SBOM": any(".dist-info/sboms/" in name and name.endswith(".json") for name in names),
    }
    failed = [label for label, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"{wheel.name}: missing {', '.join(failed)}")

    if metadata.get("Version") != aic_version:
        raise RuntimeError(
            f"{wheel.name}: version {metadata.get('Version')!r} does not match aiconfigurator {aic_version!r}"
        )
    requirements = metadata.get_all("Requires-Dist", [])
    if any(_requirement_name(requirement) == "aiconfigurator" for requirement in requirements):
        raise RuntimeError(f"{wheel.name}: standalone core must not depend on aiconfigurator: {requirements}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist_dir", type=Path)
    args = parser.parse_args()

    main_wheel = _one_wheel(args.dist_dir, "aiconfigurator-*.whl")
    core_wheel = _one_wheel(args.dist_dir, "aiconfigurator_core-*.whl")
    expected_main, expected_core = _source_payloads()
    aic_version, main_payload = _verify_main_wheel(main_wheel, expected_main)
    core_payload = _verify_core_wheel(core_wheel, aic_version, expected_core)

    overlap = sorted(main_payload & core_payload)
    if overlap:
        raise RuntimeError(f"release wheels have overlapping payload ownership: {overlap}")

    print(
        f"Verified upper {main_wheel.name} and standalone {core_wheel.name}: "
        f"{len(main_payload)} + {len(core_payload)} disjoint payload files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

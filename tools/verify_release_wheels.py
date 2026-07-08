# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the release wheel boundary and ensure every payload has one owner."""

from __future__ import annotations

import argparse
import re
import stat
import zipfile
from email import message_from_bytes
from email.message import Message
from pathlib import Path

PAYLOAD_SUFFIXES = {".css", ".csv", ".j2", ".js", ".json", ".md", ".parquet", ".py", ".rule", ".txt", ".yaml"}
CORE_AIC_DIRS = {"model_configs", "sdk", "systems"}
SDK_PREFIXES = ("aiconfigurator/sdk/", "aiconfigurator_core/sdk/")


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
    aic_source = repo_root / "src" / "aiconfigurator"
    upper: set[str] = set()
    core: set[str] = set()

    for path in aic_source.rglob("*"):
        if not path.is_file() or path.suffix not in PAYLOAD_SUFFIXES:
            continue
        relative = path.relative_to(aic_source)
        owner = core if relative.parts[0] in CORE_AIC_DIRS else upper
        owner.add((Path("aiconfigurator") / relative).as_posix())

    _add_source_tree(core, repo_root / "src" / "aiconfigurator_core", "aiconfigurator_core")
    _add_source_tree(upper, repo_root / "src" / "spica", "spica")
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
        "spica/config.py",
    }
    missing = sorted(required - payload)
    if missing:
        raise RuntimeError(f"{wheel.name}: missing upper-layer payload: {missing}")

    missing_source = sorted(expected_payload - payload)
    if missing_source:
        raise RuntimeError(f"{wheel.name}: missing upper source-tree payload: {missing_source}")

    misplaced = sorted(
        name
        for name in payload
        if name.startswith(SDK_PREFIXES)
        or name.startswith("aiconfigurator/systems/")
        or name.startswith("aiconfigurator/model_configs/")
        or name.startswith("aiconfigurator_core/")
    )
    if misplaced:
        raise RuntimeError(f"{wheel.name}: upper wheel must not own core payload: {misplaced}")

    version = metadata.get("Version")
    if not version:
        raise RuntimeError(f"{wheel.name}: missing distribution version")
    expected_requirement = f"aiconfigurator-core=={version}"
    requirements = metadata.get_all("Requires-Dist", [])
    if expected_requirement not in requirements:
        raise RuntimeError(f"{wheel.name}: expected Requires-Dist {expected_requirement!r}, found {requirements}")
    return version, payload


def _verify_materialized_sdk_mirrors(wheel: Path) -> None:
    """Require byte-identical, regular-file SDK trees under both import names."""
    prefixes = {
        "aiconfigurator": "aiconfigurator/sdk/",
        "aiconfigurator_core": "aiconfigurator_core/sdk/",
    }
    trees: dict[str, dict[str, bytes]] = {}

    with zipfile.ZipFile(wheel) as archive:
        for label, prefix in prefixes.items():
            tree: dict[str, bytes] = {}
            non_regular: list[str] = []
            for info in archive.infolist():
                if not info.filename.startswith(prefix) or not info.filename.endswith(".py"):
                    continue
                relative = info.filename.removeprefix(prefix)
                mode = (info.external_attr >> 16) & 0xFFFF
                if not stat.S_ISREG(mode):
                    non_regular.append(info.filename)
                tree[relative] = archive.read(info)
            if non_regular:
                raise RuntimeError(
                    f"{wheel.name}: SDK aliases must be materialized as regular files, found {non_regular}"
                )
            if not tree:
                raise RuntimeError(f"{wheel.name}: missing Python SDK payload under {prefix}")
            trees[label] = tree

    canonical_paths = set(trees["aiconfigurator"])
    core_paths = set(trees["aiconfigurator_core"])
    if canonical_paths != core_paths:
        canonical_only = sorted(canonical_paths - core_paths)
        core_only = sorted(core_paths - canonical_paths)
        raise RuntimeError(
            f"{wheel.name}: SDK mirror leaf sets differ: "
            f"aiconfigurator-only={canonical_only}, aiconfigurator_core-only={core_only}"
        )

    content_mismatches = sorted(
        relative
        for relative in canonical_paths
        if trees["aiconfigurator"][relative] != trees["aiconfigurator_core"][relative]
    )
    if content_mismatches:
        raise RuntimeError(f"{wheel.name}: SDK mirror contents differ: {content_mismatches}")


def _verify_core_wheel(wheel: Path, aic_version: str, expected_payload: set[str]) -> set[str]:
    names, metadata = _wheel_files(wheel)
    payload = _payload_files(names)
    required = {
        "aiconfigurator/model_configs/meta-llama--Meta-Llama-3.1-8B_config.json",
        "aiconfigurator/sdk/common.py",
        "aiconfigurator/sdk/engine.py",
        "aiconfigurator/sdk/logging_utils.py",
        "aiconfigurator/sdk/memory.py",
        "aiconfigurator/sdk/task_v2.py",
        "aiconfigurator/systems/h100_sxm.yaml",
        "aiconfigurator_core/__init__.py",
        "aiconfigurator_core/sdk/common.py",
        "aiconfigurator_core/sdk/engine.py",
        "aiconfigurator_core/sdk/logging_utils.py",
        "aiconfigurator_core/sdk/memory.py",
        "aiconfigurator_core/sdk/task_v2.py",
    }
    missing = sorted(required - payload)
    if missing:
        raise RuntimeError(f"{wheel.name}: missing standalone core payload: {missing}")

    missing_source = sorted(expected_payload - payload)
    if missing_source:
        raise RuntimeError(f"{wheel.name}: missing core source-tree payload: {missing_source}")

    _verify_materialized_sdk_mirrors(wheel)

    checks = {
        "native core extension": any(
            name.startswith("aiconfigurator_core/_aiconfigurator_core.") and name.endswith((".so", ".pyd"))
            for name in payload
        ),
        "nested performance data": any(
            name.startswith("aiconfigurator/systems/data/") and name.endswith(".parquet") for name in payload
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

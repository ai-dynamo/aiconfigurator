# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that release wheels have one owner for the complete AIC payload."""

from __future__ import annotations

import argparse
import zipfile
from email import message_from_bytes
from email.message import Message
from pathlib import Path

PAYLOAD_SUFFIXES = {".css", ".csv", ".j2", ".js", ".json", ".md", ".parquet", ".py", ".rule", ".txt", ".yaml"}
SOURCE_PACKAGES = {
    "src/aiconfigurator": "aiconfigurator",
    "src/aiconfigurator_core": "aiconfigurator_core",
    "src/spica": "spica",
}


def _wheel_files(wheel: Path) -> tuple[set[str], Message]:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_paths = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise RuntimeError(f"{wheel.name}: expected one METADATA file, found {metadata_paths}")
        metadata = message_from_bytes(archive.read(metadata_paths[0]))
    return names, metadata


def _one_wheel(dist_dir: Path, pattern: str) -> Path:
    matches = sorted(dist_dir.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one wheel matching {pattern!r}, found {[path.name for path in matches]}")
    return matches[0]


def _source_payload() -> set[str]:
    repo_root = Path(__file__).resolve().parents[1]
    expected = set()
    for source, package in SOURCE_PACKAGES.items():
        source_root = repo_root / source
        for path in source_root.rglob("*"):
            if path.is_file() and path.suffix in PAYLOAD_SUFFIXES:
                expected.add((Path(package) / path.relative_to(source_root)).as_posix())
    return expected


def _verify_main_wheel(wheel: Path) -> str:
    names, metadata = _wheel_files(wheel)
    required = {
        "aiconfigurator/cli/main.py",
        "aiconfigurator/generator/api.py",
        "aiconfigurator/model_configs/meta-llama--Meta-Llama-3.1-8B_config.json",
        "aiconfigurator/sdk/common.py",
        "aiconfigurator/systems/h100_sxm.yaml",
        "aiconfigurator_core/__init__.py",
        "spica/config.py",
    }
    missing = sorted(required - names)
    if missing:
        raise RuntimeError(f"{wheel.name}: missing required payload: {missing}")

    missing_source = sorted(_source_payload() - names)
    if missing_source:
        raise RuntimeError(f"{wheel.name}: missing source-tree payload: {missing_source}")

    checks = {
        "native core extension": any(
            name.startswith("aiconfigurator_core/_aiconfigurator_core.") and name.endswith((".so", ".pyd"))
            for name in names
        ),
        "nested performance data": any(
            name.startswith("aiconfigurator/systems/data/") and name.endswith(".parquet") for name in names
        ),
        "Rust SBOM": any(".dist-info/sboms/" in name and name.endswith(".json") for name in names),
    }
    failed = [label for label, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"{wheel.name}: missing {', '.join(failed)}")

    requirements = metadata.get_all("Requires-Dist", [])
    if any(requirement.lower().startswith("aiconfigurator-core") for requirement in requirements):
        raise RuntimeError(f"{wheel.name}: must not depend on the compatibility metapackage")

    version = metadata.get("Version")
    if not version:
        raise RuntimeError(f"{wheel.name}: missing distribution version")
    return version


def _verify_core_metapackage(wheel: Path, aic_version: str) -> None:
    names, metadata = _wheel_files(wheel)
    payload = sorted(name for name in names if ".dist-info/" not in name)
    if payload:
        raise RuntimeError(f"{wheel.name}: compatibility wheel must not own payload files: {payload}")

    if metadata.get("Version") != aic_version:
        raise RuntimeError(
            f"{wheel.name}: version {metadata.get('Version')!r} does not match aiconfigurator {aic_version!r}"
        )

    requirements = metadata.get_all("Requires-Dist", [])
    expected = f"aiconfigurator=={aic_version}"
    if expected not in requirements:
        raise RuntimeError(f"{wheel.name}: expected Requires-Dist {expected!r}, found {requirements}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist_dir", type=Path)
    args = parser.parse_args()

    main_wheel = _one_wheel(args.dist_dir, "aiconfigurator-*.whl")
    core_wheel = _one_wheel(args.dist_dir, "aiconfigurator_core-*.whl")
    aic_version = _verify_main_wheel(main_wheel)
    _verify_core_metapackage(core_wheel, aic_version)
    print(f"Verified self-contained {main_wheel.name} and payload-free {core_wheel.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

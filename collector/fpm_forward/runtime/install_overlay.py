# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install a hash-pinned Dynamo vLLM runtime overlay into an ephemeral Pod."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import py_compile
import shutil
from pathlib import Path

WORKDIR = Path("/tmp/fpm-bench")
MANIFEST = WORKDIR / "runtime-overlay-manifest.json"
RESULTS_DIR = Path("/results")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if not MANIFEST.is_file():
        return
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("invalid FPM runtime overlay manifest")
    spec = importlib.util.find_spec("dynamo.vllm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("cannot locate installed dynamo.vllm package")
    package_dir = Path(next(iter(spec.submodule_search_locations))).resolve()
    backup_dir = RESULTS_DIR / "original-runtime-files"
    backup_dir.mkdir(parents=True, exist_ok=True)
    audit = {"schema_version": 1, "package_dir": str(package_dir), "files": {}}

    for name, metadata in sorted(manifest["files"].items()):
        if Path(name).name != name or not name.endswith(".py"):
            raise ValueError(f"invalid runtime overlay filename: {name!r}")
        source = WORKDIR / str(metadata["source_name"])
        expected = str(metadata["sha256"])
        if not source.is_file() or _sha256(source) != expected:
            raise ValueError(f"runtime overlay hash mismatch: {source}")
        target = package_dir / name
        if not target.is_file():
            raise FileNotFoundError(f"runtime overlay target does not exist: {target}")
        current_sha256 = _sha256(target)
        expected_original_sha256 = str(metadata["original_sha256"])
        if current_sha256 == expected:
            # A formal collection runs its one-point capability gate and the
            # complete case list in the same ephemeral Pod.  The second run
            # therefore sees the already-installed, hash-pinned overlay.  It
            # is safe to reuse only when the target still exactly matches the
            # frozen overlay; every other non-base hash remains an error.
            py_compile.compile(str(target), doraise=True)
            audit["files"][name] = {
                "original_sha256": expected_original_sha256,
                "overlay_sha256": expected,
                "status": "already_installed",
            }
            continue
        if current_sha256 != expected_original_sha256:
            raise RuntimeError(
                f"runtime overlay base hash mismatch for {target}: "
                f"expected={expected_original_sha256}, actual={current_sha256}"
            )
        shutil.copy2(target, backup_dir / name)
        temporary = target.with_name(f".{target.name}.fpm.tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
        if _sha256(target) != expected:
            raise RuntimeError(f"installed runtime overlay hash mismatch: {target}")
        py_compile.compile(str(target), doraise=True)
        audit["files"][name] = {
            "original_sha256": current_sha256,
            "overlay_sha256": expected,
            "status": "installed",
        }

    (RESULTS_DIR / "runtime-overlay-audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

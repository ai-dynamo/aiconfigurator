# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the symlinked core SDK namespace."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SDK_ROOT = REPO_ROOT / "src" / "aiconfigurator" / "sdk"
CORE_SDK_LINK = REPO_ROOT / "src" / "aiconfigurator_core" / "sdk"


def _discover_sdk_leaves() -> tuple[str, ...]:
    """Return every non-package Python module below the physical SDK tree."""
    return tuple(
        sorted(
            ".".join(path.relative_to(SDK_ROOT).with_suffix("").parts)
            for path in SDK_ROOT.rglob("*.py")
            if path.name != "__init__.py"
        )
    )


def test_core_sdk_is_relative_symlink_to_aiconfigurator_sdk() -> None:
    assert CORE_SDK_LINK.is_symlink()

    link_target = Path(os.readlink(CORE_SDK_LINK))
    assert not link_target.is_absolute()

    resolved_repo = REPO_ROOT.resolve(strict=True)
    resolved_sdk = CORE_SDK_LINK.resolve(strict=True)
    assert resolved_sdk == SDK_ROOT.resolve(strict=True)
    assert resolved_sdk.is_relative_to(resolved_repo)


@pytest.mark.parametrize("namespace", ["aiconfigurator.sdk", "aiconfigurator_core.sdk"])
def test_every_sdk_leaf_is_importable_from_both_namespaces(namespace: str) -> None:
    """Import every leaf in a subprocess so import-time state stays isolated."""
    leaves = _discover_sdk_leaves()
    assert leaves

    script = f"""
import importlib

failures = []
for leaf in {json.dumps(leaves)}:
    module_name = f"{namespace}.{{leaf}}"
    try:
        importlib.import_module(module_name)
    except Exception as error:
        failures.append(f"{{module_name}}: {{type(error).__name__}}: {{error}}")

if failures:
    print("\\n".join(failures))
    raise SystemExit(1)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


def test_symlinked_leaf_imports_are_separate_module_objects() -> None:
    """A filesystem symlink shares source bytes, not Python module identity."""
    aic_errors = importlib.import_module("aiconfigurator.sdk.errors")
    core_errors = importlib.import_module("aiconfigurator_core.sdk.errors")

    aic_path = Path(aic_errors.__file__)
    core_path = Path(core_errors.__file__)
    source_path = SDK_ROOT / "errors.py"

    assert aic_errors is not core_errors
    assert aic_path.read_bytes() == core_path.read_bytes() == source_path.read_bytes()

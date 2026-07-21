# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layering checks for the standalone ``aiconfigurator_core`` SDK."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SDK_ROOT = Path(__file__).parents[3] / "aic-core" / "src" / "aiconfigurator_core" / "sdk"
UPPER_MODULES = {"aiconfigurator"}


def _is_cli_child(name: str) -> bool:
    return name == "cli" or name.startswith("cli.")


def _is_absolute_cli_import(name: str) -> bool:
    return name == "aiconfigurator.cli" or name.startswith("aiconfigurator.cli.")


def _sdk_package_parts(path: Path) -> tuple[str, ...]:
    return ("aiconfigurator_core", "sdk", *path.parent.parts)


def _resolve_import_from_module(node: ast.ImportFrom, package_parts: tuple[str, ...]) -> str | None:
    module = node.module or ""
    if node.level == 0:
        return module

    if node.level > len(package_parts):
        return None

    base_parts = package_parts[: len(package_parts) - node.level + 1]
    if module:
        return ".".join((*base_parts, *module.split(".")))
    return ".".join(base_parts)


def _is_upper_module(name: str) -> bool:
    return any(name == upper or name.startswith(f"{upper}.") for upper in UPPER_MODULES)


def _is_upper_import_from(node: ast.ImportFrom, package_parts: tuple[str, ...]) -> bool:
    resolved_module = _resolve_import_from_module(node, package_parts)
    if resolved_module is None:
        return False
    return _is_upper_module(resolved_module)


def _upper_import_offenders(path: Path, source: str, root: Path | None = None) -> list[str]:
    offenders: list[str] = []
    display_path = path.relative_to(root) if root is not None else path
    package_parts = _sdk_package_parts(display_path)
    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(_is_upper_module(alias.name) for alias in node.names):
                offenders.append(f"{display_path}:{node.lineno}")
        elif isinstance(node, ast.ImportFrom) and _is_upper_import_from(node, package_parts):
            offenders.append(f"{display_path}:{node.lineno}")
    return offenders


def _is_cli_import_from(node: ast.ImportFrom, package_parts: tuple[str, ...]) -> bool:
    resolved_module = _resolve_import_from_module(node, package_parts)
    if resolved_module is None:
        return False
    if _is_absolute_cli_import(resolved_module):
        return True
    if resolved_module == "aiconfigurator":
        return any(_is_cli_child(alias.name) for alias in node.names)
    return False


def _cli_import_offenders(path: Path, source: str, root: Path | None = None) -> list[str]:
    offenders: list[str] = []
    display_path = path.relative_to(root) if root is not None else path
    package_parts = _sdk_package_parts(display_path)
    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_absolute_cli_import(alias.name):
                    offenders.append(f"{display_path}:{node.lineno}")
        elif isinstance(node, ast.ImportFrom) and _is_cli_import_from(node, package_parts):
            offenders.append(f"{display_path}:{node.lineno}")
    return offenders


def test_sdk_modules_do_not_import_cli_layer() -> None:
    offenders: list[str] = []
    for path in sorted(SDK_ROOT.rglob("*.py")):
        offenders.extend(_cli_import_offenders(path, path.read_text(encoding="utf-8"), SDK_ROOT))

    assert offenders == []


def test_sdk_modules_do_not_import_upper_layer() -> None:
    offenders: list[str] = []
    for path in sorted(SDK_ROOT.rglob("*.py")):
        offenders.extend(_upper_import_offenders(path, path.read_text(encoding="utf-8"), SDK_ROOT))

    assert offenders == []


@pytest.mark.parametrize(
    ("path", "source"),
    [
        (Path("memory.py"), "import aiconfigurator\n"),
        (Path("memory.py"), "import aiconfigurator.generator\n"),
        (Path("memory.py"), "from aiconfigurator.logging_utils import setup_logging\n"),
        (Path("memory.py"), "from aiconfigurator import webapp\n"),
        (Path("memory.py"), "from aiconfigurator import main\n"),
        (Path("subpkg/module.py"), "from aiconfigurator.generator import api\n"),
    ],
)
def test_upper_import_offenders_flags_upper_packages(path: Path, source: str) -> None:
    assert _upper_import_offenders(path, source) == [f"{path}:1"]


@pytest.mark.parametrize(
    ("path", "source"),
    [
        (Path("memory.py"), "import aiconfigurator.cli\n"),
        (Path("memory.py"), "import aiconfigurator.cli.api\n"),
        (Path("memory.py"), "from aiconfigurator import cli\n"),
        (Path("memory.py"), "from aiconfigurator.cli import api\n"),
        (Path("memory.py"), "from aiconfigurator.cli.api import cli_estimate\n"),
    ],
)
def test_cli_import_offenders_flags_absolute_and_relative_cli_imports(path: Path, source: str) -> None:
    assert _cli_import_offenders(path, source) == [f"{path}:1"]


@pytest.mark.parametrize(
    ("path", "source"),
    [
        (Path("memory.py"), "import aiconfigurator_core.sdk.memory\n"),
        (Path("memory.py"), "from aiconfigurator_core import sdk\n"),
        (Path("memory.py"), "from . import cli\n"),
        (Path("subpkg/module.py"), "from .. import cli\n"),
        (Path("subpkg/module.py"), "from ..cli import api\n"),
    ],
)
def test_cli_import_offenders_ignores_non_cli_layer_imports(path: Path, source: str) -> None:
    assert _cli_import_offenders(path, source) == []

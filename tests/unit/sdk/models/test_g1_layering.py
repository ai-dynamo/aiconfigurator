# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""G1 goal verification: large-EP wiring lives in ``models/blocks/`` only.

The redesign's first goal (spec section 4.1) has two enforceable halves:

- LAYERING: the MoE block builder (``models/blocks/moe.py``) is the ONE place
  that emits the large-EP ops. No module under ``models/`` outside ``blocks/``
  may import ``operations.moe_comm`` or construct ``MoEAllToAll`` / ``MoEExpertCompute``
  directly -- model classes reach the large-EP emission only through
  ``build_moe_block_ops`` (or a ``register_moe_block`` family variant), so the
  emission stays swappable per (family, framework, system) without touching
  model files. AST scan in the ``tests/unit/sdk/test_package_layering.py``
  style, with the scanner itself pinned by synthetic offender/ignorer cases.

- DELETION: the five per-regime wideEP model classes
  (``SGLangEPMOEModel``, ``WideEPDeepSeekModel``, ``TrtllmWideEPDeepSeekModel``,
  ``WideEPDeepSeekV32Model``, ``TrtllmWideEPDeepSeekV32Model``) are gone: not
  importable from ``aiconfigurator_core.sdk.models`` nor from the legacy
  ``aiconfigurator.sdk.models`` alias, absent from their old defining
  submodules, and listed in no ``__all__``.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

MODELS_ROOT = Path(__file__).parents[4] / "aic-core" / "src" / "aiconfigurator_core" / "sdk" / "models"

#: Package parts used to resolve relative imports inside models/<module>.py.
_MODELS_PACKAGE_PARTS = ("aiconfigurator_core", "sdk", "models")

LARGE_EP_OP_NAMES = frozenset({"MoEAllToAll", "MoEExpertCompute"})

DELETED_WIDEEP_CLASSES = (
    "SGLangEPMOEModel",
    "WideEPDeepSeekModel",
    "TrtllmWideEPDeepSeekModel",
    "WideEPDeepSeekV32Model",
    "TrtllmWideEPDeepSeekV32Model",
)

MODEL_PACKAGES = ("aiconfigurator_core.sdk.models", "aiconfigurator.sdk.models")

#: The submodules the deleted classes used to live in.
FORMER_DEFINING_SUBMODULES = ("moe", "deepseek", "deepseek_v32")


# ---------------------------------------------------------------------------
# AST scanner (pattern from tests/unit/sdk/test_package_layering.py)
# ---------------------------------------------------------------------------


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


def _is_moe_comm_module(name: str) -> bool:
    return name == "operations.moe_comm" or name.endswith(".operations.moe_comm")


def _is_operations_module(name: str) -> bool:
    return name == "operations" or name.endswith(".operations")


def _module_package_parts(rel_path: Path) -> tuple[str, ...]:
    return (*_MODELS_PACKAGE_PARTS, *rel_path.parent.parts)


def _moe_comm_import_offenders(rel_path: Path, source: str) -> list[str]:
    """``operations.moe_comm`` imports, in any spelling (absolute / relative /
    ``from ...operations import moe_comm``)."""
    offenders: list[str] = []
    package_parts = _module_package_parts(rel_path)
    tree = ast.parse(source, filename=str(rel_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(_is_moe_comm_module(alias.name) for alias in node.names):
                offenders.append(f"{rel_path}:{node.lineno}")
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve_import_from_module(node, package_parts)
            if resolved is None:
                continue
            if _is_moe_comm_module(resolved) or (
                _is_operations_module(resolved) and any(alias.name == "moe_comm" for alias in node.names)
            ):
                offenders.append(f"{rel_path}:{node.lineno}")
    return offenders


def _large_ep_constructor_offenders(rel_path: Path, source: str) -> list[str]:
    """Call nodes whose callee names ``MoEAllToAll`` / ``MoEExpertCompute`` -- bare
    (``MoEAllToAll(...)``) or attribute (``ops.MoEAllToAll(...)``) form."""
    offenders: list[str] = []
    tree = ast.parse(source, filename=str(rel_path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name in LARGE_EP_OP_NAMES:
            offenders.append(f"{rel_path}:{node.lineno}")
    return offenders


def _models_modules_outside_blocks():
    for path in sorted(MODELS_ROOT.rglob("*.py")):
        rel = path.relative_to(MODELS_ROOT)
        if rel.parts[0] == "blocks":
            continue
        yield rel, path.read_text(encoding="utf-8")


def test_models_root_exists_and_scan_is_not_vacuous() -> None:
    modules = list(_models_modules_outside_blocks())
    scanned = {str(rel) for rel, _ in modules}
    # The family files the goal is about must actually be in the scan set.
    assert {"moe.py", "deepseek.py", "deepseek_v32.py", "__init__.py"} <= scanned
    assert not any(name.startswith("blocks") for name in scanned)


def test_no_models_module_outside_blocks_imports_moe_comm() -> None:
    offenders: list[str] = []
    for rel, source in _models_modules_outside_blocks():
        offenders.extend(_moe_comm_import_offenders(rel, source))
    assert offenders == []


def test_no_models_module_outside_blocks_constructs_large_ep_ops() -> None:
    offenders: list[str] = []
    for rel, source in _models_modules_outside_blocks():
        offenders.extend(_large_ep_constructor_offenders(rel, source))
    assert offenders == []


def test_blocks_moe_is_the_large_ep_emission_site() -> None:
    """The inverse guard: the builder module itself DOES import moe_comm and
    construct both ops -- if the emission moved elsewhere, the scans above
    would go vacuous without this test noticing."""
    source = (MODELS_ROOT / "blocks" / "moe.py").read_text(encoding="utf-8")
    rel = Path("blocks/moe.py")
    tree = ast.parse(source)
    imports_moe_comm = any(
        isinstance(node, ast.ImportFrom)
        and _is_moe_comm_module(_resolve_import_from_module(node, _module_package_parts(rel)) or "")
        for node in ast.walk(tree)
    )
    assert imports_moe_comm
    constructed = {offender.split(":")[0] for offender in _large_ep_constructor_offenders(rel, source)}
    assert constructed == {"blocks/moe.py"}
    assert len(_large_ep_constructor_offenders(rel, source)) >= 2  # MoEAllToAll + MoEExpertCompute


# --- scanner self-checks (synthetic sources; mirrors the layering-test style) ---


@pytest.mark.parametrize(
    ("path", "source"),
    [
        (Path("memory.py"), "import aiconfigurator_core.sdk.operations.moe_comm\n"),
        (Path("memory.py"), "from aiconfigurator_core.sdk.operations.moe_comm import MoEAllToAll\n"),
        (Path("memory.py"), "from aiconfigurator_core.sdk.operations import moe_comm\n"),
        (Path("memory.py"), "from .operations.moe_comm import nodes_for\n"),
        (Path("memory.py"), "from ..operations.moe_comm import MoEExpertCompute\n"),
        (Path("memory.py"), "from ..operations import moe_comm\n"),
    ],
)
def test_moe_comm_import_scanner_flags_all_spellings(path: Path, source: str) -> None:
    assert _moe_comm_import_offenders(path, source) == [f"{path}:1"]


@pytest.mark.parametrize(
    ("path", "source"),
    [
        (Path("memory.py"), "import aiconfigurator_core.sdk.operations as ops\n"),
        (Path("memory.py"), "from aiconfigurator_core.sdk.operations import base\n"),
        (Path("memory.py"), "x = 'operations.moe_comm'\n"),
        (Path("memory.py"), "# operations.moe_comm is documented here\npass\n"),
    ],
)
def test_moe_comm_import_scanner_ignores_non_offenders(path: Path, source: str) -> None:
    assert _moe_comm_import_offenders(path, source) == []


@pytest.mark.parametrize(
    ("path", "source"),
    [
        (Path("memory.py"), "op = MoEAllToAll('n', 1.0)\n"),
        (Path("memory.py"), "op = MoEExpertCompute('n', 1.0)\n"),
        (Path("memory.py"), "op = ops.MoEAllToAll('n', 1.0)\n"),
        (Path("memory.py"), "op = operations.moe_comm.MoEExpertCompute('n', 1.0)\n"),
    ],
)
def test_constructor_scanner_flags_call_forms(path: Path, source: str) -> None:
    assert _large_ep_constructor_offenders(path, source) == [f"{path}:1"]


@pytest.mark.parametrize(
    ("path", "source"),
    [
        (Path("memory.py"), "x = 'MoEAllToAll'\n"),
        (Path("memory.py"), "# MoEExpertCompute mentioned in a comment\npass\n"),
        (Path("memory.py"), "ok = isinstance(op, ops.MoEAllToAll)\n"),  # reference, not construction
        (Path("memory.py"), "op = MoEDispatch('n', 1.0)\n"),
    ],
)
def test_constructor_scanner_ignores_non_constructions(path: Path, source: str) -> None:
    assert _large_ep_constructor_offenders(path, source) == []


# ---------------------------------------------------------------------------
# Deletion: the five wideEP classes are gone from both namespaces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("package", MODEL_PACKAGES)
@pytest.mark.parametrize("name", DELETED_WIDEEP_CLASSES)
def test_deleted_wideep_class_is_unimportable(package: str, name: str) -> None:
    mod = importlib.import_module(package)
    with pytest.raises(AttributeError):
        getattr(mod, name)
    # The from-import spelling of the same access raises ImportError.
    with pytest.raises(ImportError):
        exec(f"from {package} import {name}", {})  # fixed template over parametrized names
    assert name not in getattr(mod, "__all__", ())


@pytest.mark.parametrize("package", MODEL_PACKAGES)
@pytest.mark.parametrize("submodule", FORMER_DEFINING_SUBMODULES)
def test_deleted_wideep_classes_absent_from_former_defining_modules(package: str, submodule: str) -> None:
    mod = importlib.import_module(f"{package}.{submodule}")
    present = [name for name in DELETED_WIDEEP_CLASSES if hasattr(mod, name)]
    assert present == []
    assert not set(DELETED_WIDEEP_CLASSES) & set(getattr(mod, "__all__", ()))


def test_surviving_family_classes_still_export() -> None:
    """Deletion control: the surviving classes still import from both
    namespaces (the alias delegates, so a broken facade would fail here, not
    silently pass the deletion checks above)."""
    for package in MODEL_PACKAGES:
        mod = importlib.import_module(package)
        for name in ("MOEModel", "DeepSeekModel", "DeepSeekV32Model"):
            assert getattr(mod, name) is not None
            assert name in mod.__all__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

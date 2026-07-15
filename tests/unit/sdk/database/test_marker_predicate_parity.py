# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-copy consistency for the "is this version dir partial/incomplete"
predicate (Collector V3 design §5/§6.3: ``collection_meta.yaml`` status:partial,
``INCOMPLETE.txt`` as the legacy fallback).

This predicate exists in FOUR independent copies (AIC-1502 PR3 Task 2 review):

- canonical: ``aiconfigurator_core.sdk.perf_database._version_dir_state()["partial"]``
- lenient:   ``aiconfigurator_core.sdk.operations.base._version_dir_is_partial()``
  (deliberately lenient hot-path duplicate — see below)
- tool:      ``tools.prediction_regression_gate.grid._dir_is_incomplete()``
- tool:      ``tools.sanity_check.create_charts._dir_is_incomplete()``

For four of the five fixtures below all four copies must agree exactly. The
fifth fixture (malformed ``collection_meta.yaml``) is the one DOCUMENTED,
DELIBERATE divergence: the canonical loader and both tool copies raise
``ValueError`` naming the file (fail loudly — this is authored data, a bad
parse is almost always a typo worth surfacing), while ``operations/base.py``'s
copy degrades to "not partial" (lenient — it runs on every op query's hot
path, not at a validation entry point, so raising there would turn one bad
sidecar file into a mass query failure). There is no automated drift check
between these copies beyond this test; a future schema change to the
partial-detection rule needs a manual sweep of all four sites.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from aiconfigurator.sdk.operations.base import _version_dir_is_partial
from aiconfigurator.sdk.perf_database import _version_dir_state
from tools.prediction_regression_gate import grid as _grid

pytestmark = pytest.mark.unit

CREATE_CHARTS_PATH = Path(__file__).resolve().parents[4] / "tools" / "sanity_check" / "create_charts.py"


@pytest.fixture
def _create_charts_module(monkeypatch):
    """Import create_charts.py without its heavy validate_database.ipynb
    dependency (see tests/unit/tools/test_create_charts_paths.py, which uses
    the same stub-and-exec pattern)."""
    monkeypatch.setitem(sys.modules, "import_ipynb", types.ModuleType("import_ipynb"))
    monkeypatch.setitem(sys.modules, "validate_database", types.ModuleType("validate_database"))

    spec = importlib.util.spec_from_file_location("create_charts_under_test", CREATE_CHARTS_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def predicates(_create_charts_module):
    """The four "is this dir partial/incomplete" predicate copies, normalized
    to a single ``(version_dir_path: str) -> bool`` signature."""
    return {
        "canonical": lambda path: _version_dir_state(path)["partial"],
        "base_lenient": _version_dir_is_partial,
        "grid_tool": _grid._dir_is_incomplete,
        "create_charts_tool": _create_charts_module._dir_is_incomplete,
    }


def _write(path: Path, name: str, content: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_text(content, encoding="utf-8")


# --- Fixture matrix: cases where all four copies must agree -----------------


def _make_valid_partial(version_dir: Path) -> None:
    _write(version_dir, "collection_meta.yaml", "tables:\n  gemm_perf:\n    status: partial\n")


def _make_valid_clean(version_dir: Path) -> None:
    _write(version_dir, "collection_meta.yaml", "tables:\n  gemm_perf:\n    status: complete\n")


def _make_legacy_incomplete_txt(version_dir: Path) -> None:
    # No collection_meta.yaml at all; only the legacy marker.
    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / "INCOMPLETE.txt").write_text("", encoding="utf-8")


def _make_missing_everything(version_dir: Path) -> None:
    # Neither marker present.
    version_dir.mkdir(parents=True, exist_ok=True)


AGREED_CASES = {
    "valid_partial": (_make_valid_partial, True),
    "valid_clean": (_make_valid_clean, False),
    "missing_collection_meta_legacy_incomplete_txt": (_make_legacy_incomplete_txt, True),
    "missing_everything": (_make_missing_everything, False),
}


@pytest.mark.parametrize("case_name", sorted(AGREED_CASES))
def test_all_four_copies_agree(case_name, predicates, tmp_path):
    setup, expected = AGREED_CASES[case_name]
    version_dir = tmp_path / "h200_sxm" / "gemm" / "sglang" / "0.5.12"
    setup(version_dir)

    for copy_name, predicate in predicates.items():
        assert predicate(str(version_dir)) is expected, f"{copy_name} disagreed for case {case_name!r}"


# --- Fixture: malformed collection_meta.yaml — the one documented divergence -


def _make_malformed_yaml(version_dir: Path) -> None:
    # Unterminated flow mapping: not parseable YAML.
    _write(version_dir, "collection_meta.yaml", "tables: [unterminated\n")


@pytest.mark.parametrize("copy_name", ["canonical", "grid_tool", "create_charts_tool"])
def test_malformed_collection_meta_raises_for_canonical_and_tools(copy_name, predicates, tmp_path):
    version_dir = tmp_path / "h200_sxm" / "gemm" / "sglang" / "0.5.12"
    _make_malformed_yaml(version_dir)

    with pytest.raises(ValueError, match=r"collection_meta\.yaml"):
        predicates[copy_name](str(version_dir))


def test_malformed_collection_meta_is_lenient_false_for_base_operations(predicates, tmp_path):
    """operations/base.py's ``_version_dir_is_partial`` is a hot per-op
    path-resolution predicate, not a validation entry point — it deliberately
    swallows a malformed collection_meta.yaml and reports "not partial"
    instead of raising, so one bad sidecar file doesn't take down every query
    against that data tree. This is the one documented divergence from the
    canonical/tool copies (see module docstring)."""
    version_dir = tmp_path / "h200_sxm" / "gemm" / "sglang" / "0.5.12"
    _make_malformed_yaml(version_dir)

    assert predicates["base_lenient"](str(version_dir)) is False

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tools/sanity_check/create_charts.py's perf-file-path aggregation
across the legacy (<system>/<backend>/<version>) and family-first
(<system>/<family>/<backend>/<version>) tree layouts.

One (backend, version) can legitimately split its perf files across several
family dirs (e.g. gemm files under <system>/gemm/<backend>/<version>/ and
attention files under <system>/attention/<backend>/<version>/). `_perf_file_paths`
must return the union of all of them rather than picking one winning directory
(unlike `_data_dir`, which still picks a single dir for single-file lookups).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

CREATE_CHARTS_PATH = Path(__file__).resolve().parents[3] / "tools" / "sanity_check" / "create_charts.py"


@pytest.fixture
def create_charts_module(monkeypatch):
    """Import create_charts.py without pulling in its real validate_database.ipynb
    dependency, which unconditionally loads a live perf database at import time
    (see tests/e2e/tools/test_sanity_check.py, which runs that import in a
    subprocess with a 300s timeout). Stub the two heavy notebook imports so the
    rest of the (otherwise lightweight) module loads normally; nothing under test
    here touches `validate_database`.
    """
    monkeypatch.setitem(sys.modules, "import_ipynb", types.ModuleType("import_ipynb"))
    monkeypatch.setitem(sys.modules, "validate_database", types.ModuleType("validate_database"))

    spec = importlib.util.spec_from_file_location("create_charts_under_test", CREATE_CHARTS_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _touch(root: Path, rel: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"stub")


def test_perf_file_paths_aggregates_split_family_dirs(create_charts_module, tmp_path):
    """gemm and attention perf files live under sibling family dirs for the same
    (backend, version); _perf_file_paths must return all three, not just whichever
    family dir happens to have the most files."""
    _touch(tmp_path, "h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
    _touch(tmp_path, "h200_sxm/attention/sglang/0.5.14/context_attention_perf.parquet")
    _touch(tmp_path, "h200_sxm/attention/sglang/0.5.14/generation_attention_perf.parquet")

    paths = create_charts_module._perf_file_paths(str(tmp_path), "h200_sxm", "sglang", "0.5.14")

    assert set(paths) == {
        "gemm_perf.parquet",
        "context_attention_perf.parquet",
        "generation_attention_perf.parquet",
    }
    assert paths["gemm_perf.parquet"] == tmp_path / "h200_sxm" / "gemm" / "sglang" / "0.5.14" / "gemm_perf.parquet"
    assert (
        paths["context_attention_perf.parquet"]
        == tmp_path / "h200_sxm" / "attention" / "sglang" / "0.5.14" / "context_attention_perf.parquet"
    )
    assert (
        paths["generation_attention_perf.parquet"]
        == tmp_path / "h200_sxm" / "attention" / "sglang" / "0.5.14" / "generation_attention_perf.parquet"
    )


def test_perf_file_paths_legacy_only_tree(create_charts_module, tmp_path):
    """A tree with no family dirs at all (pure legacy layout) still resolves."""
    _touch(tmp_path, "h200_sxm/sglang/0.5.12/gemm_perf.parquet")

    paths = create_charts_module._perf_file_paths(str(tmp_path), "h200_sxm", "sglang", "0.5.12")

    assert set(paths) == {"gemm_perf.parquet"}
    assert paths["gemm_perf.parquet"] == tmp_path / "h200_sxm" / "sglang" / "0.5.12" / "gemm_perf.parquet"


def test_perf_file_paths_prefers_legacy_on_duplicate(create_charts_module, tmp_path):
    """During the migration window the same basename may exist in both the legacy
    dir and a family dir; the legacy copy must win deterministically."""
    _touch(tmp_path, "h200_sxm/sglang/0.5.12/gemm_perf.parquet")
    _touch(tmp_path, "h200_sxm/gemm/sglang/0.5.12/gemm_perf.parquet")

    paths = create_charts_module._perf_file_paths(str(tmp_path), "h200_sxm", "sglang", "0.5.12")

    assert set(paths) == {"gemm_perf.parquet"}
    assert paths["gemm_perf.parquet"] == tmp_path / "h200_sxm" / "sglang" / "0.5.12" / "gemm_perf.parquet"


def test_should_run_cli_smoke_test_passes_on_split_tree(create_charts_module, tmp_path, monkeypatch):
    """should_run_cli_smoke_test must not falsely report required files missing
    when they legitimately live in sibling family dirs."""
    _touch(tmp_path, "h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
    _touch(tmp_path, "h200_sxm/attention/sglang/0.5.14/context_attention_perf.parquet")
    _touch(tmp_path, "h200_sxm/attention/sglang/0.5.14/generation_attention_perf.parquet")

    monkeypatch.setattr(create_charts_module, "_systems_data_root", lambda: str(tmp_path))

    run_smoke, reason = create_charts_module.should_run_cli_smoke_test("h200_sxm", "sglang", "0.5.14")

    assert run_smoke, reason

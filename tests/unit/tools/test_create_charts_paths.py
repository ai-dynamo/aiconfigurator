# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tools/sanity_check/create_charts.py's perf-file-path aggregation
across the legacy (<system>/<backend>/<version>) and family-first
(<system>/<family>/<backend>/<version>) tree layouts.

One (backend, version) can legitimately split its perf files across several
family dirs (e.g. gemm files under <system>/gemm/<backend>/<version>/ and
attention files under <system>/attention/<backend>/<version>/). `_perf_file_paths`
must return the union of all of them rather than picking one winning directory.

Also covers main()'s changed-file parser: a 5-part (legacy-shaped) changed path
must validate its INCOMPLETE.txt marker against the exact legacy dir, never a
family dir of the same (backend, version).
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


# --- main()'s changed-file parser: family-first (6-part) paths --------------
#
# main() groups PR-changed files by (system, backend, version) before calling
# create_charts(). The legacy branch only recognizes 5-part
# data/<system>/<backend>/<version>/<file> paths; family-first layout paths
# are one segment longer (data/<system>/<family>/<backend>/<version>/<file>).
# These tests drive main() end-to-end (mocking out git plumbing and the real
# create_charts()) to confirm the 6-part branch is wired up.


def _run_main_with_changed_files(create_charts_module, tmp_path, monkeypatch, changed_files):
    monkeypatch.setattr(create_charts_module, "get_changed_files", lambda base, head: changed_files)
    monkeypatch.setattr(create_charts_module, "get_csv_to_parquet_conversion_files", lambda base, head: set())
    monkeypatch.setattr(create_charts_module, "_systems_data_root", lambda: str(tmp_path))

    calls = []

    def fake_create_charts(backend, backend_version, system, perf_files, output_dir, output_md_file):
        calls.append((system, backend, backend_version, sorted(perf_files)))

    monkeypatch.setattr(create_charts_module, "create_charts", fake_create_charts)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_charts.py",
            "--output-dir",
            str(tmp_path / "charts_output"),
            "--output-md-file",
            str(tmp_path / "comment.md"),
        ],
    )

    create_charts_module.main()
    return calls


def test_main_parses_family_layout_changed_file(create_charts_module, tmp_path, monkeypatch):
    """A 6-part data/<system>/<family>/<backend>/<version>/<file> changed path is
    parsed and grouped exactly like the 5-part legacy branch."""
    changed_file = f"{create_charts_module.SYSTEMS_PREFIX}data/h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet"

    calls = _run_main_with_changed_files(create_charts_module, tmp_path, monkeypatch, [changed_file])

    assert calls == [("h200_sxm", "sglang", "0.5.14", ["gemm_perf.parquet"])]


def test_main_skips_nccl_family_layout_changed_file(create_charts_module, tmp_path, monkeypatch):
    """The 6-part branch mirrors the legacy branch's nccl/oneccl pseudo-backend
    skip: nccl/oneccl perf files are ignored, matching how the 5-part branch
    treats data/<system>/nccl/<version>/nccl_perf.parquet."""
    changed_file = f"{create_charts_module.SYSTEMS_PREFIX}data/h200_sxm/comm/nccl/2.20/nccl_perf.parquet"

    calls = _run_main_with_changed_files(create_charts_module, tmp_path, monkeypatch, [changed_file])

    assert calls == []


def test_main_suppresses_family_layout_incomplete_dir(create_charts_module, tmp_path, monkeypatch):
    """An INCOMPLETE.txt marker in the FAMILY dir the changed file lives in
    (data_root/system/family/backend/version) suppresses that entry, same as
    the legacy branch's INCOMPLETE.txt check."""
    changed_file = f"{create_charts_module.SYSTEMS_PREFIX}data/h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet"
    incomplete_dir = tmp_path / "h200_sxm" / "gemm" / "sglang" / "0.5.14"
    incomplete_dir.mkdir(parents=True)
    (incomplete_dir / "INCOMPLETE.txt").write_bytes(b"")

    calls = _run_main_with_changed_files(create_charts_module, tmp_path, monkeypatch, [changed_file])

    assert calls == []


def test_main_legacy_changed_path_ignores_family_dir_incomplete_marker(create_charts_module, tmp_path, monkeypatch):
    """A 5-part (legacy-shaped) changed path validates INCOMPLETE.txt against
    the exact legacy dir. An INCOMPLETE.txt in a SIBLING family dir of the same
    (backend, version) — even one holding more perf files — must not suppress
    the legacy change."""
    changed_file = f"{create_charts_module.SYSTEMS_PREFIX}data/h200_sxm/sglang/0.5.14/gemm_perf.parquet"
    _touch(tmp_path, "h200_sxm/sglang/0.5.14/gemm_perf.parquet")  # clean legacy dir
    # Family dir with MORE perf files than the legacy dir, marked INCOMPLETE.
    _touch(tmp_path, "h200_sxm/attention/sglang/0.5.14/context_attention_perf.parquet")
    _touch(tmp_path, "h200_sxm/attention/sglang/0.5.14/generation_attention_perf.parquet")
    _touch(tmp_path, "h200_sxm/attention/sglang/0.5.14/INCOMPLETE.txt")

    calls = _run_main_with_changed_files(create_charts_module, tmp_path, monkeypatch, [changed_file])

    assert calls == [("h200_sxm", "sglang", "0.5.14", ["gemm_perf.parquet"])]


def test_main_legacy_changed_path_suppressed_by_legacy_incomplete_marker(create_charts_module, tmp_path, monkeypatch):
    """The converse: an INCOMPLETE.txt in the exact legacy dir suppresses the
    5-part changed path even when clean, larger family dirs exist for the same
    (backend, version)."""
    changed_file = f"{create_charts_module.SYSTEMS_PREFIX}data/h200_sxm/sglang/0.5.14/gemm_perf.parquet"
    _touch(tmp_path, "h200_sxm/sglang/0.5.14/gemm_perf.parquet")
    _touch(tmp_path, "h200_sxm/sglang/0.5.14/INCOMPLETE.txt")
    _touch(tmp_path, "h200_sxm/attention/sglang/0.5.14/context_attention_perf.parquet")
    _touch(tmp_path, "h200_sxm/attention/sglang/0.5.14/generation_attention_perf.parquet")

    calls = _run_main_with_changed_files(create_charts_module, tmp_path, monkeypatch, [changed_file])

    assert calls == []

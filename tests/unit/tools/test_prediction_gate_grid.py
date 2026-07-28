# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tools/prediction_regression_gate/grid.py's directory walk.

The tier-1 grid's `enumerate_combos` must find real (system, backend,
version) combos on the family-first (Collector V3) data layout
(<system>/<family>/<backend>/<version>), not just the legacy
(<system>/<backend>/<version>) layout — and it must merge a backend's
versions across every family dir that contributes to it (e.g. both
gemm/vllm and moe/vllm feed the "vllm" backend).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.prediction_regression_gate import grid

pytestmark = pytest.mark.unit


def _touch(root: Path, rel: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"stub")


def _make_system_yaml(systems_root: Path, system: str) -> None:
    systems_root.mkdir(parents=True, exist_ok=True)
    (systems_root / f"{system}.yaml").write_text("data_dir: data\n")


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "systems" / "data"
    monkeypatch.setattr(grid, "DATA_ROOT", root)
    return root


def test_iter_backend_dirs_family_layout(data_root):
    sys_dir = data_root / "b200_sxm"
    _touch(sys_dir, "gemm/vllm/0.19.0/gemm_perf.parquet")
    _touch(sys_dir, "moe/vllm/0.19.0/moe_perf.parquet")
    _touch(sys_dir, "moe/trtllm/1.2.0/moe_perf.parquet")
    _touch(sys_dir, "comm/nccl/2.19/nccl_perf.parquet")

    found = sorted(grid._iter_backend_dirs(sys_dir))
    backends = {b for b, _ in found}
    # Family dirs (gemm, moe, comm) must never surface as "backends" —
    # only the real engine backends nested underneath them should.
    assert backends == {"vllm", "trtllm"}
    # nccl stays excluded exactly like the legacy NON_ENGINE_BACKENDS filter.
    assert not any(b == "nccl" for b, _ in found)


def test_iter_backend_dirs_legacy_layout(data_root):
    sys_dir = data_root / "h100_sxm"
    _touch(sys_dir, "sglang/0.5.12/gemm_perf.parquet")

    found = list(grid._iter_backend_dirs(sys_dir))
    assert found == [("sglang", sys_dir / "sglang")]


def test_version_dirs_merges_across_families(data_root):
    sys_dir = data_root / "b200_sxm"
    _touch(sys_dir, "gemm/vllm/0.19.0/gemm_perf.parquet")
    _touch(sys_dir, "moe/vllm/0.22.0/moe_perf.parquet")

    # "vllm" isn't a top-level dir at all here — it only exists nested under
    # two different family dirs. _version_dirs must merge across both.
    versions = grid._version_dirs("b200_sxm", "vllm")
    assert versions == ["0.19.0", "0.22.0"]


def test_version_dirs_excludes_marker_only_and_incomplete(data_root):
    sys_dir = data_root / "b200_sxm"
    _touch(sys_dir, "gemm/vllm/0.19.0/gemm_perf.parquet")
    _touch(sys_dir, "gemm/vllm/0.20.0/SHARED_LAYER_REUSE.txt")  # marker-only
    _touch(sys_dir, "gemm/vllm/0.21.0/gemm_perf.parquet")
    _touch(sys_dir, "gemm/vllm/0.21.0/INCOMPLETE.txt")  # excluded entirely

    versions = grid._version_dirs("b200_sxm", "vllm")
    assert versions == ["0.19.0"]


def test_enumerate_combos_finds_family_layout_combos(data_root, monkeypatch):
    systems_root = data_root.parent
    _make_system_yaml(systems_root, "b200_sxm")

    sys_dir = data_root / "b200_sxm"
    _touch(sys_dir, "gemm/vllm/0.19.0/gemm_perf.parquet")
    _touch(sys_dir, "gemm/vllm/0.22.0/gemm_perf.parquet")
    _touch(sys_dir, "moe/trtllm/1.2.0/moe_perf.parquet")

    # get_latest_database_version looks at the real installed SDK data via
    # its own systems path resolution, which knows nothing about this
    # synthetic tree — it returns None and enumerate_combos falls back to
    # the newest data-carrying dir it found on disk, so no monkeypatch of
    # the SDK is required for this to exercise the walk end-to-end.
    combos = grid.enumerate_combos(systems=["b200_sxm"], versions="latest")

    by_backend = {c.backend: c.version for c in combos}
    assert by_backend == {"vllm": "0.22.0", "trtllm": "1.2.0"}
    assert all(c.system == "b200_sxm" for c in combos)


def test_enumerate_combos_versions_all_merges_family_versions(data_root):
    systems_root = data_root.parent
    _make_system_yaml(systems_root, "b200_sxm")

    sys_dir = data_root / "b200_sxm"
    _touch(sys_dir, "gemm/vllm/0.19.0/gemm_perf.parquet")
    _touch(sys_dir, "moe/vllm/0.22.0/moe_perf.parquet")

    combos = grid.enumerate_combos(systems=["b200_sxm"], versions="all")
    vllm_versions = sorted(c.version for c in combos if c.backend == "vllm")
    assert vllm_versions == ["0.19.0", "0.22.0"]

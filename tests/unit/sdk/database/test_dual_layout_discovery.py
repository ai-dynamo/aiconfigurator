# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dual-layout (family-first + legacy) discovery tests — Collector V3 PR 2."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiconfigurator.sdk.operations.base import resolve_op_data_path
from aiconfigurator.sdk.perf_database import (
    _iter_database_refs_for_system,
    get_latest_database_version,
    get_supported_databases,
    is_shared_layer_marker_only_version,
)

pytestmark = pytest.mark.unit

PARQUET_STUB = b"PAR1stub"  # discovery only checks existence, never parses


def _write(root: Path, rel: str, data: bytes = PARQUET_STUB) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


@pytest.fixture
def systems_root(tmp_path):
    (tmp_path / "h200_sxm.yaml").write_text("data_dir: data/h200_sxm\n", encoding="utf-8")
    return tmp_path


def test_family_layout_versions_are_discovered_and_merged(systems_root):
    # one (backend, version) split across two family dirs = ONE version
    _write(systems_root, "data/h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
    _write(systems_root, "data/h200_sxm/attention/sglang/0.5.14/context_attention_perf.parquet")
    _write(systems_root, "data/h200_sxm/attention/sglang/0.5.12/context_attention_perf.parquet")
    supported = get_supported_databases(str(systems_root))
    assert supported["h200_sxm"]["sglang"] == ["0.5.12", "0.5.14"]


def test_legacy_layout_still_discovered(systems_root):
    _write(systems_root, "data/h200_sxm/sglang/0.5.14/gemm_perf.parquet")
    supported = get_supported_databases(str(systems_root))
    assert supported["h200_sxm"]["sglang"] == ["0.5.14"]


def test_mixed_layouts_union(systems_root):
    _write(systems_root, "data/h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
    _write(systems_root, "data/h200_sxm/sglang/0.5.12/gemm_perf.parquet")  # legacy straggler
    supported = get_supported_databases(str(systems_root))
    assert supported["h200_sxm"]["sglang"] == ["0.5.12", "0.5.14"]


def test_incomplete_family_dir_excluded_per_path(systems_root):
    # INCOMPLETE in one family dir hides that family's path; the version stays
    # declared through the other family dir (per-path semantics preserved).
    _write(systems_root, "data/h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
    _write(systems_root, "data/h200_sxm/attention/sglang/0.5.14/context_attention_perf.parquet")
    _write(systems_root, "data/h200_sxm/attention/sglang/0.5.14/INCOMPLETE.txt", b"partial")
    supported = get_supported_databases(str(systems_root))
    assert supported["h200_sxm"]["sglang"] == ["0.5.14"]


def test_all_paths_incomplete_means_undeclared(systems_root):
    _write(systems_root, "data/h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
    _write(systems_root, "data/h200_sxm/gemm/sglang/0.5.14/INCOMPLETE.txt", b"partial")
    supported = get_supported_databases(str(systems_root))
    assert supported["h200_sxm"].get("sglang", []) == []


def test_marker_only_version_across_family_dirs(systems_root):
    _write(systems_root, "data/h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
    _write(systems_root, "data/h200_sxm/gemm/sglang/0.5.12/SHARED_LAYER_REUSE.txt", b"")
    _write(systems_root, "data/h200_sxm/attention/sglang/0.5.12/SHARED_LAYER_REUSE.txt", b"")
    assert is_shared_layer_marker_only_version("h200_sxm", "sglang", "0.5.12", systems_paths=str(systems_root))
    assert not is_shared_layer_marker_only_version("h200_sxm", "sglang", "0.5.14", systems_paths=str(systems_root))
    supported = get_supported_databases(str(systems_root))
    assert supported["h200_sxm"]["sglang"] == ["0.5.12", "0.5.14"]


def test_comm_family_pseudo_backends_do_not_pollute_backends(systems_root):
    # nccl lives under the comm family; it must not appear as a backend.
    _write(systems_root, "data/h200_sxm/comm/nccl/2.19/nccl_perf.parquet")
    _write(systems_root, "data/h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
    supported = get_supported_databases(str(systems_root))
    assert set(supported["h200_sxm"].keys()) == {"sglang"}


def test_latest_version_spans_layouts(systems_root):
    _write(systems_root, "data/h200_sxm/sglang/0.5.12/gemm_perf.parquet")  # legacy
    _write(systems_root, "data/h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")  # family
    assert get_latest_database_version("h200_sxm", "sglang", systems_paths=str(systems_root)) == "0.5.14"


def test_resolve_op_path_family_first_then_legacy(systems_root):
    root = str(systems_root / "data/h200_sxm")
    _write(systems_root, "data/h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
    _write(systems_root, "data/h200_sxm/sglang/0.5.14/moe_perf.parquet")  # legacy straggler
    assert resolve_op_data_path(root, "sglang", "0.5.14", "gemm_perf.parquet").endswith(
        "gemm/sglang/0.5.14/gemm_perf.parquet"
    )
    assert resolve_op_data_path(root, "sglang", "0.5.14", "moe_perf.parquet").endswith("sglang/0.5.14/moe_perf.parquet")
    # absent file: legacy-shaped path returned, existence not required
    missing = resolve_op_data_path(root, "sglang", "0.5.14", "mla_bmm_perf.parquet")
    assert missing.endswith("sglang/0.5.14/mla_bmm_perf.parquet")


def test_resolve_op_path_skips_incomplete_family_dir(systems_root):
    root = str(systems_root / "data/h200_sxm")
    _write(systems_root, "data/h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
    _write(systems_root, "data/h200_sxm/gemm/sglang/0.5.14/INCOMPLETE.txt", b"x")
    _write(systems_root, "data/h200_sxm/sglang/0.5.14/gemm_perf.parquet")  # legacy fallback
    assert resolve_op_data_path(root, "sglang", "0.5.14", "gemm_perf.parquet").endswith(
        "sglang/0.5.14/gemm_perf.parquet"
    )


def test_resolve_op_path_txt_fallback_in_family_dir(systems_root):
    root = str(systems_root / "data/h200_sxm")
    _write(systems_root, "data/h200_sxm/gemm/sglang/0.5.14/gemm_perf.txt", b"csv")
    assert resolve_op_data_path(root, "sglang", "0.5.14", "gemm_perf.parquet").endswith(
        "gemm/sglang/0.5.14/gemm_perf.txt"
    )


def test_database_refs_found_from_family_layout_only(systems_root):
    # A database whose data lives ONLY under the family layout must still be
    # discovered by ref-discovery (not just get_supported_databases).
    _write(systems_root, "data/h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
    refs = list(_iter_database_refs_for_system(str(systems_root), "h200_sxm", {"data_dir": "data/h200_sxm"}))
    assert ("h200_sxm", "sglang", "0.5.14", str(systems_root)) in refs


def test_underscore_prefixed_version_dirs_are_skipped(systems_root):
    _write(systems_root, "data/h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
    _write(systems_root, "data/h200_sxm/gemm/sglang/_staging/gemm_perf.parquet")
    supported = get_supported_databases(str(systems_root))
    assert supported["h200_sxm"]["sglang"] == ["0.5.14"]

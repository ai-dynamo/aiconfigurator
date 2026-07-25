# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the op_backend_catalog family-map loader (Collector V3 identity)."""

import pytest

from collector.op_catalog import family_for_perf_file, load_family_map
from collector.registry_types import PerfFile

pytestmark = pytest.mark.unit

CATALOG = """
schema_version: 1
families:
  - family: attention
    op_files: [context_attention_perf, generation_attention_perf]
  - family: gemm
    op_files: [gemm_perf]
"""


def test_absent_catalog_returns_none(tmp_path):
    assert load_family_map(tmp_path / "op_backend_catalog.yaml") is None


def test_family_map_indexes_tables_by_stem(tmp_path):
    path = tmp_path / "op_backend_catalog.yaml"
    path.write_text(CATALOG, encoding="utf-8")
    family_map = load_family_map(path)
    assert family_map == {
        "context_attention_perf": "attention",
        "generation_attention_perf": "attention",
        "gemm_perf": "gemm",
    }


def test_family_for_perf_file_matches_txt_parquet_and_enum(tmp_path):
    path = tmp_path / "op_backend_catalog.yaml"
    path.write_text(CATALOG, encoding="utf-8")
    family_map = load_family_map(path)
    assert family_for_perf_file("gemm_perf.txt", family_map) == "gemm"
    assert family_for_perf_file("gemm_perf.parquet", family_map) == "gemm"
    assert family_for_perf_file(str(PerfFile.CONTEXT_ATTENTION), family_map) == "attention"
    assert family_for_perf_file("nonexistent_perf.txt", family_map) is None


def test_duplicate_table_across_families_is_rejected(tmp_path):
    path = tmp_path / "op_backend_catalog.yaml"
    path.write_text(
        CATALOG + "  - family: moe\n    op_files: [gemm_perf]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="two families"):
        load_family_map(path)

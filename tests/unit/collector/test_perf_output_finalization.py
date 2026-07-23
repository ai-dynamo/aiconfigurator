# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from collector.helper import finalize_perf_files, finalize_perf_outputs, find_perf_csv_outputs

pytestmark = pytest.mark.unit


def _write_perf_csv(path: Path, latency: float = 1.25) -> None:
    path.write_text(f"op,latency\nmatmul,{latency}\n")


def test_find_perf_csv_outputs_is_non_recursive_by_default(tmp_path):
    top_level = tmp_path / "gemm_perf.txt"
    nested = tmp_path / "src" / "aiconfigurator" / "systems" / "data" / "gemm_perf.txt"
    incomplete = tmp_path / "INCOMPLETE.txt"

    _write_perf_csv(top_level)
    nested.parent.mkdir(parents=True)
    _write_perf_csv(nested)
    incomplete.write_text("incomplete\n")

    assert find_perf_csv_outputs(tmp_path) == [top_level]
    assert find_perf_csv_outputs(tmp_path, recursive=True) == [top_level, nested]


def test_find_perf_csv_outputs_ignores_structured_provenance_markers(tmp_path):
    """collection_meta.yaml / reuse.yaml (Collector V3 design §5/§6.3) sit flat
    beside the staged CSV outputs at finalize time but are not CSVs; the
    `*_perf.txt` glob must not pick them up (they don't match the pattern, so
    this is a regression guard, not new filtering logic)."""
    top_level = tmp_path / "gemm_perf.txt"
    _write_perf_csv(top_level)
    (tmp_path / "collection_meta.yaml").write_text("schema_version: 1\n")
    (tmp_path / "reuse.yaml").write_text("schema_version: 1\n")

    assert find_perf_csv_outputs(tmp_path) == [top_level]


def test_finalize_perf_outputs_does_not_recurse_into_checked_in_assets(tmp_path):
    top_level = tmp_path / "gemm_perf.txt"
    nested = tmp_path / "src" / "aiconfigurator" / "systems" / "data" / "gemm_perf.txt"

    _write_perf_csv(top_level)
    nested.parent.mkdir(parents=True)
    _write_perf_csv(nested)

    converted = finalize_perf_outputs(tmp_path)

    assert converted == [top_level.with_suffix(".parquet")]
    assert top_level.with_suffix(".parquet").exists()
    assert not top_level.exists()
    assert nested.exists()
    assert not nested.with_suffix(".parquet").exists()


def test_finalize_perf_files_converts_only_explicit_outputs(tmp_path):
    touched = tmp_path / "gemm_perf.txt"
    untouched = tmp_path / "allreduce_perf.txt"
    nested = tmp_path / "nested" / "moe_perf.txt"

    _write_perf_csv(touched, latency=1.0)
    _write_perf_csv(untouched, latency=2.0)
    nested.parent.mkdir()
    _write_perf_csv(nested, latency=3.0)

    converted = finalize_perf_files([touched, touched, nested])

    assert converted == [touched.with_suffix(".parquet"), nested.with_suffix(".parquet")]
    assert pq.read_table(touched.with_suffix(".parquet")).to_pylist() == [{"op": "matmul", "latency": 1.0}]
    assert pq.read_table(nested.with_suffix(".parquet")).to_pylist() == [{"op": "matmul", "latency": 3.0}]
    assert untouched.exists()
    assert not untouched.with_suffix(".parquet").exists()

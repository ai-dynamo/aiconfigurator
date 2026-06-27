# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from collector.helper import finalize_perf_files, finalize_perf_outputs, find_perf_csv_outputs


def _load_require_collection_success():
    source_path = Path(__file__).resolve().parents[3] / "collector" / "collect.py"
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_require_collection_success"
    )
    namespace = {"datetime": datetime, "logger": None}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["_require_collection_success"]


def _load_select_perf_outputs_for_finalization():
    source_path = Path(__file__).resolve().parents[3] / "collector" / "collect.py"
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_select_perf_outputs_for_finalization"
    )
    namespace = {"Path": Path, "find_perf_csv_outputs": find_perf_csv_outputs}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["_select_perf_outputs_for_finalization"]


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


def test_collection_errors_block_output_finalization():
    require_collection_success = _load_require_collection_success()
    assert require_collection_success([], "sglang") == []

    with pytest.raises(RuntimeError, match="did not return a result"):
        require_collection_success(None, "sglang")
    with pytest.raises(RuntimeError, match="1 error"):
        require_collection_success([{"error_type": "WorkerFailure"}], "sglang")


def test_resume_finalizes_prior_requested_staging_files_without_touching_unrelated_files(tmp_path):
    select_outputs = _load_select_perf_outputs_for_finalization()
    prior_requested = tmp_path / "gemm_perf.txt"
    prior_unrelated = tmp_path / "allreduce_perf.txt"
    touched_now = tmp_path / "moe_perf.txt"
    for path in (prior_requested, prior_unrelated, touched_now):
        _write_perf_csv(path)
    existing = {path.resolve(): path.stat().st_mtime_ns for path in (prior_requested, prior_unrelated, touched_now)}
    _write_perf_csv(touched_now, latency=4.0)

    assert select_outputs(tmp_path, existing, {"gemm_perf.txt"}, resume=False) == [touched_now]
    assert set(select_outputs(tmp_path, existing, {"gemm_perf.txt"}, resume=True)) == {
        prior_requested,
        touched_now,
    }

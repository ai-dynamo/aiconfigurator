# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import os
from pathlib import Path

import pyarrow.parquet as pq
import pytest

import collector.helper as helper
from aiconfigurator.sdk.operations.attention import load_context_attention_data
from aiconfigurator.sdk.operations.gemm import load_gemm_data
from collector.helper import (
    COLLECTOR_INVOCATION_ENV,
    COLLECTOR_INVOCATION_FIELD,
    PhysicalRowConflictError,
    collector_invocation,
    convert_perf_csv_to_parquet,
    delta_latency_power_stats,
    finalize_perf_files,
    finalize_perf_outputs,
    find_perf_csv_outputs,
    log_perf,
    validate_perf_csv_rows,
)


def _write_perf_csv(path: Path, latency: float = 1.25) -> None:
    path.write_text(f"op,latency\nmatmul,{latency}\n")


def _write_planned_gemm_csv(path: Path, rows: list[dict], *, include_power: bool = False) -> None:
    fieldnames = [
        "framework",
        "version",
        "device",
        "op_name",
        "kernel_source",
        COLLECTOR_INVOCATION_FIELD,
        "gemm_dtype",
        "m",
        "n",
        "k",
        "latency",
    ]
    if include_power:
        fieldnames.extend(["power", "power_limit"])
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "framework": "SGLANG",
                    "version": "0.5.10",
                    "device": "B200",
                    "op_name": "gemm",
                    "kernel_source": "cutlass",
                    "gemm_dtype": "bfloat16",
                    "m": 1,
                    "n": 128,
                    "k": 128,
                    **row,
                }
            )


def _write_planned_context_attention_csv(path: Path, rows: list[dict], *, include_window: bool) -> None:
    fieldnames = [
        "framework",
        "version",
        "device",
        "op_name",
        "kernel_source",
        COLLECTOR_INVOCATION_FIELD,
        "attn_dtype",
        "kv_cache_dtype",
        "batch_size",
        "isl",
        "num_heads",
        "num_key_value_heads",
        "head_dim",
        "step",
        "latency",
    ]
    if include_window:
        fieldnames.append("window_size")
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "framework": "SGLANG",
                    "version": "0.5.10",
                    "device": "B200",
                    "op_name": "context_attention",
                    "kernel_source": "flashinfer",
                    "attn_dtype": "bfloat16",
                    "kv_cache_dtype": "bfloat16",
                    "batch_size": 1,
                    "isl": 128,
                    "num_heads": 8,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "step": 0,
                    "latency": 1.0,
                    **row,
                }
            )


def test_delta_latency_power_stats_preserves_energy_difference():
    latency, power_stats = delta_latency_power_stats(
        dynamic_latency=10.0,
        static_latency=6.0,
        dynamic_power_stats={"power": 100.0, "power_limit": 700.0},
        static_power_stats={"power": 50.0, "power_limit": 700.0},
    )

    assert latency == 4.0
    assert power_stats["power"] * latency == 700.0  # 100*10 - 50*6
    assert power_stats["power_limit"] == 700.0


def test_delta_latency_power_stats_clamps_noisy_negative_delta():
    latency, power_stats = delta_latency_power_stats(
        dynamic_latency=5.0,
        static_latency=6.0,
        dynamic_power_stats={"power": 100.0},
        static_power_stats={"power": 50.0},
    )

    assert latency == 0.0
    assert power_stats["power"] == 0.0


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


def test_validate_perf_csv_rows_coalesces_same_invocation_retry_and_strips_staging_id(tmp_path):
    path = tmp_path / "gemm_perf.txt"
    _write_planned_gemm_csv(
        path,
        [
            {COLLECTOR_INVOCATION_FIELD: "same", "latency": 1.0},
            {COLLECTOR_INVOCATION_FIELD: "same", "latency": 2.0},
        ],
    )

    report = validate_perf_csv_rows(path)

    assert report == {"rows": 1, "deduplicated_retries": 1, "validated_physical_keys": 2}
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        assert COLLECTOR_INVOCATION_FIELD not in reader.fieldnames
        assert list(reader)[0]["latency"] == "2.0"


def test_validate_perf_csv_rows_rejects_different_invocations_for_one_consumer_key(tmp_path):
    path = tmp_path / "gemm_perf.txt"
    _write_planned_gemm_csv(
        path,
        [
            {COLLECTOR_INVOCATION_FIELD: "first", "latency": 1.0},
            {COLLECTOR_INVOCATION_FIELD: "second", "latency": 1.0},
        ],
    )

    with pytest.raises(PhysicalRowConflictError, match="conflicting invocations"):
        validate_perf_csv_rows(path)

    assert COLLECTOR_INVOCATION_FIELD in path.read_text(encoding="utf-8").splitlines()[0]


def test_validate_perf_csv_rows_preserves_unattributed_legacy_duplicates(tmp_path):
    path = tmp_path / "gemm_perf.txt"
    _write_planned_gemm_csv(
        path,
        [
            {COLLECTOR_INVOCATION_FIELD: "", "latency": 1.0},
            {COLLECTOR_INVOCATION_FIELD: "", "latency": 2.0},
        ],
    )

    report = validate_perf_csv_rows(path)

    assert report == {"rows": 2, "deduplicated_retries": 0, "validated_physical_keys": 2}
    with path.open(newline="", encoding="utf-8") as source:
        assert [row["latency"] for row in csv.DictReader(source)] == ["1.0", "2.0"]


def test_validate_perf_csv_rows_rejects_planned_collision_with_unattributed_legacy_row(tmp_path):
    path = tmp_path / "gemm_perf.txt"
    _write_planned_gemm_csv(
        path,
        [
            {COLLECTOR_INVOCATION_FIELD: "", "latency": 1.0},
            {COLLECTOR_INVOCATION_FIELD: "planned", "latency": 2.0},
        ],
    )

    with pytest.raises(PhysicalRowConflictError, match="unattributed legacy row"):
        validate_perf_csv_rows(path)


def test_finalize_migrated_csv_removes_invocation_metadata_from_parquet(tmp_path):
    path = tmp_path / "gemm_perf.txt"
    _write_planned_gemm_csv(
        path,
        [{COLLECTOR_INVOCATION_FIELD: "planned", "latency": 1.0}],
    )

    [parquet_path] = finalize_perf_files([path])

    table = pq.read_table(parquet_path)
    assert COLLECTOR_INVOCATION_FIELD not in table.column_names
    assert table.num_rows == 1


def test_failed_parquet_conversion_preserves_original_staging_csv(tmp_path, monkeypatch):
    path = tmp_path / "gemm_perf.txt"
    _write_planned_gemm_csv(
        path,
        [{COLLECTOR_INVOCATION_FIELD: "planned", "latency": 1.0}],
    )
    original = path.read_bytes()

    def fail_read_csv(_path):
        raise RuntimeError("synthetic conversion failure")

    monkeypatch.setattr("pyarrow.csv.read_csv", fail_read_csv)
    with pytest.raises(RuntimeError, match="synthetic conversion failure"):
        convert_perf_csv_to_parquet(path)

    assert path.read_bytes() == original
    assert COLLECTOR_INVOCATION_FIELD in path.read_text(encoding="utf-8").splitlines()[0]
    assert not path.with_suffix(".parquet").exists()


def test_incremental_finalization_merges_existing_parquet_and_preserves_first_measurement(tmp_path):
    path = tmp_path / "gemm_perf.txt"
    _write_planned_gemm_csv(
        path,
        [{COLLECTOR_INVOCATION_FIELD: "run-a", "m": 1, "latency": 1.0}],
    )
    parquet_path = convert_perf_csv_to_parquet(path)

    _write_planned_gemm_csv(
        path,
        [
            {COLLECTOR_INVOCATION_FIELD: "run-b-overlap", "m": 1, "latency": 9.0},
            {COLLECTOR_INVOCATION_FIELD: "run-b-new", "m": 2, "latency": 2.0},
        ],
    )
    convert_perf_csv_to_parquet(path)

    rows = sorted(pq.read_table(parquet_path).to_pylist(), key=lambda row: row["m"])
    assert [(row["m"], row["latency"]) for row in rows] == [(1, 1.0), (2, 2.0)]


@pytest.mark.parametrize("existing_has_power", [False, True])
def test_incremental_finalization_fills_optional_numeric_schema_drift(tmp_path, existing_has_power):
    path = tmp_path / "gemm_perf.txt"
    existing_row = {COLLECTOR_INVOCATION_FIELD: "run-a", "m": 1, "latency": 1.0}
    new_row = {COLLECTOR_INVOCATION_FIELD: "run-b", "m": 2, "latency": 2.0}
    powered_row = {"power": 300.0, "power_limit": 1000.0}
    if existing_has_power:
        existing_row.update(powered_row)
    else:
        new_row.update(powered_row)

    _write_planned_gemm_csv(path, [existing_row], include_power=existing_has_power)
    parquet_path = convert_perf_csv_to_parquet(path)
    _write_planned_gemm_csv(path, [new_row], include_power=not existing_has_power)
    convert_perf_csv_to_parquet(path)

    rows = sorted(pq.read_table(parquet_path).to_pylist(), key=lambda row: row["m"])
    expected_power = [300.0, 0.0] if existing_has_power else [0.0, 300.0]
    expected_power_limit = [1000.0, 0.0] if existing_has_power else [0.0, 1000.0]
    assert [row["power"] for row in rows] == expected_power
    assert [row["power_limit"] for row in rows] == expected_power_limit
    assert load_gemm_data(str(parquet_path)) is not None


@pytest.mark.parametrize("existing_has_window", [False, True])
def test_incremental_finalization_fills_legacy_attention_window_size(tmp_path, existing_has_window):
    path = tmp_path / "context_attention_perf.txt"
    existing_row = {COLLECTOR_INVOCATION_FIELD: "run-a", "batch_size": 1}
    new_row = {COLLECTOR_INVOCATION_FIELD: "run-b", "batch_size": 2}
    if existing_has_window:
        existing_row["window_size"] = 128
    else:
        new_row["window_size"] = 128

    _write_planned_context_attention_csv(path, [existing_row], include_window=existing_has_window)
    parquet_path = convert_perf_csv_to_parquet(path)
    _write_planned_context_attention_csv(path, [new_row], include_window=not existing_has_window)
    convert_perf_csv_to_parquet(path)

    rows = sorted(pq.read_table(parquet_path).to_pylist(), key=lambda row: row["batch_size"])
    assert [row["window_size"] for row in rows] == ([128, 0] if existing_has_window else [0, 128])
    assert load_context_attention_data(str(parquet_path)) is not None


def test_log_perf_lock_failure_is_not_marked_as_success(tmp_path, monkeypatch):
    path = tmp_path / "gemm_perf.txt"
    Path(f"{path}.lock").write_text("held", encoding="utf-8")
    monkeypatch.setattr(helper.time, "sleep", lambda _seconds: None)

    with pytest.raises(TimeoutError, match="cannot acquire perf log lock"):
        log_perf(
            item_list=[{"gemm_dtype": "bfloat16", "m": 1, "n": 128, "k": 128, "latency": 1.0}],
            framework="SGLANG",
            version="0.5.10",
            device_name="B200",
            op_name="gemm",
            kernel_source="cutlass",
            perf_filename=str(path),
        )


def test_log_perf_migrates_existing_csv_header_before_appending_planned_row(tmp_path):
    path = tmp_path / "gemm_perf.txt"
    base_fields = ["framework", "version", "device", "op_name", "kernel_source"]
    item_fields = ["gemm_dtype", "m", "n", "k", "latency"]
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=[*base_fields, *item_fields])
        writer.writeheader()
        writer.writerow(
            {
                "framework": "SGLANG",
                "version": "0.5.10",
                "device": "B200",
                "op_name": "gemm",
                "kernel_source": "cutlass",
                "gemm_dtype": "bfloat16",
                "m": 1,
                "n": 128,
                "k": 128,
                "latency": 1.0,
            }
        )

    with collector_invocation("planned"):
        log_perf(
            item_list=[{"gemm_dtype": "bfloat16", "m": 2, "n": 128, "k": 128, "latency": 2.0}],
            framework="SGLANG",
            version="0.5.10",
            device_name="B200",
            op_name="gemm",
            kernel_source="cutlass",
            perf_filename=str(path),
        )

    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
    assert COLLECTOR_INVOCATION_FIELD in reader.fieldnames
    assert rows[0][COLLECTOR_INVOCATION_FIELD] == ""
    assert rows[1][COLLECTOR_INVOCATION_FIELD] == "planned"


def test_collector_invocation_bridges_subprocess_environment_and_restores_it(monkeypatch):
    monkeypatch.setenv(COLLECTOR_INVOCATION_ENV, "outer")

    with collector_invocation("planned"):
        assert os.environ[COLLECTOR_INVOCATION_ENV] == "planned"

    assert os.environ[COLLECTOR_INVOCATION_ENV] == "outer"


def test_log_perf_reads_inherited_invocation_when_contextvar_is_empty(tmp_path):
    path = tmp_path / "gemm_perf.txt"
    with collector_invocation("subprocess-parent"):
        token = helper._CURRENT_INVOCATION_ID.set("")
        try:
            log_perf(
                item_list=[{"gemm_dtype": "bfloat16", "m": 1, "n": 128, "k": 128, "latency": 1.0}],
                framework="SGLANG",
                version="0.5.10",
                device_name="B200",
                op_name="gemm",
                kernel_source="cutlass",
                perf_filename=str(path),
            )
        finally:
            helper._CURRENT_INVOCATION_ID.reset(token)

    with path.open(newline="", encoding="utf-8") as source:
        row = next(csv.DictReader(source))
    assert row[COLLECTOR_INVOCATION_FIELD] == "subprocess-parent"

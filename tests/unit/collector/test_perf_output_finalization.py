# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from collector.helper import (
    convert_perf_csv_to_parquet,
    finalize_perf_files,
    finalize_perf_outputs,
    find_perf_csv_outputs,
)

pytestmark = pytest.mark.unit


def _write_perf_csv(path: Path, latency: float = 1.25) -> None:
    path.write_text(f"op,latency\nmatmul,{latency}\n")


def _write_keyed_perf_csv(path: Path, rows) -> None:
    """rows: iterable of (shape, latency). Identity key is (op, shape)."""
    lines = ["op,shape,latency"] + [f"matmul,{shape},{latency}" for shape, latency in rows]
    path.write_text("\n".join(lines) + "\n")


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


def _rows_by_key(parquet_path):
    """Return {shape: latency} for a keyed perf parquet."""
    return {r["shape"]: r["latency"] for r in pq.read_table(parquet_path).to_pylist()}


def test_finalize_merges_disjoint_keys_into_existing_parquet(tmp_path):
    # Regression for the resume/retry-failed data-loss footgun: finalize deletes
    # the source .txt, so a second (partial) finalize must NOT clobber the
    # already-finalized rows — the disjoint keys from both runs must survive.
    perf = tmp_path / "gemm_perf.txt"
    parquet = perf.with_suffix(".parquet")

    _write_keyed_perf_csv(perf, [("s1", 1.0)])  # first (full) run
    finalize_perf_files([perf])
    assert not perf.exists()  # source consumed
    assert _rows_by_key(parquet) == {"s1": 1.0}

    _write_keyed_perf_csv(perf, [("s2", 2.0)])  # retry-failed run: a new key only
    finalize_perf_files([perf])
    assert _rows_by_key(parquet) == {"s1": 1.0, "s2": 2.0}  # s1 NOT lost


def test_finalize_merge_replaces_same_key_with_newest_measurement(tmp_path):
    perf = tmp_path / "gemm_perf.txt"
    parquet = perf.with_suffix(".parquet")

    _write_keyed_perf_csv(perf, [("s1", 1.0), ("s2", 2.0)])
    finalize_perf_files([perf])

    # Re-measure s1 (new value) and add s3; s2 untouched must persist.
    _write_keyed_perf_csv(perf, [("s1", 9.0), ("s3", 3.0)])
    finalize_perf_files([perf])

    assert _rows_by_key(parquet) == {"s1": 9.0, "s2": 2.0, "s3": 3.0}
    # exactly one row per identity key (no duplicate keys)
    shapes = [r["shape"] for r in pq.read_table(parquet).to_pylist()]
    assert sorted(shapes) == ["s1", "s2", "s3"]


def test_convert_without_merge_overwrites(tmp_path):
    perf = tmp_path / "gemm_perf.txt"
    parquet = perf.with_suffix(".parquet")

    _write_keyed_perf_csv(perf, [("s1", 1.0)])
    convert_perf_csv_to_parquet(perf, merge_existing=False)
    assert _rows_by_key(parquet) == {"s1": 1.0}

    _write_keyed_perf_csv(perf, [("s2", 2.0)])
    convert_perf_csv_to_parquet(perf, merge_existing=False)
    assert _rows_by_key(parquet) == {"s2": 2.0}  # legacy overwrite preserved when opted out


def test_finalize_merge_falls_back_to_overwrite_on_schema_mismatch(tmp_path):
    perf = tmp_path / "gemm_perf.txt"
    parquet = perf.with_suffix(".parquet")

    _write_keyed_perf_csv(perf, [("s1", 1.0)])
    finalize_perf_files([perf])

    # A run whose columns differ (no 'shape') cannot be safely merged; the
    # finalize must not raise and must not silently corrupt — it overwrites.
    perf.write_text("op,latency\nmatmul,5.0\n")
    finalize_perf_files([perf])
    assert pq.read_table(parquet).to_pylist() == [{"op": "matmul", "latency": 5.0}]


def test_finalize_merge_tolerates_metric_column_type_drift(tmp_path):
    """An all-empty optional metric column (e.g. power on a power-off run) is
    inferred as `null` by pyarrow.csv while populated runs infer `double`.
    That drift must NOT be treated as a schema mismatch — the old behavior
    silently overwrote the accumulated parquet with the new run's subset."""
    perf = tmp_path / "gemm_perf.txt"
    parquet = perf.with_suffix(".parquet")

    perf.write_text("shape,latency,power\ns1,1.0,5.0\ns2,2.0,6.0\n")
    finalize_perf_files([perf])

    # New run with power entirely empty -> pyarrow infers `null` for it.
    perf.write_text("shape,latency,power\ns3,3.0,\n")
    finalize_perf_files([perf])

    rows = {r["shape"]: r for r in pq.read_table(parquet).to_pylist()}
    assert sorted(rows) == ["s1", "s2", "s3"]  # accumulated, not overwritten
    assert rows["s1"]["power"] == 5.0
    assert rows["s3"]["power"] == 0.0
    # metric column keeps the existing (double) type
    assert str(pq.read_table(parquet).schema.field("power").type) == "double"


def test_finalize_merge_still_overwrites_on_identity_type_mismatch(tmp_path):
    """Identity-column type drift stays a hard mismatch (overwrite path)."""
    perf = tmp_path / "gemm_perf.txt"
    parquet = perf.with_suffix(".parquet")

    perf.write_text("shape,latency\n1,1.0\n")  # shape inferred int64
    finalize_perf_files([perf])
    perf.write_text("shape,latency\ns2,2.0\n")  # shape inferred string
    finalize_perf_files([perf])
    assert [r["shape"] for r in pq.read_table(parquet).to_pylist()] == ["s2"]


def test_finalize_merge_lock_does_not_block_sequential_runs(tmp_path):
    """The flock-based merge lock must release on close: a second finalize of
    the same target must proceed (and the lock file itself stays, by design —
    unlinking would reopen the create/steal race)."""
    perf = tmp_path / "gemm_perf.txt"
    parquet = perf.with_suffix(".parquet")
    perf.write_text("shape,latency\ns1,1.0\n")
    finalize_perf_files([perf])
    perf.write_text("shape,latency\ns2,2.0\n")
    finalize_perf_files([perf])  # would deadlock if the first run kept the flock
    assert sorted(r["shape"] for r in pq.read_table(parquet).to_pylist()) == ["s1", "s2"]


def test_finalize_merge_tolerates_reverse_metric_type_drift(tmp_path):
    """Old parquet has an all-null metric column, the NEW run has real values:
    the null side must be cast toward double — the naive cast direction
    (double -> null) raises ArrowNotImplementedError and aborted finalize."""
    perf = tmp_path / "gemm_perf.txt"
    parquet = perf.with_suffix(".parquet")

    pq.write_table(
        pa.table(
            {
                "shape": ["s1"],
                "latency": [1.0],
                "power": pa.nulls(1),
            }
        ),
        parquet,
    )
    perf.write_text("shape,latency,power\ns2,2.0,7.5\n")
    finalize_perf_files([perf])

    rows = {r["shape"]: r for r in pq.read_table(parquet).to_pylist()}
    assert sorted(rows) == ["s1", "s2"]
    assert rows["s1"]["power"] == 0.0
    assert rows["s2"]["power"] == 7.5
    assert str(pq.read_table(parquet).schema.field("power").type) == "double"


def test_finalize_normalizes_all_empty_power_metrics_to_typed_zero(tmp_path):
    perf = tmp_path / "gemm_perf.txt"
    parquet = perf.with_suffix(".parquet")

    perf.write_text("shape,latency,power,power_limit\ns1,1.0,,\ns2,2.0,,\n")
    finalize_perf_files([perf])

    table = pq.read_table(parquet)
    assert table.select(["power", "power_limit"]).to_pylist() == [
        {"power": 0.0, "power_limit": 0.0},
        {"power": 0.0, "power_limit": 0.0},
    ]
    assert str(table.schema.field("power").type) == "double"
    assert str(table.schema.field("power_limit").type) == "double"
    assert table.column("power").null_count == 0
    assert table.column("power_limit").null_count == 0


@pytest.mark.parametrize("value", ["-1.0", "inf"])
def test_finalize_rejects_invalid_power_measurements(tmp_path, value):
    perf = tmp_path / "gemm_perf.txt"
    perf.write_text(f"shape,latency,power,power_limit\ns1,1.0,{value},1000.0\n")

    with pytest.raises(ValueError, match="power must contain finite non-negative values"):
        finalize_perf_files([perf])


def test_finalize_treats_csv_nan_power_as_unavailable(tmp_path):
    perf = tmp_path / "gemm_perf.txt"
    perf.write_text("shape,latency,power,power_limit\ns1,1.0,nan,1000.0\n")

    [parquet] = finalize_perf_files([perf])

    assert pq.read_table(parquet).column("power").to_pylist() == [0.0]


def test_finalize_keeps_power_disabled_schema_column_free(tmp_path):
    perf = tmp_path / "gemm_perf.txt"
    perf.write_text("shape,latency\ns1,1.0\n")

    [parquet] = finalize_perf_files([perf])

    assert pq.read_table(parquet).column_names == ["shape", "latency"]

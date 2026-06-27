# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
import json

import pytest

from collector.planner.coverage import (
    COVERAGE_CANONICAL_VERSION,
    CoverageHeader,
    CoverageManifest,
    CoverageManifestError,
    CoverageMismatchError,
    assert_legacy_subset,
    diff,
    load,
    write,
)
from collector.planner.physical_keys import PHYSICAL_KEY_SCHEMA_VERSION, physical_row_key
from tools.collector.generate_coverage_manifest import main

pytestmark = pytest.mark.unit


def _header(**overrides) -> CoverageHeader:
    values = {
        "source_git_ref": "collector-v1-deadbeef",
        "backend_variant": "trtllm",
        "framework_version": "1.3.0rc10",
        "gpu_type": "b200_sxm",
        "sm_version": 100,
        "perf_table": "gemm_perf.parquet",
        "key_schema_version": PHYSICAL_KEY_SCHEMA_VERSION,
        "canonical_version": COVERAGE_CANONICAL_VERSION,
    }
    values.update(overrides)
    return CoverageHeader(**values)


def _gemm_key(m: int):
    key = physical_row_key(
        "gemm_perf.parquet",
        {"gemm_dtype": "bf16", "m": m, "n": 128, "k": 256},
    )
    assert key is not None
    return key


def test_manifest_is_deterministic_gzip_jsonl_and_round_trips_exact_values(tmp_path):
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"
    key_one = _gemm_key(1)
    key_two = _gemm_key(2)

    expected = CoverageManifest(_header(), frozenset({key_one, key_two}))
    write(first, expected)
    write(second, _header(), [key_two, key_one, key_two])

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes()[4:8] == b"\x00\x00\x00\x00"
    assert load(first) == expected

    lines = gzip.decompress(first.read_bytes()).decode("utf-8").splitlines()
    header = json.loads(lines[0])
    assert set(header) == {
        "source_git_ref",
        "backend_variant",
        "framework_version",
        "gpu_type",
        "sm_version",
        "perf_table",
        "key_schema_version",
        "canonical_version",
    }
    assert [json.loads(line) for line in lines[1:]] == [list(key_one.values), list(key_two.values)]


def test_diff_and_subset_assertion_report_missing_exact_key_sample():
    legacy = CoverageManifest(_header(source_git_ref="collector-v1"), frozenset({_gemm_key(1), _gemm_key(2)}))
    generated = CoverageManifest(_header(source_git_ref="current"), frozenset({_gemm_key(2), _gemm_key(3)}))

    result = diff(legacy, generated)

    assert result.legacy_count == 2
    assert result.generated_count == 2
    assert result.retained_count == 1
    assert [key.values for key in result.missing] == [("bfloat16", 1, 128, 256)]
    assert [key.values for key in result.added] == [("bfloat16", 3, 128, 256)]
    assert not result.legacy_is_subset
    with pytest.raises(CoverageMismatchError, match=r'missing key sample: \["bfloat16",1,128,256\]'):
        assert_legacy_subset(legacy, generated)

    superset = CoverageManifest(_header(source_git_ref="current"), legacy.keys | {_gemm_key(3)})
    assert assert_legacy_subset(legacy, superset).legacy_is_subset


def test_diff_rejects_different_manifest_scopes():
    legacy = CoverageManifest(_header(), frozenset({_gemm_key(1)}))
    other_gpu = CoverageManifest(_header(source_git_ref="current", sm_version=120), frozenset({_gemm_key(1)}))

    with pytest.raises(CoverageManifestError, match="sm_version"):
        diff(legacy, other_gpu)


def test_unknown_perf_table_fails_explicitly():
    with pytest.raises(CoverageManifestError, match="unknown perf table"):
        _header(perf_table="unknown_perf.parquet")


@pytest.mark.parametrize("input_format", ["json", "jsonl"])
def test_cli_builds_manifest_from_json_or_jsonl_rows(tmp_path, input_format):
    rows = [
        {"gemm_dtype": "bf16", "m": 2, "n": 128, "k": 256, "latency": 9.0},
        {"gemm_dtype": "bf16", "m": 1, "n": 128, "k": 256, "latency": 1.0},
    ]
    input_path = tmp_path / f"rows.{input_format}"
    if input_format == "json":
        input_path.write_text(json.dumps(rows), encoding="utf-8")
    else:
        input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    output_path = tmp_path / f"coverage-{input_format}.jsonl.gz"

    exit_code = main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--perf-file",
            "gemm_perf.txt",
            "--source-git-ref",
            "collector-v1",
            "--backend-variant",
            "trtllm",
            "--framework-version",
            "1.3.0rc10",
            "--gpu-type",
            "b200_sxm",
            "--sm-version",
            "100",
        ]
    )

    manifest = load(output_path)
    assert exit_code == 0
    assert manifest.header.perf_table == "gemm_perf.parquet"
    assert manifest.keys == frozenset({_gemm_key(1), _gemm_key(2)})


def test_cli_rejects_unknown_table_without_writing_output(tmp_path, capsys):
    input_path = tmp_path / "rows.json"
    input_path.write_text("[]", encoding="utf-8")
    output_path = tmp_path / "must-not-exist.jsonl.gz"

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--perf-file",
                "unknown_perf.parquet",
                "--source-git-ref",
                "collector-v1",
                "--backend-variant",
                "trtllm",
                "--framework-version",
                "1.3.0rc10",
                "--gpu-type",
                "b200_sxm",
                "--sm-version",
                "100",
            ]
        )

    assert exc_info.value.code == 2
    assert "unknown perf table" in capsys.readouterr().err
    assert not output_path.exists()

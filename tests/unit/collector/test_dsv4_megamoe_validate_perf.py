# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv

import pytest

from collector.sglang.dsv4_megamoe.validate_perf import merge_perf_files, validate_perf_file

pytestmark = pytest.mark.unit


def _row(**overrides):
    row = {
        "framework": "SGLang",
        "version": "0.5.10",
        "device": "NVIDIA GB200",
        "op_name": "dsv4_megamoe_module",
        "kernel_source": "deepgemm_megamoe",
        "phase": "context",
        "moe_dtype": "w4a8_mxfp4_mxfp8",
        "kernel_dtype": "fp8_fp4",
        "num_tokens": "1024",
        "global_num_tokens": "8192",
        "hidden_size": "7168",
        "inter_size": "3072",
        "topk": "6",
        "num_experts": "384",
        "num_fused_shared_experts": "0",
        "moe_tp_size": "1",
        "moe_ep_size": "8",
        "distribution": "balanced",
        "source_policy": "random",
        "pre_dispatch": "sglang_jit",
        "num_max_tokens_per_rank": "32768",
        "effective_num_max_tokens_per_rank": "32768",
        "routed_scaling_factor": "2.5",
        "includes_routed_scale": "true",
        "includes_gate_topk": "false",
        "buffer_policy": "cached_sglang",
        "includes_buffer_init": "false",
        "used_cuda_graph": "true",
        "latency": "1.25",
    }
    row.update(overrides)
    return row


def _write_perf(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(_row().keys())
    for row in rows:
        fieldnames.extend(field for field in row if field not in fieldnames)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_validate_perf_file_rejects_duplicate_loader_keys(tmp_path):
    perf_path = tmp_path / "dsv4_megamoe_module_perf.txt"
    _write_perf(perf_path, [_row(), _row(latency="1.50")])

    text, errors = validate_perf_file(
        perf_path=perf_path,
        prefill_eps=[8],
        decode_eps=[],
        prefill_tokens=[1024],
        decode_tokens=[],
        distributions=["balanced"],
        phases={"context"},
        summary_path=tmp_path / "validation_summary.txt",
        target_version="0.5.10",
        allow_version_mismatch=False,
        expect_single_perf_file=True,
    )

    assert "duplicate_loader_keys=1" in text
    assert any("duplicate loader key groups" in error for error in errors)


def test_validate_perf_file_passes_expected_rows(tmp_path):
    perf_path = tmp_path / "dsv4_megamoe_module_perf.txt"
    _write_perf(perf_path, [_row()])

    text, errors = validate_perf_file(
        perf_path=perf_path,
        prefill_eps=[8],
        decode_eps=[],
        prefill_tokens=[1024],
        decode_tokens=[],
        distributions=["balanced"],
        phases={"context"},
        summary_path=tmp_path / "validation_summary.txt",
        target_version="0.5.10",
        allow_version_mismatch=False,
        expect_single_perf_file=True,
    )

    assert errors == []
    assert "VALIDATION=PASS" in text


def test_merge_perf_files_preserves_union_header(tmp_path):
    input_root = tmp_path / "remote_results"
    first = input_root / "job_a" / "dsv4_megamoe_module_perf.txt"
    second = input_root / "job_b" / "dsv4_megamoe_module_perf.txt"
    _write_perf(first, [_row()])
    _write_perf(second, [_row(phase="generation", num_tokens="1", latency="2.50")])
    output = tmp_path / "merged" / "dsv4_megamoe_module_perf.txt"

    merged_files, rows = merge_perf_files(input_root, "dsv4_megamoe_module_perf.txt", output)

    assert merged_files == 2
    assert rows == 2
    with output.open(newline="") as f:
        assert len(list(csv.DictReader(f))) == 2

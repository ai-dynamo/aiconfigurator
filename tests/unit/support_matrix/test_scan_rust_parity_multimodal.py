# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
from types import SimpleNamespace

import pandas as pd
import pytest

from tools.support_matrix.scan_rust_parity import (
    PARETO_STATUS_STRICT_PASS,
    Entry,
    _run_probe,
    load_entries,
    pareto_entry,
)
from tools.support_matrix.support_matrix import (
    STATUS_PASS,
    SUPPORT_MATRIX_HEADER_WITH_SOURCE,
    SUPPORT_MATRIX_IMAGE_WORKLOAD,
    SupportMatrix,
    TestConstraints,
)

pytestmark = pytest.mark.unit


def _entry() -> Entry:
    return Entry(
        model="Qwen/Qwen3-VL-8B-Instruct",
        architecture="Qwen3VLForConditionalGeneration",
        system="b200_sxm",
        backend="vllm",
        version="0.24.0",
        mode="agg",
        baseline_status=STATUS_PASS,
        image_workload=SUPPORT_MATRIX_IMAGE_WORKLOAD,
    )


def test_legacy_parity_entry_infers_canonical_image_workload(tmp_path):
    csv_path = tmp_path / "b200_sxm.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SUPPORT_MATRIX_HEADER_WITH_SOURCE)
        writer.writerow(
            [
                "Qwen/Qwen3-VL-8B-Instruct",
                "Qwen3VLForConditionalGeneration",
                "b200_sxm",
                "vllm",
                "0.24.0",
                "agg",
                STATUS_PASS,
                "",
                "uv run aiconfigurator cli default --database-mode SILICON",
                "silicon",
            ]
        )

    [entry] = load_entries(tmp_path)

    assert entry.image_workload == SUPPORT_MATRIX_IMAGE_WORKLOAD
    assert entry.key.endswith("image=1024x1024x1")


def test_legacy_encoder_unsupported_pass_is_skipped_without_blocking_scan(tmp_path, caplog):
    csv_path = tmp_path / "b200_sxm.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SUPPORT_MATRIX_HEADER_WITH_SOURCE)
        writer.writerow(
            [
                "Qwen/Qwen3.5-27B",
                "Qwen3_5ForConditionalGeneration",
                "b200_sxm",
                "vllm",
                "0.24.0",
                "agg",
                STATUS_PASS,
                "",
                "uv run aiconfigurator cli default --database-mode SILICON",
                "silicon",
            ]
        )
        writer.writerow(
            [
                "Qwen/Qwen3-8B",
                "Qwen3ForCausalLM",
                "b200_sxm",
                "vllm",
                "0.24.0",
                "agg",
                STATUS_PASS,
                "",
                "uv run aiconfigurator cli default --database-mode SILICON",
                "silicon",
            ]
        )

    entries = load_entries(tmp_path)

    assert [entry.model for entry in entries] == ["Qwen/Qwen3-8B"]
    assert entries[0].image_workload is None
    assert "text backbones were not parity-certified" in caplog.text


def test_parity_probe_passes_image_arguments(monkeypatch):
    calls = []

    def fake_estimate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            ttft=10.0,
            tpot=2.0,
            raw={"encoder_latency": 1.25, "encoder_memory": 0.5},
        )

    monkeypatch.setattr("aiconfigurator.cli.api.cli_estimate", fake_estimate)

    ttft, tpot, error = _run_probe(
        _entry(),
        TestConstraints(total_gpus=4, isl=256, osl=256, prefix=128, ttft=1500.0, tpot=50.0),
        "python",
    )

    assert (ttft, tpot, error) == (10.0, 2.0, None)
    assert calls[0]["image_height"] == 1024
    assert calls[0]["image_width"] == 1024
    assert calls[0]["num_images"] == 1


def test_parity_pareto_runs_both_engines_with_encoder_evidence(monkeypatch):
    calls = []

    def fake_run_mode(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame(
            {
                "encoder_latency": [1.25],
                "encoder_memory": [0.5],
                "request_rate": [1.0],
            }
        )

    monkeypatch.setattr("tools.support_matrix.scan_rust_parity._get_worker_matrix", lambda: None)
    monkeypatch.setattr(
        "tools.support_matrix.scan_rust_parity._get_test_constraints",
        lambda _model: TestConstraints(4, 256, 256, 128, 1500.0, 50.0),
    )
    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))

    record = pareto_entry(_entry())

    assert record.comparison_outcome == PARETO_STATUS_STRICT_PASS
    assert [call["engine_step_backend"] for call in calls] == ["python", "rust"]
    assert all(call["image_workload"] == SUPPORT_MATRIX_IMAGE_WORKLOAD for call in calls)

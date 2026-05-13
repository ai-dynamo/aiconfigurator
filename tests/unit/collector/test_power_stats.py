# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import sys
from pathlib import Path

import pytest

_COLLECTOR_DIR = str(Path(__file__).resolve().parents[3] / "collector")
if _COLLECTOR_DIR not in sys.path:
    sys.path.insert(0, _COLLECTOR_DIR)

from helper import log_perf, summarize_power_samples


def test_summarize_power_samples_integrates_energy_with_trapezoids():
    stats = summarize_power_samples(
        [
            (2.0, 100_000),
            (0.0, 100_000),
            (1.0, 200_000),
        ],
        power_limit_mw=700_000,
    )

    assert stats == {
        "power": 150.0,
        "power_limit": 700.0,
        "power_min": 100.0,
        "power_max": 200.0,
        "power_energy_j": 300.0,
        "power_duration_s": 2.0,
        "power_sample_count": 3,
    }


def test_summarize_power_samples_uses_arithmetic_mean_without_duration():
    stats = summarize_power_samples([(10.0, 125_000)])

    assert stats["power"] == 125.0
    assert stats["power_limit"] is None
    assert stats["power_min"] == 125.0
    assert stats["power_max"] == 125.0
    assert stats["power_energy_j"] == 0.0
    assert stats["power_duration_s"] == 0.0
    assert stats["power_sample_count"] == 1


def test_summarize_power_samples_empty_returns_none():
    assert summarize_power_samples([]) is None


def test_log_perf_writes_power_analysis_fields(tmp_path):
    perf_path = tmp_path / "perf.csv"
    log_perf(
        item_list=[{"latency": 3.5}],
        framework="vllm",
        version="0.19.0",
        device_name="H100",
        op_name="gemm",
        kernel_source="triton",
        perf_filename=str(perf_path),
        power_stats={
            "power": 150.0,
            "power_limit": 700.0,
            "power_min": 100.0,
            "power_max": 200.0,
            "power_energy_j": 300.0,
            "power_duration_s": 2.0,
            "power_sample_count": 3,
        },
    )

    with perf_path.open() as f:
        rows = list(csv.DictReader(f))

    assert rows == [
        {
            "framework": "vllm",
            "version": "0.19.0",
            "device": "H100",
            "op_name": "gemm",
            "kernel_source": "triton",
            "latency": "3.5",
            "power": "150.0",
            "power_limit": "700.0",
            "power_min": "100.0",
            "power_max": "200.0",
            "power_energy_j": "300.0",
            "power_duration_s": "2.0",
            "power_sample_count": "3",
        }
    ]


@pytest.mark.parametrize("power_stats", [None, {}])
def test_log_perf_preserves_legacy_columns_without_power_stats(tmp_path, power_stats):
    perf_path = tmp_path / "perf.csv"
    log_perf(
        item_list=[{"latency": 3.5}],
        framework="vllm",
        version="0.19.0",
        device_name="H100",
        op_name="gemm",
        kernel_source="triton",
        perf_filename=str(perf_path),
        power_stats=power_stats,
    )

    with perf_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert reader.fieldnames == [
        "framework",
        "version",
        "device",
        "op_name",
        "kernel_source",
        "latency",
    ]
    assert rows[0]["latency"] == "3.5"

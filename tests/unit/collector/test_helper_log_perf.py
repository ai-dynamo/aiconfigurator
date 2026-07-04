# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv

from collector import helper


def _log_perf(perf_filename: str) -> bool:
    return helper.log_perf(
        item_list=[{"batch_size": 1, "latency": "1.25"}],
        framework="SGLang",
        version="0.5.14",
        device_name="Fake GPU",
        op_name="mla_context_module",
        kernel_source="mla_fa3",
        perf_filename=perf_filename,
    )


def test_log_perf_returns_true_after_durable_write(tmp_path):
    perf_path = tmp_path / "mla_perf.txt"

    assert _log_perf(str(perf_path)) is True
    with perf_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == [
        {
            "framework": "SGLang",
            "version": "0.5.14",
            "device": "Fake GPU",
            "op_name": "mla_context_module",
            "kernel_source": "mla_fa3",
            "batch_size": "1",
            "latency": "1.25",
        }
    ]


def test_log_perf_returns_false_when_lock_is_held(tmp_path, monkeypatch):
    perf_path = tmp_path / "mla_perf.txt"
    lock_path = tmp_path / "mla_perf.txt.lock"
    lock_path.touch()
    monkeypatch.setattr(helper.time, "sleep", lambda _seconds: None)

    assert _log_perf(str(perf_path)) is False
    assert not perf_path.exists()
    assert lock_path.exists()


def test_log_perf_returns_false_and_releases_lock_on_fsync_failure(tmp_path, monkeypatch):
    perf_path = tmp_path / "mla_perf.txt"
    lock_path = tmp_path / "mla_perf.txt.lock"

    def fail_fsync(_fd):
        raise OSError("fsync failed")

    monkeypatch.setattr(helper.os, "fsync", fail_fsync)

    assert _log_perf(str(perf_path)) is False
    assert not lock_path.exists()

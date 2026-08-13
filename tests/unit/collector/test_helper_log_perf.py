# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import os
import time
from types import SimpleNamespace

import pytest

from collector import helper
from collector.helper import PowerMonitor

pytestmark = pytest.mark.unit


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


def _make_stale_lock(lock_path):
    lock_path.touch()
    stale = time.time() - 120
    os.utime(lock_path, (stale, stale))


def test_log_perf_breaks_stale_lock_via_rename_and_writes(tmp_path, monkeypatch):
    perf_path = tmp_path / "mla_perf.txt"
    lock_path = tmp_path / "mla_perf.txt.lock"
    _make_stale_lock(lock_path)
    monkeypatch.setattr(helper.time, "sleep", lambda _seconds: None)

    assert _log_perf(str(perf_path)) is True
    assert perf_path.exists()
    assert not lock_path.exists()
    assert not list(tmp_path.glob("*.breaking-*"))


def test_log_perf_losing_breaker_never_unlinks_the_fresh_lock(tmp_path, monkeypatch):
    # Two-waiter break race: this waiter stats a stale lock, but by the time
    # it breaks, a winner has already renamed the stale lock away and a
    # sibling holds a FRESH lock at the same path. The loser's rename raises;
    # it must retry against the fresh lock (and time out) — never unlink it.
    # The retired unlink-based breaker failed exactly this: it removed the
    # fresh lock and let two writers interleave appends.
    perf_path = tmp_path / "mla_perf.txt"
    lock_path = tmp_path / "mla_perf.txt.lock"
    _make_stale_lock(lock_path)
    monkeypatch.setattr(helper.time, "sleep", lambda _seconds: None)

    state = {"raced": False}
    real_rename = os.rename

    def racing_rename(src, dst):
        if not state["raced"]:
            state["raced"] = True
            os.unlink(src)  # the winning breaker took the stale lock...
            lock_path.touch()  # ...and a sibling immediately re-acquired
            raise FileNotFoundError(src)
        return real_rename(src, dst)

    monkeypatch.setattr(helper.os, "rename", racing_rename)

    assert _log_perf(str(perf_path)) is False
    assert lock_path.exists()
    assert not perf_path.exists()


def _log_perf_with_power(perf_filename: str, power_stats) -> bool:
    return helper.log_perf(
        item_list=[{"batch_size": 1, "latency": "1.25"}],
        framework="SGLang",
        version="0.5.14",
        device_name="Fake GPU",
        op_name="moe",
        kernel_source="sglang_fused_moe_triton",
        perf_filename=perf_filename,
        power_stats=power_stats,
    )


_POWER_STATS = {"power": 450.0, "power_limit": 1000.0}


def test_log_perf_power_row_then_missing_power_row_consistent_columns(tmp_path, monkeypatch):
    """
    Regression: power-stats row written first, then a row with power_stats=None.
    Both rows must have power/power_limit columns so pyarrow can parse the file.
    Reproduces the sglang_fused_moe_triton bfloat16 smoke crash (2026-08-03).
    """
    monkeypatch.setenv("COLLECTOR_MEASURE_POWER", "true")
    perf_path = str(tmp_path / "moe_perf.txt")

    assert _log_perf_with_power(perf_path, _POWER_STATS) is True
    assert _log_perf_with_power(perf_path, None) is True

    with open(perf_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["power"] == "450.0"
    assert rows[0]["power_limit"] == "1000.0"
    # Second row must have power column present (empty, not absent)
    assert "power" in rows[1]
    assert rows[1]["power"] == ""
    assert rows[1]["power_limit"] == ""


def test_power_monitor_rejects_none_device_id():
    """
    Regression: torch.device("cuda").index returns None; PowerMonitor(None) previously
    called nvml.nvmlDeviceGetHandleByIndex(None), silently failed _init_handle(), and
    returned power_stats=None — producing a zero-power row that bypassed the smoke gate.
    Fix: PowerMonitor.__init__ raises TypeError on None so the bug is immediately visible
    (benchmark_config must use torch.cuda.current_device(), not torch.device("cuda").index).
    """
    with pytest.raises(TypeError, match="explicit integer device index"):
        PowerMonitor(None)


def test_power_monitor_rejects_noninteger_device_id():
    with pytest.raises(TypeError, match="explicit integer device index"):
        PowerMonitor("cuda:0")


def test_log_perf_missing_power_row_then_power_row_consistent_columns(tmp_path, monkeypatch):
    """
    Regression: row with power_stats=None written first, then a power-stats row.
    The header must include power columns from the start so the second row matches.
    """
    monkeypatch.setenv("COLLECTOR_MEASURE_POWER", "true")
    perf_path = str(tmp_path / "moe_perf.txt")

    assert _log_perf_with_power(perf_path, None) is True
    assert _log_perf_with_power(perf_path, _POWER_STATS) is True

    with open(perf_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    # First row has empty power (zero-sample event recorded, not silently dropped)
    assert rows[0]["power"] == ""
    assert rows[0]["power_limit"] == ""
    # Second row has actual values
    assert rows[1]["power"] == "450.0"
    assert rows[1]["power_limit"] == "1000.0"


def test_log_perf_upgrades_existing_power_off_csv_before_power_append(tmp_path, monkeypatch):
    """A resumed power run must not append wider rows under a legacy header."""
    monkeypatch.delenv("COLLECTOR_MEASURE_POWER", raising=False)
    perf_path = str(tmp_path / "moe_perf.txt")

    assert _log_perf_with_power(perf_path, None) is True
    with open(perf_path, newline="") as f:
        assert "power" not in next(csv.reader(f))

    monkeypatch.setenv("COLLECTOR_MEASURE_POWER", "true")
    assert _log_perf_with_power(perf_path, _POWER_STATS) is True

    with open(perf_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["power"] == ""
    assert rows[0]["power_limit"] == ""
    assert rows[1]["power"] == "450.0"
    assert rows[1]["power_limit"] == "1000.0"
    assert helper.convert_perf_csv_to_parquet(perf_path, delete_source=False).exists()


def test_aggregate_latency_weighted_power_is_energy_equivalent():
    result = helper.aggregate_latency_weighted_power(
        [
            (1.0, {"power": 100.0, "power_limit": 1000.0}),
            (3.0, {"power": 300.0, "power_limit": 1000.0}),
        ]
    )

    assert result == {"power": 250.0, "power_limit": 1000.0}


def test_aggregate_latency_weighted_power_rejects_partial_samples():
    with pytest.raises(RuntimeError, match="mixture of present and missing"):
        helper.aggregate_latency_weighted_power([(1.0, _POWER_STATS), (2.0, None)])


def test_zero_work_power_stats_uses_measured_limit(monkeypatch):
    class FakePowerMonitor:
        def __init__(self, device_id):
            assert device_id == 3

        def get_power_limit(self):
            return 1000.0

    monkeypatch.setattr(helper, "PowerMonitor", FakePowerMonitor)
    monkeypatch.setenv("COLLECTOR_MEASURE_POWER", "true")

    assert helper.zero_work_power_stats(SimpleNamespace(index=3)) == {
        "power": 0.0,
        "power_limit": 1000.0,
    }

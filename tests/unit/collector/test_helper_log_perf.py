# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import builtins
import csv
import os
import time

import pytest

from collector import helper

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


@pytest.mark.parametrize("first_has_power", [False, True])
def test_log_perf_rejects_optional_column_schema_changes_before_append(tmp_path, first_has_power):
    perf_path = tmp_path / "mla_perf.txt"
    kwargs = {
        "item_list": [{"batch_size": 1, "latency": "1.25"}],
        "framework": "SGLang",
        "version": "0.5.14",
        "device_name": "Fake GPU",
        "op_name": "mla_context_module",
        "kernel_source": "mla_fa3",
        "perf_filename": str(perf_path),
    }
    power_stats = {"power": 400.0, "power_limit": 700.0}

    assert helper.log_perf(**kwargs, power_stats=power_stats if first_has_power else None)
    original = perf_path.read_bytes()

    with pytest.raises(helper.PerfLogError, match="Schema mismatch"):
        helper.log_perf(**kwargs, power_stats=None if first_has_power else power_stats)
    assert perf_path.read_bytes() == original
    assert not perf_path.with_suffix(".txt.lock").exists()


def test_log_perf_raises_when_lock_is_held(tmp_path, monkeypatch):
    perf_path = tmp_path / "mla_perf.txt"
    lock_path = tmp_path / "mla_perf.txt.lock"
    lock_path.touch()
    monkeypatch.setattr(helper.time, "sleep", lambda _seconds: None)

    with pytest.raises(helper.PerfLogError, match="Can not get lock"):
        _log_perf(str(perf_path))
    assert not perf_path.exists()
    assert lock_path.exists()


def test_log_perf_raises_and_releases_lock_on_fsync_failure(tmp_path, monkeypatch):
    perf_path = tmp_path / "mla_perf.txt"
    lock_path = tmp_path / "mla_perf.txt.lock"

    def fail_fsync(_fd):
        raise OSError("fsync failed")

    monkeypatch.setattr(helper.os, "fsync", fail_fsync)

    with pytest.raises(helper.PerfLogError, match="fsync failed"):
        _log_perf(str(perf_path))
    assert not lock_path.exists()


def test_log_perf_never_reopens_reactivated_retained_file_by_path(tmp_path, monkeypatch):
    perf_path = tmp_path / "mla_perf.txt"
    retained_path = helper.collector_retained_path(perf_path)
    retained_path.write_bytes(b"retained staging")
    foreign_path = tmp_path / "foreign.csv"
    foreign_bytes = (
        b"framework,version,device,op_name,kernel_source,batch_size,latency\nFOREIGN,0,foreign,foreign,foreign,99,99\n"
    )
    foreign_path.write_bytes(foreign_bytes)
    displaced_path = tmp_path / "displaced-owned.csv"
    swapped = False
    real_open = builtins.open
    real_fdopen = helper.os.fdopen

    def swap_before_writer_open() -> None:
        nonlocal swapped
        if swapped:
            return
        swapped = True
        perf_path.rename(displaced_path)
        foreign_path.rename(perf_path)

    def racing_open(file, mode="r", *args, **kwargs):
        if os.fspath(file) == str(perf_path) and mode == "a+":
            swap_before_writer_open()
        return real_open(file, mode, *args, **kwargs)

    def racing_fdopen(fd, mode="r", *args, **kwargs):
        if mode == "r+":
            swap_before_writer_open()
        return real_fdopen(fd, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", racing_open)
    monkeypatch.setattr(helper.os, "fdopen", racing_fdopen)

    with pytest.raises(helper.PerfLogError, match="Retained performance file changed"):
        _log_perf(str(perf_path))

    assert swapped
    assert perf_path.read_bytes() == foreign_bytes
    assert displaced_path.is_file()


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

    with pytest.raises(helper.PerfLogError, match="Can not get lock"):
        _log_perf(str(perf_path))
    assert lock_path.exists()
    assert not perf_path.exists()

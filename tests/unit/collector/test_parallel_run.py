# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for parallel_run sentinel tracking and multiprocessing robustness.

Covers:
- Normal task completion
- EXIT_CODE_RESTART mid-task (worker dies, gets restarted)
- Regular exceptions (worker stays alive, error recorded)
- Mixed failure modes
- Sentinel balance under repeated restarts (the core bug-fix scenario)
"""

import json
import logging
import multiprocessing as mp
import os
import signal
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Must be set before any fork() on macOS to avoid Obj-C runtime crashes.
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import pytest

_HAS_FORK = hasattr(os, "fork")
pytestmark_fork = pytest.mark.skipif(
    not _HAS_FORK,
    reason="These tests require the 'fork' multiprocessing context (not available on Windows)",
)

# ---------------------------------------------------------------------------
# Bootstrap: mock torch so collect.py can be imported without CUDA.
# Must happen BEFORE collect.py is imported.
# ---------------------------------------------------------------------------
_COLLECTOR_DIR = str(Path(__file__).resolve().parents[3] / "collector")
if _COLLECTOR_DIR not in sys.path:
    sys.path.insert(0, _COLLECTOR_DIR)

if "torch" not in sys.modules:
    _torch = MagicMock()
    _torch.AcceleratorError = type("AcceleratorError", (Exception,), {})
    sys.modules["torch"] = _torch

import collect as _collect_mod
from collect import parallel_run

_collect_mod.logger = logging.getLogger("test_parallel_run")
_collect_mod.logger.setLevel(logging.DEBUG)
_collect_mod.logger.addHandler(logging.StreamHandler(sys.stderr))

EXIT_CODE_RESTART = 10

pytestmark = [pytest.mark.unit, pytestmark_fork]


# ---------------------------------------------------------------------------
# Task function — module-level so fork'd workers can resolve it.
# ---------------------------------------------------------------------------
def _task_fn(label, behavior, device):
    """Dispatch based on *behavior* encoded in each task's params."""
    if behavior == "exit_restart":
        sys.exit(EXIT_CODE_RESTART)
    elif behavior == "sigabrt":
        os.kill(os.getpid(), signal.SIGABRT)
    elif behavior == "error":
        raise ValueError(f"simulated: {label}")
    # "normal": return silently


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _fork_mp(monkeypatch):
    """Replace mp in collect module with a fork context so that the mocked
    ``torch`` module (and other parent-process state) propagates to workers."""
    import warnings

    warnings.filterwarnings("ignore", message=".*fork.*", category=DeprecationWarning)
    ctx = mp.get_context("fork")
    monkeypatch.setattr(_collect_mod, "mp", ctx)


@pytest.fixture(autouse=True)
def _fast_poll(monkeypatch):
    """Shrink the 2 s monitoring-loop sleep so tests finish faster."""
    _original = _collect_mod.time.sleep

    def _short(seconds):
        _original(min(seconds, 0.15))

    monkeypatch.setattr(_collect_mod.time, "sleep", _short)


@pytest.fixture(autouse=True)
def _log_dir(tmp_path, monkeypatch):
    """Redirect COLLECTOR_LOG_DIR so error-report files go to a temp dir."""
    monkeypatch.setenv("COLLECTOR_LOG_DIR", str(tmp_path))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run(tasks, num_processes, tmp_path, module_name="test"):
    return parallel_run(
        tasks,
        _task_fn,
        num_processes=num_processes,
        module_name=module_name,
        resume_options={"checkpoint_dir": str(tmp_path / ".checkpoint")},
    )


def _checkpoint_path(tmp_path, module_name, backend="unknown"):
    safe_name = module_name.replace("/", "_").replace(":", "_")
    return tmp_path / ".checkpoint" / backend / f"{safe_name}.json"


def _load_done_ids(tmp_path, module_name, backend="unknown"):
    checkpoint = _checkpoint_path(tmp_path, module_name, backend=backend)
    assert checkpoint.exists(), f"checkpoint not found: {checkpoint}"
    with checkpoint.open() as f:
        data = json.load(f)
    return set(data.get("done", []))


def _assert_all_tasks_attempted(tasks, tmp_path, module_name):
    expected = {task["id"] for task in tasks}
    actual = _load_done_ids(tmp_path, module_name)
    missing = expected - actual
    extra = actual - expected
    assert actual == expected, f"done mismatch: missing={missing}, extra={extra}"


def _run_and_assert_all_done(tasks, num_processes, tmp_path, module_name):
    errors = _run(tasks, num_processes, tmp_path, module_name=module_name)
    _assert_all_tasks_attempted(tasks, tmp_path, module_name)
    return errors


def _tasks(specs):
    """Build a task list.

    *specs* is either an int (N normal tasks) or a list of
    ``(label, behavior)`` tuples.
    """
    if isinstance(specs, int):
        return [{"id": f"t{i}", "params": (f"t{i}", "normal")} for i in range(specs)]
    return [{"id": label, "params": (label, beh)} for label, beh in specs]


def _crash_errors(errors):
    return [e for e in errors if e.get("error_type") in ("WorkerSignalCrash", "WorkerAbnormalExit")]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestNormalCompletion:
    """Baseline: all tasks succeed, no restarts needed."""

    def test_two_workers(self, tmp_path):
        tasks = _tasks(8)
        assert _run_and_assert_all_done(tasks, 2, tmp_path, module_name="normal_two_workers") == []

    def test_single_worker(self, tmp_path):
        tasks = _tasks(4)
        assert _run_and_assert_all_done(tasks, 1, tmp_path, module_name="normal_single_worker") == []


class TestExitCodeRestart:
    """Workers that sys.exit(EXIT_CODE_RESTART) mid-task get restarted.
    No surplus sentinel should be injected."""

    def test_every_task_triggers_restart(self, tmp_path):
        tasks = _tasks([(f"t{i}", "exit_restart") for i in range(6)])
        errors = _run_and_assert_all_done(tasks, 2, tmp_path, module_name="restart_all")
        assert _crash_errors(errors) == []

    def test_interleaved_restart_and_normal(self, tmp_path):
        tasks = _tasks(
            [
                ("a", "exit_restart"),
                ("b", "normal"),
                ("c", "exit_restart"),
                ("d", "normal"),
                ("e", "normal"),
                ("f", "exit_restart"),
                ("g", "normal"),
                ("h", "normal"),
            ]
        )
        errors = _run_and_assert_all_done(tasks, 2, tmp_path, module_name="restart_interleaved")
        assert _crash_errors(errors) == []


class TestTaskExceptions:
    """Regular exceptions: worker stays alive, error is recorded, next task
    is processed normally."""

    def test_all_fail(self, tmp_path):
        tasks = _tasks([(f"t{i}", "error") for i in range(4)])
        errors = _run_and_assert_all_done(tasks, 2, tmp_path, module_name="all_fail")
        assert len([e for e in errors if e.get("error_type") == "ValueError"]) == 4

    def test_mixed_success_and_fail(self, tmp_path):
        tasks = _tasks(
            [
                ("a", "normal"),
                ("b", "error"),
                ("c", "normal"),
                ("d", "error"),
                ("e", "normal"),
            ]
        )
        errors = _run_and_assert_all_done(tasks, 2, tmp_path, module_name="mixed_success_fail")
        assert len([e for e in errors if e.get("error_type") == "ValueError"]) == 2


class TestMixedFailureModes:
    """Combine EXIT_CODE_RESTART, exceptions, and normal tasks."""

    def test_restart_and_exception_combined(self, tmp_path):
        tasks = _tasks(
            [
                ("a", "normal"),
                ("b", "exit_restart"),
                ("c", "error"),
                ("d", "normal"),
                ("e", "exit_restart"),
                ("f", "error"),
                ("g", "normal"),
                ("h", "normal"),
            ]
        )
        errors = _run_and_assert_all_done(tasks, 2, tmp_path, module_name="restart_and_exception")
        assert len([e for e in errors if e.get("error_type") == "ValueError"]) == 2
        assert _crash_errors(errors) == []


class TestSentinelBalance:
    """Stress-test sentinel tracking under repeated restarts.

    Under the old (buggy) code, each EXIT_CODE_RESTART added a surplus
    sentinel.  With enough restarts the extra sentinels would kill live
    workers, stranding unfinished tasks and causing a hang.

    The fix adds a sentinel only when the dead worker had actually consumed
    its original one, keeping the count balanced.
    """

    def test_many_restarts_two_workers(self, tmp_path):
        """12 consecutive exit_restart tasks x 2 workers.
        Old code would inject 12 surplus sentinels."""
        tasks = _tasks([(f"t{i}", "exit_restart") for i in range(12)])
        errors = _run_and_assert_all_done(tasks, 2, tmp_path, module_name="many_restarts_two_workers")
        assert _crash_errors(errors) == []

    def test_many_restarts_single_worker(self, tmp_path):
        """8 consecutive exit_restart tasks x 1 worker.
        Single worker means the surplus sentinel would be consumed by the
        same worker on next restart, causing an infinite restart loop in
        the old code."""
        tasks = _tasks([(f"t{i}", "exit_restart") for i in range(8)])
        errors = _run_and_assert_all_done(tasks, 1, tmp_path, module_name="many_restarts_single_worker")
        assert _crash_errors(errors) == []

    def test_heavy_mixed_stress(self, tmp_path):
        """20 tasks with alternating failure modes across 3 workers."""
        tasks = _tasks([(f"t{i}", ["normal", "exit_restart", "error"][i % 3]) for i in range(20)])
        errors = _run_and_assert_all_done(tasks, 3, tmp_path, module_name="heavy_mixed_stress")
        assert _crash_errors(errors) == []
        n_val = len([e for e in errors if e.get("error_type") == "ValueError"])
        expected_errors = sum(1 for i in range(20) if i % 3 == 2)
        assert n_val == expected_errors


class TestSignalCrashRecovery:
    """Fatal signal exits should be accounted for exactly once."""

    def test_sigabrt_tasks_are_marked_done(self, tmp_path):
        tasks = _tasks(
            [
                ("a", "normal"),
                ("b", "sigabrt"),
                ("c", "normal"),
                ("d", "sigabrt"),
                ("e", "normal"),
            ]
        )
        errors = _run_and_assert_all_done(tasks, 2, tmp_path, module_name="sigabrt_done")
        assert len([e for e in errors if e.get("error_type") == "WorkerSignalCrash"]) >= 2

    def test_sigabrt_and_restart_mix(self, tmp_path):
        tasks = _tasks(
            [
                ("a", "sigabrt"),
                ("b", "exit_restart"),
                ("c", "normal"),
                ("d", "sigabrt"),
                ("e", "exit_restart"),
                ("f", "normal"),
            ]
        )
        errors = _run_and_assert_all_done(tasks, 2, tmp_path, module_name="sigabrt_restart_mix")
        assert len([e for e in errors if e.get("error_type") == "WorkerSignalCrash"]) >= 2

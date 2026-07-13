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
    elif behavior == "oom":
        raise sys.modules["torch"].OutOfMemoryError(f"simulated: {label}")
    # "normal": return silently


class _GroupedCase:
    """Fake case for failure-group aggregation tests.

    Carries ``model_name``/``dtype`` attributes so ``_failure_group`` labels
    it, and unpacks (via ``func(*task)``) into ``_task_fn`` args that always
    raise. Module-level so fork'd workers can unpickle it.
    """

    def __init__(self, label, model_name="test/always-fails", dtype="fp8"):
        self.label = label
        self.model_name = model_name
        self.dtype = dtype

    def __iter__(self):
        return iter((self.label, "error"))

    def __str__(self):
        return f"_GroupedCase({self.label}, model={self.model_name}, dtype={self.dtype})"


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


def _load_checkpoint_data(tmp_path, module_name, backend="unknown"):
    checkpoint = _checkpoint_path(tmp_path, module_name, backend=backend)
    assert checkpoint.exists(), f"checkpoint not found: {checkpoint}"
    with checkpoint.open() as f:
        return json.load(f)


def _load_done_ids(tmp_path, module_name, backend="unknown"):
    data = _load_checkpoint_data(tmp_path, module_name, backend=backend)
    return set(data.get("done", []))


def _load_failed_ids(tmp_path, module_name, backend="unknown"):
    data = _load_checkpoint_data(tmp_path, module_name, backend=backend)
    return set(data.get("failed", []))


def _assert_all_tasks_attempted(tasks, tmp_path, module_name):
    expected = {task["id"] for task in tasks}
    done = _load_done_ids(tmp_path, module_name)
    failed = _load_failed_ids(tmp_path, module_name)
    attempted = done | failed
    missing = expected - attempted
    extra = attempted - expected
    assert attempted == expected, f"attempted mismatch: missing={missing}, extra={extra}"


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


class TestCudaFatalExceptionDetection:
    def test_torch_accelerator_error_is_fatal(self):
        torch_mod = MagicMock()
        torch_mod.AcceleratorError = type("AcceleratorError", (Exception,), {})

        assert _collect_mod._is_cuda_fatal_exception(torch_mod.AcceleratorError("boom"), torch_mod)

    def test_torch_out_of_memory_error_is_fatal(self):
        torch_mod = MagicMock()
        torch_mod.OutOfMemoryError = type("OutOfMemoryError", (Exception,), {})

        assert _collect_mod._is_cuda_fatal_exception(torch_mod.OutOfMemoryError("boom"), torch_mod)

    @pytest.mark.parametrize(
        "message",
        [
            "CUDA error: an illegal memory access was encountered",
            "cuda error: unspecified launch failure",
            "CUDA_ERROR_LAUNCH_FAILED",
            "CUBLAS_STATUS_EXECUTION_FAILED",
            "CUBLAS_STATUS_INTERNAL_ERROR",
            "CUBLAS_STATUS_ALLOC_FAILED",
        ],
    )
    def test_cuda_fatal_markers_are_fatal(self, message):
        torch_mod = MagicMock()
        torch_mod.AcceleratorError = type("AcceleratorError", (Exception,), {})

        assert _collect_mod._is_cuda_fatal_exception(RuntimeError(message), torch_mod)

    def test_dsl_cuda_runtime_error_is_fatal(self):
        torch_mod = MagicMock()
        torch_mod.AcceleratorError = type("AcceleratorError", (Exception,), {})
        exc_cls = type("DSLCudaRuntimeError", (RuntimeError,), {})

        assert _collect_mod._is_cuda_fatal_exception(exc_cls("context corrupted"), torch_mod)

    def test_non_cuda_exception_is_not_fatal(self):
        torch_mod = MagicMock()
        torch_mod.AcceleratorError = type("AcceleratorError", (Exception,), {})

        assert not _collect_mod._is_cuda_fatal_exception(ValueError("plain failure"), torch_mod)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestNormalCompletion:
    """Baseline: all tasks succeed, no restarts needed."""

    def test_two_workers(self, tmp_path):
        tasks = _tasks(8)
        assert _run_and_assert_all_done(tasks, 2, tmp_path, module_name="normal_two_workers") == []
        assert _load_failed_ids(tmp_path, "normal_two_workers") == set()

    def test_single_worker(self, tmp_path):
        tasks = _tasks(4)
        assert _run_and_assert_all_done(tasks, 1, tmp_path, module_name="normal_single_worker") == []
        assert _load_failed_ids(tmp_path, "normal_single_worker") == set()


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
        assert _load_done_ids(tmp_path, "all_fail") == set()
        assert _load_failed_ids(tmp_path, "all_fail") == {f"t{i}" for i in range(4)}

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
        assert _load_done_ids(tmp_path, "mixed_success_fail") == {"a", "c", "e"}
        assert _load_failed_ids(tmp_path, "mixed_success_fail") == {"b", "d"}

    def test_task_failures_are_classified_unexpected(self, tmp_path):
        tasks = _tasks([("a", "error"), ("b", "normal")])
        errors = _run_and_assert_all_done(tasks, 1, tmp_path, module_name="failure_classification")
        task_errors = [e for e in errors if e.get("error_type") == "ValueError"]
        assert len(task_errors) == 1
        assert task_errors[0]["classification"] == "unexpected"
        # Plain tuple tasks carry no model/dtype attributes: no group label.
        assert task_errors[0]["group"] is None

    def test_oom_records_once_and_restarts_worker(self, tmp_path, monkeypatch):
        oom_error = type("OutOfMemoryError", (Exception,), {})
        monkeypatch.setattr(sys.modules["torch"], "OutOfMemoryError", oom_error, raising=False)
        tasks = _tasks([("oom", "oom"), ("after", "normal")])

        errors = _run_and_assert_all_done(tasks, 1, tmp_path, module_name="oom_restart")

        assert len([error for error in errors if error.get("error_type") == "OutOfMemoryError"]) == 1
        assert _crash_errors(errors) == []
        assert _load_done_ids(tmp_path, "oom_restart") == {"after"}
        assert _load_failed_ids(tmp_path, "oom_restart") == {"oom"}


class TestFailureGroups:
    """Systemic (model, dtype) group failures are all run and all labeled."""

    def test_failing_group_runs_every_case_and_labels_each_failure(self, tmp_path):
        n_tasks = 8
        tasks = [{"id": f"gr{i}", "params": _GroupedCase(f"gr{i}")} for i in range(n_tasks)]

        errors = _run(tasks, 1, tmp_path, module_name="failure_group")

        _assert_all_tasks_attempted(tasks, tmp_path, "failure_group")

        task_failures = [e for e in errors if e.get("error_type") == "ValueError"]
        # No breaker: every case runs, every failure is recorded and labeled.
        assert len(task_failures) == n_tasks
        assert all(e["classification"] == "unexpected" for e in task_failures)
        assert all(e["group"] == "test/always-fails|fp8" for e in task_failures)

        assert _load_done_ids(tmp_path, "failure_group") == set()
        assert _load_failed_ids(tmp_path, "failure_group") == {f"gr{i}" for i in range(n_tasks)}


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
        done = _load_done_ids(tmp_path, "restart_and_exception")
        failed = _load_failed_ids(tmp_path, "restart_and_exception")
        assert {"a", "b", "d", "e", "g", "h"} == done  # normal + exit_restart = passed
        assert {"c", "f"} == failed  # error = failed


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

    def test_sigabrt_tasks_are_tracked(self, tmp_path):
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
        done = _load_done_ids(tmp_path, "sigabrt_done")
        failed = _load_failed_ids(tmp_path, "sigabrt_done")
        assert {"a", "c", "e"} == done
        assert {"b", "d"} == failed

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
        done = _load_done_ids(tmp_path, "sigabrt_restart_mix")
        failed = _load_failed_ids(tmp_path, "sigabrt_restart_mix")
        assert {"b", "c", "e", "f"} == done  # exit_restart + normal = passed
        assert {"a", "d"} == failed  # sigabrt = failed


class TestResumeIntegrity:
    """Checkpoint binds runtime identity; resume surfaces unresolved failures."""

    def test_checkpoint_binds_framework_version_and_sm(self, tmp_path):
        kwargs = {
            "backend": "unknown",
            "module_name": "bind",
            "run_func_name": "f",
            "checkpoint_dir": str(tmp_path / ".checkpoint"),
        }
        first = _collect_mod.ResumeCheckpoint(framework_version="1.0.0", sm_version=90, **kwargs)
        first.mark_failed("t1")
        first.flush(force=True)

        version_changed = _collect_mod.ResumeCheckpoint(framework_version="2.0.0", sm_version=90, **kwargs)
        with pytest.raises(RuntimeError, match="framework_version"):
            version_changed.load_existing()

        sm_changed = _collect_mod.ResumeCheckpoint(framework_version="1.0.0", sm_version=100, **kwargs)
        with pytest.raises(RuntimeError, match="sm_version"):
            sm_changed.load_existing()

    def test_resume_reports_unresolved_failures(self, tmp_path):
        tasks = _tasks([("a", "error"), ("b", "normal")])
        _run(tasks, 1, tmp_path, module_name="resume_unresolved")

        # All tasks are skipped on resume (a failed, b passed) — the run must
        # still surface the unresolved failure instead of reporting clean.
        resumed = parallel_run(
            tasks,
            _task_fn,
            num_processes=1,
            module_name="resume_unresolved",
            resume_options={"checkpoint_dir": str(tmp_path / ".checkpoint"), "resume": True},
        )
        unresolved = [e for e in resumed if e.get("error_type") == "UnresolvedFailures"]
        assert len(unresolved) == 1
        assert unresolved[0]["classification"] == "unresolved_from_checkpoint"
        assert "1 unresolved" in unresolved[0]["error_message"]

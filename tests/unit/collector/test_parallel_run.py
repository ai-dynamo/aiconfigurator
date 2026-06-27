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
import types
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

from collector.model_cases import CaseSelector, OpCasePlan
from collector.planner import PopulationRule, RuleSource

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
    elif behavior == "exit_error":
        sys.exit(7)
    elif behavior == "sigabrt":
        os.kill(os.getpid(), signal.SIGABRT)
    elif behavior == "error":
        raise ValueError(f"simulated: {label}")
    elif behavior == "expected_error":
        raise RuntimeError(f"expected simulated: {label}")
    # "normal": return silently


def _exit_after_success_status_before_progress(
    queue,
    device_id,
    func,
    progress_value,
    lock,
    error_queue=None,
    done_tasks=None,
    failed_tasks=None,
    expected_failed_tasks=None,
    module_name="unknown",
    current_task_ids=None,
    consumed_sentinel=None,
    expected_failure_context=None,
    accounted_tasks=None,
):
    """Emulate a hard exit in the old done/clear/progress race window."""

    del progress_value, lock, error_queue, failed_tasks, expected_failed_tasks
    del module_name, consumed_sentinel, expected_failure_context, accounted_tasks
    task_info = queue.get()
    task_id = task_info["id"]
    if current_task_ids is not None:
        current_task_ids[device_id] = task_id
    func(*task_info["params"], device=device_id)
    done_tasks[task_id] = True
    if current_task_ids is not None:
        current_task_ids[device_id] = None
    os._exit(0)


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
def _run(tasks, num_processes, tmp_path, module_name="test", expected_failure_context=None):
    return parallel_run(
        tasks,
        _task_fn,
        num_processes=num_processes,
        module_name=module_name,
        resume_options={"checkpoint_dir": str(tmp_path / ".checkpoint")},
        expected_failure_context=expected_failure_context,
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


def _load_expected_failed_ids(tmp_path, module_name, backend="unknown"):
    data = _load_checkpoint_data(tmp_path, module_name, backend=backend)
    return set(data.get("expected_failed", []))


def _assert_all_tasks_attempted(tasks, tmp_path, module_name):
    expected = {task["id"] for task in tasks}
    done = _load_done_ids(tmp_path, module_name)
    failed = _load_failed_ids(tmp_path, module_name)
    expected_failed = _load_expected_failed_ids(tmp_path, module_name)
    attempted = done | failed | expected_failed
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


def _expected_failure_context():
    return {
        "plan": OpCasePlan(
            expected_failures=CaseSelector(
                contains={"expected_error"},
            )
        ),
        "full_module_name": "test",
        "run_func_name": "_task_fn",
        "runtime_version": None,
    }


class TestCudaFatalExceptionDetection:
    def test_torch_accelerator_error_is_fatal(self):
        torch_mod = MagicMock()
        torch_mod.AcceleratorError = type("AcceleratorError", (Exception,), {})

        assert _collect_mod._is_cuda_fatal_exception(torch_mod.AcceleratorError("boom"), torch_mod)

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

    @pytest.mark.timeout(10)
    def test_status_is_reconciled_if_worker_exits_before_progress(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_collect_mod, "worker", _exit_after_success_status_before_progress)
        tasks = _tasks(1)

        errors = _run_and_assert_all_done(tasks, 1, tmp_path, module_name="exit_between_done_and_progress")

        assert errors == []
        assert _load_done_ids(tmp_path, "exit_between_done_and_progress") == {"t0"}


class TestCentralPopulation:
    def test_raw_attention_run_is_physically_deduped_before_workers(self, monkeypatch):
        module_name = "tests.fake_attention_collector"
        fake_module = types.ModuleType(module_name)
        first = [2, 15, 8, 8, 64, True, False, False, 0]
        # SGLang generation ignores the context-FMHA flag when it persists the
        # row, so these are one consumer-visible collection point.
        duplicate_physical_row = [2, 15, 8, 8, 64, True, True, False, 0]
        fake_module.get_cases = lambda: [first, duplicate_physical_row]

        def run_case(*_args, **_kwargs):
            return None

        fake_module.run_case = run_case
        monkeypatch.setitem(sys.modules, module_name, fake_module)
        captured = []

        def capture_cases(_name, _type, get_cases, _run, _processes, **_kwargs):
            captured.append(get_cases())
            return []

        monkeypatch.setattr(_collect_mod, "collect_module_safe", capture_cases)
        collection = {
            "name": "sglang",
            "type": "attention_generation",
            "module": module_name,
            "get_func": "get_cases",
            "run_func": "run_case",
            "perf_filename": "generation_attention_perf.txt",
        }

        invocation_ids = set()
        assert (
            _collect_mod.collect_ops(
                1,
                [collection],
                backend="sglang",
                run_invocation_ids=invocation_ids,
            )
            == []
        )
        assert _collect_mod.collect_ops(1, [collection], backend="sglang") == []
        assert len(captured) == 2
        assert captured[0] == captured[1]
        assert len(captured[0]) == 1
        assert captured[0][0]["params"] == tuple(first)
        assert len(captured[0][0]["id"]) == 64
        assert invocation_ids == {captured[0][0]["id"]}

    def test_selector_limit_applies_once_after_all_rules_are_populated(self, monkeypatch):
        module_name = "tests.fake_additive_collector"
        fake_module = types.ModuleType(module_name)
        fake_module.get_cases = lambda: [[1], [2]]
        fake_module.run_case = lambda *_args, **_kwargs: None
        monkeypatch.setitem(sys.modules, module_name, fake_module)
        captured = []
        monkeypatch.setattr(
            _collect_mod,
            "collect_module_safe",
            lambda _name, _type, get_cases, _run, _processes, **_kwargs: captured.extend([get_cases()]) or [],
        )
        op_plan = OpCasePlan()
        op_plan.include.limit = 2
        op_plan.population_rules.append(PopulationRule(RuleSource("delta"), candidates=([2], [3], [4])))
        case_plan = types.SimpleNamespace(
            backend="sglang",
            model_path=None,
            requested_model_path=None,
            model_architecture=None,
            gpu_type=None,
            sm_version=None,
            full=False,
            base_cases_path=Path("base"),
            model_cases_paths=[],
            sm_exceptions_path=None,
            catalog=None,
            op_cases={"unmigrated_op": op_plan},
            population_reports={},
        )
        collection = {
            "name": "sglang",
            "type": "unmigrated_op",
            "module": module_name,
            "get_func": "get_cases",
            "run_func": "run_case",
            "perf_filename": "unmigrated_perf.txt",
        }

        assert _collect_mod.collect_ops(1, [collection], backend="sglang", case_plan=case_plan) == []

        assert [task["params"] for task in captured[0]] == [[1], [2]]
        report = case_plan.population_reports["sglang.unmigrated_op"]
        assert report["scheduled"] == 4
        assert report["selected"] == 2
        assert report["duplicate_invocations"] == 1

    def test_exclude_preserves_original_population_index_on_remaining_task(self, monkeypatch):
        module_name = "tests.fake_indexed_collector"
        fake_module = types.ModuleType(module_name)
        fake_module.get_cases = lambda: [["excluded", "normal"], ["remaining", "normal"]]
        fake_module.run_case = lambda *_args, **_kwargs: None
        monkeypatch.setitem(sys.modules, module_name, fake_module)
        captured = []
        monkeypatch.setattr(
            _collect_mod,
            "collect_module_safe",
            lambda _name, _type, get_cases, _run, _processes, **_kwargs: captured.extend([get_cases()]) or [],
        )
        op_plan = OpCasePlan(exclude=CaseSelector(indices={0}))
        case_plan = types.SimpleNamespace(
            backend="sglang",
            model_path=None,
            requested_model_path=None,
            model_architecture=None,
            gpu_type=None,
            sm_version=None,
            full=False,
            base_cases_path=Path("base"),
            model_cases_paths=[],
            sm_exceptions_path=None,
            catalog=None,
            op_cases={"unmigrated_op": op_plan},
            population_reports={},
        )
        collection = {
            "name": "sglang",
            "type": "unmigrated_op",
            "module": module_name,
            "get_func": "get_cases",
            "run_func": "run_case",
            "perf_filename": "unmigrated_perf.txt",
        }

        assert _collect_mod.collect_ops(1, [collection], backend="sglang", case_plan=case_plan) == []

        assert len(captured[0]) == 1
        assert captured[0][0]["params"] == ["remaining", "normal"]
        assert captured[0][0]["index"] == 1

    def test_shuffle_preserves_expected_failure_population_index(self, tmp_path, monkeypatch):
        module_name = "tests.fake_shuffled_expected_failure_collector"
        fake_module = types.ModuleType(module_name)
        fake_module.get_cases = lambda: [
            ["zero", "normal"],
            ["one", "expected_error"],
            ["two", "normal"],
        ]

        def run_case(label, behavior, *, perf_filename, device):
            del perf_filename
            _task_fn(label, behavior, device)

        fake_module.run_case = run_case
        monkeypatch.setitem(sys.modules, module_name, fake_module)
        op_plan = OpCasePlan(expected_failures=CaseSelector(indices={1}))
        case_plan = types.SimpleNamespace(
            backend="sglang",
            model_path=None,
            requested_model_path=None,
            model_architecture=None,
            gpu_type=None,
            sm_version=None,
            full=False,
            base_cases_path=Path("base"),
            model_cases_paths=[],
            sm_exceptions_path=None,
            catalog=None,
            op_cases={"unmigrated_op": op_plan},
            population_reports={},
        )
        collection = {
            "name": "sglang",
            "type": "unmigrated_op",
            "module": module_name,
            "get_func": "get_cases",
            "run_func": "run_case",
            "perf_filename": "unmigrated_perf.txt",
        }

        errors = _collect_mod.collect_ops(
            0,
            [collection],
            shuffle=True,
            backend="sglang",
            case_plan=case_plan,
            resume_options={"checkpoint_dir": str(tmp_path / ".checkpoint")},
        )

        assert errors == []
        assert len(_load_expected_failed_ids(tmp_path, "sglang.unmigrated_op", backend="sglang")) == 1
        assert _load_failed_ids(tmp_path, "sglang.unmigrated_op", backend="sglang") == set()


class TestResumeExecutionScope:
    def test_checkpoint_rejects_framework_or_hardware_change(self, tmp_path):
        original = _collect_mod.ResumeCheckpoint(
            "sglang",
            "sglang.attention_context",
            "run_attention_torch",
            str(tmp_path),
            framework_version="0.5.10",
            gpu_type="B200",
            sm_version=100,
        )
        original.mark_passed("case-id")
        original.flush(force=True)

        changed = _collect_mod.ResumeCheckpoint(
            "sglang",
            "sglang.attention_context",
            "run_attention_torch",
            str(tmp_path),
            framework_version="0.6.0",
            gpu_type="H200",
            sm_version=90,
        )
        with pytest.raises(RuntimeError, match=r"checkpoint mismatch .*framework_version"):
            changed.load_existing()

    def test_checkpoint_rejects_model_plan_change_with_same_task_payload(self, tmp_path):
        original = _collect_mod.ResumeCheckpoint(
            "sglang",
            "sglang.wideep_moe",
            "run_wideep_moe",
            str(tmp_path),
            plan_fingerprint=_collect_mod._plan_execution_fingerprint(None, "model/A"),
        )
        original.mark_passed("same-shape-task")
        original.flush(force=True)

        changed = _collect_mod.ResumeCheckpoint(
            "sglang",
            "sglang.wideep_moe",
            "run_wideep_moe",
            str(tmp_path),
            plan_fingerprint=_collect_mod._plan_execution_fingerprint(None, "model/B"),
        )
        with pytest.raises(RuntimeError, match=r"checkpoint mismatch .*plan_fingerprint"):
            changed.load_existing()

    def test_plan_fingerprint_includes_catalog_content(self):
        document = types.SimpleNamespace(path=Path("same.yaml"), data={"expected_failures": ["old"]})
        catalog = types.SimpleNamespace(
            base_documents=(document,),
            model_documents=(),
            sm_exceptions_document=None,
        )
        plan = types.SimpleNamespace(catalog=catalog)
        before = _collect_mod._plan_execution_fingerprint(plan, "model/A")

        document.data = {"expected_failures": []}

        assert _collect_mod._plan_execution_fingerprint(plan, "model/A") != before

    def test_plan_fingerprint_is_independent_of_worktree_path(self):
        def plan_at(path):
            document = types.SimpleNamespace(path=path, data={"cases": [{"batch_size": 1}]})
            catalog = types.SimpleNamespace(
                base_documents=(document,),
                model_documents=(),
                sm_exceptions_document=None,
            )
            return types.SimpleNamespace(catalog=catalog)

        first = _collect_mod._plan_execution_fingerprint(plan_at(Path("/worktree-a/base.yaml")), "model/A")
        second = _collect_mod._plan_execution_fingerprint(plan_at(Path("/worktree-b/base.yaml")), "model/A")

        assert first == second

    def test_checkpoint_rejects_measurement_mode_or_duration_change(self, tmp_path, monkeypatch):
        monkeypatch.setenv("COLLECTOR_MEASURE_POWER", "false")
        without_power = _collect_mod._plan_execution_fingerprint(None, "model/A")
        original = _collect_mod.ResumeCheckpoint(
            "sglang",
            "sglang.gemm",
            "run_gemm",
            str(tmp_path),
            plan_fingerprint=without_power,
        )
        original.mark_passed("case-id")
        original.flush(force=True)

        monkeypatch.setenv("COLLECTOR_MEASURE_POWER", "true")
        monkeypatch.setenv("COLLECTOR_POWER_MIN_DURATION", "2.5")
        with_power = _collect_mod._plan_execution_fingerprint(None, "model/A")
        assert with_power != without_power
        changed_mode = _collect_mod.ResumeCheckpoint(
            "sglang",
            "sglang.gemm",
            "run_gemm",
            str(tmp_path),
            plan_fingerprint=with_power,
        )
        with pytest.raises(RuntimeError, match=r"checkpoint mismatch .*plan_fingerprint"):
            changed_mode.load_existing()

        monkeypatch.setenv("COLLECTOR_POWER_MIN_DURATION", "3.0")
        assert _collect_mod._plan_execution_fingerprint(None, "model/A") != with_power


def test_runtime_hardware_identity_canonicalizes_b200_device_name(monkeypatch):
    device_module = types.SimpleNamespace(get_device_name=lambda _index: "NVIDIA B200")
    monkeypatch.setattr(_collect_mod, "get_device_module", lambda: device_module)
    monkeypatch.setattr(_collect_mod, "get_sm_version", lambda: 100)

    assert _collect_mod._runtime_hardware_identity("sglang") == ("b200_sxm", 100)


def test_runtime_hardware_identity_canonicalizes_concrete_xpu_name(monkeypatch):
    device_module = types.SimpleNamespace(get_device_name=lambda _index: "Intel(R) Data Center GPU Max 1550")
    monkeypatch.setattr(_collect_mod, "get_device_module", lambda: device_module)

    assert _collect_mod._runtime_hardware_identity("vllm_xpu") == ("xpu", 0)


def test_runtime_device_choice_rejects_planner_runtime_mismatch(monkeypatch):
    monkeypatch.setattr(_collect_mod, "_cuda_available", lambda: True)
    monkeypatch.setattr(_collect_mod, "_xpu_available", lambda: True)

    with pytest.raises(ValueError, match=r"conflicts with runtime routing to cuda"):
        _collect_mod._validate_runtime_device_choice("vllm", "xpu")

    monkeypatch.setattr(_collect_mod, "_cuda_available", lambda: False)
    with pytest.raises(ValueError, match=r"conflicts with runtime routing to xpu"):
        _collect_mod._validate_runtime_device_choice("vllm", "cuda")
    with pytest.raises(ValueError, match=r"collectors support CUDA only"):
        _collect_mod._validate_runtime_device_choice("sglang", "xpu")


def test_xpu_plan_gets_explicit_non_cuda_hardware_scope():
    assert _collect_mod._planner_hardware_scope("vllm_xpu", None, None) == ("xpu", 0)
    assert _collect_mod._planner_hardware_scope("vllm_xpu", "custom_xpu", 7) == ("custom_xpu", 7)


def test_resume_finalization_selects_untouched_staging_from_current_plan_only(tmp_path):
    current = tmp_path / "current_perf.txt"
    unrelated = tmp_path / "unrelated_perf.txt"
    touched = tmp_path / "touched_perf.txt"
    header = "framework,__collector_invocation_id,latency\n"
    current.write_text(header + "SGLANG,current-task,1.0\n", encoding="utf-8")
    unrelated.write_text(header + "SGLANG,other-task,2.0\n", encoding="utf-8")
    touched.write_text(header + "SGLANG,other-task,3.0\n", encoding="utf-8")
    existing = {path.resolve(): path.stat().st_mtime_ns for path in (current, unrelated, touched)}
    existing[touched.resolve()] -= 1

    selected = _collect_mod._select_finalizable_perf_outputs(tmp_path, existing, {"current-task"})

    assert selected == [current, touched]


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

    def test_sequential_mode_handles_restart_and_unexpected_system_exit(self, tmp_path):
        tasks = _tasks(
            [
                ("a", "exit_restart"),
                ("b", "exit_error"),
                ("c", "normal"),
            ]
        )

        errors = _run_and_assert_all_done(tasks, 0, tmp_path, module_name="sequential_system_exit")

        assert [error["task_id"] for error in errors if error.get("error_type") == "SystemExit"] == ["b"]
        assert _load_done_ids(tmp_path, "sequential_system_exit") == {"a", "c"}
        assert _load_failed_ids(tmp_path, "sequential_system_exit") == {"b"}

    def test_expected_failures_are_logged_but_not_reported_as_errors(self, tmp_path):
        tasks = _tasks(
            [
                ("a", "normal"),
                ("b", "expected_error"),
                ("c", "error"),
                ("d", "expected_error"),
            ]
        )
        errors = _run(
            tasks,
            2,
            tmp_path,
            module_name="expected_failures",
            expected_failure_context=_expected_failure_context(),
        )

        _assert_all_tasks_attempted(tasks, tmp_path, "expected_failures")
        assert len([e for e in errors if e.get("error_type") == "ValueError"]) == 1
        assert _load_done_ids(tmp_path, "expected_failures") == {"a"}
        assert _load_failed_ids(tmp_path, "expected_failures") == {"c"}
        assert _load_expected_failed_ids(tmp_path, "expected_failures") == {"b", "d"}

    def test_non_restart_system_exit_on_last_task_is_failed(self, tmp_path):
        tasks = _tasks([("last", "exit_error")])

        errors = _run_and_assert_all_done(tasks, 1, tmp_path, module_name="last_system_exit")

        assert [error["task_id"] for error in errors if error.get("error_type") == "SystemExit"] == ["last"]
        assert _load_done_ids(tmp_path, "last_system_exit") == set()
        assert _load_failed_ids(tmp_path, "last_system_exit") == {"last"}

    def test_non_restart_system_exit_does_not_skip_mixed_queue_tasks(self, tmp_path):
        tasks = _tasks(
            [
                ("a", "exit_error"),
                ("b", "normal"),
                ("c", "exit_error"),
                ("d", "normal"),
            ]
        )

        errors = _run_and_assert_all_done(tasks, 1, tmp_path, module_name="mixed_system_exit")

        assert {error["task_id"] for error in errors if error.get("error_type") == "SystemExit"} == {"a", "c"}
        assert _load_done_ids(tmp_path, "mixed_system_exit") == {"b", "d"}
        assert _load_failed_ids(tmp_path, "mixed_system_exit") == {"a", "c"}


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

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-level collector entrypoint.

This script resolves the requested backend, framework version, model/SM case
plan, and op registry entry, then runs the selected collector functions and
writes perf files. It is the orchestration layer for collector v2; individual
modules own benchmark setup, while `model_cases.py` and YAML own case selection.
"""

import contextlib
import functools
import os
import warnings

from helper import get_device_module, get_device_str


def setup_warning_filters():
    """Configure warning filters to suppress known non-critical warnings"""

    # Suppress the modelopt transformers version warning
    warnings.filterwarnings(
        "ignore",
        message="transformers version .* is incompatible with nvidia-modelopt",
        category=UserWarning,
        module="modelopt",
    )

    # Suppress the cuda.cudart deprecation warning
    warnings.filterwarnings("ignore", message="The cuda.cudart module is deprecated", category=FutureWarning)

    warnings.filterwarnings("ignore", message="The cuda.cuda module is deprecated", category=FutureWarning)

    # Suppress TensorRT-LLM specific warnings if needed
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorrt_llm")

    # Suppress flashinfer warnings
    warnings.filterwarnings("ignore", message="Prebuilt kernels not found", module="flashinfer")

    # Suppress torch operator override warnings (flash_attn kernel re-registration)
    warnings.filterwarnings(
        "ignore",
        message="Warning only once for all operators.*",
        category=UserWarning,
    )

    # Suppress pynvml deprecation warning from torch.cuda
    warnings.filterwarnings(
        "ignore",
        message="The pynvml package is deprecated",
        category=FutureWarning,
    )


import random
import resource

from tqdm import tqdm

try:
    import torch
except ModuleNotFoundError:
    torch = None

setup_warning_filters()

import argparse
import cProfile
import importlib
import importlib.util
import io
import json
import multiprocessing as mp
import pstats
import signal
import subprocess
import time
import traceback
from collections import Counter
from datetime import datetime
from inspect import Parameter, signature
from pathlib import Path

from helper import (
    EXIT_CODE_RESTART,
    WorkerRestartSignal,
    create_test_case_id,
    finalize_perf_files,
    find_perf_csv_outputs,
    save_error_report,
    setup_logging,
    setup_signal_handlers,
)

logger = None
RESUME_SCHEMA_VERSION = "collector-resume-v2"
STALL_THRESHOLD = 30  # iterations (x 0.5 s sleep = 15 s) before logging a stall warning
# Failures of one (model, dtype) group within an op before the summary flags
# it as systemic (a fix-me warning; nothing is skipped).
SYSTEMIC_GROUP_THRESHOLD = 5


def _require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is required to run collectors. Use --plan-only to inspect collector v2 YAML plans.")
    return torch


def _cuda_available() -> bool:
    return torch is not None and torch.cuda.is_available()


def _xpu_available() -> bool:
    return torch is not None and hasattr(torch, "xpu") and torch.xpu.is_available()


def _wideep_registry_for_backend(backend: str) -> list:
    module_name = f"collector.wideep.{backend}.registry"
    try:
        spec = importlib.util.find_spec(module_name)
    except ModuleNotFoundError:
        return []
    if spec is None:
        return []
    return list(importlib.import_module(module_name).REGISTRY)


def _registry_with_requested_wideep(registry: list, backend: str, ops: list[str] | None, case_plan=None) -> list:
    wideep_registry = _wideep_registry_for_backend(backend)
    if not wideep_registry:
        return registry

    requested_ops = set(ops if ops is not None else (case_plan.ops if case_plan is not None else []))
    requested_wideep_ops = requested_ops & {entry.op for entry in wideep_registry}
    if not requested_wideep_ops:
        return registry

    if logger is not None:
        logger.info(f"WideEP registry active for {backend}: {sorted(requested_wideep_ops)}")
    return [*registry, *wideep_registry]


class ResumeCheckpoint:
    """Tracks which tasks are done so a collection run can be resumed.

    Always writes checkpoint files.  When ``--resume`` is passed the existing
    checkpoint is loaded and done tasks are skipped; otherwise the checkpoint
    is overwritten from scratch (so a future ``--resume`` can pick up).
    """

    FLUSH_INTERVAL_SEC = 2.0

    def __init__(
        self,
        backend: str,
        module_name: str,
        run_func_name: str,
        checkpoint_dir: str,
        framework_version: str | None = None,
        sm_version: int | None = None,
    ):
        self.module_name = module_name
        self._dirty = False
        self._last_flush = 0.0
        # framework_version/sm_version bind the checkpoint to the runtime it
        # was collected under: resuming a plan across a version bump or on a
        # different GPU generation silently mislabels data, so it must fail.
        self._metadata = {
            "schema": RESUME_SCHEMA_VERSION,
            "backend": backend,
            "module": module_name,
            "run_func": run_func_name,
            "framework_version": framework_version,
            "sm_version": sm_version,
        }
        self._done: set[str] = set()
        self._failed: set[str] = set()

        safe_name = module_name.replace("/", "_").replace(":", "_")
        self._path = Path(checkpoint_dir).expanduser().resolve() / backend / f"{safe_name}.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def load_existing(self):
        """Load an existing checkpoint for resume.  Raises on mismatch."""
        if not self._path.exists():
            logger.info(f"{self.module_name}: no checkpoint found, starting fresh")
            return

        try:
            with open(self._path) as f:
                data = json.load(f)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint {self._path}: {e}. Run without --resume to start fresh."
            ) from e

        for key in ("schema", "backend", "module", "run_func", "framework_version", "sm_version"):
            if data.get(key) != self._metadata[key]:
                raise RuntimeError(
                    f"{self.module_name}: checkpoint mismatch "
                    f"({key}: {data.get(key)} != {self._metadata[key]}). "
                    "Run without --resume to start fresh."
                )

        self._done = set(data.get("done", []))
        self._failed = set(data.get("failed", []))
        logger.info(f"{self.module_name}: loaded checkpoint — {len(self._done)} passed, {len(self._failed)} failed")

    # -- public API -------------------------------------------------------

    def filter_done(self, task_infos: list[dict], retry_failed: bool = False) -> list[dict]:
        """Return only tasks that need to run.

        By default, skips both passed and failed tasks. With retry_failed=True,
        previously failed tasks are retried.
        """
        skip_set = self._done if retry_failed else (self._done | self._failed)
        runnable = [t for t in task_infos if t["id"] not in skip_set]
        skipped_done = sum(1 for t in task_infos if t["id"] in self._done)
        skipped_failed = sum(1 for t in task_infos if t["id"] in self._failed)
        retrying = sum(1 for t in runnable if t["id"] in self._failed) if retry_failed else 0
        if skipped_done or skipped_failed or retrying:
            parts = [f"skipping {skipped_done} passed"]
            if retry_failed:
                parts.append(f"retrying {retrying} previously failed")
            else:
                parts.append(f"skipping {skipped_failed} failed")
            parts.append(f"running {len(runnable)}")
            logger.info(f"{self.module_name}: {', '.join(parts)}")
        return runnable

    def mark_passed(self, task_id: str):
        """Mark a task as successfully completed. Skipped on resume."""
        self._done.add(task_id)
        self._failed.discard(task_id)  # if it was previously failed, it passed now
        self._dirty = True
        self.flush()

    def mark_failed(self, task_id: str):
        """Mark a task as attempted but failed."""
        self._failed.add(task_id)
        self._dirty = True
        self.flush()

    def unresolved_failed_count(self) -> int:
        """Number of tasks the checkpoint holds as failed and unresolved."""
        return len(self._failed)

    # Keep mark_done as alias for backwards compat
    mark_done = mark_passed

    def flush(self, force: bool = False):
        if not self._dirty:
            return
        now = time.time()
        if not force and (now - self._last_flush) < self.FLUSH_INTERVAL_SEC:
            return

        data = {
            **self._metadata,
            "updated_at": datetime.now().isoformat(),
            "done": sorted(self._done),
            "failed": sorted(self._failed),
        }
        tmp_path = self._path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self._path)
        self._dirty = False
        self._last_flush = now


class ProfilerContext:
    """Context manager for profiling collector execution"""

    def __init__(self, backend: str, enabled: bool = False):
        self.enabled = enabled
        self.backend = backend
        self.profiler = None
        self.start_time = None
        self.log_dir = None

    def __enter__(self):
        if self.enabled:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
            self.start_time = time.perf_counter()
            self.log_dir = os.environ.get("COLLECTOR_LOG_DIR", "")
            if not self.log_dir:
                self.log_dir = "."
            logger.info("Profiling enabled - running sequentially in main process (no parallel workers)")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled or self.profiler is None:
            return

        self.profiler.disable()
        profile_file = os.path.join(self.log_dir, f"collector_profile_{self.backend}.prof")
        self.profiler.dump_stats(profile_file)

        # Calculate elapsed time
        end_time = time.perf_counter()
        elapsed_time = end_time - self.start_time if self.start_time else 0

        logger.info("=" * 80)
        logger.info("PROFILING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total elapsed time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
        logger.info(f"Profile file: {profile_file}")
        logger.info("=" * 80)

        # Print slow operations ranked by tottime and cumtime
        stats = pstats.Stats(self.profiler)
        stats.strip_dirs()

        # Get top functions by tottime (time spent in the function itself, excluding subcalls)
        logger.info("Top 20 functions by tottime (time in function excluding subcalls):")
        logger.info("=" * 80)
        stream = io.StringIO()
        import sys

        old_stdout = sys.stdout
        sys.stdout = stream
        try:
            stats.sort_stats("tottime")
            stats.print_stats(20)
        finally:
            sys.stdout = old_stdout
        for line in stream.getvalue().split("\n"):
            if line.strip():
                logger.info(line)

        # Get top functions by cumtime (cumulative time including subcalls)
        logger.info("=" * 80)
        logger.info("Top 20 functions by cumtime (cumulative time including subcalls):")
        logger.info("=" * 80)
        stream = io.StringIO()
        sys.stdout = stream
        try:
            stats.sort_stats("cumtime")
            stats.print_stats(20)
        finally:
            sys.stdout = old_stdout
        for line in stream.getvalue().split("\n"):
            if line.strip():
                logger.info(line)

        logger.info("=" * 80)
        logger.info(f"Full profile saved to: {profile_file}")


def _failure_group(task) -> str | None:
    """Group label for failure aggregation: one (model, dtype) family within an op.

    A whole group failing is the signal that something needs FIXING (collector
    bug, unverified combo, framework gap) — the summary aggregates failures by
    this label so systemic groups are visible at a glance. Returns None when
    the task carries neither a model nor a dtype attribute (e.g. positional
    tuple cases).
    """
    from collector.capabilities import case_dtypes

    model = getattr(task, "model_name", None) or getattr(task, "model_path", None) or ""
    dtypes = ",".join(case_dtypes(task))
    if not model and not dtypes:
        return None
    return f"{model}|{dtypes}"


def _is_cuda_fatal_exception(exc, torch_mod) -> bool:
    fatal_error_types = tuple(
        error_type
        for error_type in (
            getattr(torch_mod, "AcceleratorError", None),
            getattr(torch_mod, "OutOfMemoryError", None),
        )
        if isinstance(error_type, type)
    )
    is_cuda_fatal = isinstance(exc, fatal_error_types)
    if not is_cuda_fatal:
        error_text = str(exc).lower()
        fatal_markers = (
            "illegal memory access",
            "unspecified launch failure",
            "cuda_error_launch_failed",
            "cublas_status_execution_failed",
            "cublas_status_internal_error",
            "cublas_status_alloc_failed",
        )
        is_cuda_fatal = any(marker in error_text for marker in fatal_markers)
    if not is_cuda_fatal:
        # DSLCudaRuntimeError from CUTLASS DSL also corrupts CUDA context but
        # is not a torch.AcceleratorError subclass.
        is_cuda_fatal = type(exc).__name__ == "DSLCudaRuntimeError"
    return is_cuda_fatal


def collect_module_safe(
    module_name,
    test_type,
    get_test_cases_func,
    run_func,
    num_processes,
    resume_options=None,
):
    """
    Safely collect module with comprehensive error handling

    Args:
        num_processes: Number of parallel processes to use. If 0, runs sequentially in main process.
    """
    full_name = f"{module_name}.{test_type}"
    logger.info(f"Starting collection: {full_name}")

    try:
        # Get test cases
        test_cases = get_test_cases_func()
        logger.info(f"Generated {len(test_cases)} test cases for {full_name}")

        # Run collection
        errors = parallel_run(
            test_cases,
            run_func,
            num_processes,
            full_name,
            resume_options=resume_options,
        )

        return errors

    except Exception as e:
        logger.exception(f"Failed to collect {full_name}")
        return [
            {
                "module": full_name,
                "error_type": "ModuleCollectionFailure",
                "error_message": str(e),
                "traceback": traceback.format_exc(),
            }
        ]


def worker(
    queue,
    device_id: int,
    func,
    progress_value,
    lock,
    error_queue=None,
    done_tasks=None,
    failed_tasks=None,
    module_name="unknown",
    current_task_ids=None,
    consumed_sentinel=None,
):
    """worker with automatic logging setup"""

    # Disable core dumps — GPU crashes are expected and handled via error_queue;
    # without this, each SIGSEGV/SIGABRT writes a multi-GB core file to disk.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    setup_warning_filters()  # Must run in each spawned process

    # Setup logging for this worker - reads config from environment automatically
    worker_logger = setup_logging(worker_id=device_id)

    # Setup signal handlers
    setup_signal_handlers(device_id)

    # Setup device
    torch_mod = _require_torch()
    device = torch_mod.device(f"{get_device_str()}:{device_id}")
    get_device_module().set_device(device)
    worker_logger.info(f"Worker {device_id} initialized for {module_name}")

    # Process tasks
    while True:
        task_info = queue.get()
        if task_info is None:
            if current_task_ids is not None:
                current_task_ids[device_id] = None
            if consumed_sentinel is not None:
                consumed_sentinel[device_id] = True
            worker_logger.debug("Received termination signal")
            break

        # Handle both old format (tuple) and new format (dict)
        if isinstance(task_info, dict):
            task_id = task_info.get("id", "unknown")
            task = task_info.get("params", task_info)
        else:
            task = task_info
            task_id = create_test_case_id(task, "unknown", module_name)

        if current_task_ids is not None:
            current_task_ids[device_id] = task_id

        try:
            worker_logger.debug(f"Starting task {task_id}")
            result = func(*task, device=device)
            # Only the dedicated sentinel requests a recycle: entrypoints also
            # return plain ints (row counts), which must never be mistaken for
            # EXIT_CODE_RESTART.
            if isinstance(result, WorkerRestartSignal):
                raise SystemExit(EXIT_CODE_RESTART)
            worker_logger.debug(f"Completed task {task_id}")

            # Mark done ONLY on success — failed tasks should be retried on resume
            if done_tasks is not None:
                try:
                    done_tasks[task_id] = True
                except Exception:
                    pass
            # Clear task ID on success so crash handler knows it completed
            if current_task_ids is not None:
                current_task_ids[device_id] = None
        except SystemExit as e:
            # EXIT_CODE_RESTART: task completed successfully, worker exits to free GPU memory
            # (e.g., MOE collectors call sys.exit(EXIT_CODE_RESTART) after finishing)
            if e.code == EXIT_CODE_RESTART:
                if done_tasks is not None:
                    try:
                        done_tasks[task_id] = True
                    except Exception:
                        pass
                if current_task_ids is not None:
                    current_task_ids[device_id] = None
            raise  # re-raise so the worker actually exits
        except Exception as e:
            # Build comprehensive error info
            error_info = {
                "module": module_name,
                "device_id": device_id,
                "task_id": task_id,
                "task_params": str(task),
                "error_type": type(e).__name__,
                "error_message": str(e),
                "classification": "unexpected",
                "group": _failure_group(task),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat(),
            }

            # Report error to queue BEFORE any exit
            if error_queue:
                error_queue.put(error_info)

            worker_logger.exception(f"Task {task_id} failed")

            # Track failed task for checkpoint
            if failed_tasks is not None:
                try:
                    failed_tasks[task_id] = True
                except Exception:
                    pass
            # Clear task ID so crash handler knows it was handled
            if current_task_ids is not None:
                current_task_ids[device_id] = None

            # Force flush logs before any potential exit
            for handler in worker_logger.handlers:
                handler.flush()

            if _is_cuda_fatal_exception(e, torch_mod):
                worker_logger.warning(
                    f"Fatal {type(e).__name__} encountered on task {task_id}. "
                    f"Worker {device_id} exiting to reset GPU context. "
                    f"Progress: {progress_value.value}"
                )
                # Flush logs again after warning
                for handler in worker_logger.handlers:
                    handler.flush()
                # Exiting with non-zero code will add an additional error to the summary,
                # which we don't want (error already reported above).
                exit(0)
        finally:
            with lock:
                progress_value.value += 1

            # Periodic memory cleanup to reduce fragmentation
            if progress_value.value % 100 == 0:
                import gc

                gc.collect()
                get_device_module().empty_cache()


def parallel_run(tasks, func, num_processes, module_name="unknown", resume_options=None):
    """parallel runner with error collection

    Args:
        num_processes: Number of parallel processes. If 0, runs sequentially in main process.
    """
    # func may be a functools.partial (perf_filename bound by collect_ops),
    # which lacks __name__. Fall back to partial.func to get the wrapped function.
    func_name = getattr(func, "__name__", None) or getattr(func, "func", func).__name__
    raw_task_infos = []
    for i, task in enumerate(tasks):
        if isinstance(task, dict) and "id" in task and "params" in task:
            task_id = task["id"]
            task_params = task["params"]
        else:
            task_id = create_test_case_id(task, func_name, module_name)
            task_params = task
        raw_task_infos.append({"id": task_id, "params": task_params, "index": i})

    checkpoint_dir = (
        resume_options.get("checkpoint_dir", ".collector_checkpoint") if resume_options else ".collector_checkpoint"
    )
    resume_tracker = ResumeCheckpoint(
        backend=resume_options.get("backend", "unknown") if resume_options else "unknown",
        module_name=module_name,
        run_func_name=func_name,
        checkpoint_dir=checkpoint_dir,
        framework_version=resume_options.get("framework_version") if resume_options else None,
        sm_version=resume_options.get("sm_version") if resume_options else None,
    )

    if resume_options and resume_options.get("resume"):
        resume_tracker.load_existing()
        retry_failed = resume_options.get("retry_failed", False)
        task_infos = resume_tracker.filter_done(raw_task_infos, retry_failed=retry_failed)
    else:
        task_infos = raw_task_infos

    def _unresolved_failure_errors():
        # A resumed run must not look clean while its checkpoint still holds
        # unresolved failures: completion and acceptance are distinct.
        if not (resume_options and resume_options.get("resume")):
            return []
        unresolved = resume_tracker.unresolved_failed_count()
        if not unresolved:
            return []
        logger.warning(
            f"{module_name}: checkpoint holds {unresolved} unresolved failed tasks "
            "(skipped on resume; rerun with --resume-retry-failed to retry)"
        )
        return [
            {
                "module": module_name,
                "task_id": "resume_unresolved",
                "error_type": "UnresolvedFailures",
                "error_message": f"checkpoint holds {unresolved} unresolved failed tasks",
                "classification": "unresolved_from_checkpoint",
                "timestamp": datetime.now().isoformat(),
            }
        ]

    if not task_infos:
        logger.info(f"{module_name}: no tasks to run")
        return _unresolved_failure_errors()

    queue = mp.Queue()
    error_queue = mp.Queue()
    processes = []

    manager = mp.Manager()
    progress_value = manager.Value("i", 0)
    lock = manager.Lock()

    # Track process health
    process_stats = {i: {"restarts": 0, "errors": []} for i in range(num_processes)}

    # Per-worker flag: True once a worker has consumed its None sentinel.
    # Used to decide whether a replacement sentinel is needed on restart.
    consumed_sentinel = manager.dict(dict.fromkeys(range(num_processes), False))
    current_task_ids = manager.dict(dict.fromkeys(range(num_processes), None))
    # Synchronous record of completed task IDs.  Workers write here via
    # manager RPC in their finally block — same mechanism as progress_value,
    # so it is guaranteed to be visible before the worker touches the next
    # task.  Unlike mp.Queue (async feeder thread) this cannot be lost when
    # a worker is killed by a signal on a subsequent task.
    done_tasks = manager.dict()
    failed_tasks = manager.dict()

    def start_process(device_id):
        p = mp.Process(
            target=worker,
            args=(
                queue,
                device_id,
                func,
                progress_value,
                lock,
                error_queue,
                done_tasks,
                failed_tasks,
                module_name,
                current_task_ids,
                consumed_sentinel,
            ),
        )
        p.start()
        logger.info(f"Started worker process {p.pid} on device {device_id}")
        return p

    def create_process_exit_error(device_id, exit_code):
        if exit_code in (None, 0, EXIT_CODE_RESTART):
            return None

        if exit_code < 0:
            signum = -exit_code
            try:
                signame = signal.Signals(signum).name
            except Exception:
                signame = f"SIG{signum}"
            reason = f"terminated by signal {signum} ({signame})"
            error_type = "WorkerSignalCrash"
        else:
            reason = f"exited with status {exit_code}"
            error_type = "WorkerAbnormalExit"

        logger.error(f"Process {device_id} ({module_name}) {reason}")

        return {
            "module": module_name,
            "device_id": device_id,
            "task_id": "process_exit",
            "task_params": None,
            "error_type": error_type,
            "error_message": reason,
            "traceback": "",
            "exit_code": exit_code,
            "timestamp": datetime.now().isoformat(),
        }

    def sync_done_to_checkpoint():
        for task_id in list(done_tasks.keys()):
            resume_tracker.mark_passed(task_id)
            try:
                del done_tasks[task_id]
            except KeyError:
                pass
        for task_id in list(failed_tasks.keys()):
            resume_tracker.mark_failed(task_id)
            try:
                del failed_tasks[task_id]
            except KeyError:
                pass

    # Start processes
    for device_id in range(num_processes):
        processes.append(start_process(device_id))

    # Queue tasks with IDs
    for task_info in task_infos:
        queue.put(task_info)

    # Add termination signals
    for _ in range(len(processes)):
        queue.put(None)

    # Monitor progress with error collection
    errors = []

    with tqdm(total=len(task_infos), desc=f"{module_name}", dynamic_ncols=True, leave=True) as pbar:
        last_progress = 0
        stall_count = 0
        last_error_count = 0

        if num_processes == 0:
            # Special handling for --profile
            # Run tasks sequentially in main process
            torch_mod = _require_torch()
            device = torch_mod.device(f"{get_device_str()}:0")
            get_device_module().set_device(device)

            for task_info in task_infos:
                task_id = task_info["id"]
                task_params = task_info["params"]

                try:
                    func(*task_params, device=device)
                    resume_tracker.mark_passed(task_id)
                except Exception as e:
                    resume_tracker.mark_failed(task_id)
                    error_info = {
                        "module": module_name,
                        "device_id": 0,
                        "task_id": task_id,
                        "task_params": str(task_params),
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "classification": "unexpected",
                        "group": _failure_group(task_params),
                        "traceback": traceback.format_exc(),
                        "timestamp": datetime.now().isoformat(),
                    }
                    errors.append(error_info)
                    logger.exception(f"Task {task_id} failed")

                pbar.update(1)
                progress_value.value += 1
                if len(errors) > 0:
                    pbar.set_postfix({"errors": len(errors)})
                resume_tracker.flush()
            resume_tracker.flush(force=True)

        while progress_value.value < len(task_infos):
            # Drain errors
            while not error_queue.empty():
                error = error_queue.get()
                errors.append(error)
                process_stats[error["device_id"]]["errors"].append(error["task_id"])
            sync_done_to_checkpoint()

            # Update postfix only if count changed
            if len(errors) != last_error_count:
                pbar.set_postfix({"errors": len(errors)})
                last_error_count = len(errors)

            if progress_value.value == last_progress:
                stall_count += 1
                if stall_count > STALL_THRESHOLD:
                    logger.warning(f"Progress stalled at {progress_value.value}/{len(task_infos)}")
                    stall_count = 0
            else:
                stall_count = 0
                last_progress = progress_value.value

            # Check process health — only restart if there is still work
            # remaining.  Workers that consumed a None sentinel or finished
            # via sys.exit(EXIT_CODE_RESTART) should not be restarted once
            # all tasks are dispatched, otherwise the new worker blocks
            # forever on queue.get().
            for i, p in enumerate(processes):
                if p is None:
                    continue

                if not p.is_alive():
                    exit_code = p.exitcode
                    active_task_id = current_task_ids.get(i)
                    process_stats[i]["restarts"] += 1
                    if exit_code == EXIT_CODE_RESTART:
                        logger.debug(
                            f"Process {i} completed task and exited normally for release gpu memory"
                            f"(completed tasks: {process_stats[i]['restarts']})"
                        )
                    else:
                        logger.warning(
                            f"Process {i} died (exit code: {exit_code}, "
                            f"restarts: {process_stats[i]['restarts']}, "
                            f"errors: {len(process_stats[i]['errors'])})"
                        )

                    # Mark active task as failed if the process died while running it
                    if active_task_id is not None and active_task_id not in done_tasks:
                        try:
                            failed_tasks[active_task_id] = True
                        except Exception:
                            pass
                        current_task_ids[i] = None
                        with lock:
                            progress_value.value += 1

                    crash_error = create_process_exit_error(i, exit_code)
                    if crash_error:
                        errors.append(crash_error)
                        process_stats[i]["errors"].append("process_exit")
                        pbar.set_postfix({"errors": len(errors)})
                        last_error_count = len(errors)

                    if process_stats[i]["restarts"] > 8192:
                        logger.error(f"Process {i} exceeded restart limit, not restarting")
                        processes[i] = None
                        continue

                    if consumed_sentinel.get(i, False):
                        processes[i] = None
                        continue

                    remaining = len(task_infos) - progress_value.value
                    if remaining > 0:
                        processes[i] = start_process(i)
                    else:
                        processes[i] = None

            current = progress_value.value
            if current > pbar.n:
                pbar.update(current - pbar.n)

            resume_tracker.flush()
            time.sleep(0.5)
        sync_done_to_checkpoint()

    # Collect remaining errors
    while not error_queue.empty():
        errors.append(error_queue.get())
    sync_done_to_checkpoint()
    resume_tracker.flush(force=True)

    # Wait for processes
    for p in processes:
        if p is None:
            continue
        p.join(timeout=42)
        if p.is_alive():
            logger.warning(f"Process {p.pid} did not terminate, forcing...")
            p.terminate()

    # Shutdown manager to clean up resources (semaphores, etc.)
    manager.shutdown()

    # Surface systemic failure groups — a whole (model, dtype) family failing
    # is a fix-me signal, not something to tolerate.
    group_counts = Counter(error["group"] for error in errors if error.get("group"))
    for group, count in sorted(group_counts.items()):
        if count >= SYSTEMIC_GROUP_THRESHOLD:
            logger.warning(f"{module_name}: failure group {group!r} failed {count} times — needs fixing")

    errors.extend(_unresolved_failure_errors())

    # Log summary
    if errors:
        log_dir = os.environ.get("COLLECTOR_LOG_DIR", "")
        logger.error(f"{module_name}: Completed with {len(errors)} errors")
        error_file = f"{log_dir}/errors_{module_name}.json"
        save_error_report(errors, error_file)
        logger.error(f"Error details saved to {error_file}")
    else:
        logger.info(f"{module_name}: Completed successfully with no errors")

    return errors


def collect_ops(
    num_processes: int,
    collections: list[dict],
    runtime_version: str | None = None,
    limit: int | None = None,
    shuffle: bool = False,
    shuffle_seed: int = 42,
    backend: str = "unknown",
    resume_options: dict | None = None,
    model_path: str | None = None,
    case_plan=None,
    sm_version: int | None = None,
    case_filters: list[str] | None = None,
) -> list[dict]:
    """Run collection for a list of resolved collection entries.

    Each entry must have: name, type, module, get_func, run_func.
    Version resolution and op filtering are handled upstream by
    version_resolver.build_collections(). If runtime_version is provided,
    per-module __compat__ is validated and incompatible ops fail explicitly.
    If limit is provided, the number of test cases is limited to the limit.
    If shuffle is True, the test cases are shuffled with the given seed.
    """

    class CompatibilityError(RuntimeError):
        """Raised when a resolved collector module is incompatible."""

    check_compat = None
    if runtime_version:
        from collector.version_resolver import _check_compat as check_compat

    @contextlib.contextmanager
    def _collector_model_path(model_path: str | None):
        previous = os.environ.get("COLLECTOR_MODEL_PATH")
        if model_path:
            os.environ["COLLECTOR_MODEL_PATH"] = model_path
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("COLLECTOR_MODEL_PATH", None)
            else:
                os.environ["COLLECTOR_MODEL_PATH"] = previous

    def _get_test_cases(get_func, model_path: str | None):
        if not model_path:
            return get_func()
        with _collector_model_path(model_path):
            sig = signature(get_func)
            params = sig.parameters
            if "model_path" in params or any(param.kind == Parameter.VAR_KEYWORD for param in params.values()):
                return get_func(model_path=model_path)
            return get_func()

    all_errors = []

    for collection in collections:
        try:
            if case_plan is not None and not case_plan.has_op(collection["type"]):
                logger.info(f"Skipping {collection['name']}.{collection['type']} — not in collector v2 case plan")
                continue
            unverified_here = collection.get("unverified") or (
                sm_version is not None and sm_version in (collection.get("unverified_sms") or ())
            )
            if unverified_here:
                scope = "this backend" if collection.get("unverified") else f"SM{sm_version}"
                logger.warning(
                    f"Skipping {collection['name']}.{collection['type']} — registry marks it unverified "
                    f"on {scope}; remove the OpEntry marker once the collector is debugged there"
                )
                all_errors.append(
                    {
                        "module": f"{collection['name']}.{collection['type']}",
                        "error_type": "UnverifiedCollector",
                        "error_message": f"registry marks this op unverified on {scope}; collection skipped",
                        "classification": "unverified_skipped",
                    }
                )
                continue
            module_name = collection["module"]
            get_module = __import__(module_name, fromlist=[collection["get_func"]])
            run_module = __import__(module_name, fromlist=[collection["run_func"]])

            # Fail this op explicitly if declared compatibility doesn't match runtime.
            if check_compat:
                declared = getattr(get_module, "__compat__", None)
                if declared:
                    try:
                        if not check_compat(declared, runtime_version):
                            if _xpu_available():
                                # Disable vllm xpu runtime version check for now
                                logger.warning(
                                    f"module {module_name} declares __compat__={declared!r}, \
                                    runtime is v{runtime_version}"
                                )
                            else:
                                raise CompatibilityError(
                                    f"module {module_name} declares __compat__={declared!r}, \
                                        runtime is v{runtime_version}"
                                )
                    except ValueError as e:
                        raise CompatibilityError(f"invalid __compat__ {declared!r}: {e}") from e

            get_func = getattr(get_module, collection["get_func"])
            run_func = getattr(run_module, collection["run_func"])
            run_func = functools.partial(run_func, perf_filename=collection["perf_filename"])

            def get_func_with_limit(get_func=get_func, op=collection["type"]):
                from collector.capabilities import filter_cases

                cases = _get_test_cases(get_func, model_path)
                cases, _dropped = filter_cases(cases, op=op, sm_version=sm_version)
                if case_filters:
                    before_count = len(cases)
                    cases = [case for case in cases if any(fragment in str(case) for fragment in case_filters)]
                    logger.info(f"{op}: --case-filter kept {len(cases)}/{before_count} cases")
                if shuffle:
                    rng = random.Random(shuffle_seed)
                    rng.shuffle(cases)
                if limit is not None:
                    cases = cases[:limit]
                return cases

            merged_resume = {
                **(resume_options or {}),
                "backend": backend,
                "framework_version": runtime_version,
                "sm_version": sm_version,
            }
            errors = collect_module_safe(
                collection["name"],
                collection["type"],
                get_func_with_limit,
                run_func,
                num_processes,
                resume_options=merged_resume,
            )
            all_errors.extend(errors)

        except Exception as e:
            logger.exception(f"Failed to process {collection['name']}.{collection['type']}")
            all_errors.append(
                {
                    "module": f"{collection['name']}.{collection['type']}",
                    "error_type": "CompatibilityError" if isinstance(e, CompatibilityError) else type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc(),
                }
            )

    return all_errors


def collect_sglang(
    num_processes: int,
    ops: list[str] | None = None,
    limit: int | None = None,
    shuffle: bool = False,
    resume_options: dict | None = None,
    model_path: str | None = None,
    case_plan=None,
    sm_version: int | None = None,
    case_filters: list[str] | None = None,
):
    """Collect performance data for SGLang with enhanced error tracking"""
    os.environ["FLASHINFER_LOG_LEVEL"] = "ERROR"

    # DSV4-Pro mhc-pre fast path: the DeepGEMM tf32 prenorm + TileLang fused
    # kernels must have these names present in os.environ at worker-spawn time,
    # otherwise mhc-pre collects ~53% slow and diverges from the reference
    # dataset. Both default to True in environ.py, but the effect is triggered
    # by presence in the process env (consumed downstream in the JIT path), not
    # just by .get(). setdefault so an explicit caller value still wins.
    os.environ.setdefault("SGLANG_OPT_DEEPGEMM_HC_PRENORM", "1")
    os.environ.setdefault("SGLANG_OPT_USE_TILELANG_MHC_PRE", "1")

    try:
        from importlib.metadata import version as get_version

        version = get_version("sglang")
        logger.info(f"SGLang version: {version}")
    except Exception:
        logger.exception("SGLang is not installed")
        return None, None

    from collector.framework_manifest import require_collector_runtime

    requested_ops = set(ops if ops is not None else (case_plan.ops if case_plan is not None else []))
    wideep_ops = {entry.op for entry in _wideep_registry_for_backend("sglang")}
    runtime = require_collector_runtime("sglang", version, requested_ops=requested_ops, wideep_ops=wideep_ops)

    from collector.sglang.registry import REGISTRY
    from collector.version_resolver import build_collections

    registry = _registry_with_requested_wideep(REGISTRY, "sglang", ops, case_plan)
    collections = build_collections(registry, "sglang", version, ops, logger=logger)
    all_errors = collect_ops(
        num_processes,
        collections,
        version,
        limit=limit,
        shuffle=shuffle,
        backend="sglang",
        resume_options=resume_options,
        model_path=model_path,
        case_plan=case_plan,
        sm_version=sm_version,
        case_filters=case_filters,
    )

    generate_collection_summary(all_errors, "sglang", version)
    provenance_ctx = {
        "framework": runtime.framework,
        "installed_version": version,
        "runtime": runtime,
        "collections": collections,
    }
    return all_errors, provenance_ctx


def collect_vllm(
    num_processes: int,
    ops: list[str] | None = None,
    limit: int | None = None,
    shuffle: bool = False,
    resume_options: dict | None = None,
    model_path: str | None = None,
    case_plan=None,
    sm_version: int | None = None,
    case_filters: list[str] | None = None,
):
    """Collect performance data for vLLM"""
    from collector.version_resolver import build_collections

    if _cuda_available():
        from collector.vllm.registry import REGISTRY
    elif _xpu_available():
        from collector.vllm.registry import REGISTRY_XPU as REGISTRY
    else:
        raise RuntimeError("No supported hardware detected. Neither CUDA nor XPU is available.")

    try:
        from vllm.version import __version__ as vllm_version

        version = vllm_version
    except Exception:
        logger.exception("vLLM is not installed. Please install it from https://github.com/vllm-project/vllm")
        return None, None

    from collector.framework_manifest import require_collector_runtime

    requested_ops = set(ops if ops is not None else (case_plan.ops if case_plan is not None else []))
    wideep_ops = {entry.op for entry in _wideep_registry_for_backend("vllm")}
    runtime = require_collector_runtime("vllm", version, requested_ops=requested_ops, wideep_ops=wideep_ops)

    registry = _registry_with_requested_wideep(REGISTRY, "vllm", ops, case_plan)
    collections = build_collections(registry, "vllm", version, ops, logger=logger)
    all_errors = collect_ops(
        num_processes,
        collections,
        version,
        limit=limit,
        shuffle=shuffle,
        backend="vllm",
        resume_options=resume_options,
        model_path=model_path,
        case_plan=case_plan,
        sm_version=sm_version,
        case_filters=case_filters,
    )

    generate_collection_summary(all_errors, "vllm", version)
    provenance_ctx = {
        "framework": runtime.framework,
        "installed_version": version,
        "runtime": runtime,
        "collections": collections,
    }
    return all_errors, provenance_ctx


def collect_trtllm(
    num_processes: int,
    ops: list[str] | None = None,
    limit: int | None = None,
    shuffle: bool = False,
    resume_options: dict | None = None,
    model_path: str | None = None,
    case_plan=None,
    sm_version: int | None = None,
    case_filters: list[str] | None = None,
):
    """Collect performance data for TensorRT LLM with enhanced error tracking"""
    from collector.trtllm.registry import REGISTRY
    from collector.version_resolver import build_collections

    os.environ["TLLM_LOG_LEVEL"] = "ERROR"
    os.environ["TRTLLM_DG_ENABLED"] = "1"
    os.environ["FLASHINFER_LOG_LEVEL"] = "ERROR"

    try:
        with (
            open(os.devnull, "w") as _null,
            contextlib.redirect_stdout(_null),
            contextlib.redirect_stderr(_null),
        ):
            import tensorrt_llm
        version = tensorrt_llm.__version__
        logger.info(f"TensorRT LLM version: {version}")
    except Exception:
        logger.exception("TensorRT LLM is not installed")
        return None, None

    from collector.framework_manifest import require_collector_runtime

    requested_ops = set(ops if ops is not None else (case_plan.ops if case_plan is not None else []))
    wideep_ops = {entry.op for entry in _wideep_registry_for_backend("trtllm")}
    runtime = require_collector_runtime("trtllm", version, requested_ops=requested_ops, wideep_ops=wideep_ops)

    registry = _registry_with_requested_wideep(REGISTRY, "trtllm", ops, case_plan)
    collections = build_collections(registry, "trtllm", version, ops, logger=logger)
    all_errors = collect_ops(
        num_processes,
        collections,
        version,
        limit=limit,
        shuffle=shuffle,
        backend="trtllm",
        resume_options=resume_options,
        model_path=model_path,
        case_plan=case_plan,
        sm_version=sm_version,
        case_filters=case_filters,
    )

    generate_collection_summary(all_errors, "trtllm", version)
    provenance_ctx = {
        "framework": runtime.framework,
        "installed_version": version,
        "runtime": runtime,
        "collections": collections,
    }
    return all_errors, provenance_ctx


def generate_collection_summary(all_errors, backend, version):
    """Generate comprehensive collection summary"""
    summary = {
        "backend": backend,
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "total_errors": len(all_errors),
        "errors_by_module": {},
        "errors_by_type": {},
        "errors_by_group": {},
    }

    for error in all_errors:
        module = error.get("module", "unknown")
        error_type = error.get("error_type", "unknown")

        summary["errors_by_module"][module] = summary["errors_by_module"].get(module, 0) + 1
        summary["errors_by_type"][error_type] = summary["errors_by_type"].get(error_type, 0) + 1
        group = error.get("group")
        if group:
            group_key = f"{module}:{group}"
            summary["errors_by_group"][group_key] = summary["errors_by_group"].get(group_key, 0) + 1

    log_dir = os.environ.get("COLLECTOR_LOG_DIR", "")

    # Save summary
    summary_file = f"{log_dir}/collection_summary_{backend}.json"
    with open(summary_file, "w") as f:
        json.dump({"summary": summary, "errors": all_errors}, f, indent=2)

    # Print summary
    logger.info("=" * 60)
    logger.info(f"COLLECTION SUMMARY - {backend} v{version}")
    logger.info("=" * 60)
    logger.info(f"Total errors: {summary['total_errors']}")

    if summary["errors_by_module"]:
        logger.info("\nErrors by module:")
        for module, count in sorted(summary["errors_by_module"].items()):
            logger.info(f"  {module}: {count}")

    if summary["errors_by_type"]:
        logger.info("\nErrors by type:")
        for error_type, count in sorted(summary["errors_by_type"].items()):
            logger.info(f"  {error_type}: {count}")

    if summary["errors_by_group"]:
        logger.info("\nErrors by (model, dtype) group — whole groups failing need fixing:")
        for group, count in sorted(summary["errors_by_group"].items(), key=lambda item: -item[1]):
            logger.info(f"  {group}: {count}")

    logger.info(f"\nDetailed error report saved to: {summary_file}")


def _all_op_names() -> list[str]:
    """Collect all unique op names across normal and WideEP registries."""
    from collector.sglang.registry import REGISTRY as SGLANG_REG
    from collector.trtllm.registry import REGISTRY as TRTLLM_REG
    from collector.vllm.registry import REGISTRY as VLLM_REG

    seen = set()
    ops = []
    registries = [
        TRTLLM_REG,
        VLLM_REG,
        SGLANG_REG,
        _wideep_registry_for_backend("trtllm"),
        _wideep_registry_for_backend("vllm"),
        _wideep_registry_for_backend("sglang"),
    ]
    for reg in registries:
        for entry in reg:
            if entry.op not in seen:
                seen.add(entry.op)
                ops.append(entry.op)
    return ops


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _git_collector_ref(repo_root: Path) -> str:
    """The repo SHA the collector ran from (design §5), "unknown" outside a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        logger.warning(f"collector_ref: `git rev-parse HEAD` failed ({error}); recording 'unknown'")
        return "unknown"


def _split_image_digest(image_ref: str) -> tuple[str, str | None]:
    """Split "repo/image:tag@sha256:<hex>" into (image, digest); digest is None for bare internal images."""
    image, sep, digest = image_ref.partition("@")
    return image, (digest if sep else None)


def _write_collector_provenance(
    output_root: Path,
    converted: list[Path],
    provenance_ctx: dict,
    run_errors: list[dict],
    *,
    backend: str,
    checkpoint_dir: str,
) -> None:
    """Write collection_meta.yaml (design §5) flat beside the just-finalized parquet.

    Reuses the checkpoint files ResumeCheckpoint already persisted per op
    (case ids + unresolved failures) instead of re-plumbing case-id tracking
    through parallel_run.
    """
    import pyarrow.parquet as pq
    import yaml

    from collector import provenance

    collections = provenance_ctx.get("collections") or []
    ops_by_table: dict[str, list[str]] = {}
    module_by_table: dict[str, str] = {}
    for collection in collections:
        table = Path(str(collection["perf_filename"])).stem
        full_name = f"{collection['name']}.{collection['type']}"
        ops_by_table.setdefault(table, []).append(full_name)
        module_by_table.setdefault(table, collection["module"])

    module_failure_names = {e["module"] for e in run_errors if e.get("error_type") == "ModuleCollectionFailure"}
    checkpoint_root = Path(checkpoint_dir).expanduser().resolve() / backend
    closures = provenance.load_closures(_REPO_ROOT / "collector" / "hash_closures.yaml")
    collector_ref = _git_collector_ref(_REPO_ROOT)

    runtime = provenance_ctx["runtime"]
    image, image_digest = _split_image_digest(runtime.image())
    runtime_meta = {"framework": runtime.framework, "version": runtime.version, "image": image}
    if image_digest:
        runtime_meta["image_digest"] = image_digest
    collected_at = datetime.now().strftime("%Y-%m-%d")

    tables: dict[str, dict] = {}
    for parquet_path in converted:
        table = parquet_path.stem
        full_names = ops_by_table.get(table)
        module = module_by_table.get(table)
        if not full_names or module is None:
            logger.warning(f"collection_meta: {table} has no registry mapping this run; skipping its provenance entry")
            continue

        case_ids: set[str] = set()
        unresolved_failed = 0
        for full_name in full_names:
            checkpoint_path = checkpoint_root / f"{full_name}.json"
            if not checkpoint_path.exists():
                continue
            try:
                with open(checkpoint_path) as checkpoint_file:
                    checkpoint_data = json.load(checkpoint_file)
            except Exception as error:
                logger.warning(f"collection_meta: failed to read checkpoint {checkpoint_path}: {error}")
                continue
            done = checkpoint_data.get("done", [])
            failed = checkpoint_data.get("failed", [])
            case_ids.update(done)
            case_ids.update(failed)
            unresolved_failed += len(failed)

        if not case_ids:
            # ResumeCheckpoint only writes a checkpoint file after >=1 case is
            # marked done/failed (flush is dirty-gated), so empty case_ids
            # means this table has ZERO checkpoint evidence: every op's
            # checkpoint is missing or unreadable (or was hand-emptied).
            # Finalized parquet with zero attempted cases is unattestable —
            # fail closed instead of writing a fabricated 'complete' sidecar
            # whose case_plan_hash covers an empty case set.
            raise RuntimeError(
                f"collection_meta: table '{table}' has finalized parquet ({parquet_path}) but no "
                f"readable checkpoint evidence for any of its ops ({', '.join(full_names)}) under "
                f"{checkpoint_root}. Zero attempted cases cannot explain produced parquet; writing "
                "a sidecar here would attest provenance that was never observed. Verify the "
                "checkpoint dir matches the one this collection ran with."
            )

        tables[table] = {
            "collector_ref": collector_ref,
            "collector_hash": provenance.collector_hash(module, _REPO_ROOT, closures),
            "case_plan_hash": provenance.case_plan_hash(sorted(case_ids)),
            "collected_at": collected_at,
            "rows": pq.read_metadata(parquet_path).num_rows,
            "status": provenance.derive_table_status(
                unresolved_failed_count=unresolved_failed,
                had_module_failure=any(full_name in module_failure_names for full_name in full_names),
            ),
        }

    if not tables:
        return

    # A prior invocation against the same scratch dir (e.g. --ops split across
    # runs) may have already written a sidecar for other tables; preserve them.
    existing_meta = output_root / "collection_meta.yaml"
    if existing_meta.exists():
        try:
            existing_doc = yaml.safe_load(existing_meta.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as error:
            logger.warning(f"collection_meta: could not parse existing {existing_meta}, overwriting it: {error}")
            existing_doc = {}
        if existing_doc.get("provenance") == "legacy":
            raise RuntimeError(
                f"{output_root}: existing collection_meta.yaml is a legacy-tier sidecar "
                "(provenance: legacy). A fresh collection finalizing into this directory "
                "must not silently merge into it — that would drop the legacy tier tag. "
                "Remove the legacy sidecar first if this directory is being deliberately "
                "replaced by a new collection."
            )
        existing_tables = existing_doc.get("tables") or {}
        if isinstance(existing_tables, dict):
            tables = {**existing_tables, **tables}

    meta_path = provenance.write_collection_meta(output_root, runtime_meta, tables)
    logger.info(f"Wrote collector provenance sidecar: {meta_path}")


def main():
    global logger
    parser = argparse.ArgumentParser(description="Collect performance data for backends")
    parser.add_argument("--backend", type=str, choices=["trtllm", "sglang", "vllm"], default="trtllm")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--ops",
        nargs="*",
        type=str,
        choices=_all_op_names(),
        help="Run only specified collection items. Leave empty to run all. "
        "Available ops vary by backend — see backend-specific registry.py for details.",
        default=None,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: randomly sample 4 test cases per op to verify the collector runs end-to-end",
    )
    parser.add_argument(
        "--measure_power",
        action="store_true",
        help="Enable power monitoring during kernel execution (samples at 100ms intervals)",
    )
    parser.add_argument(
        "--power_test_duration_sec",
        type=float,
        default=1.0,
        help="Minimum duration for kernel runs when power measurement is enabled (default: 1.0s)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume collection from checkpoint, skipping passed and failed tasks",
    )
    parser.add_argument(
        "--resume-retry-failed",
        action="store_true",
        help="When resuming, retry previously failed tasks instead of skipping them. Requires --resume.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=".collector_checkpoint",
        help="Directory for per-module resume checkpoints (default: .collector_checkpoint)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of test cases per collection (useful for debugging)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle test cases before applying --limit (uses seed 42 for reproducibility)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Collector v2 model path (for example 'MiniMaxAI/MiniMax-M2.5'). "
        "When set, collect.py resolves collector/cases/models/<architecture>_cases.yaml by model alias, "
        "then runs only the planned ops/cases.",
    )
    parser.add_argument(
        "--model-architecture",
        type=str,
        default=None,
        help="Collector v2 model architecture (for example 'Qwen3MoeForCausalLM'). "
        "Defaults to resolving the architecture case file from --model-path aliases.",
    )
    parser.add_argument(
        "--model-cases",
        type=str,
        default=None,
        help="Optional path to a model cases YAML file. Defaults to collector/cases/models/<architecture>_cases.yaml.",
    )
    parser.add_argument(
        "--model-cases-full",
        action="store_true",
        help="Collector v2 full mode: aggregate base op cases plus every model cases YAML file.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU type for resolving hardware capability floors, for example b200_sxm. "
        "The SM version is read from src/aiconfigurator/systems/<gpu>.yaml unless --sm is provided.",
    )
    parser.add_argument(
        "--sm",
        type=int,
        default=None,
        help="Explicit SM version for hardware capability floors, for example 100. "
        "Overrides --gpu SM resolution; defaults to the local device capability.",
    )
    parser.add_argument(
        "--case-filter",
        action="append",
        dest="case_filters",
        default=None,
        metavar="SUBSTR",
        help="Run only cases whose string form contains SUBSTR (repeatable, OR semantics). "
        "Ephemeral healing filter — never persisted to YAML.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the collector v2 case plan and exit without running collectors.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile the collector run and save output ",
    )
    parser.add_argument(
        "--keep-csv",
        action="store_true",
        help="Keep collector CSV staging files instead of finalizing *_perf.txt outputs to parquet.",
    )
    args = parser.parse_args()
    ops = args.ops
    case_plan = None
    logger_message = None
    if args.plan_only and not (args.model_path or args.model_architecture or args.model_cases or args.model_cases_full):
        parser.error("--plan-only requires --model-path, --model-architecture, --model-cases, or --model-cases-full")
    if args.model_path or args.model_architecture or args.model_cases or args.model_cases_full:
        from collector.model_cases import build_collection_case_plan

        if args.model_path:
            os.environ["COLLECTOR_MODEL_PATH"] = args.model_path
        else:
            os.environ.pop("COLLECTOR_MODEL_PATH", None)

        case_plan = build_collection_case_plan(
            backend=args.backend,
            model_path=args.model_path,
            model_architecture=args.model_architecture,
            gpu_type=args.gpu,
            sm_version=args.sm,
            model_cases_path=args.model_cases,
            full=args.model_cases_full,
        )
        if case_plan.model_path:
            os.environ["COLLECTOR_MODEL_PATH"] = case_plan.model_path

        planned_ops = case_plan.ops
        if args.ops is None:
            ops = planned_ops
        else:
            requested_ops = set(args.ops)
            ops = [op for op in planned_ops if op in requested_ops]
            missing_ops = requested_ops - set(ops)
            if missing_ops:
                parser.error(
                    "Requested ops are not present in the collector v2 case plan: " + ", ".join(sorted(missing_ops))
                )

        if (args.model_path or args.model_architecture) and not case_plan.model_cases_paths:
            logger_message = (
                "No collector v2 model cases YAML found for "
                f"model_path={args.model_path!r}, model_architecture={args.model_architecture!r}; "
                "using base op cases only plus legacy model filtering."
            )

        if args.plan_only:
            log_dict = case_plan.to_log_dict()
            log_dict["ops"] = ops
            print(json.dumps(log_dict, indent=2))
            return
    else:
        os.environ.pop("COLLECTOR_MODEL_PATH", None)

    # Setup logging - debug flag is handled inside setup_logging
    if logger is None:
        if args.model_cases_full:
            log_scope = ["model_cases_full"]
        else:
            log_scope = ops if ops else ["all"]
        logger = setup_logging(scope=log_scope, debug=args.debug)
    elif args.debug:
        # Update log level if debug flag changed
        setup_logging(debug=args.debug)

    if logger_message:
        logger.warning(logger_message)
    if case_plan is not None:
        logger.info("Collector v2 case plan active:")
        for key, value in case_plan.to_log_dict().items():
            logger.info(f"  {key}: {value}")
        if ops and args.ops is None:
            logger.info(f"  expanded to model-specific ops: {ops}")
    elif args.model_path:
        logger.info(f"Legacy model filter active: collecting only for '{args.model_path}'")

    # Hardware capability floor target: explicit --sm / --gpu wins, otherwise
    # detect from the local device (None on XPU -> filter is permissive).
    from collector.capabilities import detect_sm_version
    from collector.model_cases import resolve_sm_version

    sm_version = (
        case_plan.sm_version
        if case_plan is not None and case_plan.sm_version is not None
        else resolve_sm_version(gpu_type=args.gpu, sm_version=args.sm)
    )
    if sm_version is None:
        sm_version = detect_sm_version()
    logger.info(f"Hardware capability floors target SM version: {sm_version}")

    resume_options = {
        "resume": args.resume,
        "checkpoint_dir": args.checkpoint_dir,
        "retry_failed": args.resume_retry_failed,
    }
    if args.resume_retry_failed and not args.resume:
        parser.error("--resume-retry-failed requires --resume")
    if args.resume:
        logger.info(
            f"Resume enabled: dir={Path(args.checkpoint_dir).expanduser()}"
            + (" (retrying previously failed tasks)" if args.resume_retry_failed else "")
        )

    _require_torch()

    # Determine number of processes (0 = sequential mode for profiling)
    if args.profile:
        num_processes = 0
        logger.info("Starting collection in sequential mode (profiling enabled)")
    else:
        num_processes = get_device_module().device_count()
        logger.info(f"Starting collection with {num_processes} GPU processes")

    # Set environment variables for worker processes
    if args.measure_power:
        os.environ["COLLECTOR_MEASURE_POWER"] = "true"
        os.environ["COLLECTOR_POWER_MIN_DURATION"] = str(args.power_test_duration_sec)
        logger.info(f"Power monitoring enabled (min duration: {args.power_test_duration_sec}s)")
    else:
        os.environ["COLLECTOR_MEASURE_POWER"] = "false"

    # Suppress torch operator override warnings in spawned workers
    # (env var takes effect at interpreter startup, before any module imports)
    os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:torch.library"

    shuffle = args.shuffle
    limit = args.limit
    if args.smoke:
        shuffle = True
        limit = args.limit if args.limit is not None else 4
        logger.info(f"Smoke test mode enabled — sampling {limit} random test cases per op")

    # Warn if profiling without limit (profiling can be very slow)
    if args.profile and limit is None:
        logger.warning(
            "Profiling is enabled but --limit is not set. "
            "Profiling all test cases can be very slow. "
            "Consider using --limit to restrict the number of test cases."
        )

    # Disable core dumps — GPU crashes are expected and handled; core files waste disk.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    # Only set multiprocessing start method if not profiling (profiling uses sequential mode via num_processes=0)
    if not args.profile:
        mp.set_start_method("spawn")

    output_root = Path.cwd()
    existing_perf_outputs = {path.resolve(): path.stat().st_mtime_ns for path in find_perf_csv_outputs(output_root)}

    def was_touched_by_run(path: Path) -> bool:
        resolved = path.resolve()
        return resolved not in existing_perf_outputs or path.stat().st_mtime_ns != existing_perf_outputs[resolved]

    # Use profiling context manager
    with ProfilerContext(args.backend, enabled=args.profile):
        collect_backend = {"trtllm": collect_trtllm, "sglang": collect_sglang, "vllm": collect_vllm}[args.backend]
        run_errors, provenance_ctx = collect_backend(
            num_processes,
            ops,
            limit=limit,
            shuffle=shuffle,
            resume_options=resume_options,
            model_path=case_plan.model_path if case_plan is not None else None,
            case_plan=case_plan,
            sm_version=sm_version,
            case_filters=args.case_filters,
        )

    converted: list[Path] = []
    if args.keep_csv:
        logger.info("Keeping collector CSV staging files because --keep-csv was passed")
    else:
        touched_perf_outputs = [path for path in find_perf_csv_outputs(output_root) if was_touched_by_run(path)]
        if touched_perf_outputs:
            logger.info(
                "Finalizing collector CSV staging files as parquet:\n  "
                + "\n  ".join(str(path) for path in touched_perf_outputs)
            )
        converted = finalize_perf_files(touched_perf_outputs)
        if converted:
            logger.info(f"Finalized {len(converted)} collector perf files as parquet")

    if converted and provenance_ctx is not None:
        _write_collector_provenance(
            output_root,
            converted,
            provenance_ctx,
            run_errors or [],
            backend=args.backend,
            checkpoint_dir=args.checkpoint_dir,
        )

    # A ModuleCollectionFailure means an op failed before running a single case
    # (population raised, or the run infrastructure crashed) — the op collected
    # nothing. Exit non-zero AFTER finalization so partial data from other ops
    # is still packaged, but the job is not reported as a clean success.
    module_failures = sorted(
        {e["module"] for e in (run_errors or []) if e.get("error_type") == "ModuleCollectionFailure"}
    )
    if module_failures:
        logger.error("Module-level collection failures (no cases ran): " + ", ".join(module_failures))
        raise SystemExit(1)


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    main()

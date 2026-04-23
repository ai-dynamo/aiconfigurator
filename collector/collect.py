# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import torch
from tqdm import tqdm

setup_warning_filters()

import argparse
import cProfile
import io
import json
import multiprocessing as mp
import pstats
import signal
import time
import traceback
from datetime import datetime
from pathlib import Path

from helper import EXIT_CODE_RESTART, create_test_case_id, save_error_report, setup_logging, setup_signal_handlers

logger = None
RESUME_SCHEMA_VERSION = "collector-resume-v1"
STALL_THRESHOLD = 30  # iterations (x 0.5 s sleep = 15 s) before stall bailout


class ResumeCheckpoint:
    """Tracks which tasks are done so a collection run can be resumed.

    Always writes checkpoint files.  When ``--resume`` is passed the existing
    checkpoint is loaded and done tasks are skipped; otherwise the checkpoint
    is overwritten from scratch (so a future ``--resume`` can pick up).
    """

    FLUSH_INTERVAL_SEC = 2.0

    def __init__(self, backend: str, module_name: str, run_func_name: str, checkpoint_dir: str):
        self.module_name = module_name
        self._dirty = False
        self._last_flush = 0.0
        self._metadata = {
            "schema": RESUME_SCHEMA_VERSION,
            "backend": backend,
            "module": module_name,
            "run_func": run_func_name,
        }
        self._done: set[str] = set()

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

        for key in ("schema", "backend", "module", "run_func"):
            if data.get(key) != self._metadata[key]:
                raise RuntimeError(
                    f"{self.module_name}: checkpoint mismatch "
                    f"({key}: {data.get(key)} != {self._metadata[key]}). "
                    "Run without --resume to start fresh."
                )

        self._done = set(data.get("done", []))
        logger.info(f"{self.module_name}: loaded {len(self._done)} completed tasks from checkpoint")

    # -- public API -------------------------------------------------------

    def filter_done(self, task_infos: list[dict]) -> list[dict]:
        """Return only tasks that are not yet done."""
        runnable = [t for t in task_infos if t["id"] not in self._done]
        skipped = len(task_infos) - len(runnable)
        if skipped:
            logger.info(f"{self.module_name}: skipping {skipped} done tasks, running {len(runnable)}")
        return runnable

    def mark_done(self, task_id: str):
        self._done.add(task_id)
        self._dirty = True
        self.flush()

    def flush(self, force: bool = False):
        if not self._dirty:
            return
        now = time.time()
        if not force and (now - self._last_flush) < self.FLUSH_INTERVAL_SEC:
            return

        data = {**self._metadata, "updated_at": datetime.now().isoformat(), "done": sorted(self._done)}
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


def collect_module_safe(module_name, test_type, get_test_cases_func, run_func, num_processes, resume_options=None):
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
    device = torch.device(f"{get_device_str()}:{device_id}")
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
            func(*task, device=device)
            worker_logger.debug(f"Completed task {task_id}")
        except Exception as e:
            # Build comprehensive error info
            error_info = {
                "module": module_name,
                "device_id": device_id,
                "task_id": task_id,
                "task_params": str(task),
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat(),
            }

            # Report error to queue BEFORE any exit
            if error_queue:
                error_queue.put(error_info)

            worker_logger.exception(f"Task {task_id} failed")

            # Force flush logs before any potential exit
            for handler in worker_logger.handlers:
                handler.flush()

            # This error is could be fatal and require a process restart.
            if isinstance(e, torch.AcceleratorError):
                worker_logger.warning(
                    f"Fatal AcceleratorError encountered on task {task_id}. "
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
            # All three writes below use synchronous manager RPCs, so they
            # are guaranteed to complete before the worker picks up the next
            # task.  This means even if the *next* task kills the process
            # via signal, the bookkeeping for *this* task is already safe.
            if done_tasks is not None:
                try:
                    done_tasks[task_id] = True
                except Exception:
                    pass

            with lock:
                progress_value.value += 1
            if current_task_ids is not None:
                current_task_ids[device_id] = None

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
    )

    if resume_options and resume_options.get("resume"):
        resume_tracker.load_existing()
        task_infos = resume_tracker.filter_done(raw_task_infos)
    else:
        task_infos = raw_task_infos

    if not task_infos:
        logger.info(f"{module_name}: no tasks to run")
        return []

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
            resume_tracker.mark_done(task_id)
            try:
                del done_tasks[task_id]
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
            device = torch.device("cuda:0")
            torch.cuda.set_device(0)

            for task_info in task_infos:
                task_id = task_info["id"]
                task_params = task_info["params"]

                try:
                    func(*task_params, device=device)
                    resume_tracker.mark_done(task_id)
                except Exception as e:
                    error_info = {
                        "module": module_name,
                        "device_id": 0,
                        "task_id": task_id,
                        "task_params": str(task_params),
                        "error_type": type(e).__name__,
                        "error_message": str(e),
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

                    crash_error = create_process_exit_error(i, exit_code)
                    if crash_error:
                        errors.append(crash_error)
                        process_stats[i]["errors"].append("process_exit")
                        pbar.set_postfix({"errors": len(errors)})
                        last_error_count = len(errors)
                        if active_task_id is not None:
                            resume_tracker.mark_done(active_task_id)
                            current_task_ids[i] = None
                            with lock:
                                progress_value.value += 1

                    if process_stats[i]["restarts"] > 8192:
                        logger.error(f"Process {i} exceeded restart limit, not restarting")
                        continue

                    remaining = len(task_infos) - progress_value.value
                    if remaining > 0:
                        need_sentinel = consumed_sentinel.get(i, False)
                        consumed_sentinel[i] = False
                        processes[i] = start_process(i)
                        if need_sentinel:
                            queue.put(None)
                    else:
                        processes[i] = p  # keep dead process ref, skip restart

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
        p.join(timeout=42)
        if p.is_alive():
            logger.warning(f"Process {p.pid} did not terminate, forcing...")
            p.terminate()

    # Shutdown manager to clean up resources (semaphores, etc.)
    manager.shutdown()

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

    all_errors = []

    for collection in collections:
        try:
            module_name = collection["module"]
            get_module = __import__(module_name, fromlist=[collection["get_func"]])
            run_module = __import__(module_name, fromlist=[collection["run_func"]])

            # Fail this op explicitly if declared compatibility doesn't match runtime.
            if check_compat:
                declared = getattr(get_module, "__compat__", None)
                if declared:
                    try:
                        if not check_compat(declared, runtime_version):
                            if torch.xpu.is_available():
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

            def get_func_with_limit(get_func=get_func):
                cases = get_func()
                if shuffle:
                    rng = random.Random(shuffle_seed)
                    rng.shuffle(cases)
                if limit is not None:
                    cases = cases[:limit]
                return cases

            merged_resume = {**(resume_options or {}), "backend": backend}
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
):
    """Collect performance data for SGLang with enhanced error tracking"""
    from collector.sglang.registry import REGISTRY
    from collector.version_resolver import build_collections

    os.environ["FLASHINFER_LOG_LEVEL"] = "ERROR"

    try:
        from importlib.metadata import version as get_version

        version = get_version("sglang")
        logger.info(f"SGLang version: {version}")
    except Exception:
        logger.exception("SGLang is not installed")
        return

    collections = build_collections(REGISTRY, "sglang", version, ops, logger=logger)
    all_errors = collect_ops(
        num_processes,
        collections,
        version,
        limit=limit,
        shuffle=shuffle,
        backend="sglang",
        resume_options=resume_options,
    )

    generate_collection_summary(all_errors, "sglang", version)


def collect_vllm(
    num_processes: int,
    ops: list[str] | None = None,
    limit: int | None = None,
    shuffle: bool = False,
    resume_options: dict | None = None,
):
    """Collect performance data for vLLM"""
    from collector.version_resolver import build_collections

    if torch.cuda.is_available():
        from collector.vllm.registry import REGISTRY
    elif torch.xpu.is_available():
        from collector.vllm.registry import REGISTRY_XPU as REGISTRY
    else:
        raise RuntimeError("No supported hardware detected. Neither CUDA nor XPU is available.")

    try:
        from vllm.version import __version__ as vllm_version

        version = vllm_version
    except Exception:
        logger.exception("vLLM is not installed. Please install it from https://github.com/vllm-project/vllm")
        return

    collections = build_collections(REGISTRY, "vllm", version, ops, logger=logger)
    all_errors = collect_ops(
        num_processes, collections, version, limit=limit, shuffle=shuffle, backend="vllm", resume_options=resume_options
    )

    generate_collection_summary(all_errors, "vllm", version)


def collect_trtllm(
    num_processes: int,
    ops: list[str] | None = None,
    limit: int | None = None,
    shuffle: bool = False,
    resume_options: dict | None = None,
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
        return

    collections = build_collections(REGISTRY, "trtllm", version, ops, logger=logger)
    all_errors = collect_ops(
        num_processes,
        collections,
        version,
        limit=limit,
        shuffle=shuffle,
        backend="trtllm",
        resume_options=resume_options,
    )

    generate_collection_summary(all_errors, "trtllm", version)


def generate_collection_summary(all_errors, backend, version):
    """Generate comprehensive collection summary"""
    summary = {
        "backend": backend,
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "total_errors": len(all_errors),
        "errors_by_module": {},
        "errors_by_type": {},
    }

    for error in all_errors:
        module = error.get("module", "unknown")
        error_type = error.get("error_type", "unknown")

        summary["errors_by_module"][module] = summary["errors_by_module"].get(module, 0) + 1
        summary["errors_by_type"][error_type] = summary["errors_by_type"].get(error_type, 0) + 1

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

    logger.info(f"\nDetailed error report saved to: {summary_file}")


def _all_op_names() -> list[str]:
    """Collect all unique op names across all backend registries."""
    from collector.sglang.registry import REGISTRY as SGLANG_REG
    from collector.trtllm.registry import REGISTRY as TRTLLM_REG
    from collector.vllm.registry import REGISTRY as VLLM_REG

    seen = set()
    ops = []
    for reg in (TRTLLM_REG, VLLM_REG, SGLANG_REG):
        for entry in reg:
            if entry.op not in seen:
                seen.add(entry.op)
                ops.append(entry.op)
    return ops


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
        help="Resume collection from checkpoint, skipping already-attempted tasks",
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
        help="Filter collection to a single model (e.g. 'MiniMaxAI/MiniMax-M2.5'). "
        "Must match a model name in the test case config lists exactly. "
        "Best used together with --ops to target a specific op, since a model "
        "may only appear in one op's config list (e.g. MoE but not MLA). "
        "Default: collect all models.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile the collector run and save output ",
    )
    args = parser.parse_args()
    ops = args.ops

    if args.model_path:
        from collector.common_test_cases import get_all_model_names

        all_models = get_all_model_names()
        if args.model_path not in all_models:
            parser.error(
                f"Model '{args.model_path}' not found. Available models:\n" + "\n".join(f"  - {m}" for m in all_models)
            )
        os.environ["COLLECTOR_MODEL_PATH"] = args.model_path
    else:
        os.environ.pop("COLLECTOR_MODEL_PATH", None)

    # Setup logging - debug flag is handled inside setup_logging
    if logger is None:
        logger = setup_logging(scope=args.ops if args.ops else ["all"], debug=args.debug)
    elif args.debug:
        # Update log level if debug flag changed
        setup_logging(debug=args.debug)

    if args.model_path:
        logger.info(f"Model filter active: collecting only for '{args.model_path}'")

    resume_options = {
        "resume": args.resume,
        "checkpoint_dir": args.checkpoint_dir,
    }
    if args.resume:
        logger.info(f"Resume enabled: dir={Path(args.checkpoint_dir).expanduser()}")

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
        limit = 4
        logger.info("Smoke test mode enabled — sampling 4 random test cases per op")

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

    # Use profiling context manager
    with ProfilerContext(args.backend, enabled=args.profile):
        if args.backend == "trtllm":
            collect_trtllm(num_processes, ops, limit=limit, shuffle=shuffle, resume_options=resume_options)
        elif args.backend == "sglang":
            collect_sglang(num_processes, ops, limit=limit, shuffle=shuffle, resume_options=resume_options)
        elif args.backend == "vllm":
            collect_vllm(num_processes, ops, limit=limit, shuffle=shuffle, resume_options=resume_options)


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    main()

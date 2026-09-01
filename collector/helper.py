# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared collector runtime helpers.

This module is intentionally broad: collectors use it for benchmark timing,
power sampling, subprocess/restart control, device detection, perf logging,
case IDs, routing-logit synthesis, and small distributed-workload utilities.
Keep collector-specific policy in the per-framework collector modules or YAML
case files; this file should stay focused on reusable execution mechanics.
"""

import csv
import ctypes
import errno
import functools
import hashlib
import heapq
import json
import logging
import math
import multiprocessing as mp
import os
import shutil
import signal
import stat
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Optional, Protocol

import numpy as np

# Exit codes
EXIT_CODE_RESTART = 10  # Exit code to indicate restart is needed
PERF_TRANSACTION_FILENAME = ".perf-finalization.transaction.json"
SIDECAR_TRANSACTION_FILENAME = ".collection_meta.transaction.json"
SIDECAR_STAGING_FILENAME = ".collection_meta.pending.yaml"


class WorkerRestartSignal:
    """Return-value sentinel: task finished, recycle the worker process.

    Collector entrypoints legitimately return plain ints (row counts), so the
    executor must never interpret an int as a restart request — a task that
    logged exactly EXIT_CODE_RESTART rows would silently recycle its worker.
    Return WORKER_RESTART (or sys.exit(EXIT_CODE_RESTART)) instead.
    """


WORKER_RESTART = WorkerRestartSignal()


class PerfLogError(RuntimeError):
    """A measured performance row could not be durably persisted.

    Persistence is part of task completion, so this exception must propagate to
    the collector executor and classify the case as failed.  Keeping the
    fail-closed behavior here protects older callers that do not inspect the
    successful ``True`` return value from :func:`log_perf`.
    """


# Global NVML state per worker process
_NVML_INITIALIZED = False
_NVML_LOCK = threading.Lock()


def _parse_bool_env(env_var: str, default: bool = False) -> bool:
    """
    Robustly parse boolean environment variables.

    Accepts: "true", "True", "TRUE", "1", "yes", "Yes", "YES"
    Rejects: "false", "False", "FALSE", "0", "no", "No", "NO", or unset

    Args:
        env_var: Environment variable name to read
        default: Default value if variable is not set

    Returns:
        Boolean value
    """
    value = os.environ.get(env_var)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes")


def _ensure_nvml_initialized():
    """Initialize NVML once per process. Thread-safe."""
    global _NVML_INITIALIZED
    with _NVML_LOCK:
        if not _NVML_INITIALIZED:
            try:
                import pynvml as nvml

                nvml.nvmlInit()
                _NVML_INITIALIZED = True
                logging.getLogger(__name__).info("NVML initialized for power monitoring")
            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to initialize NVML: {e}")
                return False
        return _NVML_INITIALIZED


class PowerMonitor:
    """
    Background thread that samples GPU power using NVML at 100ms intervals.
    Designed to be reusable across multiple kernel runs within a worker process.
    """

    SAMPLE_INTERVAL_MS = 100  # Fixed sampling interval

    def __init__(self, device_id: int):
        """
        Args:
            device_id: CUDA device index to monitor
        """
        self.device_id = device_id
        self.interval_s = self.SAMPLE_INTERVAL_MS / 1000.0
        self._thread = None
        self._stop_event = threading.Event()
        self._samples = []  # List of (timestamp, power_mw) tuples
        self._lock = threading.Lock()
        self._nvml_handle = None
        self._power_limit_mw = None
        self._is_initialized = False

    def _init_handle(self):
        """Get NVML handle (called once, cached)."""
        if self._is_initialized:
            return True

        if not _ensure_nvml_initialized():
            return False

        try:
            import pynvml as nvml

            self._nvml_handle = nvml.nvmlDeviceGetHandleByIndex(self.device_id)
            self._power_limit_mw = nvml.nvmlDeviceGetPowerManagementLimit(self._nvml_handle)
            self._is_initialized = True
            return True
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to get NVML handle for device {self.device_id}: {e}")
            return False

    def start_sampling(self):
        """Start background sampling thread."""
        if not self._init_handle():
            return False

        # Clear previous samples
        with self._lock:
            self._samples.clear()

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._thread.start()
        return True

    def stop_sampling(self) -> dict | None:
        """Stop sampling and return statistics."""
        if self._thread is None:
            return None

        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None

        # Calculate statistics
        with self._lock:
            if not self._samples:
                return None
            power_values_w = [p_mw / 1000.0 for _, p_mw in self._samples]

        import numpy as np

        return {
            "power": float(np.mean(power_values_w)),
            "power_limit": float(self._power_limit_mw / 1000.0) if self._power_limit_mw else None,
        }

    def _monitoring_loop(self):
        """Background thread function that samples power every 100ms."""
        import pynvml as nvml

        while not self._stop_event.is_set():
            try:
                timestamp = time.time()
                power_mw = nvml.nvmlDeviceGetPowerUsage(self._nvml_handle)

                with self._lock:
                    self._samples.append((timestamp, power_mw))
            except Exception:
                # Skip failed samples silently
                pass

            # Wait for next interval
            self._stop_event.wait(self.interval_s)


@contextmanager
def benchmark_with_power(
    device,
    kernel_func,
    num_warmups: int = 3,
    num_runs: int = 6,
    repeat_n: int = 1,  # Default 1; GEMM files use 5
    measure_power: bool | None = None,  # Auto-detect from environment if None
    power_min_duration: float | None = None,  # Auto-detect from environment if None
    allow_graph_fail: bool = False,  # Enable graceful fallback on graph capture failure
    use_cuda_graph: bool = True,  # set False to force eager execution (ops whose captured
    # private pools retain memory across tasks — see collect_mla_module DSA context).
):
    """
    Context manager that handles warmup, graph capture, timing, and power monitoring.

    Args:
        device: torch.device object
        kernel_func: Callable that executes the kernel (e.g., lambda: gemm_op())
        num_warmups: Number of warmup iterations
        num_runs: Base number of runs (adjusted if measure_power=True)
        repeat_n: Number of repetitions per graph replay
        measure_power: Enable power monitoring (None = auto-detect from env)
        power_min_duration: Minimum duration for power measurement (None = auto-detect from env)
        allow_graph_fail: If True, gracefully fallback to eager execution when
                         CUDA graph capture fails. Power monitoring continues
                         to work in both paths. Default False for backward compatibility.
        use_cuda_graph: If False, skip CUDA graph capture entirely and run every
                         iteration eagerly. Defaults to True so all existing callers
                         (~30 collectors) keep graph-mode measurement. Callers whose
                         captured forward pass retains GiB-scale memory in the graph's
                         private pool across tasks should pass False.

    Yields:
        dict with keys:
            - 'latency_ms': Average latency in milliseconds
            - 'power_stats': Dict with power/power_limit (or None)
            - 'throttled': Boolean indicating if GPU was throttled
            - 'num_runs_executed': Actual number of runs performed
            - 'used_cuda_graph': Boolean indicating if graph was used
    """
    import torch

    # Auto-detect configuration from environment if not explicitly provided
    if measure_power is None:
        measure_power = _parse_bool_env("COLLECTOR_MEASURE_POWER", default=False)
    if power_min_duration is None:
        power_min_duration = float(os.environ.get("COLLECTOR_POWER_MIN_DURATION", "1.0"))

    # Adaptive num_runs calculation
    actual_num_runs = num_runs
    if measure_power:
        # Estimate single iteration time with warmup
        start_warmup = torch.cuda.Event(enable_timing=True)
        end_warmup = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_warmup.record()
        for _ in range(num_warmups):
            kernel_func()
        end_warmup.record()
        torch.cuda.synchronize()

        single_iter_time = start_warmup.elapsed_time(end_warmup) / num_warmups / 1000.0  # seconds

        # Adaptive duration: use shorter duration for very fast kernels to reduce memory pressure
        target_duration = power_min_duration
        if single_iter_time < 0.0001:  # < 0.1ms
            target_duration = min(power_min_duration, 0.3)
        actual_num_runs = max(num_runs, int(target_duration / (single_iter_time * repeat_n)) + 1)
        actual_num_runs = min(actual_num_runs, 3000)

        if actual_num_runs > 1000:
            logging.getLogger(__name__).warning(
                f"Kernel is very fast ({single_iter_time * 1000:.3f}ms), running {actual_num_runs} iterations"
            )
    else:
        # Normal warmup
        get_device_module().synchronize()
        for _ in range(num_warmups):
            kernel_func()
        get_device_module().synchronize()

    # ═══════════════════════════════════════════════════════════════════
    # CUDA Graph Capture with Optional Fallback
    # ═══════════════════════════════════════════════════════════════════
    g = None  # kept in scope so the finally block below can tear it down
    if torch.cuda.is_available() and use_cuda_graph:
        use_graph = True
        g = torch.cuda.CUDAGraph()

        try:
            with torch.cuda.graph(g):
                for _ in range(repeat_n):
                    kernel_func()
            torch.cuda.synchronize()
        except Exception as e:
            if allow_graph_fail:
                logging.getLogger(__name__).warning(f"CUDA graph capture failed: {e}. Falling back to eager execution.")
                g = None  # drop the partial capture so empty_cache can reclaim its private pool
                torch.cuda.empty_cache()
                use_graph = False
            else:
                # Standard behavior: re-raise exception
                raise
    else:
        use_graph = False

    # Everything from here to the yield holds live references to the captured
    # graph's private pool. A try/finally guarantees the graph (and therefore
    # its pool) is released before we return to the caller, regardless of
    # whether warmup raises, the yield body raises, or we finish cleanly.
    try:
        # ═══════════════════════════════════════════════════════════════
        # Warmup the ACTUAL execution path (after graph capture)
        # ═══════════════════════════════════════════════════════════════
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            for _ in range(num_warmups):
                if use_graph:
                    g.replay()
                else:
                    # Fallback: Direct execution matching actual execution path
                    for _ in range(repeat_n):
                        kernel_func()
            torch.cuda.synchronize()

        # Initialize power monitor if enabled
        power_monitor = None
        power_stats = None
        if measure_power:
            power_monitor = PowerMonitor(device.index)
            if not power_monitor.start_sampling():
                power_monitor = None  # Failed to start

        # Get initial clock info for throttling detection
        initial_clocks = None
        if measure_power and _NVML_INITIALIZED:
            try:
                import pynvml as nvml

                handle = nvml.nvmlDeviceGetHandleByIndex(device.index)
                initial_clocks = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM)
            except Exception:
                pass

        # ═══════════════════════════════════════════════════════════════════
        # Execute with Graph or Eager (both paths measured!)
        # ═══════════════════════════════════════════════════════════════════
        start_event = get_device_module().Event(enable_timing=True)
        end_event = get_device_module().Event(enable_timing=True)
        start_event.record()
        for _ in range(actual_num_runs):
            if use_graph:
                g.replay()
            else:
                # Fallback: Direct execution
                # This matches SGLang/VLLM pattern where kernel_func handles internal loops
                for _ in range(repeat_n):
                    kernel_func()
        end_event.record()
        get_device_module().synchronize()

        # Check for throttling
        throttled = False
        if initial_clocks is not None:
            try:
                import pynvml as nvml

                handle = nvml.nvmlDeviceGetHandleByIndex(device.index)
                final_clocks = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM)
                # If clocks dropped by more than 10%, likely throttled
                if final_clocks < initial_clocks * 0.9:
                    throttled = True
                    logging.getLogger(__name__).warning(
                        f"Clock throttling detected: {initial_clocks}MHz -> {final_clocks}MHz"
                    )
            except Exception:
                pass

        # Stop power monitoring
        if power_monitor:
            power_stats = power_monitor.stop_sampling()

        # Calculate latency
        latency_ms = start_event.elapsed_time(end_event) / actual_num_runs / repeat_n

        # Return results
        yield {
            "latency_ms": latency_ms,
            "power_stats": power_stats,
            "throttled": throttled,
            "num_runs_executed": actual_num_runs,
            "used_cuda_graph": use_graph,  # NEW: Inform caller which path was used
        }
    finally:
        # Drop the CUDA graph and reclaim its private memory pool. CUDA graph
        # captures sequester intermediate tensors into a private pool that
        # outlives Python-level GC of the CUDAGraph object in some PyTorch
        # versions — if we skip this explicit teardown, leaky ops (e.g. DSA
        # context via flashmla-sparse, which captures ~18 GiB of scratch)
        # accumulate pool memory across tasks until the worker saturates at
        # ~146 GiB pinned and subsequent tasks OOM at _ensure_workspace_size.
        if g is not None:
            g = None
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()


@contextmanager
def power_monitoring_only(device, measure_power: bool | None = None):
    """
    Lightweight context manager for TRT profiler cases.
    Only handles power monitoring, no timing/warmup.

    Args:
        device: torch.device object
        measure_power: Enable power monitoring (None = auto-detect from env)

    Yields:
        PowerMonitor instance or None
    """
    # Auto-detect from environment if not specified
    if measure_power is None:
        measure_power = _parse_bool_env("COLLECTOR_MEASURE_POWER", default=False)

    power_monitor = None

    if measure_power:
        power_monitor = PowerMonitor(device.index)
        if not power_monitor.start_sampling():
            power_monitor = None  # Failed to start
            logging.getLogger(__name__).warning("Failed to start power monitoring")

    try:
        yield power_monitor
    finally:
        # Cleanup happens after yield returns
        pass


def setup_signal_handlers(worker_id):
    """Setup signal handlers to log crashes."""
    import traceback as _tb

    logger = logging.getLogger(f"worker_{worker_id}")

    def signal_handler(signum, frame):
        try:
            try:
                sig_name = signal.Signals(signum).name
            except (ValueError, AttributeError):
                sig_name = str(signum)
            logger.error(f"Worker {worker_id} received {sig_name}")
            if frame is not None:
                logger.error("".join(_tb.format_stack(frame)))
            for handler in logger.handlers:
                handler.flush()
        finally:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    # Register handlers for common signals
    for sig in [signal.SIGTERM, signal.SIGABRT]:
        signal.signal(sig, signal_handler)

    # SIGSEGV might not be catchable on all platforms
    try:
        signal.signal(signal.SIGSEGV, signal_handler)
    except:
        pass


# Global tracking
_LOGGING_CONFIGURED = False
_LOG_DIR = None


def log_scope_dirname(scope) -> str:
    """Join the op scope for the log-directory name, capped for the filesystem.

    A long --ops list must not crash logging setup: filenames are limited to
    255 bytes on common filesystems, so summarize past the cap.
    """
    name = "+".join(scope)
    if len(name) > 80:
        name = f"{scope[0]}+{len(scope) - 1}ops"
    return name


def setup_logging(scope=["all"], debug=False, worker_id=None):
    """
    Setup structured logging - auto-configures based on process type

    Args:
        scope: types of operations targeted for collection
        debug: Enable debug logging (only used in main process)
        worker_id: If provided, configures logging for a worker process
    """
    global _LOGGING_CONFIGURED, _LOG_DIR

    # For worker processes
    if worker_id is not None:
        # Read configuration from environment
        debug = _parse_bool_env("COLLECTOR_DEBUG", default=False)
        log_dir = os.environ.get("COLLECTOR_LOG_DIR", "")

        if log_dir:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
                stdout_path = os.path.join(log_dir, "collector.log")
                stderr_path = os.path.join(log_dir, "collector_errors.log")
                so = open(stdout_path, "a", buffering=1)  # noqa: SIM115
                se = open(stderr_path, "a", buffering=1)  # noqa: SIM115
                os.dup2(so.fileno(), 1)
                os.dup2(se.fileno(), 2)
                sys.stdout = so
                sys.stderr = se
            except Exception:
                pass

        # Configure worker-specific logger
        logger = logging.getLogger(f"worker_{worker_id}")
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        logger.handlers.clear()

        # Console handler with worker ID
        console_formatter = logging.Formatter(
            f"[%(asctime)s] [%(levelname)s] [Worker-{worker_id}] [%(name)s] %(message)s"
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler - append to main log file
        if log_dir:
            file_formatter = logging.Formatter("%(asctime)s|%(levelname)s|Worker-%(name)s|%(funcName)s|%(message)s")
            file_handler = logging.FileHandler(f"{log_dir}/collector.log", mode="a")
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

            error_handler = logging.FileHandler(f"{log_dir}/collector_errors.log", mode="a")
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(file_formatter)
            logger.addHandler(error_handler)
        logging.captureWarnings(True)

        logger.propagate = False  # Prevent duplicate logs
        # Silence noisy third-party loggers even if debug is true
        _silence_noisy_loggers()

        # Configure root logger for libraries
        root = logging.getLogger()
        root.setLevel(logging.DEBUG if debug else logging.INFO)
        root.handlers.clear()

        return logger

    # Main process logging setup
    if _LOGGING_CONFIGURED and mp.current_process().name == "MainProcess":
        # Just update log level if already configured
        root = logging.getLogger()
        root.setLevel(logging.DEBUG if debug else logging.INFO)
        # Update environment for future workers
        os.environ["COLLECTOR_DEBUG"] = "true" if debug else "false"
        return root

    # Only configure once in main process
    if mp.current_process().name != "MainProcess":
        return logging.getLogger()

    # Create log directory
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _LOG_DIR = Path(f"{log_scope_dirname(scope)}_{time_stamp}")
    if not _LOG_DIR.is_dir():
        _LOG_DIR.mkdir()

    # Set environment variables for workers
    os.environ["COLLECTOR_DEBUG"] = "true" if debug else "false"
    os.environ["COLLECTOR_LOG_DIR"] = str(_LOG_DIR)

    # Create formatters
    console_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")

    file_formatter = logging.Formatter("%(asctime)s|%(levelname)s|%(name)s|%(funcName)s|%(message)s")

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Console handler (send to stdout to avoid clobbering tqdm on stderr)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(console_formatter)

    class _DropLifecycleNoise(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            if msg.startswith("Started worker process"):
                return False
            return not ("Process " in msg and " died (exit code" in msg)

    console_handler.addFilter(_DropLifecycleNoise())
    root_logger.addHandler(console_handler)

    # File handler for all logs
    file_handler = logging.FileHandler(f"{_LOG_DIR}/collector.log")
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Error file handler
    error_handler = logging.FileHandler(f"{_LOG_DIR}/collector_errors.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    root_logger.addHandler(error_handler)
    logging.captureWarnings(True)

    # Silence noisy third-party loggers globally
    _silence_noisy_loggers()

    _LOGGING_CONFIGURED = True

    return root_logger


def _silence_noisy_loggers():
    for name in ("matplotlib", "h5py", "datasets", "numexpr"):
        logging.getLogger(name).setLevel(logging.WARNING)
    for name in ("flashinfer", "tensorrt_llm"):
        logging.getLogger(name).setLevel(logging.ERROR)


def get_logging_config():
    """Get current logging configuration for passing to workers"""
    return {"debug": logging.getLogger().getEffectiveLevel() <= logging.DEBUG, "log_dir": _LOG_DIR}


def save_error_report(errors, filename):
    """Save error report"""
    with open(filename, "w") as f:
        json.dump(errors, f, indent=2)


def get_sm_version():
    """Get CUDA compute capability (SM version)"""
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            capability = torch.cuda.get_device_capability(device)
            return capability[0] * 10 + capability[1]
    except Exception:
        pass

    # fallback to cuda-python
    try:
        from cuda import cuda

        # Init
        (err,) = cuda.cuInit(0)
        if err != 0:
            raise RuntimeError(f"cuInit failed with error code: {err}")

        # Device
        err, cu_device = cuda.cuDeviceGet(0)
        if err != 0:
            raise RuntimeError(f"cuDeviceGet failed with error code: {err}")

        # Get target architecture
        err, sm_major = cuda.cuDeviceGetAttribute(
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cu_device
        )
        err, sm_minor = cuda.cuDeviceGetAttribute(
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cu_device
        )

        return sm_major * 10 + sm_minor
    except Exception as e:
        raise RuntimeError(f"Cannot get SM version: both PyTorch and cuda-python failed. Error: {e}") from e


def create_test_case_id(test_case, test_type, module_name):
    """Create a stable, cross-session identifier for a test case."""
    return f"{module_name}:{test_type}:{test_case}"


def log_perf(
    item_list: list[dict],
    framework: str,
    version: str,
    device_name: str,
    op_name: str,
    kernel_source: str,
    perf_filename: str,
    power_stats: dict | None = None,
) -> bool:
    lock_file = perf_filename + ".lock"

    # Try for 30s (300 * 0.1s). The old 1s window lost measured rows whenever
    # a sibling worker was wedged in a CUDA crash storm while other workers
    # queued behind the lock (H200 K3 moe 2026-08-01: 47 rows measured but
    # dropped). A worker SIGKILLed inside its critical section (host OOM
    # killer) skips `finally` and leaves the lock behind forever, so a lock
    # older than the stale threshold is broken instead of waited on.
    #
    # Break via rename, not unlink: with two waiters, an unlink-based break
    # lets waiter B (still holding its stat of the OLD lock) unlink the FRESH
    # lock waiter A just created, and two writers then interleave appends
    # inside the critical section. os.rename is atomic on POSIX, so exactly
    # one breaker wins the stale lock; the loser's rename raises ENOENT and
    # it simply retries against whatever fresh lock now exists.
    stale_lock_seconds = 60.0
    got_lock = False
    for _ in range(300):
        try:
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            got_lock = True
            break
        except OSError:
            try:
                if time.time() - os.path.getmtime(lock_file) > stale_lock_seconds:
                    broken_lock_file = f"{lock_file}.breaking-{os.getpid()}"
                    os.rename(lock_file, broken_lock_file)
                    print(f"Breaking stale lock for {perf_filename}")
                    os.unlink(broken_lock_file)
                    continue
            except OSError:
                # Lock vanished (or another breaker won the rename) between
                # the open attempt and the stat/rename — retry immediately.
                continue
            time.sleep(0.1)

    if not got_lock:
        message = f"Can not get lock for {perf_filename}"
        print(f"Error writing log: {message}")
        raise PerfLogError(message)

    staging_fd: int | None = None
    retained_attestation: PerfFileAttestation | None = None
    try:
        perf_path = Path(perf_filename)
        retained_path = collector_retained_path(perf_path)
        if retained_path.exists() or retained_path.is_symlink():
            if perf_path.exists() or perf_path.is_symlink():
                raise PerfLogError(f"Conflicting retained performance file for {perf_filename}")
            retained_state = retained_path.lstat()
            if not stat.S_ISREG(retained_state.st_mode) or retained_state.st_nlink != 1:
                raise PerfLogError(f"Unowned retained performance file for {perf_filename}")
            retained_attestation = _attest_regular_file(
                retained_path,
                expected_identity=(retained_state.st_dev, retained_state.st_ino),
                expected_mode=stat.S_IMODE(retained_state.st_mode),
            )
            _rename_noreplace(retained_path, perf_path)
            _attest_regular_file(
                perf_path,
                expected_identity=retained_attestation.identity,
                expected_digest=retained_attestation.digest,
                expected_mode=retained_attestation.mode,
            )
            open_flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            staging_fd = os.open(perf_path, open_flags)
            opened = os.fstat(staging_fd)
            current = perf_path.lstat()
            if (
                not stat.S_ISREG(opened.st_mode)
                or not stat.S_ISREG(current.st_mode)
                or (opened.st_dev, opened.st_ino) != retained_attestation.identity
                or (current.st_dev, current.st_ino) != retained_attestation.identity
                or opened.st_nlink != 1
                or current.st_nlink != 1
            ):
                raise PerfLogError(f"Retained performance file changed for {perf_filename}")
            os.ftruncate(staging_fd, 0)
            os.fsync(staging_fd)
        with (
            open(perf_filename, "a+", newline="") if staging_fd is None else os.fdopen(staging_fd, "r+", newline="")
        ) as f:
            staging_fd = None
            # Add header only if file is empty
            is_empty = os.fstat(f.fileno()).st_size == 0

            base_data = {
                "framework": framework,
                "version": version,
                "device": device_name,
                "op_name": op_name,
                "kernel_source": kernel_source,
            }

            # Get headers from first item if exists
            fieldnames = list(base_data.keys())
            if item_list:
                fieldnames += list(item_list[0].keys())
            # Add power_stats keys if present
            if power_stats:
                for key in ["power", "power_limit"]:
                    if key not in fieldnames:
                        fieldnames.append(key)

            # The first row freezes the staging schema. A resumed or batched
            # run must never append with a different optional-column setting
            # (most notably --measure_power), because DictWriter would emit
            # values in the NEW order under the OLD header. Validate under the
            # same writer lock before appending so the file remains unchanged
            # on mismatch and the caller can classify the failed persistence.
            if not is_empty:
                f.seek(0)
                existing_header = next(csv.reader(f), [])
                if existing_header != fieldnames:
                    message = (
                        f"Schema mismatch for {perf_filename}: "
                        f"existing header {existing_header}, requested header {fieldnames}. "
                        "Use the same measurement settings when resuming or start a fresh staging file."
                    )
                    print(f"Error writing log: {message}")
                    raise PerfLogError(message)
                f.seek(0, os.SEEK_END)

            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if is_empty:
                writer.writeheader()

            for item in item_list:
                row = base_data | item
                # Add power_stats values if present
                if power_stats:
                    for key in ["power", "power_limit"]:
                        row[key] = power_stats.get(key, "")
                writer.writerow(row)

            # Force disk write (for NFS)
            f.flush()
            os.fsync(f.fileno())
            if retained_attestation is not None:
                opened = os.fstat(f.fileno())
                current = perf_path.lstat()
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or not stat.S_ISREG(current.st_mode)
                    or (opened.st_dev, opened.st_ino) != retained_attestation.identity
                    or (current.st_dev, current.st_ino) != retained_attestation.identity
                    or opened.st_size != current.st_size
                    or stat.S_IMODE(opened.st_mode) != retained_attestation.mode
                    or stat.S_IMODE(current.st_mode) != retained_attestation.mode
                    or opened.st_nlink != 1
                    or current.st_nlink != 1
                ):
                    raise PerfLogError(f"Retained performance file changed for {perf_filename}")
    except PerfLogError:
        raise
    except Exception as e:
        print(f"Error writing log: {e}")
        raise PerfLogError(f"Failed to write {perf_filename}: {e}") from e
    finally:
        if staging_fd is not None:
            os.close(staging_fd)
        # Delete the lock file, even if writing crashed
        if got_lock and os.path.exists(lock_file):
            os.unlink(lock_file)

    return True


# Measured-value columns in a perf row. Everything else is the row's identity
# (shape/config/kernel) key. log_perf writes exactly these metrics — the timed
# ``latency`` plus the optional ``power``/``power_limit`` from power_stats.
PERF_METRIC_COLUMNS = ("latency", "power", "power_limit")


@dataclass(frozen=True)
class PerfFinalizationInfo:
    """Row contribution and exact byte/object identity of one staging file."""

    # Current-event rows that survive finalization. Compatible merges count
    # unique current identity keys; replacement paths count the written rows.
    new_rows: int
    merged_existing: bool
    # SHA-256 and descriptor identity captured from the bytes parsed into the
    # prepared parquet. Callers revalidate all three before publication.
    source_digest: str
    source_device: int
    source_inode: int


@dataclass(frozen=True)
class _PreparedPerfFile:
    """One fully rendered parquet awaiting batch publication."""

    source: Path
    target: Path
    temporary: Path
    temporary_device: int
    temporary_inode: int
    source_mode: int
    info: PerfFinalizationInfo
    merge_target: "PerfFileAttestation | None"
    merge_target_was_absent: bool


@dataclass(frozen=True)
class PerfFileAttestation:
    """Exact identity and bytes of one regular finalization artifact."""

    path: Path
    digest: str
    device: int
    inode: int
    mode: int

    @property
    def identity(self) -> tuple[int, int]:
        return self.device, self.inode


@dataclass(frozen=True)
class PerfPublication:
    """Prepared replacement plus the exact target state it may replace."""

    source: PerfFileAttestation
    target: Path
    prepared: PerfFileAttestation
    previous_target: PerfFileAttestation | None
    target_claim: Path | None
    info: PerfFinalizationInfo


class PerfPublicationTransaction(Protocol):
    """Durable owner for a collector parquet publication batch."""

    def prepare(self, publications: tuple[PerfPublication, ...]) -> None: ...

    def rollback_complete(self, publications: tuple[PerfPublication, ...]) -> None: ...

    def has_durable_journal(self) -> bool: ...


def convert_perf_csv_to_parquet(
    csv_file: str | os.PathLike,
    *,
    delete_source: bool = True,
    compression: str = "zstd",
    merge_existing: bool = False,
    finalization_info: dict[Path, PerfFinalizationInfo] | None = None,
) -> Path:
    """Convert a collector CSV staging file to parquet atomically.

    When ``merge_existing`` is True and a parquet already exists at the target,
    the new rows are merged into it instead of overwriting: the existing and new
    rows are concatenated (new last) and deduplicated on their identity key —
    every column except the measured metrics (``PERF_METRIC_COLUMNS``) — keeping
    the newest row per key. This makes finalization idempotent and accumulative:
    a resumed / ``--resume-retry-failed`` / batched collection extends the
    parquet instead of clobbering it with only the current run's subset.
    ``finalization_info``, when provided, is populated with the current event's
    finalized row contribution and whether an existing parquet took the
    compatible merge path. Compatible merges count unique current identity
    keys after deduplication; no-existing and schema-replacement paths count
    the rows actually written. The mapping is keyed by the resolved parquet
    path. Finalization deletes the source ``.txt``, so without merging a
    partial run after an earlier finalize would silently shrink the complete
    file. A full fresh run still yields the complete file either way because
    every identity key is re-measured and replaced.
    """
    csv_path = Path(csv_file)
    if csv_path.name == "INCOMPLETE.txt" or not csv_path.name.endswith("_perf.txt"):
        raise ValueError(f"Expected a collector perf CSV ending in _perf.txt, got {csv_path}")
    if not csv_path.exists() and not csv_path.is_symlink():
        raise FileNotFoundError(csv_path)
    return finalize_perf_files(
        [csv_path],
        delete_source=delete_source,
        compression=compression,
        merge_existing=merge_existing,
        finalization_info=finalization_info,
    )[0]


def _stream_digest(file, *, copy_to=None) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
        digest.update(chunk)
        if copy_to is not None:
            copy_to.write(chunk)
    return "sha256:" + digest.hexdigest()


def _fsync_directory(directory: Path) -> None:
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


@dataclass(frozen=True)
class LockedOutputRoot:
    """Capability for I/O bound to one flocked directory inode."""

    path: Path
    file_descriptor: int
    identity: tuple[int, int]

    def assert_canonical(self) -> None:
        opened = os.fstat(self.file_descriptor)
        try:
            current = self.path.lstat()
        except OSError as error:
            raise RuntimeError(f"Collector finalization output root changed while locked: {self.path}") from error
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(current.st_mode)
            or (opened.st_dev, opened.st_ino) != self.identity
            or (current.st_dev, current.st_ino) != self.identity
        ):
            raise RuntimeError(f"Collector finalization output root changed while locked: {self.path}")

    def entry(self, path: Path) -> str:
        path = path.absolute()
        name = path.name
        if path.parent != self.path or not name or name in (".", "..") or "\0" in name or Path(name).name != name:
            raise RuntimeError(f"Collector path is outside its locked output root: {path}")
        self.assert_canonical()
        return name

    def stat(self, path: Path) -> os.stat_result:
        return os.stat(self.entry(path), dir_fd=self.file_descriptor, follow_symlinks=False)

    def absent(self, path: Path) -> bool:
        try:
            self.stat(path)
        except FileNotFoundError:
            return True
        return False

    def open(self, path: Path, flags: int, mode: int = 0o600) -> int:
        return os.open(self.entry(path), flags, mode, dir_fd=self.file_descriptor)

    def rename_noreplace(self, source: Path, target: Path) -> None:
        source_name = self.entry(source)
        target_name = self.entry(target)
        _rename_noreplace_at(source_name, target_name, self.file_descriptor)
        self.assert_canonical()

    def replace(self, source: Path, target: Path) -> None:
        source_name = self.entry(source)
        target_name = self.entry(target)
        os.replace(
            source_name,
            target_name,
            src_dir_fd=self.file_descriptor,
            dst_dir_fd=self.file_descriptor,
        )
        os.fsync(self.file_descriptor)
        self.assert_canonical()

    def unlink(self, path: Path) -> None:
        os.unlink(self.entry(path), dir_fd=self.file_descriptor)
        os.fsync(self.file_descriptor)
        self.assert_canonical()

    def fsync(self) -> None:
        self.assert_canonical()
        os.fsync(self.file_descriptor)


@contextmanager
def perf_finalization_lifecycle(output_root: Path, *, resolve_path: bool = True):
    """Serialize preparation, publication, sidecar commit, and recovery."""
    import fcntl

    output_root = output_root.resolve() if resolve_path else output_root.absolute()
    open_flags = os.O_RDONLY
    open_flags |= getattr(os, "O_CLOEXEC", 0)
    open_flags |= getattr(os, "O_DIRECTORY", 0)
    open_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(output_root, open_flags)
    try:
        opened = os.fstat(directory_fd)
        current = output_root.lstat()
        identity = (opened.st_dev, opened.st_ino)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(current.st_mode)
            or (current.st_dev, current.st_ino) != identity
        ):
            raise RuntimeError(f"Collector finalization output root changed: {output_root}")
        fcntl.flock(directory_fd, fcntl.LOCK_EX)
        current = output_root.lstat()
        if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != identity:
            raise RuntimeError(f"Collector finalization output root changed after flock: {output_root}")
        locked_root = LockedOutputRoot(
            path=output_root,
            file_descriptor=directory_fd,
            identity=identity,
        )
        yield locked_root
        try:
            current = output_root.lstat()
        except OSError as error:
            raise RuntimeError(f"Collector finalization output root changed while locked: {output_root}") from error
        if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != identity:
            raise RuntimeError(f"Collector finalization output root changed while locked: {output_root}")
    finally:
        os.close(directory_fd)


def perf_transaction_artifact_paths(output_root: Path) -> tuple[Path, Path, Path]:
    """Return every marker that transfers preparation ownership to a journal."""
    return (
        output_root / PERF_TRANSACTION_FILENAME,
        output_root / SIDECAR_TRANSACTION_FILENAME,
        output_root / SIDECAR_STAGING_FILENAME,
    )


def _rename_noreplace(source: Path, target: Path) -> None:
    """Atomically move one path only while the destination is absent."""
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    target_bytes = os.fsencode(target)
    if sys.platform.startswith("linux"):
        try:
            rename = libc.renameat2
        except AttributeError as error:
            raise RuntimeError("Atomic no-replace rename is unavailable on this platform") from error
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, target_bytes, 1)
    elif sys.platform == "darwin":
        try:
            rename = libc.renamex_np
        except AttributeError as error:
            raise RuntimeError("Atomic no-replace rename is unavailable on this platform") from error
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source_bytes, target_bytes, 0x00000004)
    else:
        raise RuntimeError("Atomic no-replace rename is unavailable on this platform")
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP):
            raise RuntimeError("Atomic no-replace rename is unsupported by this filesystem")
        raise OSError(error_number, os.strerror(error_number), str(source), str(target))
    _fsync_directory(source.parent)
    if target.parent != source.parent:
        _fsync_directory(target.parent)


def _rename_noreplace_at(source: str, target: str, directory_fd: int) -> None:
    """Atomically rename two entries relative to one held directory."""
    if any(not name or name in (".", "..") or "\0" in name or Path(name).name != name for name in (source, target)):
        raise ValueError("Descriptor-relative rename requires direct entry names")
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    target_bytes = os.fsencode(target)
    if sys.platform.startswith("linux"):
        try:
            rename = libc.renameat2
        except AttributeError as error:
            raise RuntimeError("Atomic no-replace rename is unavailable on this platform") from error
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(directory_fd, source_bytes, directory_fd, target_bytes, 1)
    elif sys.platform == "darwin":
        try:
            rename = libc.renameatx_np
        except AttributeError as error:
            raise RuntimeError("Atomic no-replace rename is unavailable on this platform") from error
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(directory_fd, source_bytes, directory_fd, target_bytes, 0x00000004)
    else:
        raise RuntimeError("Atomic no-replace rename is unavailable on this platform")
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(error_number, os.strerror(error_number), target)
        unsupported_errors = {errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP}
        if sys.platform.startswith("linux"):
            unsupported_errors.add(errno.EINVAL)
        if error_number in unsupported_errors:
            raise RuntimeError("Atomic no-replace rename is unsupported by this filesystem")
        raise OSError(error_number, os.strerror(error_number), source, None, target)
    os.fsync(directory_fd)


def _lstat_path(path: Path, locked_root: LockedOutputRoot | None = None) -> os.stat_result:
    return locked_root.stat(path) if locked_root is not None else path.lstat()


def _open_path(
    path: Path,
    flags: int,
    mode: int = 0o600,
    *,
    locked_root: LockedOutputRoot | None = None,
) -> int:
    return locked_root.open(path, flags, mode) if locked_root is not None else os.open(path, flags, mode)


def _rename_path_noreplace(source: Path, target: Path, locked_root: LockedOutputRoot | None = None) -> None:
    if locked_root is not None:
        locked_root.rename_noreplace(source, target)
    else:
        _rename_noreplace(source, target)


def _replace_path(source: Path, target: Path, locked_root: LockedOutputRoot | None = None) -> None:
    if locked_root is not None:
        locked_root.replace(source, target)
    else:
        os.replace(source, target)
        _fsync_directory(target.parent)


def _unlink_path(path: Path, locked_root: LockedOutputRoot | None = None) -> None:
    if locked_root is not None:
        locked_root.unlink(path)
    else:
        path.unlink()


def _fsync_path_parent(path: Path, locked_root: LockedOutputRoot | None = None) -> None:
    if locked_root is not None:
        locked_root.fsync()
    else:
        _fsync_directory(path.parent)


def _attest_regular_file(
    path: Path,
    *,
    expected_identity: tuple[int, int] | None = None,
    expected_digest: str | None = None,
    expected_mode: int | None = None,
    locked_root: LockedOutputRoot | None = None,
) -> PerfFileAttestation:
    open_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    try:
        artifact_fd = _open_path(path, open_flags, locked_root=locked_root)
    except OSError as error:
        raise RuntimeError(f"Collector finalization artifact is not a regular file: {path}") from error
    with os.fdopen(artifact_fd, "rb") as artifact:
        opened = os.fstat(artifact.fileno())
        digest = _stream_digest(artifact)
    current = _lstat_path(path, locked_root)
    identity = (opened.st_dev, opened.st_ino)
    opened_mode = stat.S_IMODE(opened.st_mode)
    current_mode = stat.S_IMODE(current.st_mode)
    if (
        not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or identity != (current.st_dev, current.st_ino)
        or opened_mode != current_mode
        or (expected_identity is not None and identity != expected_identity)
        or (expected_digest is not None and digest != expected_digest)
        or (expected_mode is not None and opened_mode != expected_mode)
    ):
        raise RuntimeError(f"Collector finalization artifact changed: {path}")
    return PerfFileAttestation(
        path=path,
        digest=digest,
        device=opened.st_dev,
        inode=opened.st_ino,
        mode=opened_mode,
    )


def _snapshot_perf_target(
    target: Path,
    locked_root: LockedOutputRoot,
) -> tuple[PerfFileAttestation | None, Path | None]:
    if _path_absent(target, locked_root):
        return None, None
    previous_target = _attest_regular_file(target, locked_root=locked_root)
    target_claim = target.with_name(f".{target.name}.{uuid.uuid4().hex}.claim")
    if not _path_absent(target_claim, locked_root):
        raise RuntimeError(f"Collector parquet target claim already exists: {target_claim}")
    return previous_target, target_claim


def _prepare_perf_publications(
    prepared: list[_PreparedPerfFile],
    locked_roots: dict[Path, LockedOutputRoot],
) -> list[PerfPublication]:
    publications: list[PerfPublication] = []
    for item in prepared:
        locked_root = locked_roots[item.target.parent]
        prepared_attestation = _attest_regular_file(
            item.temporary,
            expected_identity=(item.temporary_device, item.temporary_inode),
            locked_root=locked_root,
        )
        previous_target, target_claim = _snapshot_perf_target(item.target, locked_root)
        if item.merge_target_was_absent and previous_target is not None:
            raise RuntimeError(f"Collector parquet target changed after merge preparation: {item.target}")
        if item.merge_target is not None and (
            previous_target is None
            or previous_target.identity != item.merge_target.identity
            or previous_target.digest != item.merge_target.digest
            or previous_target.mode != item.merge_target.mode
        ):
            raise RuntimeError(f"Collector parquet target changed after merge preparation: {item.target}")
        source = _attest_regular_file(
            item.source,
            expected_identity=(item.info.source_device, item.info.source_inode),
            expected_digest=item.info.source_digest,
            expected_mode=item.source_mode,
            locked_root=locked_root,
        )
        publications.append(
            PerfPublication(
                source=source,
                target=item.target,
                prepared=prepared_attestation,
                previous_target=previous_target,
                target_claim=target_claim,
                info=item.info,
            )
        )
    return publications


def _same_perf_file(current: PerfFileAttestation, expected: PerfFileAttestation) -> bool:
    return current.identity == expected.identity and current.digest == expected.digest and current.mode == expected.mode


def _attest_expected_perf_file(
    path: Path,
    expected: PerfFileAttestation,
    locked_root: LockedOutputRoot | None = None,
) -> PerfFileAttestation:
    return _attest_regular_file(
        path,
        expected_identity=expected.identity,
        expected_digest=expected.digest,
        expected_mode=expected.mode,
        locked_root=locked_root,
    )


def _path_absent(path: Path, locked_root: LockedOutputRoot | None = None) -> bool:
    try:
        _lstat_path(path, locked_root)
    except FileNotFoundError:
        return True
    return False


def _restore_unknown_private_path(
    private_path: Path,
    target: Path,
    *,
    context: str,
    locked_root: LockedOutputRoot | None = None,
) -> None:
    if _path_absent(target, locked_root):
        try:
            _rename_path_noreplace(private_path, target, locked_root)
        except Exception as error:
            raise RuntimeError(
                f"Collector parquet {context} changed; preserved unknown object at {private_path}"
            ) from error
    raise RuntimeError(f"Collector parquet {context} changed: {target}")


def _validate_unpublished_perf_publications(
    publications: Iterable[PerfPublication],
    locked_roots: dict[Path, LockedOutputRoot],
) -> None:
    for publication in publications:
        locked_root = locked_roots[publication.target.parent]
        _attest_expected_perf_file(publication.source.path, publication.source, locked_root)
        _attest_expected_perf_file(publication.prepared.path, publication.prepared, locked_root)
        if publication.target_claim is not None and not _path_absent(publication.target_claim, locked_root):
            raise RuntimeError(f"Collector parquet target claim changed: {publication.target_claim}")
        if publication.previous_target is None:
            if not _path_absent(publication.target, locked_root):
                raise RuntimeError(f"Collector parquet target changed before publication: {publication.target}")
        else:
            _attest_expected_perf_file(publication.target, publication.previous_target, locked_root)


def _snapshot_legacy_perf_targets(
    publications: Iterable[PerfPublication],
    *,
    snapshot_stack: ExitStack,
    locked_roots: dict[Path, LockedOutputRoot],
) -> dict[Path, BinaryIO | None]:
    """Keep anonymous old-target bytes for ordinary exception rollback."""
    snapshots: dict[Path, BinaryIO | None] = {}
    for publication in publications:
        previous = publication.previous_target
        if previous is None:
            snapshots[publication.target] = None
            continue
        locked_root = locked_roots[publication.target.parent]
        snapshot = snapshot_stack.enter_context(tempfile.TemporaryFile())  # noqa: SIM115 - owned by ExitStack
        with os.fdopen(
            locked_root.open(
                publication.target,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW,
            ),
            "rb",
        ) as target_file:
            opened = os.fstat(target_file.fileno())
            digest = _stream_digest(target_file, copy_to=snapshot)
        current = locked_root.stat(publication.target)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or (opened.st_dev, opened.st_ino) != previous.identity
            or (current.st_dev, current.st_ino) != previous.identity
            or digest != previous.digest
            or stat.S_IMODE(opened.st_mode) != previous.mode
            or stat.S_IMODE(current.st_mode) != previous.mode
        ):
            raise RuntimeError(f"Collector parquet target changed before publication: {publication.target}")
        snapshot.flush()
        os.fsync(snapshot.fileno())
        snapshot.seek(0)
        snapshots[publication.target] = snapshot
    return snapshots


def _validate_legacy_perf_rollback(
    publications: Iterable[PerfPublication],
    locked_roots: dict[Path, LockedOutputRoot],
) -> None:
    for publication in publications:
        locked_root = locked_roots[publication.target.parent]
        previous = publication.previous_target
        if previous is None:
            if not _path_absent(publication.target, locked_root):
                raise RuntimeError(f"Collector parquet target is not restored: {publication.target}")
            continue
        _attest_regular_file(
            publication.target,
            expected_digest=previous.digest,
            expected_mode=previous.mode,
            locked_root=locked_root,
        )


def _restore_legacy_perf_publications(
    publications: Iterable[PerfPublication],
    snapshots: dict[Path, BinaryIO | None],
    locked_roots: dict[Path, LockedOutputRoot],
) -> None:
    for publication in reversed(list(publications)):
        locked_root = locked_roots[publication.target.parent]
        previous = publication.previous_target
        if _path_absent(publication.target, locked_root):
            if previous is None:
                continue
            raise RuntimeError(f"Collector parquet target disappeared during rollback: {publication.target}")
        current = _attest_regular_file(publication.target, locked_root=locked_root)
        if previous is not None and _same_perf_file(current, previous):
            continue
        if not _same_perf_file(current, publication.prepared):
            raise RuntimeError(f"Collector parquet target changed during rollback: {publication.target}")
        prepared_path = publication.prepared.path
        if not _path_absent(prepared_path, locked_root):
            raise RuntimeError(f"Conflicting collector parquet rollback artifact: {prepared_path}")
        if previous is None:
            _rename_path_noreplace(publication.target, prepared_path, locked_root)
            _attest_expected_perf_file(prepared_path, publication.prepared, locked_root)
            continue

        snapshot = snapshots[publication.target]
        if snapshot is None:
            raise RuntimeError(f"Missing collector parquet rollback snapshot: {publication.target}")
        _rename_path_noreplace(publication.target, prepared_path, locked_root)
        _attest_expected_perf_file(prepared_path, publication.prepared, locked_root)
        restore_path = publication.target.with_name(f".{publication.target.name}.rollback.tmp")
        open_flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        restore_created = False
        try:
            restore_fd = locked_root.open(restore_path, open_flags | os.O_CREAT | os.O_EXCL, 0o600)
            restore_created = True
        except FileExistsError:
            restore_fd = locked_root.open(restore_path, open_flags)
        with os.fdopen(restore_fd, "r+b") as restore_file:
            if restore_created:
                locked_root.fsync()
            opened = os.fstat(restore_file.fileno())
            restore_current = locked_root.stat(restore_path)
            if (
                not stat.S_ISREG(opened.st_mode)
                or not stat.S_ISREG(restore_current.st_mode)
                or (opened.st_dev, opened.st_ino) != (restore_current.st_dev, restore_current.st_ino)
                or opened.st_nlink != 1
                or restore_current.st_nlink != 1
            ):
                raise RuntimeError(f"Collector parquet rollback artifact changed: {restore_path}")
            restore_file.seek(0)
            restore_file.truncate()
            snapshot.seek(0)
            shutil.copyfileobj(snapshot, restore_file)
            restore_file.flush()
            os.fchmod(restore_file.fileno(), previous.mode)
            os.fsync(restore_file.fileno())
        _attest_regular_file(
            restore_path,
            expected_digest=previous.digest,
            expected_mode=previous.mode,
            locked_root=locked_root,
        )
        _rename_path_noreplace(restore_path, publication.target, locked_root)
        _attest_regular_file(
            publication.target,
            expected_digest=previous.digest,
            expected_mode=previous.mode,
            locked_root=locked_root,
        )
    _validate_legacy_perf_rollback(publications, locked_roots)


def _publish_perf_publication(publication: PerfPublication, locked_root: LockedOutputRoot) -> None:
    previous = publication.previous_target
    claim = publication.target_claim
    if previous is not None:
        if claim is None:
            raise RuntimeError(f"Missing collector parquet target claim: {publication.target}")
        _rename_path_noreplace(publication.target, claim, locked_root)
        try:
            _attest_expected_perf_file(claim, previous, locked_root)
        except Exception as error:
            try:
                _restore_unknown_private_path(
                    claim,
                    publication.target,
                    context="target claim",
                    locked_root=locked_root,
                )
            except RuntimeError as restore_error:
                raise restore_error from error
            raise
    elif claim is not None:
        raise RuntimeError(f"Unexpected collector parquet target claim: {claim}")

    _rename_path_noreplace(publication.prepared.path, publication.target, locked_root)
    _attest_expected_perf_file(publication.target, publication.prepared, locked_root)


def _private_perf_state(
    path: Path,
    expected: PerfFileAttestation,
    *,
    target: Path,
    context: str,
    locked_root: LockedOutputRoot | None = None,
) -> PerfFileAttestation | None:
    if _path_absent(path, locked_root):
        return None
    try:
        return _attest_expected_perf_file(path, expected, locked_root)
    except Exception as error:
        try:
            _restore_unknown_private_path(path, target, context=context, locked_root=locked_root)
        except RuntimeError as restore_error:
            raise restore_error from error
        raise


def _restore_one_perf_publication(
    publication: PerfPublication,
    locked_root: LockedOutputRoot | None = None,
) -> None:
    target = publication.target
    previous = publication.previous_target
    claim = publication.target_claim
    claim_state = None
    if claim is not None:
        if previous is None:
            raise RuntimeError(f"Unexpected collector parquet target claim: {claim}")
        claim_state = _private_perf_state(
            claim,
            previous,
            target=target,
            context="target claim",
            locked_root=locked_root,
        )
    prepared_state = _private_perf_state(
        publication.prepared.path,
        publication.prepared,
        target=target,
        context="prepared claim",
        locked_root=locked_root,
    )

    if _path_absent(target, locked_root):
        if previous is None:
            return
        if claim_state is None or claim is None:
            raise RuntimeError(f"Collector parquet target disappeared during rollback: {target}")
        _rename_path_noreplace(claim, target, locked_root)
        _attest_expected_perf_file(target, previous, locked_root)
        return

    current = _attest_regular_file(target, locked_root=locked_root)
    if previous is not None and _same_perf_file(current, previous):
        return
    if not _same_perf_file(current, publication.prepared):
        raise RuntimeError(f"Collector parquet target changed during rollback: {target}")
    if prepared_state is not None:
        raise RuntimeError(f"Collector prepared claim unexpectedly exists during rollback: {publication.prepared.path}")

    _rename_path_noreplace(target, publication.prepared.path, locked_root)
    try:
        _attest_expected_perf_file(publication.prepared.path, publication.prepared, locked_root)
    except Exception as error:
        try:
            _restore_unknown_private_path(
                publication.prepared.path,
                target,
                context="prepared rollback claim",
                locked_root=locked_root,
            )
        except RuntimeError as restore_error:
            raise restore_error from error
        raise
    if previous is not None:
        if claim_state is None or claim is None:
            raise RuntimeError(f"Missing collector parquet target claim during rollback: {target}")
        _rename_path_noreplace(claim, target, locked_root)
        _attest_expected_perf_file(target, previous, locked_root)


def _restore_perf_publications(
    publications: Iterable[PerfPublication],
    locked_roots: dict[Path, LockedOutputRoot] | None = None,
) -> None:
    errors: list[Exception] = []
    for publication in reversed(list(publications)):
        try:
            _restore_one_perf_publication(
                publication,
                locked_roots[publication.target.parent] if locked_roots is not None else None,
            )
        except Exception as error:
            errors.append(error)
    if errors:
        raise RuntimeError(f"Collector parquet rollback retained strict recovery state: {errors[0]}") from errors[0]


def validate_restored_perf_publications(
    publications: Iterable[PerfPublication],
    locked_roots: dict[Path, LockedOutputRoot] | None = None,
) -> None:
    """Verify every target is exactly in one journaled pre-publish state."""
    for publication in publications:
        locked_root = locked_roots[publication.target.parent] if locked_roots is not None else None
        target = publication.target
        previous = publication.previous_target
        claim = publication.target_claim
        if _path_absent(target, locked_root):
            if previous is None:
                continue
            raise RuntimeError(f"Collector parquet target disappeared after rollback: {target}")
        if previous is None:
            raise RuntimeError(f"Collector parquet target is not restored: {target}")
        _attest_expected_perf_file(target, previous, locked_root)
        if claim is not None and not _path_absent(claim, locked_root):
            raise RuntimeError(f"Collector parquet target claim was not cleared after rollback: {claim}")


def restore_perf_publications(
    publications: Iterable[PerfPublication],
    locked_roots: dict[Path, LockedOutputRoot] | None = None,
) -> None:
    """Restore an attested publication batch to its exact pre-publish bytes."""
    publication_list = list(publications)
    _restore_perf_publications(publication_list, locked_roots)
    validate_restored_perf_publications(publication_list, locked_roots)


def cleanup_perf_publication_artifacts(
    publications: Iterable[PerfPublication],
    locked_roots: dict[Path, LockedOutputRoot] | None = None,
) -> None:
    """Park transaction-owned artifacts in their bounded preparation slots."""
    publication_list = list(publications)
    for publication in publication_list:
        locked_root = locked_roots[publication.target.parent] if locked_roots is not None else None
        prepared = publication.prepared
        target_is_published = False
        if not _path_absent(publication.target, locked_root):
            target_is_published = _same_perf_file(
                _attest_regular_file(publication.target, locked_root=locked_root),
                prepared,
            )
        if target_is_published:
            if not _path_absent(prepared.path, locked_root):
                raise RuntimeError(f"Collector prepared claim unexpectedly exists during cleanup: {prepared.path}")
        else:
            _attest_expected_perf_file(prepared.path, prepared, locked_root)
        claim = publication.target_claim
        previous = publication.previous_target
        if claim is not None and not _path_absent(claim, locked_root):
            if previous is None:
                raise RuntimeError(f"Unexpected collector parquet target claim: {claim}")
            if not _path_absent(prepared.path, locked_root):
                raise RuntimeError(f"Conflicting collector parquet cleanup artifacts: {prepared.path}")
            claimed_previous = PerfFileAttestation(
                path=claim,
                digest=previous.digest,
                device=previous.device,
                inode=previous.inode,
                mode=previous.mode,
            )
            _attest_expected_perf_file(claim, claimed_previous, locked_root)
            _rename_path_noreplace(claim, prepared.path, locked_root)
            _attest_expected_perf_file(prepared.path, claimed_previous, locked_root)


def _validate_prepared_source(
    item: _PreparedPerfFile,
    *,
    context: str,
    locked_root: LockedOutputRoot,
) -> None:
    open_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    try:
        source_fd = locked_root.open(item.source, open_flags)
    except OSError as error:
        raise RuntimeError(f"Collector staging file changed {context}: {item.source}") from error
    with os.fdopen(source_fd, "rb") as source_file:
        opened = os.fstat(source_file.fileno())
        source_digest = _stream_digest(source_file)
    current = locked_root.stat(item.source)
    if (
        not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        or (opened.st_dev, opened.st_ino) != (item.info.source_device, item.info.source_inode)
        or source_digest != item.info.source_digest
        or stat.S_IMODE(opened.st_mode) != item.source_mode
        or stat.S_IMODE(current.st_mode) != item.source_mode
    ):
        raise RuntimeError(f"Collector staging file changed {context}: {item.source}")


def _is_owned_temporary(
    path: Path,
    device: int,
    inode: int,
    locked_root: LockedOutputRoot | None = None,
) -> bool:
    try:
        current = _lstat_path(path, locked_root)
    except FileNotFoundError:
        return False
    return (
        stat.S_ISREG(current.st_mode) and (current.st_dev, current.st_ino) == (device, inode) and current.st_nlink == 1
    )


def _require_owned_temporary(item: _PreparedPerfFile, locked_root: LockedOutputRoot) -> None:
    if not _is_owned_temporary(
        item.temporary,
        item.temporary_device,
        item.temporary_inode,
        locked_root,
    ):
        raise RuntimeError(f"Collector temporary file changed during finalization: {item.temporary}")


def _validate_owned_temporary_cleanup(item: _PreparedPerfFile, locked_root: LockedOutputRoot) -> None:
    try:
        current = locked_root.stat(item.temporary)
    except FileNotFoundError:
        return
    if (
        not stat.S_ISREG(current.st_mode)
        or (current.st_dev, current.st_ino) != (item.temporary_device, item.temporary_inode)
        or current.st_nlink != 1
    ):
        raise RuntimeError(f"Collector temporary file changed before cleanup: {item.temporary}")
    _require_owned_temporary(item, locked_root)


def perf_preparation_path(parquet_path: Path) -> Path:
    """Return the one lock-owned pre-journal render path for a parquet."""
    return parquet_path.with_name(f".{parquet_path.name}.tmp")


def _validate_perf_file_paths(csv_path: Path, locked_root: LockedOutputRoot) -> tuple[Path, Path]:
    try:
        source_state = locked_root.stat(csv_path)
    except FileNotFoundError as error:
        raise RuntimeError(f"Cannot convert non-regular collector staging file: {csv_path}") from error
    if not stat.S_ISREG(source_state.st_mode):
        raise RuntimeError(f"Cannot convert non-regular collector staging file: {csv_path}")
    lock_path = Path(f"{csv_path}.lock")
    if not locked_root.absent(lock_path):
        raise RuntimeError(f"Cannot convert {csv_path} while lock file exists: {lock_path}")

    parquet_path = csv_path.with_suffix(".parquet")
    if not locked_root.absent(parquet_path):
        parquet_state = locked_root.stat(parquet_path)
        if not stat.S_ISREG(parquet_state.st_mode):
            raise RuntimeError(f"Cannot replace non-regular collector parquet file: {parquet_path}")
    merge_lock = parquet_path.with_name(f"{parquet_path.name}.mergelock")
    if not locked_root.absent(merge_lock):
        merge_lock_state = locked_root.stat(merge_lock)
        if not stat.S_ISREG(merge_lock_state.st_mode):
            raise RuntimeError(f"Invalid collector parquet merge lock: {merge_lock}")
    return parquet_path, merge_lock


def _read_attested_parquet(parquet_path: Path, *, pq, locked_root: LockedOutputRoot):
    parquet_fd = locked_root.open(
        parquet_path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW,
    )
    with os.fdopen(parquet_fd, "rb") as parquet_file, tempfile.TemporaryFile() as snapshot_file:
        opened = os.fstat(parquet_file.fileno())
        digest = _stream_digest(parquet_file, copy_to=snapshot_file)
        snapshot_file.flush()
        snapshot_file.seek(0)
        table = pq.read_table(snapshot_file)
    current = locked_root.stat(parquet_path)
    if (
        not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        or stat.S_IMODE(opened.st_mode) != stat.S_IMODE(current.st_mode)
    ):
        raise RuntimeError(f"Collector parquet target changed while preparing merge: {parquet_path}")
    return table, PerfFileAttestation(
        path=parquet_path,
        digest=digest,
        device=opened.st_dev,
        inode=opened.st_ino,
        mode=stat.S_IMODE(opened.st_mode),
    )


def _prepare_perf_file(
    csv_path: Path,
    parquet_path: Path,
    *,
    expected_source_identity: tuple[str, int, int] | None,
    compression: str,
    merge_existing: bool,
    pa,
    pc_compute,
    pc_csv,
    pq,
    locked_root: LockedOutputRoot,
) -> _PreparedPerfFile:
    source_fd = locked_root.open(
        csv_path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW,
    )
    with os.fdopen(source_fd, "rb") as source_file, tempfile.TemporaryFile() as snapshot_file:
        source_stat = os.fstat(source_file.fileno())
        source_mode = stat.S_IMODE(source_stat.st_mode)
        source_digest = _stream_digest(source_file, copy_to=snapshot_file)
        if (
            expected_source_identity is not None
            and (
                source_digest,
                source_stat.st_dev,
                source_stat.st_ino,
            )
            != expected_source_identity
        ):
            raise RuntimeError(f"Collector staging file changed after preflight: {csv_path}")
        snapshot_file.seek(0)
        table = pc_csv.read_csv(snapshot_file)
    new_rows = table.num_rows
    merged_existing = False
    merge_target = None
    merge_target_was_absent = False
    if merge_existing:
        try:
            locked_root.stat(parquet_path)
        except FileNotFoundError:
            merge_target_was_absent = True
        else:
            old_table, merge_target = _read_attested_parquet(parquet_path, pq=pq, locked_root=locked_root)
            table, merged_existing, new_rows = _merge_perf_rows(table, old_table, parquet_path, pa=pa)
    table = _normalize_power_metrics(table, pa=pa, pc=pc_compute)

    temporary = perf_preparation_path(parquet_path)
    temporary_identity: tuple[int, int] | None = None
    temporary_fd: int | None = None
    try:
        open_flags = os.O_RDWR
        open_flags |= getattr(os, "O_CLOEXEC", 0)
        open_flags |= getattr(os, "O_NOFOLLOW", 0)
        temporary_created = False
        try:
            temporary_fd = locked_root.open(temporary, open_flags | os.O_CREAT | os.O_EXCL, 0o600)
            temporary_created = True
        except FileExistsError:
            temporary_fd = locked_root.open(temporary, open_flags)
        temp_stat = os.fstat(temporary_fd)
        temporary_identity = (temp_stat.st_dev, temp_stat.st_ino)
        current = locked_root.stat(temporary)
        if (
            not stat.S_ISREG(temp_stat.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or (current.st_dev, current.st_ino) != temporary_identity
            or temp_stat.st_nlink != 1
            or current.st_nlink != 1
        ):
            raise RuntimeError(f"Collector temporary file changed before reuse: {temporary}")
        temp_file = os.fdopen(temporary_fd, "r+b")
        temporary_fd = None
        with temp_file:
            if temporary_created:
                locked_root.fsync()
            temp_file.seek(0)
            temp_file.truncate()
            pq.write_table(table, temp_file, compression=compression)
            temp_file.flush()
            os.fchmod(temp_file.fileno(), source_mode)
            os.fsync(temp_file.fileno())
        prepared = _PreparedPerfFile(
            source=csv_path,
            target=parquet_path,
            temporary=temporary,
            temporary_device=temporary_identity[0],
            temporary_inode=temporary_identity[1],
            source_mode=source_mode,
            info=PerfFinalizationInfo(
                new_rows=new_rows,
                merged_existing=merged_existing,
                source_digest=source_digest,
                source_device=source_stat.st_dev,
                source_inode=source_stat.st_ino,
            ),
            merge_target=merge_target,
            merge_target_was_absent=merge_target_was_absent,
        )
        _require_owned_temporary(prepared, locked_root)
        return prepared
    except Exception:
        if temporary_fd is not None:
            os.close(temporary_fd)
        if temporary_identity is not None:
            device, inode = temporary_identity
            if _is_owned_temporary(temporary, device, inode, locked_root):
                _attest_regular_file(
                    temporary,
                    expected_identity=(device, inode),
                    locked_root=locked_root,
                )
        raise


def _finalize_perf_file_batch(
    csv_paths: list[Path],
    *,
    locked_roots: dict[Path, LockedOutputRoot],
    delete_source: bool,
    compression: str,
    merge_existing: bool,
    finalization_info: dict[Path, PerfFinalizationInfo] | None,
    prepublish_validate: Callable[[], None] | None,
    expected_source_identities: dict[Path, tuple[str, int, int]] | None,
    publication_transaction: PerfPublicationTransaction | None,
) -> list[Path]:
    """Prepare every conversion before publishing any parquet target."""
    if publication_transaction is not None and delete_source:
        raise ValueError("A durable perf publication transaction must retain its staging files")
    try:
        import pyarrow as pa
        import pyarrow.compute as pc_compute
        import pyarrow.csv as pc_csv
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Finalizing collector perf data as parquet requires pyarrow. "
            "Install the project runtime dependencies before collecting perf data."
        ) from exc

    validated_inputs = [
        (csv_path, *_validate_perf_file_paths(csv_path, locked_roots[csv_path.parent])) for csv_path in csv_paths
    ]
    if expected_source_identities is not None:
        expected_paths = set(expected_source_identities)
        selected_paths = {csv_path for csv_path, _parquet_path, _merge_lock in validated_inputs}
        if expected_paths != selected_paths:
            raise RuntimeError("Collector staging preflight does not match the selected finalization inputs")

    unique_inputs: list[tuple[Path, Path, Path]] = []
    seen_inputs: set[tuple[int, int, tuple[int, int, str]]] = set()
    opened_locks: dict[
        tuple[int, int],
        tuple[int, list[tuple[Path, tuple[int, int, str]]]],
    ] = {}
    prepared: list[_PreparedPerfFile] = []
    publications: list[PerfPublication] = []
    legacy_snapshots: dict[Path, BinaryIO | None] = {}
    legacy_snapshot_stack = ExitStack()
    try:
        for item in validated_inputs:
            locked_root = locked_roots[item[0].parent]
            source_stat = locked_root.stat(item[0])
            if not stat.S_ISREG(source_stat.st_mode):
                raise RuntimeError(f"Cannot convert non-regular collector staging file: {item[0]}")
            lock_fd, lock_identity, lock_entry_key = _open_merge_lock(item[2], locked_root)
            if lock_identity in opened_locks:
                opened_locks[lock_identity][1].append((item[2], lock_entry_key))
                _release_merge_lock(lock_fd)
            else:
                opened_locks[lock_identity] = (lock_fd, [(item[2], lock_entry_key)])
            input_identity = (source_stat.st_dev, source_stat.st_ino, lock_entry_key)
            if input_identity not in seen_inputs:
                seen_inputs.add(input_identity)
                unique_inputs.append(item)
        lock_order = sorted(
            opened_locks,
            key=lambda identity: min(entry_key for _path, entry_key in opened_locks[identity][1]),
        )
        for lock_identity in lock_order:
            lock_fd, lock_entries = opened_locks[lock_identity]
            _acquire_merge_lock(lock_fd)
            for lock_path, lock_entry_key in lock_entries:
                _revalidate_merge_lock_path(
                    lock_fd,
                    lock_path,
                    expected_identity=lock_identity,
                    expected_entry_key=lock_entry_key,
                    locked_root=locked_roots[lock_path.parent],
                )
        preparation_targets = tuple(parquet_path for _csv_path, parquet_path, _lock in unique_inputs)
        transaction_paths = tuple(
            transaction_path
            for output_root in sorted({path.parent for path in preparation_targets})
            for transaction_path in perf_transaction_artifact_paths(output_root)
        )
        if not cleanup_unjournaled_perf_preparations(
            preparation_targets,
            transaction_paths=transaction_paths,
            locked_roots=locked_roots,
        ):
            raise RuntimeError("Cannot finalize collector perf files while a transaction artifact exists")
        for csv_path, parquet_path, _merge_lock in unique_inputs:
            locked_root = locked_roots[csv_path.parent]
            _validate_perf_file_paths(csv_path, locked_root)
            prepared.append(
                _prepare_perf_file(
                    csv_path,
                    parquet_path,
                    expected_source_identity=(
                        expected_source_identities[csv_path] if expected_source_identities is not None else None
                    ),
                    compression=compression,
                    merge_existing=merge_existing,
                    pa=pa,
                    pc_compute=pc_compute,
                    pc_csv=pc_csv,
                    pq=pq,
                    locked_root=locked_root,
                )
            )

        # The bytes parsed above are the bytes this batch owns. Validate the
        # whole input set again before any parquet is replaced.
        for item in prepared:
            locked_root = locked_roots[item.source.parent]
            _validate_perf_file_paths(item.source, locked_root)
            _validate_prepared_source(item, context="during finalization", locked_root=locked_root)
            _require_owned_temporary(item, locked_root)
        if prepublish_validate is not None:
            prepublish_validate()
            for item in prepared:
                locked_root = locked_roots[item.source.parent]
                _validate_perf_file_paths(item.source, locked_root)
                _validate_prepared_source(item, context="after prepublish validation", locked_root=locked_root)
                _require_owned_temporary(item, locked_root)

        publications = _prepare_perf_publications(prepared, locked_roots)
        if publication_transaction is not None:
            publication_transaction.prepare(tuple(publications))
        else:
            legacy_snapshots = _snapshot_legacy_perf_targets(
                publications,
                snapshot_stack=legacy_snapshot_stack,
                locked_roots=locked_roots,
            )
        _validate_unpublished_perf_publications(publications, locked_roots)
        try:
            for publication in publications:
                locked_root = locked_roots[publication.target.parent]
                if publication_transaction is None:
                    _replace_path(publication.prepared.path, publication.target, locked_root)
                    _attest_expected_perf_file(publication.target, publication.prepared, locked_root)
                else:
                    _publish_perf_publication(publication, locked_root)
        except Exception:
            if publication_transaction is None:
                _restore_legacy_perf_publications(publications, legacy_snapshots, locked_roots)
                cleanup_perf_publication_artifacts(publications, locked_roots)
            else:
                _restore_perf_publications(publications, locked_roots)
                publication_transaction.rollback_complete(tuple(publications))
            raise
        if publication_transaction is None:
            cleanup_perf_publication_artifacts(publications, locked_roots)

        if delete_source:
            for item in prepared:
                locked_root = locked_roots[item.source.parent]
                _validate_perf_file_paths(item.source, locked_root)
                _validate_prepared_source(item, context="before cleanup", locked_root=locked_root)
            for item in prepared:
                _unlink_path(item.source, locked_roots[item.source.parent])
        if finalization_info is not None:
            finalization_info.update({item.target: item.info for item in prepared})
        return [item.target for item in prepared]
    finally:
        legacy_snapshot_stack.close()
        for item in prepared:
            if publication_transaction is None or not publication_transaction.has_durable_journal():
                _validate_owned_temporary_cleanup(item, locked_roots[item.temporary.parent])
        for lock_fd, _lock_entries in opened_locks.values():
            _release_merge_lock(lock_fd)


def _normalize_power_metrics(table, *, pa, pc):
    """Store unavailable power metrics as typed zero sentinels.

    A running GPU workload cannot have a valid zero-watt measurement, and the
    SDK already interprets zero energy as uncovered power data.  Committed
    parquet files therefore use ``0.0`` rather than null for unavailable
    ``power``/``power_limit`` cells.  Tables that omit these optional columns
    remain unchanged.
    """
    for name in ("power", "power_limit"):
        if name not in table.column_names:
            continue

        index = table.schema.get_field_index(name)
        column = table.column(index)
        try:
            column = column.cast(pa.float64())
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError) as exc:
            raise ValueError(f"{name} must be convertible to float64") from exc

        column = pc.fill_null(column, 0.0)
        for row_index, value in enumerate(column.to_pylist()):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must contain finite non-negative values; row {row_index} has {value!r}")

        table = table.set_column(index, name, column)
    return table


def _open_merge_lock(
    lock_path: Path,
    locked_root: LockedOutputRoot | None = None,
) -> tuple[int, tuple[int, int], tuple[int, int, str]]:
    """Open and attest a merge-lock object without acquiring its flock."""
    open_flags = os.O_CREAT | os.O_WRONLY
    open_flags |= getattr(os, "O_CLOEXEC", 0)
    open_flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = _open_path(lock_path, open_flags, locked_root=locked_root)
    try:
        _fsync_path_parent(lock_path, locked_root)
        opened = os.fstat(fd)
        current = _lstat_path(lock_path, locked_root)
        identity = (opened.st_dev, opened.st_ino)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or identity != (current.st_dev, current.st_ino)
        ):
            raise RuntimeError(f"Invalid collector parquet merge lock: {lock_path}")
        return fd, identity, _merge_lock_entry_key(lock_path, identity, locked_root)
    except Exception:
        os.close(fd)
        raise


def _merge_lock_entry_key(
    lock_path: Path,
    lock_identity: tuple[int, int],
    locked_root: LockedOutputRoot | None = None,
) -> tuple[int, int, str]:
    """Identify the actual parent-directory entry naming an opened lock."""
    parent_fd = (
        os.dup(locked_root.file_descriptor) if locked_root is not None else os.open(lock_path.parent, os.O_RDONLY)
    )
    try:
        parent = os.fstat(parent_fd)
        if not stat.S_ISDIR(parent.st_mode):
            raise RuntimeError(f"Invalid collector parquet merge lock directory: {lock_path.parent}")
        names = os.listdir(parent_fd)
        same_object: list[str] = []
        for name in names:
            try:
                entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError as error:
                raise RuntimeError(f"Collector parquet merge lock directory changed: {lock_path.parent}") from error
            if stat.S_ISREG(entry.st_mode) and (entry.st_dev, entry.st_ino) == lock_identity:
                same_object.append(name)

        requested_name = lock_path.name
        if requested_name in names:
            if requested_name not in same_object:
                raise RuntimeError(f"Collector parquet merge lock changed: {lock_path}")
            actual_name = requested_name
        else:
            case_equivalent = [name for name in same_object if name.casefold() == requested_name.casefold()]
            if len(case_equivalent) == 1:
                actual_name = case_equivalent[0]
            elif len(same_object) == 1:
                actual_name = same_object[0]
            else:
                raise RuntimeError(f"Ambiguous collector parquet merge lock entry: {lock_path}")
        return parent.st_dev, parent.st_ino, actual_name
    finally:
        os.close(parent_fd)


def _acquire_merge_lock(lock_fd: int) -> None:
    """Advisory flock on a per-target lock file (blocking).

    flock is atomic, releases automatically when the holding process exits
    (no stale-lock handling needed), and unlike an O_EXCL create-and-steal
    scheme cannot let two finalizers past the gate. The lock file itself is
    left in place — unlinking it would reopen the create/steal race.
    """
    import fcntl

    fcntl.flock(lock_fd, fcntl.LOCK_EX)


def _revalidate_merge_lock_path(
    lock_fd: int,
    lock_path: Path,
    *,
    expected_identity: tuple[int, int],
    expected_entry_key: tuple[int, int, str],
    locked_root: LockedOutputRoot | None = None,
) -> None:
    """Re-attest the pathname after flock; a waiter may have seen it replaced."""
    opened = os.fstat(lock_fd)
    try:
        current = _lstat_path(lock_path, locked_root)
    except FileNotFoundError as error:
        raise RuntimeError(f"Collector parquet merge lock changed after flock: {lock_path}") from error
    if (
        not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or (opened.st_dev, opened.st_ino) != expected_identity
        or (current.st_dev, current.st_ino) != expected_identity
        or _merge_lock_entry_key(lock_path, expected_identity, locked_root) != expected_entry_key
    ):
        raise RuntimeError(f"Collector parquet merge lock changed after flock: {lock_path}")


def _release_merge_lock(lock_fd: int) -> None:
    os.close(lock_fd)  # closing releases the flock


@contextmanager
def perf_merge_locks(
    parquet_paths: Iterable[Path],
    locked_roots: dict[Path, LockedOutputRoot] | None = None,
):
    """Hold and post-attest the canonical merge locks for exact targets."""
    opened_locks: dict[
        tuple[int, int],
        tuple[int, list[tuple[Path, tuple[int, int, str]]]],
    ] = {}
    try:
        for parquet_path in sorted(set(parquet_paths)):
            lock_path = parquet_path.with_name(f"{parquet_path.name}.mergelock")
            locked_root = locked_roots[parquet_path.parent] if locked_roots is not None else None
            lock_fd, lock_identity, lock_entry_key = _open_merge_lock(lock_path, locked_root)
            if lock_identity in opened_locks:
                opened_locks[lock_identity][1].append((lock_path, lock_entry_key))
                _release_merge_lock(lock_fd)
            else:
                opened_locks[lock_identity] = (lock_fd, [(lock_path, lock_entry_key)])
        lock_order = sorted(
            opened_locks,
            key=lambda identity: min(entry_key for _path, entry_key in opened_locks[identity][1]),
        )
        for lock_identity in lock_order:
            lock_fd, lock_entries = opened_locks[lock_identity]
            _acquire_merge_lock(lock_fd)
            for lock_path, lock_entry_key in lock_entries:
                _revalidate_merge_lock_path(
                    lock_fd,
                    lock_path,
                    expected_identity=lock_identity,
                    expected_entry_key=lock_entry_key,
                    locked_root=(locked_roots[lock_path.parent] if locked_roots is not None else None),
                )
        yield
    finally:
        for lock_fd, _lock_entries in opened_locks.values():
            _release_merge_lock(lock_fd)


def perf_preparation_cleanup_path(parquet_path: Path) -> Path:
    return parquet_path.with_name(f".{parquet_path.name}.tmp.cleanup")


def atomic_write_reservation_path(destination: Path) -> Path:
    """Return the one reserved no-replace publication path for a target."""
    return destination.with_name(f".{destination.name}.tmp")


def atomic_write_cleanup_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.tmp.cleanup")


def collector_retained_path(path: Path) -> Path:
    """Return the bounded parked-inode slot for one consumed collector file."""
    return path.with_name(f".{path.name}.retained")


def _atomic_write_path_state(
    path: Path,
    locked_root: LockedOutputRoot | None = None,
) -> tuple[PerfFileAttestation, int, int]:
    try:
        before = _lstat_path(path, locked_root)
    except FileNotFoundError as error:
        raise RuntimeError(f"Collector atomic write artifact disappeared: {path}") from error
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"Unowned collector atomic write artifact: {path}")
    attestation = _attest_regular_file(
        path,
        expected_identity=(before.st_dev, before.st_ino),
        expected_mode=stat.S_IMODE(before.st_mode),
        locked_root=locked_root,
    )
    current = _lstat_path(path, locked_root)
    if (
        not stat.S_ISREG(current.st_mode)
        or (current.st_dev, current.st_ino) != attestation.identity
        or current.st_size != before.st_size
        or stat.S_IMODE(current.st_mode) != attestation.mode
        or current.st_nlink != before.st_nlink
    ):
        raise RuntimeError(f"Collector atomic write artifact changed: {path}")
    return attestation, current.st_size, current.st_nlink


def _require_atomic_write_path_state(
    path: Path,
    expected: PerfFileAttestation,
    *,
    expected_size: int,
    expected_nlink: int,
    locked_root: LockedOutputRoot | None = None,
) -> None:
    observed, observed_size, observed_nlink = _atomic_write_path_state(path, locked_root)
    if (
        observed.identity != expected.identity
        or observed.digest != expected.digest
        or observed.mode != expected.mode
        or observed_size != expected_size
        or observed_nlink != expected_nlink
    ):
        raise RuntimeError(f"Collector atomic write artifact changed: {path}")


def cleanup_atomic_write_reservations(
    destinations: Iterable[Path],
    locked_roots: dict[Path, LockedOutputRoot] | None = None,
) -> None:
    """Normalize exact interrupted reservations without deleting their inodes."""
    for destination in destinations:
        locked_root = locked_roots[destination.parent] if locked_roots is not None else None
        reserved = atomic_write_reservation_path(destination)
        cleanup_claim = atomic_write_cleanup_path(destination)
        if not _path_absent(reserved, locked_root) and not _path_absent(cleanup_claim, locked_root):
            raise RuntimeError(f"Conflicting collector atomic write artifacts: {reserved}")
        if _path_absent(reserved, locked_root) and _path_absent(cleanup_claim, locked_root):
            continue

        try:
            if not _path_absent(reserved, locked_root):
                claimed_attestation, claimed_size, claimed_links = _atomic_write_path_state(reserved, locked_root)
                _rename_path_noreplace(reserved, cleanup_claim, locked_root)
            else:
                claimed_attestation, claimed_size, claimed_links = _atomic_write_path_state(cleanup_claim, locked_root)

            _require_atomic_write_path_state(
                cleanup_claim,
                claimed_attestation,
                expected_size=claimed_size,
                expected_nlink=claimed_links,
                locked_root=locked_root,
            )
            if _path_absent(destination, locked_root):
                if claimed_links != 1:
                    raise RuntimeError(f"Unowned collector atomic write artifact: {cleanup_claim}")
            else:
                target_attestation, target_size, target_links = _atomic_write_path_state(destination, locked_root)
                if (
                    claimed_links != 2
                    or target_links != 2
                    or claimed_attestation.identity != target_attestation.identity
                    or claimed_attestation.digest != target_attestation.digest
                    or claimed_size != target_size
                    or claimed_attestation.mode != target_attestation.mode
                ):
                    raise RuntimeError(f"Collector atomic write publication changed: {destination}")
                raise RuntimeError(f"Legacy collector atomic write hardlink requires manual recovery: {destination}")

            _require_atomic_write_path_state(
                cleanup_claim,
                claimed_attestation,
                expected_size=claimed_size,
                expected_nlink=claimed_links,
                locked_root=locked_root,
            )
            _rename_path_noreplace(cleanup_claim, reserved, locked_root)
            _require_atomic_write_path_state(
                reserved,
                claimed_attestation,
                expected_size=claimed_size,
                expected_nlink=claimed_links,
                locked_root=locked_root,
            )
        except Exception as error:
            if _path_absent(cleanup_claim, locked_root):
                raise
            try:
                _restore_unknown_private_path(
                    cleanup_claim,
                    reserved,
                    context="atomic write reservation",
                    locked_root=locked_root,
                )
            except RuntimeError as restore_error:
                raise restore_error from error
            raise


def cleanup_unjournaled_perf_preparations(
    parquet_paths: Iterable[Path],
    *,
    transaction_paths: Iterable[Path],
    locked_roots: dict[Path, LockedOutputRoot] | None = None,
) -> bool:
    """Normalize exact reservations only while every ownership marker is absent."""
    parquet_paths = tuple(parquet_paths)
    if any(
        not _path_absent(path, locked_roots[path.parent] if locked_roots is not None else None)
        for path in transaction_paths
    ):
        return False
    for parquet_path in parquet_paths:
        locked_root = locked_roots[parquet_path.parent] if locked_roots is not None else None
        reserved = perf_preparation_path(parquet_path)
        cleanup_claim = perf_preparation_cleanup_path(parquet_path)
        if not _path_absent(reserved, locked_root) and not _path_absent(cleanup_claim, locked_root):
            raise RuntimeError(f"Conflicting collector reserved parquet artifacts: {reserved}")

        claimed_attestation: PerfFileAttestation | None = None
        if not _path_absent(reserved, locked_root):
            reserved_state = _lstat_path(reserved, locked_root)
            if not stat.S_ISREG(reserved_state.st_mode) or reserved_state.st_nlink != 1:
                raise RuntimeError(f"Unowned collector reserved parquet artifact: {reserved}")
            claimed_attestation = _attest_regular_file(
                reserved,
                expected_identity=(reserved_state.st_dev, reserved_state.st_ino),
                expected_mode=stat.S_IMODE(reserved_state.st_mode),
                locked_root=locked_root,
            )
            _rename_path_noreplace(reserved, cleanup_claim, locked_root)
        elif not _path_absent(cleanup_claim, locked_root):
            claim_state = _lstat_path(cleanup_claim, locked_root)
            if not stat.S_ISREG(claim_state.st_mode) or claim_state.st_nlink != 1:
                raise RuntimeError(f"Unowned collector reserved parquet artifact: {cleanup_claim}")
            claimed_attestation = _attest_regular_file(
                cleanup_claim,
                expected_identity=(claim_state.st_dev, claim_state.st_ino),
                expected_mode=stat.S_IMODE(claim_state.st_mode),
                locked_root=locked_root,
            )

        if claimed_attestation is not None:
            try:
                _attest_expected_perf_file(cleanup_claim, claimed_attestation, locked_root)
                current = _lstat_path(cleanup_claim, locked_root)
                if current.st_nlink != 1:
                    raise RuntimeError(f"Collector reserved parquet artifact changed: {cleanup_claim}")
                _rename_path_noreplace(cleanup_claim, reserved, locked_root)
                _attest_expected_perf_file(reserved, claimed_attestation, locked_root)
                if _lstat_path(reserved, locked_root).st_nlink != 1:
                    raise RuntimeError(f"Collector reserved parquet artifact changed: {reserved}")
            except Exception as error:
                if _path_absent(cleanup_claim, locked_root):
                    raise
                try:
                    _restore_unknown_private_path(
                        cleanup_claim,
                        reserved,
                        context="reserved preparation",
                        locked_root=locked_root,
                    )
                except RuntimeError as restore_error:
                    raise restore_error from error
    return True


def _merge_perf_rows(new_table, old_table, parquet_path: Path, *, pa):
    """Merge freshly-collected rows into an existing perf parquet, keeping the
    newest row per identity key. Returns ``(table, merged_existing,
    current_event_rows)``; schema incompatibility returns the new table with
    ``merged_existing=False`` and its unchanged row count."""
    import pandas as pd

    log = logging.getLogger(__name__)

    # Compare (name, type) pairs for IDENTITY columns only: matching names
    # with drifted types would otherwise round-trip through pandas and
    # silently rewrite the parquet under a different schema. Metric columns
    # are exempt from the type check — pyarrow.csv infers an all-empty
    # optional metric column as `null` while populated runs infer `double`,
    # and treating that drift as a mismatch made the merge silently OVERWRITE
    # the accumulated dataset. Metric types are reconciled for concatenation;
    # the finalizer then casts power metrics to double and replaces nulls with
    # the repository's 0.0 unavailable-measurement sentinel. Order-insensitive
    # (the merge realigns column order below); Arrow metadata is ignored
    # (pandas round-trips change it).
    def fields(schema):
        return sorted(
            (f.name, str(f.type)) if f.name not in PERF_METRIC_COLUMNS else (f.name, "<metric>") for f in schema
        )

    old_fields = fields(old_table.schema)
    new_fields = fields(new_table.schema)
    if old_fields != new_fields:
        log.warning(
            "convert_perf_csv_to_parquet: schema mismatch merging %s "
            "(existing=%s, new=%s); overwriting instead of merging.",
            parquet_path.name,
            old_fields,
            new_fields,
        )
        return new_table, False, new_table.num_rows
    # Reconcile metric-column types on BOTH sides: an all-empty metric column
    # is inferred as `null`, and Arrow can cast null -> anything (all nulls)
    # but not double -> null. Whichever side is null-typed is cast toward the
    # other; a genuine numeric-vs-numeric drift casts new toward old.
    for f in old_table.schema:
        if f.name not in PERF_METRIC_COLUMNS:
            continue
        new_field = new_table.schema.field(f.name)
        if new_field.type == f.type:
            continue
        if pa.types.is_null(f.type):
            old_table = old_table.set_column(
                old_table.schema.get_field_index(f.name),
                f.name,
                old_table.column(f.name).cast(new_field.type),
            )
        else:
            new_table = new_table.set_column(
                new_table.schema.get_field_index(f.name),
                f.name,
                new_table.column(f.name).cast(f.type),
            )

    new_df = new_table.to_pandas()
    old_df = old_table.to_pandas()

    new_df = new_df[old_df.columns.tolist()]  # align column order
    identity = [c for c in old_df.columns if c not in PERF_METRIC_COLUMNS]
    current_event_rows = len(new_df.drop_duplicates(subset=identity, keep="last"))
    combined = pd.concat([old_df, new_df], ignore_index=True)
    deduped = combined.drop_duplicates(subset=identity, keep="last").reset_index(drop=True)
    log.info(
        "convert_perf_csv_to_parquet: merged %s: %d existing + %d new -> %d rows "
        "(%d identity keys replaced by newer measurements)",
        parquet_path.name,
        len(old_df),
        len(new_df),
        len(deduped),
        len(combined) - len(deduped),
    )
    return pa.Table.from_pandas(deduped, preserve_index=False), True, current_event_rows


def find_perf_csv_outputs(output_root: str | os.PathLike = ".", *, recursive: bool = False) -> list[Path]:
    """Find collector CSV staging files directly under `output_root` by default."""
    root = Path(output_root)
    paths = root.rglob("*_perf.txt") if recursive else root.glob("*_perf.txt")
    return sorted(path for path in paths if path.name != "INCOMPLETE.txt")


def stale_output_artifacts(output_dir: str | os.PathLike, perf_filename: str) -> list[str]:
    """Artifacts a previous standalone-collector attempt left in ``output_dir``.

    ``log_perf`` opens its staging CSV in append mode, so a rerun into a
    directory holding a prior attempt's rows would append a second run after
    the stale ones and finalize both under a sidecar that attests only the
    current plan. Standalone multi-node collectors have no attempt/resume
    validation, so their only safe behavior is to refuse such a directory.
    Returns the offending names (relative to ``output_dir``), empty when clean.
    """
    directory = Path(output_dir)
    stem = Path(perf_filename).stem
    owned: list[str] = []
    for pattern in (f"{stem}.*", "collection_meta.yaml", "errors_*.json"):
        owned.extend(sorted(path.name for path in directory.glob(pattern)))
    return owned


def finalize_perf_files(
    csv_files: Iterable[str | os.PathLike],
    *,
    delete_source: bool = True,
    compression: str = "zstd",
    merge_existing: bool = True,
    finalization_info: dict[Path, PerfFinalizationInfo] | None = None,
    prepublish_validate: Callable[[], None] | None = None,
    expected_source_identities: dict[Path, tuple[str, int, int]] | None = None,
    publication_transaction: PerfPublicationTransaction | None = None,
    _locked_output_roots: dict[Path, LockedOutputRoot] | None = None,
) -> list[Path]:
    """Finalize explicit collector CSV staging files as parquet.

    ``merge_existing`` defaults to True so that finalizing accumulates into any
    pre-existing parquet (resume / retry-failed / batched collection) instead of
    overwriting it with only this run's rows — see convert_perf_csv_to_parquet.
    When provided, ``finalization_info`` receives each staging file's finalized
    current-event row contribution and whether an existing parquet took the
    compatible merge path, plus the digest of the exact staging bytes parsed.
    ``publication_transaction`` may durably bind the prepared batch before any
    target is replaced; transactional callers must retain their staging files.
    """
    selected = []
    for csv_file in sorted({Path(path) for path in csv_files}):
        if csv_file.name == "INCOMPLETE.txt" or not csv_file.name.endswith("_perf.txt"):
            continue
        if _locked_output_roots is None:
            if csv_file.exists() or csv_file.is_symlink():
                selected.append(csv_file.parent.resolve() / csv_file.name)
            continue
        csv_file = csv_file.absolute()
        locked_root = _locked_output_roots.get(csv_file.parent)
        if locked_root is None:
            raise RuntimeError(f"Collector CSV is outside its locked output root: {csv_file}")
        if not locked_root.absent(csv_file):
            selected.append(csv_file)
    if not selected:
        return []

    def finalize_selected(locked_roots: dict[Path, LockedOutputRoot]) -> list[Path]:
        return _finalize_perf_file_batch(
            selected,
            locked_roots=locked_roots,
            delete_source=delete_source,
            compression=compression,
            merge_existing=merge_existing,
            finalization_info=finalization_info,
            prepublish_validate=prepublish_validate,
            expected_source_identities=expected_source_identities,
            publication_transaction=publication_transaction,
        )

    if _locked_output_roots is not None:
        return finalize_selected(_locked_output_roots)
    with ExitStack() as lifecycle_stack:
        locked_roots = {}
        for output_root in sorted({path.parent.resolve() for path in selected}):
            locked_roots[output_root] = lifecycle_stack.enter_context(perf_finalization_lifecycle(output_root))
        return finalize_selected(locked_roots)


def finalize_perf_outputs(
    output_root: str | os.PathLike = ".",
    *,
    recursive: bool = False,
    delete_source: bool = True,
    compression: str = "zstd",
    merge_existing: bool = True,
    finalization_info: dict[Path, PerfFinalizationInfo] | None = None,
) -> list[Path]:
    """Finalize collector CSV staging files directly under `output_root` as parquet."""
    return finalize_perf_files(
        find_perf_csv_outputs(output_root, recursive=recursive),
        delete_source=delete_source,
        compression=compression,
        merge_existing=merge_existing,
        finalization_info=finalization_info,
    )


# Helper functions for MoE
def balanced_logits(num_tokens, num_experts, topk):
    import torch
    import torch.nn.functional as F

    stride = math.ceil(num_experts / topk)

    token_indices = torch.arange(num_tokens).unsqueeze(1)  # [num_tokens, 1]
    topk_indices = torch.arange(topk).unsqueeze(0)  # [1, topk]

    if num_tokens >= stride:
        h_selected_experts = (token_indices + topk_indices * stride) % num_experts
    else:
        h_selected_experts = (token_indices * stride / num_tokens + topk_indices * stride) % num_experts

    expert_map = F.one_hot(h_selected_experts.long(), num_classes=num_experts).sum(1)
    router_logits = F.softmax(expert_map.bfloat16(), dim=1)
    return router_logits


def sample_power_law(size, alpha, xmin, xmax):
    """Sample from a power law distribution using inverse CDF method.

    Args:
        size: Number of samples
        alpha: Power law exponent
        xmin: Minimum value
        xmax: Maximum value

    Returns:
        torch.Tensor of sampled values
    """
    import torch

    u = torch.rand(size)
    inv_cdf = ((xmax ** (1 - alpha) - xmin ** (1 - alpha)) * u + xmin ** (1 - alpha)) ** (1 / (1 - alpha))
    return inv_cdf


def compute_expert_replication(
    expert_tokens: np.ndarray,
    num_experts: int,
    num_slots: int,
) -> dict:
    """
    Step 1: Compute which experts should be replicated (redundant experts).

    When num_slots > num_experts, extra slots are used to replicate hot experts
    to balance load across ranks. Uses greedy algorithm to assign replicas.

    Args:
        expert_tokens: Token count array for each expert [num_experts]
        num_experts: Total number of experts (logical)
        num_slots: Total number of weight slots (physical), >= num_experts

    Returns:
        {
            'slot_to_expert': List[int],       # slot_id -> expert_id mapping [num_slots]
            'expert_replica_count': List[int], # How many slots each expert occupies
            'slot_tokens': np.ndarray,         # Token count per slot [num_slots]
            'num_redundant_slots': int,        # Number of extra slots (num_slots - num_experts)
        }
    """
    assert num_slots >= num_experts, f"num_slots ({num_slots}) must be >= num_experts ({num_experts})"

    num_redundant_slots = num_slots - num_experts

    if num_redundant_slots == 0:
        # No replication needed, 1:1 mapping
        return {
            "slot_to_expert": list(range(num_experts)),
            "expert_replica_count": [1] * num_experts,
            "slot_tokens": expert_tokens.copy(),
            "num_redundant_slots": 0,
        }

    # Initialize: each expert gets 1 slot first
    slot_to_expert = list(range(num_experts))
    expert_replica_count = [1] * num_experts

    # Use max-heap to efficiently find expert with highest effective load
    # Heap stores (-effective_load, expert_id) since heapq is min-heap
    # effective_load = expert_tokens[e] / expert_replica_count[e]
    heap = [(-expert_tokens[e], e) for e in range(num_experts)]
    heapq.heapify(heap)

    # Greedily assign redundant slots to experts with highest effective load
    for _ in range(num_redundant_slots):
        # Pop expert with highest effective load (most negative value)
        neg_load, hottest_expert = heapq.heappop(heap)

        # Add a replica for this expert
        slot_to_expert.append(hottest_expert)
        expert_replica_count[hottest_expert] += 1

        # Push back with updated effective load
        new_effective_load = expert_tokens[hottest_expert] / expert_replica_count[hottest_expert]
        heapq.heappush(heap, (-new_effective_load, hottest_expert))

    # Calculate tokens per slot (distributed among replicas of same expert)
    slot_tokens = np.zeros(num_slots, dtype=np.float64)
    for slot_id, expert_id in enumerate(slot_to_expert):
        slot_tokens[slot_id] = expert_tokens[expert_id] / expert_replica_count[expert_id]

    return {
        "slot_to_expert": slot_to_expert,
        "expert_replica_count": expert_replica_count,
        "slot_tokens": slot_tokens,
        "num_redundant_slots": num_redundant_slots,
    }


def compute_eplb_placement(
    slot_tokens: np.ndarray,
    num_slots: int,
    ep_size: int,
    slot_to_expert: Optional[list] = None,
) -> dict:
    """
    Step 2: Place slots (with replicas) onto ranks using greedy load balancing.

    Uses greedy algorithm to place slots from highest to lowest load
    onto the rank with the current minimum load.

    Args:
        slot_tokens: Token count array for each slot [num_slots]
        num_slots: Total number of slots (must be divisible by ep_size)
        ep_size: Expert parallelism size
        slot_to_expert: Optional slot_id -> expert_id mapping (for tracking)

    Returns:
        {
            'rank_slots': List[List[int]],     # Slot IDs owned by each rank
            'slot_to_rank': List[int],         # slot_id -> rank_id mapping
            'tokens_per_rank': List[float],    # Token count per rank
            'slowest_rank': int,               # ID of the slowest rank
            'slot_tokens': np.ndarray,         # Token count per slot
            'slot_to_expert': List[int],       # slot_id -> expert_id (passthrough)
        }
    """
    assert num_slots % ep_size == 0, f"num_slots ({num_slots}) must be divisible by ep_size ({ep_size})"
    slots_per_rank = num_slots // ep_size

    # EPLB greedy placement: sort slots by load descending, place on rank with min load
    sorted_slots = sorted(range(num_slots), key=lambda s: -slot_tokens[s])

    heap = [(0.0, r) for r in range(ep_size)]
    heapq.heapify(heap)

    rank_slots = [[] for _ in range(ep_size)]
    rank_slot_count = [0] * ep_size
    slot_to_rank = [-1] * num_slots

    for slot_id in sorted_slots:
        load, rank = heapq.heappop(heap)
        rank_slots[rank].append(slot_id)
        slot_to_rank[slot_id] = rank
        rank_slot_count[rank] += 1
        if rank_slot_count[rank] < slots_per_rank:
            heapq.heappush(heap, (load + slot_tokens[slot_id], rank))

    # Calculate token count per rank
    tokens_per_rank = [sum(slot_tokens[s] for s in rank_slots[r]) for r in range(ep_size)]

    # Default slot_to_expert if not provided (1:1 mapping)
    if slot_to_expert is None:
        slot_to_expert = list(range(num_slots))

    return {
        "rank_slots": rank_slots,
        "slot_to_rank": slot_to_rank,
        "tokens_per_rank": tokens_per_rank,
        "slowest_rank": int(np.argmax(tokens_per_rank)),
        "slot_tokens": slot_tokens,
        "slot_to_expert": slot_to_expert,
    }


def compute_eplb(
    expert_tokens: np.ndarray,
    num_experts: int,
    ep_size: int,
    num_slots: Optional[int] = None,
) -> dict:
    """
    Full EPLB pipeline: Replication + Placement.

    Convenience function that combines compute_expert_replication and
    compute_eplb_placement into a single call.

    Args:
        expert_tokens: Token count array for each expert [num_experts]
        num_experts: Total number of experts
        ep_size: Expert parallelism size
        num_slots: Total slots (default: num_experts, no redundancy)

    Returns:
        Combined result from both steps, plus:
        - 'rank_experts': List[List[int]] - Expert IDs (not slots) per rank
    """
    if num_slots is None:
        num_slots = num_experts

    # Step 1: Compute replication
    replication = compute_expert_replication(expert_tokens, num_experts, num_slots)

    # Step 2: Compute placement
    placement = compute_eplb_placement(
        replication["slot_tokens"],
        num_slots,
        ep_size,
        replication["slot_to_expert"],
    )

    # Build rank_experts (unique expert IDs per rank, for backward compatibility)
    rank_experts = [
        list(set(replication["slot_to_expert"][s] for s in rank_slots)) for rank_slots in placement["rank_slots"]
    ]

    return {
        # Replication info
        "slot_to_expert": replication["slot_to_expert"],
        "expert_replica_count": replication["expert_replica_count"],
        "num_redundant_slots": replication["num_redundant_slots"],
        # Placement info
        "rank_slots": placement["rank_slots"],
        "slot_to_rank": placement["slot_to_rank"],
        "tokens_per_rank": placement["tokens_per_rank"],
        "slowest_rank": placement["slowest_rank"],
        "slot_tokens": placement["slot_tokens"],
        # Derived
        "rank_experts": rank_experts,
        "expert_tokens": expert_tokens,
        "num_slots": num_slots,
        "num_experts": num_experts,
    }


def _assign_experts_from_counts(num_tokens_per_expert, num_tokens, topk):
    """Vectorized expert-to-token assignment from per-expert counts.

    Uses column-major fill: sort experts descending by count, repeat each expert
    by its count into a flat array, then reshape as (topk, num_tokens).T.

    Example: num_tokens = 5, topk = 2, num_tokens_per_expert = [4, 1, 3, 2]
    Then expert_ids_flat = [0, 0, 0, 0, 2, 2, 2, 3, 3, 1]
    and h_selected = [[0, 2],
                      [0, 2],
                      [0, 3],
                      [0, 3],
                      [2, 1]]
    Notice that there are no duplicate experts in any row.
    """
    import numpy as np
    import torch

    counts = num_tokens_per_expert.cpu().numpy().astype(np.int64)
    sorted_experts = np.argsort(-counts)
    sorted_counts = counts[sorted_experts]
    expert_ids_flat = np.repeat(sorted_experts, sorted_counts)
    h_selected = expert_ids_flat.reshape(topk, num_tokens).T.copy()
    return torch.from_numpy(h_selected).to(device=num_tokens_per_expert.device)


def _round_robin_adjust_per_rank(counts_2d, remaining, is_valid, pick_local_index, step):
    """Adjust local expert counts one rank at a time in round-robin order."""
    import torch

    # Integer redistribution can require tens of thousands of single-step updates for
    # large MoE token counts. Keep counts on CPU to avoid per-iteration GPU syncs
    # when torch.set_default_device(cuda) is active in collector workers.
    device = counts_2d.device
    counts_2d = counts_2d.cpu()

    while remaining > 0:
        progressed = False
        for rank_idx in range(counts_2d.size(0)):
            local_counts = counts_2d[rank_idx]
            valid_local = torch.nonzero(is_valid(local_counts)).flatten()
            if valid_local.numel() == 0:
                continue

            chosen_local_idx = valid_local[pick_local_index(local_counts[valid_local])].item()
            counts_2d[rank_idx, chosen_local_idx] += step
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return counts_2d.to(device)


def _generate_power_law_distribution(num_tokens, num_experts, topk, ep, alpha):
    """Core function to generate power law token distribution across experts.

    This is the shared logic used by power_law_logits_v3, power_law_deepep_prefill, and power_law_deepep_decode.

    Args:
        num_tokens: Number of tokens
        num_experts: Total number of experts
        topk: Number of experts per token
        ep: Expert parallelism size
        alpha: Power law exponent

    Returns:
        Tuple of (num_tokens_per_expert, h_selected_experts):
            - num_tokens_per_expert: Token count per expert (with EP rank 0 having max load)
            - h_selected_experts: Expert assignments matrix [num_tokens, topk]
    """
    import torch

    # Sample initial distribution
    if num_tokens * topk > num_experts:
        num_tokens_per_expert = sample_power_law(num_experts, alpha, 1, num_tokens * 0.8)
    else:
        num_tokens_per_expert = sample_power_law(num_experts, alpha, 0.01, 2)

    target_sum = num_tokens * topk
    original_distribution = num_tokens_per_expert / num_tokens_per_expert.sum()
    target_distribution = original_distribution * target_sum
    num_tokens_per_expert = torch.round(target_distribution).to(torch.int64).cpu()

    # Clamp to upper bound: each expert can be selected at most num_tokens times
    # (since each token can select an expert at most once)
    upper_bound = num_tokens
    overflow = int((num_tokens_per_expert - upper_bound).clamp(min=0).sum().item())
    num_tokens_per_expert = num_tokens_per_expert.clamp(max=upper_bound)
    experts_per_rank = num_experts // ep

    # Redistribute overflow to experts that haven't reached the bound
    if overflow > 0:
        num_tokens_per_expert_reshaped = num_tokens_per_expert.view(ep, experts_per_rank)
        num_tokens_per_expert_reshaped = _round_robin_adjust_per_rank(
            num_tokens_per_expert_reshaped,
            remaining=int(overflow),
            is_valid=lambda local_counts: local_counts < upper_bound,
            pick_local_index=torch.argmin,
            step=1,
        )
        num_tokens_per_expert = num_tokens_per_expert_reshaped.view(-1)

    # Adjust to match exact target sum (respecting upper bound)
    current_sum = num_tokens_per_expert.sum().item()
    delta = target_sum - current_sum
    if delta != 0:
        if delta > 0:
            num_tokens_per_expert_reshaped = num_tokens_per_expert.view(ep, experts_per_rank)
            num_tokens_per_expert_reshaped = _round_robin_adjust_per_rank(
                num_tokens_per_expert_reshaped,
                remaining=int(delta),
                is_valid=lambda local_counts: local_counts < upper_bound,
                pick_local_index=torch.argmin,
                step=1,
            )
            num_tokens_per_expert = num_tokens_per_expert_reshaped.view(-1)
        else:
            num_tokens_per_expert_reshaped = num_tokens_per_expert.view(ep, experts_per_rank)
            num_tokens_per_expert_reshaped = _round_robin_adjust_per_rank(
                num_tokens_per_expert_reshaped,
                remaining=int(-delta),
                is_valid=lambda local_counts: local_counts > 0,
                pick_local_index=torch.argmax,
                step=-1,
            )
            num_tokens_per_expert = num_tokens_per_expert_reshaped.view(-1)

    # Validate distribution
    if len(num_tokens_per_expert) > 1:
        sorted_tokens = torch.sort(num_tokens_per_expert, descending=True)[0]
        assert sorted_tokens[0] >= sorted_tokens[-1], "Power law distribution pattern disrupted"

    # Find EP rank with max load and swap to rank 0
    rank_sums = num_tokens_per_expert.view(ep, experts_per_rank).sum(dim=1)
    max_ep_idx = int(rank_sums.argmax().item())

    if max_ep_idx != 0:
        ep_group_size = num_experts // ep
        num_tokens_per_expert_reshaped = num_tokens_per_expert.view(ep, ep_group_size)
        num_tokens_per_expert_reshaped[0], num_tokens_per_expert_reshaped[max_ep_idx] = (
            num_tokens_per_expert_reshaped[max_ep_idx].clone(),
            num_tokens_per_expert_reshaped[0].clone(),
        )
        num_tokens_per_expert = num_tokens_per_expert_reshaped.view(-1)

    # Debug output
    aic_debug = int(os.getenv("AIC_DEBUG", "0"))
    if aic_debug >= 1:
        print("num_tokens_per_expert", num_tokens_per_expert, num_tokens_per_expert.sum().item())

    # Generate expert assignments (vectorized)
    h_selected_experts = _assign_experts_from_counts(num_tokens_per_expert, num_tokens, topk)

    return num_tokens_per_expert, h_selected_experts


def _generate_power_law_distribution_with_eplb(num_tokens, num_experts, topk, ep, alpha, num_slots=None):
    """Generate power law distribution with EPLB (Expert Parallel Load Balancer).

    EPLB has two phases:
    1. Replication: If num_slots > num_experts, hot experts are replicated to extra slots
    2. Placement: Slots are placed onto ranks using greedy load balancing

    The slowest rank's slots are then mapped to rank 0 for measurement.

    Args:
        num_tokens: Number of tokens
        num_experts: Total number of experts (logical)
        topk: Number of experts per token
        ep: Expert parallelism size
        alpha: Power law exponent
        num_slots: Total slots (default: num_experts, set higher for redundant experts)

    Returns:
        Tuple of (num_tokens_per_slot, h_selected_slots):
            - num_tokens_per_slot: Token count per slot (after remap, rank 0 is slowest) [num_slots]
            - h_selected_slots: Slot assignments matrix [num_tokens, topk]
    """
    import torch

    if num_slots is None:
        num_slots = num_experts

    num_slots // ep

    # Step 1: Sample initial power law distribution for experts
    if num_tokens * topk > num_experts:
        num_tokens_per_expert = sample_power_law(num_experts, alpha, 1, num_tokens * 0.8)
    else:
        num_tokens_per_expert = sample_power_law(num_experts, alpha, 0.01, 2)

    target_sum = num_tokens * topk
    original_distribution = num_tokens_per_expert / num_tokens_per_expert.sum()
    target_distribution = original_distribution * target_sum
    num_tokens_per_expert = torch.round(target_distribution).to(torch.int64).cpu()

    # Clamp to upper bound: each expert can be selected at most num_tokens times
    # (since each token can select an expert at most once)
    upper_bound = num_tokens
    overflow = int((num_tokens_per_expert - upper_bound).clamp(min=0).sum().item())
    num_tokens_per_expert = num_tokens_per_expert.clamp(max=upper_bound)

    # Redistribute overflow to experts that haven't reached the bound
    if overflow > 0:
        sorted_indices = torch.argsort(num_tokens_per_expert, descending=True)
        for _ in range(overflow):
            # Find an expert that hasn't reached the bound
            for j in range(len(sorted_indices)):
                expert_idx = sorted_indices[-(j + 1)]  # Start from smallest
                if num_tokens_per_expert[expert_idx] < upper_bound:
                    num_tokens_per_expert[expert_idx] += 1
                    break

    # Adjust to match exact target sum (respecting upper bound)
    current_sum = int(num_tokens_per_expert.sum().item())
    delta = target_sum - current_sum
    if delta != 0:
        sorted_indices = torch.argsort(num_tokens_per_expert, descending=True)
        if delta > 0:
            # Add to experts that haven't reached the bound
            added = 0
            for i in range(delta * len(sorted_indices)):  # Extra iterations for safety
                if added >= delta:
                    break
                expert_idx = sorted_indices[-(i % len(sorted_indices)) - 1]  # Start from smallest
                if num_tokens_per_expert[expert_idx] < upper_bound:
                    num_tokens_per_expert[expert_idx] += 1
                    added += 1
        else:
            for i in range(-delta):
                expert_idx = sorted_indices[-(i % len(sorted_indices)) - 1]
                if num_tokens_per_expert[expert_idx] > 0:
                    num_tokens_per_expert[expert_idx] -= 1
                else:
                    num_tokens_per_expert[torch.argmax(num_tokens_per_expert)] -= 1

    # Validate distribution
    if len(num_tokens_per_expert) > 1:
        sorted_tokens = torch.sort(num_tokens_per_expert, descending=True)[0]
        assert sorted_tokens[0] >= sorted_tokens[-1], "Power law distribution pattern disrupted"

    # Verify upper bound constraint
    assert num_tokens_per_expert.max().item() <= num_tokens, (
        f"Expert token count {num_tokens_per_expert.max().item()} exceeds num_tokens {num_tokens}"
    )

    # Step 2: EPLB - Replication + Placement
    expert_tokens_np = num_tokens_per_expert.cpu().numpy()
    eplb_result = compute_eplb(expert_tokens_np, num_experts, ep, num_slots)

    slowest_rank = eplb_result["slowest_rank"]
    rank_slots = eplb_result["rank_slots"]
    slot_tokens = eplb_result["slot_tokens"]
    slot_to_expert = eplb_result["slot_to_expert"]

    # Step 3: Rearrange slots so rank 0 owns the slowest rank's slots
    # Create new slot distribution array, rearranged according to EPLB result
    new_slot_tokens = torch.zeros(num_slots, dtype=torch.float64)
    new_slot_to_expert = [0] * num_slots

    new_slot_idx = 0

    # First place slowest_rank's slots into new rank 0
    for orig_slot in rank_slots[slowest_rank]:
        new_slot_tokens[new_slot_idx] = slot_tokens[orig_slot]
        new_slot_to_expert[new_slot_idx] = slot_to_expert[orig_slot]
        new_slot_idx += 1

    # Then place other ranks' slots
    for rank_id in range(ep):
        if rank_id == slowest_rank:
            continue
        for orig_slot in rank_slots[rank_id]:
            new_slot_tokens[new_slot_idx] = slot_tokens[orig_slot]
            new_slot_to_expert[new_slot_idx] = slot_to_expert[orig_slot]
            new_slot_idx += 1

    # Convert to int: use floor + distribute remainder by fractional part
    # This ensures exact sum without cumulative rounding errors
    floored = torch.floor(new_slot_tokens).to(torch.int64)
    remainder = target_sum - floored.sum().item()

    if remainder > 0:
        # Distribute remainder to slots with largest fractional parts
        fractional_parts = new_slot_tokens - floored.float()
        top_indices = torch.argsort(fractional_parts, descending=True)[:remainder]
        floored[top_indices] += 1

    num_tokens_per_slot = floored  # this num_tokens_per_slot is a list and each index means it's slot id

    # Debug output
    aic_debug = int(os.getenv("AIC_DEBUG", "0"))
    if aic_debug >= 1:
        print(f"EPLB: num_experts={num_experts}, num_slots={num_slots}, redundant={num_slots - num_experts}")
        print(f"EPLB: slowest_rank={slowest_rank}, tokens_per_rank={eplb_result['tokens_per_rank']}")
        print(f"EPLB: rank0 slots={rank_slots[slowest_rank][:5]}... (showing first 5)")
        print(f"EPLB: expert_replica_count (top 5 experts)={eplb_result['expert_replica_count'][:5]}")
        print("num_tokens_per_slot", num_tokens_per_slot[:10], "...", num_tokens_per_slot.sum().item())

    # Step 4: Generate slot assignments using per-token topk method
    # Each token selects topk DIFFERENT slots with highest remaining demand
    # This ensures no duplicate slots per token

    # Verify total count matches expected
    expected_total = num_tokens * topk
    actual_total = int(num_tokens_per_slot.sum().item())
    if actual_total != expected_total:
        raise ValueError(
            f"Slot assignment count mismatch: expected {expected_total}, got {actual_total}. "
            f"num_tokens={num_tokens}, topk={topk}, num_slots={num_slots}"
        )

    h_selected_slots = _assign_experts_from_counts(num_tokens_per_slot, num_tokens, topk)

    return num_tokens_per_slot, h_selected_slots


def power_law_logits_v3(
    num_tokens, num_experts, topk, ep, alpha, use_eplb=False, num_slots=None, return_rank0_info=False
):
    """Generate power law distributed router logits for MoE.

    Used by: sglang/collect_moe.py, vllm/collect_moe.py, trtllm/collect_moe.py

    Args:
        num_tokens: Number of tokens
        num_experts: Total number of experts
        topk: Number of experts per token
        ep: Expert parallelism size
        alpha: Power law exponent
        use_eplb: If True, use EPLB to balance load across ranks before measuring
        num_slots: Total weight slots (for redundant experts, must be >= num_experts)
                   Only used when use_eplb=True. Default: num_experts (no redundancy)
        return_rank0_info: If True, also return rank0 token indices and logits for WideEP simulation.
                           In WideEP, DP size = EP size, each DP rank has num_tokens/ep tokens.
                           This returns tokens that would be routed to EP rank 0.

    Returns:
        If return_rank0_info=False:
            router_logits: [num_tokens, num_slots] tensor of softmax probabilities
        If return_rank0_info=True:
            tuple of (router_logits, rank0_info) where rank0_info is a dict containing:
                - 'rank0_token_mask': [num_tokens] bool tensor, True for tokens routed to rank0
                - 'rank0_logits': [rank0_num_tokens, num_slots] filtered logits for rank0
                - 'rank0_num_tokens': number of tokens routed to rank0
                - 'slots_per_rank': number of slots per EP rank
    """
    import torch.nn.functional as F

    if use_eplb:
        # Use EPLB for load balanced distribution (with optional redundant experts)
        actual_num_slots = num_slots if num_slots is not None else num_experts
        num_tokens_per_slot, h_selected_slots = _generate_power_law_distribution_with_eplb(
            num_tokens, num_experts, topk, ep, alpha, num_slots=actual_num_slots
        )
        # Convert to router logits via one-hot encoding and softmax
        expert_map = F.one_hot(h_selected_slots.long(), num_classes=actual_num_slots).sum(1)
        router_logits = F.softmax(expert_map.bfloat16(), dim=1)

        if return_rank0_info:
            # Filter tokens that have ANY topk selection in rank0
            # In WideEP with EPLB, rank0 owns slots [0, slots_per_rank)
            slots_per_rank = actual_num_slots // ep
            # A token is routed to rank0 if any of its topk slots is in rank0
            rank0_selections_mask = h_selected_slots < slots_per_rank
            rank0_token_mask = rank0_selections_mask.any(dim=1)
            rank0_logits = router_logits[rank0_token_mask]
            rank0_num_tokens = rank0_logits.shape[0]
            rank0_total_selections = rank0_selections_mask.sum().item()
            # Get EPLB slot assignments for rank0 tokens
            rank0_selected_slots = h_selected_slots[rank0_token_mask]

            rank0_info = {
                "rank0_token_mask": rank0_token_mask,
                "rank0_logits": rank0_logits,
                "rank0_selected_slots": rank0_selected_slots,  # EPLB distribution for rank0 tokens
                "rank0_num_tokens": rank0_num_tokens,
                "slots_per_rank": slots_per_rank,
                "rank0_total_selections": rank0_total_selections,
            }
            return router_logits, rank0_info
        return router_logits
    else:
        # Original power law distribution (contiguous expert groups per rank)
        num_tokens_per_expert, h_selected_experts = _generate_power_law_distribution(
            num_tokens, num_experts, topk, ep, alpha
        )
        # Convert to router logits via one-hot encoding and softmax
        expert_map = F.one_hot(h_selected_experts.long(), num_classes=num_experts).sum(1)
        router_logits = F.softmax(expert_map.bfloat16(), dim=1)

        if return_rank0_info:
            # For non-EPLB, slots = experts, rank0 owns experts [0, experts_per_rank)
            experts_per_rank = num_experts // ep
            rank0_selections_mask = h_selected_experts < experts_per_rank
            rank0_token_mask = rank0_selections_mask.any(dim=1)
            rank0_logits = router_logits[rank0_token_mask]
            rank0_num_tokens = rank0_logits.shape[0]
            rank0_total_selections = rank0_selections_mask.sum().item()
            # Get expert assignments for rank0 tokens (for non-EPLB, slots = experts)
            rank0_selected_slots = h_selected_experts[rank0_token_mask]

            rank0_info = {
                "rank0_token_mask": rank0_token_mask,
                "rank0_logits": rank0_logits,
                "rank0_selected_slots": rank0_selected_slots,  # Expert distribution for rank0 tokens
                "rank0_num_tokens": rank0_num_tokens,
                "slots_per_rank": experts_per_rank,  # For non-EPLB, slots = experts
                "rank0_total_selections": rank0_total_selections,
            }
            return router_logits, rank0_info
        return router_logits


def build_rank0_local_workload(rank0_info: dict) -> dict[str, object]:
    """Convert global rank0 routing info into local-rank MoE inputs.

    Keeps the original global top-k probabilities and masks out remote experts
    so the returned tensors describe only the work executed by rank 0.
    """
    import torch

    rank0_selected_slots = rank0_info["rank0_selected_slots"].to(torch.int64)
    rank0_logits = rank0_info["rank0_logits"].to(torch.float32)
    slots_per_rank = int(rank0_info["slots_per_rank"])

    topk_weights = torch.gather(rank0_logits, 1, rank0_selected_slots.long()).to(torch.float32)

    local_mask = rank0_selected_slots < slots_per_rank
    topk_ids = rank0_selected_slots.to(torch.int32).clone()
    topk_ids[~local_mask] = -1
    topk_weights[~local_mask] = 0.0

    local_ids = topk_ids[topk_ids >= 0]
    masked_m = torch.bincount(local_ids, minlength=slots_per_rank).to(torch.int32)

    return {
        "num_tokens": int(rank0_info["rank0_num_tokens"]),
        "topk_ids": topk_ids.contiguous(),
        "topk_weights": topk_weights.contiguous(),
        "masked_m": masked_m.contiguous(),
    }


def power_law_deepep_prefill(num_tokens, num_experts, topk, ep, alpha):
    """Generate power law distribution for DeepEP MoE prefill phase.

    Used by: wideep/sglang/collect_deepep_moe.py

    Args:
        num_tokens: Number of tokens
        num_experts: Total number of experts
        topk: Number of experts per token
        ep: Expert parallelism size
        alpha: Power law exponent

    Returns:
        Tuple of (topk_idx, topk_weights, num_recv_tokens_per_expert):
            - topk_idx: [num_tokens, topk] expert indices (-1 for masked)
            - topk_weights: [num_tokens, topk] expert weights (0.0 for masked)
            - num_recv_tokens_per_expert: Padded token count per local expert
    """
    import torch

    num_tokens_per_expert, h_selected_experts = _generate_power_law_distribution(
        num_tokens, num_experts, topk, ep, alpha
    )

    # Convert to DeepEP format: topk_idx, topk_weights, num_recv
    num_local_experts = num_experts // ep
    topk_idx = h_selected_experts.clone().contiguous()
    topk_weights = torch.full_like(topk_idx, 0.1, dtype=torch.float32)

    # Mask experts not in rank 0
    mask = topk_idx >= num_local_experts
    topk_idx[mask] = -1
    topk_weights[mask] = 0.0

    # num_recv for rank 0 experts (padded to 128)
    num_recv_tokens_per_expert = num_tokens_per_expert[:num_local_experts]
    num_recv_tokens_per_expert = (num_recv_tokens_per_expert + 127) // 128 * 128

    return topk_idx, topk_weights, num_recv_tokens_per_expert


def power_law_deepep_decode(num_tokens, num_experts, topk, ep, alpha):
    """Generate power law distribution for DeepEP MoE decode phase.

    Creates a power law token distribution across all experts, then returns
    the distribution for the EP rank that has the highest total token count.

    Used by: wideep/sglang/collect_deepep_moe.py

    Args:
        num_tokens: Number of tokens
        num_experts: Total number of experts
        topk: Number of experts per token
        ep: Expert parallelism size
        alpha: Power law exponent

    Returns:
        Token count for each local expert on the max-load EP rank (rank 0 after swap)
    """
    # Reuse core distribution generation (max-load rank is swapped to rank 0)
    num_tokens_per_expert, _ = _generate_power_law_distribution(num_tokens, num_experts, topk, ep, alpha)
    experts_per_rank = num_experts // ep
    return num_tokens_per_expert.view(ep, experts_per_rank)[0]


# AIC's cached HuggingFace model configs — avoids HF downloads in CI.
_AIC_MODEL_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src",
    "aiconfigurator",
    "model_configs",
)


def _materialize_aic_cached_config(model_id: str, slug: str, cached_config: str) -> str:
    """Materialize a bundled AIC config as an immutable per-content snapshot.

    ``auto_map`` is stripped so that ``trust_remote_code=True`` consumers
    (e.g. SGLang's ServerArgs) do not try to import ``configuration_*.py``
    files that AIC does not ship.

    Snapshots live at ``<tmp>/aic_model_config_<slug>/<content-hash>/`` and
    are published with ONE atomic directory rename covering ``config.json``
    and the ``hf_quant_config.json`` side-car together, then never mutated.
    Readers (e.g. collect_mla_module's normalized-config copytree) can
    therefore never observe a torn pair — new config with an old or removed
    side-car — no matter how materialization interleaves with their reads;
    a bundled update under the same slug publishes a NEW snapshot at a new
    path while in-flight readers keep their consistent old one (#1487
    review: write-once staleness, then torn-pair publication). The path is
    deterministic per (slug, content), so parallel subprocesses /
    pytest-xdist workers converge on one directory, and losing the
    publication race is tolerated (same protocol as the normalized-config
    cache in collector/trtllm/collect_mla_module.py).
    """
    base_dir = os.path.join(tempfile.gettempdir(), f"aic_model_config_{slug}")

    with open(cached_config) as f:
        config = json.load(f)
    config.pop("auto_map", None)
    desired = json.dumps(config).encode()

    quant_side_car = os.path.join(_AIC_MODEL_CONFIG_DIR, f"{slug}_hf_quant_config.json")
    quant_desired = None
    if os.path.exists(quant_side_car):
        with open(quant_side_car, "rb") as f:
            quant_desired = f.read()

    hasher = hashlib.sha1(b"config.json\0" + desired)
    if quant_desired is not None:
        hasher.update(b"\0hf_quant_config.json\0" + quant_desired)
    snapshot = os.path.join(base_dir, hasher.hexdigest()[:16])

    if not os.path.exists(os.path.join(snapshot, "config.json")):
        # mkdtemp, not a pid-derived name: threads share a pid, and a shared
        # staging path would let one thread rename the dir out from under
        # another mid-write.
        os.makedirs(base_dir, exist_ok=True)
        staging = tempfile.mkdtemp(prefix=f"{os.path.basename(snapshot)}.stage-", dir=base_dir)
        try:
            with open(os.path.join(staging, "config.json"), "wb") as f:
                f.write(desired)
            if quant_desired is not None:
                with open(os.path.join(staging, "hf_quant_config.json"), "wb") as f:
                    f.write(quant_desired)
            os.replace(staging, snapshot)
        except OSError:
            # Another worker may have won the atomic-rename race — but
            # verify before trusting that assumption: a staged-write or
            # permission/disk-full failure would otherwise be swallowed.
            # Either way the staging directory must not linger.
            shutil.rmtree(staging, ignore_errors=True)
            if not os.path.exists(os.path.join(snapshot, "config.json")):
                raise

    print(f"Resolved {model_id} from AIC model_configs cache: {snapshot}")
    return snapshot


def config_norm_cache_key(src: str) -> str:
    """Cache key (16-hex sha1) for normalized copies of the config dir ``src``.

    Hashes the source path plus the name and bytes of every ``*.json``
    directly inside it — ``config.json`` AND side-cars like
    ``hf_quant_config.json``, because normalizers materialize the whole dir
    (``copytree``) and ``ModelConfig.from_pretrained`` reads the side-cars
    too. A path-only key would keep serving a stale normalized copy when a
    bundled config changes under an unchanged path (repo update).
    """
    if not os.path.exists(os.path.join(src, "config.json")):
        raise FileNotFoundError(f"'{src}' does not contain config.json")
    hasher = hashlib.sha1(src.encode())
    for name in sorted(os.listdir(src)):
        if not name.endswith(".json"):
            continue
        hasher.update(b"\0" + name.encode() + b"\0")
        with open(os.path.join(src, name), "rb") as f:
            hasher.update(f.read())
    return hasher.hexdigest()[:16]


def _resolve_local_model_path(model_id: str) -> str:
    """Resolve a model identifier to a local directory containing ``config.json``.

    Resolution order:
        1. Existing filesystem path. Must be a directory containing
           ``config.json`` — a file path or a directory without ``config.json``
           raises rather than silently falling through to HF download.
        2. AIC's bundled configs in ``src/aiconfigurator/model_configs/``
           (``<owner>--<name>_config.json``, with an optional
           ``..._hf_quant_config.json`` side-car).
        3. HuggingFace ``hf_hub_download``: ``config.json`` is required
           and downloaded first; tokenizer files are best-effort.

    Raises ``FileNotFoundError`` if none of the above resolves. There is
    no hardcoded ``/deepseek-v3`` (or any other model-specific) fallback —
    callers must supply a real ``model_id``.
    """
    if not model_id:
        raise ValueError("_resolve_local_model_path requires a non-empty model_id")

    # Step 1: existing filesystem path. Be strict about shape so a bogus
    # MOE_MODEL_PATH (e.g. pointing at a single file) fails loudly here
    # instead of silently triggering an HF download.
    if os.path.exists(model_id):
        if not os.path.isdir(model_id):
            raise NotADirectoryError(
                f"model_id '{model_id}' is an existing path but not a directory; "
                "expected a directory containing config.json"
            )
        if not os.path.exists(os.path.join(model_id, "config.json")):
            raise FileNotFoundError(f"model_id '{model_id}' is a directory but does not contain config.json")
        return model_id

    # Step 2: AIC bundled cache.
    slug = model_id.replace("/", "--")
    cached_config = os.path.join(_AIC_MODEL_CONFIG_DIR, f"{slug}_config.json")
    if os.path.exists(cached_config):
        return _materialize_aic_cached_config(model_id, slug, cached_config)

    # Step 3: HuggingFace download. config.json is mandatory and must
    # succeed before we accept the resulting snapshot directory.
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise FileNotFoundError(
            f"Model '{model_id}' not found under {_AIC_MODEL_CONFIG_DIR} and "
            "huggingface_hub is not installed — cannot download config."
        ) from e

    try:
        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
    except Exception as e:
        raise FileNotFoundError(
            f"Model '{model_id}' not found under {_AIC_MODEL_CONFIG_DIR} and "
            f"HuggingFace download of config.json failed: {e}"
        ) from e
    snapshot_dir = os.path.dirname(config_path)

    for filename in ("tokenizer_config.json", "tokenizer.json"):
        try:
            hf_hub_download(repo_id=model_id, filename=filename)
        except Exception as e:
            # Tokenizer files are best-effort — many MoE configs ship without them.
            print(f"Warning: failed to download {filename} for {model_id}: {e}")

    print(f"Resolved {model_id} from HuggingFace cache: {snapshot_dir}")
    return snapshot_dir


@functools.lru_cache(maxsize=1)
def get_device_module():
    import torch

    if torch.cuda.is_available():
        return torch.cuda
    elif torch.xpu.is_available():
        return torch.xpu
    raise RuntimeError("No supported device (need CUDA or XPU)")


@functools.lru_cache(maxsize=1)
def get_device_str():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif torch.xpu.is_available():
        return "xpu"
    raise RuntimeError("No supported device (need CUDA or XPU)")

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import heapq
import json
import logging
import math
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# Exit codes
EXIT_CODE_RESTART = 10  # Exit code to indicate restart is needed

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
    allow_graph_fail: bool = False,  # NEW: Enable graceful fallback on graph capture failure
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
        torch.cuda.synchronize()
        for _ in range(num_warmups):
            kernel_func()
        torch.cuda.synchronize()

    # ═══════════════════════════════════════════════════════════════════
    # CUDA Graph Capture with Optional Fallback
    # ═══════════════════════════════════════════════════════════════════
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
            torch.cuda.empty_cache()  # CRITICAL: Clean up partial allocations
            use_graph = False
        else:
            # Standard behavior: re-raise exception
            raise

    # ═══════════════════════════════════════════════════════════════════
    # Warmup the ACTUAL execution path (after graph capture)
    # ═══════════════════════════════════════════════════════════════════
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
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

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
    torch.cuda.synchronize()

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


def setup_signal_handlers(worker_id, error_queue=None):
    """Setup signal handlers to log crashes"""
    logger = logging.getLogger(f"worker_{worker_id}")

    def signal_handler(signum, frame):
        error_info = {
            "worker_id": worker_id,
            "signal": signum,
            "signal_name": signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum),
            "timestamp": datetime.now().isoformat(),
            "traceback": "".join(traceback.format_stack(frame)),
        }

        logger.error(f"Worker {worker_id} received signal {signum}")

        # Force flush all handlers
        for handler in logger.handlers:
            handler.flush()

        if error_queue:
            try:
                error_queue.put(error_info)
            except:
                pass

        # Re-raise the signal
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
    _LOG_DIR = Path(f"{'+'.join(scope)}_{time_stamp}")
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
    """Create unique identifier for test cases"""
    # Convert test case to string for hashing
    test_str = str(test_case)
    return f"{module_name}_{test_type}_{abs(hash(test_str)) % 100000}_{test_str}"


def log_perf(
    item_list: list[dict],
    framework: str,
    version: str,
    device_name: str,
    op_name: str,
    kernel_source: str,
    perf_filename: str,
    power_stats: dict | None = None,
):
    lock_file = perf_filename + ".lock"

    # Try for 1 sec (10 * 0.1s)
    got_lock = False
    for _ in range(10):
        try:
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            got_lock = True
            break
        except OSError:
            time.sleep(0.1)

    if not got_lock:
        print(f"Skipping log: can not get lock for {perf_filename}")
        return

    try:
        with open(perf_filename, "a", newline="") as f:
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
    except Exception as e:
        print(f"Error writing log: {e}")
    finally:
        # Delete the lock file, even if writing crashed
        if got_lock and os.path.exists(lock_file):
            os.unlink(lock_file)


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
    return torch.from_numpy(h_selected)


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
    num_tokens_per_expert = torch.round(target_distribution).to(torch.int64)

    # Clamp to upper bound: each expert can be selected at most num_tokens times
    # (since each token can select an expert at most once)
    upper_bound = num_tokens
    overflow = (num_tokens_per_expert - upper_bound).clamp(min=0).sum().item()
    num_tokens_per_expert = num_tokens_per_expert.clamp(max=upper_bound)

    # Redistribute overflow to experts that haven't reached the bound
    if overflow > 0:
        sorted_indices = torch.argsort(num_tokens_per_expert, descending=True)
        for i in range(int(overflow)):
            # Find an expert that hasn't reached the bound
            for j in range(len(sorted_indices)):
                expert_idx = sorted_indices[-(j + 1)]  # Start from smallest
                if num_tokens_per_expert[expert_idx] < upper_bound:
                    num_tokens_per_expert[expert_idx] += 1
                    break

    # Adjust to match exact target sum (respecting upper bound)
    current_sum = num_tokens_per_expert.sum().item()
    delta = target_sum - current_sum
    if delta != 0:
        sorted_indices = torch.argsort(num_tokens_per_expert, descending=True)
        if delta > 0:
            # Add to experts that haven't reached the bound
            added = 0
            for i in range(int(delta) * len(sorted_indices)):  # Extra iterations for safety
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

    # Find EP rank with max load and swap to rank 0
    with torch.no_grad():
        conv1d = torch.nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=num_experts // ep,
            stride=num_experts // ep,
            padding=0,
            bias=False,
        )
        conv1d_weights = torch.tensor([1 for _ in range(num_experts // ep)])
        conv1d.weight.copy_(conv1d_weights)

    res = conv1d(num_tokens_per_expert.unsqueeze(0).unsqueeze(0).float())
    max_ep_idx = torch.argmax(res).item()

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
    num_tokens_per_expert = torch.round(target_distribution).to(torch.int64)

    # Clamp to upper bound: each expert can be selected at most num_tokens times
    # (since each token can select an expert at most once)
    upper_bound = num_tokens
    overflow = (num_tokens_per_expert - upper_bound).clamp(min=0).sum().item()
    num_tokens_per_expert = num_tokens_per_expert.clamp(max=upper_bound)

    # Redistribute overflow to experts that haven't reached the bound
    if overflow > 0:
        sorted_indices = torch.argsort(num_tokens_per_expert, descending=True)
        for i in range(int(overflow)):
            # Find an expert that hasn't reached the bound
            for j in range(len(sorted_indices)):
                expert_idx = sorted_indices[-(j + 1)]  # Start from smallest
                if num_tokens_per_expert[expert_idx] < upper_bound:
                    num_tokens_per_expert[expert_idx] += 1
                    break

    # Adjust to match exact target sum (respecting upper bound)
    current_sum = num_tokens_per_expert.sum().item()
    delta = target_sum - current_sum
    if delta != 0:
        sorted_indices = torch.argsort(num_tokens_per_expert, descending=True)
        if delta > 0:
            # Add to experts that haven't reached the bound
            added = 0
            for i in range(int(delta) * len(sorted_indices)):  # Extra iterations for safety
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


def power_law_deepep_prefill(num_tokens, num_experts, topk, ep, alpha):
    """Generate power law distribution for DeepEP MoE prefill phase.

    Used by: sglang/collect_wideep_deepep_moe.py

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

    Used by: sglang/collect_wideep_deepep_moe.py

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


def _get_deepseek_model_path():
    """Get DeepSeek model path, downloading config files from HuggingFace if needed.

    If DEEPSEEK_MODEL_PATH is set, use that path.
    Otherwise, download only the necessary config files from HuggingFace.
    This allows running the collector without downloading the full model weights.
    """
    env_path = os.environ.get("DEEPSEEK_MODEL_PATH")
    if env_path:
        return env_path

    # Download config files from HuggingFace (no model weights needed)
    try:
        from huggingface_hub import hf_hub_download

        repo_id = "deepseek-ai/DeepSeek-V3"
        config_files = [
            "config.json",
            "configuration_deepseek.py",
            "tokenizer_config.json",
            "tokenizer.json",
        ]

        snapshot_dir = None
        for filename in config_files:
            try:
                path = hf_hub_download(repo_id=repo_id, filename=filename)
                if snapshot_dir is None:
                    snapshot_dir = os.path.dirname(path)
            except Exception as e:
                print(f"Warning: Failed to download {filename}: {e}")

        if snapshot_dir:
            print(f"Using DeepSeek-V3 config from HuggingFace cache: {snapshot_dir}")
            return snapshot_dir
    except ImportError:
        print("Warning: huggingface_hub not installed, cannot auto-download config")
    except Exception as e:
        print(f"Warning: Failed to download DeepSeek-V3 config: {e}")

    # Fallback to default path
    return "/deepseek-v3"

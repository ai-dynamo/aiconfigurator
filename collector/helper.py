# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import fcntl
import importlib.resources as pkg_resources
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import traceback

try:
    import cuda.bindings.driver as cuda
except:
    pass
from datetime import datetime
from pathlib import Path

import torch
import yaml


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
        debug = os.environ.get("COLLECTOR_DEBUG", "false").lower() == "true"
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

        logger.propagate = False  # Prevent duplicate logs
        # Silence noisy third-party loggers even if debug is true
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("h5py").setLevel(logging.WARNING)
        logging.getLogger("datasets").setLevel(logging.WARNING)
        logging.getLogger("flashinfer").setLevel(logging.ERROR)
        logging.getLogger("tensorrt_llm").setLevel(logging.ERROR)

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

    # Silence noisy third-party loggers globally
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("flashinfer").setLevel(logging.ERROR)
    logging.getLogger("tensorrt_llm").setLevel(logging.ERROR)

    _LOGGING_CONFIGURED = True

    return root_logger


def get_logging_config():
    """Get current logging configuration for passing to workers"""
    return {"debug": logging.getLogger().getEffectiveLevel() <= logging.DEBUG, "log_dir": _LOG_DIR}


def save_error_report(errors, filename):
    """Save error report"""
    with open(filename, "w") as f:
        json.dump(errors, f, indent=2)


def get_sm_version():
    # Init
    (err,) = cuda.cuInit(0)

    # Device
    err, cu_device = cuda.cuDeviceGet(0)

    # Get target architecture
    err, sm_major = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cu_device
    )
    err, sm_minor = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cu_device
    )

    return sm_major * 10 + sm_minor


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
):
    content_prefix = f"{framework},{version},{device_name},{op_name},{kernel_source}"
    header_prefix = "framework,version,device,op_name,kernel_source"
    for item in item_list:
        for key, value in item.items():
            content_prefix += f",{value}"
            header_prefix += f",{key}"

    with open(perf_filename, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)

        if os.fstat(f.fileno()).st_size == 0:
            f.write(header_prefix + "\n")

        f.write(content_prefix + "\n")


# ============================================================================
# Power Measurement Utilities
# ============================================================================

DTYPE_SIZES = {
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "fp8": 1,
    "fp8_block": 1,
    "int8": 1,
    "int4": 0.5,
}


def get_dtype_size(dtype: str) -> float:
    """Get size in bytes for a dtype"""
    dtype_lower = dtype.lower()
    if dtype_lower not in DTYPE_SIZES:
        raise ValueError(f"Unknown dtype: {dtype}")
    return DTYPE_SIZES[dtype_lower]


def _get_system_file_for_device(device_name: str) -> str:
    """Map GPU device name to system YAML filename.

    Args:
        device_name: GPU device name

    Returns:
        System YAML filename

    Raises:
        ValueError: If GPU is not supported
    """
    device_upper = device_name.upper()
    gpu_mappings = {
        "H200": "h200_sxm.yaml",
        "H100": "h100_sxm.yaml",
        "A100": "a100_sxm.yaml",
        "GB200": "gb200_sxm.yaml",  # Check GB200 before B200
        "B200": "b200_sxm.yaml",
    }

    for prefix, filename in gpu_mappings.items():
        if prefix in device_upper:
            return filename

    raise ValueError(f"Unsupported GPU: {device_name}")


def get_gpu_specs_from_device(device_name: str) -> dict:
    """Load GPU specifications from system YAML files.

    Dictionary keys are float16_tflops, fp8_tflops, int8_tflops, mem_bw_gbs, power_max.
    Keys follow system YAML files, except for power_max (which is just 'power' in YAML).
    """
    system_file = _get_system_file_for_device(device_name)
    systems_dir = pkg_resources.files("aiconfigurator") / "systems"
    yaml_path = systems_dir / system_file

    with open(yaml_path) as f:
        system_spec = yaml.safe_load(f)

    gpu = system_spec["gpu"]

    return {
        "float16_tflops": gpu["float16_tc_flops"] / 1e12,  # Convert to TFLOPS
        "fp8_tflops": gpu.get("fp8_tc_flops", gpu["float16_tc_flops"]) / 1e12,
        "int8_tflops": gpu["int8_tc_flops"] / 1e12,
        "mem_bw_gbs": gpu["mem_bw"] / 1e9,  # Convert to GB/s
        "power_max": gpu["power"],  # Watts
    }


def measure_kernel_power(
    power_monitor,
    kernel_fn,
    num_warmum_iters,
    target_duration_sec,
) -> tuple[float, float]:
    """
    Measure power for a memory-bound kernel by running for target_duration.

    Args:
        power_monitor: NVMLPowerMonitor instance
        kernel_fn: Kernel invocation function
        num_warmum_iters: Number of warmup iterations
        target_duration_sec: Target duration for measurement (seconds)

    Returns:
        (latency_ms, power_watts)
    """
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warmup
    start_event.record()
    for _ in range(num_warmum_iters):
        kernel_fn()
    end_event.record()
    torch.cuda.synchronize()

    # Calculate #iterations needed for target duration
    expected_kernel_latency = start_event.elapsed_time(end_event) / 1000  # seconds
    num_benchmark_iters = max(1, int(target_duration_sec / expected_kernel_latency))

    power_monitor.begin_window("kernel_benchmark", sync_execution=False)

    start_event.record()
    for _ in range(num_benchmark_iters):
        kernel_fn()
    end_event.record()
    torch.cuda.synchronize()

    measurement = power_monitor.end_window("kernel_benchmark", sync_execution=False)

    # Calculate metrics
    total_time_ms = start_event.elapsed_time(end_event)
    avg_latency_ms = total_time_ms / num_benchmark_iters

    total_energy_j = measurement.total_energy
    avg_power_watts = total_energy_j / (total_time_ms / 1000)  # J / seconds

    return avg_latency_ms, avg_power_watts


def get_system_spec_from_device(device_name: str) -> dict:
    """Load full system spec from device name.

    Args:
        device_name: GPU device name

    Returns:
        Full system_spec dict with 'gpu' key
    """
    system_file = _get_system_file_for_device(device_name)
    systems_dir = pkg_resources.files("aiconfigurator") / "systems"
    yaml_path = systems_dir / system_file

    with open(yaml_path) as f:
        system_spec = yaml.safe_load(f)

    return system_spec


def _get_gemm_quant_mode(dtype_str: str):
    """Map dtype string to GEMMQuantMode enum."""
    from aiconfigurator.sdk import common

    dtype_map = {
        "float16": common.GEMMQuantMode.float16,
        "fp8": common.GEMMQuantMode.fp8,
        "fp8_block": common.GEMMQuantMode.fp8_block,
        "nvfp4": common.GEMMQuantMode.nvfp4,
    }

    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    return dtype_map[dtype_str]


def _get_kvcache_quant_mode(dtype_str: str, use_fp8_kv_cache: bool):
    """Map dtype and fp8 flag to KVCacheQuantMode enum."""
    from aiconfigurator.sdk import common

    if use_fp8_kv_cache or "fp8" in dtype_str.lower():
        return common.KVCacheQuantMode.fp8
    else:
        return common.KVCacheQuantMode.float16


def _get_fmha_quant_mode(dtype_str: str, use_fp8_context_fmha: bool):
    """Map dtype and fp8 flag to FMHAQuantMode enum."""
    from aiconfigurator.sdk import common

    if use_fp8_context_fmha or "fp8" in dtype_str.lower():
        return common.FMHAQuantMode.fp8
    else:
        return common.FMHAQuantMode.float16


def is_gemm_compute_bound_collector(m: int, n: int, k: int, dtype: str, device_name: str) -> bool:
    """
    Determine if a GEMM operation is compute-bound.
    Wrapper for use in collectors.

    Args:
        m, n, k: GEMM dimensions (C = A @ B, A is mxk, B is kxn)
        dtype: Data type (e.g., 'float16', 'fp8')
        device_name: GPU device name

    Returns:
        True if compute-bound, False if memory-bound
    """
    from aiconfigurator.sdk import common
    from aiconfigurator.sdk.perf_database import PerfDatabase

    system_spec = get_system_spec_from_device(device_name)
    quant_mode = _get_gemm_quant_mode(dtype)

    # Create minimal PerfDatabase instance just to call query_gemm with SOL_FULL
    db = PerfDatabase.__new__(PerfDatabase)
    db.system_spec = system_spec

    sol_time, sol_math, sol_mem = db.query_gemm(m, n, k, quant_mode, sol_mode=common.SOLMode.SOL_FULL)
    return sol_math > sol_mem


def is_context_attention_compute_bound_collector(
    b: int,
    s: int,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    dtype: str,
    kv_cache_dtype: str,
    use_fp8_kv_cache: bool,
    use_fp8_context_fmha: bool,
    device_name: str,
    attention_window_size: int = 0,
) -> bool:
    """
    Determine if context (prefill) attention is compute-bound.
    Wrapper for use in collectors.

    Args:
        b: Batch size
        s: Sequence length (input)
        num_heads: Number of query heads (H_q)
        num_key_value_heads: Number of key/value heads (H_kv)
        head_dim: Head dimension
        dtype: Activation dtype
        kv_cache_dtype: KV cache dtype
        use_fp8_kv_cache: Whether using FP8 for KV cache
        use_fp8_context_fmha: Whether using FP8 for context FMHA
        device_name: GPU device name
        attention_window_size: Attention window size

    Returns:
        True if compute-bound, False if memory-bound
    """
    from aiconfigurator.sdk import common
    from aiconfigurator.sdk.perf_database import PerfDatabase

    system_spec = get_system_spec_from_device(device_name)
    kvcache_quant_mode = _get_kvcache_quant_mode(kv_cache_dtype, use_fp8_kv_cache)
    fmha_quant_mode = _get_fmha_quant_mode(dtype, use_fp8_context_fmha)

    # Create minimal PerfDatabase instance just to call query_context_attention with SOL_FULL
    db = PerfDatabase.__new__(PerfDatabase)
    db.system_spec = system_spec

    sol_time, sol_math, sol_mem = db.query_context_attention(
        b, s, num_heads, num_key_value_heads,
        kvcache_quant_mode, fmha_quant_mode,
        sol_mode=common.SOLMode.SOL_FULL,
        window_size=attention_window_size,
        head_size=head_dim
    )
    return sol_math > sol_mem


def is_generation_attention_compute_bound_collector() -> bool:
    """
    Determine if generation (decode) attention is compute-bound.
    Generation attention is ALWAYS memory-bound.

    Returns:
        False (always memory-bound)
    """
    return False

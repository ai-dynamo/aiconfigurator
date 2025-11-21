# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import fcntl
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import traceback

try:
    from cuda import cuda
except:
    from cuda.bindings import driver as cuda
from datetime import datetime
from pathlib import Path


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

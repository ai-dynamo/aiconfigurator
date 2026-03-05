# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import contextlib
import os
import warnings


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


import random

import torch
from tqdm import tqdm

setup_warning_filters()
import argparse
import json
import multiprocessing as mp
import signal
import time
import traceback
from datetime import datetime

from helper import EXIT_CODE_RESTART, create_test_case_id, save_error_report, setup_logging, setup_signal_handlers

logger = None


def collect_module_safe(module_name, test_type, get_test_cases_func, run_func, num_processes):
    """Safely collect module with comprehensive error handling"""
    full_name = f"{module_name}.{test_type}"
    logger.info(f"Starting collection: {full_name}")

    try:
        # Get test cases
        test_cases = get_test_cases_func()
        logger.info(f"Generated {len(test_cases)} test cases for {full_name}")

        # Run collection
        errors = parallel_run(test_cases, run_func, num_processes, full_name)

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


def worker(queue, device_id: int, func, progress_value, lock, error_queue=None, module_name="unknown"):
    """worker with automatic logging setup"""

    setup_warning_filters()  # Must run in each spawned process

    # Setup logging for this worker - reads config from environment automatically
    worker_logger = setup_logging(worker_id=device_id)

    # Setup signal handlers
    setup_signal_handlers(device_id, error_queue)

    # Setup device
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device_id)
    worker_logger.info(f"Worker {device_id} initialized for {module_name}")

    # Process tasks
    while True:
        task_info = queue.get()
        if task_info is None:
            worker_logger.debug("Received termination signal")
            break

        # Handle both old format (tuple) and new format (dict)
        if isinstance(task_info, dict):
            task_id = task_info.get("id", "unknown")
            task = task_info.get("params", task_info)
        else:
            task = task_info
            task_id = create_test_case_id(task, "unknown", module_name)

        try:
            worker_logger.debug(f"Starting task {task_id}")
            func(*task, device)
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
            # CRITICAL: Increment progress regardless of success or failure
            # This marks the task as "attempted" and tracks overall progress
            with lock:
                progress_value.value += 1

            # Periodic memory cleanup to reduce fragmentation
            # Only do this every 100 tasks to avoid overhead
            if progress_value.value % 100 == 0:
                import gc

                gc.collect()
                torch.cuda.empty_cache()


def parallel_run(tasks, func, num_processes, module_name="unknown"):
    """parallel runner with error collection"""
    queue = mp.Queue()
    error_queue = mp.Queue()
    processes = []
    manager = mp.Manager()
    progress_value = manager.Value("i", 0)
    lock = manager.Lock()

    # Track process health
    process_stats = {i: {"restarts": 0, "errors": []} for i in range(num_processes)}

    def start_process(device_id):
        p = mp.Process(
            target=worker,
            args=(queue, device_id, func, progress_value, lock, error_queue, module_name),
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

    # Start processes
    for device_id in range(num_processes):
        processes.append(start_process(device_id))

    # Queue tasks with IDs
    for i, task in enumerate(tasks):
        if not isinstance(task, dict):
            task_info = {
                "id": create_test_case_id(task, func.__name__, module_name),
                "params": task,
                "index": i,
            }
        else:
            task_info = task
        queue.put(task_info)

    # Add termination signals
    for _ in range(len(processes)):
        queue.put(None)

    # Monitor progress with error collection
    errors = []
    with tqdm(total=len(tasks), desc=f"{module_name}", dynamic_ncols=True, leave=True) as pbar:
        last_progress = 0
        stall_count = 0
        last_error_count = 0

        while progress_value.value < len(tasks):
            # Drain errors
            while not error_queue.empty():
                error = error_queue.get()
                errors.append(error)
                process_stats[error["device_id"]]["errors"].append(error["task_id"])

            # Update postfix only if count changed
            if len(errors) != last_error_count:
                pbar.set_postfix({"errors": len(errors)})
                last_error_count = len(errors)

            # Stall detection unchanged...
            if progress_value.value == last_progress:
                stall_count += 1
                if stall_count > 30:
                    logger.warning(f"Progress stalled at {progress_value.value}/{len(tasks)}")
            else:
                stall_count = 0
                last_progress = progress_value.value

            # Check process health
            for i, p in enumerate(processes):
                if not p.is_alive():
                    exit_code = p.exitcode
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

                    if process_stats[i]["restarts"] > 8192:
                        logger.error(f"Process {i} exceeded restart limit, not restarting")
                        continue

                    processes[i] = start_process(i)

            current = progress_value.value
            if current > pbar.n:
                pbar.update(current - pbar.n)

            time.sleep(2)

    # Collect remaining errors
    while not error_queue.empty():
        errors.append(error_queue.get())

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
                            raise CompatibilityError(
                                f"module {module_name} declares __compat__={declared!r}, runtime is v{runtime_version}"
                            )
                    except ValueError as e:
                        raise CompatibilityError(f"invalid __compat__ {declared!r}: {e}") from e

            get_func = getattr(get_module, collection["get_func"])
            run_func = getattr(run_module, collection["run_func"])

            def get_func_with_limit(get_func=get_func):
                cases = get_func()
                if shuffle:
                    rng = random.Random(shuffle_seed)
                    rng.shuffle(cases)
                if limit is not None:
                    cases = cases[:limit]
                return cases

            errors = collect_module_safe(
                collection["name"], collection["type"], get_func_with_limit, run_func, num_processes
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


def collect_sglang(num_processes: int, ops: list[str] | None = None, limit: int | None = None, shuffle: bool = False):
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
    all_errors = collect_ops(num_processes, collections, version, limit=limit, shuffle=shuffle)

    generate_collection_summary(all_errors, "sglang", version)


def collect_vllm(num_processes: int, ops: list[str] | None = None, limit: int | None = None, shuffle: bool = False):
    """Collect performance data for vLLM"""
    from collector.version_resolver import build_collections
    from collector.vllm.registry import REGISTRY

    try:
        from vllm.version import __version__ as vllm_version

        version = vllm_version
    except Exception:
        logger.exception("vLLM is not installed. Please install it from https://github.com/vllm-project/vllm")
        return

    collections = build_collections(REGISTRY, "vllm", version, ops, logger=logger)
    all_errors = collect_ops(num_processes, collections, version, limit=limit, shuffle=shuffle)

    generate_collection_summary(all_errors, "vllm", version)


def collect_trtllm(num_processes: int, ops: list[str] | None = None, limit: int | None = None, shuffle: bool = False):
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
    all_errors = collect_ops(num_processes, collections, version, limit=limit, shuffle=shuffle)

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
    args = parser.parse_args()
    ops = args.ops

    # Setup logging - debug flag is handled inside setup_logging
    if logger is None:
        logger = setup_logging(scope=args.ops if ops else ["all"], debug=args.debug)
    elif args.debug:
        # Update log level if debug flag changed
        setup_logging(debug=args.debug)

    num_processes = torch.cuda.device_count()
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

    mp.set_start_method("spawn")

    shuffle = args.shuffle
    limit = args.limit
    if args.smoke:
        shuffle = True
        limit = 4
        logger.info("Smoke test mode enabled — sampling 4 random test cases per op")

    if args.backend == "trtllm":
        collect_trtllm(num_processes, ops, limit=limit, shuffle=shuffle)
    elif args.backend == "sglang":
        collect_sglang(num_processes, ops, limit=limit, shuffle=shuffle)
    elif args.backend == "vllm":
        collect_vllm(num_processes, ops, limit=limit, shuffle=shuffle)


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    main()

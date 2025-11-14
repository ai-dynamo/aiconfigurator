# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import contextlib
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


setup_warning_filters()
import argparse
import json
import multiprocessing as mp
import os
import signal
import time
import traceback
from datetime import datetime

import torch
from tqdm import tqdm

from helper import create_test_case_id, save_error_report, setup_logging, setup_signal_handlers

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

        with lock:
            progress_value.value += 1

        try:
            worker_logger.debug(f"Starting task {task_id}")
            func(*task, device)
            worker_logger.debug(f"Completed task {task_id}")
        except Exception as e:
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

            if error_queue:
                error_queue.put(error_info)

            worker_logger.exception(f"Task {task_id} failed")

            # Force flush logs
            for handler in worker_logger.handlers:
                handler.flush()


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
        if exit_code in (None, 0):
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
                if stall_count > 30 and "moe" not in func.__name__:
                    logger.warning(f"Progress stalled at {progress_value.value}/{len(tasks)}")
                if stall_count > 900 and "moe" in func.__name__:
                    logger.warning(f"Moe Progress stalled at {progress_value.value}/{len(tasks)}")
            else:
                stall_count = 0
                last_progress = progress_value.value

            # Check process health
            for i, p in enumerate(processes):
                if not p.is_alive():
                    exit_code = p.exitcode
                    process_stats[i]["restarts"] += 1
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
        if "moe" in func.__name__:
            p.join(timeout=500) # tune + 30 tokens cases
        else:
            p.join(timeout=10)
        if p.is_alive():
            logger.warning(f"Process {p.pid} did not terminate, forcing...")
            p.terminate()

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


def collect_sglang():
    pass


def collect_vllm(num_processes: int):
    """
    Collect performance data for VLLM v1.
    """
    try:
        from vllm.version import __version__ as vllm_version

        version = vllm_version

    except:
        logger.exception("VLLM is not installed. Please install it from https://github.com/vllm-project/vllm")
        return

    # supported vllm v1 GEMM collection wich supports fp16,fp8,fp8_block_wise,awq and gptq
    try:
        import vllm_v1.collect_gemm

        test_cases = vllm_v1.collect_gemm.get_gemm_test_cases(is_unit_test=False)
        parallel_run(test_cases, vllm_v1.collect_gemm.run_gemm, num_processes)
        logger.info(f"collected gemm_vllm test cases for VLLM {version}")
    except:
        logger.warning("cannot collect gemm_vllm test cases, skipping...")

    # supported vllm v1 atten collection which supports fp16(auto) and fp8 kv cache. flashatten
    # impl for prefill and flashinfer impl for decode.
    try:
        import vllm_v1.collect_attn

        test_cases = vllm_v1.collect_attn.get_context_attention_test_cases(if_unit_test=False)
        parallel_run(test_cases, vllm_v1.collect_attn.run_attention_torch, num_processes)
        logger.info(f"collected context attention test cases for VLLM {version}")
        test_cases = vllm_v1.collect_attn.get_generation_attention_test_cases()
        parallel_run(test_cases, vllm_v1.collect_attn.run_attention_torch, num_processes)
        logger.info(f"collected generation attention test cases for VLLM {version}")
    except:
        logger.warning("cannot collect VLLM attention test cases, skipping...")


def collect_trtllm(num_processes: int, ops: list[str] | None = None):
    """Collect performance data for TensorRT LLM with enhanced error tracking"""
    all_errors = []

    os.environ["TLLM_LOG_LEVEL"] = "ERROR"
    os.environ["TRTLLM_DG_ENABLED"] = "1"
    # Suppress flashinfer logging
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
    except:
        logger.exception("TensorRT LLM is not installed")
        return

    # Define collection modules - each test type as separate entry
    collections = [
        # GEMM collections
        {
            "name": "trtllm",
            "type": "gemm_trt",
            "module": "trtllm.collect_gemm_trt",
            "get_func": "get_gemm_test_cases",
            "run_func": "run_gemm",
        },
        {
            "name": "trtllm",
            "type": "gemm",
            "module": "trtllm.collect_gemm",
            "get_func": "get_gemm_test_cases",
            "run_func": "run_gemm",
        },
        # MLA collections
        {
            "name": "trtllm",
            "type": "mla_context",
            "module": "trtllm.collect_mla",
            "get_func": "get_context_mla_test_cases",
            "run_func": "run_mla",
            "version_handler": lambda v: "trtllm.collect_mla_1_1rc2"
            if v.startswith(("1.1.0", "1.2.0"))
            else "trtllm.collect_mla",
        },
        {
            "name": "trtllm",
            "type": "mla_generation",
            "module": "trtllm.collect_mla",
            "get_func": "get_generation_mla_test_cases",
            "run_func": "run_mla",
            "version_handler": lambda v: "trtllm.collect_mla_1_1rc2"
            if v.startswith(("1.1.0", "1.2.0"))
            else "trtllm.collect_mla",
        },
        # Attention collections - separate entries for context and generation
        {
            "name": "trtllm",
            "type": "attention_context",
            "module": "trtllm.collect_attn",
            "get_func": "get_context_attention_test_cases",
            "run_func": "run_attention_torch",
        },
        {
            "name": "trtllm",
            "type": "attention_generation",
            "module": "trtllm.collect_attn",
            "get_func": "get_generation_attention_test_cases",
            "run_func": "run_attention_torch",
        },
        # MLA BMM collections
        {
            "name": "trtllm",
            "type": "mla_bmm_gen_pre",
            "module": "trtllm.collect_mla_bmm",
            "get_func": "get_mla_gen_pre_test_cases",
            "run_func": "run_mla_gen_pre",
        },
        {
            "name": "trtllm",
            "type": "mla_bmm_gen_post",
            "module": "trtllm.collect_mla_bmm",
            "get_func": "get_mla_gen_post_test_cases",
            "run_func": "run_mla_gen_post",
        },
        # MOE collection (with version handling)
        {
            "name": "trtllm",
            "type": "moe",
            "module": None,  # Will be determined based on version
            "get_func": "get_moe_test_cases",
            "run_func": "run_moe_torch",
            "version_handler": lambda v: "trtllm.collect_moe_pre_0_20"
            if v.startswith("0.20.0")
            else "trtllm.collect_moe_pre_1_0"
            if v.startswith(("0.21.0", "1.0.0"))
            else "trtllm.collect_moe"
            if v.startswith(("1.1.0", "1.2.0"))
            else None,
        },
    ]

    for collection in collections:
        if ops and (collection["type"] not in ops):
            continue
        try:
            # Handle version-specific modules
            if "version_handler" in collection:
                module_name = collection["version_handler"](version)
                if not module_name:
                    logger.warning(
                        f"Skipping {collection['name']}.{collection['type']} - unsupported version {version}",
                    )
                    continue
            else:
                module_name = collection["module"]

            get_module = __import__(module_name, fromlist=[collection["get_func"]])
            run_module = __import__(module_name, fromlist=[collection["run_func"]])

            get_func = getattr(get_module, collection["get_func"])
            run_func = getattr(run_module, collection["run_func"])

            errors = collect_module_safe(collection["name"], collection["type"], get_func, run_func, num_processes)
            all_errors.extend(errors)

        except Exception as e:
            logger.exception(f"Failed to process {collection['name']}.{collection['type']}")
            all_errors.append(
                {
                    "module": f"{collection['name']}.{collection['type']}",
                    "error_type": "ImportError",
                    "error_message": str(e),
                    "traceback": traceback.format_exc(),
                }
            )

    # Generate summary report
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


def main():
    global logger
    parser = argparse.ArgumentParser(description="Collect performance data for backends")
    parser.add_argument("--backend", type=str, choices=["trtllm", "sglang", "vllm"], default="trtllm")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--ops",
        nargs="*",
        type=str,
        choices=[
            "gemm_trt",
            "gemm",
            "mla_context",
            "mla_generation",
            "attention_context",
            "attention_generation",
            "mla_bmm_gen_pre",
            "mla_bmm_gen_post",
            "moe",
        ],
        help="Run only specified collection items. Leave empty to run all.",
        default=None,
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

    mp.set_start_method("spawn")

    if args.backend == "trtllm":
        collect_trtllm(num_processes, ops)
    elif args.backend == "sglang":
        collect_sglang(num_processes)
    elif args.backend == "vllm":
        collect_vllm(num_processes)


if __name__ == "__main__":
    main()

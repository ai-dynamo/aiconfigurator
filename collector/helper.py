# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import fcntl
import json
import logging
import math
import multiprocessing as mp
import os
import signal
import sys
import traceback

from datetime import datetime
from pathlib import Path

# Exit codes
EXIT_CODE_RESTART = 10  # Exit code to indicate restart is needed


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


# Helper functions for MoE
def balanced_logits(num_tokens, num_experts, topk):
    import torch
    import torch.nn.functional as F

    # h_selected_experts = -torch.ones([num_tokens, topk]).to(torch.device(device))
    h_selected_experts = -torch.ones([num_tokens, topk])
    stride = math.ceil(num_experts / topk)

    for token_i in range(num_tokens):
        for i in range(topk):
            if num_tokens >= stride:
                h_selected_experts[token_i][i] = (token_i + i * stride) % num_experts
            else:
                h_selected_experts[token_i][i] = (token_i * stride / num_tokens + i * stride) % num_experts

    expert_map = F.one_hot(h_selected_experts.long(), num_classes=num_experts).sum(1)
    router_logits = F.softmax(expert_map.bfloat16(), dim=1)
    return router_logits


def sample_power_law(size, alpha, xmin, xmax):
    import torch

    u = torch.rand(size)
    inv_cdf = ((xmax ** (1 - alpha) - xmin ** (1 - alpha)) * u + xmin ** (1 - alpha)) ** (1 / (1 - alpha))
    return inv_cdf


def power_law_logits_v3(num_tokens, num_experts, topk, ep, alpha):
    import torch
    import torch.nn.functional as F

    if num_tokens * topk > num_experts:
        num_tokens_per_expert = sample_power_law(num_experts, alpha, 1, num_tokens * 0.8)
    else:
        num_tokens_per_expert = sample_power_law(num_experts, alpha, 0.01, 2)

    target_sum = num_tokens * topk

    original_distribution = num_tokens_per_expert / num_tokens_per_expert.sum()

    target_distribution = original_distribution * target_sum

    num_tokens_per_expert = torch.round(target_distribution).to(torch.int64)

    current_sum = num_tokens_per_expert.sum().item()
    delta = target_sum - current_sum
    if delta != 0:
        sorted_indices = torch.argsort(num_tokens_per_expert, descending=True)

        if delta > 0:
            for i in range(delta):
                expert_idx = sorted_indices[i % len(sorted_indices)]
                num_tokens_per_expert[expert_idx] += 1
        else:
            for i in range(-delta):
                expert_idx = sorted_indices[-(i % len(sorted_indices)) - 1]
                if num_tokens_per_expert[expert_idx] > 0:
                    num_tokens_per_expert[expert_idx] -= 1
                else:
                    num_tokens_per_expert[torch.argmax(num_tokens_per_expert)] -= 1

    if len(num_tokens_per_expert) > 1:
        sorted_tokens = torch.sort(num_tokens_per_expert, descending=True)[0]
        assert sorted_tokens[0] >= sorted_tokens[-1], "Power law distribution pattern disrupted"

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

    aic_debug = int(os.getenv("AIC_DEBUG", "0"))
    if aic_debug == 1:
        print("num_tokens_per_expert", num_tokens_per_expert, num_tokens_per_expert.sum().item())

    _, num_tokens_per_expert_sorted_index = torch.sort(num_tokens_per_expert, descending=True)
    expert_assignments = []
    num_tokens_per_expert_sorted_index_lists = num_tokens_per_expert_sorted_index.tolist()
    for expert_id in num_tokens_per_expert_sorted_index_lists:
        expert_assignments.extend([expert_id] * num_tokens_per_expert[expert_id])

    expert_assignments = torch.tensor(expert_assignments, dtype=torch.long)
    h_selected_experts = expert_assignments.reshape(topk, num_tokens).T

    expert_map = F.one_hot(h_selected_experts.long(), num_classes=num_experts).sum(1)
    router_logits = F.softmax(expert_map.bfloat16(), dim=1)
    return router_logits


# NOTE: power_law_logits_v4 was copied from power_law_logits_v3 and
# modified to restrict max tokens per expert to be less than num_tokens
def power_law_logits_v4(num_tokens, num_experts, topk, ep, alpha):
    import torch

    """Generate power law distribution for token assignment to experts"""
    while True:
        if num_tokens * topk > num_experts:
            num_tokens_per_expert = sample_power_law(num_experts, alpha, 1, num_tokens * 0.8)
        else:
            num_tokens_per_expert = sample_power_law(num_experts, alpha, 0.01, 2)
        target_sum = num_tokens * topk

        original_distribution = num_tokens_per_expert / num_tokens_per_expert.sum()

        target_distribution = original_distribution * target_sum

        num_tokens_per_expert = torch.round(target_distribution).to(torch.int64)

        current_sum = num_tokens_per_expert.sum().item()
        delta = target_sum - current_sum
        if delta != 0:
            sorted_indices = torch.argsort(num_tokens_per_expert, descending=True)

            if delta > 0:
                for i in range(delta):
                    expert_idx = sorted_indices[i % len(sorted_indices)]
                    num_tokens_per_expert[expert_idx] += 1
            else:
                for i in range(-delta):
                    expert_idx = sorted_indices[-(i % len(sorted_indices)) - 1]
                    if num_tokens_per_expert[expert_idx] > 0:
                        num_tokens_per_expert[expert_idx] -= 1
                    else:
                        num_tokens_per_expert[torch.argmax(num_tokens_per_expert)] -= 1

        if len(num_tokens_per_expert) > 1:
            sorted_tokens = torch.sort(num_tokens_per_expert, descending=True)[0]
            assert sorted_tokens[0] >= sorted_tokens[-1], "Power law distribution pattern disrupted"

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
        num_tokens_per_expert_rank0 = num_tokens_per_expert.view(ep, num_experts // ep)[max_ep_idx].view(-1)
        if max(num_tokens_per_expert_rank0) <= num_tokens:
            return num_tokens_per_expert_rank0

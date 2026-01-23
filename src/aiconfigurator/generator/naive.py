# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Naive generator parameter builder for quick configuration generation.

This module provides utilities for building generator parameters using
aggressive parallelism settings (TP=gpus_per_node, PP=1) to maximize
the chance of fitting large models into memory without SLA optimization.
"""

from typing import Any, Optional

# Default GPUs per node for common systems
# gb200_sxm has 4 GPUs per node, most others have 8
_GPUS_PER_NODE_DEFAULTS = {
    "gb200_sxm": 4,
}
_DEFAULT_GPUS_PER_NODE = 8


def build_naive_generator_params(
    model_name: str,
    total_gpus: int,
    system_name: str,
    backend_name: str,
    model_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Build generator parameters for naive configuration generation.

    Uses aggressive parallelism (TP=gpus_per_node, PP=1) to maximize
    the chance of fitting large models into memory.

    Args:
        model_name: Name or HuggingFace ID of the model.
        total_gpus: Total number of GPUs available.
        system_name: Name of the system (e.g., 'h200_sxm', 'gb200_sxm').
        backend_name: Name of the backend (e.g., 'trtllm', 'sglang', 'vllm').
        model_path: Optional path to the model weights.

    Returns:
        Dictionary containing generator parameters with the structure:
        {
            "service": {...},
            "k8s": {...},
            "params": {
                "agg": {
                    "tensor_parallel_size": int,
                    "pipeline_parallel_size": int,
                    "max_batch_size": int,
                    ...
                }
            }
        }
    """
    # Get GPUs per node for this system
    gpus_per_node = _GPUS_PER_NODE_DEFAULTS.get(system_name, _DEFAULT_GPUS_PER_NODE)

    # Use aggressive parallelism: TP = gpus_per_node, PP = 1
    tensor_parallel_size = min(gpus_per_node, total_gpus)
    pipeline_parallel_size = 1

    # Default max batch size - conservative value that works for most models
    max_batch_size = 128

    # Build the generator params structure
    params = {
        "service": {
            "model_name": model_name,
            "served_model_name": model_name,
            "model_path": model_path or model_name,
        },
        "k8s": {
            "system_name": system_name,
        },
        "params": {
            "agg": {
                "tensor_parallel_size": tensor_parallel_size,
                "pipeline_parallel_size": pipeline_parallel_size,
                "max_batch_size": max_batch_size,
                "gpus_per_worker": tensor_parallel_size * pipeline_parallel_size,
            }
        },
        "dyn_config": {
            "mode": "agg",
        },
        "backend": backend_name,
    }

    return params

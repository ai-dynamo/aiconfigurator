# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Naive generator parameter builder for quick configuration generation.

This module provides utilities for building generator parameters using
aggressive parallelism settings (TP=gpus_per_node, PP=1) to maximize
the chance of fitting large models into memory without SLA optimization.
"""

import logging
import os
from importlib import resources as pkg_resources
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Default fallback if system config cannot be read
_DEFAULT_GPUS_PER_NODE = 8


def _get_gpus_per_node(system_name: str) -> int:
    """
    Read num_gpus_per_node from the system YAML config file.

    Args:
        system_name: Name of the system (e.g., 'h200_sxm', 'gb200_sxm').

    Returns:
        Number of GPUs per node for the system, or default of 8 if not found.
    """
    try:
        systems_dir = pkg_resources.files("aiconfigurator") / "systems"
        system_yaml_path = os.path.join(str(systems_dir), f"{system_name}.yaml")

        if os.path.isfile(system_yaml_path):
            with open(system_yaml_path) as f:
                system_spec = yaml.safe_load(f)
            gpus_per_node = system_spec.get("node", {}).get("num_gpus_per_node", _DEFAULT_GPUS_PER_NODE)
            return int(gpus_per_node)
    except Exception as e:
        logger.warning(f"Could not read system config for {system_name}: {e}")

    return _DEFAULT_GPUS_PER_NODE


def build_naive_generator_params(
    model_name: str,
    total_gpus: int,
    system_name: str,
    backend_name: str,
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
    # Get GPUs per node from system config
    gpus_per_node = _get_gpus_per_node(system_name)

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
            "model_path": model_name,
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

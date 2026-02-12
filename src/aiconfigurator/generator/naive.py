# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Naive generator parameter builder for quick configuration generation.

This module provides utilities for building generator parameters using
the smallest TP that fits the model in memory, with PP=1.
"""

import logging
import os
from typing import Any

import yaml

from aiconfigurator.sdk import perf_database

logger = logging.getLogger(__name__)

# Default fallbacks
_DEFAULT_GPUS_PER_NODE = 8
_DEFAULT_VRAM_BYTES = 141 * 1024 * 1024 * 1024  # 141 GiB (H200)
_MEMORY_MULTIPLIER = 1.5  # Require 1.5x model weight to fit in VRAM
_BYTES_PER_PARAM = 2  # FP16/BF16


def _get_system_config(system_name: str) -> dict[str, Any]:
    """
    Read system configuration from YAML config file.

    Args:
        system_name: Name of the system (e.g., 'h200_sxm', 'gb200').

    Returns:
        Dictionary with 'gpus_per_node' and 'vram_per_gpu' keys.
    """
    result = {
        "gpus_per_node": _DEFAULT_GPUS_PER_NODE,
        "vram_per_gpu": _DEFAULT_VRAM_BYTES,
    }

    try:
        for systems_root in perf_database.get_systems_paths():
            system_yaml_path = os.path.join(systems_root, f"{system_name}.yaml")
            if not os.path.isfile(system_yaml_path):
                continue
            with open(system_yaml_path) as f:
                system_spec = yaml.safe_load(f)
            result["gpus_per_node"] = int(system_spec.get("node", {}).get("num_gpus_per_node", _DEFAULT_GPUS_PER_NODE))
            result["vram_per_gpu"] = int(system_spec.get("gpu", {}).get("mem_capacity", _DEFAULT_VRAM_BYTES))
            break
    except Exception as e:
        logger.warning(f"Could not read system config for {system_name}: {e}")

    return result


def _estimate_model_weight_bytes(model_path: str) -> int:
    """
    Estimate model weight size in bytes based on model config.

    Formula based on DPP (Dynamo Performance Profiler):
    - Embedding: vocab_size * hidden_size
    - Per layer:
      - Attention: 4 * hidden_size^2 (Q, K, V, O projections)
      - FFN: 3 * hidden_size * inter_size (gate, up, down)
      - Layer norms: ~4 * hidden_size
    - For MoE: FFN * num_experts + router

    Args:
        model_path: HuggingFace model path or local path.

    Returns:
        Estimated model weight size in bytes.
    """
    from aiconfigurator.sdk.utils import get_model_config_from_model_path

    try:
        config = get_model_config_from_model_path(model_path)
        num_layers = config["layers"]
        hidden_size = config["hidden_size"]
        inter_size = config["inter_size"]
        vocab_size = config["vocab"]
        num_experts = config["num_experts"]
        moe_inter_size = config["moe_inter_size"]

        # Embedding parameters
        embedding_params = vocab_size * hidden_size

        # Per-layer parameters
        # Attention: Q, K, V, O projections = 4 * hidden^2
        attention_params = 4 * hidden_size * hidden_size

        # FFN parameters
        if num_experts and num_experts > 1:
            # MoE: gate + up + down for each expert, plus router
            ffn_inter = moe_inter_size if moe_inter_size else inter_size
            ffn_params = 3 * hidden_size * ffn_inter * num_experts
            # Router/gate
            ffn_params += hidden_size * num_experts
        else:
            # Dense: gate + up + down (for SwiGLU-style FFN)
            ffn_params = 3 * hidden_size * inter_size

        # Layer norms (2 per layer) + small bias terms
        norm_params = 4 * hidden_size

        # Total per layer
        per_layer_params = attention_params + ffn_params + norm_params

        # Total parameters
        total_params = embedding_params + (num_layers * per_layer_params)

        # Convert to bytes (FP16)
        weight_bytes = total_params * _BYTES_PER_PARAM

        logger.info(
            f"Estimated model weight size for {model_path}: "
            f"{weight_bytes / (1024**3):.2f} GiB ({total_params / 1e9:.2f}B params)"
        )

        return weight_bytes

    except Exception as e:
        logger.warning(f"Could not estimate model size for {model_path}: {e}")
        # Return a large fallback to be safe (assume 70B model @ FP16 = ~140GB)
        return 140 * 1024 * 1024 * 1024


def _calculate_min_tp(
    model_weight_bytes: int,
    vram_per_gpu: int,
    gpus_per_node: int,
    total_gpus: int,
) -> int:
    """
    Calculate the minimum TP size that fits the model in memory.

    Formula: tp * vram_per_gpu > memory_multiplier * model_weight_bytes

    Args:
        model_weight_bytes: Estimated model weight size in bytes.
        vram_per_gpu: VRAM per GPU in bytes.
        gpus_per_node: Number of GPUs per node.
        total_gpus: Total GPUs available.

    Returns:
        Minimum TP size (power of 2, capped at gpus_per_node and total_gpus).
    """
    # Required VRAM per model copy
    required_vram = model_weight_bytes * _MEMORY_MULTIPLIER

    # Find minimum TP where: tp * vram_per_gpu > required_vram
    # => tp > required_vram / vram_per_gpu
    min_tp_float = required_vram / vram_per_gpu
    min_tp = max(1, int(min_tp_float) + (1 if min_tp_float % 1 > 0 else 0))

    # Round up to power of 2 for efficiency
    tp = 1
    while tp < min_tp:
        tp *= 2

    # Cap at gpus_per_node (to stay within a single node) and total_gpus
    max_tp = min(gpus_per_node, total_gpus)

    # Warn if the model requires more GPUs than available in a single node
    if tp > max_tp:
        logger.warning(
            f"Model requires TP={tp} to fit in memory, but max TP is {max_tp} "
            f"(gpus_per_node={gpus_per_node}, total_gpus={total_gpus}). "
            f"The model may not fit! Consider using PP or other parallelism "
            f"strategies to fit across more than one node, or use a system "
            f"with more GPUs per node."
        )
        tp = max_tp

    logger.info(
        f"TP calculation: model={model_weight_bytes / (1024**3):.2f}GiB, "
        f"vram={vram_per_gpu / (1024**3):.2f}GiB, "
        f"required={required_vram / (1024**3):.2f}GiB (1.5x), "
        f"min_tp={min_tp}, selected_tp={tp}"
    )

    return tp


def build_naive_generator_params(
    model_name: str,
    total_gpus: int,
    system_name: str,
    backend_name: str,
) -> dict[str, Any]:
    """
    Build generator parameters for naive configuration generation.

    Calculates the smallest TP size that fits the model in memory:
    tp_size * VRAM_per_GPU > 1.5 * model_weight_size

    Args:
        model_name: Name or HuggingFace ID of the model.
        total_gpus: Total number of GPUs available.
        system_name: Name of the system (e.g., 'h200_sxm', 'gb200').
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
    # Get system config (GPUs per node and VRAM)
    system_config = _get_system_config(system_name)
    gpus_per_node = system_config["gpus_per_node"]
    vram_per_gpu = system_config["vram_per_gpu"]

    # Estimate model weight size
    model_weight_bytes = _estimate_model_weight_bytes(model_name)

    # Calculate minimum TP that fits the model
    tensor_parallel_size = _calculate_min_tp(
        model_weight_bytes=model_weight_bytes,
        vram_per_gpu=vram_per_gpu,
        gpus_per_node=gpus_per_node,
        total_gpus=total_gpus,
    )
    pipeline_parallel_size = 1

    # Default max batch size - conservative value that works for most models
    max_batch_size = 128

    # Build the generator params structure
    # Default SLA values for rule engine (used in max_num_tokens calculations)
    default_isl = 4000
    default_osl = 1000

    # Calculate number of workers (replicas) and GPUs per worker
    gpus_per_worker = tensor_parallel_size * pipeline_parallel_size
    agg_workers = total_gpus // gpus_per_worker

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
                "gpus_per_worker": gpus_per_worker,
            }
        },
        "DynConfig": {
            "mode": "agg",
        },
        "SlaConfig": {
            "isl": default_isl,
            "osl": default_osl,
        },
        "NodeConfig": {
            "num_gpus_per_node": gpus_per_node,
        },
        # WorkerConfig is used by the run script generator
        "WorkerConfig": {
            "agg_workers": agg_workers,
            "agg_gpus_per_worker": gpus_per_worker,
        },
        "backend": backend_name,
    }

    return params

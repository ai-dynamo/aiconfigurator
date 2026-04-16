# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Naive generator parameter builder for quick configuration generation.

This module provides utilities for building generator parameters using
the smallest parallelization that fits the model in memory.

For dense models, this is pure TP (tensor parallelism).  For MoE models,
the parallelization strategy depends on the model architecture and the
optimization objective:

- **Dense** (no MoE): TP
- **MLA + MoE + throughput** (DeepSeek-V3 family): DEP
- **All other sparse** (MLA + MoE + latency, GQA + MoE): TEP
"""

import logging
import os
import re
from typing import Any

import yaml

from aiconfigurator.sdk import perf_database
from aiconfigurator.sdk.utils import get_model_config_from_model_path

logger = logging.getLogger(__name__)

_RFC1123_MAX_LEN = 63

# Default fallbacks
_DEFAULT_GPUS_PER_NODE = 8
_DEFAULT_VRAM_BYTES = 141 * 1024 * 1024 * 1024  # 141 GiB (H200)
_MEMORY_MULTIPLIER = 1.5  # Require 1.5x model weight to fit in VRAM
_BYTES_PER_PARAM = 2  # FP16/BF16

# MoE architecture sets — must stay in sync with
# dynamo profiler's model_info.py (canonical source).
_MLA_MOE_ARCHITECTURES = {"DeepseekV3ForCausalLM", "DeepseekV32ForCausalLM"}
_MOE_ARCHITECTURES = _MLA_MOE_ARCHITECTURES | {
    "Qwen3MoeForCausalLM",
}


def _resolve_parallelization(
    architecture: str,
    is_moe: bool,
    num_gpus: int,
    optimization_type: str | None = None,
) -> dict[str, int]:
    """Return parallelization params for a given model architecture.

    The returned dict is suitable for merging into a worker params dict
    and contains the keys consumed by the generator (``tensor_parallel_size``,
    ``pipeline_parallel_size``, ``data_parallel_size``,
    ``moe_tensor_parallel_size``, ``moe_expert_parallel_size``).

    Rules (same for agg and disagg):
    - **Dense**: TP = num_gpus
    - **MLA + MoE + throughput** (DeepSeek-V3): DEP = num_gpus
    - **All other sparse**: TEP = num_gpus
    """
    if not is_moe:
        return {
            "tensor_parallel_size": num_gpus,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "moe_tensor_parallel_size": 1,
            "moe_expert_parallel_size": 1,
        }

    # MLA + MoE + throughput → DEP
    if architecture in _MLA_MOE_ARCHITECTURES and optimization_type == "throughput":
        return {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": num_gpus,
            "moe_tensor_parallel_size": 1,
            "moe_expert_parallel_size": num_gpus,
        }

    # All other sparse → TEP
    return {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "moe_tensor_parallel_size": num_gpus,
        "moe_expert_parallel_size": 1,
    }


def _sanitize_rfc1123(name: str) -> str:
    """Sanitize a string to be a valid RFC 1123 subdomain label prefix.

    Converts ``"Qwen/Qwen3-32B"`` → ``"qwen-qwen3-32b"``, etc.
    Falls back to ``"dynamo"`` when the input is empty or None.
    """
    if not name:
        return "dynamo"
    sanitized = name.lower()
    sanitized = re.sub(r"[^a-z0-9\-.]", "-", sanitized)
    sanitized = re.sub(r"-{2,}", "-", sanitized)
    sanitized = sanitized.strip("-.")
    sanitized = sanitized[:_RFC1123_MAX_LEN].rstrip("-.")
    return sanitized or "dynamo"


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

    Raises:
        RuntimeError: If the model config cannot be fetched (e.g. model not found
            on HuggingFace). Callers must not proceed with guessed parameters.
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
        logger.exception("Could not estimate model size for %s.", model_path)
        raise RuntimeError(f"Model {model_path!r} not found or config unavailable") from e


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
    mode: str = "agg",
    optimization_type: str | None = None,
) -> dict[str, Any]:
    """
    Build generator parameters for naive configuration generation.

    Calculates the smallest parallelization that fits the model in memory
    and selects the appropriate strategy (TP, TEP, or DEP) based on the
    model architecture and optimization objective.

    Args:
        model_name: Name or HuggingFace ID of the model.
        total_gpus: Total number of GPUs available.
        system_name: Name of the system (e.g., 'h200_sxm', 'gb200').
        backend_name: Name of the backend (e.g., 'trtllm', 'sglang', 'vllm').
        mode: Serving mode — ``"agg"`` (aggregated, single worker type) or
            ``"disagg"`` (disaggregated, separate prefill/decode workers).
        optimization_type: ``"throughput"`` or ``"latency"`` (or ``None``
            for legacy callers). Influences parallelization for MoE models.

    Returns:
        Dictionary containing generator parameters.  When ``mode="agg"``,
        ``params.agg`` is populated.  When ``mode="disagg"``, both
        ``params.prefill`` and ``params.decode`` are populated with
        identical parallelization.
    """
    # Get system config (GPUs per node and VRAM)
    system_config = _get_system_config(system_name)
    gpus_per_node = system_config["gpus_per_node"]
    vram_per_gpu = system_config["vram_per_gpu"]

    # Estimate model weight size
    model_weight_bytes = _estimate_model_weight_bytes(model_name)

    # Calculate minimum GPU count that fits the model
    min_gpus = _calculate_min_tp(
        model_weight_bytes=model_weight_bytes,
        vram_per_gpu=vram_per_gpu,
        gpus_per_node=gpus_per_node,
        total_gpus=total_gpus,
    )

    # Detect model architecture for MoE-aware parallelization
    architecture = ""
    is_moe = False
    try:
        model_config = get_model_config_from_model_path(model_name)
        architecture = model_config.get("architecture", "")
        num_experts = model_config.get("num_experts", 0)
        is_moe = bool(num_experts and num_experts > 1)
    except Exception:
        logger.warning(
            "Could not detect model architecture for %s; assuming dense (TP-only).",
            model_name,
        )

    # Resolve parallelization strategy
    parallel = _resolve_parallelization(
        architecture=architecture,
        is_moe=is_moe,
        num_gpus=min_gpus,
        optimization_type=optimization_type,
    )

    strategy = "TP" if not is_moe else ("DEP" if parallel["data_parallel_size"] > 1 else "TEP")
    logger.info(
        "Naive config: model=%s, strategy=%s=%d, optimization_type=%s, mode=%s",
        model_name,
        strategy,
        min_gpus,
        optimization_type or "default",
        mode,
    )

    # Default max batch size - conservative value that works for most models
    max_batch_size = 128

    # Build the generator params structure
    default_isl = 4000
    default_osl = 1000

    # Worker params shared by all modes
    worker_params = {
        **parallel,
        "max_batch_size": max_batch_size,
        "gpus_per_worker": min_gpus,
    }

    name_prefix = _sanitize_rfc1123(model_name)

    if mode == "disagg":
        # Disaggregated: separate prefill and decode workers with identical parallelization
        if total_gpus < 2 * min_gpus:
            logger.warning(
                "Disaggregated mode requires at least %d GPUs (%d prefill + %d decode), "
                "but only %d are available. Workers may overcommit GPU resources.",
                2 * min_gpus,
                min_gpus,
                min_gpus,
                total_gpus,
            )
        prefill_workers = 1
        decode_workers = max(1, (total_gpus // min_gpus) - 1) if total_gpus > min_gpus else 1
        params = {
            "ServiceConfig": {
                "model_name": model_name,
                "served_model_name": model_name,
                "model_path": model_name,
                "include_frontend": True,
            },
            "K8sConfig": {
                "system_name": system_name,
                "name_prefix": name_prefix,
            },
            "params": {
                # TODO: consider tuning prefill-specific defaults for
                # max_batch_size and max_num_tokens separately from decode.
                "prefill": dict(worker_params),
                "decode": dict(worker_params),
            },
            "DynConfig": {
                "mode": "disagg",
            },
            "SlaConfig": {
                "isl": default_isl,
                "osl": default_osl,
            },
            "NodeConfig": {
                "num_gpus_per_node": gpus_per_node,
            },
            "WorkerConfig": {
                "prefill_workers": prefill_workers,
                "prefill_gpus_per_worker": min_gpus,
                "decode_workers": decode_workers,
                "decode_gpus_per_worker": min_gpus,
            },
            "ModelConfig": {
                "is_moe": is_moe,
            },
            "backend": backend_name,
        }
    else:
        # Aggregated: single worker type
        agg_workers = total_gpus // min_gpus
        params = {
            "ServiceConfig": {
                "model_name": model_name,
                "served_model_name": model_name,
                "model_path": model_name,
                "include_frontend": True,
            },
            "K8sConfig": {
                "system_name": system_name,
                "name_prefix": name_prefix,
            },
            "params": {
                "agg": dict(worker_params),
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
            "WorkerConfig": {
                "agg_workers": agg_workers,
                "agg_gpus_per_worker": min_gpus,
            },
            "ModelConfig": {
                "is_moe": is_moe,
            },
            "backend": backend_name,
        }

    return params

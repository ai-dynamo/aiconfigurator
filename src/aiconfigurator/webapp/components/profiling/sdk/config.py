# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Configuration generation and validation for profiling.

This module handles:
- Generating YAML configs for profiling data points
- Validating profiling inputs
- Generating GPU configuration ranges
"""

import math
import re

import yaml

from aiconfigurator.generator.api import generate_backend_artifacts, generate_config_from_input_dict

# Constants
DECODE_MAX_CONCURRENCY = 1024
DEFAULT_NAME_PREFIX = "profiling"


def generate_config_yaml(
    model_name: str,
    system: str,
    backend: str,
    version: str,
    isl: int,
    osl: int,
    num_gpus: int,
    batch_size: int = 1,
) -> str:
    """
    Generate a k8s_deploy.yaml string for a profiling data point.

    Args:
        model_name: Model name (e.g., "QWEN3_32B")
        system: System name (e.g., "h200_sxm")
        backend: Backend name (e.g., "trtllm")
        version: Backend template version (e.g., "0.20.0")
        isl: Input sequence length
        osl: Output sequence length
        num_gpus: Number of GPUs (becomes tp_size)
        batch_size: Batch size for the configuration

    Returns:
        YAML string representation of the k8s_deploy.yaml config.
        Returns "N/A" if backend generator templates are not available.
    """
    tp_size = max(1, num_gpus)
    safe_prefix = (system or DEFAULT_NAME_PREFIX).strip() or DEFAULT_NAME_PREFIX
    name_prefix = safe_prefix.replace(" ", "-").lower()

    input_params = {
        "SlaConfig": {"isl": isl, "osl": osl},
        "ServiceConfig": {
            "model_path": model_name,
            "served_model_name": model_name,
            "model_name": model_name,
            "include_frontend": True,
        },
        "ModelConfig": {
            "is_moe": False,
            "nextn": 0,
            "prefix": 0,
        },
        "DynConfig": {
            "mode": "agg",
        },
        "K8sConfig": {
            "name_prefix": name_prefix,
            "k8s_namespace": "dynamo",
            "k8s_image": "nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.5.0",
            "k8s_model_cache": "model-cache",
            "k8s_engine_mode": "inline",
        },
        "Workers": {
            "agg": {
                "tensor_parallel_size": tp_size,
                "pipeline_parallel_size": 1,
                "data_parallel_size": 1,
                "moe_tensor_parallel_size": 1,
                "moe_expert_parallel_size": 1,
                "max_batch_size": batch_size,
                "gemm_quant_mode": "float16",
                "moe_quant_mode": "float16",
                "kvcache_quant_mode": "float16",
                "fmha_quant_mode": "float16",
                "comm_quant_mode": "half",
            }
        },
        "WorkerConfig": {
            "agg_workers": 1,
        },
    }

    try:
        params = generate_config_from_input_dict(input_params, backend=backend)
        artifacts = generate_backend_artifacts(
            params=params,
            backend=backend,
            backend_version=version,
        )
    except (FileNotFoundError, ValueError) as exc:
        message = str(exc)
        if "Unknown backend" in message or "Templates directory not found" in message:
            return "N/A"
        raise

    k8s_payload = artifacts.get("k8s_deploy.yaml")
    if not k8s_payload:
        raise ValueError("Failed to generate k8s_deploy.yaml from artifacts")

    result = (
        k8s_payload
        if isinstance(k8s_payload, str)
        else yaml.dump(k8s_payload, sort_keys=False, default_flow_style=False, width=4096)
    )

    # Clean up repeated newlines from jinja2 template rendering for better readability
    result = re.sub(r"\n{2,}", "\n", result).lstrip("\n")

    return result


def validate_inputs(model_name, system, backend, version):
    """
    Validate profiling inputs.

    Args:
        model_name: Model name
        system: System name
        backend: Backend name
        version: Backend version

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not model_name or not system or not version:
        return False, "❌ Missing required parameters (model_name, system, or version)"

    return True, None


def generate_gpu_configurations(min_num_gpus, max_num_gpus):
    """
    Generate GPU counts to profile (powers of 2).

    Args:
        min_num_gpus: Minimum number of GPUs
        max_num_gpus: Maximum number of GPUs

    Returns:
        List of GPU counts to profile
    """
    profile_num_gpus = [2**i for i in range(int(math.log2(max_num_gpus)) + 1) if min_num_gpus <= 2**i <= max_num_gpus]
    return profile_num_gpus


def get_num_request_range(attn_dp_size, engine_max_concurrency, granularity):
    """
    Generate request count range for decode profiling.

    Args:
        attn_dp_size: Attention data parallelism size (1 for dense models)
        engine_max_concurrency: Maximum concurrency for the engine
        granularity: Number of points to sample

    Returns:
        List of request counts to profile
    """
    max_concurrency = min(engine_max_concurrency, DECODE_MAX_CONCURRENCY)
    conc_per_dp = max_concurrency // attn_dp_size

    if conc_per_dp < granularity:
        ans = list(range(attn_dp_size, conc_per_dp * attn_dp_size + 1, attn_dp_size))
    else:
        step = (conc_per_dp - 1) * attn_dp_size / (granularity - 1)
        ans = [attn_dp_size + int(i * step) * attn_dp_size for i in range(granularity)]

    return ans


def enumerate_moe_configs(tp_size, attention_dp_size=1):
    """
    Enumerate all valid (moe_tp_size, moe_ep_size) pairs for a given tp_size.

    Constraint: tp_size * attention_dp_size = moe_tp_size * moe_ep_size

    Args:
        tp_size: Tensor parallel size
        attention_dp_size: Attention data parallel size (default 1)

    Returns:
        List of (moe_tp_size, moe_ep_size) tuples
    """
    target = tp_size * attention_dp_size
    configs = []

    # Find all factorizations of target
    for moe_tp in [1, 2, 4, 8, 16, 32]:
        if target % moe_tp == 0:
            moe_ep = target // moe_tp
            if moe_ep in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                configs.append((moe_tp, moe_ep))

    return configs

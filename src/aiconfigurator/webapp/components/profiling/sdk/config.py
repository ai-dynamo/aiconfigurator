"""
Configuration generation and validation for profiling.

This module handles:
- Generating YAML configs for profiling data points
- Validating profiling inputs
- Generating GPU configuration ranges
"""

import math

import yaml

# Constants
DECODE_MAX_CONCURRENCY = 1024


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
    Generate a config YAML string for a profiling data point.

    Args:
        model_name: Model name (e.g., "QWEN3_32B")
        system: System name (e.g., "h200_sxm")
        backend: Backend name (e.g., "trtllm")
        version: Backend version (e.g., "0.20.0")
        isl: Input sequence length
        osl: Output sequence length
        num_gpus: Number of GPUs (becomes tp_size)
        batch_size: Batch size for the configuration

    Returns:
        YAML string representation of the config
    """
    config = {
        "model_name": model_name,
        "system_name": system,
        "total_gpus": num_gpus,
        "serving_mode": "agg",
        "nextn": 0,
        "is_moe": False,
        "isl": isl,
        "osl": osl,
        "prefix": 0,
        "agg_worker_config": {
            "gemm_quant_mode": "float16",
            "moe_quant_mode": "float16",
            "kvcache_quant_mode": "float16",
            "fmha_quant_mode": "float16",
            "comm_quant_mode": "half",
            "bs": batch_size,
            "workers": 1,
            "tp": num_gpus,
            "pp": 1,
            "dp": 1,
            "moe_tp": 1,
            "moe_ep": 1,
            "memory": 0,  # Will be computed by aiconfigurator
        },
    }

    return yaml.dump(config, sort_keys=False, default_flow_style=False)


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

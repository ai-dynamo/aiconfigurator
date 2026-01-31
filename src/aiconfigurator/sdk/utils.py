# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.resources as pkg_resources
import json
import logging
import re
import tempfile
import urllib.request
from pathlib import Path

from aiconfigurator.sdk import common
from aiconfigurator.sdk.common import ARCHITECTURE_TO_MODEL_FAMILY, BlockConfig, DefaultHFModels

logger = logging.getLogger(__name__)


def _load_json_with_infinity(file_path) -> dict:
    """
    Load JSON file with support for JavaScript-style Infinity and NaN values.

    Standard JSON doesn't support Infinity/NaN, but HuggingFace configs may contain them (e.g., Nemotron-H-56B)
    This function pre-processes the file content to replace these values before parsing.
    """
    with open(file_path) as f:
        content = f.read()
    # Replace JavaScript-style Infinity/NaN with Python-compatible values
    # Use regex to match standalone Infinity/-Infinity/NaN (not part of a string)
    content = re.sub(r"\bInfinity\b", "null", content)
    content = re.sub(r"-Infinity\b", "null", content)
    content = re.sub(r"\bNaN\b", "null", content)
    return json.loads(content)


def enumerate_parallel_config(
    num_gpu_list: list[int],
    tp_list: list[int],
    pp_list: list[int],
    dp_list: list[int] = [1],
    moe_tp_list: list[int] = [1],
    moe_ep_list: list[int] = [1],
    is_moe: bool = False,
    backend: common.BackendName = common.BackendName.trtllm,
    enable_wideep: bool = False,
) -> list[list[int]]:
    """
    Enumerate parallel configurations based on parallel list.
    This is a helper function for agg_pareto and disagg_pareto to define search space.

    Args:
        num_gpu_list: list of number of gpus, this is used to filter out invalid parallel
            configurations
        tp_list: list of tensor parallel sizes
        pp_list: list of pipeline parallel sizes
        dp_list: list of data parallel sizes
        moe_tp_list: list of moe tensor parallel sizes
        moe_ep_list: list of moe expert parallel sizes
        is_moe: whether to use moe
        backend: backend name enum. Important for moe parallel enumeration as different backends
            have different moe parallel support.
    Returns:
        parallel_config_list: list of parallel configurations
    """
    parallel_config_list = []
    for tp in tp_list:
        for pp in pp_list:
            if is_moe:
                for dp in dp_list:
                    for moe_tp in moe_tp_list:
                        for moe_ep in moe_ep_list:
                            if dp * tp * pp in num_gpu_list and dp * tp == moe_tp * moe_ep:  # check num gpu and width
                                # backend specific filters
                                # trtllm
                                if (
                                    backend == common.BackendName.trtllm and dp > 1 and tp > 1
                                ):  # trtllm as trtllm don't supports attn tp > 1
                                    continue
                                # sglang
                                elif backend == common.BackendName.sglang:
                                    if (enable_wideep and moe_tp > 1) or (
                                        not enable_wideep and moe_ep > 1
                                    ):  # wideep only has ep
                                        continue
                                elif backend == common.BackendName.vllm:
                                    pass  # TODO
                                parallel_config_list.append([tp, pp, dp, moe_tp, moe_ep])
            else:
                if tp * pp in num_gpu_list:
                    parallel_config_list.append([tp, pp, 1, 1, 1])

    for parallel_config in parallel_config_list:
        tp, pp, dp, moe_tp, moe_ep = parallel_config
        logger.info(f"Enumerated parallel config: tp={tp}, pp={pp}, dp={dp}, moe_tp={moe_tp}, moe_ep={moe_ep}")

    return parallel_config_list


def enumerate_ttft_tpot_constraints(
    osl: int,
    request_latency: float,
    ttft: float | None = None,
) -> list[tuple[float, float]]:
    """
    Enumerate ttft and tpot constraints if given request latency.
    """
    assert osl > 1
    if ttft is None:
        ttft = request_latency * 0.95

    # typical values for ttft
    base_values = [300, 400, 500, 600, 800, 1000, 1200, 1400, 1600, 2000, 3000, 5000, 8000]
    base_min, base_max = base_values[0], base_values[-1]

    # values based on request_latency, only supplement values outside the base range
    interval_values = [request_latency * p for p in [0.1, 0.2, 0.3, 0.5, 0.7]]
    extra_values = [v for v in interval_values if v < base_min or v > base_max]

    ttft_set = set(base_values + extra_values)
    ttft_set.add(ttft)
    ttft_list = sorted([t for t in ttft_set if t < request_latency])
    return [(t, (request_latency - t) / (osl - 1)) for t in ttft_list]


def safe_mkdir(target_path: str, exist_ok: bool = True) -> Path:
    """
    Safely create a directory with path validation, sanitization, and security checks.

    This function validates the parent directory for security, sanitizes the target
    directory name, and creates the directory using pathlib.

    Args:
        target_path: The target directory path to create
        exist_ok: If True, don't raise an exception if the directory already exists

    Returns:
        Path: The resolved absolute path of the created directory

    Raises:
        ValueError: If the path is invalid or outside allowed directories
        OSError: If directory creation fails
    """

    def _sanitize_path_component(component: str) -> str:
        """
        Sanitize a path component (closure function).
        """
        if not component:
            return "unknown"

        # Replace dangerous characters with underscores
        sanitized = re.sub(r"[^\w\-_.]", "_", str(component))

        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip(". ")

        # Ensure it's not empty after sanitization
        if not sanitized:
            return "unknown"

        # Limit length to prevent extremely long filenames
        return sanitized[:100]

    if not target_path:
        raise ValueError("Target path cannot be empty")

    try:
        # Parse the target path
        target = Path(target_path)

        # Get parent directory and target directory name
        if target.is_absolute():
            # For absolute paths, validate the entire path
            parent_dir = target.parent
            dir_name = target.name
        else:
            # For relative paths, validate from current directory
            parent_dir = Path.cwd()
            # Split the relative path and sanitize each component
            parts = target.parts
            sanitized_parts = [_sanitize_path_component(part) for part in parts]

            # Build the final path
            final_target = parent_dir
            for part in sanitized_parts:
                final_target = final_target / part

            return safe_mkdir(str(final_target), exist_ok)

        # Validate parent directory security
        resolved_parent = parent_dir.resolve()

        # Security check: ensure no null bytes
        if "\x00" in str(resolved_parent):
            raise ValueError("Path contains null byte")

        # Check if the parent path is within allowed locations
        current_dir = Path.cwd().resolve()
        allowed_prefixes = [
            current_dir,
            Path.home(),
            Path("/tmp"),
            Path("/workspace"),
            Path("/var/tmp"),
            Path(tempfile.gettempdir()).resolve(),
        ]

        # Verify the parent path is under an allowed prefix
        is_allowed = any(
            resolved_parent == prefix or resolved_parent.is_relative_to(prefix) for prefix in allowed_prefixes
        )

        if not is_allowed:
            raise ValueError(f"Path is outside allowed locations: {resolved_parent}")

        # Sanitize the target directory name and create final path
        sanitized_name = _sanitize_path_component(dir_name)
        final_path = resolved_parent / sanitized_name

        # Create the directory using pathlib
        final_path.mkdir(parents=True, exist_ok=exist_ok)

        return final_path

    except (OSError, ValueError) as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Failed to create directory: {e}") from e


class HuggingFaceDownloadError(Exception):
    """
    Exception raised when a HuggingFace config.json file cannot be downloaded.
    """

    pass


def _download_hf_config(hf_id: str) -> dict:
    """
    Download a HuggingFace config.json file from the HuggingFace API.

    Args:
        hf_id: HuggingFace model ID

    Returns:
        dict: HuggingFace config.json dictionary

    Raises:
        HuggingFaceDownloadError: If the HuggingFace API returns an error
    """
    url = f"https://huggingface.co/{hf_id}/raw/main/config.json"

    # Load token from ~/.cache/huggingface/token, if available
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    hf_token = None
    if token_path.exists():
        with open(token_path) as f:
            hf_token = f.read().strip()
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        # Provide detailed error for any HTTP error code
        raise HuggingFaceDownloadError(
            f"Failed to download {hf_id}'s config.json from HuggingFace: "
            f"HuggingFace returned HTTP error {e.code}: {e.reason}. "
            f"URL: {url}. Check your authentication token in {token_path} if using a gated model."
        ) from e
    except Exception as e:
        raise HuggingFaceDownloadError(f"Failed to download {hf_id}'s config.json from HuggingFace: {e}") from e


def _parse_nemotron_block_configs(block_configs: list[dict]) -> list[BlockConfig]:
    """
    Parse Nemotron's block_configs into a list of BlockConfig objects.
    Groups consecutive blocks with the same configuration together.

    Args:
        block_configs: List of block configuration dictionaries from HuggingFace config

    Returns:
        list[BlockConfig]: Grouped block configurations
    """
    if not block_configs:
        return None

    grouped_configs = []
    current_config = None
    current_count = 0

    for block in block_configs:
        attn = block.get("attention", {})
        ffn = block.get("ffn", {})

        n_heads_in_group = attn.get("n_heads_in_group")
        attn_no_op = attn.get("no_op", False)
        ffn_mult = ffn.get("ffn_mult", 3.5)
        ffn_no_op = ffn.get("no_op", False)

        # Create a tuple to compare configurations
        config_tuple = (n_heads_in_group, attn_no_op, ffn_mult, ffn_no_op)

        if current_config == config_tuple:
            current_count += 1
        else:
            if current_config is not None:
                grouped_configs.append(
                    BlockConfig(
                        attn_n_heads_in_group=current_config[0],
                        attn_no_op=current_config[1],
                        ffn_ffn_mult=current_config[2],
                        ffn_no_op=current_config[3],
                        num_inst=current_count,
                    )
                )
            current_config = config_tuple
            current_count = 1

    # Add the last group
    if current_config is not None:
        grouped_configs.append(
            BlockConfig(
                attn_n_heads_in_group=current_config[0],
                attn_no_op=current_config[1],
                ffn_ffn_mult=current_config[2],
                ffn_no_op=current_config[3],
                num_inst=current_count,
            )
        )

    return grouped_configs if grouped_configs else None


def _parse_hf_config_json(config: dict) -> list:
    """
    Convert a HuggingFace config.json dictionary into a list of model configuration parameters:
    [architecture, l, n, n_kv, d, hidden_size, inter_size, vocab, context, topk,
    num_experts, moe_inter_size, extra_params]

    Args:
        config: HuggingFace config.json dictionary

    Returns:
        list: Model configuration parameters

    Raises:
        ValueError: If a required field is missing from the config or the architecture is not supported
    """
    architecture = config["architectures"][0]
    if architecture not in ARCHITECTURE_TO_MODEL_FAMILY:
        raise ValueError(
            f"The model's architecture {architecture} is not supported. "
            f"Supported architectures: {', '.join(ARCHITECTURE_TO_MODEL_FAMILY.keys())}"
        )

    layers = config["num_hidden_layers"]
    hidden_size = config["hidden_size"]
    n = config["num_attention_heads"]
    vocab = config["vocab_size"]
    context = config["max_position_embeddings"]

    # Handle nullable fields (e.g., Nemotron has null for these)
    n_kv = config.get("num_key_value_heads") or 0
    inter_size = config.get("intermediate_size") or 0
    d = config.get("head_dim") or config.get("attention_head_dim") or (hidden_size // n if n > 0 else 0)

    # MoE parameters
    topk = config.get("num_experts_per_tok", 0)
    num_experts = config.get("num_local_experts") or config.get("n_routed_experts") or config.get("num_experts", 0)
    moe_inter_size = config.get("moe_intermediate_size", 0)

    # Handle NemotronH-specific configuration (only fields unique to NemotronH)
    extra_params = None
    if architecture == "NemotronHForCausalLM":
        extra_params = common.NemotronHConfig(
            hybrid_override_pattern=config["hybrid_override_pattern"],
            mamba_num_heads=config["mamba_num_heads"],
            mamba_head_dim=config["mamba_head_dim"],
            ssm_state_size=config["ssm_state_size"],
            conv_kernel=config["conv_kernel"],
            n_groups=config["n_groups"],
            chunk_size=config["chunk_size"],
            # Optional: 0 for non-MoE NemotronH models (e.g., Nemotron-H-56B)
            moe_shared_expert_intermediate_size=config.get("moe_shared_expert_intermediate_size", 0),
        )
        logger.info(
            f"NemotronH hybrid config: pattern={extra_params.hybrid_override_pattern}, "
            f"mamba_heads={extra_params.mamba_num_heads}"
        )
    elif architecture == "DeciLMForCausalLM":
        if "block_configs" in config:
            extra_params = _parse_nemotron_block_configs(config["block_configs"])

    logger.info(
        f"Model architecture: architecture={architecture}, layers={layers}, n={n}, n_kv={n_kv}, d={d}, "
        f"hidden_size={hidden_size}, inter_size={inter_size}, vocab={vocab}, context={context}, "
        f"topk={topk}, num_experts={num_experts}, moe_inter_size={moe_inter_size}, "
        f"extra_params={'present' if extra_params else 'None'}"
    )
    return [
        architecture,
        layers,
        n,
        n_kv,
        d,
        hidden_size,
        inter_size,
        vocab,
        context,
        topk,
        num_experts,
        moe_inter_size,
        extra_params,
    ]


def _get_model_config_path():
    """
    Get the model config path
    """
    return pkg_resources.files("aiconfigurator") / "model_configs"


def _load_pre_downloaded_hf_config(hf_id: str) -> dict:
    config_path = _get_model_config_path() / f"{hf_id.replace('/', '--')}_config.json"
    if not config_path.exists():
        raise ValueError(f"HuggingFace model {hf_id} is not cached in model_configs directory.")
    return _load_json_with_infinity(config_path)


def _load_local_config(path: str) -> dict:
    """Load config.json from a local directory path."""
    config_path = Path(path) / "config.json"
    if not config_path.exists():
        raise ValueError(f"config.json not found at {config_path}")
    return _load_json_with_infinity(config_path)


def get_model_config_from_model_path(model_path: str) -> list:
    """
    Get model configuration from model path.

    The model_path can be:
    1. A HuggingFace model path (e.g., "Qwen/Qwen3-32B")
    2. A local directory path containing config.json

    Args:
        model_path: HuggingFace model path or local directory path

    Returns:
        list: Model configuration parameters
              [architecture, layers, n, n_kv, d, hidden_size, inter_size, vocab, context,
               topk, num_experts, moe_inter_size, extra_params]

    Raises:
        ValueError: If the model config cannot be found
        HuggingFaceDownloadError: If fetching from HuggingFace fails
    """
    import os

    # Check if it's a local path
    if os.path.isdir(model_path):
        config = _load_local_config(model_path)
        return _parse_hf_config_json(config)

    # Otherwise treat as HuggingFace path
    if model_path in DefaultHFModels:
        config = _load_pre_downloaded_hf_config(model_path)
    else:
        config = _download_hf_config(model_path)
    return _parse_hf_config_json(config)

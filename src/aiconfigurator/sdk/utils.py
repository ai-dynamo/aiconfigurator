# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.resources as pkg_resources
import json
import logging
import re
import tempfile
import urllib.request
from pathlib import Path

from aiconfigurator.sdk.common import ARCHITECTURE_TO_MODEL_FAMILY, SupportedHFModels

logger = logging.getLogger(__name__)


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


def _download_hf_config(hf_id: str) -> dict:
    """
    Download a HuggingFace config.json file from the HuggingFace API.

    Args:
        hf_id: HuggingFace model ID

    Returns:
        dict: HuggingFace config.json dictionary

    Raises:
        RuntimeError: If the HuggingFace API returns an error
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
        raise RuntimeError(
            f"HuggingFace returned HTTP error {e.code}: {e.reason}. "
            f"URL: {url}. Check your authentication token in {token_path} if using a gated model."
        ) from e


def _parse_hf_config_json(config: dict) -> list:
    """
    Convert a HuggingFace config.json dictionary into a list of model configuration parameters:
    [model_family, l, n, n_kv, d, hidden_size, inter_size, vocab, context, topk,
    num_experts, moe_inter_size, extra_params]

    Args:
        config: HuggingFace config.json dictionary

    Returns:
        list: Model configuration parameters

    Raises:
        KeyError: If a required field is missing from the config
    """
    model_family = ARCHITECTURE_TO_MODEL_FAMILY[config["architectures"][0]]
    layers = config["num_hidden_layers"]
    n_kv = config["num_key_value_heads"]
    hidden_size = config["hidden_size"]
    n = config["num_attention_heads"]
    inter_size = config["intermediate_size"]
    d = config.get("head_dim", hidden_size // n)
    vocab = config["vocab_size"]
    context = config["max_position_embeddings"]
    topk = config.get("num_experts_per_tok", 0)
    num_experts = config.get("num_local_experts") or config.get("n_routed_experts") or config.get("num_experts", 0)
    moe_inter_size = config.get("moe_intermediate_size", 0)
    return [
        model_family,
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
        None,
    ]


def get_model_config_path():
    """
    Get the model config path
    """
    return pkg_resources.files("aiconfigurator") / "model_configs"


def get_model_config_from_hf_id(hf_id: str) -> list:
    """
    Get model configuration from HuggingFace ID.
    First try to download the config from HuggingFace, if failed, use the config saved in model_configs directory.
    """
    try:
        config = _download_hf_config(hf_id)
        logger.info(f"Fetched config.json for {hf_id} from HuggingFace.")
        return _parse_hf_config_json(config)
    except Exception as e:
        logger.info(
            f"Failed to download from HuggingFace using user's HF token saved in ~/.cache/huggingface/token, "
            f"trying to use cached config: {e}"
        )
        if hf_id not in SupportedHFModels:
            raise ValueError(f"HuggingFace model {hf_id} is not cached in model_configs directory.") from e
        config_name = SupportedHFModels[hf_id]
        with open(get_model_config_path() / f"{config_name}_config.json") as f:
            return _parse_hf_config_json(json.load(f))

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate rendered extra_engine_args YAML into --trtllm.* dynamic CLI flags.

When ``use_dynamo_generator=True`` (profiler path), the rendering engine calls
:func:`yaml_to_dynamic_flags` to convert the version-specific extra_engine_args
YAML into flat ``--trtllm.<key>.<subkey> <value>`` flags accepted by dynamo >= 1.1.0.
"""

from __future__ import annotations

import logging
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Top-level YAML keys that already have direct CLI flag equivalents in cli_args.j2.
DEFAULT_SKIP_KEYS: frozenset[str] = frozenset(
    {
        "backend",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "moe_expert_parallel_size",
        "enable_attention_dp",
        "max_batch_size",
        "max_num_tokens",
        "max_seq_len",
    }
)

# Nested keys (as tuples of path components) that already have direct CLI flags.
DEFAULT_SKIP_NESTED_KEYS: frozenset[tuple[str, ...]] = frozenset(
    {
        ("kv_cache_config", "free_gpu_memory_fraction"),
    }
)


def _format_value(value: Any) -> str:
    """Serialize a scalar value for CLI consumption."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _walk(
    data: dict[str, Any],
    prefix: tuple[str, ...],
    skip_keys: set[str],
    skip_nested: set[tuple[str, ...]],
    out: list[str],
) -> None:
    """Recursively walk *data* and append ``--trtllm.key value`` pairs to *out*."""
    for key, value in data.items():
        path = prefix + (key,)

        if len(path) == 1 and key in skip_keys:
            continue

        if path in skip_nested:
            continue

        if value is None or value == "":
            continue

        if isinstance(value, dict):
            _walk(value, path, skip_keys, skip_nested, out)
            continue

        if isinstance(value, list):
            logger.debug("Skipping list value at %s", ".".join(path))
            continue

        flag = "--trtllm." + ".".join(path)
        out.append(flag)
        out.append(_format_value(value))


def yaml_to_dynamic_flags(
    yaml_content: str,
    skip_keys: set[str] | None = None,
    skip_nested_keys: set[tuple[str, ...]] | None = None,
) -> list[str]:
    """Convert rendered extra_engine_args YAML into ``--trtllm.*`` CLI flags.

    Args:
        yaml_content: Rendered YAML string (from extra_engine_args template).
        skip_keys: Top-level keys to exclude. Defaults to :data:`DEFAULT_SKIP_KEYS`.
        skip_nested_keys: Nested key paths to exclude.
            Defaults to :data:`DEFAULT_SKIP_NESTED_KEYS`.

    Returns:
        Flat list of alternating flag / value strings, e.g.
        ``["--trtllm.kv_cache_config.dtype", "auto", ...]``.
    """
    if skip_keys is None:
        skip_keys = DEFAULT_SKIP_KEYS
    if skip_nested_keys is None:
        skip_nested_keys = DEFAULT_SKIP_NESTED_KEYS

    data = yaml.safe_load(yaml_content)
    if not isinstance(data, dict):
        return []

    out: list[str] = []
    _walk(data, (), skip_keys, skip_nested_keys, out)
    return out

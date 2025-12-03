# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Memory calculations for profiling.

This module provides functions to estimate memory usage and maximum batch sizes.
"""

import aiconfigurator.sdk.config
import aiconfigurator.sdk.models


def get_max_batch_size(
    database,
    backend,
    model_name: str,
    isl: int,
    osl: int,
    **model_config_kwargs,
) -> int:
    """
    Estimate the largest batch size that fits in GPU memory.

    Uses binary search to find the maximum batch size that fits within
    the GPU memory capacity.

    Args:
        database: Performance database instance
        backend: Backend instance
        model_name: Model name (e.g., "QWEN3_32B")
        isl: Input sequence length
        osl: Output sequence length
        **model_config_kwargs: Model config kwargs (e.g., tp_size)

    Returns:
        int: Maximum batch size that fits in GPU memory
    """
    # Create model instance
    model_config = aiconfigurator.sdk.config.ModelConfig(**model_config_kwargs)
    model = aiconfigurator.sdk.models.get_model(model_name, model_config, backend)

    def get_mem_usage(bs: int):
        """Get memory usage for a given batch size."""
        return backend._get_memory_usage(model, database, bs, 1, isl, osl)["total"]

    max_memory_gb = database.system_spec["gpu"]["mem_capacity"] / (1024**3)

    bs = 1
    if get_mem_usage(bs) > max_memory_gb:
        # Model doesn't fit on GPU with given config
        return 0

    # Step 1: Find upper bound on batch size (exponential growth)
    while get_mem_usage(bs) < max_memory_gb:
        bs *= 2

    # We know bs // 2 fits but bs doesn't
    min_bs = bs // 2
    max_bs = bs

    # Step 2: Binary search for exact max batch size
    while min_bs < max_bs:
        test_bs = (min_bs + max_bs) // 2
        if get_mem_usage(test_bs) < max_memory_gb:
            min_bs = test_bs + 1
        else:
            max_bs = test_bs

    return min_bs - 1

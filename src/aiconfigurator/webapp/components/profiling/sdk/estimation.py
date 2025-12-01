# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Performance estimation using aiconfigurator SDK.

This module provides simplified wrappers around the aiconfigurator SDK
for performance estimation use cases.
"""

from typing import Any

import aiconfigurator.sdk.backends.factory
import aiconfigurator.sdk.config
import aiconfigurator.sdk.inference_session
import aiconfigurator.sdk.models


def estimate_perf(
    database,
    backend,
    model_name: str,
    isl: int,
    osl: int,
    batch_size: int,
    mode: str = "full",
    **model_config_kwargs,
) -> dict[str, Any]:
    """
    Estimate performance using aiconfigurator SDK.

    Args:
        database: Performance database instance
        backend: Backend instance
        model_name: Model name (e.g., "QWEN3_32B")
        isl: Input sequence length
        osl: Output sequence length
        batch_size: Batch size
        mode: Estimation mode - "full", "prefill", or "decode"
        **model_config_kwargs: Model config kwargs (e.g., tp_size)

    Returns:
        dict: Performance metrics from aiconfigurator
    """
    # Map user-friendly mode names to SDK mode names
    mode_to_sdk_mode = {
        "full": "static",
        "prefill": "static_ctx",
        "decode": "static_gen",
    }
    if mode not in mode_to_sdk_mode:
        raise ValueError(f"Invalid mode: {mode}. Must be one of {list(mode_to_sdk_mode.keys())}.")

    # Create model and session
    model_config = aiconfigurator.sdk.config.ModelConfig(**model_config_kwargs)
    model = aiconfigurator.sdk.models.get_model(model_name, model_config, backend)

    runtime_config = aiconfigurator.sdk.config.RuntimeConfig(
        batch_size=batch_size,
        beam_width=1,
        isl=isl,
        osl=osl,
    )

    session = aiconfigurator.sdk.inference_session.InferenceSession(model, database, backend)
    summary = session.run_static(mode=mode_to_sdk_mode[mode], runtime_config=runtime_config, stride=32)
    summary_df = summary.get_summary_df()

    # Convert DataFrame to dict (single row)
    return summary_df.to_dict(orient="records")[0]


def estimate_prefill_perf(
    database,
    backend,
    model_name: str,
    isl: int,
    **model_config_kwargs,
) -> dict[str, Any]:
    """
    Estimate prefill performance.

    Args:
        database: Performance database instance
        backend: Backend instance
        model_name: Model name (e.g., "QWEN3_32B")
        isl: Input sequence length
        **model_config_kwargs: Model config kwargs (e.g., tp_size)

    Returns:
        dict: Performance metrics with 'context_latency' (TTFT in ms)
    """
    return estimate_perf(
        database,
        backend,
        model_name,
        isl,
        5,  # small osl for prefill-only
        1,  # concurrency = 1
        mode="prefill",
        **model_config_kwargs,
    )

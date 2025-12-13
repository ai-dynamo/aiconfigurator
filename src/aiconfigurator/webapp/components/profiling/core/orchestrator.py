# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Main profiling orchestrator.

This module provides the primary entry point for generating profiling plots.
It coordinates all profiling activities and returns results for the UI.
"""

import logging
import traceback

import aiconfigurator.sdk.backends.factory
import aiconfigurator.sdk.perf_database
from aiconfigurator.webapp.components.profiling.sdk import (
    generate_gpu_configurations,
    validate_inputs,
)

from .data_preparation import (
    prepare_cost_table_data,
    prepare_decode_table_data,
    prepare_prefill_table_data,
)
from .data_serializer import serialize_profiling_data
from .profiling import profile_decode_performance, profile_prefill_performance

logger = logging.getLogger(__name__)


def format_status_message(profile_num_gpus, prefill_results):
    """
    Format success status message with profiling summary.

    Args:
        profile_num_gpus: List of GPU counts profiled
        prefill_results: Prefill profiling results

    Returns:
        Formatted status message string
    """
    _, prefill_ttft, _ = prefill_results
    prefill_num_gpus, _, _ = prefill_results

    best_prefill_idx = prefill_ttft.index(min(prefill_ttft))
    return (
        f"✅ Plots generated successfully!\n"
        f"📊 Profiled {len(profile_num_gpus)} GPU configurations: {profile_num_gpus}\n"
        f"⚡ Best prefill: {min(prefill_ttft):.1f}ms TTFT at {prefill_num_gpus[best_prefill_idx]} GPUs"
    )


def generate_profiling_plots(
    model_name: str,
    system: str,
    backend: str,
    version: str,
    min_num_gpus_per_engine: int,
    max_num_gpus_per_engine: int,
    isl: int,
    osl: int,
    ttft: float,
    itl: float,
    allow_confirm_datapoint: bool = False,
):
    """
    Generate performance plots using AI Configurator estimation.

    This is the PRIMARY ENTRY POINT for profiling. It orchestrates:
    1. Estimating prefill performance (TTFT) across different GPU counts
    2. Estimating decode performance (ITL) at various concurrency levels
    3. Computing GPU hours for cost analysis (frontend handles cost calculation)

    Args:
        model_name: Model name (e.g., "QWEN3_32B")
        system: System name (e.g., "h200_sxm")
        backend: Backend name (e.g., "trtllm")
        version: Backend version (e.g., "0.20.0")
        min_num_gpus_per_engine: Minimum TP size to profile
        max_num_gpus_per_engine: Maximum TP size to profile
        isl: Input sequence length
        osl: Output sequence length
        ttft: Target TTFT in milliseconds (for visualization)
        itl: Target ITL in milliseconds (for visualization)
        allow_confirm_datapoint: If True, adds "Select" button for user confirmation

    Returns:
        Tuple of (json_data_string, status_message)
    """

    try:
        # Validate inputs
        is_valid, error_msg = validate_inputs(model_name, system, backend, version)
        if not is_valid:
            return ("", error_msg)

        # Load database and backend
        logger.info("Loading aiconfigurator database. This might take a few seconds...")
        database = aiconfigurator.sdk.perf_database.get_database(
            system=system,
            backend=backend,
            version=version,
        )
        if not database:
            raise ValueError(f"Database not found for system: {system}, backend: {backend}, version: {version}")
        logger.info("aiconfigurator database loaded.")

        backend_instance = aiconfigurator.sdk.backends.factory.get_backend(backend)

        # Generate GPU configurations to profile
        profile_num_gpus = generate_gpu_configurations(min_num_gpus_per_engine, max_num_gpus_per_engine)

        if not profile_num_gpus:
            raise ValueError("No valid GPU configurations to profile")

        # Profile prefill performance
        prefill_results = profile_prefill_performance(database, backend_instance, model_name, profile_num_gpus, isl)

        # Profile decode performance
        decode_results = profile_decode_performance(database, backend_instance, model_name, profile_num_gpus, isl, osl)

        # Prepare table data
        prefill_table_data = prepare_prefill_table_data(prefill_results, model_name, system, backend, version, isl, osl)
        decode_table_data = prepare_decode_table_data(decode_results, model_name, system, backend, version, isl, osl)
        cost_table_data = prepare_cost_table_data(
            isl, osl, prefill_results, decode_results, model_name, system, backend, version
        )

        # Serialize all data to JSON for frontend
        json_data = serialize_profiling_data(
            prefill_results,
            decode_results,
            prefill_table_data,
            decode_table_data,
            cost_table_data,
            isl,
            osl,
            ttft,
            itl,
            allow_confirm_datapoint,
        )

        # Generate success status message
        status_msg = format_status_message(profile_num_gpus, prefill_results)

        return (json_data, status_msg)

    except Exception as e:
        # Log the full error with stack trace to console
        logger.exception("Error generating profiling plots")

        # Format detailed error message for web UI with stack trace
        tb = traceback.format_exc()

        # Get the last few lines of traceback for context
        tb_lines = tb.strip().split("\n")

        # Find the actual error location (last "File" line before the error)
        error_location = "Unknown location"
        for i in range(len(tb_lines) - 1, -1, -1):
            if tb_lines[i].strip().startswith('File "'):
                error_location = tb_lines[i].strip()
                break

        error_msg = (
            f"❌ Error during profiling:\n\n"
            f"Error Type: {type(e).__name__}\n"
            f"Error Message: {e!s}\n"
            f"Location: {error_location}\n\n"
            f"Full Stack Trace:\n{tb}"
        )

        return ("", error_msg)

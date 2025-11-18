# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Profiling orchestration logic for the webapp.

This module coordinates performance profiling, plot generation, and table building.
Uses aiconfigurator SDK directly for performance estimation.
"""

import logging
import math
import traceback
from typing import Any

import numpy as np
import yaml

import aiconfigurator.sdk.backends.factory
import aiconfigurator.sdk.config
import aiconfigurator.sdk.inference_session
import aiconfigurator.sdk.models
import aiconfigurator.sdk.perf_database
from aiconfigurator.webapp.components.profiling.constants import (
    COST_TABLE_HEADERS,
    DECODE_TABLE_HEADERS,
    PREFILL_TABLE_HEADERS,
)
from aiconfigurator.webapp.components.profiling.create_results_tabs import (
    build_table_html,
    get_empty_tables,
)
from aiconfigurator.webapp.components.profiling.plot import (
    _compute_parato,
    plot_cost_sla_interactive,
    plot_decode_performance_interactive,
    plot_prefill_performance_interactive,
)

logger = logging.getLogger(__name__)

# Constants
DEFAULT_DECODE_INTERPOLATION_GRANULARITY = 6
DECODE_MAX_CONCURRENCY = 1024


# TODO: Re-write this function to reflect the real dynamo config
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


# ============================================================================
# SDK Helper Functions
# These functions provide a simplified interface to the aiconfigurator SDK
# for the profiling use case.
# ============================================================================


def _estimate_perf(
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


def _estimate_prefill_perf(
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
    return _estimate_perf(
        database,
        backend,
        model_name,
        isl,
        5,  # small osl for prefill-only
        1,  # concurrency = 1
        mode="prefill",
        **model_config_kwargs,
    )


def _get_max_batch_size(
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


def profile_prefill_performance(database, backend, model_name, profile_num_gpus, isl):
    """
    Profile prefill performance across different GPU counts.

    Args:
        database: Performance database instance
        backend: Backend instance
        model_name: Model name
        profile_num_gpus: List of GPU counts to profile
        isl: Input sequence length

    Returns:
        Tuple of (num_gpus_list, ttft_list, thpt_per_gpu_list)
    """
    prefill_num_gpus = []
    prefill_ttft = []
    prefill_thpt_per_gpu = []

    for num_gpus in profile_num_gpus:
        # Estimate prefill performance
        perf_dict = _estimate_prefill_perf(
            database,
            backend,
            model_name,
            isl,
            tp_size=num_gpus,
        )
        ttft_val = perf_dict["context_latency"]
        # Calculate throughput: tokens/second/GPU
        thpt_val = isl / ttft_val * 1000 / num_gpus

        prefill_num_gpus.append(num_gpus)
        prefill_ttft.append(ttft_val)
        prefill_thpt_per_gpu.append(thpt_val)

    return (prefill_num_gpus, prefill_ttft, prefill_thpt_per_gpu)


def profile_decode_performance(
    database,
    backend,
    model_name,
    profile_num_gpus,
    isl,
    osl,
    decode_interpolation_granularity=DEFAULT_DECODE_INTERPOLATION_GRANULARITY,
):
    """
    Profile decode performance at various concurrency levels.

    Args:
        database: Performance database instance
        backend: Backend instance
        model_name: Model name
        profile_num_gpus: List of GPU counts to profile
        isl: Input sequence length
        osl: Output sequence length
        decode_interpolation_granularity: Number of concurrency points to sample

    Returns:
        List of tuples (num_gpus, itl_list, thpt_per_gpu_list, batch_size_list)
    """
    decode_results = []
    # For dense models (not MoE), attention_dp_size = 1
    attention_dp_size = 1

    for num_gpus in profile_num_gpus:
        # Get maximum batch size for this configuration
        max_concurrency = _get_max_batch_size(database, backend, model_name, isl, osl, tp_size=num_gpus)

        # Determine request sweep range
        sweep_num_request = get_num_request_range(
            attention_dp_size,
            max_concurrency,
            decode_interpolation_granularity,
        )

        engine_decode_itl = []
        engine_decode_thpt_per_gpu = []
        engine_batch_sizes = []

        for num_request in sweep_num_request:
            # Estimate decode performance
            perf_dict = _estimate_perf(
                database,
                backend,
                model_name,
                isl,
                osl,
                num_request,
                mode="decode",
                tp_size=num_gpus,
            )

            itl_val = perf_dict["tpot"]
            thpt_val = perf_dict["tokens/s/gpu"]

            engine_decode_itl.append(itl_val)
            engine_decode_thpt_per_gpu.append(thpt_val)
            engine_batch_sizes.append(num_request)

        # Store results for this GPU configuration
        if engine_decode_itl:
            decode_results.append((num_gpus, engine_decode_itl, engine_decode_thpt_per_gpu, engine_batch_sizes))

    return decode_results


def format_status_message(profile_num_gpus, prefill_results, gpu_cost_per_hour):
    """
    Format success status message with profiling summary.

    Args:
        profile_num_gpus: List of GPU counts profiled
        prefill_results: Prefill profiling results
        gpu_cost_per_hour: Cost per GPU per hour

    Returns:
        Formatted status message string
    """
    _, prefill_ttft, _ = prefill_results
    prefill_num_gpus, _, _ = prefill_results

    best_prefill_idx = prefill_ttft.index(min(prefill_ttft))
    return (
        f"✅ Plots generated successfully!\n"
        f"📊 Profiled {len(profile_num_gpus)} GPU configurations: {profile_num_gpus}\n"
        f"⚡ Best prefill: {min(prefill_ttft):.1f}ms TTFT at {prefill_num_gpus[best_prefill_idx]} GPUs\n"
        + "💰 GPU Cost: "
        + (f"${gpu_cost_per_hour:.2f}/hour" if gpu_cost_per_hour else "N/A")
    )


def prepare_prefill_table_data(
    prefill_results,
    model_name: str,
    system: str,
    backend: str,
    version: str,
    isl: int,
    osl: int,
):
    """
    Prepare table data for prefill performance.

    Args:
        prefill_results: Tuple of (num_gpus_list, ttft_list, thpt_per_gpu_list)
        model_name: Model name
        system: System name
        backend: Backend name
        version: Backend version
        isl: Input sequence length
        osl: Output sequence length

    Returns:
        List of rows for the table, each row includes config YAML
    """
    num_gpus_list, ttft_list, thpt_per_gpu_list = prefill_results
    rows = []
    for num_gpus, ttft, thpt in zip(num_gpus_list, ttft_list, thpt_per_gpu_list):
        config_yaml = generate_config_yaml(
            model_name=model_name,
            system=system,
            backend=backend,
            version=version,
            isl=isl,
            osl=osl,
            num_gpus=num_gpus,
            batch_size=1,  # Prefill uses batch size 1
        )
        rows.append([num_gpus, round(ttft, 3), round(thpt, 3), config_yaml])
    return rows


def prepare_decode_table_data(
    decode_results,
    model_name: str,
    system: str,
    backend: str,
    version: str,
    isl: int,
    osl: int,
):
    """
    Prepare table data for decode performance.

    Args:
        decode_results: List of tuples (num_gpus, itl_list, thpt_list)
        model_name: Model name
        system: System name
        backend: Backend name
        version: Backend version
        isl: Input sequence length
        osl: Output sequence length

    Returns:
        List of rows for the table, each row includes config YAML
    """
    table_data = []
    for decode_result in decode_results:
        num_gpus = decode_result[0]
        itl_list = decode_result[1]
        thpt_list = decode_result[2]
        batch_size_list = decode_result[3]

        for itl, thpt, batch_size in zip(itl_list, thpt_list, batch_size_list):
            config_yaml = generate_config_yaml(
                model_name=model_name,
                system=system,
                backend=backend,
                version=version,
                isl=isl,
                osl=osl,
                num_gpus=num_gpus,
                batch_size=batch_size,
            )
            table_data.append([num_gpus, round(itl, 3), round(thpt, 3), config_yaml])
    return table_data


def prepare_cost_table_data(
    isl,
    osl,
    prefill_results,
    decode_results,
    gpu_cost_per_hour,
    model_name: str,
    system: str,
    backend: str,
    version: str,
):
    """
    Prepare table data for cost analysis.

    Args:
        isl: Input sequence length
        osl: Output sequence length
        prefill_results: Tuple of (num_gpus, ttft, thpt_per_gpu) for prefill
        decode_results: List of tuples (num_gpus, itl_list, thpt_per_gpu_list) for decode
        gpu_cost_per_hour: Cost per GPU per hour in dollars
        model_name: Model name
        system: System name
        backend: Backend name
        version: Backend version

    Returns:
        List of rows for the table, each row includes config YAML
    """
    # Compute Pareto fronts with GPU tracking
    num_gpus_list, ttft_list, thpt_list = prefill_results

    # Track which GPU configuration corresponds to each pareto point
    p_ttft, p_thpt = _compute_parato(ttft_list, thpt_list)
    p_gpus = []
    for ttft_val, thpt_val in zip(p_ttft, p_thpt):
        for i, (orig_ttft, orig_thpt, orig_gpus) in enumerate(zip(ttft_list, thpt_list, num_gpus_list)):
            if abs(orig_ttft - ttft_val) < 0.001 and abs(orig_thpt - thpt_val) < 0.001:
                p_gpus.append(orig_gpus)
                break

    _d_itl, _d_thpt, _d_gpus, _d_batch_sizes = [], [], [], []
    for _d_result in decode_results:
        num_gpus = _d_result[0]
        _d_itl.extend(_d_result[1])
        _d_thpt.extend(_d_result[2])
        batch_sizes = _d_result[3]
        _d_gpus.extend([num_gpus] * len(_d_result[1]))
        _d_batch_sizes.extend(batch_sizes)
    d_itl, d_thpt = _compute_parato(_d_itl, _d_thpt)
    d_gpus = []
    d_batch_sizes = []
    for itl_val, thpt_val in zip(d_itl, d_thpt):
        for i, (orig_itl, orig_thpt, orig_gpus, orig_bs) in enumerate(zip(_d_itl, _d_thpt, _d_gpus, _d_batch_sizes)):
            if abs(orig_itl - itl_val) < 0.001 and abs(orig_thpt - thpt_val) < 0.001:
                d_gpus.append(orig_gpus)
                d_batch_sizes.append(orig_bs)
                break

    # Convert to numpy arrays
    p_ttft = np.array(p_ttft)
    p_thpt = np.array(p_thpt)
    d_itl = np.array(d_itl)
    d_thpt = np.array(d_thpt)

    # Calculate cost data
    # Handle empty strings, None, or 0 values - convert to 0 to use GPU hours
    if gpu_cost_per_hour is None or gpu_cost_per_hour == "" or gpu_cost_per_hour == 0:
        gpu_cost_per_hour = 0.0

    table_data = []
    for p_idx, (_p_ttft, _p_thpt) in enumerate(zip(p_ttft, p_thpt)):
        prefill_cost = isl * 1000 / _p_thpt * gpu_cost_per_hour / 3600
        tokens_per_user_array = 1000 / d_itl
        cost_array = osl * 1000 / d_thpt * gpu_cost_per_hour / 3600 + prefill_cost

        for i in range(len(d_itl)):
            # Use the tracked GPU counts and batch sizes for config generation
            decode_gpus = d_gpus[i] if i < len(d_gpus) else 1
            batch_size = d_batch_sizes[i] if i < len(d_batch_sizes) else 1
            # For cost table, use decode GPU config as the representative config
            config_yaml = generate_config_yaml(
                model_name=model_name,
                system=system,
                backend=backend,
                version=version,
                isl=isl,
                osl=osl,
                num_gpus=decode_gpus,
                batch_size=batch_size,
            )
            table_data.append(
                [
                    round(float(_p_ttft), 3),
                    round(float(_p_thpt), 3),
                    round(float(d_itl[i]), 3),
                    round(float(d_thpt[i]), 3),
                    round(float(tokens_per_user_array[i]), 3),
                    round(float(cost_array[i]), 3),
                    config_yaml,
                ]
            )

    return table_data


def generate_profiling_plots(
    model_name: str,
    system: str,
    backend: str,
    version: str,
    min_num_gpus_per_engine: int,
    max_num_gpus_per_engine: int,
    gpu_cost_per_hour: float,
    isl: int,
    osl: int,
    ttft: float,
    itl: float,
):
    """
    Generate performance plots using AI Configurator estimation.

    This function profiles LLM inference performance by:
    1. Estimating prefill performance (TTFT) across different GPU counts
    2. Estimating decode performance (ITL) at various concurrency levels
    3. Computing cost-vs-SLA tradeoffs based on GPU pricing

    Args:
        model_name: Model name (e.g., "QWEN3_32B")
        system: System name (e.g., "h200_sxm")
        backend: Backend name (e.g., "trtllm")
        version: Backend version (e.g., "0.20.0")
        min_num_gpus_per_engine: Minimum TP size to profile
        max_num_gpus_per_engine: Maximum TP size to profile
        gpu_cost_per_hour: Cost per GPU per hour in dollars
        isl: Input sequence length
        osl: Output sequence length
        ttft: Target TTFT in milliseconds (for visualization)
        itl: Target ITL in milliseconds (for visualization)

    Returns:
        Tuple of (prefill_plot, decode_plot, cost_plot, status_message,
                  prefill_table_html, decode_table_html, cost_table_html)
    """
    empty_prefill_html, empty_decode_html, empty_cost_html = get_empty_tables()

    try:
        # Validate inputs
        is_valid, error_msg = validate_inputs(model_name, system, backend, version)
        if not is_valid:
            return (
                None,
                None,
                None,
                error_msg,
                empty_prefill_html,
                empty_decode_html,
                empty_cost_html,
            )

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
            return (
                None,
                None,
                None,
                "❌ No valid GPU configurations to profile",
                empty_prefill_html,
                empty_decode_html,
                empty_cost_html,
            )

        # Profile prefill performance
        prefill_results = profile_prefill_performance(database, backend_instance, model_name, profile_num_gpus, isl)

        if not prefill_results[0]:
            return (
                None,
                None,
                None,
                "❌ Failed to generate prefill results",
                empty_prefill_html,
                empty_decode_html,
                empty_cost_html,
            )

        # Profile decode performance
        decode_results = profile_decode_performance(database, backend_instance, model_name, profile_num_gpus, isl, osl)

        if not decode_results:
            return (
                None,
                None,
                None,
                "❌ Failed to generate decode results",
                empty_prefill_html,
                empty_decode_html,
                empty_cost_html,
            )

        # Prepare table data (includes configs) BEFORE generating plots
        prefill_table_data = prepare_prefill_table_data(prefill_results, model_name, system, backend, version, isl, osl)
        decode_table_data = prepare_decode_table_data(decode_results, model_name, system, backend, version, isl, osl)
        cost_table_data = prepare_cost_table_data(
            isl, osl, prefill_results, decode_results, gpu_cost_per_hour, model_name, system, backend, version
        )

        # Generate interactive plots with table data (includes configs in customdata)
        prefill_plot = plot_prefill_performance_interactive(prefill_results, ttft, prefill_table_data)
        decode_plot = plot_decode_performance_interactive(decode_results, itl, decode_table_data)
        cost_plot = plot_cost_sla_interactive(
            isl, osl, prefill_results, decode_results, gpu_cost_per_hour, cost_table_data
        )

        # Generate success status message
        status_msg = format_status_message(profile_num_gpus, prefill_results, gpu_cost_per_hour)

        # Build all tables from the prepared data
        prefill_table_html = build_table_html(PREFILL_TABLE_HEADERS, prefill_table_data)
        decode_table_html = build_table_html(DECODE_TABLE_HEADERS, decode_table_data)
        cost_table_html = build_table_html(COST_TABLE_HEADERS, cost_table_data)

        return (
            prefill_plot,
            decode_plot,
            cost_plot,
            status_msg,
            prefill_table_html,
            decode_table_html,
            cost_table_html,
        )

    except Exception as e:
        error_msg = f"❌ Error generating plots:\n{e!s}\n\n{traceback.format_exc()}"
        logger.exception(error_msg)
        return (
            None,
            None,
            None,
            error_msg,
            empty_prefill_html,
            empty_decode_html,
            empty_cost_html,
        )

"""
Core profiling engines for prefill and decode performance.

This module contains the main profiling functions that orchestrate
performance measurements across different configurations.
"""

from aiconfigurator.webapp.components.profiling.sdk import (
    enumerate_moe_configs,
    estimate_perf,
    estimate_prefill_perf,
    get_max_batch_size,
    get_num_request_range,
)

# Constants
DEFAULT_DECODE_INTERPOLATION_GRANULARITY = 6


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
        # For MoE models: tp_size * attention_dp_size = moe_tp_size * moe_ep_size
        # Get valid MoE configurations and pick the first one (typically moe_tp=tp, moe_ep=1)
        attention_dp_size = 1
        moe_configs = enumerate_moe_configs(num_gpus, attention_dp_size)
        if not moe_configs:
            # Fallback if no valid config found (shouldn't happen for power of 2 tp_size)
            moe_tp_size, moe_ep_size = num_gpus, 1
        else:
            # Prefer moe_tp=tp, moe_ep=1 if available, else take first valid config
            moe_tp_size, moe_ep_size = next(
                ((moe_tp, moe_ep) for moe_tp, moe_ep in moe_configs if moe_tp == num_gpus and moe_ep == 1),
                moe_configs[0],
            )

        perf_dict = estimate_prefill_perf(
            database,
            backend,
            model_name,
            isl,
            tp_size=num_gpus,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
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
        # For MoE models: tp_size * attention_dp_size = moe_tp_size * moe_ep_size
        # Get valid MoE configurations and pick the first one
        moe_configs = enumerate_moe_configs(num_gpus, attention_dp_size)
        if not moe_configs:
            moe_tp_size, moe_ep_size = num_gpus, 1
        else:
            moe_tp_size, moe_ep_size = next(
                ((moe_tp, moe_ep) for moe_tp, moe_ep in moe_configs if moe_tp == num_gpus and moe_ep == 1),
                moe_configs[0],
            )

        max_concurrency = get_max_batch_size(
            database, backend, model_name, isl, osl, tp_size=num_gpus, moe_tp_size=moe_tp_size, moe_ep_size=moe_ep_size
        )

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
            # Estimate decode performance using the same MoE config
            perf_dict = estimate_perf(
                database,
                backend,
                model_name,
                isl,
                osl,
                num_request,
                mode="decode",
                tp_size=num_gpus,
                moe_tp_size=moe_tp_size,
                moe_ep_size=moe_ep_size,
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

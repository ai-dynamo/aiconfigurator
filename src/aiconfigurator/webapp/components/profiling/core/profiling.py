# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

        if not perf_dict:
            raise RuntimeError(
                f"estimate_prefill_perf returned empty result for num_gpus={num_gpus}, "
                f"model={model_name}, isl={isl}, moe_tp_size={moe_tp_size}, moe_ep_size={moe_ep_size}"
            )

        if "context_latency" not in perf_dict:
            raise KeyError(
                f"'context_latency' key missing in perf_dict for num_gpus={num_gpus}. "
                f"Available keys: {list(perf_dict.keys())}"
            )

        ttft_val = perf_dict["context_latency"]

        if ttft_val <= 0:
            raise ValueError(
                f"Invalid context_latency={ttft_val} for num_gpus={num_gpus}, model={model_name}, isl={isl}"
            )

        # Calculate throughput: tokens/second/GPU
        thpt_val = isl / ttft_val * 1000 / num_gpus

        prefill_num_gpus.append(num_gpus)
        prefill_ttft.append(ttft_val)
        prefill_thpt_per_gpu.append(thpt_val)

    if not prefill_num_gpus:
        raise RuntimeError(
            f"No prefill results generated for any GPU configuration. profile_num_gpus={profile_num_gpus}"
        )

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
    skipped_configs = []

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

        if max_concurrency is None or max_concurrency <= 0:
            # Skip this configuration - model doesn't fit with these settings
            skip_reason = (
                f"num_gpus={num_gpus} (moe_tp={moe_tp_size}, moe_ep={moe_ep_size}): "
                f"Model doesn't fit in memory with isl={isl}, osl={osl}"
            )
            skipped_configs.append(skip_reason)
            print(f"⚠️  Skipping decode profiling for {skip_reason}")
            continue

        # Determine request sweep range
        sweep_num_request = get_num_request_range(
            attention_dp_size,
            max_concurrency,
            decode_interpolation_granularity,
        )

        if not sweep_num_request:
            raise RuntimeError(
                f"Failed to generate request sweep range for num_gpus={num_gpus}, "
                f"attention_dp_size={attention_dp_size}, max_concurrency={max_concurrency}, "
                f"decode_interpolation_granularity={decode_interpolation_granularity}"
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

            if not perf_dict:
                raise RuntimeError(
                    f"estimate_perf returned empty result for num_gpus={num_gpus}, "
                    f"num_request={num_request}, model={model_name}, isl={isl}, osl={osl}"
                )

            if "tpot" not in perf_dict:
                raise KeyError(
                    f"'tpot' key missing in perf_dict for num_gpus={num_gpus}, "
                    f"num_request={num_request}. Available keys: {list(perf_dict.keys())}"
                )

            if "tokens/s/gpu" not in perf_dict:
                raise KeyError(
                    f"'tokens/s/gpu' key missing in perf_dict for num_gpus={num_gpus}, "
                    f"num_request={num_request}. Available keys: {list(perf_dict.keys())}"
                )

            itl_val = perf_dict["tpot"]
            thpt_val = perf_dict["tokens/s/gpu"]

            engine_decode_itl.append(itl_val)
            engine_decode_thpt_per_gpu.append(thpt_val)
            engine_batch_sizes.append(num_request)

        # Store results for this GPU configuration
        if not engine_decode_itl:
            raise RuntimeError(
                f"No decode results generated for num_gpus={num_gpus}. "
                f"sweep_num_request={sweep_num_request} but all estimations failed."
            )

        decode_results.append((num_gpus, engine_decode_itl, engine_decode_thpt_per_gpu, engine_batch_sizes))

    if not decode_results:
        error_msg = (
            f"No decode results generated - all GPU configurations were skipped.\n"
            f"Model: {model_name}, isl={isl}, osl={osl}\n"
            f"Attempted GPU configs: {profile_num_gpus}\n\n"
            f"Skipped configurations:\n"
        )
        for reason in skipped_configs:
            error_msg += f"  - {reason}\n"
        error_msg += (
            "\nSuggestions:\n"
            "  1. Increase GPU counts: Set min_num_gpus_per_engine higher (e.g., 8) and/or "
            "max_num_gpus_per_engine higher (e.g., 16, 32) to try larger configurations\n"
            "  2. Reduce sequence lengths (try isl=1024, osl=64)\n"
            "  3. Choose a smaller model if testing with few GPUs"
        )
        raise RuntimeError(error_msg)

    return decode_results

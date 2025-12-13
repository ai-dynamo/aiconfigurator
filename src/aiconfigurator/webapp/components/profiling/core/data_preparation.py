# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Data preparation for UI display.

This module formats profiling results into table data for the UI.
"""

import numpy as np

from aiconfigurator.webapp.components.profiling.sdk import generate_config_yaml


def _compute_parato(x, y):
    """
    Compute the pareto front (top-left is better) for the given x and y values.

    Returns sorted lists of the x and y values for the pareto front.
    """
    # Validate inputs
    if x is None or y is None:
        return [], []

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    if len(x) == 0:
        return [], []

    # Build point list and sort by x asc, then y desc so we prefer smaller x and larger y.
    points = list(zip(x, y))
    points.sort(key=lambda p: (p[0], -p[1]))

    # Single pass to keep only non-dominated points (minimize x, maximize y).
    pareto = []
    max_y = float("-inf")
    for px, py in points:
        if py > max_y:
            pareto.append((px, py))
            max_y = py

    # Return sorted by x ascending for convenience
    pareto.sort(key=lambda p: (p[0], p[1]))
    xs = [px for px, _ in pareto]
    ys = [py for _, py in pareto]
    return xs, ys


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
        decode_results: List of tuples (num_gpus, itl_list, thpt_list, batch_size_list)
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
    model_name: str,
    system: str,
    backend: str,
    version: str,
):
    """
    Prepare table data for cost analysis (GPU hours only, frontend handles cost calculation).

    Args:
        isl: Input sequence length
        osl: Output sequence length
        prefill_results: Tuple of (num_gpus, ttft, thpt_per_gpu) for prefill
        decode_results: List of tuples (num_gpus, itl_list, thpt_per_gpu_list, batch_size_list) for decode
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

    table_data = []
    for p_idx, (_p_ttft, _p_thpt) in enumerate(zip(p_ttft, p_thpt)):
        tokens_per_user_array = 1000 / d_itl

        # Calculate GPU hours (frontend will handle cost conversion if needed)
        prefill_gpu_hours = isl * 1000 / _p_thpt / 3600
        gpu_hours_array = osl * 1000 / d_thpt / 3600 + prefill_gpu_hours

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
                    round(float(gpu_hours_array[i]), 6),
                    config_yaml,
                ]
            )

    return table_data

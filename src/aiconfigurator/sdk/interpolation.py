# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pure interpolation/math functions extracted from PerfDatabase.

All functions are stateless — no class dependency, no side effects beyond logging.
Any mutable state (like extracted_metrics_cache) is passed as an explicit argument.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy import interpolate

from aiconfigurator.sdk.performance_result import PerformanceResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_value(data_value, metric: str = "latency"):
    """
    Extract a metric from a data value (handles both dict and float formats).

    Args:
        data_value: Either a dict {"latency": float, "power": float} or a float (legacy)
        metric: Which metric to extract ("latency" or "power")

    Returns:
        float: The requested metric value
    """
    if isinstance(data_value, dict):
        return data_value.get(metric, 0.0)
    else:
        # Legacy format: raw float is latency, power is 0
        return data_value if metric == "latency" else 0.0


def validate_interpolation_result(value: float) -> float:
    """
    Validate the value
    """
    if value < 0.0:
        logger.debug(f"Negative value detected {value}, pass")
    return value


def get_sample_leaf_value(data: dict):
    """Get a sample leaf value from nested dict to determine format."""
    current = data
    max_depth = 20  # Safety limit to prevent infinite loops
    depth = 0
    visited = set()  # Track visited dict ids to detect cycles

    while isinstance(current, dict) and current and depth < max_depth:
        dict_id = id(current)
        if dict_id in visited:
            # Circular reference detected
            logger.warning("Circular reference detected in _get_sample_leaf_value")
            break
        visited.add(dict_id)

        # Check if this is a leaf dict with latency/power keys
        if "latency" in current or "power" in current:
            return current

        try:
            key = next(iter(current))
            current = current[key]
            depth += 1
        except (StopIteration, KeyError, TypeError):
            # Handle edge cases: empty dict, missing key, or non-dict value
            break

    if depth >= max_depth:
        logger.warning(f"Maximum depth ({max_depth}) exceeded in _get_sample_leaf_value")

    return current


def nearest_1d_point_helper(x: int, values: list[int], inner_only: bool = True) -> tuple[int, int]:
    """
    Find the nearest 1d point
    """
    assert values is not None and len(values) >= 2, "values is None or len(values) < 2"
    sorted_values = sorted(values)

    if x < sorted_values[0]:
        if inner_only:
            raise ValueError(f"x is less than the smallest value in the list. {x=}, {sorted_values=}")
        else:
            return sorted_values[0], sorted_values[1]
    elif x > sorted_values[-1]:
        if inner_only:
            raise ValueError(f"x is greater than the largest value in the list. {x=}, {sorted_values=}")
        else:
            return sorted_values[-2], sorted_values[-1]

    for i, value in enumerate(sorted_values):
        if x >= value and i != len(sorted_values) - 1:
            continue
        else:
            end = value
            start = sorted_values[i - 1]
            break
    if start is None or end is None:
        raise ValueError(f"start or end is None. {x=}, {sorted_values=}, start={start=}, end={end=}")
    return start, end


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def extract_latency_and_energy_2d(data: dict) -> tuple[dict, dict]:
    """
    Extract both latency and energy from 2D dict-based data structure in a single pass.

    Args:
        data: Nested 2-level dict where leaf values are dicts {"latency": l, "power": p, "energy": e}

    Returns:
        tuple: (latency_data, energy_data) - two dicts with same structure but scalar values
    """
    latency_result = {}
    energy_result = {}

    for k1, v1 in data.items():
        latency_result[k1] = {}
        energy_result[k1] = {}

        for k2, v2 in v1.items():
            latency_result[k1][k2] = get_value(v2, "latency")
            energy_result[k1][k2] = get_value(v2, "energy")

    return latency_result, energy_result


def extract_latency_and_energy_3d(data: dict) -> tuple[dict, dict]:
    """
    Extract both latency and energy from 3D dict-based data structure in a single pass.

    This is more efficient than calling _extract_metric_data_3d twice.

    Args:
        data: Nested 3-level dict where leaf values are dicts {"latency": l, "power": p, "energy": e}

    Returns:
        tuple: (latency_data, energy_data) - two dicts with same structure but scalar values
    """
    latency_result = {}
    energy_result = {}

    for k1, v1 in data.items():
        latency_result[k1] = {}
        energy_result[k1] = {}

        for k2, v2 in v1.items():
            latency_result[k1][k2] = {}
            energy_result[k1][k2] = {}

            for k3, v3 in v2.items():
                latency_result[k1][k2][k3] = get_value(v3, "latency")
                energy_result[k1][k2][k3] = get_value(v3, "energy")

    return latency_result, energy_result


# ---------------------------------------------------------------------------
# 1-D interpolation
# ---------------------------------------------------------------------------


def interp_1d(x: list[int], y: list, value: int):
    """
    Interpolate the 1d data using linear interpolation.
    Handles both float and dict values.

    Args:
        x: list of x coordinates
        y: list of y values (can be floats or dicts)
        value: target x value

    Returns:
        float or dict: Interpolated result (dict if input was dict, float otherwise)
    """
    x0, x1 = x
    y0, y1 = y

    # Check if values are dicts (new format) or floats (legacy)
    if isinstance(y0, dict) and isinstance(y1, dict):
        # New format: interpolate latency and power separately
        lat0, lat1 = y0["latency"], y1["latency"]
        pow0, pow1 = y0["power"], y1["power"]

        # Apply interpolation logic for latency
        if (x0 - x1) * (lat0 - lat1) < 0 and (value - x0) * (value - x1) > 0:
            lat1 = lat0
        if lat0 == lat1:
            lat_result = lat0
        else:
            lat_result = lat0 + (lat1 - lat0) / (x1 - x0) * (value - x0)

        # Apply interpolation logic for power
        if (x0 - x1) * (pow0 - pow1) < 0 and (value - x0) * (value - x1) > 0:
            pow1 = pow0
        if pow0 == pow1:
            pow_result = pow0
        else:
            pow_result = pow0 + (pow1 - pow0) / (x1 - x0) * (value - x0)

        return {"latency": lat_result, "power": pow_result}
    else:
        # Legacy format: y values are floats
        if (x0 - x1) * (y0 - y1) < 0 and (value - x0) * (value - x1) > 0:
            y1 = y0
        if y0 == y1:
            return y0
        return y0 + (y1 - y0) / (x1 - x0) * (value - x0)


# ---------------------------------------------------------------------------
# Bilinear interpolation
# ---------------------------------------------------------------------------


def bilinear_interpolation(x_list: list[int], y_list: list[int], x: int, y: int, data: dict) -> float:
    """
    Interpolate the 2d data using bilinear interpolation
    """
    x1, x2 = x_list
    # assure xy has a rectengle grid
    y1, y2 = y_list
    # Calculate the weights for the corners
    Q11, Q12, Q21, Q22 = data[x1][y1], data[x1][y2], data[x2][y1], data[x2][y2]  # noqa: N806

    f_x1_y1 = Q11 * (x2 - x) * (y2 - y)
    f_x1_y2 = Q12 * (x2 - x) * (y - y1)
    f_x2_y1 = Q21 * (x - x1) * (y2 - y)
    f_x2_y2 = Q22 * (x - x1) * (y - y1)
    # Calculate the total weight
    total_weight = (x2 - x1) * (y2 - y1)
    # Calculate the interpolated value
    interpolated_value = (f_x1_y1 + f_x1_y2 + f_x2_y1 + f_x2_y2) / total_weight
    return interpolated_value


# ---------------------------------------------------------------------------
# 2-D linear interpolation
# ---------------------------------------------------------------------------


def interp_2d_linear(x: int, y: int, data: dict, extracted_metrics_cache: dict | None = None) -> dict:
    """
    Interpolate the 2D data using linear interpolation.

    Args:
        x: first dimension value
        y: second dimension value
        data: nested dict of data
        extracted_metrics_cache: optional cache dict for extracted metrics (keyed by id(data))

    Returns:
        dict: {"latency": float, "power": float, "energy": float} - interpolated values for all metrics
    """
    if extracted_metrics_cache is None:
        extracted_metrics_cache = {}

    # Check if data uses new dict format by sampling a leaf value
    sample_value = get_sample_leaf_value(data)

    if isinstance(sample_value, dict):
        # New format: interpolate latency and energy separately
        data_id = id(data)
        if data_id not in extracted_metrics_cache:
            extracted_metrics_cache[data_id] = extract_latency_and_energy_2d(data)

        latency_data, energy_data = extracted_metrics_cache[data_id]

        # Interpolate latency
        points_list = []
        latency_values = []
        x_left, x_right = nearest_1d_point_helper(x, list(latency_data.keys()))
        for i in [x_left, x_right]:
            y_left, y_right = nearest_1d_point_helper(y, list(latency_data[i].keys()))
            for j in [y_left, y_right]:
                points_list.append([i, j])
                latency_values.append(latency_data[i][j])

        latency = validate_interpolation_result(
            interpolate.griddata(np.array(points_list), np.array(latency_values), (x, y), method="linear")
        )

        # Interpolate energy using same points
        energy_values = []
        for i in [x_left, x_right]:
            y_left, y_right = nearest_1d_point_helper(y, list(energy_data[i].keys()))
            for j in [y_left, y_right]:
                energy_values.append(energy_data[i][j])

        energy = validate_interpolation_result(
            interpolate.griddata(np.array(points_list), np.array(energy_values), (x, y), method="linear")
        )

        return {"latency": latency, "power": 0.0, "energy": energy}
    else:
        # Legacy format: data values are floats
        points_list = []
        values_list = []
        x_left, x_right = nearest_1d_point_helper(x, list(data.keys()))
        for i in [x_left, x_right]:
            y_left, y_right = nearest_1d_point_helper(y, list(data[i].keys()))
            for j in [y_left, y_right]:
                points_list.append([i, j])
                values_list.append(data[i][j])

        latency = validate_interpolation_result(
            interpolate.griddata(np.array(points_list), np.array(values_list), (x, y), method="linear")
        )

        return {"latency": latency, "power": 0.0, "energy": 0.0}


# ---------------------------------------------------------------------------
# 3-D linear interpolation
# ---------------------------------------------------------------------------


def interp_3d_linear(x: int, y: int, z: int, data: dict) -> float:
    """
    Interpolate the 3d data using linear interpolation
    """
    points_list = []
    values_list = []
    x_left, x_right = nearest_1d_point_helper(x, list(data.keys()))
    for i in [x_left, x_right]:
        y_left, y_right = nearest_1d_point_helper(y, list(data[i].keys()))
        for j in [y_left, y_right]:
            z_left, z_right = nearest_1d_point_helper(z, list(data[i][j].keys()))
            points_list.append([i, j, z_left])
            points_list.append([i, j, z_right])
            values_list.append(data[i][j][z_left])
            values_list.append(data[i][j][z_right])

    return validate_interpolation_result(
        interpolate.griddata(np.array(points_list), np.array(values_list), (x, y, z), method="linear")
    )


# ---------------------------------------------------------------------------
# 2D-then-1D interpolation
# ---------------------------------------------------------------------------


def interp_2d_1d(x: int, y: int, z: int, data: dict, method="bilinear") -> float:
    """
    Interpolate the 3d data using the given method, 2d after 1d.
    """
    x_values = []
    x_left, x_right = nearest_1d_point_helper(x, list(data.keys()))

    for i in [x_left, x_right]:
        points_list = []
        values_list = []
        y_left, y_right = nearest_1d_point_helper(y, list(data[i].keys()))
        for j in [y_left, y_right]:
            z_left, z_right = nearest_1d_point_helper(z, list(data[i][j].keys()))
            points_list.append([j, z_left])
            points_list.append([j, z_right])
            values_list.append(data[i][j][z_left])
            values_list.append(data[i][j][z_right])
        if method == "cubic":
            x_values.append(
                validate_interpolation_result(
                    interpolate.griddata(np.array(points_list), np.array(values_list), (y, z), method="cubic")
                )
            )
        elif method == "bilinear":
            x_values.append(
                validate_interpolation_result(
                    bilinear_interpolation([y_left, y_right], [z_left, z_right], y, z, data[i])
                )
            )
        else:
            raise NotImplementedError

    return validate_interpolation_result(interp_1d([x_left, x_right], x_values, x))


# ---------------------------------------------------------------------------
# 3-D general interpolation (dispatches to 3d_linear or 2d_1d)
# ---------------------------------------------------------------------------


def interp_3d(x: int, y: int, z: int, data: dict, method: str, extracted_metrics_cache: dict | None = None) -> dict:
    """
    Interpolate the 3d data using the given method.

    Args:
        x, y, z: dimension values
        data: nested dict of data
        method: interpolation method ("linear", "cubic", "bilinear")
        extracted_metrics_cache: optional cache dict for extracted metrics (keyed by id(data))

    Returns:
        dict: {"latency": float, "power": float, "energy": float} - interpolated values for all metrics
        Note: power is always 0.0 as it's not currently used by callers (only latency and energy are used)
    """
    if extracted_metrics_cache is None:
        extracted_metrics_cache = {}

    # Check if data uses new dict format by sampling a leaf value
    sample_value = get_sample_leaf_value(data)

    if isinstance(sample_value, dict):
        # New format: interpolate latency and energy only (power is not used by callers)
        # Use cache to avoid repeated extraction of the same data dictionary
        data_id = id(data)
        if data_id not in extracted_metrics_cache:
            # Extract both metrics in a single pass for maximum efficiency
            extracted_metrics_cache[data_id] = extract_latency_and_energy_3d(data)

        latency_data, energy_data = extracted_metrics_cache[data_id]

        if method == "linear":
            latency = interp_3d_linear(x, y, z, latency_data)
            energy = interp_3d_linear(x, y, z, energy_data)
        else:
            latency = interp_2d_1d(x, y, z, latency_data, method)
            energy = interp_2d_1d(x, y, z, energy_data, method)

        return {"latency": latency, "power": 0.0, "energy": energy}
    else:
        # Legacy format: data values are floats
        if method == "linear":
            latency = interp_3d_linear(x, y, z, data)
        else:
            latency = interp_2d_1d(x, y, z, data, method)

        return {"latency": latency, "power": 0.0, "energy": 0.0}


# ---------------------------------------------------------------------------
# Data-grid extrapolation
# ---------------------------------------------------------------------------


def extrapolate_data_grid(
    data_dict: dict[int, dict[int, dict[int, float]]],
    target_x_list: list[int],
    target_y_list: list[int],
    target_z_list: list[int],
    sqrt_y_value: bool = False,
) -> None:
    """
    Extrapolate the data grid, we extrapolate the data grid at the initialization stage.
    Future query will based on interpolation.
    """
    x_list = sorted(data_dict.keys())
    for x in x_list:
        # z_direction
        for y in sorted(data_dict[x].keys()):
            z_dict = data_dict[x][y]
            if len(z_dict) <= 1:
                logger.warning(
                    f"only one data point for a given xy, might trigger error. "
                    f"Please revisit data collection. {x=}, {y=}, {z_dict=}"
                )
                continue
            for z in target_z_list:
                if z not in z_dict:
                    z_left, z_right = nearest_1d_point_helper(z, list(z_dict.keys()), False)
                    # Check if both left and right boundaries exist
                    if z_left not in z_dict or z_right not in z_dict:
                        logger.warning(
                            f"Skipping interpolation for z={z} as boundaries z_left={z_left} "
                            f"or z_right={z_right} do not exist in z_dict for x={x}, y={y}"
                        )
                        continue
                    value = interp_1d(
                        [z_left, z_right],
                        [data_dict[x][y][z_left], data_dict[x][y][z_right]],
                        z,
                    )
                    z_dict[z] = value

        # y_direction
        for y in target_y_list:
            if y not in data_dict[x]:
                y_left, y_right = nearest_1d_point_helper(y, list(data_dict[x].keys()), False)
                # Check if both left and right boundaries exist
                if y_left not in data_dict[x] or y_right not in data_dict[x]:
                    logger.warning(
                        f"Skipping interpolation for y={y} as boundaries y_left={y_left} "
                        f"or y_right={y_right} do not exist in data_dict[{x}]"
                    )
                    continue

                z_list = sorted(data_dict[x][y_left].keys())
                for z in z_list:
                    # Check if z exists in both y_left and y_right
                    if z not in data_dict[x][y_left] or z not in data_dict[x][y_right]:
                        logger.warning(
                            f"Skipping interpolation for z={z} as it does not exist in both "
                            f"y_left={y_left} and y_right={y_right}"
                        )
                        continue

                    y_left_value = data_dict[x][y_left][z]
                    y_right_value = data_dict[x][y_right][z]
                    assert y_right_value is not None, "y_right_value cannot be None"
                    if sqrt_y_value:
                        if isinstance(y_left_value, dict):
                            # Handle dict format: apply sqrt to both latency and power
                            y_left_value = {
                                "latency": math.sqrt(y_left_value["latency"]),
                                "power": math.sqrt(y_left_value["power"]) if y_left_value["power"] > 0 else 0.0,
                            }
                            y_right_value = {
                                "latency": math.sqrt(y_right_value["latency"]),
                                "power": math.sqrt(y_right_value["power"]) if y_right_value["power"] > 0 else 0.0,
                            }
                        else:
                            # Handle legacy float format
                            y_left_value = math.sqrt(y_left_value)
                            y_right_value = math.sqrt(y_right_value)
                    value = interp_1d([y_left, y_right], [y_left_value, y_right_value], y)
                    if sqrt_y_value:
                        if isinstance(value, dict):
                            # Square both latency and power
                            value = {
                                "latency": value["latency"] * value["latency"],
                                "power": value["power"] * value["power"],
                            }
                        else:
                            value = value * value

                    if y not in data_dict[x]:
                        data_dict[x][y] = {z: value}
                    else:
                        data_dict[x][y][z] = value

    for x in target_x_list:
        if x not in data_dict:
            x_left, x_right = nearest_1d_point_helper(x, list(data_dict.keys()), False)
            # Check if both left and right boundaries exist
            if x_left not in data_dict or x_right not in data_dict:
                logger.warning(
                    f"Skipping interpolation for x={x} as boundaries x_left={x_left} "
                    f"or x_right={x_right} do not exist in data_dict"
                )
                continue

            for y in sorted(data_dict[x_left].keys()):
                # Check if y exists in both x_left and x_right
                if y not in data_dict[x_left] or y not in data_dict[x_right]:
                    logger.warning(
                        f"Skipping interpolation for y={y} as it does not exist in both "
                        f"x_left={x_left} and x_right={x_right}"
                    )
                    continue

                for z in sorted(data_dict[x_left][y].keys()):
                    # Check if z exists in both x_left and x_right for the given y
                    if z not in data_dict[x_left][y] or z not in data_dict[x_right][y]:
                        logger.warning(
                            f"Skipping interpolation for z={z} as it does not exist in both "
                            f"x_left={x_left} and x_right={x_right} for y={y}"
                        )
                        continue

                    x_left_value = data_dict[x_left][y][z]
                    x_right_value = data_dict[x_right][y][z]
                    assert x_right_value is not None, "x_right_value cannot be None"
                    value = interp_1d([x_left, x_right], [x_left_value, x_right_value], x)
                    if x not in data_dict:
                        data_dict[x] = {y: {z: value}}
                    elif y not in data_dict[x]:
                        data_dict[x][y] = {z: value}
                    else:
                        data_dict[x][y][z] = value


# ---------------------------------------------------------------------------
# Analytical memory-bound latency estimate
# ---------------------------------------------------------------------------


def estimate_mem_op(
    mem_bytes: int,
    mem_bw: float,
    mem_bw_empirical_scaling_factor: float = 1.0,
    mem_empirical_constant_latency: float = 0.0,
    database_mode=None,
    default_database_mode=None,
) -> PerformanceResult | tuple[float, float, float]:
    """
    Compute memory-operation latency analytically (no CSV data).

    Args:
        mem_bytes: Number of bytes to transfer
        mem_bw: Memory bandwidth in bytes/s (system_spec["gpu"]["mem_bw"])
        mem_bw_empirical_scaling_factor: Empirical scaling factor (default 1.0)
        mem_empirical_constant_latency: Constant latency offset in seconds (default 0.0)
        database_mode: Requested database mode (None => use default_database_mode)
        default_database_mode: Fallback mode when database_mode is None

    Returns:
        PerformanceResult or tuple(sol_time, 0, sol_time) for SOL_FULL mode
    """
    # Import here to avoid circular imports at module level
    from aiconfigurator.sdk import common

    def get_sol(mem_bytes: int) -> tuple[float, float, float]:
        sol_time = mem_bytes / mem_bw * 1000
        return sol_time, 0, sol_time

    def get_empirical(mem_bytes: int) -> float:
        return (mem_bytes / (mem_bw * mem_bw_empirical_scaling_factor) + mem_empirical_constant_latency) * 1000

    if database_mode is None:
        database_mode = default_database_mode
    if database_mode == common.DatabaseMode.SOL:
        return PerformanceResult(get_sol(mem_bytes)[0], energy=0.0)
    elif database_mode == common.DatabaseMode.SOL_FULL:
        return get_sol(mem_bytes)
    elif database_mode == common.DatabaseMode.EMPIRICAL:
        return PerformanceResult(get_empirical(mem_bytes), energy=0.0)
    else:
        # hybrid and silicon modes have same logic
        return PerformanceResult(get_empirical(mem_bytes), energy=0.0)

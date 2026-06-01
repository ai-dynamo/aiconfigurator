# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Interpolation/math primitives extracted from PerfDatabase.

Functions here are free functions (no class dependency). Two of them mutate
caller-supplied state by design:

- ``extrapolate_data_grid`` pre-expands its ``data_dict`` argument in place
  (called once at load time so later queries can use interpolation only).
- ``interp_2d_linear`` and ``interp_3d`` populate the optional
  ``extracted_metrics_cache`` dict so repeated queries on the same data dict
  amortize the latency/energy split.

All other functions are pure.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy import interpolate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_value(data_value, metric: str = "latency"):
    """Extract a metric from a data value (handles both dict and float formats)."""
    if isinstance(data_value, dict):
        return data_value.get(metric, 0.0)
    # Legacy format: raw float is latency, power is 0
    return data_value if metric == "latency" else 0.0


def validate_interpolation_result(value):
    """Validate that an interpolated value is finite; log a debug line if negative."""
    value_array = np.asarray(value)
    if not np.all(np.isfinite(value_array)):
        raise ValueError(f"Non-finite value detected {value}")
    if np.any(value_array < 0.0):
        logger.debug(f"Negative value detected {value}, pass")
    return value


def get_sample_leaf_value(data: dict):
    """Walk the nested dict and return the first leaf-shaped value, used to
    distinguish dict-format leaves ({"latency": ..., "power": ...}) from
    legacy float leaves."""
    current = data
    max_depth = 20  # Safety limit to prevent infinite loops
    depth = 0
    visited = set()

    while isinstance(current, dict) and current and depth < max_depth:
        dict_id = id(current)
        if dict_id in visited:
            logger.warning("Circular reference detected in get_sample_leaf_value")
            break
        visited.add(dict_id)

        if "latency" in current or "power" in current:
            return current

        try:
            key = next(iter(current))
            current = current[key]
            depth += 1
        except (StopIteration, KeyError, TypeError):
            break

    if depth >= max_depth:
        logger.warning(f"Maximum depth ({max_depth}) exceeded in get_sample_leaf_value")

    return current


def _get_cached_extracted_metrics(cache: dict, ndim: int, data: dict, extractor):
    """Extract latency/energy views and cache them by object identity.

    The cache keeps the original object in the entry so an ``id(data)`` reused by
    a different short-lived dict cannot return stale extracted metrics.
    """
    cache_key = (ndim, id(data))
    cached = cache.get(cache_key)
    if cached is not None:
        cached_data, extracted = cached
        if cached_data is data:
            return extracted

    extracted = extractor(data)
    cache[cache_key] = (data, extracted)
    return extracted


def nearest_1d_point_helper(x: int, values: list[int], inner_only: bool = True) -> tuple[int, int]:
    """Return the two values bracketing ``x`` from ``values``.

    With ``inner_only=True``, raises ``ValueError`` when ``x`` is outside
    the range of ``values``. With ``inner_only=False``, returns the two
    closest values from the appropriate end (used for extrapolation).
    """
    assert values is not None and len(values) >= 1, "values is None or empty"
    if len(values) == 1:
        if inner_only and x != values[0]:
            raise ValueError(f"x is not equal to the only value in the list. {x=}, {values=}")
        return values[0], values[0]

    sorted_values = sorted(values)

    if x < sorted_values[0]:
        if inner_only:
            raise ValueError(f"x is less than the smallest value in the list. {x=}, {sorted_values=}")
        return sorted_values[0], sorted_values[1]
    if x > sorted_values[-1]:
        if inner_only:
            raise ValueError(f"x is greater than the largest value in the list. {x=}, {sorted_values=}")
        return sorted_values[-2], sorted_values[-1]

    start = end = None
    for i, value in enumerate(sorted_values):
        if x >= value and i != len(sorted_values) - 1:
            continue
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
    """Extract both latency and energy from 2D dict-based data in a single pass."""
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
    """Extract both latency and energy from 3D dict-based data in a single pass."""
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
    """Linear interpolation in 1-D. Handles both float and dict leaf values.

    When the leaves are dicts, every numeric metric present in BOTH endpoints
    is interpolated (e.g. ``latency``, ``power``, ``energy``). Dropping
    ``energy`` here used to zero out every extrapolated grid point and made
    the SDK power model return 0 W for any GEMM shape that wasn't an exact
    collection point.
    """
    x0, x1 = x
    y0, y1 = y

    def _interp_scalar(v0, v1):
        a, b = v0, v1
        if (x0 - x1) * (a - b) < 0 and (value - x0) * (value - x1) > 0:
            b = a
        if a == b:
            return a
        return a + (b - a) / (x1 - x0) * (value - x0)

    if isinstance(y0, dict) and isinstance(y1, dict):
        result = {}
        for key in y0.keys() & y1.keys():
            v0 = y0[key]
            v1 = y1[key]
            if isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
                result[key] = _interp_scalar(v0, v1)
        return result

    return _interp_scalar(y0, y1)


# ---------------------------------------------------------------------------
# Bilinear interpolation
# ---------------------------------------------------------------------------


def bilinear_interpolation(x_list: list[int], y_list: list[int], x: int, y: int, data: dict) -> float:
    """Bilinear interpolation on a 2-D rectangular grid."""
    x1, x2 = x_list
    y1, y2 = y_list
    Q11, Q12, Q21, Q22 = data[x1][y1], data[x1][y2], data[x2][y1], data[x2][y2]  # noqa: N806

    f_x1_y1 = Q11 * (x2 - x) * (y2 - y)
    f_x1_y2 = Q12 * (x2 - x) * (y - y1)
    f_x2_y1 = Q21 * (x - x1) * (y2 - y)
    f_x2_y2 = Q22 * (x - x1) * (y - y1)
    total_weight = (x2 - x1) * (y2 - y1)
    return (f_x1_y1 + f_x1_y2 + f_x2_y1 + f_x2_y2) / total_weight


# ---------------------------------------------------------------------------
# 2-D linear interpolation
# ---------------------------------------------------------------------------


def interp_2d_linear(x: int, y: int, data: dict, extracted_metrics_cache: dict | None = None) -> dict:
    """2-D linear interpolation. Returns a metrics dict (latency/power/energy)."""
    if extracted_metrics_cache is None:
        extracted_metrics_cache = {}

    sample_value = get_sample_leaf_value(data)

    if isinstance(sample_value, dict):
        latency_data, energy_data = _get_cached_extracted_metrics(
            extracted_metrics_cache, 2, data, extract_latency_and_energy_2d
        )

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

        energy_values = []
        for i in [x_left, x_right]:
            y_left, y_right = nearest_1d_point_helper(y, list(energy_data[i].keys()))
            for j in [y_left, y_right]:
                energy_values.append(energy_data[i][j])

        energy = validate_interpolation_result(
            interpolate.griddata(np.array(points_list), np.array(energy_values), (x, y), method="linear")
        )

        return {"latency": latency, "power": 0.0, "energy": energy}

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
    """3-D linear interpolation via scipy.interpolate.griddata."""
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


def interp_2d_1d(x: int, y: int, z: int, data: dict, method: str = "bilinear") -> float:
    """3-D interpolation done as 2-D (over y, z) followed by 1-D (over x)."""
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


def interp_3d(
    x: int,
    y: int,
    z: int,
    data: dict,
    method: str,
    extracted_metrics_cache: dict | None = None,
) -> dict:
    """3-D interpolation. Returns a metrics dict (latency/power/energy).

    Power is always 0.0 — current callers consume only latency and energy.
    """
    if extracted_metrics_cache is None:
        extracted_metrics_cache = {}

    sample_value = get_sample_leaf_value(data)

    if isinstance(sample_value, dict):
        latency_data, energy_data = _get_cached_extracted_metrics(
            extracted_metrics_cache, 3, data, extract_latency_and_energy_3d
        )

        if method == "linear":
            latency = interp_3d_linear(x, y, z, latency_data)
            energy = interp_3d_linear(x, y, z, energy_data)
        else:
            latency = interp_2d_1d(x, y, z, latency_data, method)
            energy = interp_2d_1d(x, y, z, energy_data, method)
        return {"latency": latency, "power": 0.0, "energy": energy}

    if method == "linear":
        latency = interp_3d_linear(x, y, z, data)
    else:
        latency = interp_2d_1d(x, y, z, data, method)
    return {"latency": latency, "power": 0.0, "energy": 0.0}


# ---------------------------------------------------------------------------
# Top-k regime-aware piecewise interpolation (DSA + DSV4 CSA)
# ---------------------------------------------------------------------------


def interp_context_topk_piecewise_from_raw(
    num_heads: int,
    full_s: int,
    b: int,
    raw_dict: dict | None,
    boundary_seq_len: int | None,
) -> dict | None:
    """Interpolate raw context module data without crossing a top-k regime boundary.

    DSA and DeepSeek-V4 CSA use different kernel paths before and after the
    top-k selected cache saturates. Smooth interpolation across that
    boundary can underestimate points just above it. This helper only
    applies when the exact raw ``(num_heads, batch)`` curve has at least
    two same-regime anchors.
    """
    if boundary_seq_len is None or raw_dict is None or num_heads not in raw_dict:
        return None

    exact_head_data = raw_dict[num_heads]
    curve = {
        seq_len: batch_dict[b]
        for seq_len, batch_dict in exact_head_data.items()
        if isinstance(batch_dict, dict) and b in batch_dict
    }
    if full_s in curve:
        value = curve[full_s]
        if isinstance(value, dict):
            return {
                "latency": get_value(value, "latency"),
                "power": get_value(value, "power"),
                "energy": get_value(value, "energy"),
            }
        return {"latency": float(value), "power": 0.0, "energy": 0.0}

    if full_s <= boundary_seq_len:
        same_regime_keys = sorted(seq_len for seq_len in curve if seq_len <= boundary_seq_len)
    else:
        same_regime_keys = sorted(seq_len for seq_len in curve if seq_len > boundary_seq_len)

    if len(same_regime_keys) < 2:
        return None

    if full_s < same_regime_keys[0]:
        if full_s <= boundary_seq_len:
            return None
        left, right = same_regime_keys[0], same_regime_keys[1]
    elif full_s > same_regime_keys[-1]:
        return None
    else:
        left, right = nearest_1d_point_helper(full_s, same_regime_keys)
    left_value = curve[left]
    right_value = curve[right]

    def _interp_metric(metric: str) -> float:
        left_metric = get_value(left_value, metric)
        right_metric = get_value(right_value, metric)
        return interp_1d([left, right], [left_metric, right_metric], full_s)

    return {
        "latency": _interp_metric("latency"),
        "power": _interp_metric("power"),
        "energy": _interp_metric("energy"),
    }


def interp_dsa_context_topk_piecewise_from_raw(
    num_heads: int,
    full_s: int,
    b: int,
    dsa_dict: dict | None,
    index_topk: int | None,
) -> dict | None:
    """DSA-context variant: same algorithm, ``index_topk`` is the boundary."""
    return interp_context_topk_piecewise_from_raw(num_heads, full_s, b, dsa_dict, index_topk)


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
    """Extrapolate the data grid in-place at initialization time so that
    later queries can rely on interpolation only."""
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
                y_keys = list(data_dict[x].keys())
                if len(y_keys) < 2:
                    logger.warning(
                        f"Skipping y-direction interpolation for x={x}: only {len(y_keys)} y-value(s), need at least 2"
                    )
                    break
                y_left, y_right = nearest_1d_point_helper(y, y_keys, False)
                if y_left not in data_dict[x] or y_right not in data_dict[x]:
                    logger.warning(
                        f"Skipping interpolation for y={y} as boundaries y_left={y_left} "
                        f"or y_right={y_right} do not exist in data_dict[{x}]"
                    )
                    continue

                z_list = sorted(data_dict[x][y_left].keys())
                for z in z_list:
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

                        def _sqrt_leaf(v):
                            if isinstance(v, dict):
                                return {
                                    key: (math.sqrt(metric) if isinstance(metric, (int, float)) and metric > 0 else 0.0)
                                    for key, metric in v.items()
                                }
                            return math.sqrt(v) if v > 0 else 0.0

                        y_left_value = _sqrt_leaf(y_left_value)
                        y_right_value = _sqrt_leaf(y_right_value)
                    value = interp_1d([y_left, y_right], [y_left_value, y_right_value], y)
                    if sqrt_y_value:
                        if isinstance(value, dict):
                            value = {key: metric * metric for key, metric in value.items()}
                        else:
                            value = value * value

                    if y not in data_dict[x]:
                        data_dict[x][y] = {z: value}
                    else:
                        data_dict[x][y][z] = value

    x_keys = list(data_dict.keys())
    for x in target_x_list:
        if x not in data_dict:
            if len(x_keys) < 2:
                logger.warning(f"Skipping x-direction interpolation: only {len(x_keys)} x-value(s), need at least 2")
                break
            x_left, x_right = nearest_1d_point_helper(x, x_keys, False)
            if x_left not in data_dict or x_right not in data_dict:
                logger.warning(
                    f"Skipping interpolation for x={x} as boundaries x_left={x_left} "
                    f"or x_right={x_right} do not exist in data_dict"
                )
                continue

            for y in sorted(data_dict[x_left].keys()):
                if y not in data_dict[x_left] or y not in data_dict[x_right]:
                    logger.warning(
                        f"Skipping interpolation for y={y} as it does not exist in both "
                        f"x_left={x_left} and x_right={x_right}"
                    )
                    continue

                for z in sorted(data_dict[x_left][y].keys()):
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

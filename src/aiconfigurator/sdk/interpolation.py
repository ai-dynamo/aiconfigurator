# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared perf-table primitives: the structured-miss error and the leaf accessor.

All interpolation itself lives in ``aiconfigurator.sdk.perf_interp`` (the
unified N-axis engine). The scipy-based interp family that used to live here
(interp_1d/2d/3d, griddata wrappers, nearest-point helpers) had its last
consumer migrated to the engine and was deleted with it.
"""

from __future__ import annotations


class InterpolationDataNotAvailableError(ValueError):
    """Raised when interpolation cannot produce a real value from available data.

    Subclasses ``ValueError`` so existing callers that catch ``ValueError``
    keep working. The perf-DB layer catches this specific class to classify
    the failure as "missing silicon data" without swallowing genuine
    programming bugs that raise plain ``ValueError`` deeper in the stack.
    """


def get_value(data_value, metric: str = "latency"):
    """Extract a metric from a data value (handles both dict and float formats)."""
    if isinstance(data_value, dict):
        return data_value.get(metric, 0.0)
    # Legacy format: raw float is latency, power is 0
    return data_value if metric == "latency" else 0.0

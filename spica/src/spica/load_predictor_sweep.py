# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planner load-predictor independent grid sweep.

A brute-force grid search that picks the best load-predictor preset *per
throughput-scaling interval*, evaluated by one-step-ahead forecast loss over
the workload's traffic windows. It reuses the real dynamo planner predictors
(so the chosen preset is what the planner will actually run) and the planner's
own (densify-fixed) trace->window tool.

Trigger rules:
- No throughput-enabled planner policy candidate -> nothing to sweep.
- Static (non-trace) workload -> ``constant_last`` for every interval (the next
  window equals this one; there is no series to learn).
- Otherwise brute-force every configured preset, once per distinct interval.

The dynamo predictors (Rust runtime + prophet/pmdarima/filterpy) are imported
lazily so ``import spica`` and the config schema stay light.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from tqdm import tqdm

from .config import SmartSearchConfig
from .planner import throughput_intervals

# preset id -> predictor family + knob overrides (the Load Predictor Search
# Space table in the design proposal). "family" selects the dynamo predictor
# class; the rest map to the config fields the predictor reads.
LOAD_PREDICTOR_PRESETS: dict[str, dict[str, Any]] = {
    "constant_last": {"family": "constant", "log1p": False},
    "arima_raw": {"family": "arima", "log1p": False},
    "arima_log1p": {"family": "arima", "log1p": True},
    "prophet_w20_raw": {"family": "prophet", "log1p": False, "prophet_window_size": 20},
    "prophet_w20_log1p": {"family": "prophet", "log1p": True, "prophet_window_size": 20},
    "prophet_w50_raw": {"family": "prophet", "log1p": False, "prophet_window_size": 50},
    "prophet_w50_log1p": {"family": "prophet", "log1p": True, "prophet_window_size": 50},
    "kalman_default_raw": {
        "family": "kalman",
        "log1p": False,
        "q_level": 1.0,
        "q_trend": 0.1,
        "r": 10.0,
        "min_points": 5,
    },
    "kalman_default_log1p": {
        "family": "kalman",
        "log1p": True,
        "q_level": 1.0,
        "q_trend": 0.1,
        "r": 10.0,
        "min_points": 5,
    },
    "kalman_reactive_raw": {
        "family": "kalman",
        "log1p": False,
        "q_level": 10.0,
        "q_trend": 1.0,
        "r": 5.0,
        "min_points": 3,
    },
    "kalman_reactive_log1p": {
        "family": "kalman",
        "log1p": True,
        "q_level": 10.0,
        "q_trend": 1.0,
        "r": 5.0,
        "min_points": 3,
    },
}

# Planner defaults for knobs a preset does not pin (mirror SLAPlannerDefaults).
_DEFAULTS = {"prophet_window_size": 50, "q_level": 1.0, "q_trend": 0.1, "r": 10.0, "min_points": 5}


@dataclass
class Window:
    """One observation window (mirrors dynamo TrafficObservation metrics)."""

    num_req: float
    isl: float
    osl: float


@dataclass
class _PredictorConfig:
    """Duck-typed stand-in for dynamo's PlannerConfig, exposing exactly the
    fields the load predictors read (see planner/core/load/predictors.py).
    Avoids depending on PlannerConfig's required-field surface."""

    load_predictor_log1p: bool
    prophet_window_size: int
    throughput_adjustment_interval_seconds: int
    kalman_q_level: float
    kalman_q_trend: float
    kalman_r: float
    kalman_min_points: int


@dataclass
class LoadPredictorResult:
    # best preset id per throughput interval (seconds)
    best_by_interval: dict[int, str] = field(default_factory=dict)
    # interval -> {preset id -> mean one-step-ahead loss}
    losses: dict[int, dict[str, float]] = field(default_factory=dict)
    # why the sweep produced this (diagnostics)
    reason: str = ""


# --------------------------------------------------------------------------
# Loss (pure; matches the design proposal's target metric)
# --------------------------------------------------------------------------
def _err(pred: float, actual: float) -> float:
    return abs(math.log1p(max(pred, 0.0)) - math.log1p(max(actual, 0.0)))


def window_loss(n_hat: float, i_hat: float, o_hat: float, n: float, i: float, o: float) -> float:
    """Weighted log-scale one-step-ahead error for one window.

    0.4 total prefill tokens (N*I) + 0.4 total decode tokens (N*O) + 0.1 isl +
    0.1 osl. num_req is scored only via the products, by design. Empty windows
    (n=i=o=0) contribute 0 when predicted as ~0 and penalize over-prediction.
    """
    return (
        0.4 * _err(n_hat * i_hat, n * i)
        + 0.4 * _err(n_hat * o_hat, n * o)
        + 0.1 * _err(i_hat, i)
        + 0.1 * _err(o_hat, o)
    )


# --------------------------------------------------------------------------
# Lazy dynamo imports (Rust runtime + prophet/pmdarima/filterpy)
# --------------------------------------------------------------------------
def _load_predictors():
    from dynamo.planner.core.load.predictors import LOAD_PREDICTORS

    return LOAD_PREDICTORS


def build_windows(trace_path: str, interval_s: int) -> list[Window]:
    """Aggregate a mooncake trace into per-interval windows via the planner's
    own (densify-fixed) tool, so middle empty windows are preserved."""
    from dynamo.planner.offline.trace_data import extract_metrics_from_mooncake

    return [
        Window(float(m["request_count"]), float(m["avg_isl"]), float(m["avg_osl"]))
        for m in extract_metrics_from_mooncake(trace_path, interval_s)
    ]


def _make_config(preset: dict[str, Any], interval_s: int) -> _PredictorConfig:
    return _PredictorConfig(
        load_predictor_log1p=preset["log1p"],
        prophet_window_size=preset.get("prophet_window_size", _DEFAULTS["prophet_window_size"]),
        throughput_adjustment_interval_seconds=interval_s,
        kalman_q_level=preset.get("q_level", _DEFAULTS["q_level"]),
        kalman_q_trend=preset.get("q_trend", _DEFAULTS["q_trend"]),
        kalman_r=preset.get("r", _DEFAULTS["r"]),
        kalman_min_points=preset.get("min_points", _DEFAULTS["min_points"]),
    )


def _new_predictors(preset: dict[str, Any], interval_s: int):
    cls = _load_predictors()[preset["family"]]
    cfg = _make_config(preset, interval_s)
    return cls(cfg), cls(cfg), cls(cfg)


def evaluate_preset(windows: list[Window], preset: dict[str, Any], interval_s: int, warmup: int) -> float:
    """Mean one-step-ahead loss for one preset over the window series.

    Mirrors the online throughput loop: feed every window (incl. middle empties)
    to three predictors (num_req/isl/osl); at each window predict the next from
    prior history, then observe. Score windows at/after ``warmup`` so all
    presets are compared on the same window set; ``predict_next`` is called once
    per window before ``add_data_point`` (required by the Kalman predict/update
    cadence).
    """
    pn, pi, po = _new_predictors(preset, interval_s)
    losses: list[float] = []
    for t, w in enumerate(windows):
        try:
            n_hat, i_hat, o_hat = pn.predict_next(), pi.predict_next(), po.predict_next()
        except Exception:
            n_hat, i_hat, o_hat = pn.get_last_value(), pi.get_last_value(), po.get_last_value()
        if t >= warmup:
            losses.append(window_loss(n_hat, i_hat, o_hat, w.num_req, w.isl, w.osl))
        pn.add_data_point(w.num_req)
        pi.add_data_point(w.isl)
        po.add_data_point(w.osl)
    return mean(losses) if losses else math.inf


def _common_warmup(preset_ids: list[str], interval_s: int) -> int:
    """Max ``minimum_data_points`` across the selected presets, so every preset
    is scored on the same windows."""
    mins = []
    for pid in preset_ids:
        pn, _, _ = _new_predictors(LOAD_PREDICTOR_PRESETS[pid], interval_s)
        mins.append(pn.minimum_data_points)
    return max(mins) if mins else 0


def sweep_load_predictor(config: SmartSearchConfig, *, show_progress: bool = True) -> LoadPredictorResult:
    """Grid-search the load-predictor presets, once per distinct throughput
    interval, picking the lowest-loss preset per interval. ``show_progress``
    draws a tqdm bar over presets per interval."""
    space = config.search_space
    intervals = throughput_intervals(space.planner_scaling_policy)
    if not intervals:
        return LoadPredictorResult(reason="no_throughput_scaling_candidate")

    if not config.workload.is_trace_based:
        return LoadPredictorResult(
            best_by_interval={iv: "constant_last" for iv in intervals},
            reason="static_workload_constant",
        )

    result = LoadPredictorResult(reason="swept")
    for iv in intervals:
        windows = build_windows(config.workload.trace_path, iv)
        warmup = _common_warmup(space.load_predictor_presets, iv)
        per_preset: dict[str, float] = {}
        bar = tqdm(
            space.load_predictor_presets,
            desc=f"load-predictor @ {iv}s ({len(windows)} windows, warmup {warmup})",
            unit="preset",
            disable=not show_progress,
        )
        for pid in bar:
            per_preset[pid] = evaluate_preset(windows, LOAD_PREDICTOR_PRESETS[pid], iv, warmup)
            bar.set_postfix_str(f"{pid}={per_preset[pid]:.3f}")
        result.losses[iv] = per_preset
        result.best_by_interval[iv] = min(per_preset, key=per_preset.__getitem__)
    return result

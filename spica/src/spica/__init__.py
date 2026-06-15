# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spica: Replay-backed smart sweeper for Dynamo deployments (Profiler V2)."""

from __future__ import annotations

from .config import (
    Candidate,
    OptimizationGoal,
    OptimizationTarget,
    SearchSpace,
    SLATarget,
    SmartSearchConfig,
    SweepConfig,
    Workload,
)
from .kv_estimate import (
    NoPerfDatabase,
    estimate_kv_tokens,
    feasible_shape_tokens,
)
from .load_predictor_sweep import (
    LoadPredictorResult,
    predictor_fields,
    sweep_load_predictor,
    window_loss,
)
from .parallel_enum import (
    DisaggParallelConfig,
    ParallelShape,
    ReplicaParallelConfig,
    enumerate_disagg_configs,
    enumerate_parallel_configs,
    enumerate_worker_shapes,
)
from .planner import FPM_SAMPLING, LOAD_SENSITIVITY, SCALING_POLICIES, ScalingPolicy, throughput_intervals
from .sample import unroll_sample
from .search import run_smart_search

__all__ = [
    "Candidate",
    "OptimizationGoal",
    "OptimizationTarget",
    "SearchSpace",
    "SLATarget",
    "SmartSearchConfig",
    "SweepConfig",
    "Workload",
    "run_smart_search",
    # parallel-config enumeration
    "ParallelShape",
    "ReplicaParallelConfig",
    "DisaggParallelConfig",
    "enumerate_parallel_configs",
    "enumerate_worker_shapes",
    "enumerate_disagg_configs",
    # planner preset decode
    "SCALING_POLICIES",
    "ScalingPolicy",
    "throughput_intervals",
    "FPM_SAMPLING",
    "LOAD_SENSITIVITY",
    # load-predictor sweep
    "LoadPredictorResult",
    "sweep_load_predictor",
    "window_loss",
    "predictor_fields",
    # KV-cache feasibility
    "NoPerfDatabase",
    "estimate_kv_tokens",
    "feasible_shape_tokens",
    # sample unroll
    "unroll_sample",
]

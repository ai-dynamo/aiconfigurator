# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input schema for a Spica smart-search run.

These are the pydantic models from the design proposal
(``docs/proposals/dgdr-profiler-smart-search-plan.md`` in ai-dynamo/dynamo),
transcribed as the single source of truth for the search inputs:

- :class:`SearchSpace`        — the knobs to sweep + pinned context, per component
- :class:`Workload`           — the traffic every candidate is evaluated against
- :class:`OptimizationGoal`   — what "better" means + the SLA constraint
- :class:`SweepConfig`        — sweep run-control
- :class:`SmartSearchConfig`  — top-level bundle; one YAML maps to this
- :class:`Candidate`          — one evaluated configuration + its replay metrics

Field names are snake_case to match AIConfigurator's ``Task`` convention so the
eventual merge into an AIC sweep task is mechanical.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class OptimizationTarget(str, Enum):
    """What the search optimizes for."""

    THROUGHPUT = "throughput"  # maximize replay throughput
    E2E_LATENCY = "e2e_latency"  # minimize mean end-to-end latency
    GOODPUT = "goodput"  # maximize SLA-satisfying throughput
    GOODPUT_PER_GPU = "goodput_per_gpu"  # maximize goodput / used_gpus

    @property
    def maximize(self) -> bool:
        """True when larger is better (throughput/goodput/goodput_per_gpu)."""
        return self is not OptimizationTarget.E2E_LATENCY


class SLATarget(BaseModel):
    """Per-request latency bounds in ms. Set ttft_ms+itl_ms, or e2e_ms."""

    model_config = ConfigDict(extra="forbid")

    ttft_ms: float | None = None
    itl_ms: float | None = None
    e2e_ms: float | None = None


class OptimizationGoal(BaseModel):
    """User-owned objective and SLA. Pinned; never searched."""

    model_config = ConfigDict(extra="forbid")

    target: OptimizationTarget = OptimizationTarget.THROUGHPUT
    sla: SLATarget | None = None  # required for goodput / goodput_per_gpu

    @model_validator(mode="after")
    def _require_sla_for_goodput(self) -> "OptimizationGoal":
        needs_sla = self.target in (
            OptimizationTarget.GOODPUT,
            OptimizationTarget.GOODPUT_PER_GPU,
        )
        has_sla = self.sla is not None and (
            self.sla.e2e_ms is not None or (self.sla.ttft_ms is not None and self.sla.itl_ms is not None)
        )
        if needs_sla and not has_sla:
            raise ValueError(f"{self.target.value} requires an SLA target (ttft_ms+itl_ms or e2e_ms)")
        return self


class Workload(BaseModel):
    """Traffic every candidate is evaluated against (pinned, never searched).

    A synthetic static workload, or a replay-ready trace — ``trace_path``
    discriminates. The trace is the dynamic-traffic path; the synthetic fields
    cover the static, backward-compatible case.
    """

    model_config = ConfigDict(extra="forbid")

    # synthetic static workload (used when trace_path is unset)
    isl: int | None = None
    osl: int | None = None
    concurrency: float | None = None  # set concurrency OR request_rate
    request_rate: float | None = None
    request_count: int | None = None
    shared_prefix_ratio: float = 0.0  # cache-locality / prefix sharing
    num_prefix_groups: int = 0
    turns_per_session: int = 1  # multi-turn sessions

    # dynamic trace source (mutually exclusive with the synthetic fields)
    trace_path: str | None = None
    trace_format: str = "mooncake"  # replay-ready trace schema
    arrival_speedup_ratio: float = 1.0  # scale trace inter-arrival times

    @property
    def is_trace_based(self) -> bool:
        return self.trace_path is not None


# Allowed choices for each swept search-space dimension. A configured value must
# be a non-empty subset of these (one or more); the field defaults below use the
# full set (or a sensible subset, e.g. ``backend``). Centralized here so the
# candidate generator can reuse it. Pinned scalars and the generated
# ``parallel_configs`` are intentionally not choice-constrained.
SEARCH_CHOICES: dict[str, tuple] = {
    "deployment_mode": ("disagg", "agg"),
    "backend": ("vllm", "sglang", "trtllm"),
    "prefill_max_num_batched_tokens": (8192, 16384, 32768),
    "prefill_max_num_seqs": (1, 2, 4, 8),
    "decode_max_num_batched_tokens": (8192,),
    "decode_max_num_seqs": (256, 512, 1024),
    "agg_max_num_batched_tokens": (8192, 16384, 32768),
    "agg_max_num_seqs": (256, 512, 1024),
    "router_mode": ("kv_router", "round_robin"),
    "overlap_score_credit": (0.0, 0.5, 1.0),
    "prefill_load_scale": (0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
    "host_cache_hit_weight": (0.5, 0.75, 1.0),
    "disk_cache_hit_weight": (0.0, 0.25, 0.5),
    "router_temperature": (0.0, 0.2, 0.5, 1.0),
    "planner_scaling_policy": (
        "disabled",
        "throughput_180_5",
        "throughput_600_5",
        "load_180_5",
        "load_180_10",
        "hybrid_180_5",
        "hybrid_600_5",
    ),
    "planner_fpm_sampling": ("small", "default", "large", "fine"),
    "planner_load_sensitivity": ("aggressive", "default", "conservative"),
    "load_predictor_presets": (
        "constant_last",
        "arima_raw",
        "arima_log1p",
        "prophet_w20_raw",
        "prophet_w20_log1p",
        "prophet_w50_raw",
        "prophet_w50_log1p",
        "kalman_default_raw",
        "kalman_default_log1p",
        "kalman_reactive_raw",
        "kalman_reactive_log1p",
    ),
}


class SearchSpace(BaseModel):
    """Inputs to one search run, grouped by component.

    Each group lists its swept knobs (list-typed candidate sets; a
    single-element list pins that knob) followed by the pinned knobs that group
    needs (scalars). When ``deployment_mode`` lists both branches the optimizer
    runs one flat study per branch and ranks across both. Most fields drive the
    main Vizier sweep; ``load_predictor_presets`` is swept by a separate
    forecast-loss grid, with its winner pinned into the main sweep.
    """

    model_config = ConfigDict(extra="forbid")

    # deployment: branch + backend + legal parallel shapes
    deployment_mode: list[str] = ["disagg", "agg"]  # branches to explore; pin with one
    backend: list[str] = ["vllm"]  # vllm | sglang | trtllm
    parallel_configs: list[dict[str, Any]] = Field(default_factory=list)  # generated when empty
    # pinned
    model_name: str  # HF id or private model name
    hardware_sku: str  # e.g. "h200_sxm"
    gpu_budget: int = 32  # max GPUs per candidate
    min_gpu_budget: int | None = None
    min_endpoint: int | None = None
    context_length: int | None = None
    startup_time: float | None = None
    aic_nextn: int | None = None  # speculative-decode (MTP) depth, 1..5

    # prefill engine (disagg branch): scheduler batching capacity
    prefill_max_num_batched_tokens: list[int] = [8192, 16384, 32768]
    prefill_max_num_seqs: list[int] = [1, 2, 4, 8]
    # pinned
    prefill_block_size: int = 64
    prefill_gpu_memory_utilization: float = 0.9
    prefill_enable_prefix_caching: bool = True

    # decode engine (disagg branch): scheduler batching capacity
    decode_max_num_batched_tokens: list[int] = [8192]
    decode_max_num_seqs: list[int] = [256, 512, 1024]
    # pinned
    decode_block_size: int = 64
    decode_gpu_memory_utilization: float = 0.9
    decode_enable_prefix_caching: bool = False  # forced off for decode workers

    # agg engine (agg branch): scheduler batching capacity
    agg_max_num_batched_tokens: list[int] = [8192, 16384, 32768]
    agg_max_num_seqs: list[int] = [256, 512, 1024]
    # pinned
    agg_block_size: int = 64
    agg_gpu_memory_utilization: float = 0.9
    agg_enable_prefix_caching: bool = True

    # kv manager: multi-tier offload policy (all pinned; G3/G4 extend G2)
    num_g2_blocks: int = 0  # 0 disables host offload
    bandwidth_g1_to_g2_gbps: float | None = None
    bandwidth_g2_to_g1_gbps: float | None = None
    offload_batch_size: int | None = None

    # router (KV-router knobs are ignored under round_robin)
    router_mode: list[str] = ["kv_router", "round_robin"]
    overlap_score_credit: list[float] = [0.0, 0.5, 1.0]
    prefill_load_scale: list[float] = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    host_cache_hit_weight: list[float] = [0.5, 0.75, 1.0]
    disk_cache_hit_weight: list[float] = [0.0, 0.25, 0.5]
    router_temperature: list[float] = [0.0, 0.2, 0.5, 1.0]
    # pinned (admission control)
    active_decode_blocks_threshold: int | None = None
    active_prefill_tokens_threshold: int | None = None
    active_prefill_tokens_threshold_frac: float | None = None
    no_admission_control: bool = False

    # planner: preset ids expanded by the candidate generator
    # "disabled" = planner not enabled (no autoscaling, static replica count)
    planner_scaling_policy: list[str] = [
        "disabled",
        "throughput_180_5",
        "throughput_600_5",
        "load_180_5",
        "load_180_10",
        "hybrid_180_5",
        "hybrid_600_5",
    ]
    planner_fpm_sampling: list[str] = ["small", "default", "large", "fine"]
    planner_load_sensitivity: list[str] = ["aggressive", "default", "conservative"]

    # planner load predictor — independent grid sweep (ranked by one-step-ahead
    # forecast loss, NOT the main Vizier loop); the winning preset is pinned
    # into the main sweep. Only relevant under predictive throughput scaling.
    load_predictor_presets: list[str] = [
        "constant_last",
        "arima_raw",
        "arima_log1p",
        "prophet_w20_raw",
        "prophet_w20_log1p",
        "prophet_w50_raw",
        "prophet_w50_log1p",
        "kalman_default_raw",
        "kalman_default_log1p",
        "kalman_reactive_raw",
        "kalman_reactive_log1p",
    ]

    @model_validator(mode="after")
    def _validate_search_choices(self) -> "SearchSpace":
        """Each swept dimension must be a non-empty subset of its listed choices."""
        for field_name, allowed in SEARCH_CHOICES.items():
            values = getattr(self, field_name)
            if not values:
                raise ValueError(f"{field_name} must list at least one choice; allowed: {list(allowed)}")
            invalid = [v for v in values if v not in allowed]
            if invalid:
                raise ValueError(f"{field_name} has invalid choices {invalid}; allowed: {list(allowed)}")
        return self


class SweepConfig(BaseModel):
    """Sweep run-control."""

    model_config = ConfigDict(extra="forbid")

    max_rounds: int = 20  # total Vizier suggestion/evaluation rounds
    parallel_evals: int = 16  # concurrent CPU replay evaluations
    candidates_per_round: int | None = None  # defaults to parallel_evals
    random_seed: int = 1


class Candidate(BaseModel):
    """One evaluated configuration and its replay performance."""

    model_config = ConfigDict(extra="forbid")

    config: dict[str, Any]  # the decoded knob assignment (engine/router/planner)
    used_gpus: int
    score: float  # objective score, normalized so higher is better
    metrics: dict[str, float]  # replay performance: throughput, ttft, itl, e2e, goodput


class SmartSearchConfig(BaseModel):
    """Top-level config integrating every search input; one YAML maps to this."""

    model_config = ConfigDict(extra="forbid")

    search_space: SearchSpace
    workload: Workload
    goal: OptimizationGoal = Field(default_factory=OptimizationGoal)
    sweep: SweepConfig = Field(default_factory=SweepConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SmartSearchConfig":
        """Load + validate one YAML file into the nested config."""
        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data)

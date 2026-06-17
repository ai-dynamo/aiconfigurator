# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the per-branch candidate space the sampler searches over.

A *branch* is one ``(deployment_mode, backend)`` combination from the SearchSpace
lists. For each branch we enumerate the **KV-feasible** parallel configs
(:func:`spica.model_hw.parallel_configs_for`) — that legal set is the categorical
domain the sampler picks an index into — and collect the branch's searchable
atomic knobs (each a configured choice list straight off the SearchSpace). The
sampler turns one :class:`BranchSpace` into one Vizier study (one study per
branch, per the design proposal).

``backend`` is fixed per branch (not a searched knob); ``load_predictor_presets``
is resolved separately by the load-predictor sub-sweep and is not a sampler
dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import SmartSearchConfig
from .model_hw import parallel_configs_for
from .parallel_enum import DisaggParallelConfig, ReplicaParallelConfig

# Searchable atomic knobs, by group. Names are SearchSpace list-typed fields.
_ROUTER_KNOBS = (
    "router_mode",
    "overlap_score_credit",
    "prefill_load_scale",
    "host_cache_hit_weight",
    "disk_cache_hit_weight",
    "router_temperature",
)
_PLANNER_KNOBS = ("planner_scaling_policy", "planner_fpm_sampling", "planner_load_sensitivity")
_AGG_ENGINE = ("agg_max_num_batched_tokens", "agg_max_num_seqs")
_DISAGG_ENGINE = (
    "prefill_max_num_batched_tokens",
    "prefill_max_num_seqs",
    "decode_max_num_batched_tokens",
    "decode_max_num_seqs",
)


@dataclass(frozen=True)
class BranchSpace:
    """One ``(deployment_mode, backend)`` branch of the search."""

    deployment_mode: str
    backend: str
    # KV-feasible parallel configs — the categorical domain (pick an index).
    parallel_configs: tuple[ReplicaParallelConfig | DisaggParallelConfig, ...]
    # Searchable atomic knob -> its configured choice list (from the SearchSpace).
    knob_choices: dict[str, list[Any]]


def _engine_knobs(deployment_mode: str) -> tuple[str, ...]:
    return _AGG_ENGINE if deployment_mode == "agg" else _DISAGG_ENGINE


def branch_knob_choices(search_space, deployment_mode: str) -> dict[str, list[Any]]:
    """The searchable atomic knobs for a branch (router + planner + the active
    mode's engine batching), each mapped to its configured choice list."""
    names = (*_ROUTER_KNOBS, *_PLANNER_KNOBS, *_engine_knobs(deployment_mode))
    return {name: list(getattr(search_space, name)) for name in names}


def enumerate_branches(config: SmartSearchConfig, *, max_seq_len: int | None = None) -> list[BranchSpace]:
    """One :class:`BranchSpace` per ``(deployment_mode, backend)``.

    ``max_seq_len`` is forwarded to :func:`parallel_configs_for` for the KV
    feasibility filter; ``None`` uses the model's max context length (the
    conservative default).
    """
    ss = config.search_space
    branches: list[BranchSpace] = []
    for deployment_mode in ss.deployment_mode:
        for backend in ss.backend:
            parallel_configs = parallel_configs_for(
                ss.model_name,
                ss.hardware_sku,
                gpu_budget=ss.gpu_budget,
                deployment_mode=deployment_mode,
                backend=backend,
                min_gpu_budget=ss.min_gpu_budget,
                max_seq_len=max_seq_len,
            )
            branches.append(
                BranchSpace(
                    deployment_mode=deployment_mode,
                    backend=backend,
                    parallel_configs=tuple(parallel_configs),
                    knob_choices=branch_knob_choices(ss, deployment_mode),
                )
            )
    return branches

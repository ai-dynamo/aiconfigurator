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
from .model_hw import NoViableParallelConfig, parallel_configs_for
from .parallel_enum import DisaggParallelConfig, ParallelShape, ReplicaParallelConfig

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


def _shape_from_dict(d: dict[str, Any]) -> ParallelShape:
    """A per-worker :class:`ParallelShape` from a pinned shape dict. Omitted dims
    default to 1 (so dense models can write just ``{tp: N}``); ``pp`` defaults to 1."""
    if "tp" not in d:
        raise ValueError(f"a parallel_configs shape needs a 'tp' field, got {d}")
    return ParallelShape(
        tp=int(d["tp"]),
        dp=int(d.get("attention_dp", 1)),
        moe_tp=int(d.get("moe_tp", 1)),
        moe_ep=int(d.get("moe_ep", 1)),
        pp=int(d.get("pp", 1)),
    )


def _replica_from_dict(d: dict[str, Any]) -> ReplicaParallelConfig:
    return ReplicaParallelConfig(shape=_shape_from_dict(d), replicas=int(d.get("replicas", 1)))


def _parse_parallel_entry(entry: dict[str, Any], deployment_mode: str):
    """Parse one pinned ``parallel_configs`` entry into the config object: a flat
    shape dict for agg, or a ``{prefill, decode}`` pair for disagg."""
    if deployment_mode == "agg":
        return _replica_from_dict(entry)
    return DisaggParallelConfig(
        prefill=_replica_from_dict(entry["prefill"]),
        decode=_replica_from_dict(entry["decode"]),
    )


def _pinned_parallel_configs(parallel_configs, deployment_mode, legal):
    """Parse the user's pinned ``parallel_configs`` and keep only those that appear
    in the enumerated ``legal`` set (KV-feasible + shape/backend-legal within budget).
    Raises if any pinned config is not legal/feasible for this branch."""
    legal_set = set(legal)
    pinned = [_parse_parallel_entry(entry, deployment_mode) for entry in parallel_configs]
    illegal = [p for p in pinned if p not in legal_set]
    if illegal:
        raise NoViableParallelConfig(
            f"pinned parallel_configs are not legal/KV-feasible for this branch: {illegal}. "
            "Each pinned shape must satisfy the MoE-width / backend / GPU-ladder rules and "
            "hold a max_seq_len sequence within gpu_budget (same constraints the enumerator applies)."
        )
    return pinned


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
            legal = parallel_configs_for(
                ss.model_name,
                ss.hardware_sku,
                gpu_budget=ss.gpu_budget,
                deployment_mode=deployment_mode,
                backend=backend,
                min_gpu_budget=ss.min_gpu_budget,
                max_seq_len=max_seq_len,
            )
            # A pinned parallel_configs replaces the enumerated menu (validated against
            # it); empty -> use the full enumerated menu. (deployment_mode is pinned to
            # one mode when parallel_configs is set, enforced in SearchSpace.)
            if ss.parallel_configs:
                parallel_configs = _pinned_parallel_configs(ss.parallel_configs, deployment_mode, legal)
            else:
                parallel_configs = list(legal)
            branches.append(
                BranchSpace(
                    deployment_mode=deployment_mode,
                    backend=backend,
                    parallel_configs=tuple(parallel_configs),
                    knob_choices=branch_knob_choices(ss, deployment_mode),
                )
            )
    return branches

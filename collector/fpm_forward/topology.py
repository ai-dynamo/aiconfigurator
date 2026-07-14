# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIC-owned parallel topology enumeration for FPM cells."""

from __future__ import annotations

from aiconfigurator.sdk import common
from aiconfigurator.sdk.utils import enumerate_parallel_config

from .config import FPMCollectionOptions
from .types import ParallelTopology


def _power_sizes(maximum: int) -> list[int]:
    values = []
    value = 1
    while value <= maximum:
        values.append(value)
        value *= 2
    return values


def _axis_sizes(
    options: FPMCollectionOptions,
    axis: str,
    explicit: tuple[int, ...] | None,
) -> list[int]:
    if axis not in options.parallel_axes:
        if explicit not in {None, (1,)}:
            raise ValueError(f"{axis} size filter was supplied but {axis} is not enabled")
        return [1]
    values = list(explicit) if explicit is not None else _power_sizes(options.max_gpus)
    over_limit = [value for value in values if value > options.max_gpus]
    if over_limit:
        raise ValueError(f"{axis} sizes exceed --fpm-max-gpus={options.max_gpus}: {over_limit}")
    return values


def enumerate_fpm_topologies(
    *,
    backend: str,
    is_moe: bool,
    options: FPMCollectionOptions,
) -> tuple[ParallelTopology, ...]:
    """Delegate validity to AIC's shared parallel enumerator."""

    try:
        backend_name = common.BackendName(backend)
    except ValueError as error:
        raise ValueError(f"unsupported AIC backend for FPM topology enumeration: {backend}") from error

    tp = _axis_sizes(options, "tp", options.tp_sizes)
    pp = _axis_sizes(options, "pp", options.pp_sizes)
    dp = _axis_sizes(options, "dp", options.dp_sizes)
    cp = _axis_sizes(options, "cp", options.cp_sizes)
    if is_moe:
        moe_tp = _axis_sizes(options, "moe_tp", options.moe_tp_sizes)
        moe_ep = _axis_sizes(options, "moe_ep", options.moe_ep_sizes)
    else:
        moe_tp = [1]
        moe_ep = [1]

    raw = enumerate_parallel_config(
        num_gpu_list=list(options.gpu_counts),
        tp_list=tp,
        pp_list=pp,
        dp_list=dp,
        moe_tp_list=moe_tp,
        moe_ep_list=moe_ep,
        cp_list=cp,
        is_moe=is_moe,
        backend=backend_name,
    )
    # vLLM has one TP group for dense attention and MoE tensor parallelism.
    # It cannot deploy a separate MoE-TP group across data-parallel replicas;
    # such rows carry dp>1 and moe_tp>1 in the generic AIC search space but
    # render as ordinary DP, so collecting them would mislabel the database.
    if backend_name == common.BackendName.vllm and is_moe:
        raw = [row for row in raw if not (row[2] > 1 and row[3] > 1)]
    topologies = tuple(
        ParallelTopology(tp=row[0], pp=row[1], dp=row[2], moe_tp=row[3], moe_ep=row[4], cp=row[5]) for row in raw
    )
    if not topologies:
        raise ValueError(
            "AIC parallel enumeration produced no valid FPM topology for "
            f"backend={backend}, gpu_counts={list(options.gpu_counts)}, axes={list(options.parallel_axes)}"
        )
    return topologies

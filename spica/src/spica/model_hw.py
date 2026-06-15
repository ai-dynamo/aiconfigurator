# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a (model, hardware SKU) into the facts the parallel enumeration needs.

This is where Spica **reads the model weights** to bound the parallel configs to
shapes that actually hold the model. It directly reuses AIConfigurator:

- ``check_is_moe`` / architecture  -> is_moe, and whether pure expert-TP is allowed
- ``_estimate_model_weight_bytes``  -> model weight size
- ``_get_system_config``            -> the SKU's VRAM / GPUs-per-node
- ``_calculate_min_tp``             -> the memory-fit floor (smallest worker that
                                       holds the weights) and whether it fits the budget

The result feeds :func:`spica.parallel_enum.enumerate_parallel_configs` /
``enumerate_disagg_configs`` as ``min_gpus_per_worker`` + ``mla`` + ``enable_wideep``.
"""

from __future__ import annotations

from dataclasses import dataclass

from aiconfigurator.generator.naive import (
    _calculate_min_tp,
    _estimate_model_weight_bytes,
    _get_system_config,
)
from aiconfigurator.sdk.models import check_is_moe
from aiconfigurator.sdk.utils import get_model_config_from_model_path

from .parallel_enum import (
    DisaggParallelConfig,
    ReplicaParallelConfig,
    enumerate_disagg_configs,
    enumerate_parallel_configs,
)

# GQA+MoE architectures that allow pure expert-TP; others (e.g. MLA) only TEP/DEP.
_GQA_MOE_ARCHITECTURES = frozenset({"Qwen3MoeForCausalLM"})


class NoViableParallelConfig(ValueError):
    """The model cannot fit within the GPU budget at any parallel config."""


@dataclass(frozen=True)
class ModelHardware:
    """Per-(model, hardware, backend) facts that bound the parallel search."""

    model_name: str
    hardware_sku: str
    backend: str
    is_moe: bool
    mla: bool  # MoE that only allows TEP/DEP (no pure expert-TP)
    enable_wideep: bool
    weight_bytes: int
    vram_per_gpu: int
    gpus_per_node: int
    min_gpus_per_worker: int  # memory-fit floor: smallest worker that holds the weights
    fits: bool  # False -> model can't fit within gpu_budget at any config


def resolve_model_hardware(model_name: str, hardware_sku: str, *, gpu_budget: int, backend: str) -> ModelHardware:
    """Read the model weights + SKU spec (via AIC) to derive is_moe / mla /
    wideep and the memory-fit floor for ``gpu_budget``."""
    is_moe = check_is_moe(model_name)
    architecture = get_model_config_from_model_path(model_name).get("architecture", "")
    allow_pure_tp = is_moe and architecture in _GQA_MOE_ARCHITECTURES
    mla = is_moe and not allow_pure_tp

    sys_cfg = _get_system_config(hardware_sku)
    vram_per_gpu = sys_cfg["vram_per_gpu"]
    gpus_per_node = sys_cfg["gpus_per_node"]

    weight_bytes = _estimate_model_weight_bytes(model_name)
    # Large MoE (a node can't hold ~2x the weights) auto-enables multi-node wideEP.
    enable_wideep = is_moe and gpus_per_node * vram_per_gpu < 2 * weight_bytes

    min_gpus, fits, _required_tp = _calculate_min_tp(
        model_weight_bytes=weight_bytes,
        vram_per_gpu=vram_per_gpu,
        gpus_per_node=gpus_per_node,
        total_gpus=gpu_budget,
        allow_multi_node=is_moe and enable_wideep,
    )
    return ModelHardware(
        model_name=model_name,
        hardware_sku=hardware_sku,
        backend=backend,
        is_moe=is_moe,
        mla=mla,
        enable_wideep=enable_wideep,
        weight_bytes=weight_bytes,
        vram_per_gpu=vram_per_gpu,
        gpus_per_node=gpus_per_node,
        min_gpus_per_worker=min_gpus,
        fits=fits,
    )


def parallel_configs_for(
    model_name: str,
    hardware_sku: str,
    *,
    gpu_budget: int,
    deployment_mode: str,
    backend: str,
    min_gpu_budget: int | None = None,
) -> list[ReplicaParallelConfig] | list[DisaggParallelConfig]:
    """Resolve the model/hardware, then enumerate the parallel configs that both
    fit the GPU budget and hold the model weights (memory-fit floor applied).

    ``deployment_mode`` is ``"agg"`` (-> ``list[ReplicaParallelConfig]``) or
    ``"disagg"`` (-> ``list[DisaggParallelConfig]``). Raises
    :class:`NoViableParallelConfig` when the model cannot fit the budget.
    """
    mh = resolve_model_hardware(model_name, hardware_sku, gpu_budget=gpu_budget, backend=backend)
    if not mh.fits:
        raise NoViableParallelConfig(
            f"{model_name} ({mh.weight_bytes / 1024**3:.0f} GiB weights) needs more than "
            f"{gpu_budget} GPUs to fit on {hardware_sku}"
        )

    common = dict(
        is_moe=mh.is_moe,
        backend=backend,
        gpu_budget=gpu_budget,
        min_gpu_budget=min_gpu_budget,
        min_gpus_per_worker=mh.min_gpus_per_worker,
        enable_wideep=mh.enable_wideep,
    )
    if deployment_mode == "disagg":
        return enumerate_disagg_configs(**common)
    if deployment_mode == "agg":
        return enumerate_parallel_configs(**common)
    raise ValueError(f"deployment_mode must be 'agg' or 'disagg', got {deployment_mode!r}")

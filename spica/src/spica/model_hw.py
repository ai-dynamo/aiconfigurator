# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a (model, hardware SKU) into the facts the parallel enumeration needs,
and bound the parallel configs to shapes that actually hold the model.

It directly reuses AIConfigurator:

- ``check_is_moe``                  -> is_moe
- ``_estimate_model_weight_bytes``  -> model weight size (-> wideEP heuristic)
- ``_get_system_config``            -> the SKU's VRAM / GPUs-per-node

Validity is **KV-cache based**: :func:`parallel_configs_for` enumerates shapes
from 1 GPU/worker and keeps a shape iff its estimated KV capacity exceeds the
workload's ``max_seq_len`` (:mod:`spica.kv_estimate`). That is per-shape (TEP /
DEP / TP differ at the same GPU count) and uses the real quantized weights — it
replaces the old BF16 min-GPU weight floor entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

from aiconfigurator.generator.naive import (
    _estimate_model_weight_bytes,
    _get_system_config,
)
from aiconfigurator.sdk.models import check_is_moe
from aiconfigurator.sdk.utils import get_model_config_from_model_path

from .kv_estimate import (
    DEFAULT_MAX_BATCH_SIZE,
    DEFAULT_MAX_NUM_TOKENS,
    DEFAULT_MEMORY_FRACTION,
    feasible_shape_tokens,
)
from .parallel_enum import (
    DisaggParallelConfig,
    ReplicaParallelConfig,
    enumerate_disagg_configs,
    enumerate_parallel_configs,
)

# GQA+MoE architectures that allow pure expert-TP; others (e.g. MLA) only TEP/DEP.
_GQA_MOE_ARCHITECTURES = frozenset({"Qwen3MoeForCausalLM"})


class NoViableParallelConfig(ValueError):
    """No parallel config can hold the model+sequence within the GPU budget."""


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


def resolve_model_hardware(model_name: str, hardware_sku: str, *, backend: str) -> ModelHardware:
    """Read the model weights + SKU spec (via AIC) to derive is_moe / mla / wideep."""
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
    )


def parallel_configs_for(
    model_name: str,
    hardware_sku: str,
    *,
    gpu_budget: int,
    deployment_mode: str,
    backend: str,
    max_seq_len: int,
    min_gpu_budget: int | None = None,
    max_num_tokens: int = DEFAULT_MAX_NUM_TOKENS,
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    memory_fraction: float = DEFAULT_MEMORY_FRACTION,
) -> list[ReplicaParallelConfig] | list[DisaggParallelConfig]:
    """Resolve the model/hardware, then enumerate the parallel configs that fit
    the GPU budget and can hold a ``max_seq_len``-token sequence.

    Validity is **KV-cache based**: shapes are enumerated from 1 GPU/worker and a
    shape is kept iff its estimated KV capacity exceeds ``max_seq_len`` (the
    accurate, per-shape check; see :mod:`spica.kv_estimate`). ``max_num_tokens`` /
    ``max_batch_size`` / ``memory_fraction`` are the runtime knobs the estimate
    reserves around the KV budget.

    ``deployment_mode`` is ``"agg"`` (-> ``list[ReplicaParallelConfig]``) or
    ``"disagg"`` (-> ``list[DisaggParallelConfig]``). Raises
    :class:`NoViableParallelConfig` when no shape can hold the sequence within the
    budget.
    """
    mh = resolve_model_hardware(model_name, hardware_sku, backend=backend)

    # Enumerate from 1 GPU/worker; the KV estimate is the sole feasibility filter.
    common = dict(
        is_moe=mh.is_moe,
        backend=backend,
        gpu_budget=gpu_budget,
        min_gpu_budget=min_gpu_budget,
        enable_wideep=mh.enable_wideep,
    )
    if deployment_mode == "disagg":
        configs = enumerate_disagg_configs(**common)
    elif deployment_mode == "agg":
        configs = enumerate_parallel_configs(**common)
    else:
        raise ValueError(f"deployment_mode must be 'agg' or 'disagg', got {deployment_mode!r}")

    # KV-cache validity: keep configs whose every role-shape holds a max_seq_len sequence.
    if deployment_mode == "agg":
        shapes = [c.shape for c in configs]
    else:
        shapes = [c.prefill.shape for c in configs] + [c.decode.shape for c in configs]
    feasible = feasible_shape_tokens(
        shapes,
        model_name=model_name,
        hardware_sku=hardware_sku,
        backend=backend,
        max_seq_len=max_seq_len,
        max_num_tokens=max_num_tokens,
        max_batch_size=max_batch_size,
        memory_fraction=memory_fraction,
    )
    if deployment_mode == "agg":
        kept = [c for c in configs if c.shape in feasible]
    else:
        kept = [c for c in configs if c.prefill.shape in feasible and c.decode.shape in feasible]
    if not kept:
        raise NoViableParallelConfig(
            f"{model_name} on {hardware_sku}: no parallel config holds a {max_seq_len}-token "
            f"sequence within {gpu_budget} GPUs ({backend} KV-cache estimate)"
        )
    return kept

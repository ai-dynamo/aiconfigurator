# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIC-validated typical parallel topologies for FPM cells."""

from __future__ import annotations

from aiconfigurator.sdk import common
from aiconfigurator.sdk.utils import enumerate_parallel_config

from .config import FPMCollectionOptions
from .types import ParallelTopology


def _requested_presets(options: FPMCollectionOptions, *, is_moe: bool, allow_pure_tp: bool) -> tuple[str, ...]:
    if options.parallel_axes:
        axes = frozenset(options.parallel_axes)
        legacy = {
            frozenset({"tp"}): ("tp",),
            frozenset({"tp", "moe_ep"}): ("tep",),
            frozenset({"dp", "moe_ep"}): ("dep",),
            frozenset({"tp", "moe_tp"}): ("pure_tp",),
            frozenset({"tp", "dp", "moe_tp", "moe_ep"}): ("auto",),
        }.get(axes)
        if legacy is None:
            raise ValueError(
                "legacy --fpm-parallel-axes must describe one typical preset: "
                "tp, tp+moe_ep (TEP), dp+moe_ep (DEP), or tp+moe_tp"
            )
        requested = legacy
    else:
        requested = options.parallel_presets

    if requested == ("auto",):
        if not is_moe:
            return ("tp",)
        return ("pure_tp", "tep", "dep") if allow_pure_tp else ("tep", "dep")
    if not is_moe and any(value != "tp" for value in requested):
        raise ValueError(f"dense models support only the TP typical preset, got {list(requested)}")
    if is_moe and "tp" in requested:
        raise ValueError("MoE models use TEP or DEP; plain TP is not an independent typical preset")
    if "pure_tp" in requested and not allow_pure_tp:
        raise ValueError(
            "pure TP was requested, but the AIC model/runtime capability profile does not explicitly admit it"
        )
    return requested


_PRESET_AXES = {
    "tp": frozenset({"tp"}),
    "tep": frozenset({"tp", "moe_ep"}),
    "dep": frozenset({"dp", "moe_ep"}),
    "pure_tp": frozenset({"tp", "moe_tp"}),
}


def _axis_filters(options: FPMCollectionOptions) -> dict[str, tuple[int, ...] | None]:
    return {
        "tp": options.tp_sizes,
        "dp": options.dp_sizes,
        "moe_tp": options.moe_tp_sizes,
        "moe_ep": options.moe_ep_sizes,
    }


def _validate_filters(options: FPMCollectionOptions, presets: tuple[str, ...]) -> None:
    active_axes = set().union(*(_PRESET_AXES[preset] for preset in presets))
    for axis, values in _axis_filters(options).items():
        if values is not None and axis not in active_axes and values != (1,):
            raise ValueError(f"{axis} size filter does not apply to selected presets {list(presets)}")
        if values is not None:
            over_limit = [value for value in values if value > options.max_gpus]
            if over_limit:
                raise ValueError(f"{axis} sizes exceed --fpm-max-gpus={options.max_gpus}: {over_limit}")


def _width_allowed(options: FPMCollectionOptions, preset: str, width: int) -> bool:
    filters = _axis_filters(options)
    return all(filters[axis] is None or width in filters[axis] for axis in _PRESET_AXES[preset])


def _preset_values(preset: str, width: int) -> tuple[int, int, int, int, int, int]:
    if preset == "tp":
        return width, 1, 1, 1, 1, 1
    if preset == "tep":
        return width, 1, 1, 1, width, 1
    if preset == "dep":
        return 1, 1, width, 1, width, 1
    if preset == "pure_tp":
        return width, 1, 1, width, 1, 1
    raise AssertionError(f"unhandled FPM parallel preset: {preset}")


def topology_strategy(topology: ParallelTopology, *, is_moe: bool) -> str:
    if topology.total_gpus == 1:
        return "single"
    if not is_moe:
        return "tp"
    if topology.tp == topology.moe_ep and topology.dp == topology.moe_tp == 1:
        return "tep"
    if topology.dp == topology.moe_ep and topology.tp == topology.moe_tp == 1:
        return "dep"
    if topology.tp == topology.moe_tp and topology.dp == topology.moe_ep == 1:
        return "pure_tp"
    raise ValueError(f"topology is not a recognized typical FPM preset: {topology.to_dict()}")


def enumerate_fpm_topologies(
    *,
    backend: str,
    is_moe: bool,
    options: FPMCollectionOptions,
    allow_pure_tp: bool = False,
) -> tuple[ParallelTopology, ...]:
    """Generate semantic presets, then ask AIC to admit each exact tuple."""

    try:
        backend_name = common.BackendName(backend)
    except ValueError as error:
        raise ValueError(f"unsupported AIC backend for FPM topology enumeration: {backend}") from error

    presets = _requested_presets(options, is_moe=is_moe, allow_pure_tp=allow_pure_tp)
    _validate_filters(options, presets)
    admitted: dict[tuple[int, int, int, int, int, int], ParallelTopology] = {}
    for width in options.gpu_counts:
        for preset in presets:
            if not _width_allowed(options, preset, width):
                continue
            tp, pp, dp, moe_tp, moe_ep, cp = _preset_values(preset, width)
            raw = enumerate_parallel_config(
                num_gpu_list=[width],
                tp_list=[tp],
                pp_list=[pp],
                dp_list=[dp],
                moe_tp_list=[moe_tp],
                moe_ep_list=[moe_ep],
                cp_list=[cp],
                is_moe=is_moe,
                backend=backend_name,
            )
            for row in raw:
                key = tuple(int(value) for value in row)
                admitted[key] = ParallelTopology(
                    tp=key[0], pp=key[1], dp=key[2], moe_tp=key[3], moe_ep=key[4], cp=key[5]
                )

    topologies = tuple(
        sorted(
            admitted.values(),
            key=lambda item: (item.total_gpus, topology_strategy(item, is_moe=is_moe), tuple(item.to_dict().values())),
        )
    )
    if not topologies:
        raise ValueError(
            "AIC parallel enumeration admitted no typical FPM topology for "
            f"backend={backend}, gpu_counts={list(options.gpu_counts)}, presets={list(presets)}"
        )
    return topologies

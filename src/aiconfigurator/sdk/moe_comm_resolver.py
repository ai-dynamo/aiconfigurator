# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared data-driven MoE communication resolution for exact model configs."""

from __future__ import annotations

from collections.abc import Mapping, Set
from typing import Any

from aiconfigurator.sdk import common
from aiconfigurator.sdk.config import ModelConfig
from aiconfigurator.sdk.models import (
    _architecture_to_model_family,
    _get_model_info,
    check_is_moe,
)
from aiconfigurator.sdk.models.blocks.moe import LARGE_EP_READY_FAMILIES, MoEBlockShape
from aiconfigurator.sdk.models.helpers import _apply_model_quant_defaults
from aiconfigurator.sdk.operations.moe_comm import MOE_A2A_BACKENDS, nodes_for
from aiconfigurator.sdk.perf_database import PerfDataNotAvailableError

_DEEPEP_NODE1_FALLBACK_BACKENDS = frozenset(("deepep_ht", "deepep_ll"))
_LEGACY_DEEPEP_NODE1_COORDINATE = (8, 1)

LargeEpCoverage = Mapping[str, Mapping[str, Set[int]]]


def a2a_covers_parallel(
    pairs: set[tuple[int, int]],
    *,
    framework: str,
    comm_backend: str,
    moe_ep_size: int,
    expected_nodes: int,
) -> bool:
    """Whether A2A data can serve a target EP/node scale.

    SGLang DeepEP preserves the marked node-1 substitution introduced by
    PR #1314: prefer an exact scale, otherwise let the canonical legacy
    ``(ep=8, node_num=1)`` row for the already shape-filtered coverage probe
    represent a multi-node request. The Rust legacy adapter uses the same
    coordinate and marks that substitution as estimated. Other frameworks
    and communication backends remain exact-scale only.
    """
    if (moe_ep_size, expected_nodes) in pairs:
        return True
    return (
        framework == "sglang"
        and comm_backend in _DEEPEP_NODE1_FALLBACK_BACKENDS
        and expected_nodes > 1
        and _LEGACY_DEEPEP_NODE1_COORDINATE in pairs
    )


def select_moe_comm_backend(
    coverage: LargeEpCoverage,
    *,
    backend_name: str,
    attention_dp_size: int,
    moe_ep_size: int,
) -> dict[str, str]:
    """Select one communication backend for every covered phase at one EP."""
    if backend_name == "trtllm" and attention_dp_size <= 1:
        return {}

    resolved: dict[str, str] = {}
    for phase, per_backend in coverage.items():
        for comm_backend, eps in per_backend.items():
            if moe_ep_size in eps:
                resolved[phase] = comm_backend
                break
    return resolved


def resolve_model_config_moe_comm(
    model_config: ModelConfig,
    *,
    model_path: str,
    backend_name: str,
    database: Any,
    required_phases: tuple[str, ...],
    fmha_quant_mode_explicit: bool = False,
    kvcache_quant_mode_explicit: bool = False,
    coverage_snapshot: LargeEpCoverage | None = None,
) -> dict[str, str] | None:
    """Resolve DeepEP data, requiring it when EP spans physical nodes."""
    if not check_is_moe(model_path):
        return None

    moe_tp_size = int(model_config.moe_tp_size or 1)
    moe_ep_size = int(model_config.moe_ep_size or 1)
    if moe_ep_size <= 1:
        return None

    gpus_per_node = 0
    if database is not None and hasattr(database, "system_spec"):
        gpus_per_node = int(database.system_spec.get("node", {}).get("num_gpus_per_node", 0) or 0)
    if not gpus_per_node:
        gpus_per_node = int(model_config.num_gpus_per_node or 0)
    if not gpus_per_node:
        raise ValueError("Cannot resolve MoE communication: num_gpus_per_node is missing from the system spec.")
    model_config.num_gpus_per_node = gpus_per_node

    cross_node = moe_ep_size > gpus_per_node
    if not cross_node and coverage_snapshot is None:
        # Exact CLI estimates historically keep intra-node EP on the fused
        # path. Task supplies its cached coverage snapshot and can still
        # resolve an explicitly covered intra-node tuple.
        return None
    if moe_tp_size != 1:
        if cross_node:
            raise ValueError(
                "Cross-node EP requires pure expert parallelism "
                f"(moe_tp_size=1); got moe_tp_size={moe_tp_size}, moe_ep_size={moe_ep_size}."
            )
        return None

    info = _get_model_info(model_path)
    _apply_model_quant_defaults(
        model_config,
        info.get("raw_config", {}),
        info["architecture"],
        backend_name,
        worker_name=model_path,
    )
    family = _architecture_to_model_family(info["architecture"])
    shape = MoEBlockShape.from_model_info(info)
    sm_version = (
        database.system_spec.get("gpu", {}).get("sm_version")
        if database is not None and hasattr(database, "system_spec")
        else None
    )
    sm_version = int(sm_version) if sm_version is not None else None
    expected_nodes = nodes_for(moe_ep_size, gpus_per_node)

    resolved: dict[str, str]
    if coverage_snapshot is not None:
        resolved = select_moe_comm_backend(
            coverage_snapshot,
            backend_name=backend_name,
            attention_dp_size=int(model_config.attention_dp_size or 1),
            moe_ep_size=moe_ep_size,
        )
    else:
        coverage: dict[str, dict[str, set[int]]] = {}
        resolved = {}
    if coverage_snapshot is None and family in LARGE_EP_READY_FAMILIES and database is not None:
        a2a_probe = getattr(database, "moe_a2a_coverage", None)
        compute_probe = getattr(database, "moe_expert_compute_coverage", None)
        for phase in dict.fromkeys(required_phases):
            a2a = (
                a2a_probe(
                    shape.hidden_size,
                    shape.topk,
                    shape.num_experts,
                    model_config.moe_quant_mode,
                    phase,
                )
                if a2a_probe is not None
                else {}
            )
            compute_eps = (
                compute_probe(
                    shape.hidden_size,
                    shape.moe_inter_size,
                    shape.topk,
                    shape.num_experts,
                    model_config.moe_quant_mode,
                    phase,
                )
                if compute_probe is not None
                else {moe_ep_size}
            )
            for comm_backend, backend_spec in MOE_A2A_BACKENDS.items():
                if backend_name not in backend_spec.frameworks or phase not in backend_spec.inference_phases:
                    continue
                if backend_name == "trtllm" and int(model_config.attention_dp_size or 1) <= 1:
                    # Keep parity with Task._resolve_moe_comm_backend: TRT-LLM
                    # large EP requires attention DP. Do not return early;
                    # cross-node EP must reach the missing-coverage error.
                    continue
                if (
                    a2a_covers_parallel(
                        a2a.get(comm_backend, set()),
                        framework=backend_name,
                        comm_backend=comm_backend,
                        moe_ep_size=moe_ep_size,
                        expected_nodes=expected_nodes,
                    )
                    and moe_ep_size in compute_eps
                    and backend_spec.feasible(
                        topk=shape.topk,
                        num_experts=shape.num_experts,
                        moe_tp_size=moe_tp_size,
                        moe_ep_size=moe_ep_size,
                        sm_version=sm_version,
                    )
                ):
                    coverage.setdefault(phase, {}).setdefault(comm_backend, set()).add(moe_ep_size)
                    break
        resolved = select_moe_comm_backend(
            coverage,
            backend_name=backend_name,
            attention_dp_size=int(model_config.attention_dp_size or 1),
            moe_ep_size=moe_ep_size,
        )

    missing = set(required_phases) - set(resolved)
    if missing:
        if cross_node:
            system_name = getattr(database, "system", "") if database is not None else ""
            version = getattr(database, "version", "") if database is not None else ""
            raise PerfDataNotAvailableError(
                "Cross-node EP requires DeepEP A2A data, but no compatible exact or supported node-1 coverage "
                "was found for "
                f"model={model_path!r}, system={system_name!r}, backend={backend_name!r}, version={version!r}, "
                f"moe_ep={moe_ep_size}, gpus_per_node={gpus_per_node}, "
                f"phase(s)={','.join(sorted(missing))}."
            )
        return None

    model_config.moe_comm_backend = resolved
    if database is not None:
        model_config.system = getattr(database, "system", None)
    if backend_name == "sglang" and info["architecture"] == "DeepseekV3ForCausalLM":
        # The SGLang WideEP MLA collectors label their rows fp8_block/fp8.
        # Preserve explicit user modes, but restore the PR #1314 defaults for
        # inferred DeepSeek-V3/R1 configs once this tuple selects DeepEP.
        if not fmha_quant_mode_explicit:
            model_config.fmha_quant_mode = common.FMHAQuantMode.fp8_block
        if not kvcache_quant_mode_explicit:
            model_config.kvcache_quant_mode = common.KVCacheQuantMode.fp8
    return resolved

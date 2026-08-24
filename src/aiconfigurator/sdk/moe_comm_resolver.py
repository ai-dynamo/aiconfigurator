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

# These vLLM imports were measured on one NVLink domain at EP=4. Until
# multi-node measurements land, their reviewed dataset policy reuses that
# exact curve for cross-node EP configurations without rescaling its token
# axis. Keep this allow-list scoped to the physical dataset identity so an
# incomplete table can never opt into the approximation accidentally.
_UNSCALED_SINGLE_NODE_PROXY_DATASETS = frozenset(
    {
        ("gb200", "vllm", "0.24.0"),
        ("gb300", "vllm", "0.24.0"),
    }
)

LargeEpCoverage = Mapping[str, Mapping[str, Set[int]]]


def resolve_a2a_query_profile(
    pairs: set[tuple[int, int]],
    *,
    framework: str,
    comm_backend: str,
    moe_ep_size: int,
    expected_nodes: int,
    dataset_identity: tuple[str, str, str] | None = None,
) -> tuple[int, int] | None:
    """Return the measured ``(ep, node)`` coordinate serving a request.

    Exact coverage always wins. SGLang keeps the legacy DeepEP EP8/node-1
    fallback from #1578. The only other substitution is the explicitly
    allow-listed, unscaled vLLM EP4/node-1 dataset policy above.
    """
    exact = (moe_ep_size, expected_nodes)
    if exact in pairs:
        return exact
    if (
        framework == "sglang"
        and comm_backend in _DEEPEP_NODE1_FALLBACK_BACKENDS
        and expected_nodes > 1
        and _LEGACY_DEEPEP_NODE1_COORDINATE in pairs
    ):
        return _LEGACY_DEEPEP_NODE1_COORDINATE
    if dataset_identity in _UNSCALED_SINGLE_NODE_PROXY_DATASETS and expected_nodes > 1:
        donors = sorted((ep, node_num) for ep, node_num in pairs if node_num == 1)
        if donors:
            return donors[-1]
    return None


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
    return (
        resolve_a2a_query_profile(
            pairs,
            framework=framework,
            comm_backend=comm_backend,
            moe_ep_size=moe_ep_size,
            expected_nodes=expected_nodes,
        )
        is not None
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
    query_profiles: dict[str, tuple[int, int]] = {}
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
        a2a = a2a_probe(shape.hidden_size, shape.topk, shape.num_experts) if a2a_probe is not None else {}
        dataset_identity = (
            str(getattr(database, "system", getattr(model_config, "system", "") or "")),
            str(getattr(database, "backend", backend_name)),
            str(getattr(database, "version", "")),
        )
        for phase in dict.fromkeys(required_phases):
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
                query_profile = resolve_a2a_query_profile(
                    a2a.get(comm_backend, set()),
                    framework=backend_name,
                    comm_backend=comm_backend,
                    moe_ep_size=moe_ep_size,
                    expected_nodes=expected_nodes,
                    dataset_identity=dataset_identity,
                )
                if (
                    query_profile is not None
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
                    query_profiles[phase] = query_profile
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
                "Cross-node EP requires compatible A2A and expert-compute data, but no exact or approved "
                "proxy coverage "
                "was found for "
                f"model={model_path!r}, system={system_name!r}, backend={backend_name!r}, version={version!r}, "
                f"moe_ep={moe_ep_size}, gpus_per_node={gpus_per_node}, "
                f"phase(s)={','.join(sorted(missing))}."
            )
        return None

    model_config.moe_comm_backend = resolved
    model_config.moe_comm_query_profile = query_profiles or None
    if backend_name == "sglang" and info["architecture"] == "DeepseekV3ForCausalLM":
        # The SGLang WideEP MLA collectors label their rows fp8_block/fp8.
        # Preserve explicit user modes, but restore the PR #1314 defaults for
        # inferred DeepSeek-V3/R1 configs once this tuple selects DeepEP.
        if not fmha_quant_mode_explicit:
            model_config.fmha_quant_mode = common.FMHAQuantMode.fp8_block
        if not kvcache_quant_mode_explicit:
            model_config.kvcache_quant_mode = common.KVCacheQuantMode.fp8
    return resolved

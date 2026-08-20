# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage-gated selection of measured large-EP MoE communication paths."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from aiconfigurator.sdk.models import _get_model_info, get_model_family
from aiconfigurator.sdk.models.blocks.moe import LARGE_EP_READY_FAMILIES, MoEBlockShape
from aiconfigurator.sdk.operations.moe_comm import MOE_A2A_BACKENDS, nodes_for
from aiconfigurator.sdk.perf_database import load_system_spec

logger = logging.getLogger(__name__)

ParallelChoice = tuple[int, int, int, int, int, int]
A2AProfile = tuple[int, int]

# These two imports were collected on one NVLink domain at EP=4.  Until
# multi-node measurements land, the approved approximation is to reuse that
# exact curve for cross-node EP configurations without changing its token
# axis.  Keep the allow-list at the dataset identity (rather than silently
# applying the fallback to every incomplete table) so the approximation is
# explicit and reviewable.
UNSCALED_SINGLE_NODE_PROXY_DATASETS = frozenset(
    {
        ("gb200", "vllm", "0.24.0"),
        ("gb300", "vllm", "0.24.0"),
    }
)


@dataclass(frozen=True)
class LargeEPCoverage:
    """Measured A2A coverage and the topology/shape used to derive it."""

    phases: dict[str, dict[str, set[int]]]
    query_profiles: dict[str, dict[str, dict[int, A2AProfile]]]
    num_gpus_per_node: int | None
    shape: MoEBlockShape | None


def compute_large_ep_coverage(
    *,
    model_path: str,
    model_family: str,
    system_name: str,
    backend_name: str,
    database,
) -> LargeEPCoverage:
    """Return measured A2A coverage plus explicitly approved proxy profiles."""
    spec = load_system_spec(system_name)
    gpus_per_node = int(spec.get("node", {}).get("num_gpus_per_node", 0) or 0)
    num_gpus_per_node = gpus_per_node or None
    if model_family not in LARGE_EP_READY_FAMILIES:
        return LargeEPCoverage({}, {}, num_gpus_per_node, None)

    try:
        shape = MoEBlockShape.from_model_info(_get_model_info(model_path))
    except Exception as exc:  # not a MoE checkpoint / unparsable config
        logger.debug("large-EP coverage: no MoE shape for %s: %s", model_path, exc)
        return LargeEPCoverage({}, {}, num_gpus_per_node, None)

    sm_version = spec.get("gpu", {}).get("sm_version")
    sm_version = int(sm_version) if sm_version is not None else None
    a2a_probe = getattr(database, "moe_a2a_coverage", None)
    coverage: dict[str, dict[str, set[int]]] = {}
    query_profiles: dict[str, dict[str, dict[int, A2AProfile]]] = {}
    dataset_identity = (
        str(getattr(database, "system", system_name)),
        str(getattr(database, "backend", backend_name)),
        str(getattr(database, "version", "")),
    )
    allow_unscaled_proxy = dataset_identity in UNSCALED_SINGLE_NODE_PROXY_DATASETS
    if gpus_per_node and a2a_probe is not None:
        a2a = a2a_probe(shape.hidden_size, shape.topk, shape.num_experts)
        for phase in ("context", "generation"):
            per_backend: dict[str, set[int]] = {}
            per_backend_profiles: dict[str, dict[int, A2AProfile]] = {}
            for name, backend_spec in MOE_A2A_BACKENDS.items():
                if backend_name not in backend_spec.frameworks or phase not in backend_spec.inference_phases:
                    continue
                measured_pairs = a2a.get(name, ())
                profiles = {
                    ep: (ep, node_num)
                    for ep, node_num in measured_pairs
                    if node_num == nodes_for(ep, gpus_per_node)
                    and backend_spec.feasible(
                        topk=shape.topk,
                        num_experts=shape.num_experts,
                        moe_tp_size=1,
                        moe_ep_size=ep,
                        sm_version=sm_version,
                    )
                }
                if allow_unscaled_proxy:
                    single_node_donors = sorted(
                        (ep, node_num)
                        for ep, node_num in measured_pairs
                        if node_num == 1
                        and backend_spec.feasible(
                            topk=shape.topk,
                            num_experts=shape.num_experts,
                            moe_tp_size=1,
                            moe_ep_size=ep,
                            sm_version=sm_version,
                        )
                    )
                    if single_node_donors:
                        donor = single_node_donors[-1]
                        for target_ep in range(2, shape.num_experts + 1):
                            if target_ep in profiles or nodes_for(target_ep, gpus_per_node) <= 1:
                                continue
                            if backend_spec.feasible(
                                topk=shape.topk,
                                num_experts=shape.num_experts,
                                moe_tp_size=1,
                                moe_ep_size=target_ep,
                                sm_version=sm_version,
                            ):
                                profiles[target_ep] = donor
                if profiles:
                    per_backend[name] = set(profiles)
                    per_backend_profiles[name] = profiles
            if per_backend:
                coverage[phase] = per_backend
                query_profiles[phase] = per_backend_profiles
    return LargeEPCoverage(coverage, query_profiles, num_gpus_per_node, shape)


def resolve_moe_comm_query_profiles(
    *,
    coverage: LargeEPCoverage,
    resolved_backends: dict[str, str] | None,
    moe_ep_size: int,
) -> dict[str, A2AProfile] | None:
    """Return the measured ``(ep, node)`` keys used by a resolved config."""
    if resolved_backends is None:
        return None
    return {phase: coverage.query_profiles[phase][backend][moe_ep_size] for phase, backend in resolved_backends.items()}


def resolve_moe_comm_backend(
    *,
    coverage: dict[str, dict[str, set[int]]],
    backend_name: str,
    parallel: ParallelChoice,
    required_phases: tuple[str, ...],
) -> dict[str, str] | None:
    """Resolve one explicit parallel tuple, falling back to fused when uncovered."""
    _tp, _pp, attention_dp, moe_tp, moe_ep, _cp = parallel
    if moe_tp != 1 or moe_ep <= 1:
        return None
    if backend_name == "trtllm" and attention_dp <= 1:
        return None

    resolved: dict[str, str] = {}
    for phase in ("context", "generation"):
        for name, eps in coverage.get(phase, {}).items():
            if moe_ep in eps:
                resolved[phase] = name
                break
    if set(required_phases) - set(resolved):
        return None
    return resolved


def resolve_model_config_moe_comm(
    model_config,
    *,
    model_path: str,
    system_name: str,
    backend_name: str,
    database,
    required_phases: tuple[str, ...] = ("context", "generation"),
) -> dict[str, str] | None:
    """Inject the measured large-EP backend/topology into an explicit config."""
    result = compute_large_ep_coverage(
        model_path=model_path,
        model_family=get_model_family(model_path),
        system_name=system_name,
        backend_name=backend_name,
        database=database,
    )
    model_config.num_gpus_per_node = result.num_gpus_per_node
    parallel = (
        model_config.tp_size,
        model_config.pp_size,
        model_config.attention_dp_size,
        model_config.moe_tp_size,
        model_config.moe_ep_size,
        model_config.cp_size,
    )
    model_config.moe_comm_backend = resolve_moe_comm_backend(
        coverage=result.phases,
        backend_name=backend_name,
        parallel=parallel,
        required_phases=required_phases,
    )
    model_config.moe_comm_query_profile = resolve_moe_comm_query_profiles(
        coverage=result,
        resolved_backends=model_config.moe_comm_backend,
        moe_ep_size=model_config.moe_ep_size,
    )
    if model_config.moe_comm_backend is not None:
        logger.info(
            "Resolved coverage-gated large-EP MoE communication for %s on %s/%s "
            "(moe_ep=%d, gpus_per_node=%d): backends=%s, query_profiles=%s",
            model_path,
            system_name,
            backend_name,
            model_config.moe_ep_size,
            result.num_gpus_per_node,
            model_config.moe_comm_backend,
            model_config.moe_comm_query_profile,
        )
    return model_config.moe_comm_backend

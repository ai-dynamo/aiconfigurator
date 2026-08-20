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


@dataclass(frozen=True)
class LargeEPCoverage:
    """Measured A2A coverage and the topology/shape used to derive it."""

    phases: dict[str, dict[str, set[int]]]
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
    """Return exact, topology-matched A2A coverage for a model deployment."""
    spec = load_system_spec(system_name)
    gpus_per_node = int(spec.get("node", {}).get("num_gpus_per_node", 0) or 0)
    num_gpus_per_node = gpus_per_node or None
    if model_family not in LARGE_EP_READY_FAMILIES:
        return LargeEPCoverage({}, num_gpus_per_node, None)

    try:
        shape = MoEBlockShape.from_model_info(_get_model_info(model_path))
    except Exception as exc:  # not a MoE checkpoint / unparsable config
        logger.debug("large-EP coverage: no MoE shape for %s: %s", model_path, exc)
        return LargeEPCoverage({}, num_gpus_per_node, None)

    sm_version = spec.get("gpu", {}).get("sm_version")
    sm_version = int(sm_version) if sm_version is not None else None
    a2a_probe = getattr(database, "moe_a2a_coverage", None)
    coverage: dict[str, dict[str, set[int]]] = {}
    if gpus_per_node and a2a_probe is not None:
        a2a = a2a_probe(shape.hidden_size, shape.topk, shape.num_experts)
        for phase in ("context", "generation"):
            per_backend: dict[str, set[int]] = {}
            for name, backend_spec in MOE_A2A_BACKENDS.items():
                if backend_name not in backend_spec.frameworks or phase not in backend_spec.inference_phases:
                    continue
                eps = {
                    ep
                    for ep, node_num in a2a.get(name, ())
                    if node_num == nodes_for(ep, gpus_per_node)
                    and backend_spec.feasible(
                        topk=shape.topk,
                        num_experts=shape.num_experts,
                        moe_tp_size=1,
                        moe_ep_size=ep,
                        sm_version=sm_version,
                    )
                }
                if eps:
                    per_backend[name] = eps
            if per_backend:
                coverage[phase] = per_backend
    return LargeEPCoverage(coverage, num_gpus_per_node, shape)


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
    if model_config.moe_comm_backend is not None:
        logger.info(
            "Resolved coverage-gated large-EP MoE communication for %s on %s/%s (moe_ep=%d, gpus_per_node=%d): %s",
            model_path,
            system_name,
            backend_name,
            model_config.moe_ep_size,
            result.num_gpus_per_node,
            model_config.moe_comm_backend,
        )
    return model_config.moe_comm_backend

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build an immutable cell matrix for Dynamo-native FPM self-benchmarks."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .capabilities import ModelCapabilityProfile, ResolvedDTypeProfile, resolve_model_capability
from .config import FPMCollectionOptions
from .memory_admission import TopologyMemoryDecision, filter_memory_infeasible_topologies
from .topology import enumerate_fpm_topologies, topology_strategy
from .types import ParallelTopology


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_revision() -> str:
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


@dataclass(frozen=True, slots=True)
class BackendPolicy:
    axis: str
    policy_id: str
    generator_overrides: dict[str, Any]
    expected_markers: dict[str, str]
    aic_fields: dict[str, object] = field(default_factory=dict)
    admission_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "axis": self.axis,
            "policy_id": self.policy_id,
            "generator_overrides": self.generator_overrides,
            "expected_markers": self.expected_markers,
            "aic_fields": self.aic_fields,
            "admission_reason": self.admission_reason,
        }


def _backend_policies(
    options: FPMCollectionOptions,
    collector_config: dict[str, Any],
    *,
    backend: str,
) -> tuple[BackendPolicy, ...]:
    declarations = collector_config.get("backend_variants", {})
    if not isinstance(declarations, dict):
        raise TypeError("FpmCollector.backend_variants must be a mapping")
    if declarations:
        raise ValueError(
            "FpmCollector.backend_variants is no longer an admission mechanism; backend policies must come from "
            "AIC structured capabilities"
        )
    if "auto" in options.backend_axes and len(options.backend_axes) > 1:
        raise ValueError("backend axis 'auto' cannot be combined with explicit backend axes")
    requested = {"baseline"} if options.backend_axes == ("auto",) else set(options.backend_axes)
    unsupported = requested - {"baseline"}
    if unsupported:
        raise ValueError(
            f"AIC exposes no structured {backend} FPM backend variant for axes {sorted(unsupported)}; "
            "arbitrary runtime overrides are not treated as modeled support"
        )
    return (
        BackendPolicy(
            "baseline",
            "baseline_auto",
            {},
            {},
            {
                "moe_backend": None,
                "attention_backend": None,
                "enable_wideep": False,
                "enable_eplb": False,
            },
            "AIC automatic baseline for the selected model/backend",
        ),
    )


@dataclass(frozen=True, slots=True)
class FPMCell:
    cell_id: str
    workload_kind: str
    topology: ParallelTopology
    weight_quantization: str
    kv_cache_dtype: str
    backend_policy: BackendPolicy
    parallel_strategy: str = "unspecified"
    gemm_quant_mode: str | None = None
    moe_quant_mode: str | None = None
    fmha_quant_mode: str | None = None
    comm_quant_mode: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "workload_kind": self.workload_kind,
            "point_source": "dynamo_native_self_benchmark",
            "topology": self.topology.to_dict(),
            "parallel_strategy": self.parallel_strategy,
            "weight_quantization": self.weight_quantization,
            "kv_cache_dtype": self.kv_cache_dtype,
            "resolved_dtypes": {
                "gemm_quant_mode": self.gemm_quant_mode or self.weight_quantization,
                "moe_quant_mode": self.moe_quant_mode,
                "fmha_quant_mode": self.fmha_quant_mode,
                "comm_quant_mode": self.comm_quant_mode,
                "kvcache_quant_mode": self.kv_cache_dtype,
            },
            "backend_policy": self.backend_policy.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class FPMCollectionPlan:
    backend: str
    model_path: str
    system: str
    aic_revision: str
    generator_config_sha256: str
    options: FPMCollectionOptions
    capability: ModelCapabilityProfile
    dtype_profile: ResolvedDTypeProfile
    topologies: tuple[ParallelTopology, ...]
    topology_memory_admission: tuple[TopologyMemoryDecision, ...]
    backend_policies: tuple[BackendPolicy, ...]
    cells: tuple[FPMCell, ...]
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_name": "aic_fpm_collection_plan",
            "schema_version": 9,
            "backend": self.backend,
            "model_path": self.model_path,
            "system": self.system,
            "aic_revision": self.aic_revision,
            "generator_config_sha256": self.generator_config_sha256,
            "options": self.options.to_dict(),
            "capability": self.capability.to_dict(),
            "dtype_profile": self.dtype_profile.to_dict(),
            "point_generation": {
                "owner": "dynamo.vllm.instrumented_scheduler.InstrumentedScheduler",
                "method": "native_self_benchmark",
                "coordinates": [
                    "batch_size",
                    "total_prefill_tokens",
                    "total_kv_read_tokens",
                ],
                "partition_policy": "balanced_v1",
                "point_admission": "dynamo_live_scheduler",
                "precondition": "vllm_engine_initialized",
                "prefill_sampling": self.options.prefill_sampling.to_dict(),
                "planned_point_count": None,
            },
            "topologies": [
                {
                    **topology.to_dict(),
                    "strategy": topology_strategy(topology, is_moe=self.capability.is_moe),
                }
                for topology in self.topologies
            ],
            "topology_memory_admission": [decision.to_dict() for decision in self.topology_memory_admission],
            "backend_policies": [policy.to_dict() for policy in self.backend_policies],
            "cells": [cell.to_dict() for cell in self.cells],
            "counts": {
                "candidate_topologies": len(self.topology_memory_admission),
                "topologies": len(self.topologies),
                "memory_rejected_topologies": sum(
                    decision.disposition == "rejected" for decision in self.topology_memory_admission
                ),
                "memory_unknown_topologies": sum(
                    decision.disposition == "unknown" for decision in self.topology_memory_admission
                ),
                "backend_policies": len(self.backend_policies),
                "cells": len(self.cells),
                "prefill_cudagraph_capture_sizes": len(self.options.prefill_sampling.cudagraph_capture_sizes),
                "prefill_new_token_axis_points": len(self.options.prefill_sampling.new_token_axis_points),
                "points": "runtime-determined",
            },
            "sha256": self.sha256,
        }


def _cell_id(
    *,
    backend: str,
    model_path: str,
    system: str,
    phase: str,
    topology: ParallelTopology,
    weight_quantization: str,
    kv_cache_dtype: str,
    policy: BackendPolicy,
) -> str:
    payload = {
        "backend": backend,
        "model_path": model_path,
        "system": system,
        "phase": phase,
        "topology": topology.to_dict(),
        "weight_quantization": weight_quantization,
        "kv_cache_dtype": kv_cache_dtype,
        "backend_axis": policy.axis,
        "backend_policy": policy.policy_id,
        "point_source": "dynamo_native_self_benchmark",
    }
    return f"fpm-{_canonical_hash(payload)[:16]}"


def build_collection_plan(
    *,
    backend: str,
    model_path: str,
    system: str,
    selected_ops: set[str],
    options: FPMCollectionOptions,
    model_architecture: str | None = None,
    has_model_cases: bool = True,
    collector_config: dict[str, Any] | None = None,
    generator_overrides: dict[str, Any] | None = None,
) -> FPMCollectionPlan:
    if backend != "vllm":
        raise ValueError("FPM Generator V1 currently supports only backend=vllm")
    collector_config = collector_config or {}
    generator_config_sha256 = _canonical_hash(generator_overrides or {})
    capability = resolve_model_capability(
        backend=backend,
        model_path=model_path,
        model_architecture=model_architecture,
        selected_ops=selected_ops,
        has_model_cases=has_model_cases,
        system=system,
        requested_weight_quantizations=options.weight_quantizations,
        requested_kv_cache_dtypes=options.kv_cache_dtypes,
        database_version=(
            str(collector_config["aic_database_version"]) if "aic_database_version" in collector_config else None
        ),
    )
    candidate_topologies = enumerate_fpm_topologies(
        backend=backend,
        is_moe=capability.is_moe,
        options=options,
        allow_pure_tp=capability.allow_pure_tp,
    )
    topologies, topology_memory_admission = filter_memory_infeasible_topologies(
        backend=backend,
        model_path=model_path,
        system=system,
        capability=capability,
        topologies=candidate_topologies,
        max_new_tokens=options.prefill_sampling.max_total_prefill_tokens,
    )
    policies = _backend_policies(options, collector_config, backend=backend)
    weight_quantization = capability.dtype.gemm_quant_mode
    cells = tuple(
        FPMCell(
            cell_id=_cell_id(
                backend=backend,
                model_path=model_path,
                system=system,
                phase=phase,
                topology=topology,
                weight_quantization=weight_quantization,
                kv_cache_dtype=kv_cache_dtype,
                policy=policy,
            ),
            workload_kind=phase,
            topology=topology,
            weight_quantization=weight_quantization,
            kv_cache_dtype=kv_cache_dtype,
            backend_policy=policy,
            parallel_strategy=topology_strategy(topology, is_moe=capability.is_moe),
            gemm_quant_mode=capability.dtype.gemm_quant_mode,
            moe_quant_mode=capability.dtype.moe_quant_mode,
            fmha_quant_mode=capability.dtype.fmha_quant_mode,
            comm_quant_mode=capability.dtype.comm_quant_mode,
        )
        for phase in ("prefill", "decode")
        for topology in topologies
        for kv_cache_dtype in capability.dtype.kv_cache_dtypes
        for policy in policies
    )
    revision = _git_revision()
    canonical = {
        "backend": backend,
        "model_path": model_path,
        "system": system,
        "aic_revision": revision,
        "generator_config_sha256": generator_config_sha256,
        "options": options.to_dict(),
        "capability": capability.to_dict(),
        "dtype_profile": capability.dtype.to_dict(),
        "point_generation": "dynamo_native_self_benchmark",
        "topology_memory_admission": [decision.to_dict() for decision in topology_memory_admission],
        "topologies": [topology.to_dict() for topology in topologies],
        "policies": [policy.to_dict() for policy in policies],
        "cells": [cell.to_dict() for cell in cells],
    }
    return FPMCollectionPlan(
        backend=backend,
        model_path=model_path,
        system=system,
        aic_revision=revision,
        generator_config_sha256=generator_config_sha256,
        options=options,
        capability=capability,
        dtype_profile=capability.dtype,
        topologies=topologies,
        topology_memory_admission=topology_memory_admission,
        backend_policies=policies,
        cells=cells,
        sha256=_canonical_hash(canonical),
    )

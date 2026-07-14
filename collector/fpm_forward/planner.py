# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a complete immutable FPM campaign plan."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .capabilities import ModelCapabilityProfile, ResolvedDTypeProfile, resolve_model_capability
from .capacity import (
    CapacityDecision,
    FPMCapacityModel,
    FPMExecutionProfile,
    PointCapacityDecision,
    admit_model_residency,
    build_capacity_model,
    build_execution_profile,
    build_memory_selection_admission,
    filter_memory_feasible_points,
)
from .config import FPMCollectionOptions
from .population import AttentionPopulation, build_attention_population
from .sampling import SamplingDesign, build_sampling_design
from .topology import enumerate_fpm_topologies, topology_strategy
from .types import ParallelTopology

RUNTIME_OVERLAY_FILES = ("args.py", "backend_args.py", "instrumented_scheduler.py", "worker_factory.py")


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


def _runtime_overlay_files(
    collector_config: dict[str, Any],
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    raw = collector_config.get("runtime_overlay_dir")
    if raw is None:
        return (), ()
    directory = Path(str(raw)).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"FpmCollector.runtime_overlay_dir is not a directory: {directory}")
    raw_base = collector_config.get("runtime_overlay_original_sha256")
    if not isinstance(raw_base, dict) or set(raw_base) != set(RUNTIME_OVERLAY_FILES):
        raise ValueError(f"FpmCollector.runtime_overlay_original_sha256 must map exactly {list(RUNTIME_OVERLAY_FILES)}")
    files = []
    base_files = []
    for name in RUNTIME_OVERLAY_FILES:
        path = directory / name
        if not path.is_file():
            raise ValueError(f"FpmCollector.runtime_overlay_dir is missing {name}: {directory}")
        files.append((name, hashlib.sha256(path.read_bytes()).hexdigest()))
        base_sha256 = str(raw_base[name])
        if len(base_sha256) != 64 or any(character not in "0123456789abcdef" for character in base_sha256.lower()):
            raise ValueError(f"invalid original SHA256 for runtime overlay target {name}")
        base_files.append((name, base_sha256.lower()))
    return tuple(files), tuple(base_files)


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
    sampling_sha256: str
    execution_profile: FPMExecutionProfile
    parallel_strategy: str = "unspecified"
    gemm_quant_mode: str | None = None
    moe_quant_mode: str | None = None
    fmha_quant_mode: str | None = None
    comm_quant_mode: str | None = None
    sampling_design: SamplingDesign | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "workload_kind": self.workload_kind,
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
            "sampling_sha256": self.sampling_sha256,
            "sampling_design": self.sampling_design.to_dict() if self.sampling_design is not None else None,
            "execution_profile": self.execution_profile.to_dict(),
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
    population: AttentionPopulation
    prefill_design: SamplingDesign
    decode_design: SamplingDesign
    topologies: tuple[ParallelTopology, ...]
    capacity_decisions: tuple[CapacityDecision, ...]
    capacity_memory_fraction: float
    backend_policies: tuple[BackendPolicy, ...]
    runtime_overlay_files: tuple[tuple[str, str], ...]
    runtime_overlay_base_files: tuple[tuple[str, str], ...]
    cells: tuple[FPMCell, ...]
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_name": "aic_fpm_collection_plan",
            "schema_version": 4,
            "backend": self.backend,
            "model_path": self.model_path,
            "system": self.system,
            "aic_revision": self.aic_revision,
            "generator_config_sha256": self.generator_config_sha256,
            "options": self.options.to_dict(),
            "capability": self.capability.to_dict(),
            "dtype_profile": self.dtype_profile.to_dict(),
            "population": self.population.to_summary(),
            "sampling": {
                "scope": "reference_before_per_cell_memory_filter",
                "prefill": self.prefill_design.to_dict(),
                "decode": self.decode_design.to_dict(),
            },
            "topologies": [
                {
                    **topology.to_dict(),
                    "strategy": topology_strategy(topology, is_moe=self.capability.is_moe),
                }
                for topology in self.topologies
            ],
            "capacity_admission": {
                "scope": "model_residency_then_per_cell_execution_envelope",
                "max_num_tokens": 1,
                "max_batch_size": 1,
                "memory_fraction_kind": "of_total",
                "memory_fraction_ceiling": self.capacity_memory_fraction,
                "decisions": [decision.to_dict() for decision in self.capacity_decisions],
            },
            "backend_policies": [policy.to_dict() for policy in self.backend_policies],
            "runtime_overlay": {
                "overlay_sha256": dict(self.runtime_overlay_files),
                "original_sha256": dict(self.runtime_overlay_base_files),
            },
            "cells": [cell.to_dict() for cell in self.cells],
            "counts": {
                "population_points": len(self.population.points),
                "reference_selected_prefill_points": len(self.prefill_design.selected),
                "reference_selected_decode_points": len(self.decode_design.selected),
                "selected_points_by_cell": {
                    cell.cell_id: cell.execution_profile.selected_point_count for cell in self.cells
                },
                "topologies": len(self.topologies),
                "topology_candidates": len(self.capacity_decisions),
                "capacity_rejected": sum(not decision.admitted for decision in self.capacity_decisions),
                "point_capacity_rejected": sum(
                    not decision.admitted for cell in self.cells for decision in cell.execution_profile.decisions
                ),
                "backend_policies": len(self.backend_policies),
                "cells": len(self.cells),
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
    population = build_attention_population(
        backend=backend,
        model_path=model_path,
        selected_ops=selected_ops,
        kv_block_size=options.kv_block_size,
        attention_source=capability.attention_source,
    )
    weight_quantizations = (capability.dtype.gemm_quant_mode,)

    prefill = build_sampling_design(
        population.prefill_points,
        block_size=options.kv_block_size,
        active_budget=options.sampling_budget,
    )
    decode = build_sampling_design(
        population.decode_points,
        block_size=options.kv_block_size,
        active_budget=options.sampling_budget,
    )
    topology_candidates = enumerate_fpm_topologies(
        backend=backend,
        is_moe=capability.is_moe,
        options=options,
        allow_pure_tp=capability.allow_pure_tp,
    )
    topologies, capacity_decisions, capacity_memory_fraction = admit_model_residency(
        model_path=model_path,
        system=system,
        backend=backend,
        capability=capability,
        topologies=topology_candidates,
    )
    policies = _backend_policies(options, collector_config, backend=backend)
    runtime_overlay_files, runtime_overlay_base_files = _runtime_overlay_files(collector_config)

    phase_populations = {
        "prefill": population.prefill_points,
        "decode": population.decode_points,
    }
    execution_profiles: dict[
        tuple[str, ParallelTopology, str],
        tuple[SamplingDesign, FPMExecutionProfile],
    ] = {}
    capacity_models: dict[tuple[ParallelTopology, str], FPMCapacityModel] = {}
    all_design_points = population.points
    maximum_design_batch = max(point.batch_size for point in all_design_points)
    maximum_design_tokens = max(point.batch_size * point.suffix_length for point in all_design_points)
    cells_list = []
    for phase in ("prefill", "decode"):
        for topology in topologies:
            for weight_quantization in weight_quantizations:
                for kv_cache_dtype in capability.dtype.kv_cache_dtypes:
                    profile_key = (phase, topology, kv_cache_dtype)
                    cached_profile = execution_profiles.get(profile_key)
                    if cached_profile is None:
                        capacity_key = (topology, kv_cache_dtype)
                        capacity_model = capacity_models.get(capacity_key)
                        if capacity_model is None:
                            capacity_model = build_capacity_model(
                                model_path=model_path,
                                system=system,
                                backend=backend,
                                capability=capability,
                                topology=topology,
                                kv_cache_dtype=kv_cache_dtype,
                                max_batch_size_hint=maximum_design_batch,
                                max_num_tokens_hint=maximum_design_tokens,
                            )
                            capacity_models[capacity_key] = capacity_model
                        feasible_points, pointwise_decisions = filter_memory_feasible_points(
                            phase_populations[phase],
                            capacity_model=capacity_model,
                            block_size=options.kv_block_size,
                        )
                        sampling_design = build_sampling_design(
                            feasible_points,
                            block_size=options.kv_block_size,
                            active_budget=options.sampling_budget,
                            selection_admission=build_memory_selection_admission(
                                capacity_model=capacity_model,
                                block_size=options.kv_block_size,
                            ),
                        )
                        jointly_excluded = set(sampling_design.excluded_points)
                        pointwise_decisions = tuple(
                            PointCapacityDecision(
                                point=decision.point,
                                admitted=False,
                                disposition="rejected_by_joint_sampling_envelope",
                                reason="point cannot join the capacity-constrained sparse design below the ceiling",
                                required_memory_fraction=None,
                            )
                            if decision.point in jointly_excluded
                            else decision
                            for decision in pointwise_decisions
                        )
                        execution_profile = build_execution_profile(
                            model_path=model_path,
                            system=system,
                            backend=backend,
                            capability=capability,
                            topology=topology,
                            kv_cache_dtype=kv_cache_dtype,
                            design=sampling_design,
                            block_size=options.kv_block_size,
                            capacity_model=capacity_model,
                            pre_admission_decisions=pointwise_decisions,
                        )
                        cached_profile = (sampling_design, execution_profile)
                        execution_profiles[profile_key] = cached_profile
                    sampling_design, execution_profile = cached_profile
                    for policy in policies:
                        cells_list.append(
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
                                sampling_sha256=sampling_design.sha256,
                                execution_profile=execution_profile,
                                parallel_strategy=topology_strategy(topology, is_moe=capability.is_moe),
                                gemm_quant_mode=capability.dtype.gemm_quant_mode,
                                moe_quant_mode=capability.dtype.moe_quant_mode,
                                fmha_quant_mode=capability.dtype.fmha_quant_mode,
                                comm_quant_mode=capability.dtype.comm_quant_mode,
                                sampling_design=sampling_design,
                            )
                        )
    cells = tuple(cells_list)
    canonical = {
        "backend": backend,
        "model_path": model_path,
        "system": system,
        "aic_revision": _git_revision(),
        "generator_config_sha256": generator_config_sha256,
        "options": options.to_dict(),
        "capability": capability.to_dict(),
        "dtype_profile": capability.dtype.to_dict(),
        "population": population.to_summary(),
        "prefill_sampling": prefill.sha256,
        "decode_sampling": decode.sha256,
        "topologies": [topology.to_dict() for topology in topologies],
        "capacity_admission": {
            "memory_fraction": capacity_memory_fraction,
            "decisions": [decision.to_dict() for decision in capacity_decisions],
        },
        "policies": [policy.to_dict() for policy in policies],
        "runtime_overlay": {
            "overlay_sha256": dict(runtime_overlay_files),
            "original_sha256": dict(runtime_overlay_base_files),
        },
        "cells": [cell.to_dict() for cell in cells],
    }
    revision = canonical["aic_revision"]
    return FPMCollectionPlan(
        backend=backend,
        model_path=model_path,
        system=system,
        aic_revision=str(revision),
        generator_config_sha256=generator_config_sha256,
        options=options,
        capability=capability,
        dtype_profile=capability.dtype,
        population=population,
        prefill_design=prefill,
        decode_design=decode,
        topologies=topologies,
        capacity_decisions=capacity_decisions,
        capacity_memory_fraction=capacity_memory_fraction,
        backend_policies=policies,
        runtime_overlay_files=runtime_overlay_files,
        runtime_overlay_base_files=runtime_overlay_base_files,
        cells=cells,
        sha256=_canonical_hash(canonical),
    )

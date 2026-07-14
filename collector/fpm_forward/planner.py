# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a complete immutable FPM campaign plan."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import FPMCollectionOptions
from .population import AttentionPopulation, build_attention_population
from .sampling import SamplingDesign, build_sampling_design
from .topology import enumerate_fpm_topologies
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

    def to_dict(self) -> dict[str, object]:
        return {
            "axis": self.axis,
            "policy_id": self.policy_id,
            "generator_overrides": self.generator_overrides,
            "expected_markers": self.expected_markers,
        }


def _backend_policies(
    options: FPMCollectionOptions,
    collector_config: dict[str, Any],
) -> tuple[BackendPolicy, ...]:
    policies = [BackendPolicy("baseline", "baseline_auto", {}, {})]
    declarations = collector_config.get("backend_variants", {})
    if not isinstance(declarations, dict):
        raise TypeError("FpmCollector.backend_variants must be a mapping")

    for axis in options.backend_axes:
        if axis == "baseline":
            continue
        variants = declarations.get(axis)
        if not isinstance(variants, list) or not variants:
            raise ValueError(
                f"--fpm-backend-axes requested {axis!r}, but FpmCollector.backend_variants.{axis} "
                "has no declared variants"
            )
        for index, variant in enumerate(variants):
            if not isinstance(variant, dict):
                raise TypeError(f"backend variant {axis}[{index}] must be a mapping")
            policy_id = variant.get("id")
            if not policy_id:
                raise ValueError(f"backend variant {axis}[{index}] requires an id")
            overrides = variant.get("generator_overrides", {})
            markers = variant.get("expected_markers", {})
            if not isinstance(overrides, dict) or not isinstance(markers, dict):
                raise TypeError(f"backend variant {axis}[{index}] overrides/markers must be mappings")
            policies.append(BackendPolicy(axis, str(policy_id), overrides, markers))
    return tuple(policies)


@dataclass(frozen=True, slots=True)
class FPMCell:
    cell_id: str
    workload_kind: str
    topology: ParallelTopology
    weight_quantization: str
    kv_cache_dtype: str
    backend_policy: BackendPolicy
    sampling_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "workload_kind": self.workload_kind,
            "topology": self.topology.to_dict(),
            "weight_quantization": self.weight_quantization,
            "kv_cache_dtype": self.kv_cache_dtype,
            "backend_policy": self.backend_policy.to_dict(),
            "sampling_sha256": self.sampling_sha256,
        }


@dataclass(frozen=True, slots=True)
class FPMCollectionPlan:
    backend: str
    model_path: str
    system: str
    aic_revision: str
    generator_config_sha256: str
    options: FPMCollectionOptions
    population: AttentionPopulation
    prefill_design: SamplingDesign
    decode_design: SamplingDesign
    topologies: tuple[ParallelTopology, ...]
    backend_policies: tuple[BackendPolicy, ...]
    runtime_overlay_files: tuple[tuple[str, str], ...]
    runtime_overlay_base_files: tuple[tuple[str, str], ...]
    cells: tuple[FPMCell, ...]
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_name": "aic_fpm_collection_plan",
            "schema_version": 1,
            "backend": self.backend,
            "model_path": self.model_path,
            "system": self.system,
            "aic_revision": self.aic_revision,
            "generator_config_sha256": self.generator_config_sha256,
            "options": self.options.to_dict(),
            "population": self.population.to_summary(),
            "sampling": {
                "prefill": self.prefill_design.to_dict(),
                "decode": self.decode_design.to_dict(),
            },
            "topologies": [topology.to_dict() for topology in self.topologies],
            "backend_policies": [policy.to_dict() for policy in self.backend_policies],
            "runtime_overlay": {
                "overlay_sha256": dict(self.runtime_overlay_files),
                "original_sha256": dict(self.runtime_overlay_base_files),
            },
            "cells": [cell.to_dict() for cell in self.cells],
            "counts": {
                "population_points": len(self.population.points),
                "selected_prefill_points": len(self.prefill_design.selected),
                "selected_decode_points": len(self.decode_design.selected),
                "topologies": len(self.topologies),
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
    collector_config: dict[str, Any] | None = None,
    generator_overrides: dict[str, Any] | None = None,
) -> FPMCollectionPlan:
    if backend != "vllm":
        raise ValueError("FPM Generator V1 currently supports only backend=vllm")
    collector_config = collector_config or {}
    generator_config_sha256 = _canonical_hash(generator_overrides or {})
    population = build_attention_population(
        backend=backend,
        model_path=model_path,
        selected_ops=selected_ops,
        kv_block_size=options.kv_block_size,
    )

    requested_quantizations = set(options.weight_quantizations)
    if requested_quantizations and requested_quantizations != {population.native_weight_quantization}:
        raise ValueError(
            "one --model-path represents one weight artifact; requested quantizations must exactly match "
            f"its native value {population.native_weight_quantization!r}, got {sorted(requested_quantizations)}"
        )
    weight_quantizations = (population.native_weight_quantization,)

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
    topologies = enumerate_fpm_topologies(
        backend=backend,
        is_moe=population.is_moe,
        options=options,
    )
    policies = _backend_policies(options, collector_config)
    runtime_overlay_files, runtime_overlay_base_files = _runtime_overlay_files(collector_config)

    designs = {"prefill": prefill, "decode": decode}
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
            sampling_sha256=designs[phase].sha256,
        )
        for phase in ("prefill", "decode")
        for topology in topologies
        for weight_quantization in weight_quantizations
        for kv_cache_dtype in options.kv_cache_dtypes
        for policy in policies
    )
    canonical = {
        "backend": backend,
        "model_path": model_path,
        "system": system,
        "aic_revision": _git_revision(),
        "generator_config_sha256": generator_config_sha256,
        "options": options.to_dict(),
        "population": population.to_summary(),
        "prefill_sampling": prefill.sha256,
        "decode_sampling": decode.sha256,
        "topologies": [topology.to_dict() for topology in topologies],
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
        population=population,
        prefill_design=prefill,
        decode_design=decode,
        topologies=topologies,
        backend_policies=policies,
        runtime_overlay_files=runtime_overlay_files,
        runtime_overlay_base_files=runtime_overlay_base_files,
        cells=cells,
        sha256=_canonical_hash(canonical),
    )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed public input contract for the generator (design: generator_request_design.md).

`GeneratorRequest` is the boundary type every caller (CLI, SDK bridge, naive)
lowers into. It is a thin, validated, self-documenting contract; the adapter in
`legacy.py` lowers it to the exact legacy params dict the render pipeline already
consumes, so adopting the request changes nothing about generated output.

Frozen dataclasses (no new dependency; consistent with ir.py / environment/).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

SCHEMA_VERSION = "v1beta1"

_ROLES = ("agg", "prefill", "decode")


@dataclass(frozen=True)
class ModelSpec:
    """The model to serve."""

    model_path: str
    served_model_name: Optional[str] = None
    # NEW resolver input; carried but inert until the resolver consumes it.
    precision: Optional[str] = None


@dataclass(frozen=True)
class BackendSpec:
    """Backend engine + version intent."""

    name: str
    # Target Dynamo release -> generator_dynamo_version (image + template).
    dynamo_version: Optional[str] = None
    # Expert override of the generated-artifact template version (the legacy
    # `backend_version` arg to generate_backend_artifacts).
    generated_config_version: Optional[str] = None


@dataclass(frozen=True)
class RoleSizing:
    """Per-worker parallelism / batch sizing (lowers to params.<role>.*)."""

    tensor_parallel_size: Optional[int] = None
    pipeline_parallel_size: Optional[int] = None
    data_parallel_size: Optional[int] = None
    moe_tensor_parallel_size: Optional[int] = None
    moe_expert_parallel_size: Optional[int] = None
    max_batch_size: Optional[int] = None
    # Any other per-role engine key (gpus_per_worker, quant modes, kv_cache_dtype,
    # extra_engine_args, ...) lowered verbatim into params.<role>.
    extra: dict[str, Any] = field(default_factory=dict)

    def to_params(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
            "moe_tensor_parallel_size",
            "moe_expert_parallel_size",
            "max_batch_size",
        ):
            v = getattr(self, k)
            if v is not None:
                out[k] = v
        out.update(self.extra)
        return out

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "RoleSizing":
        known = {
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
            "moe_tensor_parallel_size",
            "moe_expert_parallel_size",
            "max_batch_size",
        }
        kwargs = {k: params.get(k) for k in known if params.get(k) is not None}
        extra = {k: v for k, v in params.items() if k not in known}
        return cls(extra=extra, **kwargs)


@dataclass(frozen=True)
class Topology:
    """Serving topology + sizing source (total_gpus drives naive sizing; roles is
    SDK-supplied explicit sizing)."""

    mode: str = "disagg"
    total_gpus: Optional[int] = None
    roles: dict[str, RoleSizing] = field(default_factory=dict)
    workers: dict[str, int] = field(default_factory=dict)  # {agg|prefill|decode: count}


@dataclass(frozen=True)
class SlaSpec:
    isl: Optional[int] = None
    osl: Optional[int] = None


@dataclass(frozen=True)
class Platform:
    hardware_profile: Optional[str] = None  # system name (e.g. h200_sxm)
    transport: Optional[str] = None  # nvlink | ib | efa
    # Path to an EnvironmentProfile YAML; only fields with a legacy home are
    # lowered today (namespace, image pull secret) — the rest are inert.
    environment_profile: Optional[str] = None


@dataclass(frozen=True)
class CacheSpec:
    policy: Optional[str] = None  # NEW; carried but inert
    pvc_name: Optional[str] = None
    model_cache: Optional[str] = None


@dataclass(frozen=True)
class EmitTargets:
    output_dir: Optional[str] = None
    deployment_target: str = "dynamo-j2"
    k8s: bool = True
    run_script: bool = True
    perf_job: bool = False
    llm_d: bool = False
    sflow: bool = False
    report: bool = False


@dataclass(frozen=True)
class ModelFacts:
    """SDK-owned facts (not a hand-authored field); lowers to ModelConfig.*."""

    is_moe: Optional[bool] = None
    nextn: Optional[int] = None
    prefix: Optional[int] = None
    architecture: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Overrides:
    """Highest-precedence escape hatch. `raw` is a flat {"Section.key": value}
    map that can reach any legacy/ADAPTER-FILLED key; applied last."""

    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratorRequest:
    """The root public contract."""

    model: ModelSpec
    backend: BackendSpec
    topology: Topology = field(default_factory=Topology)
    sla: SlaSpec = field(default_factory=SlaSpec)
    platform: Platform = field(default_factory=Platform)
    cache: CacheSpec = field(default_factory=CacheSpec)
    emit: EmitTargets = field(default_factory=EmitTargets)
    model_facts: Optional[ModelFacts] = None
    overrides: Overrides = field(default_factory=Overrides)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> list[str]:
        """Return a list of structural errors (empty == valid). Pure data checks;
        precision/transport/backend specifics are validated later at resolution."""
        errors: list[str] = []
        if not self.model.model_path:
            errors.append("model.model_path is required")
        if not self.backend.name:
            errors.append("backend.name is required")
        if self.topology.mode not in ("agg", "disagg"):
            errors.append(f"topology.mode must be 'agg' or 'disagg', got {self.topology.mode!r}")

        bad_roles = set(self.topology.roles) - set(_ROLES)
        if bad_roles:
            errors.append(f"unknown topology.roles: {sorted(bad_roles)}")
        roles = set(self.topology.roles) & set(_ROLES)
        if roles:
            if self.topology.mode == "agg" and roles != {"agg"}:
                errors.append("agg mode must use only the 'agg' role")
            if self.topology.mode == "disagg" and not roles <= {"prefill", "decode"}:
                errors.append("disagg mode roles must be a subset of {prefill, decode}")
        # A sizing source is required (naive total_gpus OR explicit roles).
        if not roles and not self.topology.total_gpus:
            errors.append("topology requires either total_gpus or explicit roles")
        return errors

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned, language-neutral contracts for server-config adaptation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "aic-estimate-request/1.0.0"
ADAPTER_VERSION = "1.0.0"
AGGREGATED_REPLICAS_LOWERING_ASSUMPTION = (
    "Aggregated worker replicas are omitted during cli_estimate lowering because "
    "cli_estimate has no aggregated worker-count parameter."
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ModelSettingsV1(_StrictModel):
    path: str = Field(min_length=1)
    nextn: int | Literal["auto"] = 0
    nextn_accepted: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_speculation(self) -> ModelSettingsV1:
        if self.nextn == "auto":
            if self.nextn_accepted is None:
                raise ValueError("nextn_accepted is required when nextn is 'auto'")
            return self
        if self.nextn < 0:
            raise ValueError("nextn must be non-negative")
        if self.nextn == 0 and self.nextn_accepted is not None:
            raise ValueError("nextn_accepted is only valid when speculative decoding is enabled")
        if self.nextn > 0 and self.nextn_accepted is None:
            raise ValueError("nextn_accepted is required when nextn is greater than zero")
        if self.nextn_accepted is not None and self.nextn_accepted > self.nextn:
            raise ValueError("nextn_accepted cannot exceed nextn")
        return self


class QuantizationSettingsV1(_StrictModel):
    gemm: str | None = None
    kvcache: str | None = None
    fmha: str | None = None
    moe: str | None = None
    communication: str | None = None


class BackendSettingsV1(_StrictModel):
    name: Literal["trtllm", "vllm", "sglang"]
    version: str | None = None
    database_mode: Literal["SILICON", "HYBRID", "EMPIRICAL", "SOL"] = "SILICON"
    transfer_policy: str | list[Any] | None = None


class SystemSettingsV1(_StrictModel):
    prefill: str = Field(min_length=1)
    decode: str | None = None


class WorkloadSettingsV1(_StrictModel):
    isl: int = Field(gt=0)
    osl: int = Field(gt=0)
    concurrency: int = Field(gt=0)
    prefix: int = Field(default=0, ge=0)
    image_height: int = Field(default=0, ge=0)
    image_width: int = Field(default=0, ge=0)
    num_images: int = Field(default=1, gt=0)

    @model_validator(mode="after")
    def _validate_prefix(self) -> WorkloadSettingsV1:
        if self.prefix > self.isl:
            raise ValueError("prefix cannot exceed isl")
        if (self.image_height == 0) != (self.image_width == 0):
            raise ValueError("image_height and image_width must either both be zero or both be positive")
        return self


class WorkerSettingsV1(_StrictModel):
    replicas: int = Field(gt=0)
    gpus_per_replica: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    tp_size: int = Field(gt=0)
    pp_size: int = Field(default=1, gt=0)
    attention_dp_size: int = Field(default=1, gt=0)
    moe_tp_size: int | None = Field(default=None, gt=0)
    moe_ep_size: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_width(self) -> WorkerSettingsV1:
        attention_width = self.tp_size * self.pp_size * self.attention_dp_size
        if attention_width != self.gpus_per_replica:
            raise ValueError(
                "tp_size * pp_size * attention_dp_size must equal gpus_per_replica "
                f"({attention_width} != {self.gpus_per_replica})"
            )
        if (self.moe_tp_size is None) != (self.moe_ep_size is None):
            raise ValueError("moe_tp_size and moe_ep_size must be set together")
        if self.moe_tp_size is not None and self.moe_ep_size is not None:
            moe_width = self.moe_tp_size * self.moe_ep_size * self.pp_size
            if moe_width != self.gpus_per_replica:
                raise ValueError(
                    "moe_tp_size * moe_ep_size * pp_size must equal gpus_per_replica "
                    f"({moe_width} != {self.gpus_per_replica})"
                )
        return self


class AggregatedTopologyV1(_StrictModel):
    kind: Literal["agg"] = "agg"
    worker: WorkerSettingsV1


class DisaggregatedTopologyV1(_StrictModel):
    kind: Literal["disagg"] = "disagg"
    prefill: WorkerSettingsV1
    decode: WorkerSettingsV1


TopologyV1 = Annotated[AggregatedTopologyV1 | DisaggregatedTopologyV1, Field(discriminator="kind")]


class RuntimeSettingsV1(_StrictModel):
    enable_encoder_dp: bool = True
    systems_paths: str | None = None
    free_gpu_memory_fraction: float | None = Field(default=None, gt=0, le=1)
    max_seq_len: int | None = Field(default=None, gt=0)
    engine_step_backend: Literal["rust"] | None = None


class SourceProvenanceV1(_StrictModel):
    source_type: Literal["inferencex", "dynamo", "custom"]
    source_reference: str | None = None
    source_ids: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    adapter_version: Literal["1.0.0"] = ADAPTER_VERSION
    assumptions: tuple[str, ...] = ()


class EstimateRequestV1(_StrictModel):
    """Canonical request accepted by adapter v1 and lowered into ``cli_estimate`` kwargs."""

    schema_version: Literal["aic-estimate-request/1.0.0"] = SCHEMA_VERSION
    model: ModelSettingsV1
    quantization: QuantizationSettingsV1 = Field(default_factory=QuantizationSettingsV1)
    backend: BackendSettingsV1
    systems: SystemSettingsV1
    workload: WorkloadSettingsV1
    topology: TopologyV1
    runtime: RuntimeSettingsV1 = Field(default_factory=RuntimeSettingsV1)
    provenance: SourceProvenanceV1

    @classmethod
    def schema_path(cls) -> Path:
        return Path(__file__).with_name("schemas") / "estimate-request-v1.schema.json"


class WorkloadPointOverride(_StrictModel):
    point_id: str | None = None
    isl: int = Field(gt=0)
    osl: int = Field(gt=0)
    concurrency: int = Field(gt=0)
    prefix: int | None = Field(default=None, ge=0)


class AdapterOverrides(_StrictModel):
    """Explicit values with precedence over source values and adapter defaults."""

    model_path: str | None = Field(default=None, min_length=1)
    system_name: str | None = Field(default=None, min_length=1)
    decode_system_name: str | None = Field(default=None, min_length=1)
    backend_name: Literal["trtllm", "vllm", "sglang"] | None = None
    backend_version: str | None = None
    database_mode: Literal["SILICON", "HYBRID", "EMPIRICAL", "SOL"] | None = None
    isl: int | None = Field(default=None, gt=0)
    osl: int | None = Field(default=None, gt=0)
    concurrency: int | None = Field(default=None, gt=0)
    workload_points: tuple[WorkloadPointOverride, ...] | None = None
    prefix: int | None = Field(default=None, ge=0)
    nextn: int | Literal["auto"] | None = None
    nextn_accepted: float | None = Field(default=None, ge=0)
    gemm_quant_mode: str | None = None
    kvcache_quant_mode: str | None = None
    fmha_quant_mode: str | None = None
    moe_quant_mode: str | None = None
    comm_quant_mode: str | None = None
    prefill_batch_size: int | None = Field(default=None, gt=0)
    systems_paths: str | None = None
    free_gpu_memory_fraction: float | None = Field(default=None, gt=0, le=1)
    max_seq_len: int | None = Field(default=None, gt=0)
    engine_step_backend: Literal["rust"] | None = None


class AdaptationDiagnostic(_StrictModel):
    severity: Literal["warning", "error"]
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    path: str | None = None
    hint: str | None = None


class AdaptationOutcome(_StrictModel):
    point_id: str = Field(min_length=1)
    status: Literal["adapted", "rejected"]
    request: EstimateRequestV1 | None = None
    diagnostics: tuple[AdaptationDiagnostic, ...] = ()

    @model_validator(mode="after")
    def _validate_status(self) -> AdaptationOutcome:
        if self.status == "adapted" and self.request is None:
            raise ValueError("an adapted outcome must contain a request")
        if self.status == "rejected" and self.request is not None:
            raise ValueError("a rejected outcome cannot contain a request")
        if self.status == "rejected" and not any(item.severity == "error" for item in self.diagnostics):
            raise ValueError("a rejected outcome must contain an error diagnostic")
        return self


class AdaptationReport(_StrictModel):
    outcomes: tuple[AdaptationOutcome, ...]

    @property
    def requests(self) -> tuple[EstimateRequestV1, ...]:
        return tuple(outcome.request for outcome in self.outcomes if outcome.request is not None)

    @property
    def rejections(self) -> tuple[AdaptationOutcome, ...]:
        return tuple(outcome for outcome in self.outcomes if outcome.status == "rejected")

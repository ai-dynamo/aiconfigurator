# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class EstimateRequest(BaseModel):
    model_path: str
    systems_paths: str | None = None
    total_gpus: int = Field(..., ge=1)
    system: str
    decode_system: str | None = None
    backend: str = "trtllm"
    backend_version: str | None = None
    database_mode: str = "SILICON"
    isl: int = Field(4000, ge=1)
    osl: int = Field(1000, ge=1)
    ttft: float = Field(2000.0, gt=0)
    tpot: float = Field(30.0, gt=0)
    request_latency: float | None = Field(None, gt=0)
    prefix: int = Field(0, ge=0)
    strict_sla: bool = False
    free_gpu_memory_fraction: float | None = Field(None, gt=0, le=1)
    max_seq_len: int | None = Field(None, gt=0)
    top_n: int = Field(5, ge=1, le=50)


class ServiceConfigInput(BaseModel):
    model_path: str | None = None
    hf_token: str | None = None
    served_model_name: str | None = None
    model_name: str | None = None
    served_model_path: str | None = None
    head_node_ip: str | None = None
    port: int | None = Field(None, ge=1, le=65535)
    include_frontend: bool | None = None
    dyn_vllm_kv_event_port: int | None = None
    vllm_nixl_side_channel_port: int | None = None
    prefix: int | None = Field(None, ge=0)


class K8sConfigInput(BaseModel):
    name: str | None = None
    name_prefix: str | None = None
    k8s_namespace: str | None = None
    k8s_image: str | None = None
    k8s_image_pull_secret: str | None = None
    frontend_image_pull_policy: str | None = None
    worker_image_pull_policy: str | None = None
    k8s_engine_mode: str | None = None
    k8s_pvc_name: str | None = None
    k8s_model_path_in_pvc: str | None = None
    k8s_pvc_mount_path: str | None = None
    k8s_model_cache: str | None = None
    k8s_etcd_endpoints: str | None = None
    k8s_hf_home: str | None = None
    working_dir: str | None = None
    frontend_working_dir: str | None = None
    worker_working_dir: str | None = None
    frontend_node_selector: dict[str, str] | None = None
    worker_node_selector: dict[str, str] | None = None
    frontend_extra_volumes: list[dict[str, Any]] | None = None
    worker_extra_volumes: list[dict[str, Any]] | None = None
    frontend_extra_volume_mounts: list[dict[str, Any]] | None = None
    worker_extra_volume_mounts: list[dict[str, Any]] | None = None
    frontend_extra_envs: list[dict[str, Any]] | None = None
    worker_extra_envs: list[dict[str, Any]] | None = None
    oss_enabled: bool | None = None
    oss_endpoint_url: str | None = None
    oss_region: str | None = None
    oss_secret_name: str | None = None
    oss_model_express_url: str | None = None
    oss_streamer_concurrency: int | None = Field(None, ge=1)


class WorkerConfigInput(BaseModel):
    prefill_workers: int | None = Field(None, ge=0)
    decode_workers: int | None = Field(None, ge=0)
    agg_workers: int | None = Field(None, ge=0)


class SlaConfigInput(BaseModel):
    isl: int | None = Field(None, ge=1)
    osl: int | None = Field(None, ge=1)
    ttft: float | None = Field(None, gt=0)
    tpot: float | None = Field(None, gt=0)


class BenchConfigInput(BaseModel):
    name: str | None = None
    image: str | None = None
    profile_start_timeout: int | None = Field(None, ge=0)
    endpoint_type: str | None = None
    endpoint_url: str | None = None
    model: str | None = None
    tokenizer: str | None = None
    isl: int | None = Field(None, ge=1)
    osl: int | None = Field(None, ge=1)
    isl_stddev: int | None = Field(None, ge=0)
    osl_stddev: int | None = Field(None, ge=0)
    concurrency: list[int] | None = None
    estimated_concurrency: int | None = Field(None, ge=1)
    num_requests: list[int] | None = None
    ui: str | None = None


class DynConfigInput(BaseModel):
    mode: str | None = None
    enable_router: bool | None = None


class ModelConfigInput(BaseModel):
    is_moe: bool | None = None
    nextn: int | None = Field(None, ge=0)
    nextn_accept_rates: list[float] | None = None
    prefix: int | None = Field(None, ge=0)


class GenerationOverrides(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    service_config: ServiceConfigInput | None = None
    k8s_config: K8sConfigInput | None = None
    worker_config: WorkerConfigInput | None = None
    sla_config: SlaConfigInput | None = None
    bench_config: BenchConfigInput | None = None
    dyn_config: DynConfigInput | None = None
    model_cfg: ModelConfigInput | None = Field(default=None, alias="model_config")
    params: dict[str, Any] | None = None
    workers: dict[str, Any] | None = None
    generator_dynamo_version: str | None = None
    rule: str | None = None


class ArtifactRequest(BaseModel):
    backend: str
    backend_version: str | None = None
    deployment_target: Literal["dynamo-j2", "dynamo-python", "llm-d"] = "dynamo-j2"
    estimate_request: EstimateRequest | None = None
    selected_mode: str | None = None
    row_index: int = Field(0, ge=0)
    generator_overrides: GenerationOverrides | None = None
    direct_generator_params: dict[str, Any] | None = None


class AggregateArtifactRequest(ArtifactRequest):
    artifact_types: list[Literal["deployment", "benchmark"]] = Field(
        default_factory=lambda: ["deployment", "benchmark"]
    )


class ApplyDeploymentRequest(BaseModel):
    content: str = Field(..., min_length=1)
    namespace: str | None = None
    timeout_seconds: int = Field(180, ge=1, le=1800)
    jid: str | None = None
    dgd_id: str | None = None
    mode: str | None = None
    row_index: int | None = Field(None, ge=0)


class DeleteDeploymentRequest(ApplyDeploymentRequest):
    pass


class DGDPreviewRequest(BaseModel):
    mode: str
    row_index: int = Field(0, ge=0)

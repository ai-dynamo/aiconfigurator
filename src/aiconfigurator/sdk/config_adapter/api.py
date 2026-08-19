# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public dispatch and lowering API for config adaptation."""

from __future__ import annotations

from typing import Any

from .dynamo import DynamoRecipeSource, adapt_dynamo
from .inferencex import InferenceXSource, adapt_inferencex
from .schema import AdaptationReport, AdapterOverrides, EstimateRequestV1


def adapt_config(
    source: InferenceXSource | DynamoRecipeSource,
    overrides: AdapterOverrides | None = None,
) -> AdaptationReport:
    """Adapt source config without executing an estimate."""
    resolved_overrides = overrides or AdapterOverrides()
    if isinstance(source, InferenceXSource):
        return adapt_inferencex(source, resolved_overrides)
    if isinstance(source, DynamoRecipeSource):
        return adapt_dynamo(source, resolved_overrides)
    raise TypeError(f"unsupported config source type: {type(source).__name__}")


def _set_if_not_none(target: dict[str, Any], name: str, value: Any) -> None:
    if value is not None:
        target[name] = value


def to_cli_estimate_kwargs(request: EstimateRequestV1) -> dict[str, Any]:
    """Lower a validated request into the existing ``cli_estimate`` arguments."""
    if not isinstance(request, EstimateRequestV1):
        request = EstimateRequestV1.model_validate(request)
    kwargs: dict[str, Any] = {
        "model_path": request.model.path,
        "system_name": request.systems.prefill,
        "mode": request.topology.kind,
        "backend_name": request.backend.name,
        "database_mode": request.backend.database_mode,
        "isl": request.workload.isl,
        "osl": request.workload.osl,
        "image_height": request.workload.image_height,
        "image_width": request.workload.image_width,
        "num_images": request.workload.num_images,
        "enable_encoder_dp": request.runtime.enable_encoder_dp,
        "prefix": request.workload.prefix,
        "nextn": request.model.nextn,
    }
    _set_if_not_none(kwargs, "backend_version", request.backend.version)
    _set_if_not_none(kwargs, "transfer_policy", request.backend.transfer_policy)
    _set_if_not_none(kwargs, "nextn_accepted", request.model.nextn_accepted)
    _set_if_not_none(kwargs, "gemm_quant_mode", request.quantization.gemm)
    _set_if_not_none(kwargs, "kvcache_quant_mode", request.quantization.kvcache)
    _set_if_not_none(kwargs, "fmha_quant_mode", request.quantization.fmha)
    _set_if_not_none(kwargs, "moe_quant_mode", request.quantization.moe)
    _set_if_not_none(kwargs, "comm_quant_mode", request.quantization.communication)
    _set_if_not_none(kwargs, "systems_paths", request.runtime.systems_paths)
    _set_if_not_none(kwargs, "free_gpu_memory_fraction", request.runtime.free_gpu_memory_fraction)
    _set_if_not_none(
        kwargs,
        "prefill_free_gpu_memory_fraction",
        request.runtime.prefill_free_gpu_memory_fraction,
    )
    _set_if_not_none(
        kwargs,
        "decode_free_gpu_memory_fraction",
        request.runtime.decode_free_gpu_memory_fraction,
    )
    _set_if_not_none(kwargs, "max_seq_len", request.runtime.max_seq_len)
    _set_if_not_none(kwargs, "prefill_max_seq_len", request.runtime.prefill_max_seq_len)
    _set_if_not_none(kwargs, "decode_max_seq_len", request.runtime.decode_max_seq_len)
    _set_if_not_none(kwargs, "engine_step_backend", request.runtime.engine_step_backend)

    if request.topology.kind == "agg":
        worker = request.topology.worker
        # cli_estimate has no aggregated worker-count argument. Source adapters
        # preserve worker.replicas in the canonical request and record that
        # lowering limitation in provenance.
        kwargs.update(
            batch_size=worker.batch_size,
            tp_size=worker.tp_size,
            pp_size=worker.pp_size,
            attention_dp_size=worker.attention_dp_size,
        )
        _set_if_not_none(kwargs, "moe_tp_size", worker.moe_tp_size)
        _set_if_not_none(kwargs, "moe_ep_size", worker.moe_ep_size)
        return kwargs

    prefill = request.topology.prefill
    decode = request.topology.decode
    kwargs.update(
        decode_system_name=request.systems.decode or request.systems.prefill,
        prefill_tp_size=prefill.tp_size,
        prefill_pp_size=prefill.pp_size,
        prefill_attention_dp_size=prefill.attention_dp_size,
        prefill_batch_size=prefill.batch_size,
        prefill_num_workers=prefill.replicas,
        decode_tp_size=decode.tp_size,
        decode_pp_size=decode.pp_size,
        decode_attention_dp_size=decode.attention_dp_size,
        decode_batch_size=decode.batch_size,
        decode_num_workers=decode.replicas,
    )
    _set_if_not_none(kwargs, "prefill_moe_tp_size", prefill.moe_tp_size)
    _set_if_not_none(kwargs, "prefill_moe_ep_size", prefill.moe_ep_size)
    _set_if_not_none(kwargs, "decode_moe_tp_size", decode.moe_tp_size)
    _set_if_not_none(kwargs, "decode_moe_ep_size", decode.moe_ep_size)
    return kwargs

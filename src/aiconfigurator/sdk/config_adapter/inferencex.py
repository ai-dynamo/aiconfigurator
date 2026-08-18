# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed InferenceX DB-record adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from .schema import (
    AGGREGATED_REPLICAS_LOWERING_ASSUMPTION,
    AdaptationDiagnostic,
    AdaptationOutcome,
    AdaptationReport,
    AdapterOverrides,
    AggregatedTopologyV1,
    BackendSettingsV1,
    DisaggregatedTopologyV1,
    EstimateRequestV1,
    ModelSettingsV1,
    QuantizationSettingsV1,
    RuntimeSettingsV1,
    SourceProvenanceV1,
    SystemSettingsV1,
    WorkerSettingsV1,
    WorkloadSettingsV1,
)

HARDWARE_TO_SYSTEM = {
    "h100": "h100_sxm",
    "h200": "h200_sxm",
    "b200": "b200_sxm",
    "b300": "b300_sxm",
    "gb200": "gb200",
    "gb300": "gb300",
}
FRAMEWORK_TO_BACKEND = {
    "trt": "trtllm",
    "trtllm": "trtllm",
    "vllm": "vllm",
    "sglang": "sglang",
    "dynamo-trt": "trtllm",
    "dynamo-trtllm": "trtllm",
    "dynamo-vllm": "vllm",
    "dynamo-sglang": "sglang",
}
MODEL_PATHS = {
    ("minimaxm2.5", "bf16"): "MiniMaxAI/MiniMax-M2.5",
    ("minimaxm2.5", "fp4"): "MiniMaxAI/MiniMax-M2.5",
    ("minimaxm2.5", "fp8"): "MiniMaxAI/MiniMax-M2.5",
    ("dsr1", "bf16"): "deepseek-ai/DeepSeek-V3",
    ("dsr1", "fp4"): "deepseek-ai/DeepSeek-V3",
    ("dsr1", "fp8"): "deepseek-ai/DeepSeek-V3",
    ("kimik2.5", "bf16"): "moonshotai/Kimi-K2.5",
    ("kimik2.5", "fp4"): "moonshotai/Kimi-K2.5",
    ("kimik2.5", "fp8"): "moonshotai/Kimi-K2.5",
    ("kimik2.5", "int4"): "moonshotai/Kimi-K2.5",
    ("qwen3.5", "bf16"): "Qwen/Qwen3.5-397B-A17B",
    ("qwen3.5", "fp4"): "Qwen/Qwen3.5-397B-A17B",
    ("qwen3.5", "fp8"): "Qwen/Qwen3.5-397B-A17B",
    ("llama70b", "bf16"): "meta-llama/Meta-Llama-3.1-70B",
    ("llama70b", "fp4"): "meta-llama/Meta-Llama-3.1-70B",
    ("llama70b", "fp8"): "meta-llama/Meta-Llama-3.1-70B",
    ("gptoss120b", "fp4"): "openai/gpt-oss-120b",
    ("dsv4", "fp4"): "deepseek-ai/DeepSeek-V4-Pro",
    ("dsv4", "fp8"): "sgl-project/DeepSeek-V4-Pro-FP8",
    ("glm5", "bf16"): "zai-org/GLM-5",
    ("glm5", "fp4"): "nvidia/GLM-5-NVFP4",
    ("glm5", "fp8"): "zai-org/GLM-5-FP8",
    ("glm5.1", "bf16"): "zai-org/GLM-5.1",
    ("glm5.1", "fp4"): "nvidia/GLM-5.1-NVFP4",
    ("glm5.1", "fp8"): "zai-org/GLM-5.1-FP8",
    ("glm5.2", "bf16"): "zai-org/GLM-5.2",
    ("glm5.2", "fp4"): "nvidia/GLM-5.2-NVFP4",
    ("glm5.2", "fp8"): "zai-org/GLM-5.2-FP8",
    ("minimaxm3", "bf16"): "MiniMaxAI/MiniMax-M3",
    ("minimaxm3", "fp4"): "MiniMaxAI/MiniMax-M3",
    ("minimaxm3", "fp8"): "MiniMaxAI/MiniMax-M3",
}
MOE_MODELS = frozenset(
    {"minimaxm2.5", "dsr1", "kimik2.5", "qwen3.5", "gptoss120b", "dsv4", "glm5", "glm5.1", "glm5.2", "minimaxm3"}
)
PRECISION_QUANT = {
    "bf16": (None, None),
    "fp4": ("nvfp4", "nvfp4"),
    "fp8": ("fp8", "fp8"),
    "int4": ("int4_wo", "int4_wo"),
}
NATIVE_QUANT_MODELS = frozenset({("gptoss120b", "fp4"), ("dsv4", "fp4"), ("dsv4", "fp8")})


@dataclass(frozen=True)
class InferenceXSource:
    config: Mapping[str, Any]
    benchmark: Mapping[str, Any]
    source_reference: str | None = None


def _integer(record: Mapping[str, Any], key: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TypeError(f"{key} must be an integer")
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{key} must be an integer") from error


def _worker(
    config: Mapping[str, Any],
    *,
    role: str,
    backend: str,
    is_moe: bool,
    concurrency: int,
    disagg: bool,
    prefill_batch_override: int | None,
) -> WorkerSettingsV1:
    gpu_key = f"num_{role}_gpu"
    workers_key = f"{role}_num_workers"
    gpus = _integer(config, gpu_key)
    source_workers = _integer(config, workers_key)
    if source_workers <= 0:
        raise ValueError(f"{workers_key} must be positive")
    replicas = source_workers
    if gpus <= 0 or gpus % replicas:
        raise ValueError(f"{gpu_key} must be positive and divisible by {workers_key}")
    gpus_per_replica = gpus // replicas
    pp = 1
    tp = _integer(config, f"{role}_tp")
    ep = _integer(config, f"{role}_ep")
    if tp <= 0 or ep <= 0:
        raise ValueError(f"{role} TP and EP must be positive")
    attention_dp_enabled = bool(config.get(f"{role}_dp_attention", False))
    if backend == "vllm":
        if gpus_per_replica % tp:
            raise ValueError(f"{role} GPUs per worker ({gpus_per_replica}) must be divisible by TP ({tp})")
        attention_tp = tp
        attention_dp = gpus_per_replica // tp
    elif attention_dp_enabled:
        if tp not in {1, gpus_per_replica}:
            raise ValueError(f"{role} declared TP ({tp}) conflicts with attention-DP width ({gpus_per_replica})")
        attention_tp = 1
        attention_dp = gpus_per_replica
    else:
        if tp != gpus_per_replica:
            raise ValueError(f"{role} declared TP ({tp}) does not match GPUs per worker ({gpus_per_replica})")
        attention_tp = gpus_per_replica
        attention_dp = 1

    moe_tp: int | None = None
    moe_ep: int | None = None
    if is_moe:
        if backend == "vllm":
            moe_tp, moe_ep = (1, attention_tp * attention_dp) if ep > 1 else (attention_tp * attention_dp, 1)
        else:
            if gpus_per_replica % ep:
                raise ValueError(f"{role} GPUs per worker ({gpus_per_replica}) must be divisible by EP ({ep})")
            moe_tp, moe_ep = gpus_per_replica // ep, ep

    if role == "prefill" and disagg:
        batch_size = prefill_batch_override or 1
    else:
        denominator = replicas * attention_dp
        if concurrency % denominator:
            raise ValueError(
                f"concurrency ({concurrency}) must be divisible by {role} replicas * attention DP ({denominator})"
            )
        batch_size = concurrency // denominator
    return WorkerSettingsV1(
        replicas=replicas,
        gpus_per_replica=gpus_per_replica,
        batch_size=batch_size,
        tp_size=attention_tp,
        pp_size=pp,
        attention_dp_size=attention_dp,
        moe_tp_size=moe_tp,
        moe_ep_size=moe_ep,
    )


def adapt_inferencex(source: InferenceXSource, overrides: AdapterOverrides) -> AdaptationReport:
    config = source.config
    benchmark = source.benchmark
    point_id = ":".join(
        str(benchmark.get(key, config.get(key, "unknown"))) for key in ("config_id", "isl", "osl", "conc")
    )
    diagnostics: list[AdaptationDiagnostic] = []
    assumptions = ["InferenceX does not expose pipeline parallelism; pp_size defaults to 1."]
    try:
        hardware = str(config.get("hardware", "")).lower()
        system = overrides.system_name or HARDWARE_TO_SYSTEM.get(hardware)
        if not system:
            raise ValueError(f"unsupported or missing InferenceX hardware {hardware!r}")
        decode_system = overrides.decode_system_name or system

        framework = str(config.get("framework", "")).lower()
        source_backend = FRAMEWORK_TO_BACKEND.get(framework)
        backend = overrides.backend_name or source_backend
        if not backend:
            raise ValueError(f"unsupported or missing InferenceX framework {framework!r}")

        model_alias = str(config.get("silicon_model", config.get("model", ""))).lower()
        precision = str(config.get("precision", "")).lower()
        model_path = overrides.model_path or MODEL_PATHS.get((model_alias, precision))
        if not model_path:
            raise ValueError(f"unsupported InferenceX model/precision pair {(model_alias, precision)!r}")
        if precision not in PRECISION_QUANT:
            raise ValueError(f"unsupported InferenceX precision {precision!r}")

        isl = overrides.isl or _integer(benchmark, "isl")
        osl = overrides.osl or _integer(benchmark, "osl")
        concurrency = overrides.concurrency or _integer(benchmark, "conc")
        if min(isl, osl, concurrency) <= 0:
            raise ValueError("isl, osl, and concurrency must be positive")

        spec_method = str(config.get("spec_method", "none")).lower()
        speculative = spec_method not in {"", "none", "false", "disabled", "no_spec"}
        if speculative and (overrides.nextn is None or overrides.nextn_accepted is None):
            raise ValueError("InferenceX speculative decoding requires explicit nextn and nextn_accepted overrides")
        nextn = overrides.nextn if overrides.nextn is not None else 0

        disagg = bool(config.get("disagg", False))
        is_moe = model_alias in MOE_MODELS
        if disagg:
            assumptions.append("InferenceX disaggregated prefill batch defaults to 1 unless explicitly overridden.")
            topology = DisaggregatedTopologyV1(
                prefill=_worker(
                    config,
                    role="prefill",
                    backend=backend,
                    is_moe=is_moe,
                    concurrency=concurrency,
                    disagg=True,
                    prefill_batch_override=overrides.prefill_batch_size,
                ),
                decode=_worker(
                    config,
                    role="decode",
                    backend=backend,
                    is_moe=is_moe,
                    concurrency=concurrency,
                    disagg=True,
                    prefill_batch_override=None,
                ),
            )
            systems = SystemSettingsV1(prefill=system, decode=decode_system)
        else:
            agg_config = config
            if _integer(config, "decode_num_workers") == 0:
                assumptions.append(
                    "InferenceX aggregated decode_num_workers=0 is an irrelevant sentinel; replicas default to 1."
                )
                agg_config = {**config, "decode_num_workers": 1}
            assumptions.append(AGGREGATED_REPLICAS_LOWERING_ASSUMPTION)
            topology = AggregatedTopologyV1(
                worker=_worker(
                    agg_config,
                    role="decode",
                    backend=backend,
                    is_moe=is_moe,
                    concurrency=concurrency,
                    disagg=False,
                    prefill_batch_override=None,
                )
            )
            systems = SystemSettingsV1(prefill=system)

        gemm, moe = (None, None) if (model_alias, precision) in NATIVE_QUANT_MODELS else PRECISION_QUANT[precision]
        if not is_moe:
            moe = None
        if backend == "sglang" and moe == "fp8":
            moe = "fp8_block"
        backend_version = overrides.backend_version
        if backend_version is None:
            diagnostics.append(
                AdaptationDiagnostic(
                    severity="warning",
                    code="backend_version_unpinned",
                    message="Backend version is not pinned; AIC will select its latest compatible database version.",
                    path="backend.version",
                )
            )

        request = EstimateRequestV1(
            model=ModelSettingsV1(path=model_path, nextn=nextn, nextn_accepted=overrides.nextn_accepted),
            quantization=QuantizationSettingsV1(
                gemm=overrides.gemm_quant_mode if overrides.gemm_quant_mode is not None else gemm,
                kvcache=overrides.kvcache_quant_mode,
                fmha=overrides.fmha_quant_mode,
                moe=overrides.moe_quant_mode if overrides.moe_quant_mode is not None else moe,
                communication=overrides.comm_quant_mode,
            ),
            backend=BackendSettingsV1(
                name=backend,
                version=backend_version,
                database_mode=overrides.database_mode or "SILICON",
            ),
            systems=systems,
            workload=WorkloadSettingsV1(
                isl=isl,
                osl=osl,
                concurrency=concurrency,
                prefix=overrides.prefix or 0,
            ),
            topology=topology,
            runtime=RuntimeSettingsV1(
                systems_paths=overrides.systems_paths,
                free_gpu_memory_fraction=overrides.free_gpu_memory_fraction,
                max_seq_len=overrides.max_seq_len,
                engine_step_backend=overrides.engine_step_backend,
            ),
            provenance=SourceProvenanceV1(
                source_type="inferencex",
                source_reference=source.source_reference,
                source_ids={
                    "config_id": config.get("config_id"),
                    "benchmark_id": benchmark.get("bench_id", benchmark.get("id")),
                },
                assumptions=tuple(assumptions),
            ),
        )
    except (ValueError, TypeError, ValidationError) as error:
        diagnostics.append(
            AdaptationDiagnostic(
                severity="error",
                code="inferencex_mapping_failed",
                message=str(error),
                hint="Provide a supported DB-export record and operating point, or add an explicit override.",
            )
        )
        return AdaptationReport(
            outcomes=(AdaptationOutcome(point_id=point_id, status="rejected", diagnostics=tuple(diagnostics)),)
        )
    return AdaptationReport(
        outcomes=(
            AdaptationOutcome(
                point_id=point_id,
                status="adapted",
                request=request,
                diagnostics=tuple(diagnostics),
            ),
        )
    )

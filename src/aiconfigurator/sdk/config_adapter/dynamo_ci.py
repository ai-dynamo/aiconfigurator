# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter for concrete dynamo-ci benchmark recipe documents."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from .schema import (
    AdaptationDiagnostic,
    AdaptationOutcome,
    AdaptationReport,
    AdapterOverrides,
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

_SYSTEMS = {
    "h100": "h100_sxm",
    "h200": "h200_sxm",
    "b200": "b200_sxm",
    "b300": "b300_sxm",
    "gb200": "gb200",
    "gb300": "gb300",
}
_VERSION_PATTERN = re.compile(r"^\d+\.\d+(?:\.\d+)?(?:[-._a-zA-Z0-9]+)?$")


@dataclass(frozen=True)
class _Point:
    point_id: str
    isl: Any
    osl: Any
    concurrency: Any
    prefix: Any = 0


def is_dynamo_ci_recipe(document: Mapping[str, Any]) -> bool:
    """Return whether a document is a concrete, already-rendered benchmark recipe."""
    return {"model", "resources", "backend", "benchmark"}.issubset(document)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{label} must be an integer")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer") from error
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _flag(config: Mapping[str, Any], *names: str) -> Any:
    present = [(name, config[name]) for name in names if name in config]
    if not present:
        return None
    first = str(present[0][1]).lower()
    if any(str(value).lower() != first for _, value in present[1:]):
        details = ", ".join(f"{name}={value!r}" for name, value in present)
        raise ValueError(f"conflicting aliases: {details}")
    return present[0][1]


def _boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in {"true", "1", "yes"}:
            return True
        if value.lower() in {"false", "0", "no", ""}:
            return False
    raise ValueError(f"expected a boolean, got {value!r}")


def _points(document: Mapping[str, Any], overrides: AdapterOverrides) -> list[_Point]:
    if overrides.workload_points is not None:
        return [
            _Point(point.point_id or f"point-{index}", point.isl, point.osl, point.concurrency, point.prefix or 0)
            for index, point in enumerate(overrides.workload_points)
        ]
    benchmark = _mapping(document.get("benchmark"), "benchmark")
    raw_concurrencies = benchmark.get("concurrencies", benchmark.get("concurrency"))
    if isinstance(raw_concurrencies, str):
        concurrency_values: list[Any] = [item.strip() for item in raw_concurrencies.split("x") if item.strip()]
    elif isinstance(raw_concurrencies, list):
        concurrency_values = raw_concurrencies
    elif raw_concurrencies is None:
        concurrency_values = [None]
    else:
        concurrency_values = [raw_concurrencies]
    if not concurrency_values:
        concurrency_values = [None]
    return [
        _Point(
            point_id=f"concurrency-{value}" if value is not None else f"point-{index}",
            isl=benchmark.get("isl"),
            osl=benchmark.get("osl"),
            concurrency=value,
            prefix=benchmark.get("prefix", 0),
        )
        for index, value in enumerate(concurrency_values)
    ]


def _role_configs(document: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    backend = _mapping(document.get("backend"), "backend")
    config = _mapping(backend.get("sglang_config"), "backend.sglang_config")
    roles: dict[str, Mapping[str, Any]] = {}
    for role in ("prefill", "decode"):
        roles[role] = _mapping(config.get(role), f"backend.sglang_config.{role}")
    return roles


def _worker(
    *,
    role: str,
    config: Mapping[str, Any],
    resources: Mapping[str, Any],
    concurrency: int,
    prefill_batch_override: int | None,
    assumptions: list[str],
) -> WorkerSettingsV1:
    nodes = _positive_int(resources.get(f"{role}_nodes"), f"resources.{role}_nodes")
    replicas = _positive_int(resources.get(f"{role}_workers"), f"resources.{role}_workers")
    gpus_per_node = _positive_int(resources.get("gpus_per_node"), "resources.gpus_per_node")
    total_gpus = nodes * gpus_per_node
    if total_gpus % replicas:
        raise ValueError(f"{role} nodes * GPUs per node must be divisible by {role} workers")
    gpus_per_replica = total_gpus // replicas

    pp_raw = _flag(config, "pp-size", "pipeline-parallel-size")
    pp = _positive_int(pp_raw, f"{role} pipeline parallelism") if pp_raw is not None else 1
    if pp_raw is None:
        assumptions.append(f"{role} pipeline parallelism defaults to 1.")
    if gpus_per_replica % pp:
        raise ValueError(f"{role} GPUs per worker must be divisible by pipeline parallelism")
    width_without_pp = gpus_per_replica // pp

    source_tp_raw = _flag(config, "tp-size", "tensor-parallel-size")
    source_tp = (
        _positive_int(source_tp_raw, f"{role} tensor parallelism") if source_tp_raw is not None else width_without_pp
    )
    if source_tp != width_without_pp:
        raise ValueError(
            f"{role} tensor parallelism ({source_tp}) * PP ({pp}) does not match GPUs per worker ({gpus_per_replica})"
        )
    if source_tp_raw is None:
        assumptions.append(f"{role} tensor parallelism is derived from GPUs per worker and PP.")

    dp_enabled_raw = _flag(config, "enable-dp-attention", "enable-attention-dp")
    attention_dp_enabled = _boolean(dp_enabled_raw) if dp_enabled_raw is not None else False
    source_dp_raw = _flag(config, "dp-size", "data-parallel-size")
    if attention_dp_enabled:
        attention_dp = (
            _positive_int(source_dp_raw, f"{role} data parallelism") if source_dp_raw is not None else width_without_pp
        )
        if attention_dp != width_without_pp:
            raise ValueError(
                f"{role} attention-DP ({attention_dp}) * PP ({pp}) does not match GPUs per worker ({gpus_per_replica})"
            )
        attention_tp = 1
    else:
        if source_dp_raw is not None and _positive_int(source_dp_raw, f"{role} data parallelism") != 1:
            raise ValueError(f"{role} declares data parallelism without enabling DP attention")
        attention_dp = 1
        attention_tp = width_without_pp

    ep_raw = _flag(config, "ep-size", "expert-parallel-size")
    moe_ep = _positive_int(ep_raw, f"{role} expert parallelism") if ep_raw is not None else 1
    if width_without_pp % moe_ep:
        raise ValueError(f"{role} GPUs per worker / PP must be divisible by expert parallelism")
    moe_tp = width_without_pp // moe_ep

    if role == "prefill":
        batch_size = prefill_batch_override or 1
        if prefill_batch_override is None:
            assumptions.append("dynamo-ci disaggregated prefill batch defaults to 1.")
    else:
        denominator = replicas * attention_dp
        if concurrency % denominator:
            raise ValueError(
                f"concurrency ({concurrency}) must be divisible by {role} workers * attention DP ({denominator})"
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


def _quantization(
    document: Mapping[str, Any], roles: Mapping[str, Mapping[str, Any]]
) -> tuple[str | None, str | None, str | None]:
    model = _mapping(document.get("model"), "model")
    precision = str(model.get("precision", "")).lower()
    role_quant = {str(config.get("quantization")).lower() for config in roles.values() if config.get("quantization")}
    if len(role_quant) > 1:
        raise ValueError(f"prefill and decode use different quantization modes: {sorted(role_quant)}")
    source_quant = next(iter(role_quant), precision)
    if source_quant in {"fp4", "nvfp4", "modelopt_fp4"}:
        gemm, moe = "nvfp4", "nvfp4"
    elif source_quant in {"fp8", "modelopt_fp8"}:
        gemm, moe = "fp8", "fp8_block"
    elif source_quant in {"", "bf16", "none"}:
        gemm, moe = None, None
    else:
        raise ValueError(f"unsupported dynamo-ci quantization mode {source_quant!r}")

    kv_values = {
        str(value).lower()
        for config in roles.values()
        if (value := _flag(config, "kv-cache-dtype", "kv_cache_dtype")) is not None
    }
    if len(kv_values) > 1:
        raise ValueError(f"prefill and decode use different KV cache dtypes: {sorted(kv_values)}")
    kv = next(iter(kv_values), None)
    if kv in {"fp8_e4m3", "fp8_e5m2", "fp8"}:
        kv = "fp8"
    elif kv in {None, "auto", "none"}:
        kv = None
    else:
        raise ValueError(f"unsupported dynamo-ci KV cache dtype {kv!r}")
    return gemm, moe, kv


def _shared_runtime(
    roles: Mapping[str, Mapping[str, Any]], overrides: AdapterOverrides
) -> tuple[float | None, float | None, float | None, int | None, int | None, int | None]:
    fraction = overrides.free_gpu_memory_fraction
    prefill_fraction = overrides.prefill_free_gpu_memory_fraction
    decode_fraction = overrides.decode_free_gpu_memory_fraction
    if fraction is None:
        prefill_value = _flag(roles["prefill"], "mem-fraction-static", "gpu-memory-utilization")
        decode_value = _flag(roles["decode"], "mem-fraction-static", "gpu-memory-utilization")
        if prefill_fraction is None:
            prefill_fraction = float(prefill_value) if prefill_value is not None else None
        if decode_fraction is None:
            decode_fraction = float(decode_value) if decode_value is not None else None

    if overrides.max_seq_len is not None:
        max_seq_len = overrides.max_seq_len
        prefill_max_seq_len = overrides.prefill_max_seq_len
        decode_max_seq_len = overrides.decode_max_seq_len
    else:
        max_seq_len = None
        prefill_value = _flag(roles["prefill"], "context-length", "max-model-len", "max-seq-len")
        decode_value = _flag(roles["decode"], "context-length", "max-model-len", "max-seq-len")
        prefill_max_seq_len = (
            overrides.prefill_max_seq_len
            if overrides.prefill_max_seq_len is not None
            else (_positive_int(prefill_value, "prefill context length") if prefill_value is not None else None)
        )
        decode_max_seq_len = (
            overrides.decode_max_seq_len
            if overrides.decode_max_seq_len is not None
            else (_positive_int(decode_value, "decode context length") if decode_value is not None else None)
        )
    return (
        fraction,
        prefill_fraction,
        decode_fraction,
        max_seq_len,
        prefill_max_seq_len,
        decode_max_seq_len,
    )


def _speculation(roles: Mapping[str, Mapping[str, Any]], overrides: AdapterOverrides) -> int | str:
    disabled = {"", "0", "false", "none", "disabled"}
    depth_values: set[int] = set()
    active = False
    for role, config in roles.items():
        depth = _flag(config, "num-speculative-tokens", "speculative-num-steps", "speculative-token-num")
        algorithm = _flag(config, "speculative-algorithm", "speculative-draft-model-path", "speculative-model")
        if str(depth).lower() not in disabled:
            depth_values.add(_positive_int(depth, f"{role} speculative depth"))
            active = True
        if str(algorithm).lower() not in disabled:
            active = True
    if len(depth_values) > 1:
        raise ValueError(f"prefill and decode use different speculative depths: {sorted(depth_values)}")
    if not active:
        return overrides.nextn if overrides.nextn is not None else 0
    if overrides.nextn_accepted is None:
        raise ValueError("speculative decoding requires an explicit nextn_accepted override")
    if overrides.nextn is not None:
        return overrides.nextn
    if not depth_values:
        raise ValueError("speculative decoding depth is missing; provide a nextn override")
    return next(iter(depth_values))


def _request(
    *,
    document: Mapping[str, Any],
    point: _Point,
    overrides: AdapterOverrides,
    source_reference: str | None,
) -> EstimateRequestV1:
    resources = _mapping(document.get("resources"), "resources")
    roles = _role_configs(document)
    assumptions: list[str] = []

    isl = _positive_int(overrides.isl if overrides.isl is not None else point.isl, "workload ISL")
    osl = _positive_int(overrides.osl if overrides.osl is not None else point.osl, "workload OSL")
    concurrency = _positive_int(
        overrides.concurrency if overrides.concurrency is not None else point.concurrency,
        "workload concurrency",
    )
    prefix = int(overrides.prefix if overrides.prefix is not None else point.prefix or 0)

    served_models = {
        str(config.get("served-model-name")) for config in roles.values() if config.get("served-model-name")
    }
    if len(served_models) > 1:
        raise ValueError(f"prefill and decode use different served models: {sorted(served_models)}")
    model_path = overrides.model_path or next(iter(served_models), None)
    if not model_path:
        model = _mapping(document.get("model"), "model")
        raw_model_path = model.get("path")
        if isinstance(raw_model_path, str) and "/" in raw_model_path:
            model_path = raw_model_path
    if not model_path:
        raise ValueError("model identity is missing; provide a model_path override")

    hardware = str(resources.get("gpu_type", "")).lower()
    system = overrides.system_name or _SYSTEMS.get(hardware)
    if not system:
        raise ValueError(f"unsupported or missing dynamo-ci GPU type {hardware!r}")

    backend_version = overrides.backend_version
    model = _mapping(document.get("model"), "model")
    container = model.get("container")
    if backend_version is None and isinstance(container, str) and _VERSION_PATTERN.fullmatch(container):
        backend_version = container

    gemm, moe, kvcache = _quantization(document, roles)
    (
        fraction,
        prefill_fraction,
        decode_fraction,
        max_seq_len,
        prefill_max_seq_len,
        decode_max_seq_len,
    ) = _shared_runtime(roles, overrides)
    nextn = _speculation(roles, overrides)
    topology = DisaggregatedTopologyV1(
        prefill=_worker(
            role="prefill",
            config=roles["prefill"],
            resources=resources,
            concurrency=concurrency,
            prefill_batch_override=overrides.prefill_batch_size,
            assumptions=assumptions,
        ),
        decode=_worker(
            role="decode",
            config=roles["decode"],
            resources=resources,
            concurrency=concurrency,
            prefill_batch_override=None,
            assumptions=assumptions,
        ),
    )
    return EstimateRequestV1(
        model=ModelSettingsV1(
            path=model_path,
            nextn=nextn,
            nextn_accepted=overrides.nextn_accepted,
        ),
        quantization=QuantizationSettingsV1(
            gemm=overrides.gemm_quant_mode if overrides.gemm_quant_mode is not None else gemm,
            kvcache=overrides.kvcache_quant_mode if overrides.kvcache_quant_mode is not None else kvcache,
            fmha=overrides.fmha_quant_mode,
            moe=overrides.moe_quant_mode if overrides.moe_quant_mode is not None else moe,
            communication=overrides.comm_quant_mode,
        ),
        backend=BackendSettingsV1(
            name=overrides.backend_name or "sglang",
            version=backend_version,
            database_mode=overrides.database_mode or "SILICON",
        ),
        systems=SystemSettingsV1(prefill=system, decode=overrides.decode_system_name or system),
        workload=WorkloadSettingsV1(isl=isl, osl=osl, concurrency=concurrency, prefix=prefix),
        topology=topology,
        runtime=RuntimeSettingsV1(
            systems_paths=overrides.systems_paths,
            free_gpu_memory_fraction=fraction,
            prefill_free_gpu_memory_fraction=prefill_fraction,
            decode_free_gpu_memory_fraction=decode_fraction,
            max_seq_len=max_seq_len,
            prefill_max_seq_len=prefill_max_seq_len,
            decode_max_seq_len=decode_max_seq_len,
            engine_step_backend=overrides.engine_step_backend,
        ),
        provenance=SourceProvenanceV1(
            source_type="dynamo",
            source_reference=source_reference,
            source_ids={
                "format": "dynamo-ci-concrete",
                "recipe": document.get("name"),
                "operating_point": point.point_id,
            },
            assumptions=tuple(dict.fromkeys(assumptions)),
        ),
    )


def adapt_dynamo_ci(
    document: Mapping[str, Any],
    overrides: AdapterOverrides,
    *,
    source_reference: str | None,
) -> AdaptationReport:
    """Adapt all operating points in one concrete dynamo-ci benchmark recipe."""
    try:
        points = _points(document, overrides)
    except (TypeError, ValueError) as error:
        diagnostic = AdaptationDiagnostic(
            severity="error",
            code="dynamo_ci_workload_rejected",
            message=str(error),
            hint="Use a concrete benchmark recipe with literal ISL, OSL, and concurrency values.",
        )
        return AdaptationReport(
            outcomes=(AdaptationOutcome(point_id="point-0", status="rejected", diagnostics=(diagnostic,)),)
        )

    outcomes: list[AdaptationOutcome] = []
    for point in points:
        diagnostics: list[AdaptationDiagnostic] = []
        try:
            request = _request(
                document=document,
                point=point,
                overrides=overrides,
                source_reference=source_reference,
            )
        except (TypeError, ValueError, ValidationError) as error:
            diagnostics.append(
                AdaptationDiagnostic(
                    severity="error",
                    code="dynamo_ci_mapping_failed",
                    message=str(error),
                    hint="Fix inconsistent recipe values or supply an explicit adapter override.",
                )
            )
            outcomes.append(
                AdaptationOutcome(point_id=point.point_id, status="rejected", diagnostics=tuple(diagnostics))
            )
            continue
        if request.backend.version is None:
            diagnostics.append(
                AdaptationDiagnostic(
                    severity="warning",
                    code="backend_version_unpinned",
                    message="Backend version is not pinned; AIC will select its latest compatible database version.",
                    path="backend.version",
                )
            )
        outcomes.append(
            AdaptationOutcome(
                point_id=point.point_id,
                status="adapted",
                request=request,
                diagnostics=tuple(diagnostics),
            )
        )
    return AdaptationReport(outcomes=tuple(outcomes))

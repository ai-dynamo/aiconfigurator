# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe adapter for standard DynamoGraphDeployment recipes."""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from pydantic import ValidationError

from .dynamo_ci import adapt_dynamo_ci, is_dynamo_ci_recipe
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

DocumentInput = str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]]
_ENV_PATTERN = re.compile(r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))")
_BACKEND_MODULE_PATTERN = re.compile(r"(?:python\d*\s+-m\s+)?dynamo\.(vllm|sglang|trtllm)\b")
_SYSTEM_MARKERS = {
    "H100": "h100_sxm",
    "H200": "h200_sxm",
    "B200": "b200_sxm",
    "B300": "b300_sxm",
    "GB200": "gb200",
    "GB300": "gb300",
}


@dataclass(frozen=True)
class DynamoRecipeSource:
    deployment: DocumentInput
    performance: DocumentInput | None = None
    source_reference: str | None = None


@dataclass(frozen=True)
class _Point:
    point_id: str
    isl: Any
    osl: Any
    concurrency: Any
    prefix: Any = 0
    error: str | None = None


@dataclass(frozen=True)
class _Service:
    name: str
    role: str
    config: Mapping[str, Any]
    env: Mapping[str, str]
    flags: Mapping[str, str | bool]
    engine_config: Mapping[str, Any]


def _documents(value: DocumentInput | None) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Path):
        text = value.read_text()
        loaded = list(yaml.safe_load_all(text))
    elif isinstance(value, str):
        loaded = list(yaml.safe_load_all(value))
    elif isinstance(value, Mapping):
        loaded = [value]
    else:
        loaded = list(value)
    documents: list[Mapping[str, Any]] = []
    for index, document in enumerate(loaded):
        if document is None:
            continue
        if not isinstance(document, Mapping):
            raise TypeError(f"YAML document {index} must be an object")
        documents.append(document)
    return documents


def _env(entries: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    if not isinstance(entries, list):
        return result
    for entry in entries:
        if not isinstance(entry, Mapping) or "value" not in entry:
            continue
        name = entry.get("name")
        value = entry.get("value")
        if isinstance(name, str) and isinstance(value, (str, int, float, bool)):
            result[name] = str(value)
    return result


def _container(service: Mapping[str, Any]) -> Mapping[str, Any]:
    extra = service.get("extraPodSpec", {})
    if not isinstance(extra, Mapping):
        return {}
    main = extra.get("mainContainer", {})
    return main if isinstance(main, Mapping) else {}


def _command(container: Mapping[str, Any]) -> str:
    values: list[str] = []
    for key in ("command", "args"):
        value = container.get(key, [])
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(str(item) for item in value)
    return " ".join(values).replace("\\\n", " ")


def _expand_literal(command: str, env: Mapping[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group("braced") or match.group("plain")
        if name not in env:
            raise ValueError(f"command depends on unresolved environment variable {name!r}")
        value = env[name]
        if "$(" in value or "`" in value:
            raise ValueError(f"environment variable {name!r} is shell-derived, not literal")
        return value

    expanded = _ENV_PATTERN.sub(replace, command)
    if "$" in expanded:
        raise ValueError("command contains an unsupported shell parameter expansion")
    return expanded


def _flags(command: str) -> dict[str, str | bool]:
    if "$(" in command or "`" in command:
        raise ValueError("engine arguments contain a shell-derived value")
    try:
        tokens = shlex.split(command, comments=True)
    except ValueError as error:
        raise ValueError(f"cannot parse engine arguments: {error}") from error
    result: dict[str, str | bool] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        name_value = token[2:].split("=", 1)
        name = name_value[0].replace("_", "-")
        if len(name_value) == 2:
            value: str | bool = name_value[1]
        elif index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            index += 1
            value = tokens[index]
        else:
            value = True
        if name in result and result[name] != value:
            raise ValueError(f"engine argument --{name} has conflicting values")
        result[name] = value
        index += 1
    return result


def _config_maps(documents: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Mapping[str, Any]]]:
    result: dict[str, dict[str, Mapping[str, Any]]] = {}
    for document in documents:
        if document.get("kind") != "ConfigMap":
            continue
        metadata = document.get("metadata", {})
        name = metadata.get("name") if isinstance(metadata, Mapping) else None
        if not isinstance(name, str):
            continue
        parsed: dict[str, Mapping[str, Any]] = {}
        data = document.get("data", {})
        if isinstance(data, Mapping):
            for filename, text in data.items():
                if not isinstance(filename, str) or not isinstance(text, str):
                    continue
                value = yaml.safe_load(text)
                if isinstance(value, Mapping):
                    parsed[filename] = value
        result[name] = parsed
    return result


def _mounted_config(service: Mapping[str, Any], config_maps: Mapping[str, Any]) -> Mapping[str, Any]:
    main = _container(service)
    extra = service.get("extraPodSpec", {})
    volumes = extra.get("volumes", []) if isinstance(extra, Mapping) else []
    config_volumes: dict[str, tuple[str, Mapping[str, str] | None]] = {}
    if isinstance(volumes, list):
        for volume in volumes:
            if not isinstance(volume, Mapping):
                continue
            config_map = volume.get("configMap", {})
            volume_name = volume.get("name")
            config_map_name = config_map.get("name") if isinstance(config_map, Mapping) else None
            if not isinstance(volume_name, str) or not isinstance(config_map_name, str):
                continue
            projected_paths: dict[str, str] = {}
            items = config_map.get("items")
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, Mapping):
                        continue
                    key = item.get("key")
                    path = item.get("path")
                    if isinstance(key, str) and isinstance(path, str):
                        projected_paths[key] = path
            config_volumes[volume_name] = (config_map_name, projected_paths or None)

    env = _env(main.get("env", []))
    engine_path: Any = env.get("ENGINE_ARGS", "")
    try:
        flags = _flags(_expand_literal(_command(main), env))
    except ValueError:
        flags = _flags(_command(main))
    if flags.get("extra-engine-args") not in (None, True):
        engine_path = flags["extra-engine-args"]
    engine_path = str(engine_path)
    candidates: list[Mapping[str, Any]] = []
    mounts = main.get("volumeMounts", [])
    if isinstance(mounts, list):
        for mount in mounts:
            if not isinstance(mount, Mapping):
                continue
            volume_name = mount.get("name")
            mount_path = mount.get("mountPath")
            if not isinstance(volume_name, str) or not isinstance(mount_path, str):
                continue
            volume_config = config_volumes.get(volume_name)
            if volume_config is None:
                continue
            config_map_name, projected_paths = volume_config
            sub_path = mount.get("subPath")
            for filename, config in config_maps.get(config_map_name, {}).items():
                if projected_paths is not None and filename not in projected_paths:
                    continue
                projected_path = projected_paths[filename] if projected_paths is not None else filename
                if isinstance(sub_path, str):
                    if sub_path != projected_path:
                        continue
                    resolved_path = PurePosixPath(mount_path)
                else:
                    resolved_path = PurePosixPath(mount_path) / projected_path
                if not engine_path or resolved_path == PurePosixPath(engine_path):
                    candidates.append(config)
    if engine_path and not candidates:
        raise ValueError(f"engine config path {engine_path!r} does not resolve to a mounted ConfigMap file")
    if len(candidates) > 1:
        raise ValueError("worker mounts multiple ambiguous engine ConfigMap files")
    return candidates[0] if candidates else {}


def _normalize_backend(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.lower().replace("_", "-")
    return {"vllm": "vllm", "sglang": "sglang", "trtllm": "trtllm", "trt-llm": "trtllm"}.get(normalized)


def _role(name: str, service: Mapping[str, Any], flags: Mapping[str, Any]) -> str:
    subcomponent = str(service.get("subComponentType", "")).lower()
    mode = str(flags.get("disaggregation-mode", "")).lower()
    text = f"{name} {subcomponent} {mode}".lower()
    if any(marker in text for marker in ("encode", "epd", "afd", "attentionworker", "ffnworker")):
        raise ValueError(f"unsupported special topology in service {name!r}")
    role_text = f"{subcomponent} {mode}"
    if "prefill" in role_text:
        return "prefill"
    if "decode" in role_text:
        return "decode"
    return "worker"


def _service_records(dgd: Mapping[str, Any], config_maps: Mapping[str, Any]) -> list[_Service]:
    spec = dgd.get("spec", {})
    services = spec.get("services", {}) if isinstance(spec, Mapping) else {}
    if not isinstance(services, Mapping):
        raise TypeError("DynamoGraphDeployment spec.services must be an object")
    records: list[_Service] = []
    for name, raw in services.items():
        if not isinstance(name, str) or not isinstance(raw, Mapping):
            continue
        component = str(raw.get("componentType", "")).lower()
        if component == "main":
            raise ValueError("componentType: main is not supported by adapter v1")
        if component != "worker":
            continue
        main = _container(raw)
        env = {**_env(raw.get("envs", [])), **_env(main.get("env", []))}
        command = _expand_literal(_command(main), env)
        flags = _flags(command)
        records.append(
            _Service(
                name=name,
                role=_role(name, raw, flags),
                config=raw,
                env=env,
                flags=flags,
                engine_config=_mounted_config(raw, config_maps),
            )
        )
    if not records:
        raise ValueError("DynamoGraphDeployment has no worker services")
    if len(records) == 1 and records[0].role == "worker":
        records[0] = _Service(**{**records[0].__dict__, "role": "agg"})
    elif any(record.role == "worker" for record in records):
        raise ValueError("multiple workers require explicit prefill/decode roles")
    roles = [record.role for record in records]
    if roles != ["agg"] and (set(roles) != {"prefill", "decode"} or len(roles) != 2):
        raise ValueError("adapter v1 supports one agg worker or one prefill plus one decode worker")
    return records


def _as_int(value: Any, label: str) -> int:
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


def _as_bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no", ""}:
            return False
    raise ValueError(f"{label} must be a boolean")


def _as_float(value: Any, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error


def _coalesce(label: str, values: Sequence[tuple[str, Any]], default: Any = None) -> Any:
    present = [(source, value) for source, value in values if value is not None]
    if not present:
        return default
    first = str(present[0][1]).lower()
    if any(str(value).lower() != first for _, value in present[1:]):
        detail = ", ".join(f"{source}={value!r}" for source, value in present)
        raise ValueError(f"conflicting {label}: {detail}")
    return present[0][1]


def _flag(service: _Service, *names: str) -> Any:
    values = [(f"--{name}", service.flags.get(name)) for name in names]
    return _coalesce(names[0], values)


def _gpu_limit(config: Mapping[str, Any]) -> int:
    candidates: list[Any] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            limits = value.get("limits")
            if isinstance(limits, Mapping):
                for key in ("gpu", "nvidia.com/gpu"):
                    if key in limits:
                        candidates.append(limits[key])
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(config)
    if not candidates:
        raise ValueError("worker is missing a GPU resource limit")
    normalized = {_as_int(item, "GPU limit") for item in candidates}
    if len(normalized) != 1:
        raise ValueError(f"worker has conflicting GPU limits: {sorted(normalized)}")
    return normalized.pop()


def _node_count(config: Mapping[str, Any]) -> int:
    multinode = config.get("multinode", {})
    if not isinstance(multinode, Mapping) or multinode.get("nodeCount") is None:
        return 1
    return _as_int(multinode.get("nodeCount"), "multinode.nodeCount")


def _system(config: Mapping[str, Any]) -> str | None:
    found: set[str] = set()

    def visit(value: Any, parent_key: str = "") -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                visit(nested, str(key))
        elif isinstance(value, list):
            for nested in value:
                visit(nested, parent_key)
        elif "product" in parent_key.lower() and isinstance(value, str):
            upper = value.upper()
            for marker in sorted(_SYSTEM_MARKERS, key=len, reverse=True):
                if marker in upper:
                    found.add(_SYSTEM_MARKERS[marker])
                    break

    visit(config)
    if len(found) > 1:
        raise ValueError(f"worker selects multiple GPU products: {sorted(found)}")
    return next(iter(found), None)


def _model(service: _Service) -> str | None:
    config_model = service.engine_config.get("model") or service.engine_config.get("model_path")
    return _coalesce(
        "model identity",
        (
            ("--model", _flag(service, "model", "model-path")),
            ("MODEL_PATH", service.env.get("MODEL_PATH")),
            ("engine ConfigMap", config_model),
        ),
    )


def _config_value(service: _Service, name: str) -> Any:
    return service.engine_config.get(name)


def _worker_settings(
    service: _Service,
    *,
    backend: str,
    concurrency: int,
    prefill_batch_override: int | None,
    assumptions: list[str],
) -> WorkerSettingsV1:
    replicas = _as_int(service.config.get("replicas", 1), f"{service.name}.replicas")
    gpus = _gpu_limit(service.config) * _node_count(service.config)
    tp_raw = _coalesce(
        "tensor parallelism",
        (
            ("command", _flag(service, "tensor-parallel-size", "tp-size", "tp")),
            ("engine ConfigMap", _config_value(service, "tensor_parallel_size")),
        ),
    )
    if tp_raw is None:
        if gpus != 1:
            raise ValueError(f"{service.name} does not declare tensor parallelism for {gpus} GPUs")
        tp = 1
        assumptions.append(f"{service.name} tensor parallelism defaults to 1 for a one-GPU worker.")
    else:
        tp = _as_int(tp_raw, f"{service.name} tensor parallelism")
    pp_raw = _coalesce(
        "pipeline parallelism",
        (
            ("command", _flag(service, "pipeline-parallel-size", "pp-size", "pp")),
            ("engine ConfigMap", _config_value(service, "pipeline_parallel_size")),
        ),
    )
    pp = _as_int(pp_raw, f"{service.name} pipeline parallelism") if pp_raw is not None else 1
    if pp_raw is None:
        assumptions.append(f"{service.name} pipeline parallelism defaults to 1.")
    dp_raw = _flag(service, "data-parallel-size", "dp-size", "dp")
    config_attention_dp = _config_value(service, "enable_attention_dp")
    flag_attention_dp = _flag(service, "enable-dp-attention", "enable-attention-dp")
    enabled_attention_dp = _as_bool(
        _coalesce(
            "attention DP",
            (("command", flag_attention_dp), ("engine ConfigMap", config_attention_dp)),
            False,
        ),
        f"{service.name} attention DP",
    )
    if backend == "sglang" and enabled_attention_dp:
        if tp * pp != gpus:
            raise ValueError(f"{service.name} SGLang TP*PP ({tp * pp}) does not match GPUs per replica ({gpus})")
        attention_dp = _as_int(dp_raw, f"{service.name} data parallelism") if dp_raw is not None else tp
        if attention_dp != tp:
            raise ValueError(f"{service.name} SGLang attention-DP must match its world-size TP")
        tp = 1
    elif dp_raw is not None:
        attention_dp = _as_int(dp_raw, f"{service.name} data parallelism")
    elif enabled_attention_dp:
        if gpus % (tp * pp):
            raise ValueError(f"{service.name} GPU width is not divisible by TP*PP")
        attention_dp = gpus // (tp * pp)
    else:
        attention_dp = 1
    if tp * pp * attention_dp != gpus:
        raise ValueError(
            f"{service.name} TP*PP*attention-DP ({tp * pp * attention_dp}) does not match GPUs per replica ({gpus})"
        )

    ep_raw = _coalesce(
        "expert parallelism",
        (
            ("command", _flag(service, "expert-parallel-size", "ep-size", "ep")),
            ("engine ConfigMap", _config_value(service, "expert_parallel_size")),
        ),
    )
    expert_flag = service.flags.get("enable-expert-parallel")
    expert_enabled = _as_bool(expert_flag, f"{service.name} expert parallelism") if expert_flag is not None else False
    if ep_raw is None:
        moe_ep = gpus // pp if expert_enabled else 1
    else:
        moe_ep = _as_int(ep_raw, f"{service.name} expert parallelism")
    if gpus % (pp * moe_ep):
        raise ValueError(f"{service.name} GPU width is not divisible by PP*EP")
    moe_tp = gpus // (pp * moe_ep)

    max_batch = _coalesce(
        "max batch size",
        (
            ("command", _flag(service, "max-batch-size", "max-num-seqs")),
            ("engine ConfigMap", _config_value(service, "max_batch_size")),
        ),
    )
    if service.role == "prefill":
        raw_batch = prefill_batch_override if prefill_batch_override is not None else max_batch
        if raw_batch is None:
            raw_batch = 1
            assumptions.append(f"{service.name} prefill batch size defaults to 1.")
        batch = _as_int(raw_batch, f"{service.name} prefill batch size")
    else:
        denominator = replicas * attention_dp
        if concurrency % denominator:
            raise ValueError(
                f"concurrency ({concurrency}) must be divisible by "
                f"{service.name} replicas * attention DP ({denominator})"
            )
        batch = concurrency // denominator
        if max_batch is not None and batch > _as_int(max_batch, f"{service.name} max batch size"):
            raise ValueError(f"active batch size {batch} exceeds {service.name} max batch size {max_batch}")
    return WorkerSettingsV1(
        replicas=replicas,
        gpus_per_replica=gpus,
        batch_size=batch,
        tp_size=tp,
        pp_size=pp,
        attention_dp_size=attention_dp,
        moe_tp_size=moe_tp,
        moe_ep_size=moe_ep,
    )


def _point_from_mapping(value: Mapping[str, Any], index: int) -> _Point:
    return _Point(
        point_id=str(value.get("point_id", value.get("name", f"point-{index}"))),
        isl=value.get("isl", value.get("ISL")),
        osl=value.get("osl", value.get("OSL")),
        concurrency=value.get("concurrency", value.get("total_concurrency", value.get("TOTAL_CONCURRENCY"))),
        prefix=value.get("prefix", value.get("PREFIX", 0)),
    )


def _performance_containers(documents: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    found: list[Mapping[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            containers = value.get("containers")
            if isinstance(containers, list):
                found.extend(item for item in containers if isinstance(item, Mapping))
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    for document in documents:
        visit(document)
    return found


def _discover_points(documents: Sequence[Mapping[str, Any]], overrides: AdapterOverrides) -> list[_Point]:
    if overrides.workload_points is not None:
        return [
            _Point(point.point_id or f"point-{index}", point.isl, point.osl, point.concurrency, point.prefix or 0)
            for index, point in enumerate(overrides.workload_points)
        ]
    explicit: list[_Point] = []
    for document in documents:
        points = document.get("points")
        if isinstance(points, list):
            for point in points:
                point_index = len(explicit)
                if isinstance(point, Mapping):
                    explicit.append(_point_from_mapping(point, point_index))
                else:
                    explicit.append(
                        _Point(
                            f"point-{point_index}",
                            None,
                            None,
                            None,
                            error=f"workload point {point_index} must be an object",
                        )
                    )
    if explicit:
        return explicit

    for document in documents:
        pipeline = document.get("toolPipeline")
        if not isinstance(pipeline, list):
            continue
        for pipeline_index, item in enumerate(pipeline):
            config = item.get("config") if isinstance(item, Mapping) else None
            if not isinstance(config, Mapping):
                explicit.append(_Point(f"pipeline-{pipeline_index}", None, None, None))
                continue
            raw_concurrencies = config.get("concurrency")
            concurrencies = raw_concurrencies if isinstance(raw_concurrencies, list) else [raw_concurrencies]
            if not concurrencies:
                concurrencies = [None]
            name = config.get("name") or f"pipeline-{pipeline_index}"
            for concurrency in concurrencies:
                explicit.append(
                    _Point(
                        f"{name}-concurrency-{concurrency}",
                        config.get("isl"),
                        config.get("osl"),
                        concurrency,
                        config.get("prefix", 0),
                    )
                )
    if explicit:
        return explicit

    for container in _performance_containers(documents):
        env = _env(container.get("env", []))
        command = _command(container)
        try:
            expanded = _expand_literal(command, env)
        except ValueError:
            expanded = command
        calls = re.findall(r"\brun_perf\s+(\d+)\s+(\d+)\s+(\d+)\b", expanded)
        if calls:
            for concurrency, isl, osl in calls:
                explicit.append(_Point(f"point-{len(explicit)}", isl, osl, concurrency))
            continue
        if "CONCURRENCIES" in env:
            raw_concurrencies = env["CONCURRENCIES"]
            if "$" in raw_concurrencies or "`" in raw_concurrencies:
                explicit.append(
                    _Point(
                        "concurrency-sweep",
                        env.get("ISL"),
                        env.get("OSL"),
                        None,
                        error="CONCURRENCIES must contain literal integers",
                    )
                )
                continue
            concurrencies = raw_concurrencies.split()
            if not concurrencies:
                explicit.append(
                    _Point(
                        "concurrency-sweep",
                        env.get("ISL"),
                        env.get("OSL"),
                        None,
                        error="CONCURRENCIES must contain at least one integer",
                    )
                )
                continue
            explicit.extend(
                _Point(
                    f"concurrency-{concurrency}",
                    env.get("ISL"),
                    env.get("OSL"),
                    concurrency,
                    env.get("PREFIX", 0),
                )
                for concurrency in concurrencies
            )
            continue
        if not any(key in env for key in ("ISL", "OSL", "TOTAL_CONCURRENCY", "CONCURRENCY", "CONCURRENCY_PER_GPU")):
            continue
        concurrency: Any = env.get("TOTAL_CONCURRENCY", env.get("CONCURRENCY"))
        if concurrency is None and "CONCURRENCY_PER_GPU" in env and "DEPLOYMENT_GPU_COUNT" in env:
            try:
                concurrency = int(env["CONCURRENCY_PER_GPU"]) * int(env["DEPLOYMENT_GPU_COUNT"])
            except ValueError:
                concurrency = None
        explicit.append(
            _Point(
                f"point-{len(explicit)}",
                env.get("ISL"),
                env.get("OSL"),
                concurrency,
                env.get("PREFIX", 0),
            )
        )
    if explicit:
        return explicit
    if overrides.isl is not None or overrides.osl is not None or overrides.concurrency is not None:
        return [_Point("point-0", overrides.isl, overrides.osl, overrides.concurrency, overrides.prefix or 0)]
    return [_Point("point-0", None, None, None)]


def _runtime_value(services: Sequence[_Service], flag_names: tuple[str, ...], config_path: tuple[str, ...]) -> Any:
    values: list[tuple[str, Any]] = []
    for service in services:
        value = _flag(service, *flag_names)
        config_value: Any = service.engine_config
        for part in config_path:
            config_value = config_value.get(part) if isinstance(config_value, Mapping) else None
        values.extend(((f"{service.name} command", value), (f"{service.name} ConfigMap", config_value)))
    return _coalesce(flag_names[0], values)


def _request_for_point(
    *,
    point: _Point,
    dgd: Mapping[str, Any],
    services: Sequence[_Service],
    backend: str,
    overrides: AdapterOverrides,
    source: DynamoRecipeSource,
    diagnostics: list[AdaptationDiagnostic],
) -> EstimateRequestV1:
    if point.error is not None:
        raise ValueError(point.error)

    isl = _as_int(overrides.isl if overrides.isl is not None else point.isl, "workload ISL")
    osl = _as_int(overrides.osl if overrides.osl is not None else point.osl, "workload OSL")
    concurrency = _as_int(
        overrides.concurrency if overrides.concurrency is not None else point.concurrency,
        "workload concurrency",
    )
    prefix = int(overrides.prefix if overrides.prefix is not None else point.prefix or 0)

    source_models = {_model(service) for service in services}
    source_models.discard(None)
    if len(source_models) > 1:
        raise ValueError(f"worker services use different models: {sorted(source_models)}")
    model_path = overrides.model_path or next(iter(source_models), None)
    if not model_path:
        raise ValueError("model identity is missing; set MODEL_PATH/--model or provide model_path override")

    source_systems = [_system(service.config) for service in services]
    explicit_systems = {system for system in source_systems if system is not None}
    if len(explicit_systems) > 1:
        raise ValueError(f"heterogeneous hardware is not supported: {sorted(explicit_systems)}")
    system = overrides.system_name or next(iter(explicit_systems), None)
    if not system:
        raise ValueError("GPU system is missing; provide an explicit system_name override")

    assumptions: list[str] = []
    by_role = {service.role: service for service in services}
    if "agg" in by_role:
        topology = AggregatedTopologyV1(
            worker=_worker_settings(
                by_role["agg"],
                backend=backend,
                concurrency=concurrency,
                prefill_batch_override=None,
                assumptions=assumptions,
            )
        )
        assumptions.append(AGGREGATED_REPLICAS_LOWERING_ASSUMPTION)
        systems = SystemSettingsV1(prefill=system)
    else:
        topology = DisaggregatedTopologyV1(
            prefill=_worker_settings(
                by_role["prefill"],
                backend=backend,
                concurrency=concurrency,
                prefill_batch_override=overrides.prefill_batch_size,
                assumptions=assumptions,
            ),
            decode=_worker_settings(
                by_role["decode"],
                backend=backend,
                concurrency=concurrency,
                prefill_batch_override=None,
                assumptions=assumptions,
            ),
        )
        systems = SystemSettingsV1(prefill=system, decode=overrides.decode_system_name or system)

    speculative_flag_names = {"num-speculative-tokens", "speculative-num-steps", "speculative-token-num"}
    speculation_disabled = {"", "0", "false", "none", "disabled"}

    def speculation_enabled(value: Any) -> bool:
        return value is not None and value is not False and str(value).lower() not in speculation_disabled

    speculative_values: list[tuple[str, Any]] = []
    speculative_enable_values: list[Any] = []
    for service in services:
        config = service.engine_config.get("speculative_config")
        speculative_config = config if isinstance(config, Mapping) else {}
        speculative_values.append(
            (
                service.name,
                _coalesce(
                    "speculative depth",
                    (
                        (
                            f"{service.name} command",
                            _flag(
                                service,
                                "num-speculative-tokens",
                                "speculative-num-steps",
                                "speculative-token-num",
                            ),
                        ),
                        (f"{service.name} engine ConfigMap", speculative_config.get("max_draft_len")),
                        (
                            f"{service.name} engine ConfigMap",
                            speculative_config.get("num_nextn_predict_layers"),
                        ),
                    ),
                ),
            )
        )
        speculative_enable_values.extend(
            speculative_config.get(key) for key in ("decoding_type", "speculative_model_dir")
        )
    speculative_enable_values.extend(
        value
        for service in services
        for key, value in service.flags.items()
        if "specul" in key and key not in speculative_flag_names
    )
    active_depths = [
        (service_name, _as_int(value, f"{service_name} speculative depth"))
        for service_name, value in speculative_values
        if speculation_enabled(value)
    ]
    depth_values = {value for _, value in active_depths}
    if len(depth_values) > 1:
        detail = ", ".join(f"{service_name}={value}" for service_name, value in active_depths)
        raise ValueError(f"worker services use different speculative depths: {detail}")
    source_nextn = next(iter(depth_values), None)
    active_speculation = (
        speculation_enabled(overrides.nextn)
        or source_nextn is not None
        or any(speculation_enabled(value) for value in speculative_enable_values)
    )
    if active_speculation and overrides.nextn_accepted is None:
        raise ValueError("speculative decoding requires an explicit nextn_accepted override")
    nextn = overrides.nextn if overrides.nextn is not None else (source_nextn if source_nextn is not None else 0)

    free_fraction = overrides.free_gpu_memory_fraction
    if free_fraction is None:
        raw_fraction = _runtime_value(
            services,
            ("free-gpu-memory-fraction", "gpu-memory-utilization", "mem-fraction-static"),
            ("kv_cache_config", "free_gpu_memory_fraction"),
        )
        free_fraction = _as_float(raw_fraction, "free GPU memory fraction") if raw_fraction is not None else None
    max_seq_len = overrides.max_seq_len
    if max_seq_len is None:
        raw_max_seq = _runtime_value(
            services,
            ("max-model-len", "max-seq-len", "context-length"),
            ("max_seq_len",),
        )
        max_seq_len = _as_int(raw_max_seq, "max sequence length") if raw_max_seq is not None else None

    metadata = dgd.get("metadata", {})
    deployment_name = metadata.get("name") if isinstance(metadata, Mapping) else None
    return EstimateRequestV1(
        model=ModelSettingsV1(path=str(model_path), nextn=nextn, nextn_accepted=overrides.nextn_accepted),
        quantization=QuantizationSettingsV1(
            gemm=overrides.gemm_quant_mode,
            kvcache=overrides.kvcache_quant_mode,
            fmha=overrides.fmha_quant_mode,
            moe=overrides.moe_quant_mode,
            communication=overrides.comm_quant_mode,
        ),
        backend=BackendSettingsV1(
            name=backend,
            version=overrides.backend_version,
            database_mode=overrides.database_mode or "SILICON",
        ),
        systems=systems,
        workload=WorkloadSettingsV1(isl=isl, osl=osl, concurrency=concurrency, prefix=prefix),
        topology=topology,
        runtime=RuntimeSettingsV1(
            systems_paths=overrides.systems_paths,
            free_gpu_memory_fraction=free_fraction,
            max_seq_len=max_seq_len,
            engine_step_backend=overrides.engine_step_backend,
        ),
        provenance=SourceProvenanceV1(
            source_type="dynamo",
            source_reference=source.source_reference,
            source_ids={"deployment": deployment_name, "operating_point": point.point_id},
            assumptions=tuple(dict.fromkeys(assumptions)),
        ),
    )


def adapt_dynamo(source: DynamoRecipeSource, overrides: AdapterOverrides) -> AdaptationReport:
    try:
        deployment_documents = _documents(source.deployment)
        performance_documents = _documents(source.performance)
        points = _discover_points(performance_documents, overrides)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as error:
        diagnostic = AdaptationDiagnostic(
            severity="error",
            code="dynamo_parse_failed",
            message=str(error),
            hint="Use safe, literal YAML values from a standard Dynamo recipe.",
        )
        return AdaptationReport(
            outcomes=(AdaptationOutcome(point_id="point-0", status="rejected", diagnostics=(diagnostic,)),)
        )

    if len(deployment_documents) == 1 and is_dynamo_ci_recipe(deployment_documents[0]):
        source_reference = source.source_reference
        if source_reference is None and isinstance(source.deployment, Path):
            source_reference = str(source.deployment)
        return adapt_dynamo_ci(
            deployment_documents[0],
            overrides,
            source_reference=source_reference,
        )

    try:
        dgds = [document for document in deployment_documents if document.get("kind") == "DynamoGraphDeployment"]
        if len(dgds) != 1:
            raise ValueError(f"expected exactly one DynamoGraphDeployment, found {len(dgds)}")
        dgd = dgds[0]
        config_maps = _config_maps(deployment_documents)
        services = _service_records(dgd, config_maps)
        spec = dgd.get("spec", {})
        declared_backend = _normalize_backend(spec.get("backendFramework") if isinstance(spec, Mapping) else None)
        command_backends: set[str] = set()
        for service in services:
            command_backends.update(_BACKEND_MODULE_PATTERN.findall(_command(_container(service.config))))
        if len(command_backends) > 1:
            raise ValueError(f"workers use conflicting backends: {sorted(command_backends)}")
        command_backend = next(iter(command_backends), None)
        if declared_backend and command_backend and declared_backend != command_backend:
            raise ValueError(
                f"backendFramework {declared_backend!r} conflicts with worker command backend {command_backend!r}"
            )
        backend = overrides.backend_name or declared_backend or command_backend
        if backend not in {"vllm", "sglang", "trtllm"}:
            raise ValueError(f"unsupported or missing Dynamo backend {backend!r}")
    except (ValueError, TypeError, yaml.YAMLError) as error:
        diagnostic = AdaptationDiagnostic(
            severity="error",
            code="dynamo_topology_rejected",
            message=str(error),
            hint="Use a standard agg or P/D-disaggregated vLLM, SGLang, or TRT-LLM DGD.",
        )
        return AdaptationReport(
            outcomes=tuple(
                AdaptationOutcome(point_id=point.point_id, status="rejected", diagnostics=(diagnostic,))
                for point in points
            )
        )

    outcomes: list[AdaptationOutcome] = []
    for point in points:
        diagnostics: list[AdaptationDiagnostic] = []
        if overrides.backend_version is None:
            diagnostics.append(
                AdaptationDiagnostic(
                    severity="warning",
                    code="backend_version_unpinned",
                    message="Backend version is not pinned; AIC will select its latest compatible database version.",
                    path="backend.version",
                )
            )
        try:
            request = _request_for_point(
                point=point,
                dgd=dgd,
                services=services,
                backend=backend,
                overrides=overrides,
                source=source,
                diagnostics=diagnostics,
            )
        except (ValueError, TypeError, ValidationError) as error:
            diagnostics.append(
                AdaptationDiagnostic(
                    severity="error",
                    code="dynamo_mapping_failed",
                    message=str(error),
                    hint=(
                        "Provide explicit overrides for missing model, system, workload, "
                        "or speculative acceptance values."
                    ),
                )
            )
            outcomes.append(
                AdaptationOutcome(point_id=point.point_id, status="rejected", diagnostics=tuple(diagnostics))
            )
            continue
        outcomes.append(
            AdaptationOutcome(
                point_id=point.point_id,
                status="adapted",
                request=request,
                diagnostics=tuple(diagnostics),
            )
        )
    return AdaptationReport(outcomes=tuple(outcomes))

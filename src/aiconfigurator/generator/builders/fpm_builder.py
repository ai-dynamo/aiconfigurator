# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the minimal V1 artifacts used for FPM collection.

The FPM resource pod deliberately does not launch an engine.  It reserves the
same infrastructure as the normal vLLM worker and stays alive while an agent
streams the generated ``run.sh`` into it, potentially more than once.
"""

from __future__ import annotations

import copy
import re
import shlex
from typing import Any

from .dgd_model import DGD, DGDService, MainContainer, _dump_k8s_yaml
from .k8s_builder import build_dgd

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MISSING = object()


def build_fpm_artifacts(
    context: dict[str, Any],
    backend: str,
    resolved_facts: Any = None,
    param_values: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Return a reusable resource Pod and a complete FPM engine script.

    V1 intentionally supports only one single-node aggregated vLLM worker.
    The existing DGD builder remains the source of truth for infrastructure;
    this function lowers its worker service to a plain Pod while moving every
    concrete environment variable and the engine command into ``run.sh``.
    """
    if backend != "vllm":
        raise ValueError("FPM V1 supports only the vllm backend")

    dyn_config = context.get("DynConfig") or {}
    if not isinstance(dyn_config, dict) or dyn_config.get("mode") != "agg":
        raise ValueError("FPM V1 supports only DynConfig.mode=agg")
    unsupported_dyn_features = [
        key for key in ("enable_router", "router_mode", "router_config", "planner_config") if dyn_config.get(key)
    ]
    if unsupported_dyn_features:
        fields = ", ".join(f"DynConfig.{key}" for key in unsupported_dyn_features)
        raise ValueError(f"FPM V1 does not support router or planner configuration: {fields}")

    extra_cli_args = _extract_extra_cli_args(param_values)
    worker = _build_single_worker(context, backend, resolved_facts)
    main_container = _require_main_container(worker)

    command = list(main_container.command or [])
    args = list(main_container.args or [])
    if command[:3] != ["python3", "-m", "dynamo.vllm"]:
        raise ValueError("FPM V1 requires the normal vLLM worker command")
    if not all(isinstance(token, str) for token in command + args):
        raise ValueError("The resolved vLLM command must contain only string tokens")
    args.extend(extra_cli_args)

    env = _collect_concrete_env(worker, main_container)
    _require_cli_option(args, "--benchmark-mode", expected="agg")
    benchmark_output_path = _ensure_benchmark_output_path(args, env)
    wait_timeout_seconds = _benchmark_wait_timeout_seconds(args)
    run_script = _render_run_script(
        command + args,
        env,
        benchmark_output_path,
        wait_timeout_seconds,
    )
    pod = _lower_worker_to_pod(context, worker, main_container)

    return {
        "k8s_deploy.yaml": _dump_k8s_yaml(pod),
        "run.sh": run_script,
    }


def _extract_extra_cli_args(param_values: dict[str, Any] | None) -> list[str]:
    if param_values is None:
        return []
    if not isinstance(param_values, dict):
        raise TypeError("param_values must be a mapping")

    params = param_values.get("params") or {}
    if not isinstance(params, dict):
        raise TypeError("param_values.params must be a mapping")
    agg = params.get("agg") or {}
    if not isinstance(agg, dict):
        raise TypeError("param_values.params.agg must be a mapping")

    value = agg.get("extra_cli_args", _MISSING)
    if value is _MISSING:
        return []
    if not isinstance(value, list) or not all(isinstance(token, str) for token in value):
        raise ValueError("params.agg.extra_cli_args must be a list[str]")
    return list(value)


def _build_single_worker(context: dict[str, Any], backend: str, resolved_facts: Any) -> DGDService:
    docs = build_dgd(context, backend, resolved_facts=resolved_facts)
    dgd_docs = [doc for doc in docs if isinstance(doc, DGD)]
    if len(dgd_docs) != 1:
        raise ValueError("FPM V1 requires exactly one DynamoGraphDeployment document")

    workers = [(name, service) for name, service in dgd_docs[0].services.items() if service.component_type == "worker"]
    if len(workers) != 1 or workers[0][0] != "VllmWorker":
        raise ValueError("FPM V1 requires exactly one aggregated VllmWorker")

    worker = workers[0][1]
    if worker.replicas != 1:
        raise ValueError("FPM V1 requires worker replicas=1")
    if worker.multinode:
        raise ValueError("FPM V1 does not support multinode workers")
    if worker.resources and worker.resources.get("claims"):
        raise ValueError("FPM V1 does not support resource claims")
    if worker.extra_pod_spec and worker.extra_pod_spec.resource_claims:
        raise ValueError("FPM V1 does not support Pod resourceClaims")
    return worker


def _require_main_container(worker: DGDService) -> MainContainer:
    pod_spec = worker.extra_pod_spec
    if pod_spec is None or pod_spec.main_container is None:
        raise ValueError("The resolved vLLM worker has no main container")
    return pod_spec.main_container


def _collect_concrete_env(worker: DGDService, main_container: MainContainer) -> list[tuple[str, str]]:
    resolved: list[tuple[str, str]] = []
    entries = list(worker.envs or []) + list(main_container.env or [])
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("FPM environment entries must be mappings")
        if "valueFrom" in entry:
            raise ValueError("FPM V1 does not support valueFrom environment entries")

        name = entry.get("name")
        if not isinstance(name, str) or not _ENV_NAME_RE.fullmatch(name):
            raise ValueError(f"Invalid shell environment variable name: {name!r}")
        if "value" not in entry or entry["value"] is None:
            raise ValueError(f"Environment variable {name} must have a concrete value")

        value = entry["value"]
        if isinstance(value, bool):
            resolved.append((name, "true" if value else "false"))
        elif isinstance(value, (str, int, float)):
            resolved.append((name, str(value)))
        else:
            raise TypeError(f"Environment variable {name} must have a scalar value")
    return resolved


def _cli_option_value(args: list[str], flag: str) -> str | None:
    value: str | None = None
    for index, token in enumerate(args):
        if token == flag:
            if index + 1 >= len(args):
                raise ValueError(f"{flag} requires a value")
            candidate = args[index + 1]
            if candidate.startswith("--"):
                raise ValueError(f"{flag} requires a value")
            value = candidate
        elif token.startswith(f"{flag}="):
            value = token.split("=", 1)[1]
    return value


def _require_cli_option(args: list[str], flag: str, *, expected: str | None = None) -> None:
    value = _cli_option_value(args, flag)
    if value is None:
        raise ValueError(f"FPM V1 requires {flag}")
    if expected is not None and value != expected:
        raise ValueError(f"FPM V1 requires {flag} {expected}")


def _last_env_value(env: list[tuple[str, str]], name: str) -> str | None:
    for env_name, value in reversed(env):
        if env_name == name:
            return value
    return None


def _ensure_benchmark_output_path(args: list[str], env: list[tuple[str, str]]) -> str:
    flag = "--benchmark-output-path"
    cli_value = _cli_option_value(args, flag)
    env_value = _last_env_value(env, "DYN_FPM_BENCHMARK_OUTPUT_PATH")
    if cli_value is not None and env_value is not None and cli_value != env_value:
        raise ValueError(f"{flag} and DYN_FPM_BENCHMARK_OUTPUT_PATH must resolve to the same path")
    value = cli_value or env_value

    if value is None:
        value = "/results/benchmark.json"
    if cli_value is None:
        # Waiting for an output path that the engine does not know about would
        # hang forever.  Make the V1 default explicit in the resolved command.
        args.extend([flag, value])
    if env_value is None:
        env.append(("DYN_FPM_BENCHMARK_OUTPUT_PATH", value))
    if not value:
        raise ValueError(f"{flag} must not be empty")
    return value


def _benchmark_wait_timeout_seconds(args: list[str]) -> int:
    raw = _cli_option_value(args, "--benchmark-timeout")
    if raw is None:
        return 7800
    try:
        benchmark_timeout = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("--benchmark-timeout must be an integer number of seconds") from exc
    if benchmark_timeout <= 0:
        raise ValueError("--benchmark-timeout must be positive")
    # Give the engine time to initialize and flush the final result after its
    # own collector deadline expires.
    return benchmark_timeout + 600


def _render_run_script(
    command: list[str],
    env: list[tuple[str, str]],
    benchmark_output_path: str,
    wait_timeout_seconds: int,
) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -Eeuo pipefail",
        "",
        "ulimit -l unlimited || true",
        "ulimit -n 1048576 || true",
    ]
    for name, value in env:
        lines.append(f"export {name}={shlex.quote(value)}")

    lines.extend(
        [
            "",
            f"benchmark_output_path={shlex.quote(benchmark_output_path)}",
            f"wait_timeout_seconds={wait_timeout_seconds}",
            'if [[ -e "$benchmark_output_path" || -L "$benchmark_output_path" ]]; then',
            '  echo "Refusing to overwrite existing benchmark output: $benchmark_output_path" >&2',
            "  exit 1",
            "fi",
            'mkdir -p -- "$(dirname -- "$benchmark_output_path")"',
            "",
            'engine_pid=""',
            "cleanup() {",
            "  local status=$?",
            "  trap - EXIT INT TERM",
            '  if [[ -n "${engine_pid:-}" ]] && kill -0 "$engine_pid" 2>/dev/null; then',
            '    kill -TERM "$engine_pid" 2>/dev/null || true',
            '    wait "$engine_pid" 2>/dev/null || true',
            "  fi",
            '  exit "$status"',
            "}",
            "trap cleanup EXIT",
            "trap 'exit 130' INT",
            "trap 'exit 143' TERM",
            "",
            f"{' '.join(shlex.quote(token) for token in command)} &",
            "engine_pid=$!",
            "deadline=$((SECONDS + wait_timeout_seconds))",
            "",
            "while true; do",
            (
                '  if [[ -s "$benchmark_output_path" ]] && '
                "python3 -c 'import json, sys; json.load(open(sys.argv[1], encoding=\"utf-8\"))' "
                '"$benchmark_output_path" 2>/dev/null; then'
            ),
            "    break",
            "  fi",
            '  if ! kill -0 "$engine_pid" 2>/dev/null; then',
            "    set +e",
            '    wait "$engine_pid"',
            "    engine_status=$?",
            "    set -e",
            '    engine_pid=""',
            '    echo "Engine exited before writing benchmark output: $benchmark_output_path" >&2',
            '    if (( engine_status == 0 )); then exit 1; else exit "$engine_status"; fi',
            "  fi",
            "  if (( SECONDS >= deadline )); then",
            '    echo "Timed out waiting for benchmark output: $benchmark_output_path" >&2',
            "    exit 124",
            "  fi",
            "  sleep 2",
            "done",
            "",
            'kill -TERM "$engine_pid" 2>/dev/null || true',
            'wait "$engine_pid" 2>/dev/null || true',
            'engine_pid=""',
            "trap - EXIT INT TERM",
            "",
        ]
    )
    return "\n".join(lines)


def _lower_worker_to_pod(
    context: dict[str, Any],
    worker: DGDService,
    main_container: MainContainer,
) -> dict[str, Any]:
    extra_pod_spec = worker.extra_pod_spec
    if extra_pod_spec is None:
        raise ValueError("The resolved vLLM worker has no extraPodSpec")

    pod_spec = extra_pod_spec.to_dict()
    pod_spec.pop("mainContainer", None)
    if pod_spec.get("resourceClaims"):
        raise ValueError("FPM V1 does not support Pod resourceClaims")
    pod_spec.pop("resourceClaims", None)

    container = main_container.to_dict()
    if container.get("envFrom"):
        raise ValueError("FPM V1 does not support mainContainer.envFrom")
    for key in (
        "command",
        "args",
        "env",
        "envFrom",
        "startupProbe",
        "livenessProbe",
        "readinessProbe",
        "lifecycle",
        "resources",
    ):
        container.pop(key, None)
    if not container.get("image"):
        raise ValueError("The resolved vLLM worker has no container image")

    volumes = copy.deepcopy(pod_spec.get("volumes") or [])
    volume_mounts = copy.deepcopy(container.get("volumeMounts") or [])
    if not isinstance(volumes, list):
        raise TypeError("Worker volumes must be a list")
    if not isinstance(volume_mounts, list):
        raise TypeError("Worker volumeMounts must be a list")
    _add_volume_mount(
        volumes,
        volume_mounts,
        name="results",
        mount_path="/results",
        volume_source={"emptyDir": {}},
    )

    # A vLLM resource pod always needs a real /dev/shm mount; when the normal
    # DGD resolved a hardware-specific size, preserve it as sizeLimit.
    shared_memory = worker.shared_memory
    empty_dir: dict[str, Any] = {"medium": "Memory"}
    if shared_memory is not None:
        if not isinstance(shared_memory, dict):
            raise ValueError("sharedMemory must be a mapping")
        size = shared_memory.get("size")
        if size:
            empty_dir["sizeLimit"] = size
    _add_volume_mount(
        volumes,
        volume_mounts,
        name="dshm",
        mount_path="/dev/shm",
        volume_source={"emptyDir": empty_dir},
    )

    container.update(
        {
            "name": "fpm-resource",
            "resources": _lower_resources(worker.resources),
            "volumeMounts": volume_mounts,
            "command": ["/bin/bash", "-lc"],
            "args": ["exec sleep infinity"],
        }
    )
    pod_spec["volumes"] = volumes
    pod_spec["containers"] = [container]
    pod_spec["restartPolicy"] = "Always"

    k8s = context.get("K8sConfig") or {}
    if not isinstance(k8s, dict):
        raise TypeError("K8sConfig must be a mapping")
    name = context.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("FPM resource Pod requires a non-empty context name")

    metadata: dict[str, Any] = {
        "name": name,
        "labels": {
            "app.kubernetes.io/name": name,
            "app.kubernetes.io/component": "fpm-resource",
        },
    }
    namespace = k8s.get("k8s_namespace")
    if namespace:
        metadata["namespace"] = namespace

    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": metadata,
        "spec": pod_spec,
    }


def _lower_resources(resources: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(resources, dict):
        raise TypeError("The resolved vLLM worker has no resources")
    if resources.get("claims"):
        raise ValueError("FPM V1 does not support resource claims")

    lowered: dict[str, Any] = {}
    for section_name in ("limits", "requests"):
        section = resources.get(section_name)
        if section is None:
            continue
        if not isinstance(section, dict):
            raise TypeError(f"resources.{section_name} must be a mapping")
        section = copy.deepcopy(section)
        custom = section.pop("custom", None)
        gpu = section.pop("gpu", None)
        if gpu is not None:
            section["nvidia.com/gpu"] = str(gpu)
        if custom is not None:
            if not isinstance(custom, dict):
                raise TypeError(f"resources.{section_name}.custom must be a mapping")
            section.update(copy.deepcopy(custom))
        if section:
            lowered[section_name] = section

    limits = lowered.get("limits") or {}
    if "nvidia.com/gpu" not in limits:
        raise ValueError("FPM resource Pod requires a GPU limit")
    return lowered


def _add_volume_mount(
    volumes: list[Any],
    volume_mounts: list[Any],
    *,
    name: str,
    mount_path: str,
    volume_source: dict[str, Any],
) -> None:
    for volume in volumes:
        if isinstance(volume, dict) and volume.get("name") == name:
            raise ValueError(f"worker_extra_pod_spec already defines reserved volume {name}")
    for mount in volume_mounts:
        if not isinstance(mount, dict):
            raise TypeError("Worker volumeMount entries must be mappings")
        if mount.get("name") == name or mount.get("mountPath") == mount_path:
            raise ValueError(f"worker_extra_pod_spec conflicts with reserved mount {mount_path}")

    volumes.append({"name": name, **copy.deepcopy(volume_source)})
    volume_mounts.append({"name": name, "mountPath": mount_path})

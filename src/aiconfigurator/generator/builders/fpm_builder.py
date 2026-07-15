# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the artifacts used for FPM collection.

The FPM resource Pod or LeaderWorkerSet deliberately does not launch an engine.
It reserves the same infrastructure as the normal vLLM worker and stays alive
while a collector streams the generated ``run.sh`` into it.
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
_NODE_RANK_SENTINEL = "__FPM_NODE_RANK__"
_FPM_BENCHMARK_MODES = ("agg", "prefill", "decode")
_FPM_OWNED_ORCHESTRATION_FLAGS = frozenset(
    {
        "--nnodes",
        "--node-rank",
        "--master-addr",
        "--master-port",
        "--headless",
        "--data-parallel-size-local",
        "--data-parallel-start-rank",
        "--data-parallel-address",
        "--data-parallel-rpc-port",
        "--data-parallel-hybrid-lb",
    }
)


def build_fpm_artifacts(
    context: dict[str, Any],
    backend: str,
    resolved_facts: Any = None,
    param_values: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Return a reusable resource workload and a complete FPM engine script.

    The existing DGD builder remains the source of truth for infrastructure.
    This function lowers its one aggregated worker to a Pod (single node) or a
    LeaderWorkerSet (multiple nodes), while moving every concrete environment
    variable and the engine command into ``run.sh``.
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
    _reject_owned_orchestration_args(extra_cli_args)
    worker = _build_worker(context, backend, resolved_facts)
    main_container = _require_main_container(worker)

    command = list(main_container.command or [])
    args = list(main_container.args or [])
    if command[:3] != ["python3", "-m", "dynamo.vllm"]:
        raise ValueError("FPM V1 requires the normal vLLM worker command")
    if not all(isinstance(token, str) for token in command + args):
        raise ValueError("The resolved vLLM command must contain only string tokens")
    args.extend(extra_cli_args)

    env = _collect_concrete_env(worker, main_container)
    benchmark_mode = _require_benchmark_mode(args)
    topology = _resolve_topology(context, worker, args)
    if topology["data_parallel_size"] > 1 and _cli_option_value(args, "--data-parallel-backend") is None:
        args.extend(["--data-parallel-backend", "mp"])
    if topology["data_parallel_size"] > 1:
        _require_cli_option(args, "--data-parallel-backend", expected="mp")
    _ensure_dump_config_path(args, topology["node_count"])
    benchmark_output_path = _ensure_benchmark_output_path(args, env)
    wait_timeout_seconds = _benchmark_wait_timeout_seconds(args)
    run_script = _render_run_script(
        command + args,
        env,
        benchmark_mode,
        benchmark_output_path,
        wait_timeout_seconds,
        topology,
    )
    workload = _lower_worker_to_resource(
        context,
        worker,
        main_container,
        topology,
        efa_resource_name=_efa_resource_name(resolved_facts),
    )

    return {
        "k8s_deploy.yaml": _dump_k8s_yaml(workload),
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


def _reject_owned_orchestration_args(args: list[str]) -> None:
    for token in args:
        for flag in _FPM_OWNED_ORCHESTRATION_FLAGS:
            if token == flag or token.startswith(f"{flag}="):
                raise ValueError(f"FPM owns orchestration option {flag}; do not pass it through extra_cli_args")


def _ensure_dump_config_path(args: list[str], node_count: int) -> None:
    flag = "--dump-config-to"
    occurrences: list[tuple[int, bool]] = []
    for index, token in enumerate(args):
        if token == flag:
            if index + 1 >= len(args) or args[index + 1].startswith("--"):
                raise ValueError(f"{flag} requires a value")
            occurrences.append((index + 1, False))
        elif token.startswith(f"{flag}="):
            occurrences.append((index, True))
    if len(occurrences) > 1:
        raise ValueError(f"FPM accepts at most one {flag} option")

    default = (
        "/results/resolved-config-node{node_rank}.json" if node_count > 1 else "/results/resolved-config-node0.json"
    )
    if not occurrences:
        value = default
        args.extend([flag, value])
        occurrences.append((len(args) - 1, False))

    index, joined = occurrences[0]
    value = args[index].split("=", 1)[1] if joined else args[index]
    if node_count > 1 and "{node_rank}" not in value:
        raise ValueError(f"Multinode FPM {flag} must contain the {{node_rank}} placeholder")
    if node_count == 1:
        value = value.replace("{node_rank}", "0")
    else:
        value = value.replace("{node_rank}", _NODE_RANK_SENTINEL)
    args[index] = f"{flag}={value}" if joined else value


def _build_worker(context: dict[str, Any], backend: str, resolved_facts: Any) -> DGDService:
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
    return worker


def _positive_cli_int(args: list[str], flag: str, *, default: int = 1) -> int:
    raw = _cli_option_value(args, flag)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{flag} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{flag} must be a positive integer")
    return value


def _worker_gpu_limit(resources: dict[str, Any] | None) -> int:
    if not isinstance(resources, dict):
        raise TypeError("The resolved vLLM worker has no resources")
    limits = resources.get("limits")
    if not isinstance(limits, dict):
        raise TypeError("The resolved vLLM worker has no GPU limits")
    raw = limits.get("gpu")
    if raw is None:
        custom = limits.get("custom")
        if isinstance(custom, dict):
            raw = custom.get("nvidia.com/gpu")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("The resolved vLLM worker has an invalid GPU limit") from exc
    if value <= 0:
        raise ValueError("The resolved vLLM worker GPU limit must be positive")
    return value


def _resolve_topology(context: dict[str, Any], worker: DGDService, args: list[str]) -> dict[str, int]:
    total_gpus = _worker_gpu_limit(worker.resources)
    multinode = worker.multinode
    if multinode is None:
        node_count = 1
    elif isinstance(multinode, dict):
        try:
            node_count = int(multinode.get("nodeCount"))
        except (TypeError, ValueError) as exc:
            raise ValueError("FPM multinode.nodeCount must be a positive integer") from exc
        if node_count <= 1:
            raise ValueError("FPM multinode.nodeCount must be greater than one")
    else:
        raise TypeError("FPM worker.multinode must be a mapping")

    if total_gpus % node_count:
        raise ValueError("FPM total GPU count must be divisible by node count")
    gpus_per_node = total_gpus // node_count
    configured_gpus_per_node = int((context.get("NodeConfig") or {}).get("num_gpus_per_node") or gpus_per_node)
    if node_count > 1 and gpus_per_node > configured_gpus_per_node:
        raise ValueError("FPM per-node GPU count exceeds NodeConfig.num_gpus_per_node")

    tensor_parallel_size = _positive_cli_int(args, "--tensor-parallel-size")
    pipeline_parallel_size = _positive_cli_int(args, "--pipeline-parallel-size")
    data_parallel_size = _positive_cli_int(args, "--data-parallel-size")
    expected_gpus = tensor_parallel_size * pipeline_parallel_size * data_parallel_size
    if expected_gpus != total_gpus:
        raise ValueError(
            "FPM topology does not match the resolved GPU count: "
            f"tp({tensor_parallel_size}) * pp({pipeline_parallel_size}) * dp({data_parallel_size}) "
            f"!= gpus({total_gpus})"
        )

    if data_parallel_size > 1:
        if data_parallel_size % node_count:
            raise ValueError("FPM data parallel size must be divisible by node count")
        local_data_parallel_size = data_parallel_size // node_count
    else:
        local_data_parallel_size = 1
    if (
        data_parallel_size > 1
        and local_data_parallel_size * tensor_parallel_size * pipeline_parallel_size > gpus_per_node
    ):
        raise ValueError("FPM local parallel topology exceeds the per-node GPU count")

    return {
        "node_count": node_count,
        "total_gpus": total_gpus,
        "gpus_per_node": gpus_per_node,
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "data_parallel_size": data_parallel_size,
        "local_data_parallel_size": local_data_parallel_size,
    }


def _require_main_container(worker: DGDService) -> MainContainer:
    pod_spec = worker.extra_pod_spec
    if pod_spec is None or pod_spec.main_container is None:
        raise ValueError("The resolved vLLM worker has no main container")
    return pod_spec.main_container


def _collect_concrete_env(worker: DGDService, main_container: MainContainer) -> list[tuple[str, str]]:
    # build_dgd sets the operator-only envFromSecret="hf-token-secret" on
    # every vLLM worker. FPM V1 intentionally accepts only concrete values:
    # it cannot safely materialize a Secret into run.sh, and rejecting that
    # built-in marker would make every FPM render fail.
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


def _require_benchmark_mode(args: list[str]) -> str:
    flag = "--benchmark-mode"
    value = _cli_option_value(args, flag)
    if value is None:
        raise ValueError(f"FPM V1 requires {flag}")
    if value not in _FPM_BENCHMARK_MODES:
        choices = ", ".join(_FPM_BENCHMARK_MODES)
        raise ValueError(f"FPM V1 requires {flag} to be one of: {choices}")
    return value


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
    value = cli_value if cli_value is not None else env_value

    if value is None:
        value = "/results/benchmark.json"
    if not value:
        raise ValueError(f"{flag} must not be empty")
    if cli_value is None:
        # Waiting for an output path that the engine does not know about would
        # hang forever.  Make the V1 default explicit in the resolved command.
        args.extend([flag, value])
    if env_value is None:
        env.append(("DYN_FPM_BENCHMARK_OUTPUT_PATH", value))
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
    benchmark_mode: str,
    benchmark_output_path: str,
    wait_timeout_seconds: int,
    topology: dict[str, int],
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
            f"node_count={topology['node_count']}",
            f"data_parallel_size={topology['data_parallel_size']}",
            f"local_data_parallel_size={topology['local_data_parallel_size']}",
            "if (( node_count > 1 )); then",
            '  node_rank="${LWS_WORKER_INDEX:?LWS_WORKER_INDEX is required for multinode FPM}"',
            '  master_addr="${LWS_LEADER_ADDRESS:?LWS_LEADER_ADDRESS is required for multinode FPM}"',
            "else",
            '  node_rank="${LWS_WORKER_INDEX:-0}"',
            '  master_addr="${LWS_LEADER_ADDRESS:-127.0.0.1}"',
            "fi",
            'if ! [[ "$node_rank" =~ ^[0-9]+$ ]] || (( node_rank >= node_count )); then',
            '  echo "Invalid FPM node rank: $node_rank (node_count=$node_count)" >&2',
            "  exit 2",
            "fi",
            'export DYN_FPM_WORKER_ID="${DYN_FPM_WORKER_ID:-${FPM_RUN_ID:-fpm}-node${node_rank}}"',
            f"benchmark_mode={shlex.quote(benchmark_mode)}",
            f"benchmark_output_path={shlex.quote(benchmark_output_path)}",
            f"wait_timeout_seconds={wait_timeout_seconds}",
            f"engine_command=({' '.join(shlex.quote(token) for token in command)})",
            'for index in "${!engine_command[@]}"; do',
            f'  engine_command[$index]="${{engine_command[$index]//{_NODE_RANK_SENTINEL}/$node_rank}}"',
            "done",
            "",
            "if (( node_count > 1 )); then",
            "  if (( data_parallel_size > 1 )); then",
            '    engine_command+=(--data-parallel-size-local "$local_data_parallel_size")',
            '    engine_command+=(--data-parallel-start-rank "$((node_rank * local_data_parallel_size))")',
            '    engine_command+=(--data-parallel-address "$master_addr" --data-parallel-rpc-port 29510)',
            "    engine_command+=(--data-parallel-hybrid-lb)",
            "  else",
            '    engine_command+=(--nnodes "$node_count" --node-rank "$node_rank")',
            '    engine_command+=(--master-addr "$master_addr" --master-port 29500)',
            "    if (( node_rank > 0 )); then",
            '      exec "${engine_command[@]}" --headless',
            "    fi",
            "  fi",
            "fi",
            "",
            "benchmark_path_for_dp_rank() {",
            "  local dp_rank=$1",
            '  local directory=""',
            '  local filename="$benchmark_output_path"',
            '  if [[ "$benchmark_output_path" == */* ]]; then',
            '    directory="${benchmark_output_path%/*}/"',
            '    filename="${benchmark_output_path##*/}"',
            "  fi",
            "  if (( dp_rank == 0 )); then",
            '    printf "%s\\n" "$benchmark_output_path"',
            '  elif [[ "$filename" == *.* ]]; then',
            '    printf "%s%s_dp%s.%s\\n" "$directory" "${filename%.*}" "$dp_rank" "${filename##*.}"',
            "  else",
            '    printf "%s%s_dp%s\\n" "$directory" "$filename" "$dp_rank"',
            "  fi",
            "}",
            "",
            "expected_results=()",
            "local_dp_start=$((node_rank * local_data_parallel_size))",
            "local_dp_end=$((local_dp_start + local_data_parallel_size))",
            "for ((dp_rank=local_dp_start; dp_rank<local_dp_end; dp_rank++)); do",
            '  expected_results+=("$(benchmark_path_for_dp_rank "$dp_rank")")',
            "done",
            'for path in "${expected_results[@]}"; do',
            '  if [[ -e "$path" || -L "$path" ]]; then',
            '    echo "Refusing to overwrite existing benchmark output: $path" >&2',
            "    exit 1",
            "  fi",
            '  mkdir -p -- "$(dirname -- "$path")"',
            "done",
            "",
            "check_result_files() {",
            '  python3 - "$local_dp_start" "$benchmark_mode" "${expected_results[@]}" <<\'PY\'',
            "import json",
            "import pathlib",
            "import sys",
            "",
            "start_rank = int(sys.argv[1])",
            "expected_mode = sys.argv[2]",
            'allowed_point_types = {"prefill", "decode"} if expected_mode == "agg" else {expected_mode}',
            "",
            "def invalid(path, message):",
            '    print(f"Invalid FPM benchmark result {path}: {message}", file=sys.stderr)',
            "    raise SystemExit(20)",
            "",
            "def validate_schema_v1(path, value, expected_rank):",
            '    if value.get("status") != "complete" or value.get("valid") is not True:',
            "        invalid(path,",
            "                f\"schema_version=1 status={value.get('status')!r} \"",
            "                f\"valid={value.get('valid')!r} errors={value.get('errors')!r}\")",
            '    config = value.get("config")',
            '    actual_mode = config.get("mode") if isinstance(config, dict) else None',
            "    if actual_mode != expected_mode:",
            '        invalid(path, f"benchmark mode {actual_mode!r} != {expected_mode!r}")',
            '    coverage = value.get("coverage")',
            "    if not isinstance(coverage, dict):",
            '        invalid(path, "coverage must be an object")',
            '    expected_points = coverage.get("expected_points")',
            '    completed_points = coverage.get("completed_points")',
            '    skipped_points = coverage.get("skipped_points")',
            "    if (type(expected_points) is not int or expected_points <= 0",
            "            or type(completed_points) is not int or completed_points != expected_points",
            "            or type(skipped_points) is not int or skipped_points != 0):",
            '        invalid(path, f"invalid coverage {coverage!r}")',
            '    results = value.get("results")',
            "    result_count = len(results) if isinstance(results, list) else None",
            "    if not isinstance(results, list) or result_count != completed_points:",
            '        invalid(path, f"results count {result_count!r} != {completed_points!r}")',
            "    observed_ranks = set()",
            "    for result in results:",
            "        if not isinstance(result, dict):",
            '            invalid(path, "result entry must be an object")',
            '        point = result.get("point")',
            '        point_type = point.get("point_type") if isinstance(point, dict) else None',
            "        if point_type not in allowed_point_types:",
            '            invalid(path, f"point type {point_type!r} is not valid for {expected_mode!r}")',
            '        fpms = result.get("fpms")',
            "        if not isinstance(fpms, list) or not fpms:",
            '            invalid(path, "each result must contain at least one FPM sample")',
            "        for fpm in fpms:",
            '            rank = fpm.get("dp_rank") if isinstance(fpm, dict) else None',
            "            if type(rank) is not int:",
            '                invalid(path, f"invalid FPM dp_rank {rank!r}")',
            "            observed_ranks.add(rank)",
            "    if observed_ranks != {expected_rank}:",
            '        invalid(path, f"FPM dp_ranks {sorted(observed_ranks)!r} != [{expected_rank}]")',
            "",
            "def validate_legacy_schema_v2(path, value, expected_rank):",
            '    if value.get("status") != "passed":',
            "        invalid(path,",
            "                f\"schema_version=2 status={value.get('status')!r} \"",
            "                f\"errors={value.get('errors')!r}\")",
            '    config = value.get("config")',
            '    actual_rank = config.get("dp_rank") if isinstance(config, dict) else None',
            "    if actual_rank != expected_rank:",
            '        invalid(path, f"FPM dp_rank {actual_rank!r} != {expected_rank}")',
            "",
            "for offset, raw_path in enumerate(sys.argv[3:]):",
            "    path = pathlib.Path(raw_path)",
            "    if not path.is_file() or path.stat().st_size == 0:",
            "        raise SystemExit(10)",
            "    try:",
            '        value = json.loads(path.read_text(encoding="utf-8"))',
            "    except (OSError, json.JSONDecodeError):",
            "        raise SystemExit(10)",
            "    if not isinstance(value, dict):",
            '        invalid(path, f"top-level JSON must be an object, got {type(value).__name__}")',
            "    expected_rank = start_rank + offset",
            '    schema_version = value.get("schema_version")',
            "    if schema_version == 1:",
            "        validate_schema_v1(path, value, expected_rank)",
            "    elif schema_version == 2:",
            "        validate_legacy_schema_v2(path, value, expected_rank)",
            "    else:",
            '        invalid(path, f"unsupported schema_version {schema_version!r}")',
            "PY",
            "}",
            "",
            'engine_pid=""',
            "engine_shutdown_grace_seconds=30",
            "terminate_engine() {",
            "  local pid=$1",
            '  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true',
            "  local shutdown_deadline=$((SECONDS + engine_shutdown_grace_seconds))",
            '  while kill -0 "$pid" 2>/dev/null || kill -0 -- "-$pid" 2>/dev/null; do',
            "    if (( SECONDS >= shutdown_deadline )); then",
            '      echo "Engine did not stop within ${engine_shutdown_grace_seconds}s; sending SIGKILL" >&2',
            '      kill -KILL -- "-$pid" 2>/dev/null || true',
            '      kill -KILL "$pid" 2>/dev/null || true',
            "      break",
            "    fi",
            "    sleep 1",
            "  done",
            '  wait "$pid" 2>/dev/null || true',
            "}",
            "cleanup() {",
            "  local status=$?",
            "  trap - EXIT INT TERM",
            '  if [[ -n "${engine_pid:-}" ]]; then',
            '    terminate_engine "$engine_pid"',
            "  fi",
            '  exit "$status"',
            "}",
            "trap cleanup EXIT",
            "trap 'exit 130' INT",
            "trap 'exit 143' TERM",
            "",
            "python3 -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' \"${engine_command[@]}\" &",
            "engine_pid=$!",
            "deadline=$((SECONDS + wait_timeout_seconds))",
            "",
            "while true; do",
            "  set +e",
            "  check_result_files",
            "  result_status=$?",
            "  set -e",
            "  if (( result_status == 0 )); then",
            "    break",
            "  fi",
            "  if (( result_status == 20 )); then",
            "    exit 1",
            "  fi",
            '  if ! kill -0 "$engine_pid" 2>/dev/null; then',
            "    set +e",
            '    wait "$engine_pid"',
            "    engine_status=$?",
            "    set -e",
            '    terminate_engine "$engine_pid"',
            '    engine_pid=""',
            '    echo "Engine exited before writing all FPM benchmark outputs" >&2',
            '    if (( engine_status == 0 )); then exit 1; else exit "$engine_status"; fi',
            "  fi",
            "  if (( SECONDS >= deadline )); then",
            '    echo "Timed out waiting for all FPM benchmark outputs" >&2',
            "    exit 124",
            "  fi",
            "  sleep 2",
            "done",
            "",
            'terminate_engine "$engine_pid"',
            'engine_pid=""',
            "trap - EXIT INT TERM",
            "",
        ]
    )
    return "\n".join(lines)


def _lower_worker_to_resource(
    context: dict[str, Any],
    worker: DGDService,
    main_container: MainContainer,
    topology: dict[str, int],
    *,
    efa_resource_name: str | None,
) -> dict[str, Any]:
    k8s = context.get("K8sConfig") or {}
    if not isinstance(k8s, dict):
        raise TypeError("K8sConfig must be a mapping")
    pod_spec = _lower_worker_pod_spec(
        worker,
        main_container,
        topology["gpus_per_node"],
        shared_memory_size=k8s.get("fpm_shared_memory_size"),
        compute_domain_name=(f"{context.get('name')}-compute-domain-channel" if topology["node_count"] > 1 else None),
        efa_resource_name=efa_resource_name,
    )
    metadata = _resource_metadata(context)
    if topology["node_count"] == 1:
        return {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": metadata,
            "spec": pod_spec,
        }

    if len(metadata["name"]) > 50:
        raise ValueError("FPM LeaderWorkerSet names must be at most 50 characters")
    pod_labels = copy.deepcopy(metadata["labels"])
    pod_template = {
        "metadata": {"labels": pod_labels},
        "spec": pod_spec,
    }
    return {
        "apiVersion": "leaderworkerset.x-k8s.io/v1",
        "kind": "LeaderWorkerSet",
        "metadata": metadata,
        "spec": {
            "replicas": 1,
            "startupPolicy": "LeaderCreated",
            "networkConfig": {"subdomainPolicy": "Shared"},
            "leaderWorkerTemplate": {
                "size": topology["node_count"],
                "restartPolicy": "None",
                "leaderTemplate": copy.deepcopy(pod_template),
                "workerTemplate": copy.deepcopy(pod_template),
            },
        },
    }


def _resource_metadata(context: dict[str, Any]) -> dict[str, Any]:
    k8s = context.get("K8sConfig") or {}
    if not isinstance(k8s, dict):
        raise TypeError("K8sConfig must be a mapping")
    name = context.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("FPM resource workload requires a non-empty context name")

    labels = {
        "app.kubernetes.io/name": name,
        "app.kubernetes.io/component": "fpm-resource",
    }
    extra_labels = k8s.get("fpm_resource_labels") or {}
    if not isinstance(extra_labels, dict):
        raise TypeError("K8sConfig.fpm_resource_labels must be a mapping")
    for label_name, label_value in extra_labels.items():
        if not isinstance(label_name, str) or not isinstance(label_value, str):
            raise TypeError("FPM resource label names and values must be strings")
        if label_name in labels and labels[label_name] != label_value:
            raise ValueError(f"K8sConfig.fpm_resource_labels cannot replace reserved label {label_name}")
        labels[label_name] = label_value

    metadata: dict[str, Any] = {
        "name": name,
        "labels": labels,
    }
    namespace = k8s.get("k8s_namespace")
    if namespace:
        metadata["namespace"] = namespace
    return metadata


def _drop_compute_domain_claims(pod_spec: dict[str, Any], expected_template_name: str | None) -> None:
    claims = pod_spec.pop("resourceClaims", None) or []
    if not isinstance(claims, list):
        raise TypeError("Pod resourceClaims must be a list")
    expected = (
        [
            {
                "name": "compute-domain-channel",
                "resourceClaimTemplateName": expected_template_name,
            }
        ]
        if expected_template_name is not None
        else []
    )
    if claims != expected:
        raise ValueError("FPM does not support user-provided Pod resourceClaims")


def _lower_worker_pod_spec(
    worker: DGDService,
    main_container: MainContainer,
    gpus_per_node: int,
    *,
    shared_memory_size: Any = None,
    compute_domain_name: str | None = None,
    efa_resource_name: str | None = None,
) -> dict[str, Any]:
    extra_pod_spec = worker.extra_pod_spec
    if extra_pod_spec is None:
        raise ValueError("The resolved vLLM worker has no extraPodSpec")

    pod_spec = extra_pod_spec.to_dict()
    pod_spec.pop("mainContainer", None)
    _drop_compute_domain_claims(pod_spec, compute_domain_name)

    container = main_container.to_dict()
    if container.get("envFrom"):
        raise ValueError("FPM V1 does not support mainContainer.envFrom")
    resource_override = container.get("resources")
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
    if shared_memory_size is not None:
        if not isinstance(shared_memory_size, str) or not shared_memory_size:
            raise ValueError("K8sConfig.fpm_shared_memory_size must be a non-empty string")
        empty_dir["sizeLimit"] = shared_memory_size
    elif shared_memory is not None:
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
            "resources": _merge_container_resources(
                _lower_resources(
                    worker.resources,
                    gpu_limit=gpus_per_node,
                    allow_compute_domain_claim=compute_domain_name is not None,
                    efa_resource_name=efa_resource_name,
                ),
                resource_override,
                expected_gpu_limit=gpus_per_node,
                efa_resource_name=efa_resource_name,
            ),
            "volumeMounts": volume_mounts,
            "command": ["/bin/bash", "-lc"],
            "args": ["exec sleep infinity"],
        }
    )
    pod_spec["volumes"] = volumes
    pod_spec["containers"] = [container]
    pod_spec["restartPolicy"] = "Always"
    return pod_spec


def _lower_resources(
    resources: dict[str, Any] | None,
    *,
    gpu_limit: int,
    allow_compute_domain_claim: bool,
    efa_resource_name: str | None,
) -> dict[str, Any]:
    if not isinstance(resources, dict):
        raise TypeError("The resolved vLLM worker has no resources")
    claims = resources.get("claims") or []
    if not isinstance(claims, list):
        raise TypeError("resources.claims must be a list")
    expected_claims = [{"name": "compute-domain-channel"}] if allow_compute_domain_claim else []
    if claims != expected_claims:
        raise ValueError("FPM does not support user-provided resource claims")

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
            section["nvidia.com/gpu"] = str(gpu_limit)
        if custom is not None:
            if not isinstance(custom, dict):
                raise TypeError(f"resources.{section_name}.custom must be a mapping")
            if efa_resource_name is not None and efa_resource_name in custom:
                custom[efa_resource_name] = str(gpu_limit)
            section.update(copy.deepcopy(custom))
        if section:
            lowered[section_name] = section

    limits = lowered.get("limits") or {}
    if "nvidia.com/gpu" not in limits:
        raise ValueError("FPM resource Pod requires a GPU limit")
    return lowered


def _merge_container_resources(
    base: dict[str, Any],
    override: Any,
    *,
    expected_gpu_limit: int,
    efa_resource_name: str | None,
) -> dict[str, Any]:
    if override is None:
        return base
    if not isinstance(override, dict):
        raise TypeError("worker_extra_pod_spec.mainContainer.resources must be a mapping")

    merged = copy.deepcopy(base)
    for section_name, section_override in override.items():
        if section_name not in ("limits", "requests"):
            raise ValueError(f"Unsupported container resource section: {section_name}")
        if not isinstance(section_override, dict):
            raise TypeError(f"mainContainer.resources.{section_name} must be a mapping")
        section = merged.setdefault(section_name, {})
        for name, value in section_override.items():
            if name == "nvidia.com/gpu":
                try:
                    requested_gpu = int(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError("nvidia.com/gpu must be an integer") from exc
                if requested_gpu != expected_gpu_limit:
                    raise ValueError("worker_extra_pod_spec cannot override the Generator-resolved per-node GPU count")
                value = str(expected_gpu_limit)
            elif efa_resource_name is not None and name == efa_resource_name:
                try:
                    requested_efa = int(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{efa_resource_name} must be an integer") from exc
                if requested_efa != expected_gpu_limit:
                    raise ValueError("worker_extra_pod_spec cannot override the Generator-resolved per-node EFA count")
                value = str(expected_gpu_limit)
            section[name] = copy.deepcopy(value)
    return merged


def _efa_resource_name(resolved_facts: Any) -> str | None:
    transport = getattr(resolved_facts, "transport", None)
    if not isinstance(transport, dict):
        return None
    pod = transport.get("pod")
    if pod is None:
        return None
    if not isinstance(pod, dict):
        raise TypeError("Resolved transport pod facts must be a mapping")
    resource_name = pod.get("efa_resource")
    if resource_name is None:
        return None
    if not isinstance(resource_name, str) or not resource_name:
        raise ValueError("Resolved transport efa_resource must be a non-empty string")
    return resource_name


def _add_volume_mount(
    volumes: list[Any],
    volume_mounts: list[Any],
    *,
    name: str,
    mount_path: str,
    volume_source: dict[str, Any],
) -> None:
    matching_volume = None
    for volume in volumes:
        if isinstance(volume, dict) and volume.get("name") == name:
            matching_volume = volume
            break

    matching_mount = None
    for mount in volume_mounts:
        if not isinstance(mount, dict):
            raise TypeError("Worker volumeMount entries must be mappings")
        if mount.get("name") == name or mount.get("mountPath") == mount_path:
            if mount.get("name") == name and mount.get("mountPath") == mount_path:
                matching_mount = mount
                break
            raise ValueError(f"worker_extra_pod_spec conflicts with reserved mount {mount_path}")

    if matching_volume is not None or matching_mount is not None:
        if matching_volume is None or matching_mount is None:
            raise ValueError(
                f"worker_extra_pod_spec must define both volume {name} and mount {mount_path} when overriding it"
            )
        return

    volumes.append({"name": name, **copy.deepcopy(volume_source)})
    volume_mounts.append({"name": name, "mountPath": mount_path})

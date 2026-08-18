# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the artifacts used for FPM collection.

The FPM resource Pod, LeaderWorkerSet, or PodCliqueSet deliberately does not
launch an engine. It reserves the same infrastructure as the normal vLLM
worker and stays alive while a collector stages the generated ``fpm_env.sh``
and ``run.sh`` into it.
"""

from __future__ import annotations

import copy
import json
import re
import shlex
from typing import Any

from aiconfigurator.fpm_contract import (
    FPM_BARRIER_TIMEOUT_ENV,
    FPM_ENGINE_BENCHMARK_OUTPUT_ENV,
    FPM_ENV_EXPORTED_VARS,
    FPM_ENV_FILENAME,
    FPM_MANIFEST_FILENAME,
    FPM_NATIVE_BENCHMARK_RESULT_SCHEMA_VERSION,
    FPM_RESOLVED_CONFIG_TEMPLATE,
    FPM_RESULTS_DIR,
    FPM_RUN_SCRIPT_FILENAME,
    fpm_validate_benchmark_output_path,
)

from .dgd_model import DGD, ComputeDomainDoc, DGDService, MainContainer, _dump_k8s_yaml
from .k8s_builder import build_dgd

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MISSING = object()
_NODE_RANK_SENTINEL = "__FPM_NODE_RANK__"
_FPM_BENCHMARK_MODES = ("agg", "prefill", "decode")
_FPM_ORCHESTRATORS = frozenset({"lws", "grove"})
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
    """Return the contract artifact trio for one FPM cell.

    The existing DGD builder remains the source of truth for infrastructure.
    This function lowers its one aggregated worker to a Pod (single node) or
    the selected multi-node orchestrator, publishes every collection-relevant
    render fact through ``fpm_env.sh`` (exactly the contract's
    ``FPM_ENV_EXPORTED_VARS``), and moves every concrete environment variable
    and the engine command into ``run.sh``, which only launches the engine.
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
    worker, compute_domain = _build_worker(context, backend, resolved_facts)
    main_container = _require_main_container(worker)

    command = list(main_container.command or [])
    args = list(main_container.args or [])
    if command[:3] != ["python3", "-m", "dynamo.vllm"]:
        raise ValueError("FPM V1 requires the normal vLLM worker command")
    if not all(isinstance(token, str) for token in command + args):
        raise ValueError("The resolved vLLM command must contain only string tokens")
    args.extend(extra_cli_args)

    env, barrier_timeout_override = _collect_concrete_env(worker, main_container)
    benchmark_mode = _require_benchmark_mode(args)
    topology = _resolve_topology(context, worker, args)
    if topology["data_parallel_size"] > 1 and _cli_option_value(args, "--data-parallel-backend") is None:
        args.extend(["--data-parallel-backend", "mp"])
    if topology["data_parallel_size"] > 1:
        _require_cli_option(args, "--data-parallel-backend", expected="mp")
    _ensure_dump_config_path(args, topology["node_count"])
    if topology["node_count"] > 1 and not _requires_mnnvl_compute_domain(resolved_facts):
        _force_disable_allreduce_fusion(args)
    benchmark_output_path = _ensure_benchmark_output_path(args, env)
    wait_timeout_seconds = _benchmark_wait_timeout_seconds(args)
    orchestrator = _fpm_orchestrator(context)
    preserve_compute_domain = topology["node_count"] > 1 and _requires_mnnvl_compute_domain(resolved_facts)
    env_script = _render_env_script(
        benchmark_mode,
        benchmark_output_path,
        wait_timeout_seconds,
        topology,
        barrier_timeout_override=barrier_timeout_override,
    )
    run_script = _render_run_script(command + args, env)
    workload = _lower_worker_to_resource(
        context,
        worker,
        main_container,
        topology,
        orchestrator=orchestrator,
        compute_domain=compute_domain,
        preserve_compute_domain=preserve_compute_domain,
        efa_resource_name=_efa_resource_name(resolved_facts),
    )
    resource_documents = _resource_documents(
        workload,
        compute_domain,
        topology,
        preserve_compute_domain=preserve_compute_domain,
    )

    return {
        FPM_MANIFEST_FILENAME: "---\n".join(_dump_k8s_yaml(document) for document in resource_documents),
        FPM_ENV_FILENAME: env_script,
        FPM_RUN_SCRIPT_FILENAME: run_script,
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
        # vLLM's FlexibleArgumentParser accepts underscore and dash spellings
        # of the same option, so normalize before matching the owned set.
        head = token.split("=", 1)[0]
        if head.startswith("--"):
            head = "--" + head[2:].replace("_", "-")
        for flag in _FPM_OWNED_ORCHESTRATION_FLAGS:
            if head == flag:
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
        f"{FPM_RESULTS_DIR}/{FPM_RESOLVED_CONFIG_TEMPLATE}"
        if node_count > 1
        else f"{FPM_RESULTS_DIR}/{FPM_RESOLVED_CONFIG_TEMPLATE.format(node_rank=0)}"
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


def _build_worker(
    context: dict[str, Any],
    backend: str,
    resolved_facts: Any,
) -> tuple[DGDService, ComputeDomainDoc | None]:
    docs = build_dgd(context, backend, resolved_facts=resolved_facts)
    dgd_docs = [doc for doc in docs if isinstance(doc, DGD)]
    if len(dgd_docs) != 1:
        raise ValueError("FPM V1 requires exactly one DynamoGraphDeployment document")
    compute_domains = [doc for doc in docs if isinstance(doc, ComputeDomainDoc)]
    if len(compute_domains) > 1:
        raise ValueError("FPM V1 supports at most one ComputeDomain document")

    workers = [(name, service) for name, service in dgd_docs[0].services.items() if service.component_type == "worker"]
    if len(workers) != 1 or workers[0][0] != "VllmWorker":
        raise ValueError("FPM V1 requires exactly one aggregated VllmWorker")

    worker = workers[0][1]
    if worker.replicas != 1:
        raise ValueError("FPM V1 requires worker replicas=1")
    return worker, compute_domains[0] if compute_domains else None


def _requires_mnnvl_compute_domain(resolved_facts: Any) -> bool:
    hardware = getattr(resolved_facts, "hardware", None)
    if not isinstance(hardware, dict):
        return False
    nccl_env = hardware.get("nccl_env") or {}
    if not isinstance(nccl_env, dict):
        return False
    return str(nccl_env.get("NCCL_MNNVL_ENABLE", "")).strip() == "1"


def _fpm_orchestrator(context: dict[str, Any]) -> str:
    k8s = context.get("K8sConfig") or {}
    if not isinstance(k8s, dict):
        raise TypeError("K8sConfig must be a mapping")
    orchestrator = k8s.get("fpm_orchestrator", "lws")
    # Older request/template versions materialize absent optional values as an
    # empty string. Treat that representation exactly like the historical
    # default so existing frozen requests continue to render LWS artifacts.
    if orchestrator in (None, ""):
        orchestrator = "lws"
    if not isinstance(orchestrator, str) or orchestrator not in _FPM_ORCHESTRATORS:
        allowed = ", ".join(sorted(_FPM_ORCHESTRATORS))
        raise ValueError(f"K8sConfig.fpm_orchestrator must be one of: {allowed}")
    return orchestrator


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


def _force_disable_allreduce_fusion(args: list[str]) -> None:
    # Guard: (HANG) multinode vLLM without an MNNVL fabric has no working
    # fused-allreduce path in the pinned engine: backend auto-selection picks
    # mnnvl unconditionally (vllm flashinfer_all_reduce.py:119) and hangs at
    # the first cross-node collective without NVSwitch multicast, while the
    # trtllm backend refuses multi-node outright and no environment variable
    # can turn the fusion off. Disable the fusion pass so allreduce falls
    # back to NCCL; single-node and MNNVL-fabric systems keep the fusion.
    flag = "--compilation-config"
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
    disabled = {"fuse_allreduce_rms": False}
    if not occurrences:
        args.extend([flag, json.dumps({"pass_config": disabled}, sort_keys=True, separators=(",", ":"))])
        return
    index, joined = occurrences[0]
    raw = args[index].split("=", 1)[1] if joined else args[index]
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{flag} must contain a JSON object") from exc
    if not isinstance(config, dict):
        raise TypeError(f"{flag} must contain a JSON object")
    pass_config = config.setdefault("pass_config", {})
    if not isinstance(pass_config, dict):
        raise TypeError(f"{flag} pass_config must be a JSON object")
    pass_config.update(disabled)
    value = json.dumps(config, sort_keys=True, separators=(",", ":"))
    args[index] = f"{flag}={value}" if joined else value


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


def _collect_concrete_env(
    worker: DGDService,
    main_container: MainContainer,
) -> tuple[list[tuple[str, str]], str | None]:
    # build_dgd sets the operator-only envFromSecret="hf-token-secret" on
    # every vLLM worker. FPM V1 intentionally accepts only concrete values:
    # it cannot safely materialize a Secret into run.sh, and rejecting that
    # built-in marker would make every FPM render fail.
    resolved: list[tuple[str, str]] = []
    barrier_timeout_override: str | None = None
    entries = list(worker.envs or []) + list(main_container.env or [])
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("FPM environment entries must be mappings")
        if "valueFrom" in entry:
            raise ValueError("FPM V1 does not support valueFrom environment entries")

        name = entry.get("name")
        if not isinstance(name, str) or not _ENV_NAME_RE.fullmatch(name):
            raise ValueError(f"Invalid shell environment variable name: {name!r}")
        if name in FPM_ENV_EXPORTED_VARS:
            # An export in run.sh would shadow the fpm_env.sh contract value
            # for the engine while the collection runtime keeps the original,
            # splitting the engine's flags from the gate's expectations.
            raise ValueError(f"Environment variable {name} is a reserved FPM contract variable")
        if "value" not in entry or entry["value"] is None:
            raise ValueError(f"Environment variable {name} must have a concrete value")

        value = entry["value"]
        if isinstance(value, bool):
            text = "true" if value else "false"
        elif isinstance(value, (str, int, float)):
            text = str(value)
        else:
            raise TypeError(f"Environment variable {name} must have a scalar value")
        if name == FPM_BARRIER_TIMEOUT_ENV:
            # The DP completion barrier lives in the collection runtime, which
            # sources fpm_env.sh; an export in run.sh would never reach it.
            barrier_timeout_override = text
            continue
        resolved.append((name, text))
    return resolved, barrier_timeout_override


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
    env_value = _last_env_value(env, FPM_ENGINE_BENCHMARK_OUTPUT_ENV)
    if cli_value is not None and env_value is not None and cli_value != env_value:
        raise ValueError(f"{flag} and {FPM_ENGINE_BENCHMARK_OUTPUT_ENV} must resolve to the same path")
    value = cli_value if cli_value is not None else env_value

    if value is None:
        value = f"{FPM_RESULTS_DIR}/benchmark.json"
    if not value:
        raise ValueError(f"{flag} must not be empty")
    # Fail closed on paths collector discovery could never find (Ethan's
    # cross-boundary review finding on the result-path contract).
    fpm_validate_benchmark_output_path(value)
    if cli_value is None:
        # Waiting for an output path that the engine does not know about would
        # hang forever.  Make the V1 default explicit in the resolved command.
        args.extend([flag, value])
    if env_value is None:
        env.append((FPM_ENGINE_BENCHMARK_OUTPUT_ENV, value))
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


def _render_env_script(
    benchmark_mode: str,
    benchmark_output_path: str,
    wait_timeout_seconds: int,
    topology: dict[str, int],
    *,
    barrier_timeout_override: str | None = None,
) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "# Rendered by the Generator FPM target. Sourced (not executed) by both the",
        "# collector's in-pod runtime and the generated run.sh. Exports exactly the",
        "# variables listed in fpm_contract.FPM_ENV_EXPORTED_VARS.",
        f"export FPM_NODE_COUNT={topology['node_count']}",
        f"export FPM_DATA_PARALLEL_SIZE={topology['data_parallel_size']}",
        f"export FPM_LOCAL_DATA_PARALLEL_SIZE={topology['local_data_parallel_size']}",
        f"export FPM_BENCHMARK_MODE={shlex.quote(benchmark_mode)}",
        f"export FPM_BENCHMARK_OUTPUT_PATH={shlex.quote(benchmark_output_path)}",
        f"export FPM_WAIT_TIMEOUT_SECONDS={wait_timeout_seconds}",
        f"export FPM_RESULT_SCHEMA_VERSION={FPM_NATIVE_BENCHMARK_RESULT_SCHEMA_VERSION}",
    ]
    if barrier_timeout_override is not None:
        # Operator tunable consumed by the collection runtime's DP barrier;
        # it travels here (not run.sh) because the runtime sources this file.
        lines.append(f"export {FPM_BARRIER_TIMEOUT_ENV}={shlex.quote(barrier_timeout_override)}")
    lines += [
        "",
        "if (( FPM_NODE_COUNT > 1 )); then",
        '  fpm_node_rank="${FPM_NODE_RANK:-${LWS_WORKER_INDEX:-${GROVE_PCLQ_POD_INDEX:-}}}"',
        '  fpm_master_addr="${FPM_MASTER_ADDR:-${LWS_LEADER_ADDRESS:-}}"',
        '  if [[ -z "$fpm_master_addr" && -n "${GROVE_PCLQ_NAME:-}" && -n "${GROVE_HEADLESS_SERVICE:-}" ]]; then',
        '    fpm_master_addr="${GROVE_PCLQ_NAME}-0.${GROVE_HEADLESS_SERVICE}"',
        "  fi",
        '  if [[ -z "$fpm_node_rank" || -z "$fpm_master_addr" ]]; then',
        '    echo "Multinode FPM requires rank and leader discovery from FPM_NODE_*, LWS, or Grove" >&2',
        "    exit 2",
        "  fi",
        "else",
        '  fpm_node_rank="${FPM_NODE_RANK:-${LWS_WORKER_INDEX:-${GROVE_PCLQ_POD_INDEX:-}}}"',
        '  fpm_master_addr="${FPM_MASTER_ADDR:-${LWS_LEADER_ADDRESS:-}}"',
        '  if [[ -z "$fpm_master_addr" && -n "${GROVE_PCLQ_NAME:-}" && -n "${GROVE_HEADLESS_SERVICE:-}" ]]; then',
        '    fpm_master_addr="${GROVE_PCLQ_NAME}-0.${GROVE_HEADLESS_SERVICE}"',
        "  fi",
        "  # Defaults apply only to a fully undiscovered environment; a partial",
        "  # answer is a misconfigured orchestrator and must fail closed.",
        '  if [[ -z "$fpm_node_rank" && -z "$fpm_master_addr" \\',
        '    && -z "${GROVE_PCLQ_NAME:-}" && -z "${GROVE_HEADLESS_SERVICE:-}" ]]; then',
        "    fpm_node_rank=0",
        "    fpm_master_addr=127.0.0.1",
        '  elif [[ -z "$fpm_node_rank" || -z "$fpm_master_addr" ]]; then',
        '    echo "FPM runtime requires complete rank and leader discovery from FPM_NODE_*, LWS, or Grove" >&2',
        "    exit 2",
        "  fi",
        "fi",
        'if ! [[ "$fpm_node_rank" =~ ^[0-9]+$ ]] || (( fpm_node_rank >= FPM_NODE_COUNT )); then',
        '  echo "Invalid FPM node rank: $fpm_node_rank (node_count=$FPM_NODE_COUNT)" >&2',
        "  exit 2",
        "fi",
        'export FPM_NODE_RANK="$fpm_node_rank"',
        'export FPM_MASTER_ADDR="$fpm_master_addr"',
        "",
    ]
    return "\n".join(lines)


def _render_run_script(
    command: list[str],
    env: list[tuple[str, str]],
) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -Eeuo pipefail",
        "",
        "# fpm_env.sh owns every render-time collection fact (topology, rank and",
        "# leader discovery, benchmark identity); discovery failures exit 2 here.",
        f'source "$(dirname "${{BASH_SOURCE[0]}}")/{FPM_ENV_FILENAME}"',
        "",
        "ulimit -l unlimited || true",
        "ulimit -n 1048576 || true",
    ]
    for name, value in env:
        lines.append(f"export {name}={shlex.quote(value)}")

    lines.extend(
        [
            "",
            "# FlashInfer downloads missing cubins at first use; its default cache",
            "# lives inside site-packages, which is read-only in the deployed image",
            "# and crashes every engine worker with EACCES. Default the cache to the",
            "# writable model-cache volume so pods reuse previously fetched cubins.",
            'if [[ -z "${FLASHINFER_CUBIN_DIR:-}" && -n "${HF_HOME:-}" ]]; then',
            '  export FLASHINFER_CUBIN_DIR="${HF_HOME}/flashinfer-cubins"',
            "fi",
        ]
    )

    lines.extend(
        [
            "",
            'export DYN_FPM_WORKER_ID="${DYN_FPM_WORKER_ID:-${FPM_RUN_ID:-fpm}-node${FPM_NODE_RANK}}"',
            f"engine_command=({' '.join(shlex.quote(token) for token in command)})",
            'for index in "${!engine_command[@]}"; do',
            f'  engine_command[$index]="${{engine_command[$index]//{_NODE_RANK_SENTINEL}/$FPM_NODE_RANK}}"',
            "done",
            "",
            "if (( FPM_NODE_COUNT > 1 )); then",
            "  if (( FPM_DATA_PARALLEL_SIZE > 1 )); then",
            '    engine_command+=(--data-parallel-size-local "$FPM_LOCAL_DATA_PARALLEL_SIZE")',
            '    engine_command+=(--data-parallel-start-rank "$((FPM_NODE_RANK * FPM_LOCAL_DATA_PARALLEL_SIZE))")',
            '    engine_command+=(--data-parallel-address "$FPM_MASTER_ADDR" --data-parallel-rpc-port 29510)',
            "    engine_command+=(--data-parallel-hybrid-lb)",
            "  else",
            '    engine_command+=(--nnodes "$FPM_NODE_COUNT" --node-rank "$FPM_NODE_RANK")',
            '    engine_command+=(--master-addr "$FPM_MASTER_ADDR" --master-port 29500)',
            "    if (( FPM_NODE_RANK > 0 )); then",
            "      # Headless followers never write results; classifying their exit",
            "      # against the leader's teardown belongs to the collector runtime.",
            "      engine_command+=(--headless)",
            "    fi",
            "  fi",
            "fi",
            "",
            "# Multinode transport profiles rewrite PATH to load the fabric-patched",
            "# libfabric first; a profile that drops /usr/local/cuda/bin silently",
            "# starves deep_gemm's runtime nvcc JIT, which only surfaces minutes",
            "# later as an opaque DG_HOST_ASSERT(!cubin.empty()) engine crash. Fail",
            "# fast with an actionable message instead. Single-node runs keep the",
            "# image default PATH and skip the check.",
            "if (( FPM_NODE_COUNT > 1 )) && ! command -v nvcc >/dev/null 2>&1; then",
            '  echo "run.sh: nvcc is not on PATH (PATH=$PATH); deep_gemm JIT will'
            ' fail. Ensure the transport PATH keeps /usr/local/cuda/bin." >&2',
            "  exit 2",
            "fi",
            "",
            "# Replace this shell so run.sh's exit code is the engine's exit code;",
            "# setsid makes the engine its process-group leader so the collector",
            "# runtime can terminate the whole group.",
            "exec python3 -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])'"
            ' "${engine_command[@]}"',
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
    orchestrator: str,
    compute_domain: ComputeDomainDoc | None,
    preserve_compute_domain: bool,
    efa_resource_name: str | None,
) -> dict[str, Any]:
    k8s = context.get("K8sConfig") or {}
    if not isinstance(k8s, dict):
        raise TypeError("K8sConfig must be a mapping")
    compute_domain_channel = compute_domain.channel_name if compute_domain is not None else None
    if compute_domain_channel is not None and (
        not isinstance(compute_domain_channel, str) or not compute_domain_channel
    ):
        raise ValueError("FPM ComputeDomain requires a non-empty resource claim template name")
    pod_spec = _lower_worker_pod_spec(
        worker,
        main_container,
        topology["gpus_per_node"],
        shared_memory_size=k8s.get("fpm_shared_memory_size"),
        compute_domain_name=compute_domain_channel,
        preserve_compute_domain=preserve_compute_domain,
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
        raise ValueError("FPM multi-node resource names must be at most 50 characters")
    pod_labels = copy.deepcopy(metadata["labels"])
    if orchestrator == "grove":
        return {
            "apiVersion": "grove.io/v1alpha1",
            "kind": "PodCliqueSet",
            "metadata": metadata,
            "spec": {
                "replicas": 1,
                "template": {
                    "cliqueStartupType": "CliqueStartupTypeAnyOrder",
                    "headlessServiceConfig": {"publishNotReadyAddresses": True},
                    "cliques": [
                        {
                            "name": "worker",
                            "labels": pod_labels,
                            "spec": {
                                "roleName": "worker",
                                "replicas": topology["node_count"],
                                "minAvailable": topology["node_count"],
                                "podSpec": pod_spec,
                            },
                        }
                    ],
                },
            },
        }

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


def _resource_documents(
    workload: dict[str, Any],
    compute_domain: ComputeDomainDoc | None,
    topology: dict[str, int],
    *,
    preserve_compute_domain: bool,
) -> list[dict[str, Any]]:
    if not preserve_compute_domain:
        return [workload]
    if topology["node_count"] <= 1:
        raise ValueError("FPM single-node workload must not require a ComputeDomain document")
    if compute_domain is None:
        raise ValueError("FPM multinode workload requires a ComputeDomain document")
    return [compute_domain.to_dict(), workload]


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


def _validate_compute_domain_claims(pod_spec: dict[str, Any], expected_template_name: str | None) -> None:
    claims = pod_spec.get("resourceClaims") or []
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
    preserve_compute_domain: bool = False,
    efa_resource_name: str | None = None,
) -> dict[str, Any]:
    extra_pod_spec = worker.extra_pod_spec
    if extra_pod_spec is None:
        raise ValueError("The resolved vLLM worker has no extraPodSpec")

    pod_spec = extra_pod_spec.to_dict()
    pod_spec.pop("mainContainer", None)
    _validate_compute_domain_claims(pod_spec, compute_domain_name)
    if not preserve_compute_domain:
        pod_spec.pop("resourceClaims", None)

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
        mount_path=FPM_RESULTS_DIR,
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
                    preserve_compute_domain_claim=preserve_compute_domain,
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
    preserve_compute_domain_claim: bool,
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
    if claims and preserve_compute_domain_claim:
        lowered["claims"] = copy.deepcopy(claims)
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

    # Guaranteed QoS by default: a GPU-only pod runs BestEffort, and FPM's
    # launch-bound band then disperses +-8-18% across boots (host launch speed
    # is decided by neighbour contention; R15 F-arm verdict: requests==limits
    # brings the in-band spread to 3.3%). CPU/memory scale per GPU and remain
    # overridable via worker_extra_pod_spec.mainContainer.resources. 14
    # cores/GPU (not 16) keeps a whole 8-GPU-node pod schedulable on real
    # fleets (127.9 allocatable cores on the H200 pool) with daemonset
    # headroom; 64Gi/GPU matches the validated F-arm pod.
    limits = lowered.setdefault("limits", limits)
    requests = lowered.setdefault("requests", {})
    if "cpu" not in limits and "cpu" not in requests:
        limits["cpu"] = str(gpu_limit * 14)
        requests["cpu"] = str(gpu_limit * 14)
    if "memory" not in limits and "memory" not in requests:
        limits["memory"] = f"{gpu_limit * 64}Gi"
        requests["memory"] = f"{gpu_limit * 64}Gi"
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

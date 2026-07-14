# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generator-driven Kubernetes execution for an immutable FPM plan."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from .planner import FPMCell, FPMCollectionPlan

CHECKPOINT_SCHEMA = "aic-fpm-collector-checkpoint-v1"
DEFAULT_BENCHMARK_TIMEOUT_SECONDS = 3600
_FPM_VLLM_RUNTIME_ARGS = (
    "--distributed-executor-backend",
    "mp",
    "--distributed-timeout-seconds",
    "1800",
    "--no-async-scheduling",
    "--enable-prefix-caching",
    "--no-scheduler-reserve-full-isl",
    "--no-enable-log-requests",
)
REMOTE_EXIT_MARKER = "__FPM_REMOTE_EXIT_CODE__="
REMOTE_FILES_MARKER = "__FPM_REMOTE_FILES__="
REMOTE_WORKDIR = "/tmp/fpm-bench"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _file_manifest(root: Path) -> dict[str, dict[str, int | str]]:
    manifest: dict[str, dict[str, int | str]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        manifest[str(path.relative_to(root))] = {
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return manifest


def _command_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(name, None)
    return env


def _kubectl_command() -> list[str]:
    override = os.environ.get("FPM_KUBECTL")
    if override:
        return override.split()
    if shutil.which("kubectl"):
        return ["kubectl"]
    if shutil.which("tsh"):
        return ["tsh", "kubectl"]
    raise RuntimeError("neither kubectl nor tsh is available")


def _run_command(
    args: list[str],
    *,
    check: bool = True,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=check,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_command_env(),
    )


class KubernetesCellRunner:
    """Own one generated Pod/LWS from apply through verified deletion."""

    def __init__(self, manifest: Path, cell_dir: Path) -> None:
        self.manifest = manifest
        self.cell_dir = cell_dir
        workload = yaml.safe_load(manifest.read_text())
        if not isinstance(workload, dict):
            raise TypeError("generated k8s_deploy.yaml must be one YAML mapping")
        metadata = workload.get("metadata") or {}
        self.name = str(metadata["name"])
        self.namespace = str(metadata.get("namespace") or "default")
        self.kind = str(workload["kind"])
        self.selector = f"app.kubernetes.io/name={self.name}"
        self.kubectl = _kubectl_command()

    def _kubectl(self, *args: str, check: bool = True, timeout: int | None = None):
        return _run_command([*self.kubectl, *args], check=check, timeout=timeout)

    def _exec_checked(self, pod: str, command: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
        """Run a pod command without trusting the local kubectl wrapper's exit code.

        ``tsh kubectl exec`` can return zero after printing a non-zero remote
        exit status.  Emit and parse an explicit marker from inside the pod so
        staging and benchmark failures remain fail-closed.
        """

        remote_command = shlex.join(command)
        script = f"{remote_command}; rc=$?; printf '\\n{REMOTE_EXIT_MARKER}%s\\n' \"$rc\"; exit 0"
        completed = self._kubectl(
            "exec",
            "-n",
            self.namespace,
            pod,
            "--",
            "bash",
            "-lc",
            script,
            check=False,
            timeout=timeout,
        )
        matches = re.findall(rf"{re.escape(REMOTE_EXIT_MARKER)}(\d+)", completed.stdout + completed.stderr)
        remote_exit = int(matches[-1]) if matches else None
        if completed.returncode != 0 or remote_exit != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(
                f"pod command failed for {pod}: local_exit={completed.returncode}, "
                f"remote_exit={remote_exit}, command={command!r}, output={detail!r}"
            )
        return completed

    def apply(self) -> None:
        applied = self._kubectl(
            "apply",
            "--validate=false",
            "-f",
            str(self.manifest),
            check=False,
            timeout=120,
        )
        observed = self._kubectl(
            "get",
            f"{self.kind}/{self.name}",
            "-n",
            self.namespace,
            "-o",
            "json",
            check=False,
            timeout=60,
        )
        try:
            payload = json.loads(observed.stdout)
        except json.JSONDecodeError as error:
            detail = (applied.stderr or applied.stdout or observed.stderr or observed.stdout).strip()
            raise RuntimeError(
                f"kubectl apply did not create {self.kind}/{self.name}: "
                f"apply_exit={applied.returncode}, get_exit={observed.returncode}, output={detail!r}"
            ) from error
        metadata = payload.get("metadata") if isinstance(payload, dict) else None
        if not isinstance(metadata, dict) or metadata.get("name") != self.name:
            raise RuntimeError(f"kubectl apply returned the wrong object for {self.kind}/{self.name}: {payload!r}")

    def pods(self) -> list[str]:
        result = self._kubectl(
            "get",
            "pods",
            "-n",
            self.namespace,
            "-l",
            self.selector,
            "-o",
            "json",
            timeout=60,
        )
        payload = json.loads(result.stdout)
        return sorted(item["metadata"]["name"] for item in payload.get("items", []))

    def wait_ready(self, expected_nodes: int, timeout_seconds: int = 900) -> list[str]:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            result = self._kubectl(
                "get",
                "pods",
                "-n",
                self.namespace,
                "-l",
                self.selector,
                "-o",
                "json",
                check=False,
                timeout=60,
            )
            if result.returncode == 0:
                payload = json.loads(result.stdout)
                items = payload.get("items", [])
                ready = [
                    item
                    for item in items
                    if item.get("status", {}).get("phase") == "Running"
                    and any(
                        condition.get("type") == "Ready" and condition.get("status") == "True"
                        for condition in item.get("status", {}).get("conditions", [])
                    )
                ]
                if len(ready) == expected_nodes:
                    return sorted(item["metadata"]["name"] for item in ready)
                failed = [item for item in items if item.get("status", {}).get("phase") in {"Failed", "Succeeded"}]
                if failed:
                    raise RuntimeError(f"FPM resource pods terminated before readiness: {failed}")
            time.sleep(5)
        raise TimeoutError(f"timed out waiting for {expected_nodes} FPM pods for {self.name}")

    def stage(self, pods: list[str], files: list[Path]) -> None:
        for pod in pods:
            self._exec_checked(pod, ["mkdir", "-p", REMOTE_WORKDIR, "/results"], timeout=60)
            for path in files:
                copied = self._kubectl(
                    "cp",
                    str(path),
                    f"{self.namespace}/{pod}:{REMOTE_WORKDIR}/{path.name}",
                    check=False,
                    timeout=180,
                )
                remote_path = f"{REMOTE_WORKDIR}/{path.name}"
                expected_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
                verifier = (
                    "import hashlib, pathlib, sys; "
                    "path = pathlib.Path(sys.argv[1]); "
                    "actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None; "
                    "raise SystemExit(0 if actual == sys.argv[2] else 1)"
                )
                try:
                    self._exec_checked(
                        pod,
                        ["python3", "-c", verifier, remote_path, expected_sha256],
                        timeout=60,
                    )
                except RuntimeError as error:
                    raise RuntimeError(
                        f"failed to stage an exact copy of {path.name} to {pod}: "
                        f"local_exit={copied.returncode}, "
                        f"output={(copied.stderr or copied.stdout).strip()!r}"
                    ) from error

    def _remote_result_manifest(self, pod: str) -> dict[str, dict[str, int | str]]:
        script = (
            "import hashlib, json, pathlib; "
            "root = pathlib.Path('/results'); "
            "files = {str(path.relative_to(root)): "
            "{'size': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} "
            "for path in sorted(root.rglob('*')) if path.is_file()}; "
            f"print('{REMOTE_FILES_MARKER}' + json.dumps(files, sort_keys=True))"
        )
        completed = self._exec_checked(pod, ["python3", "-c", script], timeout=120)
        matches = re.findall(
            rf"^{re.escape(REMOTE_FILES_MARKER)}(.+)$",
            completed.stdout,
            flags=re.MULTILINE,
        )
        if not matches:
            raise RuntimeError(f"pod {pod} did not return a /results file manifest")
        payload = json.loads(matches[-1])
        if not isinstance(payload, dict):
            raise TypeError(f"pod {pod} returned a non-object /results file manifest")
        return payload

    def _run_pod(self, pod: str, timeout_seconds: int) -> tuple[str, subprocess.CompletedProcess[str]]:
        completed = self._exec_checked(
            pod,
            ["bash", f"{REMOTE_WORKDIR}/run_with_etcd.sh"],
            timeout=timeout_seconds,
        )
        return pod, completed

    def execute(self, pods: list[str], timeout_seconds: int = 14400) -> None:
        logs_dir = self.cell_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        failures = []
        with ThreadPoolExecutor(max_workers=len(pods)) as pool:
            futures = {pool.submit(self._run_pod, pod, timeout_seconds): pod for pod in pods}
            for future in as_completed(futures):
                pod = futures[future]
                try:
                    _, completed = future.result()
                except Exception as error:
                    (logs_dir / f"{pod}.run.stderr.log").write_text(str(error) + "\n")
                    failures.append((pod, str(error)))
                    continue
                (logs_dir / f"{pod}.run.stdout.log").write_text(completed.stdout)
                (logs_dir / f"{pod}.run.stderr.log").write_text(completed.stderr)
        if failures:
            raise RuntimeError(f"generated FPM run.sh failed: {failures}")

    def collect(
        self,
        pods: list[str],
        *,
        destination: str = "raw",
        require_benchmark: bool = True,
    ) -> None:
        results_root = self.cell_dir / destination
        results_root.mkdir(parents=True, exist_ok=True)
        logs_dir = self.cell_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        for pod in pods:
            if require_benchmark:
                self._exec_checked(
                    pod,
                    [
                        "bash",
                        "-lc",
                        "find /results -maxdepth 1 -type f -name 'benchmark*.json' -size +0c | grep -q .",
                    ],
                    timeout=60,
                )
            remote_manifest = self._remote_result_manifest(pod)
            if require_benchmark and not any(
                Path(name).name.startswith("benchmark") and name.endswith(".json") for name in remote_manifest
            ):
                raise RuntimeError(f"pod {pod} result manifest contains no benchmark JSON files")
            pod_root = results_root / pod
            pod_root.mkdir(parents=True, exist_ok=True)
            copied = self._kubectl(
                "cp",
                f"{self.namespace}/{pod}:/results/.",
                str(pod_root),
                check=False,
                timeout=300,
            )
            local_manifest = _file_manifest(pod_root)
            if local_manifest != remote_manifest:
                missing = sorted(set(remote_manifest) - set(local_manifest))
                unexpected = sorted(set(local_manifest) - set(remote_manifest))
                mismatched = sorted(
                    name
                    for name in set(remote_manifest) & set(local_manifest)
                    if remote_manifest[name] != local_manifest[name]
                )
                raise RuntimeError(
                    f"failed to collect an exact /results copy from {pod}: "
                    f"local_exit={copied.returncode}, missing={missing!r}, "
                    f"unexpected={unexpected!r}, mismatched={mismatched!r}, "
                    f"output={(copied.stderr or copied.stdout).strip()!r}"
                )
            logs = self._kubectl(
                "logs",
                "-n",
                self.namespace,
                pod,
                check=False,
                timeout=120,
            )
            (logs_dir / f"{pod}.container.log").write_text(logs.stdout + logs.stderr)

    def cleanup(self) -> None:
        self._kubectl(
            "delete",
            "-f",
            str(self.manifest),
            "--ignore-not-found=true",
            "--wait=true",
            "--timeout=180s",
            check=False,
            timeout=240,
        )
        remaining = self.pods()
        if remaining:
            raise RuntimeError(f"owned FPM pods remain after cleanup: {remaining}")


def _case_payload(
    plan: FPMCollectionPlan,
    cell: FPMCell,
    *,
    selected_point_count: int | None = None,
) -> dict[str, object]:
    profile = cell.execution_profile
    selected_point_count = profile.selected_point_count if selected_point_count is None else selected_point_count
    if selected_point_count < 1 or selected_point_count > profile.selected_point_count:
        raise ValueError(
            f"selected_point_count={selected_point_count} is outside the frozen profile for {cell.cell_id}"
        )
    return {
        "schema_name": "aic_fpm_case_manifest",
        "schema_version": 1,
        "plan_sha256": plan.sha256,
        "cell_id": cell.cell_id,
        "workload_kind": cell.workload_kind,
        "global_warmup_iterations": plan.options.warmup_iterations,
        "warmup_repeats": 0,
        "measured_repeats": plan.options.measurement_repeats,
        "selected_point_count": selected_point_count,
        "ordered_shapes": [point.to_dict() for point in profile.ordered_points],
        "sampling_sha256": cell.sampling_sha256,
        "execution_profile": {
            "max_batch_size": profile.max_batch_size,
            "max_num_tokens": profile.max_num_tokens,
            "max_seq_len": profile.max_seq_len,
            "gpu_memory_utilization": profile.gpu_memory_utilization,
        },
    }


def _cell_generator_overrides(
    plan: FPMCollectionPlan,
    cell: FPMCell,
    base: dict[str, Any],
    *,
    selected_point_count: int | None = None,
) -> dict[str, Any]:
    profile = cell.execution_profile
    if selected_point_count is not None and not 1 <= selected_point_count <= profile.selected_point_count:
        raise ValueError(
            f"selected_point_count={selected_point_count} is outside the frozen profile for {cell.cell_id}"
        )
    unsupported_base = set(base) - {"K8sConfig", "generator_dynamo_version"}
    if unsupported_base:
        raise ValueError(f"FPM runner accepts deployment-only Generator inputs, got {sorted(unsupported_base)}")
    service = {"include_frontend": False}
    deployment = base.get("K8sConfig") or {}
    mount_path = deployment.get("k8s_pvc_mount_path")
    model_path_in_pvc = deployment.get("k8s_model_path_in_pvc")
    if mount_path is not None or model_path_in_pvc is not None:
        if not mount_path or not model_path_in_pvc:
            raise ValueError("K8sConfig.k8s_pvc_mount_path and k8s_model_path_in_pvc must be provided together")
        relative_model_path = PurePosixPath(str(model_path_in_pvc))
        if relative_model_path.is_absolute() or ".." in relative_model_path.parts:
            raise ValueError("K8sConfig.k8s_model_path_in_pvc must be a relative path without '..'")
        deployed_model_path = str(PurePosixPath(str(mount_path)) / relative_model_path)
        service.update(
            {
                "model_path": deployed_model_path,
                "served_model_path": deployed_model_path,
                "served_model_name": plan.model_path,
            }
        )
    scheduler_args = [
        "--scheduler-cls",
        "fpm_scheduler.InstrumentedScheduler",
        "--benchmark-mode",
        cell.workload_kind,
        "--benchmark-warmup-iterations",
        str(plan.options.warmup_iterations),
    ]
    model_args = []
    architecture = getattr(getattr(plan, "capability", None), "architecture", None)
    if architecture == "GlmMoeDsaForCausalLM":
        # This is the serving path validated by the pinned GLM-5.2 vLLM image.
        # The parser does not alter FPM scheduling, but keeping the model's
        # native runtime initialization avoids measuring a different engine.
        model_args.extend(["--trust-remote-code", "--reasoning-parser=glm45"])
    env = [
        {"name": "PYTHONPATH", "value": REMOTE_WORKDIR},
        {"name": "DYN_FPM_CASE_CONFIG", "value": f"{REMOTE_WORKDIR}/cases.json"},
        {"name": "DYN_FPM_BENCHMARK_OUTPUT_PATH", "value": "/results/benchmark.json"},
        {"name": "FPM_RUN_ID", "value": cell.cell_id},
        {"name": "FPM_RESULT_GRACE_SECONDS", "value": "0"},
    ]
    total_gpus = cell.topology.total_gpus
    generated = {
        # FPM cases are capacity-planned against these explicit engine bounds.
        # Letting Generator's SLA rules recompute them can silently make a
        # frozen point illegal or change CUDA Graph coverage.
        "preserve_engine_limits": True,
        "ServiceConfig": service,
        "DynConfig": {"mode": "agg"},
        "WorkerConfig": {"agg_workers": 1, "agg_gpus_per_worker": total_gpus},
        "K8sConfig": {
            "name_prefix": cell.cell_id,
            "extra_env": env,
            "fpm_resource_labels": {
                "aiconfigurator.nvidia.com/owned-by": "fpm-forward-collector",
                "aiconfigurator.nvidia.com/plan": plan.sha256[:16],
                "aiconfigurator.nvidia.com/cell": cell.cell_id,
            },
        },
        "params": {
            "agg": {
                "tensor_parallel_size": cell.topology.tp,
                "pipeline_parallel_size": cell.topology.pp,
                "data_parallel_size": cell.topology.dp,
                "moe_tensor_parallel_size": cell.topology.moe_tp,
                "moe_expert_parallel_size": cell.topology.moe_ep,
                "gpus_per_worker": total_gpus,
                "max_batch_size": profile.max_batch_size,
                "max_num_tokens": profile.max_num_tokens,
                "max_seq_len": profile.max_seq_len,
                "tokens_per_block": plan.options.kv_block_size,
                "kv_cache_dtype": cell.kv_cache_dtype,
                "gpu_memory_utilization": profile.gpu_memory_utilization,
                "compilation_config": profile.compilation_config,
                "extra_cli_args": [],
            }
        },
    }
    policy = cell.backend_policy.generator_overrides
    merged = _deep_merge(_deep_merge(base, generated), policy)

    policy_env = (policy.get("K8sConfig") or {}).get("extra_env") or []
    resolved_env: dict[str, dict[str, str]] = {}
    for item in [*policy_env, *env]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise TypeError("K8sConfig.extra_env entries must be {name, value} mappings")
        name = item["name"]
        existing = resolved_env.get(name)
        if existing is not None and existing != item:
            raise ValueError(f"conflicting FPM environment value for {name}")
        resolved_env[name] = copy.deepcopy(item)
    merged.setdefault("K8sConfig", {})["extra_env"] = list(resolved_env.values())

    policy_args = ((policy.get("params") or {}).get("agg") or {}).get("extra_cli_args") or []
    resolved_args = [*_FPM_VLLM_RUNTIME_ARGS, *model_args, *policy_args]
    has_benchmark_timeout = any(
        str(argument) == "--benchmark-timeout" or str(argument).startswith("--benchmark-timeout=")
        for argument in resolved_args
    )
    if not has_benchmark_timeout:
        # Dynamo's engine-side default is 300 seconds. A formal sparse design
        # can legitimately run longer even though the outer Kubernetes exec
        # timeout is four hours, so make the inner deadline campaign-safe.
        resolved_args.extend(["--benchmark-timeout", str(DEFAULT_BENCHMARK_TIMEOUT_SECONDS)])
    resolved_args.extend(["--gpu-memory-utilization", str(profile.gpu_memory_utilization)])
    compilation_value = json.dumps(profile.compilation_config, separators=(",", ":"), sort_keys=True)
    resolved_args.extend(["--compilation-config", compilation_value])
    resolved_args.extend(scheduler_args)
    merged_agg = merged.setdefault("params", {}).setdefault("agg", {})
    merged_agg.update(
        {
            "max_batch_size": profile.max_batch_size,
            "max_num_tokens": profile.max_num_tokens,
            "max_seq_len": profile.max_seq_len,
            "tokens_per_block": plan.options.kv_block_size,
            "gpu_memory_utilization": profile.gpu_memory_utilization,
            "compilation_config": profile.compilation_config,
            "extra_cli_args": resolved_args,
        }
    )
    return merged


def _render_cell(
    plan: FPMCollectionPlan,
    cell: FPMCell,
    cell_dir: Path,
    generator_overrides: dict[str, Any],
    *,
    selected_point_count: int | None = None,
) -> dict[str, Any]:
    from aiconfigurator.generator.api import generate_from_request
    from aiconfigurator.generator.naive import build_naive_generator_params
    from aiconfigurator.generator.request import from_legacy_params

    overrides = _cell_generator_overrides(
        plan,
        cell,
        generator_overrides,
        selected_point_count=selected_point_count,
    )
    params = build_naive_generator_params(
        model_name=plan.model_path,
        total_gpus=cell.topology.total_gpus,
        system_name=plan.system,
        backend_name=plan.backend,
        mode="agg",
        generator_dynamo_version=overrides.get("generator_dynamo_version"),
        generator_overrides=overrides,
    )
    # ``build_naive_generator_params`` merges section and role overrides but
    # intentionally does not propagate arbitrary top-level keys.  Carry this
    # rule control explicitly across the typed request boundary so rendering
    # cannot recompute limits after the FPM plan has been frozen.
    params["preserve_engine_limits"] = True
    request = from_legacy_params(params, plan.backend)
    request = replace(
        request,
        emit=replace(request.emit, deployment_target="fpm", output_dir=str(cell_dir)),
        backend=replace(
            request.backend,
            generated_config_version=overrides.get("generated_config_version"),
        ),
    )
    errors = request.validate()
    if errors:
        raise ValueError(f"invalid GeneratorRequest for {cell.cell_id}: {errors}")
    artifacts = generate_from_request(request, output_dir=str(cell_dir))
    _allow_graph_aware_benchmark_results(cell_dir / "run.sh")
    _atomic_json(cell_dir / "generator-request.json", params)
    return artifacts


def _allow_graph_aware_benchmark_results(run_script: Path) -> None:
    """Teach the generated strict result gate about Dynamo's schema-v2 rank artifact.

    The Generator still owns process launch, timeout, validation, and teardown.
    This compatibility shim changes only the accepted upstream schema version;
    its existing complete/valid/coverage/rank checks remain unchanged.
    """

    script = run_script.read_text()
    legacy = 'value.get("schema_version") != 1'
    compatible = 'value.get("schema_version") not in {1, 2}'
    if compatible in script:
        return
    occurrences = script.count(legacy)
    if occurrences != 1:
        raise RuntimeError(
            "generated FPM run.sh has an unsupported benchmark result gate: "
            f"expected one legacy schema check, found {occurrences}"
        )
    run_script.write_text(script.replace(legacy, compatible))


def _expected_nodes(manifest: Path) -> int:
    payload = yaml.safe_load(manifest.read_text())
    if payload["kind"] == "Pod":
        return 1
    if payload["kind"] == "LeaderWorkerSet":
        return int(payload["spec"]["leaderWorkerTemplate"]["size"])
    raise ValueError(f"unsupported generated FPM workload kind: {payload.get('kind')}")


def _validate_runtime_collection(
    cell: FPMCell,
    raw_root: Path,
    *,
    selected_point_count: int | None = None,
) -> None:
    """Validate integrated runtime capacity, canary, and formal results across DP ranks."""

    rank_payloads = []
    for path in sorted(raw_root.glob("**/benchmark*.json")):
        payload = json.loads(path.read_text())
        if payload.get("artifact_type") == "merged" or path.stem.endswith("_merged"):
            continue
        rank_payloads.append((path, payload))
    if not rank_payloads:
        raise ValueError(f"integrated runtime emitted no benchmark results for {cell.cell_id}")
    seen_ranks = set()
    seen_schema_versions = set()
    graph_aware_identity = None
    ordered_population = [point.to_dict() for point in cell.execution_profile.ordered_points]
    formal_points = None
    target = cell.execution_profile.selected_point_count if selected_point_count is None else selected_point_count

    def point_key(point: dict[str, object]) -> tuple[str, int, int, int]:
        return (
            str(point["workload_kind"]),
            int(point["batch_size"]),
            int(point["suffix_length"]),
            int(point["prefix_length"]),
        )

    for path, payload in rank_payloads:
        schema_version = payload.get("schema_version")
        if schema_version not in {1, 2}:
            raise ValueError(f"unsupported integrated result schema for {cell.cell_id}: {path}")
        seen_schema_versions.add(schema_version)
        if schema_version == 2 and payload.get("artifact_type") != "rank":
            raise ValueError(f"schema-v2 integrated result is not a rank artifact for {cell.cell_id}: {path}")
        if payload.get("status") != "complete" or payload.get("valid") is not True:
            raise ValueError(f"invalid integrated result for {cell.cell_id}: {path}")
        collector = payload.get("collector")
        if not isinstance(collector, dict) or collector.get("cell_id") != cell.cell_id:
            raise ValueError(f"integrated result identity mismatch for {cell.cell_id}: {path}")
        if schema_version == 2 and collector.get("runtime_contract") != "dynamo_pr11509_schema_v2":
            raise ValueError(
                f"schema-v2 integrated result has no PR11509 collector contract for {cell.cell_id}: {path}"
            )
        if int(collector.get("selected_point_count", -1)) != target:
            raise ValueError(f"integrated selected-count mismatch for {cell.cell_id}: {path}")
        eligible = int(collector.get("capacity_eligible_count", -1))
        if eligible < target:
            raise ValueError(
                f"runtime capacity admits only {eligible}/{target} frozen points for {cell.cell_id}; "
                "formal collection was not started"
            )
        cancelled = payload.get("cancelled_points")
        if not isinstance(cancelled, list):
            raise TypeError(f"integrated result has no capacity decisions for {cell.cell_id}: {path}")
        if int(collector.get("capacity_cancelled_count", -1)) != len(cancelled):
            raise ValueError(f"integrated cancelled-count mismatch for {cell.cell_id}: {path}")
        cancelled_keys = {point_key(item["point"]) for item in cancelled}
        rank_eligible = [point for point in ordered_population if point_key(point) not in cancelled_keys]
        if len(rank_eligible) != eligible:
            raise ValueError(f"integrated eligible-count mismatch for {cell.cell_id}: {path}")
        rank_formal_points = rank_eligible[:target]
        if formal_points is None:
            formal_points = rank_formal_points
        elif rank_formal_points != formal_points:
            raise ValueError(f"integrated DP ranks selected different formal point sets for {cell.cell_id}: {path}")
        rows = payload.get("campaign_results")
        if not isinstance(rows, list) or len(rows) != target:
            raise ValueError(
                f"integrated collection did not execute {target} runtime-admitted points for {cell.cell_id}: {path}"
            )
        if [row.get("point") for row in rows] != rank_formal_points:
            raise ValueError(f"integrated formal point order mismatch for {cell.cell_id}: {path}")
        row_ranks = set()
        for row in rows:
            warmups = row.get("warmup_fpms")
            measurements = row.get("fpms")
            if not isinstance(warmups, list) or warmups:
                raise ValueError(f"integrated collection emitted warmup FPMs for {cell.cell_id}: {path}")
            if not isinstance(measurements, list) or len(measurements) != 1:
                raise ValueError(f"integrated collection did not emit one measured FPM for {cell.cell_id}: {path}")
            row_ranks.add(int(measurements[0]["dp_rank"]))
        if len(row_ranks) != 1:
            raise ValueError(f"integrated result mixes DP ranks for {cell.cell_id}: {path}")
        rank = row_ranks.pop()
        if schema_version == 2:
            dp = payload.get("dp")
            if not isinstance(dp, dict) or dp.get("rank") != rank or dp.get("size") != cell.topology.dp:
                raise ValueError(f"schema-v2 DP metadata mismatch for {cell.cell_id}: {path}")
            identity = (payload.get("run_id"), payload.get("grid_digest"))
            if not all(isinstance(value, str) and value for value in identity):
                raise ValueError(f"schema-v2 run identity is missing for {cell.cell_id}: {path}")
            if graph_aware_identity is None:
                graph_aware_identity = identity
            elif identity != graph_aware_identity:
                raise ValueError(f"schema-v2 DP ranks have different run identities for {cell.cell_id}: {path}")
        if rank in seen_ranks:
            raise ValueError(f"duplicate integrated dp_rank={rank} for {cell.cell_id}: {path}")
        seen_ranks.add(rank)
    if len(seen_schema_versions) != 1:
        raise ValueError(f"integrated DP ranks emitted mixed result schemas for {cell.cell_id}: {seen_schema_versions}")
    expected_ranks = set(range(cell.topology.dp))
    if seen_ranks != expected_ranks:
        raise ValueError(
            f"integrated DP rank set mismatch for {cell.cell_id}: "
            f"actual={sorted(seen_ranks)} expected={sorted(expected_ranks)}"
        )


def _load_checkpoint(path: Path, plan: FPMCollectionPlan, resume: bool) -> dict[str, Any]:
    if resume and path.exists():
        payload = json.loads(path.read_text())
        if payload.get("schema") != CHECKPOINT_SCHEMA or payload.get("plan_sha256") != plan.sha256:
            raise ValueError("FPM checkpoint does not match the current frozen plan")
        return payload
    return {"schema": CHECKPOINT_SCHEMA, "plan_sha256": plan.sha256, "cells": {}}


def _runtime_timing_summary(raw_root: Path) -> dict[str, int | float]:
    """Summarize validated rank timing without treating merged artifacts as ranks."""

    rank_timings = []
    for path in sorted(raw_root.glob("**/benchmark*.json")):
        payload = json.loads(path.read_text())
        if payload.get("artifact_type") == "merged" or path.stem.endswith("_merged"):
            continue
        timing = payload.get("timing")
        if not isinstance(timing, dict):
            continue
        benchmark_elapsed = timing.get("benchmark_elapsed_seconds")
        measured_iterations = timing.get("measured_iteration_seconds")
        if not isinstance(benchmark_elapsed, (int, float)) or benchmark_elapsed < 0:
            continue
        if not isinstance(measured_iterations, (int, float)) or measured_iterations < 0:
            continue
        rank_timings.append((float(benchmark_elapsed), float(measured_iterations)))
    if not rank_timings:
        return {}
    return {
        "runtime_rank_count": len(rank_timings),
        "benchmark_elapsed_seconds": max(value[0] for value in rank_timings),
        "measured_iteration_seconds": max(value[1] for value in rank_timings),
    }


def run_collection(
    plan: FPMCollectionPlan,
    *,
    generator_overrides: dict[str, Any],
    checkpoint_dir: str,
    artifact_root: str,
    resume: bool,
    retry_failed: bool,
    smoke: bool = False,
    cell_limit: int | None = None,
    runtime_overlay_dir: str | None = None,
    database_root: str | None = None,
) -> list[dict[str, object]]:
    """Render and run every cell, always tearing down owned resources."""

    root = Path(artifact_root).expanduser().resolve() / plan.sha256[:16]
    if smoke:
        root /= "smoke"
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "collection-plan.json", plan.to_dict())
    checkpoint_name = "fpm_forward_smoke.json" if smoke else "fpm_forward.json"
    checkpoint_path = Path(checkpoint_dir).expanduser().resolve() / checkpoint_name
    checkpoint = _load_checkpoint(checkpoint_path, plan, resume)
    errors: list[dict[str, object]] = []
    runtime_scheduler = Path(__file__).resolve().parent / "runtime" / "vllm_scheduler.py"
    runtime_wrapper = Path(__file__).resolve().parent / "runtime" / "run_with_etcd.sh"
    runtime_preflight = Path(__file__).resolve().parent / "runtime" / "preflight.py"
    runtime_installer = Path(__file__).resolve().parent / "runtime" / "install_overlay.py"
    overlay_source = Path(runtime_overlay_dir).expanduser().resolve() if runtime_overlay_dir else None
    if bool(plan.runtime_overlay_files) != bool(overlay_source):
        raise ValueError("runtime overlay source does not match the frozen plan")
    target_cells = plan.cells[: (cell_limit or (1 if smoke else len(plan.cells)))]

    checkpoint_changed = False
    for cell in target_cells:
        entry = checkpoint["cells"].get(cell.cell_id)
        if not isinstance(entry, dict) or entry.get("status") != "passed":
            continue
        metadata = {
            "total_gpus": cell.topology.total_gpus,
            "planned_point_count": cell.execution_profile.selected_point_count,
            "global_warmup_iterations": plan.options.warmup_iterations,
            **_runtime_timing_summary(root / "cells" / cell.cell_id / "raw"),
        }
        for key, value in metadata.items():
            if entry.get(key) != value:
                entry[key] = value
                checkpoint_changed = True
    if checkpoint_changed:
        _atomic_json(checkpoint_path, checkpoint)

    for cell in target_cells:
        previous = checkpoint["cells"].get(cell.cell_id, {})
        if resume and previous.get("status") == "passed":
            continue
        if resume and previous.get("status") == "failed" and not retry_failed:
            continue

        cell_dir = root / "cells" / cell.cell_id
        if cell_dir.exists() and not resume:
            shutil.rmtree(cell_dir)
        cell_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(cell_dir / "cell.json", cell.to_dict())
        overlay_files: list[Path] = []
        if overlay_source is not None:
            overlay_manifest = {"schema_version": 1, "files": {}}
            expected_base = dict(plan.runtime_overlay_base_files)
            for name, expected_sha256 in plan.runtime_overlay_files:
                source = overlay_source / name
                actual_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
                if actual_sha256 != expected_sha256:
                    raise ValueError(f"runtime overlay changed after planning: {source}")
                destination = cell_dir / f"runtime-overlay-{name}"
                shutil.copy2(source, destination)
                overlay_files.append(destination)
                overlay_manifest["files"][name] = {
                    "source_name": destination.name,
                    "original_sha256": expected_base[name],
                    "sha256": expected_sha256,
                }
            _atomic_json(cell_dir / "runtime-overlay-manifest.json", overlay_manifest)
            overlay_files.append(cell_dir / "runtime-overlay-manifest.json")
        selected_count = plan.options.smoke_points if smoke else None
        planned_point_count = selected_count or cell.execution_profile.selected_point_count
        _atomic_json(
            cell_dir / "cases.json",
            _case_payload(plan, cell, selected_point_count=selected_count),
        )
        cell_started = time.monotonic()
        started_at = _utc_now()
        base_record = {
            "status": "running",
            "started_at": started_at,
            "total_gpus": cell.topology.total_gpus,
            "planned_point_count": planned_point_count,
            "global_warmup_iterations": plan.options.warmup_iterations,
        }
        checkpoint["cells"][cell.cell_id] = base_record
        _atomic_json(checkpoint_path, checkpoint)
        resource = None
        try:
            _render_cell(
                plan,
                cell,
                cell_dir,
                generator_overrides,
                selected_point_count=selected_count,
            )
            shutil.copy2(runtime_scheduler, cell_dir / "fpm_scheduler.py")
            manifest = cell_dir / "k8s_deploy.yaml"
            run_script = cell_dir / "run.sh"
            if not manifest.exists() or not run_script.exists():
                raise RuntimeError("Generator FPM target did not emit k8s_deploy.yaml and run.sh")
            resource = KubernetesCellRunner(manifest, cell_dir)
            resource.apply()
            pods = resource.wait_ready(_expected_nodes(manifest))
            resource.stage(
                pods,
                [
                    cell_dir / "cases.json",
                    cell_dir / "fpm_scheduler.py",
                    run_script,
                    runtime_wrapper,
                    runtime_preflight,
                    runtime_installer,
                    *overlay_files,
                ],
            )
            resource.execute(pods)
            resource.collect(pods)
            _validate_runtime_collection(
                cell,
                cell_dir / "raw",
                selected_point_count=selected_count,
            )
            checkpoint["cells"][cell.cell_id] = {
                **base_record,
                "status": "passed",
                "artifact_dir": str(cell_dir),
                "pods": pods,
                **_runtime_timing_summary(cell_dir / "raw"),
            }
        except KeyboardInterrupt:
            if resource is not None:
                try:
                    resource.collect(resource.pods(), require_benchmark=False)
                except Exception:
                    pass
            checkpoint["cells"][cell.cell_id] = {
                **base_record,
                "status": "interrupted",
                "artifact_dir": str(cell_dir),
            }
            raise
        except Exception as error:
            if resource is not None:
                try:
                    resource.collect(resource.pods(), require_benchmark=False)
                except Exception:
                    pass
            checkpoint["cells"][cell.cell_id] = {
                **base_record,
                "status": "failed",
                "artifact_dir": str(cell_dir),
                "error_type": type(error).__name__,
                "error": str(error),
            }
            errors.append(
                {
                    "module": "fpm_forward",
                    "cell_id": cell.cell_id,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "classification": "campaign_cell_failed",
                }
            )
        finally:
            if resource is not None:
                try:
                    resource.cleanup()
                except Exception as cleanup_error:
                    checkpoint["cells"][cell.cell_id]["cleanup_error"] = str(cleanup_error)
                    errors.append(
                        {
                            "module": "fpm_forward",
                            "cell_id": cell.cell_id,
                            "error_type": type(cleanup_error).__name__,
                            "error_message": str(cleanup_error),
                            "classification": "resource_cleanup_failed",
                        }
                    )
            checkpoint["cells"][cell.cell_id]["completed_at"] = _utc_now()
            checkpoint["cells"][cell.cell_id]["duration_seconds"] = round(time.monotonic() - cell_started, 3)
            _atomic_json(checkpoint_path, checkpoint)
    all_passed = all(checkpoint["cells"].get(cell.cell_id, {}).get("status") == "passed" for cell in target_cells)
    if smoke and not errors and all_passed:
        checkpoint["smoke"] = {
            "status": "passed",
            "cell_count": len(target_cells),
            "points_per_cell": plan.options.smoke_points,
            "formal_database_written": False,
        }
        _atomic_json(checkpoint_path, checkpoint)
    elif not errors and all_passed:
        from .database import aggregate_cell, write_formal_database

        try:
            formal_rows = []
            for cell in plan.cells:
                formal_rows.extend(aggregate_cell(plan, cell, root / "cells" / cell.cell_id))
            systems_root = Path(database_root).expanduser().resolve() if database_root else None
            parquet_path, metadata_path = write_formal_database(plan, formal_rows, systems_root=systems_root)
            checkpoint["database"] = {
                "status": "passed",
                "parquet": str(parquet_path),
                "metadata": str(metadata_path),
                "row_count": len(formal_rows),
            }
        except Exception as error:
            checkpoint["database"] = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            errors.append(
                {
                    "module": "fpm_forward.database",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "classification": "formal_database_failed",
                }
            )
        _atomic_json(checkpoint_path, checkpoint)
    elif not errors and not all_passed:
        errors.append(
            {
                "module": "fpm_forward",
                "error_type": "IncompleteCampaign",
                "error_message": "not every frozen FPM cell is passed; formal database was not written",
                "classification": "campaign_incomplete",
            }
        )
    return errors

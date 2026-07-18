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
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from .native_artifact import COLLECTOR_PROVENANCE_FILENAME, validate_native_collection
from .planner import FPMCell, FPMCollectionPlan

CHECKPOINT_SCHEMA = "aic-fpm-collector-checkpoint-v3"
DEFAULT_BENCHMARK_TIMEOUT_SECONDS = 3600
RESULT_COPY_ATTEMPTS = 3
RESULT_COPY_TIMEOUT_SECONDS = 300
_FPM_VLLM_RUNTIME_ARGS = (
    "--distributed-executor-backend",
    "mp",
    "--distributed-timeout-seconds",
    "1800",
    "--no-async-scheduling",
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


def _file_metadata(path: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return {
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _file_manifest(root: Path) -> dict[str, dict[str, int | str]]:
    manifest: dict[str, dict[str, int | str]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        manifest[str(path.relative_to(root))] = _file_metadata(path)
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
        self.expected_labels = dict(metadata.get("labels") or {})
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
            "-o",
            "json",
            check=False,
            timeout=120,
        )
        if applied.returncode != 0:
            detail = (applied.stderr or applied.stdout).strip()
            raise RuntimeError(
                f"kubectl apply failed for {self.kind}/{self.name}: apply_exit={applied.returncode}, output={detail!r}"
            )
        try:
            payload = json.loads(applied.stdout)
        except json.JSONDecodeError as error:
            detail = (applied.stderr or applied.stdout).strip()
            raise RuntimeError(
                f"kubectl apply returned no verifiable object for {self.kind}/{self.name}: output={detail!r}"
            ) from error
        metadata = payload.get("metadata") if isinstance(payload, dict) else None
        labels = metadata.get("labels") if isinstance(metadata, dict) else None
        if (
            not isinstance(payload, dict)
            or payload.get("kind") != self.kind
            or not isinstance(metadata, dict)
            or metadata.get("name") != self.name
            or not isinstance(labels, dict)
            or any(labels.get(key) != value for key, value in self.expected_labels.items())
        ):
            raise RuntimeError(f"kubectl apply returned the wrong object for {self.kind}/{self.name}: {payload!r}")

    def pods(self, *, include_terminating: bool = True) -> list[str]:
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
        items = payload.get("items", [])
        if not include_terminating:
            items = [
                item
                for item in items
                if not isinstance(item.get("metadata"), dict) or not item["metadata"].get("deletionTimestamp")
            ]
        return sorted(item["metadata"]["name"] for item in items)

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

    def prepare_attempt(
        self,
        pods: list[str],
        *,
        cell_id: str,
        plan_sha256: str,
        attempt_id: str,
    ) -> None:
        """Clear stale results and bind every Pod to this Collector attempt."""

        payload = json.dumps(
            {
                "schema_name": "aic_fpm_collector_provenance",
                "schema_version": 1,
                "cell_id": cell_id,
                "plan_sha256": plan_sha256,
                "attempt_id": attempt_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        script = (
            "import importlib.metadata, json, pathlib, sys; "
            "payload = json.loads(sys.argv[1]); "
            "payload['runtime'] = {'backend': 'vllm', "
            "'backend_version': importlib.metadata.version('vllm')}; "
            "path = pathlib.Path('/results') / sys.argv[2]; "
            "path.write_text(json.dumps(payload, sort_keys=True) + '\\n')"
        )
        for pod in pods:
            self._exec_checked(
                pod,
                [
                    "find",
                    "/results",
                    "-mindepth",
                    "1",
                    "-maxdepth",
                    "1",
                    "-exec",
                    "rm",
                    "-rf",
                    "--",
                    "{}",
                    "+",
                ],
                timeout=60,
            )
            self._exec_checked(
                pod,
                ["python3", "-c", script, payload, COLLECTOR_PROVENANCE_FILENAME],
                timeout=60,
            )

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

    def _copy_result_file(
        self,
        pod: str,
        remote_name: str,
        expected: dict[str, int | str],
        pod_root: Path,
    ) -> None:
        relative = PurePosixPath(remote_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"pod {pod} returned an unsafe /results path: {remote_name!r}")
        target = pod_root.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and _file_metadata(target) == expected:
            return

        partial = target.with_name(f".{target.name}.part")
        failures = []
        for attempt in range(1, RESULT_COPY_ATTEMPTS + 1):
            partial.unlink(missing_ok=True)
            try:
                copied = self._kubectl(
                    "cp",
                    f"{self.namespace}/{pod}:/results/{relative.as_posix()}",
                    str(partial),
                    check=False,
                    timeout=RESULT_COPY_TIMEOUT_SECONDS,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                failures.append(
                    {
                        "attempt": attempt,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                continue
            actual = _file_metadata(partial) if partial.is_file() else None
            if actual == expected:
                os.replace(partial, target)
                return
            failures.append(
                {
                    "attempt": attempt,
                    "local_exit": copied.returncode,
                    "actual": actual,
                    "output": (copied.stderr or copied.stdout).strip(),
                }
            )
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"failed to collect exact result file {remote_name!r} from {pod} "
            f"after {RESULT_COPY_ATTEMPTS} attempts: expected={expected!r}, "
            f"failures={failures!r}"
        )

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
        benchmark_observed = False
        for pod in pods:
            remote_manifest = self._remote_result_manifest(pod)
            pod_has_benchmark = any(
                Path(name).name.startswith("benchmark")
                and name.endswith(".json")
                and isinstance(metadata, dict)
                and int(metadata.get("size", 0)) > 0
                for name, metadata in remote_manifest.items()
            )
            benchmark_observed = benchmark_observed or pod_has_benchmark
            pod_root = results_root / pod
            pod_root.mkdir(parents=True, exist_ok=True)
            for remote_name, expected in sorted(remote_manifest.items()):
                if not isinstance(expected, dict):
                    raise TypeError(f"pod {pod} returned invalid metadata for {remote_name!r}: {expected!r}")
                self._copy_result_file(pod, remote_name, expected, pod_root)
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
                    f"missing={missing!r}, unexpected={unexpected!r}, "
                    f"mismatched={mismatched!r}"
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
        if require_benchmark and not benchmark_observed:
            raise RuntimeError("FPM cell result manifests contain no benchmark JSON files")

    def cleanup(self) -> None:
        deleted = self._kubectl(
            "delete",
            "-f",
            str(self.manifest),
            "--ignore-not-found=true",
            "--cascade=foreground",
            "--wait=true",
            "--timeout=180s",
            check=False,
            timeout=240,
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
        detail = (observed.stderr or observed.stdout).strip()
        if observed.returncode == 0:
            try:
                payload = json.loads(observed.stdout)
            except json.JSONDecodeError as error:
                if "notfound" not in detail.replace(" ", "").lower():
                    raise RuntimeError(
                        f"could not verify deletion of {self.kind}/{self.name}: "
                        f"delete_exit={deleted.returncode}, output={detail!r}"
                    ) from error
            else:
                metadata = payload.get("metadata") if isinstance(payload, dict) else None
                if isinstance(metadata, dict) and metadata.get("name") == self.name:
                    raise RuntimeError(f"owned FPM resource remains after cleanup: {self.kind}/{self.name}")
                raise RuntimeError(
                    f"kubectl returned an unexpected deletion probe for {self.kind}/{self.name}: {payload!r}"
                )
        elif "notfound" not in detail.replace(" ", "").lower():
            raise RuntimeError(
                f"could not verify deletion of {self.kind}/{self.name}: "
                f"delete_exit={deleted.returncode}, get_exit={observed.returncode}, output={detail!r}"
            )
        # Foreground cascading keeps the parent alive until its dependants are
        # deleted.  An eventually-consistent list may still expose Pod objects
        # that already carry deletionTimestamp; those no longer represent a
        # live reservation and must not downgrade a successful cell.
        remaining = self.pods(include_terminating=False)
        if remaining:
            raise RuntimeError(f"owned FPM pods remain after cleanup: {remaining}")


def _cell_generator_overrides(
    plan: FPMCollectionPlan,
    cell: FPMCell,
    base: dict[str, Any],
    *,
    smoke: bool = False,
) -> dict[str, Any]:
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
        "--benchmark-mode",
        cell.workload_kind,
        "--benchmark-warmup-iterations",
        str(plan.options.warmup_iterations),
        "--max-model-len",
        str(plan.options.vllm_max_model_len),
    ]
    if cell.workload_kind == "prefill" and not smoke:
        profile = plan.options.prefill_sampling
        compilation_config = {
            "cudagraph_capture_sizes": list(profile.cudagraph_capture_sizes),
            "max_cudagraph_capture_size": profile.max_cudagraph_capture_size,
        }
        scheduler_args.extend(
            [
                "--max-num-batched-tokens",
                str(profile.max_total_prefill_tokens),
                "--compilation-config",
                json.dumps(compilation_config, sort_keys=True, separators=(",", ":")),
                "--prefill-max-new-token-samples",
                str(profile.max_new_token_samples),
                "--prefill-max-kv-read-token-samples",
                str(profile.max_kv_read_token_samples),
            ]
        )
        if profile.max_batch_size is not None:
            scheduler_args.extend(["--max-num-seqs", str(profile.max_batch_size)])
    elif smoke:
        if cell.workload_kind == "prefill":
            scheduler_args.extend(
                [
                    "--prefill-max-new-token-samples",
                    "2",
                    "--prefill-max-kv-read-token-samples",
                    "2",
                    "--prefix-max-batch-size-samples",
                    "1",
                ]
            )
        else:
            scheduler_args.extend(
                [
                    "--decode-max-kv-read-token-samples",
                    "2",
                    "--decode-max-batch-size-samples",
                    "2",
                ]
            )
    model_args = []
    architecture = getattr(getattr(plan, "capability", None), "architecture", None)
    if architecture == "GlmMoeDsaForCausalLM":
        # This is the serving path validated by the pinned GLM-5.2 vLLM image.
        # The parser does not alter FPM scheduling, but keeping the model's
        # native runtime initialization avoids measuring a different engine.
        model_args.extend(["--trust-remote-code", "--reasoning-parser=glm45"])
    env = [
        {"name": "DYN_FPM_BENCHMARK_OUTPUT_PATH", "value": "/results/benchmark.json"},
        {"name": "FPM_RUN_ID", "value": cell.cell_id},
    ]
    total_gpus = cell.topology.total_gpus
    generated = {
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
                "kv_cache_dtype": cell.kv_cache_dtype,
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
        # Dynamo's engine-side default is 300 seconds. A formal native grid
        # can legitimately run longer even though the outer Kubernetes exec
        # timeout is four hours, so make the inner deadline campaign-safe.
        resolved_args.extend(["--benchmark-timeout", str(DEFAULT_BENCHMARK_TIMEOUT_SECONDS)])
    resolved_args.extend(scheduler_args)
    merged_agg = merged.setdefault("params", {}).setdefault("agg", {})
    merged_agg.update({"extra_cli_args": resolved_args})
    return merged


def _configured_sampling_metadata(
    plan: FPMCollectionPlan,
    cell: FPMCell,
    *,
    smoke: bool,
) -> dict[str, int]:
    if cell.workload_kind != "prefill":
        return {}
    if smoke:
        return {"prefill_max_new_token_samples": 2}
    profile = plan.options.prefill_sampling
    return {
        "prefill_cudagraph_capture_size_count": len(profile.cudagraph_capture_sizes),
        "prefill_requested_new_token_axis_count": len(profile.new_token_axis_points),
        "prefill_max_new_token_samples": profile.max_new_token_samples,
    }


def _render_cell(
    plan: FPMCollectionPlan,
    cell: FPMCell,
    cell_dir: Path,
    generator_overrides: dict[str, Any],
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    from aiconfigurator.generator.api import generate_from_request
    from aiconfigurator.generator.naive import build_naive_generator_params
    from aiconfigurator.generator.request import from_legacy_params

    overrides = _cell_generator_overrides(
        plan,
        cell,
        generator_overrides,
        smoke=smoke,
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
    # The Generator owns deployment rendering, but its naive serving defaults
    # intentionally synthesize max batch/token/sequence limits from an SLA.
    # Native self-benchmarking must instead observe the limits resolved by the
    # target vLLM image.  Remove only those optional engine-policy fields and
    # use the Generator's supported guard to keep its rule plugin from
    # reintroducing them; topology, dtype, model identity, and Pod rendering
    # continue through the normal typed request path.
    params["preserve_engine_limits"] = True
    role_params = params.setdefault("params", {}).setdefault("agg", {})
    for key in (
        "max_batch_size",
        "max_num_tokens",
        "max_seq_len",
        "tokens_per_block",
        "gpu_memory_utilization",
        "compilation_config",
        "cuda_graph_batch_sizes",
    ):
        role_params.pop(key, None)
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
    _atomic_json(cell_dir / "generator-request.json", params)
    return artifacts


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
    expected_plan_sha256: str | None = None,
    expected_attempt_id: str | None = None,
) -> int:
    """Validate PR11509 native rank artifacts and return the runtime point count."""
    return _runtime_collection_summary(
        cell,
        raw_root,
        expected_plan_sha256=expected_plan_sha256,
        expected_attempt_id=expected_attempt_id,
    )["measured_point_count"]


def _runtime_collection_summary(
    cell: FPMCell,
    raw_root: Path,
    *,
    expected_plan_sha256: str | None = None,
    expected_attempt_id: str | None = None,
) -> dict[str, int]:
    """Return auditable unique-axis counts from a validated native grid."""

    collection = validate_native_collection(
        cell,
        raw_root,
        expected_plan_sha256=expected_plan_sha256,
        expected_attempt_id=expected_attempt_id,
    )
    points = tuple(measurement.point for measurement in collection.points)
    summary = {
        "measured_point_count": len(points),
        "measured_batch_size_axis_count": len({int(point["batch_size"]) for point in points}),
        "measured_kv_read_axis_count": len({int(point["total_kv_read_tokens"]) for point in points}),
    }
    if cell.workload_kind == "prefill":
        summary["measured_new_token_axis_count"] = len({int(point["total_prefill_tokens"]) for point in points})
    return summary


def _load_checkpoint(path: Path, plan: FPMCollectionPlan, resume: bool) -> dict[str, Any]:
    if resume and path.exists():
        payload = json.loads(path.read_text())
        if payload.get("schema") != CHECKPOINT_SCHEMA or payload.get("plan_sha256") != plan.sha256:
            raise ValueError("FPM checkpoint does not match the current frozen plan")
        return payload
    return {"schema": CHECKPOINT_SCHEMA, "plan_sha256": plan.sha256, "cells": {}}


def _required_attempt_id(entry: dict[str, Any], cell_id: str) -> str:
    attempt_id = entry.get("attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise ValueError(f"passed FPM checkpoint cell {cell_id!r} has no attempt identity")
    return attempt_id


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
    runtime_wrapper = Path(__file__).resolve().parent / "runtime" / "run_with_etcd.sh"
    runtime_preflight = Path(__file__).resolve().parent / "runtime" / "preflight.py"
    target_cells = plan.cells[: (cell_limit or (1 if smoke else len(plan.cells)))]

    checkpoint_changed = False
    for cell in target_cells:
        entry = checkpoint["cells"].get(cell.cell_id)
        if not isinstance(entry, dict) or entry.get("status") != "passed":
            continue
        metadata = {
            "total_gpus": cell.topology.total_gpus,
            "point_source": "dynamo_native_self_benchmark",
            "global_warmup_iterations": plan.options.warmup_iterations,
            **_configured_sampling_metadata(plan, cell, smoke=smoke),
            **_runtime_timing_summary(root / "cells" / cell.cell_id / "raw"),
        }
        metadata.update(
            _runtime_collection_summary(
                cell,
                root / "cells" / cell.cell_id / "raw",
                expected_plan_sha256=plan.sha256,
                expected_attempt_id=_required_attempt_id(entry, cell.cell_id),
            )
        )
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
        if resume and previous.get("status") in {"failed", "cleanup_failed"} and not retry_failed:
            continue

        cell_dir = root / "cells" / cell.cell_id
        if cell_dir.exists() and not resume:
            shutil.rmtree(cell_dir)
        cell_dir.mkdir(parents=True, exist_ok=True)
        for stale_dir in (cell_dir / "raw", cell_dir / "logs"):
            if stale_dir.exists():
                shutil.rmtree(stale_dir)
        _atomic_json(cell_dir / "cell.json", cell.to_dict())
        cell_started = time.monotonic()
        started_at = _utc_now()
        attempt_id = uuid.uuid4().hex
        base_record = {
            "status": "running",
            "started_at": started_at,
            "attempt_id": attempt_id,
            "total_gpus": cell.topology.total_gpus,
            "point_source": "dynamo_native_self_benchmark",
            "global_warmup_iterations": plan.options.warmup_iterations,
            **_configured_sampling_metadata(plan, cell, smoke=smoke),
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
                smoke=smoke,
            )
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
                    run_script,
                    runtime_wrapper,
                    runtime_preflight,
                ],
            )
            resource.prepare_attempt(
                pods,
                cell_id=cell.cell_id,
                plan_sha256=plan.sha256,
                attempt_id=attempt_id,
            )
            resource.execute(pods)
            resource.collect(pods)
            runtime_collection = _runtime_collection_summary(
                cell,
                cell_dir / "raw",
                expected_plan_sha256=plan.sha256,
                expected_attempt_id=attempt_id,
            )
            checkpoint["cells"][cell.cell_id] = {
                **base_record,
                "status": "passed",
                "artifact_dir": str(cell_dir),
                "pods": pods,
                **runtime_collection,
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
                    record = checkpoint["cells"][cell.cell_id]
                    if record.get("status") == "passed":
                        record["status"] = "cleanup_failed"
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
            "sampling_profile": "dynamo_native_minimal_axes",
            "formal_database_written": False,
        }
        _atomic_json(checkpoint_path, checkpoint)
    elif not errors and all_passed:
        from .database import aggregate_cell, write_formal_database

        try:
            formal_rows = []
            for cell in plan.cells:
                entry = checkpoint["cells"].get(cell.cell_id)
                if not isinstance(entry, dict) or entry.get("status") != "passed":
                    raise ValueError(f"cannot publish non-passed FPM cell {cell.cell_id!r}")
                formal_rows.extend(
                    aggregate_cell(
                        plan,
                        cell,
                        root / "cells" / cell.cell_id,
                        expected_attempt_id=_required_attempt_id(entry, cell.cell_id),
                    )
                )
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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generator-driven Kubernetes execution for an immutable FPM plan."""

from __future__ import annotations

import copy
import fnmatch
import gzip
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
import uuid
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, suppress
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from aiconfigurator.fpm_contract import (
    FPM_AUXILIARY_KINDS,
    FPM_BENCHMARK_RESULT_GLOB,
    FPM_CELL_LABEL,
    FPM_ENGINE_BENCHMARK_OUTPUT_ENV,
    FPM_ENV_FILENAME,
    FPM_MANIFEST_FILENAME,
    FPM_RESULTS_DIR,
    FPM_RUN_ID_ENV,
    FPM_RUN_SCRIPT_FILENAME,
    FPM_WORKLOAD_KINDS,
    fpm_workload_node_count,
)

from .native_artifact import COLLECTOR_PROVENANCE_FILENAME, validate_native_collection
from .planner import FPMCell, FPMCollectionPlan

logger = logging.getLogger(__name__)

CHECKPOINT_SCHEMA = "aic-fpm-collector-checkpoint-v3"
# KV warm-up dominates a decode cell's wall clock (~80 min for a tep4 decode
# sweep on MiniMax M2.7); one hour would kill the engine mid-warm-up. 10800
# matches the r15 parity protocol's budget.
DEFAULT_BENCHMARK_TIMEOUT_SECONDS = 10800
RESULT_COPY_ATTEMPTS = 3
RESULT_COPY_TIMEOUT_SECONDS = 300
# Convergence budget for the background-delete escalation in cleanup(); the
# escalated parent vanishes immediately, but the GC still needs a moment to
# collect the orphaned children.
CLEANUP_ESCALATION_TIMEOUT_SECONDS = 120.0
CLEANUP_PROBE_INTERVAL_SECONDS = 5.0
# kubectl-cp truncation has only been observed on multi-MB files; smaller
# files skip the in-pod gzip and its extra proxied round-trips entirely.
RESULT_COMPRESSION_MIN_BYTES = 1024 * 1024
_FPM_VLLM_RUNTIME_ARGS = (
    "--distributed-executor-backend",
    "mp",
    "--distributed-timeout-seconds",
    "1800",
    "--no-enable-log-requests",
)
# Prefill and decode cells need opposite engine policies.
#
# Prefill keeps vLLM's defaults (prefix caching on, synchronous scheduling):
# points with total_kv_read_tokens > 0 are staged through the fake prefix
# cache, and `_bench_cached_kv_read_tokens` reads `Request.block_hashes`,
# which only exists while prefix caching builds a block hasher. Disabling it
# would fail every cached-prefill point's seed validation. Prefill is also
# insensitive to both flags (measured 100-108 ms across all four combinations
# at 8192 new tokens, M2.7 tp4+EP).
_FPM_VLLM_PREFILL_ARGS = ("--no-async-scheduling",)
# Decode keeps async scheduling on (vLLM's default) -- the steady-state
# second step models a production decode iteration, and production overlaps
# scheduler CPU work with the GPU. Against real traffic at (256, 2.1M KV):
# async 26.5 ms (1.03x of the 25.8 ms measured), sync 31.3 ms (1.21x).
#
# Decode also keeps prefix caching on (vLLM's default). The engine's KV
# warm-up reuses warmed prefixes across points and refuses to warm without
# prefix caching (skip_reason="prefix_caching_disabled"), collapsing every
# decode point back to the fake-KV fallback regime that underestimates
# capture-mode decode. The old `--no-enable-prefix-caching` pin dated from
# that fallback protocol, where each point admits batch_size synthetic
# full-context requests and `Request.__init__` block hashing would run
# inside the measured step (26.5 ms -> 121 ms at (batch 256, 2.1M KV) on
# M2.7 tp4+EP); under KV warm-up the hashing happens at seed time, outside
# the measured step, and the r15 parity baseline runs with the default.
_FPM_VLLM_DECODE_ARGS = ()
REMOTE_EXIT_MARKER = "__FPM_REMOTE_EXIT_CODE__="
REMOTE_FILES_MARKER = "__FPM_REMOTE_FILES__="
REMOTE_WORKDIR = "/tmp/fpm-bench"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


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


# Live kubectl children, registered so an interrupt can terminate them: a
# worker thread blocked in a subprocess wait cannot be interrupted by the main
# thread's signal, so without this a detached campaign stopped with SIGINT or
# SIGTERM would sit in ThreadPoolExecutor joins for up to the full exec
# timeout before salvage/teardown could start.
_ACTIVE_COMMANDS: set[subprocess.Popen[str]] = set()
_ACTIVE_COMMANDS_LOCK = threading.Lock()


def terminate_active_commands() -> int:
    """TERM every live kubectl child; returns how many were signalled."""

    with _ACTIVE_COMMANDS_LOCK:
        processes = list(_ACTIVE_COMMANDS)
    for process in processes:
        with suppress(OSError):
            process.terminate()
    return len(processes)


def _manifest_documents(manifest: Path) -> list[dict[str, Any]]:
    documents = list(yaml.safe_load_all(manifest.read_text()))
    if not documents or any(not isinstance(document, dict) for document in documents):
        raise TypeError("generated k8s_deploy.yaml must contain only YAML mappings")
    unsupported = {
        str(document.get("kind"))
        for document in documents
        if document.get("kind") not in FPM_WORKLOAD_KINDS | FPM_AUXILIARY_KINDS
    }
    if unsupported:
        raise ValueError(f"unsupported generated FPM resource kinds: {sorted(unsupported)}")
    return documents


def _workload_document(documents: list[dict[str, Any]]) -> dict[str, Any]:
    workloads = [document for document in documents if document.get("kind") in FPM_WORKLOAD_KINDS]
    if len(workloads) != 1:
        raise ValueError("generated k8s_deploy.yaml must contain exactly one Pod, LeaderWorkerSet, or PodCliqueSet")
    workload = workloads[0]
    compute_domains = [document for document in documents if document.get("kind") in FPM_AUXILIARY_KINDS]
    if len(compute_domains) > 1:
        raise ValueError("generated k8s_deploy.yaml supports at most one ComputeDomain")
    if compute_domains and workload.get("kind") not in {"LeaderWorkerSet", "PodCliqueSet"}:
        raise ValueError("generated ComputeDomain requires a multi-node workload")
    workload_namespace = str((workload.get("metadata") or {}).get("namespace") or "default")
    for document in documents:
        namespace = str((document.get("metadata") or {}).get("namespace") or "default")
        if namespace != workload_namespace:
            raise ValueError("all generated FPM resources must use the workload namespace")
    return workload


def _resource_identity(document: dict[str, Any]) -> tuple[str, str, str]:
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError("generated FPM resources require metadata mappings")
    kind = document.get("kind")
    name = metadata.get("name")
    if not isinstance(kind, str) or not kind or not isinstance(name, str) or not name:
        raise ValueError("generated FPM resources require non-empty kind and metadata.name")
    return kind, name, str(metadata.get("namespace") or "default")


def _run_command(
    args: list[str],
    *,
    check: bool = True,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    def _stop(process: subprocess.Popen[str]) -> None:
        # Never wait on pipes here and never rely on SIGKILL alone: `tsh
        # kubectl` re-execs itself as a pipe-sharing grandchild that SIGKILL
        # on the wrapper cannot reach (an unbounded drain would then hang on
        # the orphan's open pipe forever), while terminate() IS forwarded.
        # Signal politely, give the wrapper a moment, then kill and reap the
        # direct child only.
        process.terminate()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=10)
        process.kill()
        process.wait()

    with subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_command_env(),
    ) as process:
        with _ACTIVE_COMMANDS_LOCK:
            _ACTIVE_COMMANDS.add(process)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            _stop(process)
            raise subprocess.TimeoutExpired(args, timeout or 0, output=error.output, stderr=error.stderr) from None
        except BaseException:
            # Mirror subprocess.run: never abandon a live child (a hung
            # kubectl would otherwise block Popen.__exit__'s untimed wait
            # forever, unreachable after the registry discard below).
            _stop(process)
            raise
        finally:
            with _ACTIVE_COMMANDS_LOCK:
                _ACTIVE_COMMANDS.discard(process)
    if check and process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, args, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)


class KubernetesCellRunner:
    """Own one generated Pod/LWS/PCS and auxiliary resources through deletion."""

    def __init__(self, manifest: Path, cell_dir: Path) -> None:
        self.manifest = manifest
        self.cell_dir = cell_dir
        documents = _manifest_documents(manifest)
        workload = _workload_document(documents)
        metadata = workload.get("metadata") or {}
        self.name = str(metadata["name"])
        self.namespace = str(metadata.get("namespace") or "default")
        self.kind = str(workload["kind"])
        self.expected_labels = dict(metadata.get("labels") or {})
        # Pods are selected exclusively through the collector-owned identity
        # label the Generator propagates to every pod template; the collector
        # never depends on Generator-chosen labels or on the workload kind.
        cell_label = self.expected_labels.get(FPM_CELL_LABEL)
        if not isinstance(cell_label, str) or not cell_label:
            raise ValueError(
                f"generated FPM workload {self.kind}/{self.name} does not carry the collector cell "
                f"label {FPM_CELL_LABEL!r}; the Generator must apply K8sConfig.fpm_resource_labels "
                "to workload metadata and every pod template"
            )
        self.selector = f"{FPM_CELL_LABEL}={cell_label}"
        self.resources = [_resource_identity(document) for document in documents]
        if len(set(self.resources)) != len(self.resources):
            raise ValueError("generated k8s_deploy.yaml contains duplicate resources")
        self.kubectl = _kubectl_command()

    def _owned_resources(self) -> list[tuple[str, str, str]]:
        resources = getattr(self, "resources", None)
        if resources is not None:
            return resources
        return [(self.kind, self.name, self.namespace)]

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
        if applied.returncode != 0:
            detail = (applied.stderr or applied.stdout).strip()
            raise RuntimeError(
                f"kubectl apply failed for {self.kind}/{self.name}: apply_exit={applied.returncode}, output={detail!r}"
            )
        for kind, name, namespace in self._owned_resources():
            observed = self._kubectl(
                "get",
                f"{kind}/{name}",
                "-n",
                namespace,
                "-o",
                "json",
                check=False,
                timeout=60,
            )
            try:
                payload = json.loads(observed.stdout)
            except json.JSONDecodeError as error:
                detail = (observed.stderr or observed.stdout).strip()
                raise RuntimeError(
                    f"kubectl apply returned no verifiable object for {kind}/{name}: output={detail!r}"
                ) from error
            metadata = payload.get("metadata") if isinstance(payload, dict) else None
            labels = metadata.get("labels") if isinstance(metadata, dict) else None
            verify_labels = kind == self.kind and name == self.name
            if (
                observed.returncode != 0
                or not isinstance(payload, dict)
                or payload.get("kind") != kind
                or not isinstance(metadata, dict)
                or metadata.get("name") != name
                or (
                    verify_labels
                    and (
                        not isinstance(labels, dict)
                        or any(labels.get(key) != value for key, value in self.expected_labels.items())
                    )
                )
            ):
                raise RuntimeError(f"kubectl apply returned the wrong object for {kind}/{name}: {payload!r}")

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
            self._exec_checked(pod, ["mkdir", "-p", REMOTE_WORKDIR, FPM_RESULTS_DIR], timeout=60)
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
            f"path = pathlib.Path('{FPM_RESULTS_DIR}') / sys.argv[2]; "
            "path.write_text(json.dumps(payload, sort_keys=True) + '\\n')"
        )
        for pod in pods:
            self._exec_checked(
                pod,
                [
                    "find",
                    FPM_RESULTS_DIR,
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
            f"root = pathlib.Path('{FPM_RESULTS_DIR}'); "
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
            ["bash", f"{REMOTE_WORKDIR}/fpm_exec.sh"],
            timeout=timeout_seconds,
        )
        return pod, completed

    def _compress_remote_result(self, pod: str, remote_path: str, remote_gz: str) -> bool:
        """Best-effort in-pod gzip of a result file before transfer.

        kubectl cp through proxied clusters silently truncates large files
        (observed repeatedly at 18-148 MB); shipping a ~10x smaller archive
        shrinks the truncation window and the sha check below still verifies
        the decompressed bytes against the pod-side manifest. The probe is
        retried once so a transient proxy hiccup is not mistaken for a missing
        gzip binary. Returns False when compression is unavailable so the
        caller falls back to the raw copy path.
        """
        last_error: Exception | None = None
        for _probe_attempt in range(2):
            try:
                self._exec_checked(
                    pod,
                    ["sh", "-c", f"gzip -cf {shlex.quote(remote_path)} > {shlex.quote(remote_gz)}"],
                    timeout=RESULT_COPY_TIMEOUT_SECONDS,
                )
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                last_error = error
                continue
            return True
        logger.warning(
            "in-pod gzip unavailable for %s on %s (%s); falling back to raw copy",
            remote_path,
            pod,
            last_error,
        )
        return False

    def _remove_remote_file(self, pod: str, remote_path: str) -> None:
        try:
            self._exec_checked(pod, ["rm", "-f", remote_path], timeout=RESULT_COPY_TIMEOUT_SECONDS)
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            logger.warning("failed to remove transfer temp %s on %s: %s", remote_path, pod, error)

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

        remote_path = f"{FPM_RESULTS_DIR}/{relative.as_posix()}"
        # The transfer temp must live outside /results: the pod-side result
        # manifest rglobs /results, so a temp leaked there would enter a later
        # salvage manifest as a phantom result file and abort the salvage.
        remote_gz = f"{REMOTE_WORKDIR}/{relative.as_posix().replace('/', '__')}.__xfer.gz"
        should_compress = int(expected.get("size", 0)) >= RESULT_COMPRESSION_MIN_BYTES
        compressed = should_compress and self._compress_remote_result(pod, remote_path, remote_gz)

        partial = target.with_name(f".{target.name}.part")
        partial_gz = target.with_name(f".{target.name}.part.gz")
        failures = []
        try:
            for attempt in range(1, RESULT_COPY_ATTEMPTS + 1):
                partial.unlink(missing_ok=True)
                partial_gz.unlink(missing_ok=True)
                source = remote_gz if compressed else remote_path
                destination = partial_gz if compressed else partial
                try:
                    copied = self._kubectl(
                        "cp",
                        f"{self.namespace}/{pod}:{source}",
                        str(destination),
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
                if compressed and destination.is_file():
                    try:
                        with gzip.open(partial_gz, "rb") as src, open(partial, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                    except (OSError, EOFError, zlib.error) as error:
                        failures.append(
                            {
                                "attempt": attempt,
                                "local_exit": copied.returncode,
                                "gz_size": partial_gz.stat().st_size,
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
                        "gz_size": partial_gz.stat().st_size if partial_gz.is_file() else None,
                        "output": (copied.stderr or copied.stdout).strip(),
                    }
                )
        finally:
            partial.unlink(missing_ok=True)
            partial_gz.unlink(missing_ok=True)
            if should_compress:
                # Success or failure alike: `gzip -cf X > Y` pre-creates Y via
                # shell redirection even when gzip itself fails, so the temp
                # must be removed on every path.
                self._remove_remote_file(pod, remote_gz)
        raise RuntimeError(
            f"failed to collect exact result file {remote_name!r} from {pod} "
            f"after {RESULT_COPY_ATTEMPTS} attempts: expected={expected!r}, "
            f"failures={failures!r}"
        )

    def execute(self, pods: list[str], timeout_seconds: int = 14400) -> None:
        logs_dir = self.cell_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        failures = []
        pool = ThreadPoolExecutor(max_workers=len(pods))
        try:
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
        except BaseException:
            # An interrupt lands here while worker threads sit in kubectl
            # waits they cannot be signalled out of; kill the children first
            # so the pool join below returns promptly and salvage/teardown
            # can start.
            terminate_active_commands()
            raise
        finally:
            pool.shutdown(wait=True)
        if failures:
            raise RuntimeError(f"staged FPM fpm_exec.sh failed: {failures}")

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
                fnmatch.fnmatch(Path(name).name, FPM_BENCHMARK_RESULT_GLOB)
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

    def _resource_remains(self, kind: str, name: str, namespace: str, *, delete_exit: int) -> bool:
        """Fail-closed deletion probe: False only on a verified NotFound."""
        observed = self._kubectl(
            "get",
            f"{kind}/{name}",
            "-n",
            namespace,
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
                        f"could not verify deletion of {kind}/{name}: delete_exit={delete_exit}, output={detail!r}"
                    ) from error
                return False
            metadata = payload.get("metadata") if isinstance(payload, dict) else None
            if isinstance(metadata, dict) and metadata.get("name") == name:
                return True
            raise RuntimeError(f"kubectl returned an unexpected deletion probe for {kind}/{name}: {payload!r}")
        if "notfound" not in detail.replace(" ", "").lower():
            raise RuntimeError(
                f"could not verify deletion of {kind}/{name}: "
                f"delete_exit={delete_exit}, get_exit={observed.returncode}, output={detail!r}"
            )
        return False

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
        stuck = [
            (kind, name, namespace)
            for kind, name, namespace in self._owned_resources()
            if self._resource_remains(kind, name, namespace, delete_exit=deleted.returncode)
        ]
        if stuck:
            # A controller that keeps reconciling children of a terminating
            # parent can stall a foreground cascade forever (observed live:
            # LWS v0.6.0 recreated its child StatefulSet against the GC for
            # 40+ minutes while the workload was still unscheduled). Switch
            # the stuck parents to a background delete: the parent object goes
            # away immediately, the controller loses its reconcile target, and
            # the GC collects the orphans. Deliberately no finalizer stripping
            # here — when even this cannot converge, fail loudly below.
            for kind, name, namespace in stuck:
                self._kubectl(
                    "delete",
                    f"{kind}/{name}",
                    "-n",
                    namespace,
                    "--ignore-not-found=true",
                    "--cascade=background",
                    "--wait=false",
                    check=False,
                    timeout=60,
                )
            deadline = time.monotonic() + CLEANUP_ESCALATION_TIMEOUT_SECONDS
            while True:
                stuck = [
                    (kind, name, namespace)
                    for kind, name, namespace in stuck
                    if self._resource_remains(kind, name, namespace, delete_exit=deleted.returncode)
                ]
                if not stuck or time.monotonic() >= deadline:
                    break
                time.sleep(CLEANUP_PROBE_INTERVAL_SECONDS)
            if stuck:
                names = ", ".join(f"{kind}/{name}" for kind, name, _namespace in stuck)
                raise RuntimeError(f"owned FPM resource remains after cleanup: {names}")
        # Foreground cascading keeps the parent alive until its dependants are
        # deleted.  An eventually-consistent list may still expose Pod objects
        # that already carry deletionTimestamp; those no longer represent a
        # live reservation and must not downgrade a successful cell. After a
        # background escalation the orphaned children may outlive the parent
        # for a moment, so give the GC the same bounded window.
        deadline = time.monotonic() + CLEANUP_ESCALATION_TIMEOUT_SECONDS
        while True:
            remaining = self.pods(include_terminating=False)
            if not remaining:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"owned FPM pods remain after cleanup: {remaining}")
            time.sleep(CLEANUP_PROBE_INTERVAL_SECONDS)


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
        {"name": FPM_ENGINE_BENCHMARK_OUTPUT_ENV, "value": f"{FPM_RESULTS_DIR}/benchmark.json"},
        {"name": FPM_RUN_ID_ENV, "value": cell.cell_id},
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
                FPM_CELL_LABEL: cell.cell_id,
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

    # Pod-selection isolation rests on this label equaling the cell id; a
    # policy override that changed it would make the runner adopt (or miss)
    # another cell's pods, so re-assert the invariant after the merge.
    merged_labels = (merged.get("K8sConfig") or {}).get("fpm_resource_labels") or {}
    if merged_labels.get(FPM_CELL_LABEL) != cell.cell_id:
        raise ValueError(f"FPM resource label {FPM_CELL_LABEL} must equal the cell id after overrides")

    # User-supplied entries (the declared K8sConfig.extra_env input) resolve
    # first, then backend-policy entries, then Collector-owned identities; a
    # name claimed twice with different values fails closed rather than
    # silently overriding.
    base_env = (base.get("K8sConfig") or {}).get("extra_env") or []
    policy_env = (policy.get("K8sConfig") or {}).get("extra_env") or []
    resolved_env: dict[str, dict[str, str]] = {}
    for item in [*base_env, *policy_env, *env]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise TypeError("K8sConfig.extra_env entries must be {name, value} mappings")
        name = item["name"]
        existing = resolved_env.get(name)
        if existing is not None and existing != item:
            raise ValueError(f"conflicting FPM environment value for {name}")
        resolved_env[name] = copy.deepcopy(item)
    merged.setdefault("K8sConfig", {})["extra_env"] = list(resolved_env.values())

    policy_args = ((policy.get("params") or {}).get("agg") or {}).get("extra_cli_args") or []
    workload_args = _FPM_VLLM_DECODE_ARGS if cell.workload_kind == "decode" else _FPM_VLLM_PREFILL_ARGS
    resolved_args = [*_FPM_VLLM_RUNTIME_ARGS, *workload_args, *model_args, *policy_args]
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
) -> dict[str, int | str]:
    if cell.workload_kind != "prefill":
        # Auditable measurement condition: decode engines run with prefix
        # caching pinned off (see _cell_generator_overrides).
        return {"decode_prefix_caching": "disabled"}
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
    # The Generator owns deployment rendering, but its naive serving defaults
    # intentionally synthesize max batch/token/sequence limits from an SLA.
    # Native self-benchmarking must instead observe the limits resolved by the
    # target vLLM image, so declare preserve_engine_limits: the Generator
    # strips those optional engine-policy fields and keeps its rule plugin
    # from reintroducing them; topology, dtype, model identity, and Pod
    # rendering continue through the normal typed request path.
    params = build_naive_generator_params(
        model_name=plan.model_path,
        total_gpus=cell.topology.total_gpus,
        system_name=plan.system,
        backend_name=plan.backend,
        mode="agg",
        generator_dynamo_version=overrides.get("generator_dynamo_version"),
        generator_overrides=overrides,
        preserve_engine_limits=True,
        # Render is a pure function of the frozen plan: model metadata comes
        # from the plan's resolved config evidence, never from re-resolving
        # plan.model_path — which may identify a checkpoint reachable only
        # inside the cluster (--fpm-model-config campaigns).
        model_config=plan.capability.model_config.parsed_payload(),
    )
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
    return fpm_workload_node_count(_workload_document(_manifest_documents(manifest)))


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
    for path in sorted(raw_root.glob(f"**/{FPM_BENCHMARK_RESULT_GLOB}")):
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


def _salvage_artifacts(resource: KubernetesCellRunner, cell_id: str) -> None:
    """Best-effort artifact salvage after a failed or interrupted attempt.

    kubectl-exec disconnects do not stop pod processes, so the runtime may
    still be writing when the failure handler runs; a file that grows between
    the manifest snapshot and the copy can never match its size/sha entry and
    would abort the whole salvage. Wait for two consecutive identical result
    manifests per pod before collecting, and log every failure — salvage runs
    inside failure handling and must never raise or fall silent.
    """

    try:
        pods = resource.pods()
    except Exception as error:
        logger.warning("FPM salvage for %s could not list pods: %s", cell_id, error)
        return
    for pod in pods:
        try:
            previous = None
            for _ in range(6):
                current = resource._remote_result_manifest(pod)
                if current == previous:
                    break
                previous = current
                time.sleep(3)
            else:
                logger.warning(
                    "FPM salvage for %s: results on %s were still changing after the settle window",
                    cell_id,
                    pod,
                )
        except Exception as error:
            logger.warning("FPM salvage for %s could not settle %s: %s", cell_id, pod, error)
    try:
        resource.collect(pods, require_benchmark=False)
    except Exception as error:
        logger.warning("FPM salvage for %s did not capture a complete artifact set: %s", cell_id, error)


# Statuses whose salvaged artifacts may be acknowledged without a rerun. A
# cell that failed, was interrupted, or lost its host process can leave a
# complete, attempt-matched artifact set behind; "cleanup_failed" is excluded
# because only a rerun re-drives the verified deletion of leaked resources.
FPM_RECOVERABLE_STATUSES = frozenset({"failed", "interrupted", "running"})


def _recover_completed_attempt(
    plan: FPMCollectionPlan,
    cell: FPMCell,
    root: Path,
    entry: dict[str, Any],
) -> dict[str, Any] | None:
    """Recover a non-passed checkpoint cell whose salvaged artifacts are complete.

    Recovery only flips identity fields; the passed-entry refresh loop that
    follows in ``run_collection`` owns the metadata of every passed record.
    Entries carrying ``cleanup_error`` are never recovered, whatever their
    status: their Kubernetes teardown was not verified, and only a rerun
    re-applies and re-deletes the leaked resource. ``running`` entries carry
    the same unverified-teardown hazard implicitly (the status is only
    overwritten after cleanup runs), so recovering one requires a verified
    delete of the abandoned workload first.
    """

    status = entry.get("status")
    if status not in FPM_RECOVERABLE_STATUSES or "cleanup_error" in entry:
        return None
    cell_dir = root / "cells" / cell.cell_id
    try:
        attempt_id = _required_attempt_id(entry, cell.cell_id)
        _runtime_collection_summary(
            cell,
            cell_dir / "raw",
            expected_plan_sha256=plan.sha256,
            expected_attempt_id=attempt_id,
        )
    except (OSError, TypeError, ValueError) as error:
        logger.info(
            "FPM cell %s (status=%s) is not recoverable from local artifacts: %s",
            cell.cell_id,
            status,
            error,
        )
        return None

    if status == "running":
        # A persistent "running" status means the attempt's finally block
        # never ran: its Kubernetes teardown is unverified by construction
        # (the checkpoint is only rewritten AFTER cleanup), and once this
        # entry flips to passed the cell is skipped forever, so no later
        # rerun would re-drive the delete. Verify the teardown now or refuse
        # recovery — the un-recovered cell then reruns, and the rerun's
        # unconditional pre-apply cleanup owns the leak.
        manifest = cell_dir / FPM_MANIFEST_FILENAME
        if not manifest.exists():
            logger.info(
                "FPM cell %s (status=running) has no manifest to verify teardown against; not recovering",
                cell.cell_id,
            )
            return None
        try:
            KubernetesCellRunner(manifest, cell_dir).cleanup()
        except Exception as error:
            logger.warning(
                "FPM cell %s recovery refused: teardown of the abandoned workload failed: %s",
                cell.cell_id,
                error,
            )
            return None

    recovered = dict(entry)
    original_error = {key: recovered.pop(key) for key in ("error_type", "error") if key in recovered}
    recovered.update(
        {
            "status": "passed",
            "artifact_dir": str(cell_dir),
            "artifact_recovery": {
                "recovered_at": _utc_now(),
                "validation": "native_collection_plan_and_attempt_identity",
                "original_status": status,
                **original_error,
            },
        }
    )
    return recovered


@contextmanager
def _sigterm_as_interrupt():
    """Route SIGTERM through the same salvage -> interrupted-checkpoint ->
    verified-cleanup path as Ctrl-C. Detached campaigns are stopped with plain
    ``kill``; the default handler would skip every finally block and leave a
    live workload behind a checkpoint stuck at status=running. No-op off the
    main thread, where signal handlers cannot be installed (e.g. tests driving
    the runner from a worker thread)."""

    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def _handle(signum: int, frame: object) -> None:
        # Raise only — no work in signal context. Acquiring the (non-reentrant)
        # command-registry lock here could deadlock against the main thread's
        # own critical section. Child termination is owned by the exception
        # paths: _run_command kills its child on any abandonment, and
        # execute() terminates all registered children before joining workers.
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, _handle)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


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
    publish_partial: bool = False,
) -> list[dict[str, object]]:
    """Render and run every cell, always tearing down owned resources."""

    with _sigterm_as_interrupt():
        return _run_collection_impl(
            plan,
            generator_overrides=generator_overrides,
            checkpoint_dir=checkpoint_dir,
            artifact_root=artifact_root,
            resume=resume,
            retry_failed=retry_failed,
            smoke=smoke,
            cell_limit=cell_limit,
            database_root=database_root,
            publish_partial=publish_partial,
        )


def _run_collection_impl(
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
    publish_partial: bool = False,
) -> list[dict[str, object]]:
    root = Path(artifact_root).expanduser().resolve() / plan.sha256[:16]
    if smoke:
        root /= "smoke"
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "collection-plan.json", plan.to_dict())
    checkpoint_name = "fpm_forward_smoke.json" if smoke else "fpm_forward.json"
    checkpoint_path = Path(checkpoint_dir).expanduser().resolve() / checkpoint_name
    checkpoint = _load_checkpoint(checkpoint_path, plan, resume)
    errors: list[dict[str, object]] = []
    runtime_exec = Path(__file__).resolve().parent / "runtime" / "fpm_exec.sh"
    runtime_preflight = Path(__file__).resolve().parent / "runtime" / "preflight.py"
    target_cells = plan.cells[: (cell_limit or (1 if smoke else len(plan.cells)))]

    checkpoint_changed = False
    # Recovery costs zero cluster time, so it runs on every resume: it only
    # acknowledges strictly validated, attempt-matched artifacts that already
    # exist on disk. Rerunning (--resume-retry-failed) remains the only path
    # that schedules cluster work.
    if resume:
        for cell in target_cells:
            entry = checkpoint["cells"].get(cell.cell_id)
            if not isinstance(entry, dict):
                continue
            recovered = _recover_completed_attempt(plan, cell, root, entry)
            if recovered is None:
                continue
            checkpoint["cells"][cell.cell_id] = recovered
            checkpoint_changed = True
            logger.info(
                "Recovered completed FPM cell %s from strictly validated artifacts for attempt %s",
                cell.cell_id,
                recovered["attempt_id"],
            )
        if checkpoint_changed:
            _atomic_json(checkpoint_path, checkpoint)
            checkpoint_changed = False

    for cell in target_cells:
        entry = checkpoint["cells"].get(cell.cell_id)
        if not isinstance(entry, dict) or entry.get("status") != "passed":
            continue
        # This refresh only polishes checkpoint metadata for cells whose
        # results were already validated and published; raw artifacts that
        # are no longer readable (disk reclaimed, resume from a different
        # working directory) must not abort the whole resume and never
        # demote the cell's passed status.
        try:
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
        except (OSError, TypeError, ValueError) as error:
            logger.warning(
                "Skipping metadata refresh for passed FPM cell %s: %s",
                cell.cell_id,
                error,
            )
            continue
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
        # Collector-side phase segmentation for the run manifest (R16 §3):
        # external segments only - engine-internal phases (kvwarm, seeding)
        # come from the engine's own artifact fields once dynamo-fpm phase
        # instrumentation lands. Plain monotonic marks; nothing is written
        # on the measurement path.
        phase_marks: dict[str, float] = {}
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
            manifest = cell_dir / FPM_MANIFEST_FILENAME
            run_script = cell_dir / FPM_RUN_SCRIPT_FILENAME
            env_script = cell_dir / FPM_ENV_FILENAME
            if not manifest.exists() or not run_script.exists() or not env_script.exists():
                raise RuntimeError(
                    f"Generator FPM target did not emit {FPM_MANIFEST_FILENAME}, "
                    f"{FPM_ENV_FILENAME}, and {FPM_RUN_SCRIPT_FILENAME}"
                )
            resource = KubernetesCellRunner(manifest, cell_dir)
            # A prior invocation may have left the same-named workload alive
            # (cleanup timeout, killed collector host) even when THIS
            # checkpoint has no record of the cell: workload names derive
            # deterministically from the cell_id, so a fresh checkpoint dir
            # proves nothing about the cluster. apply() would adopt such pods
            # and let the stale engine write into this attempt's freshly wiped
            # /results, so always drive a verified ignore-not-found delete
            # before applying.
            phase_marks["render_s"] = round(time.monotonic() - cell_started, 3)
            resource.cleanup()
            resource.apply()
            mark = time.monotonic()
            pods = resource.wait_ready(_expected_nodes(manifest))
            phase_marks["schedule_s"] = round(time.monotonic() - mark, 3)
            mark = time.monotonic()
            resource.stage(
                pods,
                [
                    run_script,
                    env_script,
                    runtime_exec,
                    runtime_preflight,
                ],
            )
            resource.prepare_attempt(
                pods,
                cell_id=cell.cell_id,
                plan_sha256=plan.sha256,
                attempt_id=attempt_id,
            )
            phase_marks["stage_s"] = round(time.monotonic() - mark, 3)
            mark = time.monotonic()
            resource.execute(pods)
            phase_marks["execute_wall_s"] = round(time.monotonic() - mark, 3)
            mark = time.monotonic()
            resource.collect(pods)
            phase_marks["collect_s"] = round(time.monotonic() - mark, 3)
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
                "collector_phase_seconds": dict(phase_marks),
            }
        except KeyboardInterrupt:
            if resource is not None:
                _salvage_artifacts(resource, cell.cell_id)
            checkpoint["cells"][cell.cell_id] = {
                **base_record,
                "status": "interrupted",
                "artifact_dir": str(cell_dir),
            }
            raise
        except Exception as error:
            if resource is not None:
                _salvage_artifacts(resource, cell.cell_id)
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
    # R16 §3 run manifest: the machine-readable timing map for the speed-up
    # campaign. Collector-observable segments land now; engine-internal
    # phases (kvwarm/seeding splits) are declared interface fields sourced
    # from the engine artifact once dynamo-fpm phase instrumentation exists.
    manifest_cells: dict[str, Any] = {}
    run_total = 0.0
    for cell in target_cells:
        entry = checkpoint["cells"].get(cell.cell_id)
        if not isinstance(entry, dict):
            continue
        duration = float(entry.get("duration_seconds") or 0.0)
        run_total += duration
        manifest_cells[cell.cell_id] = {
            "status": entry.get("status"),
            "topology": cell.topology.to_dict(),
            "workload_kind": cell.workload_kind,
            "cell_total_s": duration,
            "collector_phase_seconds": entry.get("collector_phase_seconds"),
            "engine_timing": {
                "benchmark_elapsed_seconds": entry.get("benchmark_elapsed_seconds"),
                "measured_iteration_seconds": entry.get("measured_iteration_seconds"),
            },
            "engine_phase_seconds": {
                "engine_launch_s": None,
                "kvwarm_warmup_s": None,
                "seeding_s": None,
                "inference_s": entry.get("measured_iteration_seconds"),
                "note": "pending dynamo-fpm phase instrumentation; interface per R16 §3",
            },
        }
    _atomic_json(
        root / "run-manifest.json",
        {
            "schema_name": "aic_fpm_run_manifest",
            "schema_version": 1,
            "plan_sha256": plan.sha256,
            "smoke": smoke,
            "run_total_s": round(run_total, 3),
            "cells": manifest_cells,
        },
    )
    all_passed = all(checkpoint["cells"].get(cell.cell_id, {}).get("status") == "passed" for cell in target_cells)
    # Formal publication eligibility must agree with completion: a deliberate
    # partial run (cell_limit below the frozen plan) can pass every targeted
    # cell yet cannot publish, and deserves the honest campaign_incomplete
    # classification rather than a formal_database_failed error.
    covers_full_plan = len(target_cells) == len(plan.cells)
    if smoke and not errors and all_passed:
        checkpoint["smoke"] = {
            "status": "passed",
            "cell_count": len(target_cells),
            "sampling_profile": "dynamo_native_minimal_axes",
            "formal_database_written": False,
        }
        _atomic_json(checkpoint_path, checkpoint)
    elif (not errors and all_passed and covers_full_plan) or (
        # Smoke rows must never reach the formal database, no matter which
        # publication escape hatches are set.
        not smoke
        and publish_partial
        and any(
            isinstance(checkpoint["cells"].get(cell.cell_id), dict)
            and checkpoint["cells"][cell.cell_id].get("status") == "passed"
            for cell in plan.cells
        )
    ):
        from .database import aggregate_cell, write_formal_database

        # Explicit partial publication: rows from passed cells only; the
        # missing cells are recorded so coverage is auditable, never implied.
        publishable_cells = [
            cell
            for cell in plan.cells
            if isinstance(checkpoint["cells"].get(cell.cell_id), dict)
            and checkpoint["cells"][cell.cell_id].get("status") == "passed"
        ]
        missing_cells = [cell.cell_id for cell in plan.cells if cell not in publishable_cells]
        partial = bool(missing_cells)
        try:
            if partial and not publish_partial:
                raise ValueError("internal: partial publication reached without --fpm-publish-partial")
            formal_rows = []
            for cell in publishable_cells:
                entry = checkpoint["cells"].get(cell.cell_id)
                formal_rows.extend(
                    aggregate_cell(
                        plan,
                        cell,
                        root / "cells" / cell.cell_id,
                        expected_attempt_id=_required_attempt_id(entry, cell.cell_id),
                    )
                )
            systems_root = Path(database_root).expanduser().resolve() if database_root else None
            parquet_path, metadata_path, first_wins_skipped = write_formal_database(
                plan, formal_rows, systems_root=systems_root
            )
            checkpoint["database"] = {
                "status": "passed",
                "parquet": str(parquet_path),
                "metadata": str(metadata_path),
                "row_count": len(formal_rows),
                "published_cells": len(publishable_cells) - len(first_wins_skipped),
                "plan_cells": len(plan.cells),
                "missing_cells": missing_cells,
                "skipped_first_publisher_wins": list(first_wins_skipped),
            }
            if partial:
                logger.warning(
                    "fpm_forward: PARTIAL publication (--fpm-publish-partial): %d/%d cells published, missing: %s",
                    len(publishable_cells),
                    len(plan.cells),
                    ", ".join(missing_cells),
                )
                # Publication succeeded, but the run must still exit nonzero:
                # a resumed run skips previously-failed cells without re-adding
                # their execution errors, and a clean exit would hide them.
                errors.append(
                    {
                        "module": "fpm_forward",
                        "error_type": "IncompleteCampaign",
                        "error_message": (
                            "partial publication: "
                            f"{len(missing_cells)} of {len(plan.cells)} plan cells are not in the "
                            f"formal database: {', '.join(missing_cells)}"
                        ),
                        "classification": "campaign_incomplete",
                    }
                )
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
    elif not errors:
        errors.append(
            {
                "module": "fpm_forward",
                "error_type": "IncompleteCampaign",
                "error_message": "not every frozen FPM cell is passed; formal database was not written",
                "classification": "campaign_incomplete",
            }
        )
    return errors

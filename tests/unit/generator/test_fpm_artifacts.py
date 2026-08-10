# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public-contract tests for the reusable-Pod FPM artifact target.

The Generator side of the FPM contract renders exactly three artifacts:
``k8s_deploy.yaml`` (the resource workload), ``fpm_env.sh`` (the contract's
``FPM_ENV_EXPORTED_VARS``), and a thin ``run.sh`` that only launches the
engine. Collection behavior (gate, result checker, barrier, follower
classification, engine termination) lives in the collector's staged runtime
and is tested there.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shlex
import stat
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml

from aiconfigurator.fpm_contract import (
    FPM_BARRIER_TIMEOUT_ENV,
    FPM_CELL_LABEL,
    FPM_ENV_EXPORTED_VARS,
    FPM_ENV_FILENAME,
    FPM_MANIFEST_FILENAME,
    FPM_NATIVE_BENCHMARK_RESULT_SCHEMA_VERSION,
    FPM_RUN_ID_ENV,
    FPM_RUN_SCRIPT_FILENAME,
)
from aiconfigurator.generator.aggregators import generate_config_from_input_dict
from aiconfigurator.generator.api import generate_backend_artifacts
from aiconfigurator.generator.main import main as generator_main
from aiconfigurator.generator.rendering.engine import render_backend_templates

pytestmark = pytest.mark.unit

_BACKEND_VERSION = "0.20.1"
_COMPILATION_CONFIG = json.dumps(
    {
        "cudagraph_mode": "FULL",
        "max_capture_size": 1024,
        "compile_sizes": [1, 2, 4, 8],
    }
)
_ARTIFACT_TRIO = {FPM_MANIFEST_FILENAME, FPM_ENV_FILENAME, FPM_RUN_SCRIPT_FILENAME}
_COLLECTION_VOCABULARY = (
    "check_result_files",
    "29511",
    "2379",
    "terminate_engine",
    "benchmark_path_for_dp_rank",
    "trap",
)
_FOREGROUND_EXEC_LINE = (
    "exec python3 -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' \"${engine_command[@]}\""
)
_FPM_VARIABLE_RE = re.compile(r"\bFPM_[A-Z0-9_]+")
_EXPORT_ASSIGNMENT_RE = re.compile(r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_TOPOLOGIES = ("single-node", "multinode-tp", "multinode-dp")


def _params() -> dict:
    return {
        "ServiceConfig": {
            "model_path": "/workspace/model_cache/GLM-5",
            "served_model_path": "/workspace/model_cache/GLM-5",
            "served_model_name": "glm52-fpm",
            "include_frontend": False,
        },
        "K8sConfig": {
            "name_prefix": "glm52-fpm",
            "k8s_namespace": "default",
            "k8s_image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:test",
            "k8s_pvc_name": "model-cache-pvc",
            "k8s_pvc_mount_path": "/workspace/model_cache",
            "k8s_model_path_in_pvc": "GLM-5",
            # Normalized backward-compatible aliases consumed by the typed
            # vLLM K8s builder, which remains FPM's infrastructure source.
            "k8s_model_cache": "model-cache-pvc",
            "k8s_hf_home": "/workspace/model_cache/GLM-5",
            "extra_env": [
                {"name": "FPM_RUN_ID", "value": "glm52-fpm-a3-example"},
                {"name": "FPM_STAGE", "value": "aligned validation"},
                {"name": "DYN_FPM_BENCHMARK_OUTPUT_PATH", "value": "/results/benchmark.json"},
                {"name": "NCCL_DEBUG", "value": "INFO"},
            ],
        },
        "DynConfig": {"mode": "agg"},
        "WorkerConfig": {
            "agg_workers": 1,
            "agg_gpus_per_worker": 4,
            "prefill_workers": 0,
            "decode_workers": 0,
        },
        "NodeConfig": {"system_name": "b200_sxm", "num_gpus_per_node": 8},
        "SlaConfig": {"isl": 1024, "osl": 256},
        "ModelConfig": {"is_moe": True, "prefix": 0, "nextn": 0},
        "BenchConfig": {},
        "params": {
            "agg": {
                "tensor_parallel_size": 4,
                "pipeline_parallel_size": 1,
                "data_parallel_size": 1,
                "gpus_per_worker": 4,
                "max_batch_size": 64,
                "max_num_tokens": 4096,
                "max_seq_len": 8192,
                "tokens_per_block": 64,
                "trust_remote_code": True,
                "extra_cli_args": [
                    "--scheduler-cls",
                    "fpm.scheduler.InstrumentedScheduler",
                    "--benchmark-mode",
                    "agg",
                    "--compilation-config",
                    _COMPILATION_CONFIG,
                ],
            }
        },
    }


def _multinode_params(*, data_parallel: bool = False) -> dict:
    params = _params()
    params["WorkerConfig"]["agg_gpus_per_worker"] = 16
    params["params"]["agg"].update({"gpus_per_worker": 16, "tensor_parallel_size": 16})
    if data_parallel:
        params["params"]["agg"].update({"tensor_parallel_size": 1, "data_parallel_size": 16})
    return params


def _topology_params(topology: str) -> dict:
    if topology == "single-node":
        return _params()
    return _multinode_params(data_parallel=topology == "multinode-dp")


def _render(params: dict | None = None, backend: str = "vllm") -> dict[str, str]:
    return render_backend_templates(
        copy.deepcopy(params or _params()),
        backend,
        version=_BACKEND_VERSION,
        deployment_target="fpm",
    )


def _k8s_documents(artifacts: dict[str, str]) -> list[dict]:
    documents = list(yaml.safe_load_all(artifacts[FPM_MANIFEST_FILENAME]))
    assert all(isinstance(document, dict) for document in documents)
    return documents


def _k8s_document(artifacts: dict[str, str], kind: str) -> dict:
    matches = [document for document in _k8s_documents(artifacts) if document.get("kind") == kind]
    assert len(matches) == 1
    return matches[0]


def _set_benchmark_mode(params: dict, mode: str) -> None:
    args = params["params"]["agg"]["extra_cli_args"]
    args[args.index("--benchmark-mode") + 1] = mode


def _pod(artifacts: dict[str, str]) -> dict:
    documents = _k8s_documents(artifacts)
    assert len(documents) == 1
    assert documents[0]["kind"] == "Pod"
    return documents[0]


def _main_container(pod: dict) -> dict:
    containers = pod["spec"]["containers"]
    assert len(containers) == 1
    return containers[0]


def _export_assignments(script: str) -> list[str]:
    assignments = []
    for line in script.splitlines():
        try:
            tokens = shlex.split(line, comments=True)
        except ValueError:
            continue
        if not tokens or tokens[0] != "export":
            continue
        assignments.extend(tokens[1:])
    return assignments


def _export_value(script: str, name: str) -> str:
    prefix = f"{name}="
    values = [assignment[len(prefix) :] for assignment in _export_assignments(script) if assignment.startswith(prefix)]
    if not values:
        raise AssertionError(f"missing export for {name}")
    if len(values) > 1:
        raise AssertionError(f"duplicate exports for {name}: {values!r}")
    return values[0]


_NVCC_STUB_DIR: Path | None = None


def _nvcc_stub_dir() -> Path:
    """Execution tests run the rendered run.sh on hosts without a CUDA
    toolchain, which would trip the multinode nvcc fail-fast guard. A stub
    nvcc on PATH exercises the guard's healthy path instead; the guard's
    failure path is covered by the rendering test."""
    global _NVCC_STUB_DIR
    if _NVCC_STUB_DIR is None:
        stub_dir = Path(tempfile.mkdtemp(prefix="fpm-nvcc-stub-"))
        stub = stub_dir / "nvcc"
        stub.write_text("#!/usr/bin/env bash\nexit 0\n")
        stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        _NVCC_STUB_DIR = stub_dir
    return _NVCC_STUB_DIR


def _clean_env(**overrides: str) -> dict[str, str]:
    """Subprocess environment without host FPM/orchestrator contamination."""
    env = {name: value for name, value in os.environ.items() if not name.startswith(("FPM_", "LWS_", "GROVE_"))}
    env["PATH"] = f"{_nvcc_stub_dir()}{os.pathsep}{env.get('PATH', '')}"
    env.update(overrides)
    return env


def _write_runtime(tmp_path: Path, artifacts: dict[str, str]) -> Path:
    """Stage run.sh next to the fpm_env.sh it sources; return the run.sh path."""
    (tmp_path / FPM_ENV_FILENAME).write_text(artifacts[FPM_ENV_FILENAME])
    script_path = tmp_path / FPM_RUN_SCRIPT_FILENAME
    script_path.write_text(artifacts[FPM_RUN_SCRIPT_FILENAME])
    return script_path


def _write_fake_engine(tmp_path: Path, body: str) -> Path:
    """Create an importable fake ``dynamo.vllm`` package; return the PYTHONPATH root."""
    fake_package = tmp_path / "fake-package" / "dynamo" / "vllm"
    fake_package.mkdir(parents=True)
    (fake_package.parent / "__init__.py").write_text("")
    (fake_package / "__init__.py").write_text("")
    (fake_package / "__main__.py").write_text(body)
    return tmp_path / "fake-package"


def _source_env_script(
    tmp_path: Path,
    env_script: str,
    case_env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    env_path = tmp_path / FPM_ENV_FILENAME
    env_path.write_text(env_script)
    probe = 'source "$1" && printf "%s %s" "$FPM_NODE_RANK" "$FPM_MASTER_ADDR"'
    return subprocess.run(
        ["bash", "-c", probe, "bash", str(env_path)],
        text=True,
        capture_output=True,
        env=_clean_env(**case_env),
        timeout=5,
        check=False,
    )


def test_fpm_render_returns_exactly_the_contract_artifact_trio():
    artifacts = _render()

    assert set(artifacts) == _ARTIFACT_TRIO


def test_fpm_env_script_exports_exactly_the_contract_variables():
    env_script = _render()[FPM_ENV_FILENAME]

    exported = {assignment.split("=", 1)[0] for assignment in _export_assignments(env_script)}

    assert exported == set(FPM_ENV_EXPORTED_VARS)
    assert _export_value(env_script, "FPM_NODE_COUNT") == "1"
    assert _export_value(env_script, "FPM_DATA_PARALLEL_SIZE") == "1"
    assert _export_value(env_script, "FPM_LOCAL_DATA_PARALLEL_SIZE") == "1"
    assert _export_value(env_script, "FPM_BENCHMARK_MODE") == "agg"
    assert _export_value(env_script, "FPM_BENCHMARK_OUTPUT_PATH") == "/results/benchmark.json"
    assert _export_value(env_script, "FPM_WAIT_TIMEOUT_SECONDS") == "7800"
    assert _export_value(env_script, "FPM_RESULT_SCHEMA_VERSION") == str(FPM_NATIVE_BENCHMARK_RESULT_SCHEMA_VERSION)


def test_fpm_forwards_barrier_timeout_override_through_fpm_env_only():
    """FPM_COMPLETION_BARRIER_TIMEOUT_SECONDS is a collection-runtime tunable:
    it must land in fpm_env.sh (which the runtime sources) and never in run.sh
    (which only the engine's environment sees)."""

    params = _params()
    params["K8sConfig"]["extra_env"].append({"name": FPM_BARRIER_TIMEOUT_ENV, "value": "45"})

    artifacts = _render(params)
    env_script = artifacts[FPM_ENV_FILENAME]

    assert f"export {FPM_BARRIER_TIMEOUT_ENV}=45" in env_script.splitlines()
    exported = {assignment.split("=", 1)[0] for assignment in _export_assignments(env_script)}
    assert exported == set(FPM_ENV_EXPORTED_VARS) | {FPM_BARRIER_TIMEOUT_ENV}
    assert FPM_BARRIER_TIMEOUT_ENV not in artifacts[FPM_RUN_SCRIPT_FILENAME]


def test_fpm_single_node_data_parallel_exports_full_local_dp_size():
    params = _params()
    params["params"]["agg"].update({"tensor_parallel_size": 1, "data_parallel_size": 4})

    env_script = _render(params)[FPM_ENV_FILENAME]

    assert _export_value(env_script, "FPM_NODE_COUNT") == "1"
    assert _export_value(env_script, "FPM_DATA_PARALLEL_SIZE") == "4"
    assert _export_value(env_script, "FPM_LOCAL_DATA_PARALLEL_SIZE") == "4"


@pytest.mark.parametrize("topology", _TOPOLOGIES)
def test_fpm_run_script_is_collection_free_and_ends_with_foreground_exec(topology):
    script = _render(_topology_params(topology))[FPM_RUN_SCRIPT_FILENAME]

    assert 'source "$(dirname "${BASH_SOURCE[0]}")/fpm_env.sh"' in script
    # Rank and leader discovery moved to fpm_env.sh in full.
    assert "LWS_WORKER_INDEX" not in script
    assert "GROVE_PCLQ" not in script
    for token in _COLLECTION_VOCABULARY:
        assert token not in script, f"collection-side token {token!r} leaked into run.sh"
    last_line = [line for line in script.splitlines() if line.strip()][-1]
    assert last_line == _FOREGROUND_EXEC_LINE


@pytest.mark.unit
def test_fpm_run_script_guards_multinode_cuda_toolchain():
    """Multinode transport profiles rewrite PATH; a profile that drops
    /usr/local/cuda/bin starves deep_gemm's runtime nvcc JIT and only crashes
    minutes later. run.sh must fail fast with an nvcc check before launching
    the engine. The guard is rendered into every script but gated on
    FPM_NODE_COUNT at runtime, the same shape as the other multinode blocks."""

    script = _render(_topology_params("multinode-tp"))[FPM_RUN_SCRIPT_FILENAME]
    assert "FPM_NODE_COUNT > 1 )) && ! command -v nvcc" in script
    assert script.index("command -v nvcc") < script.index("exec python3 -c")


def _consumed_fpm_variables(script: str) -> set[str]:
    """FPM_* names the script reads.

    Names assigned by an ``export NAME=value`` passthrough line (the
    collector-supplied ``extra_env``) are not reads, but their values are
    still scanned (e.g. the DYN_FPM_WORKER_ID default reads FPM_RUN_ID).
    """
    consumed: set[str] = set()
    for line in script.splitlines():
        match = _EXPORT_ASSIGNMENT_RE.match(line)
        scan_target = match.group(2) if match else line
        consumed.update(_FPM_VARIABLE_RE.findall(scan_target))
    return consumed


@pytest.mark.parametrize("topology", _TOPOLOGIES)
def test_fpm_run_script_consumes_only_contract_environment(topology):
    script = _render(_topology_params(topology))[FPM_RUN_SCRIPT_FILENAME]

    consumed = _consumed_fpm_variables(script)

    assert consumed
    assert consumed <= set(FPM_ENV_EXPORTED_VARS) | {FPM_RUN_ID_ENV}


@pytest.mark.parametrize("topology", _TOPOLOGIES)
def test_fpm_scripts_pass_bash_syntax_check(topology):
    artifacts = _render(_topology_params(topology))

    for name in (FPM_ENV_FILENAME, FPM_RUN_SCRIPT_FILENAME):
        syntax = subprocess.run(
            ["bash", "-n"],
            input=artifacts[name],
            text=True,
            capture_output=True,
            check=False,
        )
        assert syntax.returncode == 0, f"{name}: {syntax.stderr}"


def _pod_template_labels(artifacts: dict[str, str], kind: str) -> list[dict]:
    document = _k8s_document(artifacts, kind)
    if kind == "Pod":
        return [document["metadata"]["labels"]]
    if kind == "LeaderWorkerSet":
        group = document["spec"]["leaderWorkerTemplate"]
        return [group[name]["metadata"]["labels"] for name in ("leaderTemplate", "workerTemplate")]
    assert kind == "PodCliqueSet"
    return [clique["labels"] for clique in document["spec"]["template"]["cliques"]]


@pytest.mark.parametrize("kind", ["Pod", "LeaderWorkerSet", "PodCliqueSet"])
def test_fpm_cell_label_reaches_every_pod_template(kind):
    params = _params()
    params["K8sConfig"]["fpm_resource_labels"] = {FPM_CELL_LABEL: "glm52-fpm-cell-0"}
    if kind != "Pod":
        params["NodeConfig"].update({"system_name": "gb200", "num_gpus_per_node": 4})
        params["WorkerConfig"]["agg_gpus_per_worker"] = 8
        params["params"]["agg"].update({"gpus_per_worker": 8, "tensor_parallel_size": 8})
    if kind == "PodCliqueSet":
        params["K8sConfig"]["fpm_orchestrator"] = "grove"

    label_sets = _pod_template_labels(_render(params), kind)

    assert label_sets
    for labels in label_sets:
        assert labels[FPM_CELL_LABEL] == "glm52-fpm-cell-0"


@pytest.mark.parametrize(
    ("case_env", "expected_output"),
    [
        pytest.param(
            {"LWS_WORKER_INDEX": "1", "LWS_LEADER_ADDRESS": "leader.example"},
            "1 leader.example",
            id="lws",
        ),
        pytest.param(
            {
                "GROVE_PCLQ_POD_INDEX": "1",
                "GROVE_PCLQ_NAME": "glm52-fpm-agg-0-worker",
                "GROVE_HEADLESS_SERVICE": "glm52-fpm-agg-0.default.svc.cluster.local",
            },
            "1 glm52-fpm-agg-0-worker-0.glm52-fpm-agg-0.default.svc.cluster.local",
            id="grove",
        ),
        pytest.param(
            {
                "FPM_NODE_RANK": "0",
                "FPM_MASTER_ADDR": "10.0.0.9",
                "LWS_WORKER_INDEX": "1",
                "LWS_LEADER_ADDRESS": "ignored.example",
            },
            "0 10.0.0.9",
            id="explicit-fpm-presets-win",
        ),
    ],
)
def test_fpm_env_script_multinode_discovery_cascade(tmp_path, case_env, expected_output):
    env_script = _render(_multinode_params())[FPM_ENV_FILENAME]

    completed = _source_env_script(tmp_path, env_script, case_env)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == expected_output


def test_fpm_env_script_multinode_fails_closed_without_discovery(tmp_path):
    env_script = _render(_multinode_params())[FPM_ENV_FILENAME]

    completed = _source_env_script(tmp_path, env_script, {})

    assert completed.returncode == 2
    assert "requires rank and leader discovery" in completed.stderr


@pytest.mark.parametrize(
    "case_env",
    [
        pytest.param({"LWS_WORKER_INDEX": "7", "LWS_LEADER_ADDRESS": "leader.example"}, id="out-of-range"),
        pytest.param({"FPM_NODE_RANK": "one", "FPM_MASTER_ADDR": "leader.example"}, id="non-numeric"),
    ],
)
def test_fpm_env_script_rejects_invalid_node_rank(tmp_path, case_env):
    env_script = _render(_multinode_params())[FPM_ENV_FILENAME]

    completed = _source_env_script(tmp_path, env_script, case_env)

    assert completed.returncode == 2
    assert "Invalid FPM node rank" in completed.stderr


def test_fpm_env_script_single_node_defaults_to_local_leader(tmp_path):
    env_script = _render()[FPM_ENV_FILENAME]

    completed = _source_env_script(tmp_path, env_script, {})

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "0 127.0.0.1"


@pytest.mark.parametrize(
    "partial_env",
    [
        pytest.param({"LWS_WORKER_INDEX": "0"}, id="rank-only-lws"),
        pytest.param({"FPM_MASTER_ADDR": "10.0.0.1"}, id="leader-only-explicit"),
        pytest.param({"GROVE_PCLQ_NAME": "glm52-fpm-agg-0-worker"}, id="grove-name-without-service"),
    ],
)
def test_fpm_env_script_single_node_fails_closed_on_partial_discovery(tmp_path, partial_env):
    """The rank-0/127.0.0.1 defaults apply only to a fully undiscovered
    environment; any partial orchestrator answer is a misconfiguration."""

    env_script = _render()[FPM_ENV_FILENAME]

    completed = _source_env_script(tmp_path, env_script, partial_env)

    assert completed.returncode == 2
    assert "FPM runtime requires complete rank and leader discovery" in completed.stderr


def test_fpm_env_script_single_node_accepts_complete_explicit_discovery(tmp_path):
    env_script = _render()[FPM_ENV_FILENAME]

    completed = _source_env_script(
        tmp_path,
        env_script,
        {"FPM_NODE_RANK": "0", "FPM_MASTER_ADDR": "10.0.0.1"},
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "0 10.0.0.1"


def test_fpm_env_script_multinode_fails_closed_with_rank_but_no_leader(tmp_path):
    """A rank without a leader source must not pass the multinode gate; this
    pins the OR in its incomplete-discovery check."""

    env_script = _render(_multinode_params())[FPM_ENV_FILENAME]

    completed = _source_env_script(tmp_path, env_script, {"LWS_WORKER_INDEX": "1"})

    assert completed.returncode == 2
    assert "Multinode FPM requires rank and leader discovery" in completed.stderr


def test_fpm_pinned_vllm_024_uses_floor_template_and_preserves_fpm_overlay():
    artifacts = render_backend_templates(
        copy.deepcopy(_params()),
        "vllm",
        version="0.24.0",
        deployment_target="fpm",
    )

    assert yaml.safe_load(artifacts[FPM_MANIFEST_FILENAME])["kind"] == "Pod"
    assert "--tensor-parallel-size 4" in artifacts[FPM_RUN_SCRIPT_FILENAME]
    assert "--scheduler-cls fpm.scheduler.InstrumentedScheduler" in artifacts[FPM_RUN_SCRIPT_FILENAME]
    assert "--dump-config-to /results/resolved-config-node0.json" in artifacts[FPM_RUN_SCRIPT_FILENAME]


def test_fpm_resource_pod_is_keepalive_only_and_preserves_resources():
    artifacts = _render()
    pod = _pod(artifacts)
    container = _main_container(pod)

    assert pod["apiVersion"] == "v1"
    assert pod["kind"] == "Pod"
    keepalive = " ".join([*container.get("command", []), *container.get("args", [])])
    assert "sleep" in keepalive
    assert "infinity" in keepalive

    assert int(container["resources"]["limits"]["nvidia.com/gpu"]) == 4
    assert "claims" not in container["resources"]
    assert "resourceClaims" not in pod["spec"]
    assert pod["spec"]["nodeSelector"]["nvidia.com/gpu.product"] == "NVIDIA-B200"
    assert not container.get("env")
    assert not container.get("envFrom")
    assert "dynamo.vllm" not in keepalive
    assert "--scheduler-cls" not in keepalive
    assert "--benchmark-mode" not in keepalive

    volumes = {volume["name"]: volume for volume in pod["spec"]["volumes"]}
    mounts = {mount["mountPath"]: mount["name"] for mount in container["volumeMounts"]}

    model_volume = volumes[mounts["/workspace/model_cache"]]
    assert model_volume["persistentVolumeClaim"]["claimName"] == "model-cache-pvc"
    assert volumes[mounts["/results"]]["emptyDir"] == {}
    assert volumes[mounts["/dev/shm"]]["emptyDir"]["medium"] == "Memory"
    assert volumes[mounts["/dev/shm"]]["emptyDir"]["sizeLimit"] == "64Gi"


def test_fpm_resource_overlays_preserve_requests_mount_path_shm_and_labels():
    params = _params()
    params["K8sConfig"].update(
        {
            "k8s_pvc_mount_path": "/model-cache",
            "fpm_shared_memory_size": "200Gi",
            "fpm_resource_labels": {
                "fpm.nvidia.com/run-id": "glm52-fpm-a3-example",
                "fpm.nvidia.com/stage": "probe",
            },
            "worker_extra_pod_spec": {
                "mainContainer": {
                    "resources": {
                        "requests": {
                            "memory": "448Gi",
                            "ephemeral-storage": "30Gi",
                        }
                    }
                }
            },
        }
    )

    pod = _pod(_render(params))
    container = _main_container(pod)
    requests = container["resources"]["requests"]
    assert requests["memory"] == "448Gi"
    assert requests["ephemeral-storage"] == "30Gi"
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "4"
    assert {mount["mountPath"] for mount in container["volumeMounts"]} >= {
        "/model-cache",
        "/results",
        "/dev/shm",
    }
    volumes = {volume["name"]: volume for volume in pod["spec"]["volumes"]}
    assert volumes["dshm"]["emptyDir"] == {"medium": "Memory", "sizeLimit": "200Gi"}
    assert pod["metadata"]["labels"]["fpm.nvidia.com/run-id"] == "glm52-fpm-a3-example"
    assert pod["metadata"]["labels"]["fpm.nvidia.com/stage"] == "probe"


def test_fpm_resource_overlay_cannot_change_resolved_gpu_count():
    params = _params()
    params["K8sConfig"]["worker_extra_pod_spec"] = {"mainContainer": {"resources": {"limits": {"nvidia.com/gpu": "8"}}}}

    with pytest.raises(ValueError, match="per-node GPU count"):
        _render(params)


def test_fpm_run_script_contains_resolved_args_passthrough_and_exports():
    script = _render()[FPM_RUN_SCRIPT_FILENAME]

    # Service-level fields plus the normal versioned vLLM template/rule output.
    assert "python3 -m dynamo.vllm" in script
    assert "--model /workspace/model_cache/GLM-5" in script
    assert "--served-model-name glm52-fpm" in script
    assert "--tensor-parallel-size 4" in script
    assert "--block-size 64" in script

    # FPM-only argv is appended without losing token boundaries. The JSON has
    # spaces and nested values specifically to catch unsafe string joining.
    assert "--scheduler-cls fpm.scheduler.InstrumentedScheduler" in script
    assert "--benchmark-mode agg" in script
    assert f"--compilation-config {shlex.quote(_COMPILATION_CONFIG)}" in script
    assert script.count(_COMPILATION_CONFIG) == 1

    assert _export_value(script, "FPM_RUN_ID") == "glm52-fpm-a3-example"
    assert _export_value(script, "FPM_STAGE") == "aligned validation"
    assert _export_value(script, "DYN_FPM_BENCHMARK_OUTPUT_PATH") == "/results/benchmark.json"
    assert _export_value(script, "NCCL_DEBUG") == "INFO"
    assert _export_value(script, "NCCL_CUMEM_ENABLE") == "1"
    assert "ulimit -n 1048576" in script


def test_fpm_preserves_duplicate_environment_export_order():
    params = _params()
    params["K8sConfig"]["extra_env"].extend(
        [
            {"name": "NCCL_DEBUG", "value": "WARN"},
            {"name": "NCCL_DEBUG", "value": "TRACE"},
        ]
    )

    exports = [
        line for line in _render(params)[FPM_RUN_SCRIPT_FILENAME].splitlines() if line.startswith("export NCCL_DEBUG=")
    ]

    assert exports == ["export NCCL_DEBUG=INFO", "export NCCL_DEBUG=WARN", "export NCCL_DEBUG=TRACE"]


def test_fpm_keeps_cli_and_environment_output_paths_aligned():
    params = _params()
    params["K8sConfig"]["extra_env"] = [
        entry for entry in params["K8sConfig"]["extra_env"] if entry["name"] != "DYN_FPM_BENCHMARK_OUTPUT_PATH"
    ]
    params["params"]["agg"]["extra_cli_args"].extend(["--benchmark-output-path", "/results/benchmark_custom.json"])

    artifacts = _render(params)

    engine_output = _export_value(artifacts[FPM_RUN_SCRIPT_FILENAME], "DYN_FPM_BENCHMARK_OUTPUT_PATH")
    assert engine_output == "/results/benchmark_custom.json"
    assert _export_value(artifacts[FPM_ENV_FILENAME], "FPM_BENCHMARK_OUTPUT_PATH") == "/results/benchmark_custom.json"


def test_fpm_rejects_conflicting_output_paths():
    params = _params()
    params["params"]["agg"]["extra_cli_args"].extend(["--benchmark-output-path", "/results/different.json"])

    with pytest.raises(ValueError, match="same path"):
        _render(params)


@pytest.mark.parametrize(
    "empty_cli_args",
    [
        ["--benchmark-output-path", ""],
        ["--benchmark-output-path="],
    ],
)
def test_fpm_rejects_explicit_empty_cli_output_path(empty_cli_args):
    params = _params()
    params["K8sConfig"]["extra_env"] = [
        entry for entry in params["K8sConfig"]["extra_env"] if entry["name"] != "DYN_FPM_BENCHMARK_OUTPUT_PATH"
    ]
    params["params"]["agg"]["extra_cli_args"].extend(empty_cli_args)

    with pytest.raises(ValueError, match="must not be empty"):
        _render(params)


def test_fpm_rejects_explicit_empty_environment_output_path():
    params = _params()
    for entry in params["K8sConfig"]["extra_env"]:
        if entry["name"] == "DYN_FPM_BENCHMARK_OUTPUT_PATH":
            entry["value"] = ""

    with pytest.raises(ValueError, match="must not be empty"):
        _render(params)


def test_fpm_api_writes_exact_filenames_and_executable_scripts(tmp_path):
    artifacts = generate_backend_artifacts(
        copy.deepcopy(_params()),
        "vllm",
        output_dir=str(tmp_path),
        backend_version=_BACKEND_VERSION,
        deployment_target="fpm",
    )

    assert set(artifacts) == _ARTIFACT_TRIO
    assert {path.name for path in tmp_path.iterdir()} == _ARTIFACT_TRIO
    assert not (tmp_path / "run_x.sh").exists()
    for script_name in (FPM_RUN_SCRIPT_FILENAME, FPM_ENV_FILENAME):
        assert (tmp_path / script_name).stat().st_mode & stat.S_IXUSR
    assert yaml.safe_load((tmp_path / FPM_MANIFEST_FILENAME).read_text())["kind"] == "Pod"


def test_fpm_run_script_execs_engine_in_foreground_with_setsid(tmp_path):
    report_path = tmp_path / "engine-report.json"
    pythonpath = _write_fake_engine(
        tmp_path,
        """\
import json
import os
import pathlib
import sys

pathlib.Path(os.environ["FAKE_REPORT_PATH"]).write_text(json.dumps({
    "argv": sys.argv[1:],
    "worker_id": os.environ.get("DYN_FPM_WORKER_ID"),
    "group_leader": os.getpgrp() == os.getpid(),
}))
raise SystemExit(23)
""",
    )
    script_path = _write_runtime(tmp_path, _render())

    completed = subprocess.run(
        ["bash", str(script_path)],
        text=True,
        capture_output=True,
        env=_clean_env(PYTHONPATH=str(pythonpath), FAKE_REPORT_PATH=str(report_path)),
        timeout=8,
        check=False,
    )

    # run.sh replaces itself with the engine: its exit code is the engine's,
    # and setsid made the engine its own process-group leader.
    assert completed.returncode == 23
    report = json.loads(report_path.read_text())
    assert report["worker_id"] == "glm52-fpm-a3-example-node0"
    assert report["group_leader"] is True
    assert "--benchmark-mode" in report["argv"]


def test_default_and_explicit_normal_targets_remain_identical():
    params = _params()
    params["K8sConfig"].pop("extra_env")
    params["params"]["agg"].pop("extra_cli_args")

    default = render_backend_templates(copy.deepcopy(params), "vllm", version=_BACKEND_VERSION)
    explicit = render_backend_templates(
        copy.deepcopy(params),
        "vllm",
        version=_BACKEND_VERSION,
        deployment_target="dynamo-j2",
    )

    assert explicit == default
    assert "k8s_deploy.yaml" in explicit
    assert "run_0.sh" in explicit
    assert "run.sh" not in explicit
    assert FPM_ENV_FILENAME not in explicit


def test_legacy_yaml_normalization_preserves_extra_cli_args():
    raw = {
        "ServiceConfig": {"model_path": "/models/glm52", "served_model_name": "glm52"},
        "DynConfig": {"mode": "agg"},
        "Workers": {
            "agg": {
                "tensor_parallel_size": 4,
                "extra_cli_args": ["--scheduler-cls", "fpm.scheduler.InstrumentedScheduler"],
            }
        },
    }

    normalized = generate_config_from_input_dict(raw, backend="vllm")

    assert normalized["params"]["agg"]["extra_cli_args"] == raw["Workers"]["agg"]["extra_cli_args"]


def test_render_artifacts_cli_accepts_fpm_target(tmp_path, capsys):
    config = {
        "ServiceConfig": {
            "model_path": "/workspace/model_cache/GLM-5",
            "served_model_name": "glm52-fpm",
        },
        "K8sConfig": {
            "name_prefix": "glm52-fpm",
            "k8s_namespace": "default",
            "k8s_image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:test",
            "extra_env": [
                {"name": "DYN_FPM_BENCHMARK_OUTPUT_PATH", "value": "/results/benchmark.json"},
            ],
        },
        "DynConfig": {"mode": "agg"},
        "WorkerConfig": {"agg_workers": 1},
        "NodeConfig": {"num_gpus_per_node": 8},
        "Workers": {
            "agg": {
                "tensor_parallel_size": 4,
                "pipeline_parallel_size": 1,
                "gpus_per_worker": 4,
                "max_batch_size": 64,
                "max_num_tokens": 4096,
                "max_seq_len": 8192,
                "tokens_per_block": 64,
                "extra_cli_args": ["--benchmark-mode", "agg"],
            }
        },
    }
    config_path = tmp_path / "fpm-request.yaml"
    output_dir = tmp_path / "artifacts"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    generator_main(
        [
            "render-artifacts",
            "--backend",
            "vllm",
            "--version",
            _BACKEND_VERSION,
            "--deployment-target",
            "fpm",
            "--config",
            str(config_path),
            "--output",
            str(output_dir),
        ]
    )
    capsys.readouterr()

    assert {path.name for path in output_dir.iterdir()} == _ARTIFACT_TRIO


def _disagg_params() -> dict:
    params = _params()
    agg = params["params"].pop("agg")
    params["params"]["prefill"] = copy.deepcopy(agg)
    params["params"]["decode"] = copy.deepcopy(agg)
    params["DynConfig"]["mode"] = "disagg"
    params["WorkerConfig"].update(
        {
            "agg_workers": 0,
            "prefill_workers": 1,
            "decode_workers": 1,
            "prefill_gpus_per_worker": 4,
            "decode_gpus_per_worker": 4,
        }
    )
    return params


@pytest.mark.parametrize(
    ("backend", "params"),
    [
        pytest.param("sglang", _params(), id="non-vllm"),
        pytest.param("vllm", _disagg_params(), id="disaggregated"),
    ],
)
def test_fpm_rejects_unsupported_backend_or_mode(backend, params):
    with pytest.raises(ValueError):
        _render(params, backend=backend)


def test_fpm_rejects_multiple_workers():
    params = _params()
    params["WorkerConfig"]["agg_workers"] = 2

    with pytest.raises(ValueError):
        _render(params)


@pytest.mark.parametrize("orchestrator", [None, "lws"], ids=["default", "explicit-lws"])
def test_fpm_multinode_worker_emits_keepalive_leaderworkerset_and_rank_aware_scripts(orchestrator):
    params = _params()
    if orchestrator is not None:
        params["K8sConfig"]["fpm_orchestrator"] = orchestrator
    params["NodeConfig"].update({"system_name": "gb200", "num_gpus_per_node": 4})
    params["WorkerConfig"]["agg_gpus_per_worker"] = 8
    params["params"]["agg"]["gpus_per_worker"] = 8
    params["params"]["agg"]["tensor_parallel_size"] = 8

    artifacts = _render(params)
    documents = _k8s_documents(artifacts)
    assert [document["kind"] for document in documents] == ["ComputeDomain", "LeaderWorkerSet"]
    compute_domain = _k8s_document(artifacts, "ComputeDomain")
    workload = _k8s_document(artifacts, "LeaderWorkerSet")

    assert compute_domain == {
        "apiVersion": "resource.nvidia.com/v1beta1",
        "kind": "ComputeDomain",
        "metadata": {
            "name": "glm52-fpm-agg-compute-domain",
            "namespace": "default",
        },
        "spec": {
            "channel": {
                "resourceClaimTemplate": {
                    "name": "glm52-fpm-agg-compute-domain-channel",
                }
            },
            "numNodes": 0,
        },
    }
    assert workload["apiVersion"] == "leaderworkerset.x-k8s.io/v1"
    assert workload["kind"] == "LeaderWorkerSet"
    group = workload["spec"]["leaderWorkerTemplate"]
    assert group["size"] == 2
    for template_name in ("leaderTemplate", "workerTemplate"):
        pod_spec = group[template_name]["spec"]
        assert pod_spec["resourceClaims"] == [
            {
                "name": "compute-domain-channel",
                "resourceClaimTemplateName": "glm52-fpm-agg-compute-domain-channel",
            }
        ]
        container = pod_spec["containers"][0]
        assert container["resources"]["limits"]["nvidia.com/gpu"] == "4"
        assert container["resources"]["claims"] == [{"name": "compute-domain-channel"}]
        assert container["command"] == ["/bin/bash", "-lc"]
        assert container["args"] == ["exec sleep infinity"]

    # fpm_env.sh owns rank/leader discovery; run.sh only consumes its exports.
    env_script = artifacts[FPM_ENV_FILENAME]
    assert _export_value(env_script, "FPM_NODE_COUNT") == "2"
    assert 'fpm_node_rank="${FPM_NODE_RANK:-${LWS_WORKER_INDEX:-${GROVE_PCLQ_POD_INDEX:-}}}"' in env_script
    assert 'fpm_master_addr="${FPM_MASTER_ADDR:-${LWS_LEADER_ADDRESS:-}}"' in env_script
    script = artifacts[FPM_RUN_SCRIPT_FILENAME]
    assert '--nnodes "$FPM_NODE_COUNT" --node-rank "$FPM_NODE_RANK"' in script
    assert '--master-addr "$FPM_MASTER_ADDR" --master-port 29500' in script
    # Followers get the engine's --headless flag and nothing more; exit
    # classification against the leader belongs to the collector runtime.
    assert "engine_command+=(--headless)" in script


def test_fpm_multinode_gb200_grove_emits_compute_domain_and_keepalive_podcliqueset():
    params = _params()
    params["K8sConfig"]["fpm_orchestrator"] = "grove"
    params["NodeConfig"].update({"system_name": "gb200", "num_gpus_per_node": 4})
    params["WorkerConfig"]["agg_gpus_per_worker"] = 8
    params["params"]["agg"].update({"gpus_per_worker": 8, "tensor_parallel_size": 8})

    artifacts = _render(params)
    documents = _k8s_documents(artifacts)

    assert [document["kind"] for document in documents] == ["ComputeDomain", "PodCliqueSet"]
    compute_domain = _k8s_document(artifacts, "ComputeDomain")
    workload = _k8s_document(artifacts, "PodCliqueSet")
    assert compute_domain == {
        "apiVersion": "resource.nvidia.com/v1beta1",
        "kind": "ComputeDomain",
        "metadata": {
            "name": "glm52-fpm-agg-compute-domain",
            "namespace": "default",
        },
        "spec": {
            "channel": {
                "resourceClaimTemplate": {
                    "name": "glm52-fpm-agg-compute-domain-channel",
                }
            },
            "numNodes": 0,
        },
    }

    assert workload["apiVersion"] == "grove.io/v1alpha1"
    assert workload["metadata"]["name"] == "glm52-fpm-agg"
    assert workload["metadata"]["namespace"] == "default"
    assert workload["spec"]["replicas"] == 1
    template = workload["spec"]["template"]
    assert template["cliqueStartupType"] == "CliqueStartupTypeAnyOrder"
    assert template["headlessServiceConfig"] == {"publishNotReadyAddresses": True}
    assert len(template["cliques"]) == 1

    clique = template["cliques"][0]
    assert clique["name"] == "worker"
    assert clique["labels"]["app.kubernetes.io/name"] == "glm52-fpm-agg"
    assert clique["labels"]["app.kubernetes.io/component"] == "fpm-resource"
    assert "kai.scheduler/queue" not in clique["labels"]
    assert clique["spec"]["roleName"] == "worker"
    assert clique["spec"]["replicas"] == 2
    assert clique["spec"]["minAvailable"] == 2

    pod_spec = clique["spec"]["podSpec"]
    assert "schedulerName" not in pod_spec
    assert pod_spec["nodeSelector"]["nvidia.com/gpu.product"] == "NVIDIA-GB200"
    assert pod_spec["resourceClaims"] == [
        {
            "name": "compute-domain-channel",
            "resourceClaimTemplateName": "glm52-fpm-agg-compute-domain-channel",
        }
    ]
    assert len(pod_spec["containers"]) == 1
    container = pod_spec["containers"][0]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "4"
    assert container["resources"]["claims"] == [{"name": "compute-domain-channel"}]
    assert container["command"] == ["/bin/bash", "-lc"]
    assert container["args"] == ["exec sleep infinity"]


@pytest.mark.parametrize(
    ("scheduler_name", "queue_label", "queue_name"),
    [
        ("kai-scheduler", "kai.scheduler/queue", "dynamo"),
        ("custom-scheduler", "scheduler.example.com/queue", "fpm"),
    ],
)
def test_fpm_grove_preserves_explicit_scheduler_and_queue_label(
    scheduler_name,
    queue_label,
    queue_name,
):
    params = _params()
    params["K8sConfig"].update(
        {
            "fpm_orchestrator": "grove",
            "fpm_resource_labels": {queue_label: queue_name},
            "worker_extra_pod_spec": {"schedulerName": scheduler_name},
        }
    )
    params["NodeConfig"].update({"system_name": "gb200", "num_gpus_per_node": 4})
    params["WorkerConfig"]["agg_gpus_per_worker"] = 8
    params["params"]["agg"].update({"gpus_per_worker": 8, "tensor_parallel_size": 8})

    workload = _k8s_document(_render(params), "PodCliqueSet")
    clique = workload["spec"]["template"]["cliques"][0]

    assert workload["metadata"]["labels"][queue_label] == queue_name
    assert clique["labels"][queue_label] == queue_name
    assert clique["spec"]["podSpec"]["schedulerName"] == scheduler_name


def test_fpm_rejects_unknown_orchestrator():
    params = _params()
    params["K8sConfig"]["fpm_orchestrator"] = "deployment"

    with pytest.raises(ValueError, match="fpm_orchestrator"):
        _render(params)


def test_fpm_b200_multinode_does_not_require_compute_domain():
    params = _params()
    params["WorkerConfig"]["agg_gpus_per_worker"] = 16
    params["params"]["agg"].update({"gpus_per_worker": 16, "tensor_parallel_size": 16})

    artifacts = _render(params)
    documents = _k8s_documents(artifacts)

    assert [document["kind"] for document in documents] == ["LeaderWorkerSet"]
    group = documents[0]["spec"]["leaderWorkerTemplate"]
    for template_name in ("leaderTemplate", "workerTemplate"):
        pod_spec = group[template_name]["spec"]
        assert "resourceClaims" not in pod_spec
        assert "claims" not in pod_spec["containers"][0]["resources"]


def test_fpm_multinode_efa_resource_matches_per_node_gpu_count():
    params = _params()
    params["K8sConfig"]["transport"] = "efa"
    params["K8sConfig"]["worker_extra_pod_spec"] = {
        "mainContainer": {
            "resources": {
                "limits": {"example.com/unrelated": "1"},
            }
        }
    }
    params["WorkerConfig"]["agg_gpus_per_worker"] = 16
    params["params"]["agg"].update({"gpus_per_worker": 16, "tensor_parallel_size": 16})

    workload = _k8s_document(_render(params), "LeaderWorkerSet")
    group = workload["spec"]["leaderWorkerTemplate"]

    for template_name in ("leaderTemplate", "workerTemplate"):
        limits = group[template_name]["spec"]["containers"][0]["resources"]["limits"]
        assert limits["nvidia.com/gpu"] == "8"
        assert limits["vpc.amazonaws.com/efa"] == "8"
        assert limits["example.com/unrelated"] == "1"


def test_fpm_efa_resource_overlay_cannot_change_resolved_per_node_count():
    params = _params()
    params["K8sConfig"]["transport"] = "efa"
    params["K8sConfig"]["worker_extra_pod_spec"] = {
        "mainContainer": {
            "resources": {
                "limits": {"vpc.amazonaws.com/efa": "16"},
            }
        }
    }
    params["WorkerConfig"]["agg_gpus_per_worker"] = 16
    params["params"]["agg"].update({"gpus_per_worker": 16, "tensor_parallel_size": 16})

    with pytest.raises(ValueError, match="per-node EFA count"):
        _render(params)


def test_fpm_multinode_dump_config_override_requires_rank_placeholder():
    params = _params()
    params["WorkerConfig"]["agg_gpus_per_worker"] = 16
    params["params"]["agg"].update({"gpus_per_worker": 16, "tensor_parallel_size": 16})
    params["params"]["agg"]["extra_cli_args"].extend(["--dump-config-to", "/results/resolved-config.json"])

    with pytest.raises(ValueError, match="node_rank"):
        _render(params)


def test_fpm_multinode_rejects_name_too_long_for_lws_revision_labels():
    params = _params()
    params["K8sConfig"]["name_prefix"] = "f" * 51
    params["WorkerConfig"]["agg_gpus_per_worker"] = 16
    params["params"]["agg"].update({"gpus_per_worker": 16, "tensor_parallel_size": 16})

    with pytest.raises(ValueError, match="at most 50"):
        _render(params)


def test_fpm_multinode_requires_runtime_discovery_environment(tmp_path):
    script_path = _write_runtime(tmp_path, _render(_multinode_params()))

    completed = subprocess.run(
        ["bash", str(script_path)],
        text=True,
        capture_output=True,
        env=_clean_env(),
        timeout=5,
        check=False,
    )

    # The sourced fpm_env.sh fails closed and terminates run.sh with exit 2.
    assert completed.returncode == 2
    assert "requires rank and leader discovery" in completed.stderr


@pytest.mark.parametrize(
    ("orchestrator", "discovery_env", "expected_master_addr"),
    [
        pytest.param(
            "lws",
            {
                "LWS_WORKER_INDEX": "1",
                "LWS_LEADER_ADDRESS": "leader.example",
            },
            "leader.example",
            id="lws",
        ),
        pytest.param(
            "grove",
            {
                "GROVE_PCLQ_POD_INDEX": "1",
                "GROVE_PCLQ_NAME": "glm52-fpm-agg-0-worker",
                "GROVE_HEADLESS_SERVICE": "glm52-fpm-agg-0.default.svc.cluster.local",
            },
            "glm52-fpm-agg-0-worker-0.glm52-fpm-agg-0.default.svc.cluster.local",
            id="grove",
        ),
    ],
)
def test_fpm_multinode_model_parallel_follower_receives_rank_and_headless(
    tmp_path,
    orchestrator,
    discovery_env,
    expected_master_addr,
):
    args_path = tmp_path / "engine-args.json"
    params = _multinode_params()
    params["K8sConfig"]["fpm_orchestrator"] = orchestrator

    pythonpath = _write_fake_engine(
        tmp_path,
        """\
import json
import os
import pathlib
import sys

pathlib.Path(os.environ["FAKE_ARGS_PATH"]).write_text(json.dumps(sys.argv[1:]))
""",
    )
    script_path = _write_runtime(tmp_path, _render(params))
    env = _clean_env(
        PYTHONPATH=str(pythonpath),
        FAKE_ARGS_PATH=str(args_path),
        **discovery_env,
    )

    completed = subprocess.run(
        ["bash", str(script_path)],
        text=True,
        capture_output=True,
        env=env,
        timeout=5,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    engine_args = json.loads(args_path.read_text())
    assert engine_args[engine_args.index("--nnodes") + 1] == "2"
    assert engine_args[engine_args.index("--node-rank") + 1] == "1"
    assert engine_args[engine_args.index("--master-addr") + 1] == expected_master_addr
    assert engine_args[engine_args.index("--master-port") + 1] == "29500"
    assert engine_args[engine_args.index("--dump-config-to") + 1].endswith("resolved-config-node1.json")
    assert engine_args[-1] == "--headless"


def test_fpm_multinode_dp_rank_receives_local_data_parallel_flags(tmp_path):
    args_path = tmp_path / "engine-args.json"
    params = _multinode_params(data_parallel=True)
    _set_benchmark_mode(params, "prefill")

    pythonpath = _write_fake_engine(
        tmp_path,
        """\
import json
import os
import pathlib
import sys

pathlib.Path(os.environ["FAKE_ARGS_PATH"]).write_text(json.dumps(sys.argv[1:]))
""",
    )
    artifacts = _render(params)
    assert _export_value(artifacts[FPM_ENV_FILENAME], "FPM_DATA_PARALLEL_SIZE") == "16"
    assert _export_value(artifacts[FPM_ENV_FILENAME], "FPM_LOCAL_DATA_PARALLEL_SIZE") == "8"
    script_path = _write_runtime(tmp_path, artifacts)
    env = _clean_env(
        PYTHONPATH=str(pythonpath),
        FAKE_ARGS_PATH=str(args_path),
        LWS_WORKER_INDEX="1",
        LWS_LEADER_ADDRESS="leader.example",
    )

    completed = subprocess.run(
        ["bash", str(script_path)],
        text=True,
        capture_output=True,
        env=env,
        timeout=5,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    engine_args = json.loads(args_path.read_text())
    assert engine_args[engine_args.index("--data-parallel-size-local") + 1] == "8"
    assert engine_args[engine_args.index("--data-parallel-start-rank") + 1] == "8"
    assert engine_args[engine_args.index("--data-parallel-address") + 1] == "leader.example"
    assert engine_args[engine_args.index("--data-parallel-rpc-port") + 1] == "29510"
    assert engine_args[engine_args.index("--dump-config-to") + 1].endswith("resolved-config-node1.json")
    assert engine_args[-1] == "--data-parallel-hybrid-lb"
    assert "--headless" not in engine_args


@pytest.mark.parametrize(
    "flag",
    sorted(
        [
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
        ]
    ),
)
def test_fpm_rejects_passthrough_of_generator_owned_orchestration_flags(flag):
    params = _params()
    params["params"]["agg"]["extra_cli_args"].append(flag)

    with pytest.raises(ValueError, match="owns orchestration option"):
        _render(params)


@pytest.mark.parametrize("spelling", ["--master_port", "--data_parallel_size_local=2", "--node_rank"])
def test_fpm_rejects_underscore_spellings_of_owned_orchestration_flags(spelling):
    """vLLM's FlexibleArgumentParser treats underscore and dash spellings as
    the same option, so the guard must normalize before matching."""

    params = _params()
    params["params"]["agg"]["extra_cli_args"].append(spelling)

    with pytest.raises(ValueError, match="owns orchestration option"):
        _render(params)


@pytest.mark.parametrize("reserved_name", ["FPM_NODE_RANK", "FPM_BENCHMARK_OUTPUT_PATH"])
def test_fpm_rejects_reserved_contract_names_in_extra_env(reserved_name):
    """An extra_env export of a contract variable would shadow the fpm_env.sh
    value for the engine while the collection runtime keeps the original."""

    params = _params()
    params["K8sConfig"]["extra_env"].append({"name": reserved_name, "value": "0"})

    with pytest.raises(ValueError, match="reserved FPM contract variable"):
        _render(params)


def test_fpm_rejects_value_from_environment_entry():
    params = _params()
    params["K8sConfig"]["extra_env"].append(
        {
            "name": "POD_NAME",
            "valueFrom": {
                "fieldRef": {
                    "apiVersion": "v1",
                    "fieldPath": "metadata.name",
                }
            },
        }
    )

    with pytest.raises(ValueError):
        _render(params)


def test_fpm_rejects_env_from_environment_sources():
    params = _params()
    params["K8sConfig"]["worker_extra_pod_spec"] = {
        "mainContainer": {
            "envFrom": [{"secretRef": {"name": "fpm-secret"}}],
        }
    }

    with pytest.raises(ValueError, match="envFrom"):
        _render(params)


def test_fpm_rejects_user_resource_claims():
    params = _params()
    params["K8sConfig"]["worker_extra_pod_spec"] = {
        "resourceClaims": [
            {
                "name": "compute-domain-channel",
                "resourceClaimTemplateName": "user-owned",
            }
        ]
    }

    with pytest.raises(ValueError, match="resourceClaims"):
        _render(params)


def test_fpm_rejects_user_resource_claims_for_mnnvl_multinode():
    params = _params()
    params["NodeConfig"].update({"system_name": "gb200", "num_gpus_per_node": 4})
    params["WorkerConfig"]["agg_gpus_per_worker"] = 8
    params["params"]["agg"].update({"gpus_per_worker": 8, "tensor_parallel_size": 8})
    params["K8sConfig"]["worker_extra_pod_spec"] = {
        "resourceClaims": [
            {
                "name": "compute-domain-channel",
                "resourceClaimTemplateName": "user-owned",
            }
        ]
    }

    with pytest.raises(ValueError, match="resourceClaims"):
        _render(params)


def test_fpm_requires_mp_data_parallel_backend():
    params = _params()
    params["params"]["agg"].update(
        {
            "tensor_parallel_size": 1,
            "data_parallel_size": 4,
            "gpus_per_worker": 4,
        }
    )
    params["params"]["agg"]["extra_cli_args"].extend(["--data-parallel-backend", "ray"])

    with pytest.raises(ValueError, match="data-parallel-backend mp"):
        _render(params)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("enable_router", True),
        ("router_mode", "kv"),
        ("router_config", {"router_reset_states": True}),
        ("planner_config", {"environment": "kubernetes"}),
    ],
)
def test_fpm_rejects_router_and_planner_configuration(field, value):
    params = _params()
    params["DynConfig"][field] = value

    with pytest.raises(ValueError, match="router or planner"):
        _render(params)


def test_fpm_requires_benchmark_mode():
    params = _params()
    args = params["params"]["agg"]["extra_cli_args"]
    index = args.index("--benchmark-mode")
    del args[index : index + 2]

    with pytest.raises(ValueError, match="--benchmark-mode"):
        _render(params)


@pytest.mark.parametrize("benchmark_mode", ["agg", "prefill", "decode"])
def test_fpm_accepts_supported_benchmark_modes(benchmark_mode):
    params = _params()
    _set_benchmark_mode(params, benchmark_mode)

    artifacts = _render(params)

    assert set(artifacts) == _ARTIFACT_TRIO
    assert f"--benchmark-mode {benchmark_mode}" in artifacts[FPM_RUN_SCRIPT_FILENAME]
    assert _export_value(artifacts[FPM_ENV_FILENAME], "FPM_BENCHMARK_MODE") == benchmark_mode


def test_fpm_rejects_unsupported_benchmark_mode():
    params = _params()
    _set_benchmark_mode(params, "disagg")

    with pytest.raises(ValueError, match="agg, prefill, decode"):
        _render(params)


def test_fpm_rejects_another_flag_in_a_required_value_position():
    params = _params()
    _set_benchmark_mode(params, "--scheduler-cls")

    with pytest.raises(ValueError, match="requires a value"):
        _render(params)


def test_fpm_uses_benchmark_timeout_with_startup_grace():
    params = _params()
    params["params"]["agg"]["extra_cli_args"].extend(["--benchmark-timeout", "3600"])

    assert _export_value(_render(params)[FPM_ENV_FILENAME], "FPM_WAIT_TIMEOUT_SECONDS") == "4200"


@pytest.mark.parametrize("timeout", ["zero", "0", "-1"])
def test_fpm_rejects_invalid_benchmark_timeout(timeout):
    params = _params()
    params["params"]["agg"]["extra_cli_args"].extend(["--benchmark-timeout", timeout])

    with pytest.raises(ValueError, match="benchmark-timeout"):
        _render(params)


@pytest.mark.parametrize("invalid", ["--scheduler-cls InstrumentedScheduler", {"--scheduler-cls": "x"}, None])
def test_fpm_rejects_non_list_extra_cli_args(invalid):
    params = _params()
    params["params"]["agg"]["extra_cli_args"] = invalid

    with pytest.raises((TypeError, ValueError)):
        _render(params)


def test_fpm_orchestrator_schema_default_resolves_to_lws():
    # Defaults are Jinja-evaluated; an unquoted identifier silently resolves
    # to None and only the builder's None fallback saves it. Pin the schema
    # default itself so the declaration stays truthful on its own.
    from aiconfigurator.generator.rendering.schemas import apply_defaults

    resolved = apply_defaults("K8sConfig", {}, backend="vllm")
    assert resolved.get("fpm_orchestrator") == "lws"


def test_fpm_multinode_without_mnnvl_fabric_disables_allreduce_fusion():
    # Guard pin: on multinode systems without an MNNVL fabric the pinned
    # engine's fused allreduce either hangs (auto -> mnnvl) or refuses to
    # start (trtllm), so the render must disable the fusion pass.
    script = _render(_multinode_params())[FPM_RUN_SCRIPT_FILENAME]
    assert '"fuse_allreduce_rms":false' in script


def test_fpm_single_node_keeps_allreduce_fusion():
    script = _render(_params())[FPM_RUN_SCRIPT_FILENAME]
    assert "fuse_allreduce_rms" not in script


def test_fpm_multinode_mnnvl_fabric_keeps_allreduce_fusion():
    params = _params()
    params["K8sConfig"]["fpm_orchestrator"] = "grove"
    params["NodeConfig"].update({"system_name": "gb200", "num_gpus_per_node": 4})
    params["WorkerConfig"]["agg_gpus_per_worker"] = 8
    params["params"]["agg"].update({"gpus_per_worker": 8, "tensor_parallel_size": 8})

    script = _render(params)[FPM_RUN_SCRIPT_FILENAME]
    assert "fuse_allreduce_rms" not in script


def test_fpm_multinode_fusion_guard_merges_into_existing_compilation_config():
    # The base fixture already carries the collector's compilation config
    # (cudagraph capture policy); the guard must merge into that single flag
    # rather than appending a second occurrence.
    script = _render(_multinode_params())[FPM_RUN_SCRIPT_FILENAME]

    assert script.count("--compilation-config") == 1
    assert "cudagraph_mode" in script
    assert '"fuse_allreduce_rms":false' in script

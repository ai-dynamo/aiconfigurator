# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public-contract tests for the reusable-Pod FPM artifact target."""

from __future__ import annotations

import copy
import json
import os
import shlex
import signal
import stat
import subprocess
import time
from pathlib import Path

import pytest
import yaml

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


def _render(params: dict | None = None, backend: str = "vllm") -> dict[str, str]:
    return render_backend_templates(
        copy.deepcopy(params or _params()),
        backend,
        version=_BACKEND_VERSION,
        deployment_target="fpm",
    )


def _set_benchmark_mode(params: dict, mode: str) -> None:
    args = params["params"]["agg"]["extra_cli_args"]
    args[args.index("--benchmark-mode") + 1] = mode


def _benchmark_result(
    *,
    mode: str = "prefill",
    dp_rank: int = 0,
    point_types: list[str] | None = None,
    status: str = "complete",
    valid: bool = True,
) -> dict:
    if point_types is None:
        point_types = ["prefill" if mode == "agg" else mode]
    results = [
        {
            "point": {"point_type": point_type},
            "fpms": [{"dp_rank": dp_rank, "wall_time": 0.001}],
        }
        for point_type in point_types
    ]
    return {
        "schema_version": 1,
        "status": status,
        "valid": valid,
        "coverage": {
            "expected_points": len(results),
            "completed_points": len(results),
            "skipped_points": 0,
        },
        "config": {"mode": mode},
        "results": results,
        "skipped_points": [],
        "errors": [] if valid else ["boom"],
    }


def _run_script_with_static_result(
    tmp_path: Path,
    result: object,
    *,
    benchmark_mode: str = "prefill",
) -> subprocess.CompletedProcess[str]:
    output_path = tmp_path / "benchmark.json"
    params = _params()
    _set_benchmark_mode(params, benchmark_mode)
    for entry in params["K8sConfig"]["extra_env"]:
        if entry["name"] == "DYN_FPM_BENCHMARK_OUTPUT_PATH":
            entry["value"] = str(output_path)

    fake_package = tmp_path / "fake-package" / "dynamo" / "vllm"
    fake_package.mkdir(parents=True)
    (fake_package.parent / "__init__.py").write_text("")
    (fake_package / "__init__.py").write_text("")
    serialized = json.dumps(result)
    (fake_package / "__main__.py").write_text(
        f"""\
import pathlib
import signal
import sys
import time

path = pathlib.Path(sys.argv[sys.argv.index("--benchmark-output-path") + 1])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text({serialized!r})
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
while True:
    time.sleep(0.1)
"""
    )
    script_path = tmp_path / "run.sh"
    script_path.write_text(_render(params)["run.sh"])
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path / "fake-package")
    return subprocess.run(
        ["bash", str(script_path)],
        text=True,
        capture_output=True,
        env=env,
        timeout=8,
        check=False,
    )


def _pod(artifacts: dict[str, str]) -> dict:
    document = yaml.safe_load(artifacts["k8s_deploy.yaml"])
    assert isinstance(document, dict)
    return document


def _main_container(pod: dict) -> dict:
    containers = pod["spec"]["containers"]
    assert len(containers) == 1
    return containers[0]


def _export_value(script: str, name: str) -> str:
    prefix = f"{name}="
    for line in script.splitlines():
        try:
            tokens = shlex.split(line, comments=True)
        except ValueError:
            continue
        if not tokens or tokens[0] != "export":
            continue
        for assignment in tokens[1:]:
            if assignment.startswith(prefix):
                return assignment[len(prefix) :]
    raise AssertionError(f"missing export for {name}")


def test_fpm_render_returns_only_resource_pod_and_run_script():
    artifacts = _render()

    assert set(artifacts) == {"k8s_deploy.yaml", "run.sh"}


def test_fpm_pinned_vllm_024_uses_floor_template_and_preserves_fpm_overlay():
    artifacts = render_backend_templates(
        copy.deepcopy(_params()),
        "vllm",
        version="0.24.0",
        deployment_target="fpm",
    )

    assert yaml.safe_load(artifacts["k8s_deploy.yaml"])["kind"] == "Pod"
    assert "--tensor-parallel-size 4" in artifacts["run.sh"]
    assert "--scheduler-cls fpm.scheduler.InstrumentedScheduler" in artifacts["run.sh"]
    assert "--dump-config-to /results/resolved-config-node0.json" in artifacts["run.sh"]


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
    script = _render()["run.sh"]

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
    assert "wait_timeout_seconds=7800" in script

    syntax = subprocess.run(
        ["bash", "-n"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr


def test_fpm_preserves_duplicate_environment_export_order():
    params = _params()
    params["K8sConfig"]["extra_env"].extend(
        [
            {"name": "NCCL_DEBUG", "value": "WARN"},
            {"name": "NCCL_DEBUG", "value": "TRACE"},
        ]
    )

    exports = [line for line in _render(params)["run.sh"].splitlines() if line.startswith("export NCCL_DEBUG=")]

    assert exports == ["export NCCL_DEBUG=INFO", "export NCCL_DEBUG=WARN", "export NCCL_DEBUG=TRACE"]


def test_fpm_keeps_cli_and_environment_output_paths_aligned():
    params = _params()
    params["K8sConfig"]["extra_env"] = [
        entry for entry in params["K8sConfig"]["extra_env"] if entry["name"] != "DYN_FPM_BENCHMARK_OUTPUT_PATH"
    ]
    params["params"]["agg"]["extra_cli_args"].extend(["--benchmark-output-path", "/results/custom.json"])

    script = _render(params)["run.sh"]

    assert _export_value(script, "DYN_FPM_BENCHMARK_OUTPUT_PATH") == "/results/custom.json"


def test_fpm_rejects_conflicting_output_paths():
    params = _params()
    params["params"]["agg"]["extra_cli_args"].extend(["--benchmark-output-path", "/results/different.json"])

    with pytest.raises(ValueError, match="same path"):
        _render(params)


def _process_is_running(process_pid: int) -> bool:
    """Return false for an exited process, including an unreaped Linux zombie."""
    try:
        os.kill(process_pid, 0)
    except ProcessLookupError:
        return False

    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return True

    try:
        status_lines = (proc_root / str(process_pid) / "status").read_text().splitlines()
    except FileNotFoundError:
        return False
    except OSError:
        return True

    state = next((line for line in status_lines if line.startswith("State:")), None)
    if state is None:
        return True
    state_parts = state.split()
    return len(state_parts) < 2 or state_parts[1] not in {"Z", "X", "x"}


def _assert_process_stopped(pid_path) -> None:
    process_pid = int(pid_path.read_text())
    exit_deadline = time.monotonic() + 2
    while time.monotonic() < exit_deadline:
        if not _process_is_running(process_pid):
            return
        time.sleep(0.05)

    if not _process_is_running(process_pid):
        return
    try:
        os.kill(process_pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    pytest.fail(f"process {process_pid} survived FPM run.sh cleanup")


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


def test_fpm_api_writes_exact_filenames_and_executable_script(tmp_path):
    artifacts = generate_backend_artifacts(
        copy.deepcopy(_params()),
        "vllm",
        output_dir=str(tmp_path),
        backend_version=_BACKEND_VERSION,
        deployment_target="fpm",
    )

    assert set(artifacts) == {"k8s_deploy.yaml", "run.sh"}
    assert {path.name for path in tmp_path.iterdir()} == {"k8s_deploy.yaml", "run.sh"}
    assert not (tmp_path / "run_x.sh").exists()
    assert (tmp_path / "run.sh").stat().st_mode & stat.S_IXUSR
    assert yaml.safe_load((tmp_path / "k8s_deploy.yaml").read_text())["kind"] == "Pod"


def test_fpm_run_script_accepts_legacy_schema_v2_and_stops_fake_engine(tmp_path):
    output_path = tmp_path / "benchmark.json"
    params = _params()
    for entry in params["K8sConfig"]["extra_env"]:
        if entry["name"] == "DYN_FPM_BENCHMARK_OUTPUT_PATH":
            entry["value"] = str(output_path)

    fake_package = tmp_path / "fake-package" / "dynamo" / "vllm"
    fake_package.mkdir(parents=True)
    (fake_package.parent / "__init__.py").write_text("")
    (fake_package / "__init__.py").write_text("")
    (fake_package / "__main__.py").write_text(
        """\
import json
import pathlib
import signal
import sys
import time

flag = "--benchmark-output-path"
index = sys.argv.index(flag)
path = pathlib.Path(sys.argv[index + 1])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({"schema_version": 2, "status": "passed", "config": {"dp_rank": 0}}))
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
while True:
    time.sleep(0.1)
"""
    )
    script_path = tmp_path / "run.sh"
    script_path.write_text(_render(params)["run.sh"])
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path / "fake-package")

    completed = subprocess.run(
        ["bash", str(script_path)],
        text=True,
        capture_output=True,
        env=env,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(output_path.read_text()) == {
        "schema_version": 2,
        "status": "passed",
        "config": {"dp_rank": 0},
    }

    repeated = subprocess.run(
        ["bash", str(script_path)],
        text=True,
        capture_output=True,
        env=env,
        timeout=10,
        check=False,
    )
    assert repeated.returncode == 1
    assert "Refusing to overwrite" in repeated.stderr


@pytest.mark.parametrize(
    ("benchmark_mode", "point_types"),
    [
        ("prefill", ["prefill"]),
        ("decode", ["decode"]),
        ("agg", ["prefill", "decode"]),
    ],
)
def test_fpm_run_script_accepts_schema_v1_for_supported_benchmark_modes(
    tmp_path,
    benchmark_mode,
    point_types,
):
    result = _benchmark_result(mode=benchmark_mode, point_types=point_types)

    completed = _run_script_with_static_result(
        tmp_path,
        result,
        benchmark_mode=benchmark_mode,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("case", "expected_message"),
    [
        ("schema", "unsupported schema_version 3"),
        ("mode", "benchmark mode 'decode' != 'prefill'"),
        ("coverage", "invalid coverage"),
        ("result_count", "results count 0 != 1"),
        ("point_type", "point type 'decode' is not valid for 'prefill'"),
        ("dp_rank", "FPM dp_ranks [1] != [0]"),
        ("non_object", "top-level JSON must be an object"),
    ],
)
def test_fpm_run_script_rejects_invalid_schema_v1_result(
    tmp_path,
    case,
    expected_message,
):
    result = _benchmark_result()
    payload: object = result
    if case == "schema":
        result["schema_version"] = 3
    elif case == "mode":
        result["config"]["mode"] = "decode"
    elif case == "coverage":
        result["coverage"]["completed_points"] = 0
    elif case == "result_count":
        result["results"] = []
    elif case == "point_type":
        result["results"][0]["point"]["point_type"] = "decode"
    elif case == "dp_rank":
        result["results"][0]["fpms"][0]["dp_rank"] = 1
    elif case == "non_object":
        payload = []
    else:  # pragma: no cover - protects the test table itself
        raise AssertionError(f"unknown test case: {case}")

    completed = _run_script_with_static_result(tmp_path, payload)

    assert completed.returncode == 1
    assert expected_message in completed.stderr


def test_fpm_run_script_waits_for_every_single_node_dp_result(tmp_path):
    output_path = tmp_path / "benchmark.json"
    params = _params()
    _set_benchmark_mode(params, "prefill")
    params["params"]["agg"].update(
        {
            "tensor_parallel_size": 1,
            "data_parallel_size": 4,
            "gpus_per_worker": 4,
        }
    )
    for entry in params["K8sConfig"]["extra_env"]:
        if entry["name"] == "DYN_FPM_BENCHMARK_OUTPUT_PATH":
            entry["value"] = str(output_path)

    fake_package = tmp_path / "fake-package" / "dynamo" / "vllm"
    fake_package.mkdir(parents=True)
    (fake_package.parent / "__init__.py").write_text("")
    (fake_package / "__init__.py").write_text("")
    (fake_package / "__main__.py").write_text(
        """\
import json
import pathlib
import signal
import sys
import time

output_flag = sys.argv.index("--benchmark-output-path")
base = pathlib.Path(sys.argv[output_flag + 1])
dp_flag = sys.argv.index("--data-parallel-size")
dp_size = int(sys.argv[dp_flag + 1])
base.parent.mkdir(parents=True, exist_ok=True)
for rank in range(dp_size):
    path = base if rank == 0 else base.with_name(f"{base.stem}_dp{rank}{base.suffix}")
    path.write_text(json.dumps({
        "schema_version": 1,
        "status": "complete",
        "valid": True,
        "coverage": {"expected_points": 1, "completed_points": 1, "skipped_points": 0},
        "config": {"mode": "prefill"},
        "results": [{"point": {"point_type": "prefill"}, "fpms": [{"dp_rank": rank}]}],
        "skipped_points": [],
        "rank": rank,
    }))
    time.sleep(1)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
while True:
    time.sleep(0.1)
"""
    )
    script_path = tmp_path / "run.sh"
    script_path.write_text(_render(params)["run.sh"])
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path / "fake-package")

    completed = subprocess.run(
        ["bash", str(script_path)],
        text=True,
        capture_output=True,
        env=env,
        timeout=12,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert [
        json.loads((output_path if rank == 0 else tmp_path / f"benchmark_dp{rank}.json").read_text())["rank"]
        for rank in range(4)
    ] == [0, 1, 2, 3]


def test_fpm_run_script_rejects_terminal_failed_result(tmp_path):
    completed = _run_script_with_static_result(
        tmp_path,
        _benchmark_result(status="failed", valid=False),
    )

    assert completed.returncode == 1
    assert "status='failed'" in completed.stderr
    assert "valid=False" in completed.stderr
    assert "boom" in completed.stderr


@pytest.mark.parametrize(
    ("result_status", "expected_returncode"),
    [
        ("passed", 0),
        ("failed", 1),
    ],
)
def test_fpm_run_script_bounds_stubborn_engine_shutdown(
    tmp_path,
    result_status,
    expected_returncode,
):
    output_path = tmp_path / "benchmark.json"
    pid_path = tmp_path / "engine.pid"
    child_pid_path = tmp_path / "engine-child.pid"
    params = _params()
    for entry in params["K8sConfig"]["extra_env"]:
        if entry["name"] == "DYN_FPM_BENCHMARK_OUTPUT_PATH":
            entry["value"] = str(output_path)

    fake_package = tmp_path / "fake-package" / "dynamo" / "vllm"
    fake_package.mkdir(parents=True)
    (fake_package.parent / "__init__.py").write_text("")
    (fake_package / "__init__.py").write_text("")
    (fake_package / "__main__.py").write_text(
        f"""\
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

signal.signal(signal.SIGTERM, signal.SIG_IGN)
pathlib.Path(os.environ["FAKE_ENGINE_PID_PATH"]).write_text(str(os.getpid()))
child = subprocess.Popen([
    sys.executable,
    "-c",
    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(3600)",
])
pathlib.Path(os.environ["FAKE_ENGINE_CHILD_PID_PATH"]).write_text(str(child.pid))
index = sys.argv.index("--benchmark-output-path")
path = pathlib.Path(sys.argv[index + 1])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({{
    "schema_version": 2,
    "status": "{result_status}",
    "config": {{"dp_rank": 0}},
}}))
while True:
    time.sleep(0.1)
"""
    )
    script_path = tmp_path / "run.sh"
    script = _render(params)["run.sh"]
    assert script.count("engine_shutdown_grace_seconds=30") == 1
    script = script.replace(
        "engine_shutdown_grace_seconds=30",
        "engine_shutdown_grace_seconds=1",
    )
    script_path.write_text(script)
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(tmp_path / "fake-package"),
            "FAKE_ENGINE_PID_PATH": str(pid_path),
            "FAKE_ENGINE_CHILD_PID_PATH": str(child_pid_path),
        }
    )

    started = time.monotonic()
    completed = subprocess.run(
        ["bash", str(script_path)],
        text=True,
        capture_output=True,
        env=env,
        timeout=8,
        check=False,
    )
    elapsed = time.monotonic() - started

    assert completed.returncode == expected_returncode
    assert elapsed < 7
    assert "Engine did not stop within 1s; sending SIGKILL" in completed.stderr
    for process_pid_path in (pid_path, child_pid_path):
        _assert_process_stopped(process_pid_path)


def test_fpm_run_script_cleans_process_group_when_engine_parent_exits(tmp_path):
    output_path = tmp_path / "benchmark.json"
    child_pid_path = tmp_path / "engine-child.pid"
    params = _params()
    for entry in params["K8sConfig"]["extra_env"]:
        if entry["name"] == "DYN_FPM_BENCHMARK_OUTPUT_PATH":
            entry["value"] = str(output_path)

    fake_package = tmp_path / "fake-package" / "dynamo" / "vllm"
    fake_package.mkdir(parents=True)
    (fake_package.parent / "__init__.py").write_text("")
    (fake_package / "__init__.py").write_text("")
    (fake_package / "__main__.py").write_text(
        """\
import os
import pathlib
import signal
import subprocess
import sys

child = subprocess.Popen(
    [
        sys.executable,
        "-c",
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(3600)",
    ],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
pathlib.Path(os.environ["FAKE_ENGINE_CHILD_PID_PATH"]).write_text(str(child.pid))
raise SystemExit(23)
"""
    )
    script_path = tmp_path / "run.sh"
    script = _render(params)["run.sh"]
    assert script.count("engine_shutdown_grace_seconds=30") == 1
    script = script.replace(
        "engine_shutdown_grace_seconds=30",
        "engine_shutdown_grace_seconds=1",
    )
    script_path.write_text(script)
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(tmp_path / "fake-package"),
            "FAKE_ENGINE_CHILD_PID_PATH": str(child_pid_path),
        }
    )

    try:
        completed = subprocess.run(
            ["bash", str(script_path)],
            text=True,
            capture_output=True,
            env=env,
            timeout=8,
            check=False,
        )
    except subprocess.TimeoutExpired:
        if child_pid_path.exists():
            os.kill(int(child_pid_path.read_text()), signal.SIGKILL)
        raise

    assert completed.returncode == 23
    assert "Engine exited before writing all FPM benchmark outputs" in completed.stderr
    assert "Engine did not stop within 1s; sending SIGKILL" in completed.stderr
    _assert_process_stopped(child_pid_path)


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

    assert {path.name for path in output_dir.iterdir()} == {"k8s_deploy.yaml", "run.sh"}


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


def test_fpm_multinode_worker_emits_keepalive_leaderworkerset_and_rank_aware_script():
    params = _params()
    params["WorkerConfig"]["agg_gpus_per_worker"] = 16
    params["params"]["agg"]["gpus_per_worker"] = 16
    params["params"]["agg"]["tensor_parallel_size"] = 16

    artifacts = _render(params)
    workload = yaml.safe_load(artifacts["k8s_deploy.yaml"])

    assert workload["apiVersion"] == "leaderworkerset.x-k8s.io/v1"
    assert workload["kind"] == "LeaderWorkerSet"
    group = workload["spec"]["leaderWorkerTemplate"]
    assert group["size"] == 2
    for template_name in ("leaderTemplate", "workerTemplate"):
        container = group[template_name]["spec"]["containers"][0]
        assert container["resources"]["limits"]["nvidia.com/gpu"] == "8"
        assert container["command"] == ["/bin/bash", "-lc"]
        assert container["args"] == ["exec sleep infinity"]

    script = artifacts["run.sh"]
    assert 'node_rank="${LWS_WORKER_INDEX:?LWS_WORKER_INDEX is required for multinode FPM}"' in script
    assert 'master_addr="${LWS_LEADER_ADDRESS:?LWS_LEADER_ADDRESS is required for multinode FPM}"' in script
    assert '--nnodes "$node_count" --node-rank "$node_rank"' in script
    assert 'exec "${engine_command[@]}" --headless' in script


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

    workload = yaml.safe_load(_render(params)["k8s_deploy.yaml"])
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


def test_fpm_multinode_requires_lws_runtime_environment(tmp_path):
    params = _params()
    params["WorkerConfig"]["agg_gpus_per_worker"] = 16
    params["params"]["agg"].update({"gpus_per_worker": 16, "tensor_parallel_size": 16})
    script_path = tmp_path / "run.sh"
    script_path.write_text(_render(params)["run.sh"])

    completed = subprocess.run(
        ["bash", str(script_path)],
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert completed.returncode != 0
    assert "LWS_WORKER_INDEX is required" in completed.stderr


def test_fpm_multinode_model_parallel_follower_receives_rank_and_headless(tmp_path):
    args_path = tmp_path / "engine-args.json"
    params = _params()
    params["WorkerConfig"]["agg_gpus_per_worker"] = 16
    params["params"]["agg"].update({"gpus_per_worker": 16, "tensor_parallel_size": 16})

    fake_package = tmp_path / "fake-package" / "dynamo" / "vllm"
    fake_package.mkdir(parents=True)
    (fake_package.parent / "__init__.py").write_text("")
    (fake_package / "__init__.py").write_text("")
    (fake_package / "__main__.py").write_text(
        """\
import json
import os
import pathlib
import sys

pathlib.Path(os.environ["FAKE_ARGS_PATH"]).write_text(json.dumps(sys.argv[1:]))
"""
    )
    script_path = tmp_path / "run.sh"
    script_path.write_text(_render(params)["run.sh"])
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(tmp_path / "fake-package"),
            "FAKE_ARGS_PATH": str(args_path),
            "LWS_WORKER_INDEX": "1",
            "LWS_LEADER_ADDRESS": "leader.example",
        }
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
    assert engine_args[engine_args.index("--master-addr") + 1] == "leader.example"
    assert engine_args[engine_args.index("--dump-config-to") + 1].endswith("resolved-config-node1.json")
    assert engine_args[-1] == "--headless"


def test_fpm_multinode_dp_rank_executes_local_result_range(tmp_path):
    output_path = tmp_path / "run.v1" / "metrics.final.json"
    args_path = tmp_path / "engine-args.json"
    params = _params()
    _set_benchmark_mode(params, "prefill")
    params["WorkerConfig"]["agg_gpus_per_worker"] = 16
    params["params"]["agg"].update(
        {
            "gpus_per_worker": 16,
            "tensor_parallel_size": 1,
            "data_parallel_size": 16,
        }
    )
    for entry in params["K8sConfig"]["extra_env"]:
        if entry["name"] == "DYN_FPM_BENCHMARK_OUTPUT_PATH":
            entry["value"] = str(output_path)

    fake_package = tmp_path / "fake-package" / "dynamo" / "vllm"
    fake_package.mkdir(parents=True)
    (fake_package.parent / "__init__.py").write_text("")
    (fake_package / "__init__.py").write_text("")
    (fake_package / "__main__.py").write_text(
        """\
import json
import os
import pathlib
import signal
import sys
import time

pathlib.Path(os.environ["FAKE_ARGS_PATH"]).write_text(json.dumps(sys.argv[1:]))
output = pathlib.Path(sys.argv[sys.argv.index("--benchmark-output-path") + 1])
local = int(sys.argv[sys.argv.index("--data-parallel-size-local") + 1])
start = int(sys.argv[sys.argv.index("--data-parallel-start-rank") + 1])
output.parent.mkdir(parents=True, exist_ok=True)
for rank in range(start, start + local):
    path = output if rank == 0 else output.with_name(f"{output.stem}_dp{rank}{output.suffix}")
    path.write_text(json.dumps({
        "schema_version": 1,
        "status": "complete",
        "valid": True,
        "coverage": {"expected_points": 1, "completed_points": 1, "skipped_points": 0},
        "config": {"mode": "prefill"},
        "results": [{"point": {"point_type": "prefill"}, "fpms": [{"dp_rank": rank}]}],
        "skipped_points": [],
    }))
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
while True:
    time.sleep(0.1)
"""
    )
    script_path = tmp_path / "run.sh"
    script_path.write_text(_render(params)["run.sh"])
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(tmp_path / "fake-package"),
            "FAKE_ARGS_PATH": str(args_path),
            "LWS_WORKER_INDEX": "1",
            "LWS_LEADER_ADDRESS": "leader.example",
        }
    )

    completed = subprocess.run(
        ["bash", str(script_path)],
        text=True,
        capture_output=True,
        env=env,
        timeout=8,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    engine_args = json.loads(args_path.read_text())
    assert engine_args[engine_args.index("--data-parallel-size-local") + 1] == "8"
    assert engine_args[engine_args.index("--data-parallel-start-rank") + 1] == "8"
    assert engine_args[engine_args.index("--data-parallel-address") + 1] == "leader.example"
    assert engine_args[engine_args.index("--dump-config-to") + 1].endswith("resolved-config-node1.json")
    assert engine_args[-1] == "--data-parallel-hybrid-lb"
    assert "--headless" not in engine_args
    assert {path.name for path in output_path.parent.glob("metrics.final_dp*.json")} == {
        f"metrics.final_dp{rank}.json" for rank in range(8, 16)
    }


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
    args = params["params"]["agg"]["extra_cli_args"]
    index = args.index("--benchmark-mode")
    args[index + 1] = benchmark_mode

    artifacts = _render(params)

    assert set(artifacts) == {"k8s_deploy.yaml", "run.sh"}
    assert f"--benchmark-mode {benchmark_mode}" in artifacts["run.sh"]


def test_fpm_rejects_unsupported_benchmark_mode():
    params = _params()
    args = params["params"]["agg"]["extra_cli_args"]
    index = args.index("--benchmark-mode")
    args[index + 1] = "disagg"

    with pytest.raises(ValueError, match="agg, prefill, decode"):
        _render(params)


def test_fpm_rejects_another_flag_in_a_required_value_position():
    params = _params()
    args = params["params"]["agg"]["extra_cli_args"]
    index = args.index("--benchmark-mode")
    args[index + 1] = "--scheduler-cls"

    with pytest.raises(ValueError, match="requires a value"):
        _render(params)


def test_fpm_uses_benchmark_timeout_with_startup_grace():
    params = _params()
    params["params"]["agg"]["extra_cli_args"].extend(["--benchmark-timeout", "3600"])

    assert "wait_timeout_seconds=4200" in _render(params)["run.sh"]


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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public-contract tests for the reusable-Pod FPM artifact target."""

from __future__ import annotations

import copy
import json
import os
import shlex
import stat
import subprocess

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


def test_fpm_run_script_completes_and_stops_fake_engine(tmp_path):
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
path.write_text(json.dumps({"status": "ok"}))
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
    assert json.loads(output_path.read_text()) == {"status": "ok"}

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


def test_fpm_rejects_multinode_worker():
    params = _params()
    params["WorkerConfig"]["agg_gpus_per_worker"] = 16
    params["params"]["agg"]["gpus_per_worker"] = 16
    params["params"]["agg"]["tensor_parallel_size"] = 16

    with pytest.raises(ValueError):
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


def test_fpm_requires_aggregated_benchmark_mode():
    params = _params()
    args = params["params"]["agg"]["extra_cli_args"]
    index = args.index("--benchmark-mode")
    args[index + 1] = "disagg"

    with pytest.raises(ValueError, match="--benchmark-mode agg"):
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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM workers must advertise the configured served model name.

The vLLM cli_args templates do not emit ``--served-model-name`` (unlike the
sglang/trtllm run scripts), so the k8s worker args and the run.sh launcher must
carry this service-level flag; otherwise the deployed model only answers to the
HF model id and requests using the configured alias 404.
"""

from __future__ import annotations

import copy
import shlex

import yaml

from aiconfigurator.generator.api import generate_backend_artifacts

_BACKEND_VERSION = "0.20.1"

_PARAMS = {
    "ServiceConfig": {
        "model_path": "Qwen/Qwen3-32B-FP8",
        "served_model_path": "Qwen/Qwen3-32B-FP8",
        "served_model_name": "Qwen3-32B-FP8",
        "include_frontend": True,
    },
    "K8sConfig": {"name_prefix": "test", "k8s_namespace": "default"},
    "DynConfig": {"mode": "agg"},
    "WorkerConfig": {"agg_workers": 1, "agg_gpus_per_worker": 1, "prefill_workers": 0, "decode_workers": 0},
    "NodeConfig": {"num_gpus_per_node": 8},
    "SlaConfig": {"isl": 1024, "osl": 256},
    "ModelConfig": {"is_moe": False, "prefix": 0, "nextn": 0},
    "BenchConfig": {},
    "params": {
        "agg": {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "max_batch_size": 64,
            "max_num_tokens": 4096,
            "max_seq_len": 4096,
        }
    },
}


def _render():
    return generate_backend_artifacts(
        copy.deepcopy(_PARAMS),
        "vllm",
        backend_version=_BACKEND_VERSION,
        deployment_target="dynamo-j2",
    )


def test_k8s_worker_args_carry_served_model_name():
    artifacts = _render()
    k8s = yaml.safe_load(artifacts["k8s_deploy.yaml"])
    services = k8s["spec"]["services"]
    worker = next(svc for name, svc in services.items() if name != "Frontend")
    args = worker["extraPodSpec"]["mainContainer"]["args"]

    assert "--served-model-name" in args
    assert args[args.index("--served-model-name") + 1] == "Qwen3-32B-FP8"


def test_run_script_carries_served_model_name():
    artifacts = _render()
    run_script = next(v for k, v in artifacts.items() if k.startswith("run_") and k.endswith(".sh"))

    assert 'export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen3-32B-FP8"}' in run_script
    assert '--served-model-name "$SERVED_MODEL_NAME"' in run_script


def test_served_model_name_appears_once_per_worker():
    # The templates own only --model; the builder owns --served-model-name.
    # Guard against a future duplicate if a template starts emitting it too.
    artifacts = _render()
    k8s = yaml.safe_load(artifacts["k8s_deploy.yaml"])
    services = k8s["spec"]["services"]
    worker = next(svc for name, svc in services.items() if name != "Frontend")
    args = worker["extraPodSpec"]["mainContainer"]["args"]
    assert args.count("--served-model-name") == 1

    run_script = next(v for k, v in artifacts.items() if k.startswith("run_") and k.endswith(".sh"))
    # One worker loop in agg mode -> exactly one occurrence in the rendered script.
    assert run_script.count("--served-model-name") == 1
    # Sanity: the run script is shell-parseable.
    shlex.split(run_script.split("python3 -m dynamo.vllm", 1)[0])


def test_empty_served_model_name_omits_flag():
    # An empty served_model_name must NOT emit `--served-model-name ""` (which
    # vLLM treats as an explicit empty alias -> 404), matching the k8s builder's
    # `if served_model_name` guard. Both the k8s worker args and run.sh must
    # fall back to `--model` only.
    params = copy.deepcopy(_PARAMS)
    params["ServiceConfig"]["served_model_name"] = ""

    artifacts = generate_backend_artifacts(
        params, "vllm", backend_version=_BACKEND_VERSION, deployment_target="dynamo-j2"
    )

    k8s = yaml.safe_load(artifacts["k8s_deploy.yaml"])
    services = k8s["spec"]["services"]
    worker = next(svc for name, svc in services.items() if name != "Frontend")
    assert "--served-model-name" not in worker["extraPodSpec"]["mainContainer"]["args"]

    run_script = next(v for k, v in artifacts.items() if k.startswith("run_") and k.endswith(".sh"))
    assert "--served-model-name" not in run_script

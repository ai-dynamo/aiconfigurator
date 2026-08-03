# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for vLLM disaggregated worker role arguments."""

from __future__ import annotations

import copy

import pytest
import yaml

from aiconfigurator.generator.api import generate_backend_artifacts

_VERSION_CASES = [
    pytest.param("0.9.0", "0.14.1", "legacy", id="dynamo-0.9"),
    pytest.param("1.0.0", "0.16.0", "disaggregation-mode", id="dynamo-1.0"),
    pytest.param("1.4.0", "0.26.0", "disaggregation-mode", id="dynamo-1.4"),
]

_PARAMS = {
    "ServiceConfig": {
        "model_path": "Qwen/Qwen3-32B-FP8",
        "served_model_path": "Qwen/Qwen3-32B-FP8",
        "served_model_name": "Qwen3-32B-FP8",
        "include_frontend": True,
    },
    "K8sConfig": {"name_prefix": "test", "k8s_namespace": "default"},
    "DynConfig": {"mode": "disagg"},
    "WorkerConfig": {
        "agg_workers": 0,
        "agg_gpus_per_worker": 0,
        "prefill_workers": 1,
        "prefill_gpus_per_worker": 1,
        "decode_workers": 1,
        "decode_gpus_per_worker": 1,
    },
    "NodeConfig": {"num_gpus_per_node": 8},
    "SlaConfig": {"isl": 1024, "osl": 256},
    "ModelConfig": {"is_moe": False, "prefix": 0, "nextn": 0},
    "BenchConfig": {},
    "params": {
        "prefill": {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "max_batch_size": 1,
            "max_num_tokens": 2524,
            "max_seq_len": 4096,
        },
        "decode": {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "max_batch_size": 512,
            "max_num_tokens": 512,
            "max_seq_len": 4096,
        },
    },
}


def _render(dynamo_version: str, backend_version: str) -> dict[str, str]:
    params = copy.deepcopy(_PARAMS)
    params["generator_dynamo_version"] = dynamo_version
    return generate_backend_artifacts(
        params,
        "vllm",
        backend_version=backend_version,
        deployment_target="dynamo-j2",
    )


def _value_after(args: list[str], flag: str) -> str:
    assert args.count(flag) == 1
    return args[args.index(flag) + 1]


@pytest.mark.parametrize("dynamo_version,backend_version,interface", _VERSION_CASES)
def test_vllm_disaggregated_k8s_workers_use_versioned_role_interface(
    dynamo_version,
    backend_version,
    interface,
):
    artifacts = _render(dynamo_version, backend_version)
    manifest = yaml.safe_load(artifacts["k8s_deploy.yaml"])
    services = manifest["spec"]["services"]

    prefill_args = services["VllmPrefillWorker"]["extraPodSpec"]["mainContainer"]["args"]
    decode_args = services["VllmDecodeWorker"]["extraPodSpec"]["mainContainer"]["args"]

    if interface == "legacy":
        assert "--is-prefill-worker" in prefill_args
        assert "--is-decode-worker" in decode_args
        assert "--disaggregation-mode" not in prefill_args
        assert "--disaggregation-mode" not in decode_args
    else:
        assert "--is-prefill-worker" not in prefill_args
        assert "--is-decode-worker" not in decode_args
        assert _value_after(prefill_args, "--disaggregation-mode") == "prefill"
        assert _value_after(decode_args, "--disaggregation-mode") == "decode"
    assert "--kv-transfer-config" in prefill_args
    assert "--kv-transfer-config" in decode_args


@pytest.mark.parametrize("dynamo_version,backend_version,interface", _VERSION_CASES)
def test_vllm_disaggregated_launch_artifacts_use_versioned_role_interface(
    dynamo_version,
    backend_version,
    interface,
):
    artifacts = _render(dynamo_version, backend_version)
    run_scripts = "\n".join(content for name, content in artifacts.items() if name.startswith("run_"))
    sflow = artifacts["sflow.yaml"]

    for content in (run_scripts, sflow):
        if interface == "legacy":
            assert content.count("--is-prefill-worker") == 1
            assert content.count("--is-decode-worker") == 1
            assert "--disaggregation-mode" not in content
        else:
            assert "--is-prefill-worker" not in content
            assert "--is-decode-worker" not in content
            assert content.count("--disaggregation-mode prefill") == 1
            assert content.count("--disaggregation-mode decode") == 1
        assert content.count("--kv-transfer-config") >= 2

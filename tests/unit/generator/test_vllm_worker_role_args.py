# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for vLLM disaggregated worker role arguments."""

from __future__ import annotations

import copy

import yaml

from aiconfigurator.generator.api import generate_backend_artifacts

_BACKEND_VERSION = "0.26.0"
_REMOVED_ROLE_FLAGS = {"--is-prefill-worker", "--is-decode-worker"}

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


def _render() -> dict[str, str]:
    return generate_backend_artifacts(
        copy.deepcopy(_PARAMS),
        "vllm",
        backend_version=_BACKEND_VERSION,
        deployment_target="dynamo-j2",
    )


def _value_after(args: list[str], flag: str) -> str:
    assert args.count(flag) == 1
    return args[args.index(flag) + 1]


def test_vllm_disaggregated_k8s_workers_use_disaggregation_mode():
    artifacts = _render()
    manifest = yaml.safe_load(artifacts["k8s_deploy.yaml"])
    services = manifest["spec"]["services"]

    prefill_args = services["VllmPrefillWorker"]["extraPodSpec"]["mainContainer"]["args"]
    decode_args = services["VllmDecodeWorker"]["extraPodSpec"]["mainContainer"]["args"]

    assert _REMOVED_ROLE_FLAGS.isdisjoint(prefill_args)
    assert _REMOVED_ROLE_FLAGS.isdisjoint(decode_args)
    assert _value_after(prefill_args, "--disaggregation-mode") == "prefill"
    assert _value_after(decode_args, "--disaggregation-mode") == "decode"
    assert "--kv-transfer-config" in prefill_args
    assert "--kv-transfer-config" in decode_args


def test_vllm_disaggregated_launch_artifacts_use_disaggregation_mode():
    artifacts = _render()
    run_scripts = "\n".join(content for name, content in artifacts.items() if name.startswith("run_"))
    sflow = artifacts["sflow.yaml"]

    for content in (run_scripts, sflow):
        assert "--is-prefill-worker" not in content
        assert "--is-decode-worker" not in content
        assert content.count("--disaggregation-mode prefill") == 1
        assert content.count("--disaggregation-mode decode") == 1
        assert content.count("--kv-transfer-config") >= 2

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prove the typed k8s builders emit the four hardware/transport
pod facts (nodeSelector, tolerations, mainContainer.env, sharedMemory) correctly
per hardware profile, end-to-end through ``generate_backend_artifacts``."""

import copy
from types import SimpleNamespace

import yaml

from aiconfigurator.generator.api import generate_backend_artifacts
from tests.baseline.canary import CANARY_CASES


def _services(name):
    c = next(c for c in CANARY_CASES if c.name == name)
    arts = generate_backend_artifacts(copy.deepcopy(c.params), c.backend, backend_version=c.backend_version)
    doc = [d for d in yaml.safe_load_all(arts["k8s_deploy.yaml"]) if d and d.get("kind") == "DynamoGraphDeployment"][0]
    return doc["spec"]["services"]


def _worker(svcs):
    return next(v for k, v in svcs.items() if "frontend" not in k.lower())


def _frontend(svcs):
    return next(v for k, v in svcs.items() if "frontend" in k.lower())


def test_h200_worker_nodeselector_env_shm():
    s = _worker(_services("vllm_dense_agg"))
    assert s["extraPodSpec"]["nodeSelector"]["nvidia.com/gpu.product"] == "NVIDIA-H200"
    env = {e["name"]: e["value"] for e in s["extraPodSpec"]["mainContainer"]["env"]}
    assert env.get("NCCL_CUMEM_ENABLE") == "1"
    assert s["sharedMemory"]["size"] == "64Gi"


def test_gb200_worker_arm64_and_mnnvl_env():
    s = _worker(_services("deepseek_sglang_gb200_agg"))
    assert s["extraPodSpec"]["nodeSelector"]["kubernetes.io/arch"] == "arm64"
    env = {e["name"]: e["value"] for e in s["extraPodSpec"]["mainContainer"]["env"]}
    assert env.get("NCCL_MNNVL_ENABLE") == "1"
    assert any(t["key"] == "kubernetes.io/arch" for t in s["extraPodSpec"]["tolerations"])


def test_frontend_gets_nodeselector_tolerations_but_no_worker_only_facts():
    # Frontend carries placement (nodeSelector + tolerations) but NOT the
    # worker-only facts (mainContainer.env / sharedMemory).
    fe = _frontend(_services("vllm_dense_agg"))
    assert fe["extraPodSpec"]["nodeSelector"]["nvidia.com/gpu.product"] == "NVIDIA-H200"
    assert fe["extraPodSpec"]["tolerations"]
    assert "env" not in fe["extraPodSpec"]["mainContainer"]
    assert "sharedMemory" not in fe


def test_gb200_disagg_decode_gets_128gi_prefill_64gi():
    svcs = _services("deepseek_trtllm_gb200_disagg")
    prefill = next(v for k, v in svcs.items() if "prefill" in k.lower())
    decode = next(v for k, v in svcs.items() if "decode" in k.lower())
    assert prefill["sharedMemory"]["size"] == "64Gi"
    assert decode["sharedMemory"]["size"] == "128Gi"


def test_h100_empty_tolerations_not_emitted():
    # h100's tolerations fact is [] -> emit nothing (None), never an empty key.
    # No h100 canary exists today, so assert the builder rule directly via a
    # fact stub through the builder.
    from aiconfigurator.generator.builders.k8s_builder import build_dgd

    facts = SimpleNamespace(
        hardware={"node_selector": {"x": "y"}, "tolerations": [], "nccl_env": {}},
        transport={"env": {}},
    )

    ctx = {
        "name": "t",
        "K8sConfig": {"k8s_namespace": "ns"},
        "DynConfig": {"mode": "agg"},
        "ServiceConfig": {"model_path": "m", "served_model_name": "m"},
        "agg_gpu": 1,
        "agg_workers": 1,
        "agg_cli_args_list": [],
    }
    docs = build_dgd(ctx, "vllm", resolved_facts=facts)
    svcs = docs[0].to_dict()["spec"]["services"]
    for svc in svcs.values():
        assert "tolerations" not in svc.get("extraPodSpec", {})

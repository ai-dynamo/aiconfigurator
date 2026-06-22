# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EPD phases E3 (vllm) + E4 (sglang): the encode worker renders into the k8s DGD."""
from __future__ import annotations

import yaml

from aiconfigurator.generator.api import generate_backend_artifacts
from aiconfigurator.generator.request import (
    BackendSpec,
    GeneratorRequest,
    ModelSpec,
    RoleSizing,
    Topology,
    to_legacy_params,
)


def _params(backend: str, mode: str):
    roles = {"encode": RoleSizing(tensor_parallel_size=1, extra={"gpus_per_worker": 1, "chat_template": "qwen2-vl"})}
    workers = {"encode": 1}
    if mode == "disagg":
        roles["prefill"] = RoleSizing(tensor_parallel_size=2, extra={"gpus_per_worker": 2})
        roles["decode"] = RoleSizing(tensor_parallel_size=2, extra={"gpus_per_worker": 2})
        workers.update({"prefill": 1, "decode": 1})
    else:
        roles["agg"] = RoleSizing(tensor_parallel_size=1, extra={"gpus_per_worker": 1})
        workers["agg"] = 1
    req = GeneratorRequest(
        model=ModelSpec(model_path="Qwen/Qwen2.5-VL-7B"),
        backend=BackendSpec(name=backend),
        topology=Topology(mode=mode, roles=roles, workers=workers),
    )
    assert req.validate() == []
    return to_legacy_params(req)


def _dgd(arts):
    for doc in yaml.safe_load_all(arts["k8s_deploy.yaml"]):
        if doc and doc.get("kind") == "DynamoGraphDeployment":
            return doc
    raise AssertionError("no DGD doc")


def _args(svc):
    return "\n".join(svc["extraPodSpec"]["mainContainer"]["args"])


def test_vllm_epd_3stage():
    arts = generate_backend_artifacts(_params("vllm", "disagg"), "vllm")
    svcs = _dgd(arts)["spec"]["services"]
    assert "VllmEncodeWorker" in svcs
    enc = _args(svcs["VllmEncodeWorker"])
    assert "--multimodal-encode-worker" in enc and "--enable-multimodal" in enc
    pf = _args(svcs["VllmPrefillWorker"])
    assert "--route-to-encoder" in pf and "--enable-multimodal" in pf and "--enable-mm-embeds" in pf
    dc = _args(svcs["VllmDecodeWorker"])
    assert "--enable-multimodal" in dc and "--enable-mm-embeds" in dc and "--route-to-encoder" not in dc


def test_sglang_epd_2stage():
    arts = generate_backend_artifacts(_params("sglang", "agg"), "sglang")
    svcs = _dgd(arts)["spec"]["services"]
    assert "SGLangEncodeWorker" in svcs
    enc = _args(svcs["SGLangEncodeWorker"])
    assert "--multimodal-encode-worker" in enc and "--skip-tokenizer-init" in enc
    assert '--chat-template "qwen2-vl"' in enc
    pd = _args(svcs["SGLangWorker"])
    assert "--multimodal-worker" in pd and "--disaggregation-transfer-backend nixl" in pd


def test_sglang_epd_chat_template_defaults_when_unset():
    """The sglang encode worker requires --chat-template (crashes KeyError: None
    without it). It must be emitted even when the user does not set one — default
    to dynamo's Qwen-VL E/PD default 'qwen2-vl'."""
    roles = {
        "encode": RoleSizing(tensor_parallel_size=1, extra={"gpus_per_worker": 1}),  # no chat_template
        "agg": RoleSizing(tensor_parallel_size=1, extra={"gpus_per_worker": 1}),
    }
    req = GeneratorRequest(
        model=ModelSpec(model_path="Qwen/Qwen3-VL-2B-Instruct"),
        backend=BackendSpec(name="sglang"),
        topology=Topology(mode="agg", roles=roles, workers={"encode": 1, "agg": 1}),
    )
    arts = generate_backend_artifacts(to_legacy_params(req), "sglang")
    enc = _args(_dgd(arts)["spec"]["services"]["SGLangEncodeWorker"])
    assert '--chat-template "qwen2-vl"' in enc

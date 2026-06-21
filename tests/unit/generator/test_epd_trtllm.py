# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EPD phase E2 (trtllm): the encode worker renders into the k8s DGD.

No dynamo k8s EPD recipe exists to match; these assert the synthesized encode
worker shape (subComponentType + launch-script command flags).
"""
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


def _epd_params(mode: str):
    roles = {
        "encode": RoleSizing(
            tensor_parallel_size=1,
            extra={"gpus_per_worker": 1, "allowed_local_media_path": "/tmp/media", "max_file_size_mb": 50},
        )
    }
    workers = {"encode": 1}
    if mode == "disagg":
        roles["prefill"] = RoleSizing(tensor_parallel_size=4, extra={"gpus_per_worker": 4})
        roles["decode"] = RoleSizing(tensor_parallel_size=4, extra={"gpus_per_worker": 4})
        workers.update({"prefill": 1, "decode": 2})
    else:
        roles["agg"] = RoleSizing(tensor_parallel_size=4, extra={"gpus_per_worker": 4})
        workers["agg"] = 2
    req = GeneratorRequest(
        model=ModelSpec(model_path="Qwen/Qwen2.5-VL-7B"),
        backend=BackendSpec(name="trtllm"),
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


def test_trtllm_epd_3stage_encode_worker():
    arts = generate_backend_artifacts(_epd_params("disagg"), "trtllm", backend_version="1.2.0rc5")
    svcs = _dgd(arts)["spec"]["services"]
    assert "TRTLLMEncodeWorker" in svcs
    enc = svcs["TRTLLMEncodeWorker"]
    assert enc["subComponentType"] == "encode"
    enc_s = _args(enc)
    assert "--disaggregation-mode encode" in enc_s
    assert "--modality multimodal" in enc_s
    assert '--allowed-local-media-path "/tmp/media"' in enc_s
    assert "--max-file-size-mb 50" in enc_s
    # prefill is the encode entry point: modality + encode-endpoint
    pf = _args(svcs["TRTLLMPrefillWorker"])
    assert "--modality multimodal" in pf and "--encode-endpoint" in pf
    # decode carries modality (no encode-endpoint)
    dc = _args(svcs["TRTLLMDecodeWorker"])
    assert "--modality multimodal" in dc and "--encode-endpoint" not in dc


def test_trtllm_epd_2stage_encode_worker():
    arts = generate_backend_artifacts(_epd_params("agg"), "trtllm", backend_version="1.2.0rc5")
    svcs = _dgd(arts)["spec"]["services"]
    assert "TRTLLMEncodeWorker" in svcs
    agg = _args(svcs["TRTLLMWorker"])
    assert "--modality multimodal" in agg and "--encode-endpoint" in agg

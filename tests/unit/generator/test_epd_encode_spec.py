# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EPD: first-class EncodeSpec lowers into the encode role params."""

from __future__ import annotations

import yaml

from aiconfigurator.generator.api import generate_backend_artifacts
from aiconfigurator.generator.request import (
    BackendSpec,
    EncodeSpec,
    GeneratorRequest,
    ModelSpec,
    RoleSizing,
    Topology,
    to_legacy_params,
)


def _req(encode_spec=None, encode_extra=None):
    return GeneratorRequest(
        model=ModelSpec(model_path="Qwen/Qwen2.5-VL-7B"),
        backend=BackendSpec(name="trtllm"),
        topology=Topology(
            mode="disagg",
            roles={
                "encode": RoleSizing(tensor_parallel_size=1, extra={"gpus_per_worker": 1, **(encode_extra or {})}),
                "prefill": RoleSizing(tensor_parallel_size=4, extra={"gpus_per_worker": 4}),
                "decode": RoleSizing(tensor_parallel_size=4, extra={"gpus_per_worker": 4}),
            },
            workers={"encode": 1, "prefill": 1, "decode": 1},
        ),
        encode=encode_spec,
    )


def _encode_args(params):
    arts = generate_backend_artifacts(params, "trtllm", backend_version="1.2.0rc5")
    for doc in yaml.safe_load_all(arts["k8s_deploy.yaml"]):
        if doc and doc.get("kind") == "DynamoGraphDeployment":
            enc = doc["spec"]["services"]["TRTLLMEncodeWorker"]
            return "\n".join(enc["extraPodSpec"]["mainContainer"]["args"])
    raise AssertionError("no encode worker")


def test_encode_spec_lowers_to_named_params_and_artifact():
    req = _req(EncodeSpec(allowed_local_media_path="/data", max_file_size_mb=100, modality="multimodal"))
    params = to_legacy_params(req)
    enc = params["params"]["encode"]
    assert enc["allowed_local_media_path"] == "/data"
    assert enc["max_file_size_mb"] == 100
    args = _encode_args(params)
    assert '--allowed-local-media-path "/data"' in args
    assert "--max-file-size-mb 100" in args
    assert "--modality multimodal" in args


def test_role_extra_wins_over_encode_spec():
    # An explicit per-role value overrides the EncodeSpec default.
    req = _req(EncodeSpec(max_file_size_mb=999), encode_extra={"max_file_size_mb": 50})
    params = to_legacy_params(req)
    assert params["params"]["encode"]["max_file_size_mb"] == 50

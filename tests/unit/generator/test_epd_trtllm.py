# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EPD (trtllm): the encode worker renders into the k8s DGD.

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
    # The encode worker MUST carry --model-path/--served-model-name: dynamo.trtllm
    # otherwise falls back to its default model (TinyLlama) and the multimodal
    # encoder rejects the architecture. Regression guard for that silent bug.
    assert '--model-path "Qwen/Qwen2.5-VL-7B"' in enc_s
    assert '--served-model-name "Qwen/Qwen2.5-VL-7B"' in enc_s
    assert "--modality multimodal" in enc_s
    assert '--allowed-local-media-path "/tmp/media"' in enc_s
    assert "--max-file-size-mb 50" in enc_s
    # The encode worker PINS its --endpoint (stable `encode` component) so prefill
    # can address it deterministically across dynamo image versions.
    import re

    m = re.search(r'--endpoint "(dyn://[^"]+\.encode\.generate)"', enc_s)
    assert m, f"encode must pin --endpoint to dyn://...encode.generate; got:\n{enc_s}"
    encode_ep = m.group(1)
    # prefill is the encode entry point: modality + the SAME encode-endpoint
    pf = _args(svcs["TRTLLMPrefillWorker"])
    assert "--modality multimodal" in pf
    assert f'--encode-endpoint "{encode_ep}"' in pf
    # decode carries modality (no encode-endpoint)
    dc = _args(svcs["TRTLLMDecodeWorker"])
    assert "--modality multimodal" in dc and "--encode-endpoint" not in dc


def test_trtllm_epd_2stage_encode_worker():
    arts = generate_backend_artifacts(_epd_params("agg"), "trtllm", backend_version="1.2.0rc5")
    svcs = _dgd(arts)["spec"]["services"]
    assert "TRTLLMEncodeWorker" in svcs
    enc_s = _args(svcs["TRTLLMEncodeWorker"])
    assert '--model-path "Qwen/Qwen2.5-VL-7B"' in enc_s
    assert '--served-model-name "Qwen/Qwen2.5-VL-7B"' in enc_s
    import re

    m = re.search(r'--endpoint "(dyn://[^"]+\.encode\.generate)"', enc_s)
    assert m, f"encode must pin --endpoint; got:\n{enc_s}"
    encode_ep = m.group(1)
    agg = _args(svcs["TRTLLMWorker"])
    assert "--modality multimodal" in agg
    assert f'--encode-endpoint "{encode_ep}"' in agg


def test_trtllm_epd_single_pod_colocated_artifacts():
    """EPD also emits a single-pod (colocated) launch script + Pod: trtllm's
    image-URL embedding transfer uses CUDA IPC, which needs encode + prefill to
    share GPU memory — impossible across GPU-isolated DGD pods, so EPD runs as
    one Pod with overlapping GPUs."""
    arts = generate_backend_artifacts(_epd_params("disagg"), "trtllm", backend_version="1.2.0rc5")
    assert "epd_run.sh" in arts and "epd_pod.yaml" in arts

    run = arts["epd_run.sh"]
    # encode pinned on GPU 0 with a pinned --endpoint; prefill colocates on GPU 0
    # (BASE starts at 0) and points --encode-endpoint at the same value.
    assert "CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.trtllm" in run
    assert "--disaggregation-mode encode" in run
    assert "--encode-endpoint" in run and "--disaggregation-mode prefill" in run
    # decode is placed AFTER all prefill GPUs (true cross-GPU P->D disagg)
    assert "DECODE_OFFSET=$(( PREFILL_WORKERS * PREFILL_GPU ))" in run

    pod_docs = [d for d in yaml.safe_load_all(arts["epd_pod.yaml"]) if d]
    kinds = [d.get("kind") for d in pod_docs]
    assert kinds == ["ConfigMap", "Pod", "Service"]
    cm, pod, svc = pod_docs
    # the frontend Service exposes the in-pod frontend (selector matches the pod)
    assert svc["spec"]["selector"]["app"] == pod["metadata"]["labels"]["app"]
    assert svc["spec"]["ports"][0]["port"] == 8000
    # engine configs + launch script are mounted via the ConfigMap
    assert "encode_config.yaml" in cm["data"] and "epd_run.sh" in cm["data"]
    assert "max_batch_size: 4" in cm["data"]["encode_config.yaml"]
    # pod-local etcd + nats sidecars + the worker container
    names = [c["name"] for c in pod["spec"]["containers"]]
    assert {"nats", "etcd", "main"} <= set(names)
    main = next(c for c in pod["spec"]["containers"] if c["name"] == "main")
    # encode colocates with prefill (no extra GPU) -> prefill_workers*tp +
    # decode_workers*tp = 1*4 + 2*4 = 12 for this fixture.
    assert main["resources"]["limits"]["nvidia.com/gpu"] == "12"
    assert main["command"] == ["/bin/bash", "/cfg/epd_run.sh"]

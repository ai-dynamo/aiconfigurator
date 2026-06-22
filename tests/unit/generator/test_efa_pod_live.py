# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live (end-to-end) checks that transport=efa emits worker-only pod requirements."""

import copy

import yaml

from aiconfigurator.generator.api import generate_backend_artifacts
from tests.baseline.canary import CANARY_CASES


def _doc(name):
    c = next(c for c in CANARY_CASES if c.name == name)
    arts = generate_backend_artifacts(copy.deepcopy(c.params), c.backend, backend_version=c.backend_version)
    return [d for d in yaml.safe_load_all(arts["k8s_deploy.yaml"]) if d and d.get("kind") == "DynamoGraphDeployment"][0]


def _svc(doc, frontend):
    items = doc["spec"]["services"].items()
    return next(v for k, v in items if ("frontend" in k.lower()) == frontend)


def test_efa_worker_has_pod_requirements():
    w = _svc(_doc("deepseek_sglang_gb200_efa"), frontend=False)
    assert w["extraPodSpec"]["hostIPC"] is True
    sc = w["extraPodSpec"]["mainContainer"]["securityContext"]
    assert sc["privileged"] is True and "IPC_LOCK" in sc["capabilities"]["add"]
    gpu = w["resources"]["limits"]["gpu"]
    assert w["resources"]["limits"]["custom"]["vpc.amazonaws.com/efa"] == gpu
    env = {e["name"] for e in w["extraPodSpec"]["mainContainer"]["env"]}
    assert any(n.startswith("FI_") for n in env)


def test_efa_frontend_has_no_pod_requirements():
    f = _svc(_doc("deepseek_sglang_gb200_efa"), frontend=True)
    assert "hostIPC" not in f["extraPodSpec"]
    assert "securityContext" not in f["extraPodSpec"].get("mainContainer", {})
    assert "custom" not in (f.get("resources", {}).get("limits", {}))

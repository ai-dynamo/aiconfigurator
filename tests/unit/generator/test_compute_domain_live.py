# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live (end-to-end) test that a multinode deployment emits the ComputeDomain CRD
+ per-worker resourceClaims + multinode.nodeCount, while the frontend stays clean."""

import copy

import yaml

from aiconfigurator.generator.api import generate_backend_artifacts
from tests.baseline.canary import CANARY_CASES


def _docs(name):
    c = next(c for c in CANARY_CASES if c.name == name)
    arts = generate_backend_artifacts(copy.deepcopy(c.params), c.backend, backend_version=c.backend_version)
    return list(yaml.safe_load_all(arts["k8s_deploy.yaml"]))


def test_multinode_worker_and_compute_domain_doc():
    docs = _docs("deepseek_trtllm_b200_multinode")
    dgd = [d for d in docs if d and d.get("kind") == "DynamoGraphDeployment"][0]
    cd = [d for d in docs if d and d.get("kind") == "ComputeDomain"]
    assert cd, "ComputeDomain doc must be emitted for multinode"
    # numNodes=0 is the intentional DRA on-demand value — the driver sizes the
    # domain as pods schedule. Do NOT "fix" this to a non-zero value.
    assert cd[0]["spec"]["numNodes"] == 0, "numNodes must be 0 (DRA on-demand mode)"
    chan = cd[0]["spec"]["channel"]["resourceClaimTemplate"]["name"]
    mn_workers = [v for k, v in dgd["spec"]["services"].items() if v.get("multinode")]
    assert mn_workers, "at least one multinode worker"
    w = mn_workers[0]
    assert w["multinode"]["nodeCount"] >= 2
    assert w["resources"]["claims"][0]["name"] == "compute-domain-channel"
    assert w["extraPodSpec"]["resourceClaims"][0]["resourceClaimTemplateName"] == chan
    # frontend has no claims/multinode
    fe = next(v for k, v in dgd["spec"]["services"].items() if "frontend" in k.lower())
    assert "multinode" not in fe and "resourceClaims" not in fe.get("extraPodSpec", {})

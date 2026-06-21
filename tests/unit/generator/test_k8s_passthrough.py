# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional user K8s passthrough (extra_env / *_extra_pod_spec).

These are additive and presence-guarded: absent -> output byte-identical to the
baseline; present -> the values land on the worker/frontend services.
"""
from __future__ import annotations

import copy

import pytest
import yaml

from aiconfigurator.generator.api import generate_backend_artifacts
from tests.baseline.canary import CANARY_CASES

BACKENDS = ["trtllm", "vllm", "sglang"]


def _agg_case(backend):
    return next(c for c in CANARY_CASES if c.backend == backend and "agg" in c.name)


def _gen(params, case):
    return generate_backend_artifacts(
        copy.deepcopy(params), case.backend, backend_version=case.backend_version
    )


def _dgd(arts):
    # First YAML doc that is a DynamoGraphDeployment.
    for doc in yaml.safe_load_all(arts["k8s_deploy.yaml"]):
        if doc and doc.get("kind") == "DynamoGraphDeployment":
            return doc
    raise AssertionError("no DGD doc found")


@pytest.mark.parametrize("backend", BACKENDS)
def test_absent_passthrough_is_byte_identical(backend):
    case = _agg_case(backend)
    base = _gen(case.params, case)["k8s_deploy.yaml"]
    # Setting the keys to empty/None must not change anything either.
    p = copy.deepcopy(case.params)
    p.setdefault("K8sConfig", {}).update(
        {"extra_env": None, "worker_extra_pod_spec": None, "frontend_extra_pod_spec": {}}
    )
    assert _gen(p, case)["k8s_deploy.yaml"] == base


@pytest.mark.parametrize("backend", BACKENDS)
def test_extra_env_lands_on_worker(backend):
    case = _agg_case(backend)
    p = copy.deepcopy(case.params)
    p.setdefault("K8sConfig", {})["extra_env"] = [{"name": "NCCL_SOCKET_IFNAME", "value": "eth0"}]
    dgd = _dgd(_gen(p, case))
    workers = [s for s in dgd["spec"]["services"].values() if s.get("componentType") == "worker"]
    assert workers, "expected at least one worker service"
    for w in workers:
        env = w["extraPodSpec"]["mainContainer"].get("env", [])
        assert {"name": "NCCL_SOCKET_IFNAME", "value": "eth0"} in env
    # frontend must NOT receive extra_env
    fe = [s for s in dgd["spec"]["services"].values() if s.get("componentType") == "frontend"]
    for f in fe:
        env = (f.get("extraPodSpec", {}).get("mainContainer", {}) or {}).get("env", []) or []
        assert {"name": "NCCL_SOCKET_IFNAME", "value": "eth0"} not in env


@pytest.mark.parametrize("backend", BACKENDS)
def test_worker_extra_pod_spec_merges(backend):
    case = _agg_case(backend)
    p = copy.deepcopy(case.params)
    p.setdefault("K8sConfig", {})["worker_extra_pod_spec"] = {
        "nodeSelector": {"disktype": "ssd"},
        "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists"}],
    }
    dgd = _dgd(_gen(p, case))
    workers = [s for s in dgd["spec"]["services"].values() if s.get("componentType") == "worker"]
    assert workers
    for w in workers:
        eps = w["extraPodSpec"]
        # User key is merged in (additively) without clobbering any
        # fact-derived nodeSelector entries (e.g. the GPU product label).
        assert eps["nodeSelector"]["disktype"] == "ssd"
        assert {"key": "nvidia.com/gpu", "operator": "Exists"} in eps["tolerations"]

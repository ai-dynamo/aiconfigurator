# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""P3 wiring: api.generate_from_request and request.from_cli."""
from __future__ import annotations

import argparse
import copy
import dataclasses

import pytest

from aiconfigurator.generator.api import generate_backend_artifacts, generate_from_request
from aiconfigurator.generator.request import from_cli, from_legacy_params, to_legacy_params
from tests.baseline.canary import CANARY_CASES


@pytest.mark.parametrize("name", ["trtllm_dense_agg", "vllm_moe_disagg", "deepseek_sglang_gb200_agg"])
def test_generate_from_request_matches_direct(name):
    case = next(c for c in CANARY_CASES if c.name == name)
    req = from_legacy_params(copy.deepcopy(case.params), backend=case.backend)
    # carry the template version through the request (the artifact-version intent)
    req = dataclasses.replace(
        req, backend=dataclasses.replace(req.backend, generated_config_version=case.backend_version)
    )
    arts = generate_from_request(req)
    expected = generate_backend_artifacts(
        to_legacy_params(req), case.backend, backend_version=case.backend_version,
        deployment_target="dynamo-j2",
    )
    assert arts == expected


@pytest.mark.parametrize(
    "name", ["trtllm_dense_agg", "deepseek_trtllm_b200_disagg", "deepseek_sglang_gb200_agg"]
)
def test_save_results_render_path_is_equivalent(name):
    """P5a: the rewired `default`-mode render step (from_legacy_params(cfg) ->
    generate_from_request) produces artifacts byte-identical to the old direct
    generate_backend_artifacts(cfg, ...) call."""
    case = next(c for c in CANARY_CASES if c.name == name)
    cfg = copy.deepcopy(case.params)  # stands in for the dict-bridge output

    # NEW path (exactly what save_results now does)
    req = from_legacy_params(cfg, backend=case.backend)
    req = dataclasses.replace(
        req,
        backend=dataclasses.replace(req.backend, generated_config_version=case.backend_version),
        emit=dataclasses.replace(req.emit, deployment_target="dynamo-j2", output_dir=None),
    )
    arts_new = generate_from_request(req)

    # OLD path
    arts_old = generate_backend_artifacts(
        copy.deepcopy(case.params), case.backend, backend_version=case.backend_version,
        output_dir=None, deployment_target="dynamo-j2",
    )
    assert arts_new == arts_old


def _cli_args(**kw):
    base = dict(
        model_path="Qwen/Qwen3-32B", backend="trtllm", system="h200_sxm", total_gpus=16,
        generator_dynamo_version="1.1.0", generated_config_version=None,
        generator_config=None, generator_set=None,
        namespace=None, transport=None, image_pull_secret=None, model_cache=None,
        save_dir="/tmp/out", deployment_target="dynamo-j2",
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_from_cli_maps_keep_fields_and_overrides():
    args = _cli_args(
        namespace="myns", transport="ib", model_cache="mypvc",
        generator_set=["K8sConfig.k8s_etcd_endpoints=http://etcd:2379"],
    )
    req = from_cli(args)
    assert req.model.model_path == "Qwen/Qwen3-32B"
    assert req.backend.name == "trtllm"
    assert req.backend.dynamo_version == "1.1.0"
    assert req.platform.hardware_profile == "h200_sxm"
    assert req.topology.total_gpus == 16
    assert req.emit.output_dir == "/tmp/out"
    # promoted deployment flags + --generator-set land in overrides.raw (flat Section.key)
    raw = req.overrides.raw
    assert raw["K8sConfig.k8s_namespace"] == "myns"
    assert raw["K8sConfig.transport"] == "ib"
    assert raw["K8sConfig.k8s_pvc_name"] == "mypvc"
    assert raw["K8sConfig.k8s_etcd_endpoints"] == "http://etcd:2379"
    # generator_dynamo_version is a KEEP field, not duplicated into raw
    assert "generator_dynamo_version" not in raw
    assert req.validate() == []

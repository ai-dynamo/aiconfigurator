# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GeneratorRequest schema construction, defaults, and validate()."""

from __future__ import annotations

from aiconfigurator.generator.request import (
    BackendSpec,
    GeneratorRequest,
    ModelSpec,
    RoleSizing,
    Topology,
)


def _req(**topo):
    return GeneratorRequest(
        model=ModelSpec(model_path="Qwen/Qwen3-32B"),
        backend=BackendSpec(name="trtllm"),
        topology=Topology(**topo),
    )


def test_defaults_and_schema_version():
    r = _req(total_gpus=8)
    assert r.schema_version == "v1beta1"
    assert r.topology.mode == "disagg"
    assert r.emit.deployment_target == "dynamo-j2"
    assert r.cache.pvc_name is None
    assert r.validate() == []


def test_missing_model_path_is_error():
    r = GeneratorRequest(
        model=ModelSpec(model_path=""), backend=BackendSpec(name="trtllm"), topology=Topology(total_gpus=8)
    )
    assert any("model.model_path" in e for e in r.validate())


def test_missing_sizing_source_is_error():
    # no total_gpus and no roles
    r = _req()
    assert any("total_gpus or explicit roles" in e for e in r.validate())


def test_mode_role_consistency():
    # agg mode with prefill/decode roles -> error
    r = _req(mode="agg", roles={"prefill": RoleSizing(tensor_parallel_size=1)})
    assert any("agg mode" in e for e in r.validate())
    # disagg with an unknown role
    r2 = _req(mode="disagg", roles={"weird": RoleSizing()})
    assert any("unknown topology.roles" in e for e in r2.validate())
    # valid disagg
    r3 = _req(
        mode="disagg",
        roles={"prefill": RoleSizing(tensor_parallel_size=1), "decode": RoleSizing(tensor_parallel_size=1)},
    )
    assert r3.validate() == []


def test_role_sizing_roundtrips_params():
    params = {
        "tensor_parallel_size": 4,
        "pipeline_parallel_size": 1,
        "gpus_per_worker": 4,
        "max_batch_size": 128,
        "kv_cache_dtype": "fp8",
    }
    rs = RoleSizing.from_params(params)
    assert rs.tensor_parallel_size == 4
    assert rs.extra["gpus_per_worker"] == 4
    assert rs.extra["kv_cache_dtype"] == "fp8"
    # to_params reproduces the input (order-independent)
    assert rs.to_params() == params

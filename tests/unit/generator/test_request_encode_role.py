# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EPD Phase E1: the optional `encode` role flows through the data layer.

Encode is presence-guarded: absent -> params dict byte-identical to before.
"""
from __future__ import annotations

from aiconfigurator.generator.aggregators import collect_generator_params
from aiconfigurator.generator.request import (
    BackendSpec,
    GeneratorRequest,
    ModelSpec,
    RoleSizing,
    Topology,
    to_legacy_params,
)


def test_collect_without_encode_has_no_encode_keys():
    params = collect_generator_params(
        service={"model_path": "m"}, k8s={}, agg_params={"tensor_parallel_size": 1},
        agg_workers=2, backend="trtllm",
    )
    assert "encode" not in params["params"]
    assert "encode_workers" not in params["WorkerConfig"]


def test_collect_with_encode_adds_role_and_worker_count():
    params = collect_generator_params(
        service={"model_path": "m"}, k8s={},
        prefill_params={"tensor_parallel_size": 4}, decode_params={"tensor_parallel_size": 4},
        prefill_workers=1, decode_workers=1,
        encode_params={"tensor_parallel_size": 1, "gpus_per_worker": 1}, encode_workers=1,
        backend="trtllm",
    )
    assert params["params"]["encode"] == {"tensor_parallel_size": 1, "gpus_per_worker": 1}
    assert params["WorkerConfig"]["encode_workers"] == 1
    assert params["WorkerConfig"]["encode_gpus_per_worker"] == 1


def test_request_with_encode_role_validates_and_lowers():
    req = GeneratorRequest(
        model=ModelSpec(model_path="Qwen/Qwen2.5-VL-7B"),
        backend=BackendSpec(name="trtllm"),
        topology=Topology(
            mode="disagg",
            roles={
                "encode": RoleSizing(tensor_parallel_size=1),
                "prefill": RoleSizing(tensor_parallel_size=4),
                "decode": RoleSizing(tensor_parallel_size=4),
            },
            workers={"encode": 1, "prefill": 1, "decode": 2},
        ),
    )
    assert req.validate() == []
    params = to_legacy_params(req)
    assert params["params"]["encode"]["tensor_parallel_size"] == 1
    assert params["WorkerConfig"]["encode_workers"] == 1


def test_encode_role_allowed_in_agg_mode():
    # encode + agg = 2-stage E/PD
    req = GeneratorRequest(
        model=ModelSpec(model_path="m"), backend=BackendSpec(name="sglang"),
        topology=Topology(mode="agg", roles={"agg": RoleSizing(tensor_parallel_size=1),
                                             "encode": RoleSizing(tensor_parallel_size=1)},
                          workers={"agg": 1, "encode": 1}),
    )
    assert req.validate() == []
    # prefill role in agg mode is still rejected
    bad = GeneratorRequest(
        model=ModelSpec(model_path="m"), backend=BackendSpec(name="sglang"),
        topology=Topology(mode="agg", roles={"prefill": RoleSizing()}),
    )
    assert any("agg mode" in e for e in bad.validate())

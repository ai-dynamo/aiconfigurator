# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from aiconfigurator.sdk.backends.sglang_backend import SGLANGBackend
from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend
from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("backend", "moe_coefficient", "overhead"),
    [
        (TRTLLMBackend(), 22, 1.0),
        (VLLMBackend(), 22, 1.0),
        (SGLANGBackend(), 28, 1.15),
    ],
    ids=("trtllm", "vllm", "sglang"),
)
def test_step3p7_uses_moe_activation_and_dispatch_workspace(backend, moe_coefficient, overhead):
    """STEP3P7 must not fall through to dense activation memory accounting."""
    num_tokens = 8192
    attention_width = 64 * 128
    residual_width = 4096
    model = SimpleNamespace(
        context_ops=[],
        model_family="STEP3P7",
        _num_heads=64,
        _head_size=128,
        _hidden_size=residual_width,
        _num_experts=288,
        _topk=8,
        config=SimpleNamespace(
            pp_size=1,
            tp_size=1,
            attention_dp_size=8,
            moe_ep_size=8,
            nextn=0,
        ),
        get_kvcache_bytes_per_sequence=lambda _seq: 0,
        _cp_kv_memory_divisor=lambda: 1,
    )
    database = SimpleNamespace(system_spec={"misc": {"nccl_mem": {1: 0}, "other_mem": 0}})

    memory = backend._get_memory_usage(
        model,
        database,
        batch_size=1,
        beam_width=1,
        isl=num_tokens,
        osl=1,
    )

    base_activation = 2 * num_tokens * attention_width * moe_coefficient
    dispatch_workspace = (
        num_tokens
        * residual_width
        * model.config.attention_dp_size
        * model._num_experts
        * model._topk
        / model.config.moe_ep_size
        / 128
        * 4
    )
    expected_activation_gib = (base_activation + dispatch_workspace) * overhead / (1 << 30)

    assert memory["activations"] == pytest.approx(expected_activation_gib)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MTP activation scaling must apply to the decode-token share only on mixed steps.

Aggregated serving processes context and decode tokens in one step
(``num_tokens = ctx_tokens + num_gen_requests``). Speculative decoding verifies
``nextn + 1`` tokens per *decode* request; context tokens are processed once.
Multiplying the whole activation footprint by ``(nextn + 1)`` models
``(nextn+1) x (context + decode)`` instead of ``context + (nextn+1) x decode``,
which at long ISL inflates activations ~(nextn+1)x and over-prunes concurrency
(observed: GLM-5.2 agg recommendation pinned at concurrency 8 vs 64 measured).

Decode-only steps (disagg decode workers, ``mtp_scaled_tokens=None``) keep the
full multiplier: every token in those steps is part of verification.
"""

import pytest

from aiconfigurator_core.sdk import config as sdk_config
from aiconfigurator_core.sdk.models import get_model


GPTOSS = "openai/gpt-oss-120b"  # any registered model works; we only need activations


def _memory(backend, model, database, *, nextn, num_tokens, mtp_scaled_tokens):
    model.config.nextn = nextn
    return backend._get_memory_usage(
        model,
        database,
        batch_size=4,
        beam_width=1,
        isl=65_536,
        osl=400,
        num_tokens=num_tokens,
        mtp_scaled_tokens=mtp_scaled_tokens,
    )["activations"]


@pytest.fixture
def setup():
    from types import SimpleNamespace

    from aiconfigurator_core.sdk.backends.factory import get_backend

    model_config = sdk_config.ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1)
    model = get_model(GPTOSS, model_config, "vllm")
    backend = get_backend("vllm")
    database = SimpleNamespace(
        backend="vllm",
        version="test-version",
        system="b200_sxm",
        system_spec={
            "gpu": {"mem_capacity": 180 * (1 << 30)},
            "misc": {"nccl_mem": {1: 0, 2: 0, 4: 0, 8: 0}, "other_mem": 0},
        },
    )
    return backend, model, database


def test_context_dominated_step_barely_scales(setup):
    """65,536 ctx + 3 decode tokens: (nextn+1) on the whole step is ~4x too big."""
    backend, model, database = setup
    num_tokens = 65_536 + 3
    base = _memory(backend, model, database, nextn=0, num_tokens=num_tokens, mtp_scaled_tokens=3)
    spec = _memory(backend, model, database, nextn=3, num_tokens=num_tokens, mtp_scaled_tokens=3)
    # Correct scaling: (65536 + 3*4) / 65539 ~= 1.00014, nowhere near 4x.
    assert spec / base == pytest.approx((65_536 + 3 * 4) / num_tokens, rel=1e-6)
    assert spec / base < 1.01


def test_decode_only_step_keeps_full_multiplier(setup):
    """mtp_scaled_tokens=None (disagg decode worker): full (nextn+1) applies."""
    backend, model, database = setup
    base = _memory(backend, model, database, nextn=0, num_tokens=512, mtp_scaled_tokens=None)
    spec = _memory(backend, model, database, nextn=3, num_tokens=512, mtp_scaled_tokens=None)
    assert spec / base == pytest.approx(4.0, rel=1e-6)


def test_all_decode_share_equals_full_multiplier(setup):
    """mtp_scaled_tokens == num_tokens degenerates to the old behavior."""
    backend, model, database = setup
    base = _memory(backend, model, database, nextn=0, num_tokens=512, mtp_scaled_tokens=512)
    spec = _memory(backend, model, database, nextn=3, num_tokens=512, mtp_scaled_tokens=512)
    assert spec / base == pytest.approx(4.0, rel=1e-6)


def test_prefill_only_step_does_not_scale(setup):
    """mtp_scaled_tokens=0 (static_ctx / disagg prefill worker): no draft share."""
    backend, model, database = setup
    base = _memory(backend, model, database, nextn=0, num_tokens=65_536, mtp_scaled_tokens=0)
    spec = _memory(backend, model, database, nextn=3, num_tokens=65_536, mtp_scaled_tokens=0)
    assert spec == pytest.approx(base, rel=1e-9)

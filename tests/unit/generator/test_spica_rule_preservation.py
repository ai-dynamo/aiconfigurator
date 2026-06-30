# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy

import pytest

from aiconfigurator.generator.rendering.rule_engine import apply_rule_plugins
from aiconfigurator.generator.request import from_legacy_params, to_legacy_params


def _params(*, preserve: bool, rule: str | None = None) -> dict:
    params = {
        "preserve_engine_limits": preserve,
        "ServiceConfig": {},
        "DynConfig": {"mode": "disagg"},
        "WorkerConfig": {"prefill_workers": 1, "decode_workers": 1},
        "SlaConfig": {"isl": 128, "osl": 16},
        "ModelConfig": {"is_moe": False, "prefix": 0, "nextn": 0},
        "params": {
            "prefill": {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "data_parallel_size": 1,
                "max_batch_size": 128,
                "max_num_tokens": 32768,
                "max_prefill_tokens": 32768,
                "max_seq_len": 8192,
                "tokens_per_block": 64,
                "cache_transceiver_max_tokens_in_buffer": 32768,
            },
            "decode": {
                "tensor_parallel_size": 2,
                "pipeline_parallel_size": 1,
                "data_parallel_size": 1,
                "max_batch_size": 256,
                "max_num_tokens": 8192,
                "max_prefill_tokens": 8192,
                "max_seq_len": 8192,
                "tokens_per_block": 64,
                "cache_transceiver_max_tokens_in_buffer": 8192,
            },
        },
    }
    if rule is not None:
        params["rule"] = rule
    return params


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
@pytest.mark.parametrize("rule", [None, "benchmark"])
def test_spica_preserves_evaluated_engine_limits(backend: str, rule: str | None):
    params = _params(preserve=True, rule=rule)

    result = apply_rule_plugins(copy.deepcopy(params), backend)

    for role in ("prefill", "decode"):
        expected = params["params"][role]
        actual = result["params"][role]
        for key in (
            "max_batch_size",
            "max_num_tokens",
            "max_prefill_tokens",
            "max_seq_len",
            "cache_transceiver_max_tokens_in_buffer",
        ):
            assert actual[key] == expected[key]


def test_default_vllm_rules_still_resize_non_spica_requests():
    result = apply_rule_plugins(_params(preserve=False), "vllm")

    assert result["params"]["prefill"]["max_num_tokens"] == 1628
    assert result["params"]["decode"]["max_batch_size"] == 512
    assert result["params"]["decode"]["max_num_tokens"] == 512


def test_preserve_engine_limits_survives_typed_request_round_trip():
    params = _params(preserve=True)

    result = to_legacy_params(from_legacy_params(params, backend="vllm"))

    assert result["preserve_engine_limits"] is True

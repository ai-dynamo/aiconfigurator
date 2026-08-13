# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from aiconfigurator.sdk.config_adapter import (
    AdapterOverrides,
    InferenceXSource,
    adapt_config,
    to_cli_estimate_kwargs,
)

pytestmark = pytest.mark.unit


def _config(**overrides):
    value = {
        "config_id": 7,
        "hardware": "h200",
        "framework": "vllm",
        "silicon_model": "llama70b",
        "precision": "fp8",
        "spec_method": "none",
        "disagg": False,
        "prefill_tp": 2,
        "prefill_ep": 1,
        "prefill_dp_attention": False,
        "prefill_num_workers": 2,
        "decode_tp": 4,
        "decode_ep": 1,
        "decode_dp_attention": False,
        "decode_num_workers": 0,
        "num_prefill_gpu": 4,
        "num_decode_gpu": 4,
    }
    value.update(overrides)
    return value


def _benchmark(**overrides):
    value = {"id": "bench-9", "isl": 1024, "osl": 128, "conc": 16}
    value.update(overrides)
    return value


def test_dense_agg_alias_quantization_and_topology():
    report = adapt_config(InferenceXSource(_config(), _benchmark()))
    outcome = report.outcomes[0]

    assert outcome.status == "adapted"
    assert outcome.request is not None
    assert outcome.request.model.path == "meta-llama/Meta-Llama-3.1-70B"
    assert outcome.request.quantization.gemm == "fp8"
    assert outcome.request.quantization.moe is None
    assert outcome.request.provenance.assumptions == (
        "InferenceX does not expose pipeline parallelism; pp_size defaults to 1.",
        "InferenceX aggregated decode_num_workers=0 is an irrelevant sentinel; replicas defaults to 1.",
        "Aggregated worker replicas are omitted during cli_estimate lowering because "
        "cli_estimate has no aggregated worker-count parameter.",
    )
    assert to_cli_estimate_kwargs(outcome.request)["batch_size"] == 16


def test_moe_disagg_backend_folding_and_worker_arithmetic():
    report = adapt_config(
        InferenceXSource(
            _config(
                framework="dynamo-sglang",
                silicon_model="minimaxm2.5",
                disagg=True,
                prefill_tp=2,
                prefill_ep=2,
                prefill_num_workers=2,
                num_prefill_gpu=4,
                decode_tp=4,
                decode_ep=2,
                decode_num_workers=2,
                num_decode_gpu=8,
            ),
            _benchmark(conc=16),
        )
    )
    request = report.requests[0]
    kwargs = to_cli_estimate_kwargs(request)

    assert request.backend.name == "sglang"
    assert request.quantization.moe == "fp8_block"
    assert kwargs["prefill_num_workers"] == 2
    assert kwargs["prefill_batch_size"] == 1
    assert kwargs["decode_num_workers"] == 2
    assert kwargs["decode_batch_size"] == 8
    assert kwargs["decode_moe_tp_size"] == 2
    assert kwargs["decode_moe_ep_size"] == 2


@pytest.mark.parametrize(
    ("config", "concurrency", "message"),
    [
        (_config(hardware="amd-mi300x"), 16, "hardware"),
        (_config(silicon_model="unknown"), 16, "model/precision"),
        (_config(decode_tp=2, num_decode_gpu=4), 3, "divisible"),
        (
            _config(framework="dynamo-trtllm", decode_tp=2, num_decode_gpu=4),
            4,
            "does not match GPUs per worker",
        ),
        (_config(decode_num_workers=-1), 16, "must be positive"),
        (_config(disagg=True, prefill_num_workers=0, decode_num_workers=1), 16, "must be positive"),
    ],
)
def test_invalid_sources_are_rejected(config, concurrency, message):
    outcome = adapt_config(InferenceXSource(config, _benchmark(conc=concurrency))).outcomes[0]

    assert outcome.status == "rejected"
    assert message in outcome.diagnostics[-1].message


def test_mtp_requires_acceptance_and_can_be_explicitly_adapted():
    source = InferenceXSource(_config(spec_method="mtp"), _benchmark())
    rejected = adapt_config(source).outcomes[0]
    accepted = adapt_config(source, AdapterOverrides(nextn=1, nextn_accepted=0.8)).outcomes[0]

    assert rejected.status == "rejected"
    assert "nextn_accepted" in rejected.diagnostics[-1].message
    assert accepted.request is not None
    assert accepted.request.model.nextn == 1
    assert accepted.request.model.nextn_accepted == 0.8


def test_unpinned_backend_version_is_warning_not_rejection():
    outcome = adapt_config(InferenceXSource(_config(), _benchmark())).outcomes[0]

    assert outcome.status == "adapted"
    assert [diagnostic.code for diagnostic in outcome.diagnostics] == ["backend_version_unpinned"]

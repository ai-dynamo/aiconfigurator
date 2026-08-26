# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64

import numpy as np
import pytest

from collector.expert_popularity.response_capture import (
    aggregate_routed_experts,
    decode_routed_experts,
    repeat_stability,
)

pytestmark = pytest.mark.unit


def test_decode_and_aggregate_response_routed_experts():
    routed = np.asarray(
        [
            [[0, 0], [0, 2], [3, 1]],
            [[0, 0], [2, 2], [1, 0]],
            [[0, 0], [3, 0], [1, 3]],
        ],
        dtype=np.int32,
    )
    encoded = base64.b64encode(routed.tobytes()).decode("ascii")

    decoded = decode_routed_experts(encoded, prompt_tokens=3, num_layers=3, top_k=2)
    counts = aggregate_routed_experts(
        decoded,
        num_layers=3,
        num_experts=4,
        top_k=2,
        moe_layer_ids=[1, 2],
    )

    assert np.array_equal(decoded, routed)
    assert counts.tolist() == [[0, 0, 0, 0], [2, 0, 3, 1], [1, 3, 0, 2]]


def test_decode_rejects_wrong_response_size():
    encoded = base64.b64encode(np.zeros((3,), dtype=np.int32).tobytes()).decode("ascii")

    with pytest.raises(RuntimeError, match="wrong byte length"):
        decode_routed_experts(encoded, prompt_tokens=2, num_layers=3, top_k=2)


def test_decode_rejects_invalid_base64():
    with pytest.raises(RuntimeError, match="not valid base64"):
        decode_routed_experts("not-base64!", prompt_tokens=1, num_layers=1, top_k=1)


def test_aggregate_rejects_invalid_expert_ids():
    routed = np.asarray([[[0, 4]]], dtype=np.int32)

    with pytest.raises(RuntimeError, match="out-of-range"):
        aggregate_routed_experts(
            routed,
            num_layers=1,
            num_experts=4,
            top_k=2,
            moe_layer_ids=[0],
        )


def test_repeat_stability_reports_exact_and_aggregate_differences():
    baseline = np.asarray([[20, 10, 2], [4, 8, 20]], dtype=np.int64)
    candidate = np.asarray([[19, 11, 2], [4, 9, 19]], dtype=np.int64)

    exact = repeat_stability([[baseline], [baseline.copy()]], [0, 1])
    aggregate = repeat_stability([[baseline], [candidate]], [0, 1])

    assert exact["all_counts_exact"] is True
    assert exact["minimum_mean_layer_pearson"] == pytest.approx(1.0)
    assert exact["maximum_mean_layer_jensen_shannon_divergence_bits"] == pytest.approx(0.0)
    assert aggregate["all_counts_exact"] is False
    assert 0.99 < aggregate["minimum_mean_layer_pearson"] < 1.0
    assert 0.0 < aggregate["maximum_mean_layer_jensen_shannon_divergence_bits"] < 0.001
    assert aggregate["comparisons"][0]["top1_expert_match_rate"] == 1.0

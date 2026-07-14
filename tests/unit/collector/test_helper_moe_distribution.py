# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for MoE token distribution helpers in collector/helper.py.

Covers:
- _round_robin_adjust_per_rank: device preservation and exact-total invariant
- _generate_power_law_distribution: sum == num_tokens * topk, per-expert upper bound

The add_sequence_batch shim in collect_mla._run_attn_for_backend is embedded
inside a large function that takes live TRT-LLM objects; testing it in isolation
requires either extracting the dispatch to a helper or mocking the full TRT-LLM
KV-cache stack. That is left for a follow-up if the function is ever refactored.
"""

import pytest

torch = pytest.importorskip("torch")

from collector.helper import (
    _generate_power_law_distribution,
    _round_robin_adjust_per_rank,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _round_robin_adjust_per_rank
# ---------------------------------------------------------------------------


def _make_counts(rows, cols, fill=0):
    return torch.full((rows, cols), fill, dtype=torch.int64)


def test_round_robin_adjust_preserves_cpu_device():
    counts = _make_counts(2, 4)
    result = _round_robin_adjust_per_rank(
        counts, remaining=1, is_valid=lambda c: c < 10, pick_local_index=torch.argmin, step=1
    )
    assert result.device.type == "cpu"


def test_round_robin_adjust_exact_total_add():
    counts = _make_counts(2, 4)  # sum = 0
    remaining = 5
    result = _round_robin_adjust_per_rank(
        counts, remaining=remaining, is_valid=lambda c: c < 10, pick_local_index=torch.argmin, step=1
    )
    assert result.sum().item() == remaining


def test_round_robin_adjust_exact_total_subtract():
    counts = _make_counts(2, 4, fill=3)  # sum = 24
    remaining = 5
    result = _round_robin_adjust_per_rank(
        counts, remaining=remaining, is_valid=lambda c: c > 0, pick_local_index=torch.argmax, step=-1
    )
    assert result.sum().item() == 24 - remaining


def test_round_robin_adjust_zero_remaining_is_noop():
    counts = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    result = _round_robin_adjust_per_rank(
        counts, remaining=0, is_valid=lambda c: c < 10, pick_local_index=torch.argmin, step=1
    )
    assert result.equal(counts)


def test_round_robin_adjust_stops_when_no_valid_slot():
    # All slots at upper bound — remaining cannot be exhausted
    counts = _make_counts(2, 4, fill=10)
    result = _round_robin_adjust_per_rank(
        counts, remaining=3, is_valid=lambda c: c < 10, pick_local_index=torch.argmin, step=1
    )
    # No slot was valid, so sum unchanged
    assert result.sum().item() == counts.sum().item()


# ---------------------------------------------------------------------------
# _generate_power_law_distribution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_tokens, num_experts, topk, ep, alpha",
    [
        (128, 8, 2, 2, 1.5),
        (256, 16, 4, 4, 1.2),
        (64, 4, 1, 1, 2.0),
        (512, 32, 2, 8, 1.8),
    ],
)
def test_power_law_distribution_exact_sum(num_tokens, num_experts, topk, ep, alpha):
    counts, _ = _generate_power_law_distribution(num_tokens, num_experts, topk, ep, alpha)
    assert counts.sum().item() == num_tokens * topk
    rank_sums = counts.view(ep, num_experts // ep).sum(dim=1)
    assert rank_sums[0] == rank_sums.max()


@pytest.mark.parametrize(
    "num_tokens, num_experts, topk, ep, alpha",
    [
        (128, 8, 2, 2, 1.5),
        (256, 16, 4, 4, 1.2),
    ],
)
def test_power_law_distribution_per_expert_upper_bound(num_tokens, num_experts, topk, ep, alpha):
    counts, _ = _generate_power_law_distribution(num_tokens, num_experts, topk, ep, alpha)
    assert counts.max().item() <= num_tokens


@pytest.mark.parametrize(
    "num_tokens, num_experts, topk, ep, alpha",
    [
        (128, 8, 2, 2, 1.5),
        (256, 16, 4, 4, 1.2),
    ],
)
def test_power_law_distribution_length(num_tokens, num_experts, topk, ep, alpha):
    counts, _ = _generate_power_law_distribution(num_tokens, num_experts, topk, ep, alpha)
    assert len(counts) == num_experts


def test_power_law_distribution_assignment_shape():
    num_tokens, num_experts, topk, ep, alpha = 128, 8, 2, 2, 1.5
    _, assignments = _generate_power_law_distribution(num_tokens, num_experts, topk, ep, alpha)
    assert assignments.shape == (num_tokens, topk)

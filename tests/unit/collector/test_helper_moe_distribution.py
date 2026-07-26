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

import sys
from unittest.mock import MagicMock

import pytest

# test_collect_provenance_writer deliberately leaves a MagicMock cached as
# sys.modules["torch"] (collect.py's fork-worker tests depend on it), so a
# plain importorskip("torch") can "succeed" with the mock and every tensor
# assertion below dies with `TypeError: '<' not supported between instances
# of 'MagicMock' and 'int'` — whether that happens depends only on which
# module pytest-xdist imports first on the worker (same pattern as
# test_dsv4_megamoe_workload). Evict the mock, import the real torch (or
# skip when it isn't installed), then put the mock back for the siblings
# that rely on it.
_saved_mock = sys.modules.get("torch")
_restore_mock = isinstance(_saved_mock, MagicMock)
if _restore_mock:
    sys.modules.pop("torch")
try:
    import torch
except ImportError:
    if _restore_mock:
        sys.modules["torch"] = _saved_mock
    pytest.skip("real torch required for tensor operations", allow_module_level=True)
if _restore_mock:
    sys.modules["torch"] = _saved_mock

from collector.helper import (
    _generate_power_law_distribution,
    _round_robin_adjust_per_rank,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _real_torch_in_sys_modules(monkeypatch):
    """Pin the real torch for the duration of each test.

    ``collector/helper.py`` imports torch lazily inside the functions under
    test, so binding the real module at collection time is not enough — the
    call-time ``import torch`` resolves whatever sits in ``sys.modules`` at
    that moment.
    """
    monkeypatch.setitem(sys.modules, "torch", torch)


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

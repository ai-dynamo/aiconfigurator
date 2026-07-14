# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the AFD default-mode search space enumeration."""

import pytest

from aiconfigurator.sdk.task_v2 import build_afd_parallel_lists

pytestmark = pytest.mark.unit


def test_dense_candidates_respect_budget_and_divisibility():
    candidates = build_afd_parallel_lists(total_gpus=32, gpus_per_node=8, is_moe=False)
    assert candidates
    for n_a, n_f, tp_a, f_ep, mb, pipe in candidates:
        assert n_a >= 1 and n_f >= 1
        assert (n_a + n_f) * 8 <= 32
        assert 8 % tp_a == 0
        assert f_ep == 1  # dense models never shard experts
        assert mb in (2, 3, 4)
        assert pipe in ("optimistic", "conservative")


def test_moe_expert_divisibility():
    candidates = build_afd_parallel_lists(total_gpus=32, gpus_per_node=8, is_moe=True, num_experts=256)
    assert candidates
    for _n_a, n_f, _tp_a, f_ep, _mb, _pipe in candidates:
        tp_f = n_f * 8
        assert tp_f % f_ep == 0
        assert 256 % f_ep == 0


def test_partial_node_splits_are_enumerated():
    """Combined-with-PD needs headroom: splits using < all nodes must exist."""
    candidates = build_afd_parallel_lists(total_gpus=32, gpus_per_node=8, is_moe=False)
    used_nodes = {n_a + n_f for n_a, n_f, *_ in candidates}
    assert {2, 3, 4} <= used_nodes


def test_skewed_splits_are_pruned():
    candidates = build_afd_parallel_lists(total_gpus=64, gpus_per_node=8, is_moe=False)
    assert all(n_a / n_f <= 4 for n_a, n_f, *_ in candidates)


def test_search_config_controls_candidate_axes():
    candidates = build_afd_parallel_lists(
        total_gpus=32,
        gpus_per_node=8,
        is_moe=True,
        num_experts=256,
        search_config={
            "tp_a_list": [4],
            "microbatch_list": [3],
            "pipeline_model_list": ["optimistic"],
            "f_moe_ep_size_list": [1, "n_f_nodes"],
            "max_af_ratio": 3,
        },
    )

    assert candidates
    for n_a, n_f, tp_a, f_ep, mb, pipe in candidates:
        assert n_a / n_f <= 3
        assert tp_a == 4
        assert f_ep in {1, n_f}
        assert mb == 3
        assert pipe == "optimistic"


def test_search_config_errors_when_candidate_count_exceeds_limit():
    with pytest.raises(ValueError, match="max_candidates=1"):
        build_afd_parallel_lists(
            total_gpus=32,
            gpus_per_node=8,
            is_moe=False,
            search_config={"max_candidates": 1},
        )


def test_search_config_can_truncate_candidate_overflow():
    candidates = build_afd_parallel_lists(
        total_gpus=32,
        gpus_per_node=8,
        is_moe=False,
        search_config={"max_candidates": 1, "candidate_overflow": "truncate"},
    )

    assert len(candidates) == 1


def test_single_node_returns_empty():
    assert build_afd_parallel_lists(total_gpus=8, gpus_per_node=8, is_moe=True, num_experts=64) == []


def test_invalid_inputs_return_empty():
    assert build_afd_parallel_lists(total_gpus=0, gpus_per_node=8, is_moe=False) == []
    assert build_afd_parallel_lists(total_gpus=16, gpus_per_node=0, is_moe=False) == []

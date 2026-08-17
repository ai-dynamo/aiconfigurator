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


def test_skewed_splits_are_kept_by_default():
    """The A:F ratio is uncapped by default.

    This replaces an earlier assertion that every candidate satisfied a 4:1
    bound. That bound was the default, and it excluded every measured AFD
    optimum: FastAFD reports 7:1 and 11:1 for Qwen3-235B and 17:1 for
    MiniMax-M2.5 on GB200 NVL72. The cap is now opt-in.
    """
    candidates = build_afd_parallel_lists(total_gpus=64, gpus_per_node=8, is_moe=False)
    ratios = {n_a / n_f for n_a, n_f, *_ in candidates}
    assert max(ratios) == 7.0  # 8 nodes -> 7 A nodes + 1 F node


def test_max_af_ratio_still_prunes_when_set():
    candidates = build_afd_parallel_lists(
        total_gpus=64,
        gpus_per_node=8,
        is_moe=False,
        search_config={"max_af_ratio": 4},
    )
    assert candidates
    assert all(n_a / n_f <= 4 for n_a, n_f, *_ in candidates)


def test_max_af_ratio_rejects_non_positive():
    with pytest.raises(ValueError, match="max_af_ratio must be > 0 when set"):
        build_afd_parallel_lists(
            total_gpus=64,
            gpus_per_node=8,
            is_moe=False,
            search_config={"max_af_ratio": 0},
        )


def test_nvl72_domain_reaches_the_measured_optima():
    """GB200 NVL72: 18 nodes x 4 GPUs, so a single F node allows 17:1."""
    candidates = build_afd_parallel_lists(
        total_gpus=72,
        gpus_per_node=4,
        is_moe=True,
        num_experts=128,
    )
    ratios = {n_a / n_f for n_a, n_f, *_ in candidates}
    assert max(ratios) == 17.0
    # The three ratios FastAFD measured as optimal per workload.
    assert {7.0, 11.0, 17.0} <= ratios


def test_mb2_optimistic_is_enumerated():
    """mb=2 + optimistic used to be skipped outright.

    The strict K=3 occupancy bound needs mb >= 3 whenever t_c > 0, so the
    combination was pruned as a guaranteed duplicate of mb=2 + conservative.
    ``AFDConfig.comm_hiding_tolerance`` now keeps the overlapped cycle when
    the round trip is negligible against compute, which is the measured
    behavior on NVLink-class fabrics -- so the candidate has to exist.
    """
    candidates = build_afd_parallel_lists(total_gpus=32, gpus_per_node=8, is_moe=False)
    assert any(mb == 2 and pipe == "optimistic" for *_, mb, pipe in candidates)


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


def test_search_config_rejects_invalid_candidate_limit():
    with pytest.raises(ValueError, match="max_candidates must be >= 1"):
        build_afd_parallel_lists(
            total_gpus=32,
            gpus_per_node=8,
            is_moe=False,
            search_config={"max_candidates": 0},
        )


def test_search_config_rejects_invalid_overflow_policy():
    with pytest.raises(ValueError, match="candidate_overflow must be 'error' or 'truncate'"):
        build_afd_parallel_lists(
            total_gpus=32,
            gpus_per_node=8,
            is_moe=False,
            search_config={"candidate_overflow": "ignore"},
        )


def test_default_limit_covers_128_gpu_dense_search():
    candidates = build_afd_parallel_lists(total_gpus=128, gpus_per_node=8, is_moe=False)

    # Grew from 2040 when the default A:F cap was dropped (skewed splits are
    # now enumerated) and mb=2 + optimistic stopped being pruned. Still well
    # inside the 20k default limit, which is the point of this test.
    assert len(candidates) == 2880


def test_default_limit_covers_96_gpu_moe_search():
    candidates = build_afd_parallel_lists(
        total_gpus=96,
        gpus_per_node=8,
        is_moe=True,
        num_experts=256,
    )

    assert len(candidates) > 2000


def test_single_node_returns_empty():
    assert build_afd_parallel_lists(total_gpus=8, gpus_per_node=8, is_moe=True, num_experts=64) == []


def test_invalid_inputs_return_empty():
    assert build_afd_parallel_lists(total_gpus=0, gpus_per_node=8, is_moe=False) == []
    assert build_afd_parallel_lists(total_gpus=16, gpus_per_node=0, is_moe=False) == []

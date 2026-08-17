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


def test_mb2_optimistic_is_pruned():
    """``optimistic`` below mb=3 is a duplicate, so it is not enumerated.

    The K=3 occupancy bound needs ``mb >= 2 + t_c / max(t_a, t_f)``, which
    exceeds 2 whenever the round trip is non-zero. At mb=2 the session demotes
    to ``conservative`` and returns exactly the cadence the mb=2 +
    conservative candidate already carries, so enumerating both spends a sweep
    slot on a guaranteed duplicate.
    """
    candidates = build_afd_parallel_lists(total_gpus=32, gpus_per_node=8, is_moe=False)
    assert not any(mb == 2 and pipe == "optimistic" for *_, mb, pipe in candidates)
    # The pruning is specific to mb < 3 -- higher occupancies must survive it.
    assert any(mb == 3 and pipe == "optimistic" for *_, mb, pipe in candidates)
    # ...and mb=2 is still reachable through the cadence it actually gets.
    assert any(mb == 2 and pipe == "conservative" for *_, mb, pipe in candidates)


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

    # Grew from 2040 when the default A:F cap was dropped, so skewed splits are
    # now enumerated. Five of the six (mb, pipeline) pairs survive: optimistic
    # below mb=3 is pruned as a duplicate of conservative at the same mb, which
    # is what took this from 2880 to 2400. Still well inside the 20k default
    # limit, which is the point of this test.
    assert len(candidates) == 2400


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


# A config that never aborts on the candidate ceiling: these cases care about
# which topologies are admitted, not about the truncation policy.
_NO_CEILING = {"max_candidates": 10**9, "candidate_overflow": "truncate"}


class TestPerPoolNodeWidths:
    """Node width is a per-pool hardware fact.

    A single width silently conflates three jobs -- the GPU budget, ``tp_a``
    divisibility and ``tp_f`` -- so under hetero A/F the enumerator budgeted a
    footprint the candidate would not actually occupy, and no downstream check
    caught it (the rejection categories cover oom / fixed_batch / tpot /
    low_batch_oom only).
    """

    def test_budget_uses_each_pool_real_width(self):
        """gb200 A pool (4/node) with a b200_sxm F pool (8/node) under 72 GPUs.

        The single-width enumerator computed ``72 // 4 = 18`` nodes and admitted
        ``(n_a=10, n_f=8)``, a real footprint of ``10*4 + 8*8 = 104`` GPUs -- 44%
        over budget, silently.
        """
        candidates = build_afd_parallel_lists(
            72, 4, True, 256, a_gpus_per_node=4, f_gpus_per_node=8, search_config=_NO_CEILING
        )
        assert candidates, "expected some feasible topology at 72 GPUs"
        over = [(na, nf) for (na, nf, *_rest) in candidates if na * 4 + nf * 8 > 72]
        assert over == [], f"admitted over-budget topologies: {sorted(set(over))[:5]}"

    def test_the_cited_over_budget_candidate_is_rejected(self):
        candidates = build_afd_parallel_lists(
            72, 4, True, 256, a_gpus_per_node=4, f_gpus_per_node=8, search_config=_NO_CEILING
        )
        assert not any(na == 10 and nf == 8 for (na, nf, *_rest) in candidates)

    def test_tp_a_divides_the_a_pool_width(self):
        """``tp_a`` is an A-pool tensor-parallel width, so it must divide the A
        node -- taking it from the F pool's width would admit shapes the A nodes
        cannot host."""
        candidates = build_afd_parallel_lists(
            96, 8, False, 0, a_gpus_per_node=4, f_gpus_per_node=8, search_config=_NO_CEILING
        )
        assert candidates
        assert all(4 % tp_a == 0 for (_na, _nf, tp_a, *_rest) in candidates)

    def test_tp_f_uses_the_f_pool_width(self):
        """``tp_f`` is F-side ranks, so it must be ``n_f_nodes * f_width``.

        Checking only "ep divides tp_f" is too weak: the ep set produced by a
        narrower width is a subset of the wider one, so divisibility still holds
        if the implementation regresses to the A-pool width. Assert instead on
        the value the enumerator resolves for the symbolic ``"tp_f"`` candidate,
        which is exactly the quantity that has to track ``f_width``.
        """
        a_width, f_width = 4, 8
        candidates = build_afd_parallel_lists(
            96,
            a_width,
            True,
            256,
            a_gpus_per_node=a_width,
            f_gpus_per_node=f_width,
            search_config={**_NO_CEILING, "f_moe_ep_size_list": ["tp_f"]},
        )
        assert candidates, "expected candidates with ep resolved from tp_f"
        for _na, nf, _tp_a, ep, *_rest in candidates:
            # ep came from the symbolic "tp_f", so it *is* tp_f.
            assert ep == nf * f_width, f"n_f={nf}: ep={ep}, expected {nf * f_width}"
            assert ep != nf * a_width or a_width == f_width

    def test_ep_from_n_f_nodes_is_unaffected_by_the_width(self):
        """Sanity companion: the ``"n_f_nodes"`` symbol is a node count, so it
        must NOT scale with either width."""
        candidates = build_afd_parallel_lists(
            96,
            4,
            True,
            256,
            a_gpus_per_node=4,
            f_gpus_per_node=8,
            search_config={**_NO_CEILING, "f_moe_ep_size_list": ["n_f_nodes"]},
        )
        assert candidates
        for _na, nf, _tp_a, ep, *_rest in candidates:
            assert ep == nf

    def test_budget_below_one_node_each_yields_nothing(self):
        """4 + 8 = 12 GPUs is the floor for this pairing."""
        assert build_afd_parallel_lists(11, 4, False, 0, a_gpus_per_node=4, f_gpus_per_node=8) == []
        assert build_afd_parallel_lists(12, 4, False, 0, a_gpus_per_node=4, f_gpus_per_node=8) != []

    @pytest.mark.parametrize("total_gpus", [16, 24, 32, 64, 72])
    @pytest.mark.parametrize("width", [2, 4, 8])
    @pytest.mark.parametrize("is_moe,num_experts", [(False, 0), (True, 256)])
    def test_equal_widths_reproduce_the_single_width_enumeration(self, total_gpus, width, is_moe, num_experts):
        """The gate that makes this change safe: naming the widths explicitly
        must not alter homogeneous results, element for element and in order."""
        implicit = build_afd_parallel_lists(total_gpus, width, is_moe, num_experts, search_config=_NO_CEILING)
        explicit = build_afd_parallel_lists(
            total_gpus,
            width,
            is_moe,
            num_experts,
            a_gpus_per_node=width,
            f_gpus_per_node=width,
            search_config=_NO_CEILING,
        )
        assert implicit == explicit

    def test_every_admitted_candidate_fits_the_budget(self):
        """Property form, over a spread of asymmetric pairings."""
        for a_w, f_w, total in ((4, 8, 72), (8, 4, 72), (2, 8, 40), (8, 2, 40), (4, 4, 32)):
            candidates = build_afd_parallel_lists(
                total, max(a_w, f_w), True, 256, a_gpus_per_node=a_w, f_gpus_per_node=f_w, search_config=_NO_CEILING
            )
            for na, nf, *_rest in candidates:
                assert na * a_w + nf * f_w <= total, (
                    f"a_w={a_w} f_w={f_w} total={total}: ({na}, {nf}) needs {na * a_w + nf * f_w}"
                )

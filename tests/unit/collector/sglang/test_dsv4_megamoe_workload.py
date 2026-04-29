# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collector.sglang.dsv4_megamoe_workload import (
    build_dsv4_power_law_megamoe_workload,
    build_dsv4_power_law_megamoe_workload_from_global_tokens,
    build_dsv4_uniform_megamoe_workload,
    build_dsv4_uniform_megamoe_workload_from_global_tokens,
)


def test_power_law_workload_uses_target_ep_and_remaps_bottleneck_to_rank0():
    workload = build_dsv4_power_law_megamoe_workload(
        num_tokens_per_rank=16,
        routed_num_experts=8,
        routed_topk=2,
        moe_ep_size=4,
        alpha=1.2,
        seed=7,
    )

    assert workload.num_global_tokens == 64
    assert workload.experts_per_rank == 2
    assert sum(workload.routed_expert_counts) == workload.num_global_tokens * workload.routed_topk
    assert workload.routed_rank_loads[0] == max(workload.routed_rank_loads)
    assert len(workload.routed_topk_ids_by_src_rank) == 4
    assert all(len(rank_rows) == 16 for rank_rows in workload.routed_topk_ids_by_src_rank)


def test_uniform_workload_uses_target_ep_and_local_rank0_workload():
    workload = build_dsv4_uniform_megamoe_workload(
        num_tokens_per_rank=8,
        routed_num_experts=8,
        routed_topk=2,
        moe_ep_size=4,
    )

    assert workload.num_global_tokens == 32
    assert workload.experts_per_rank == 2
    assert workload.routed_rank_loads == (16, 16, 16, 16)
    assert sum(workload.rank0_masked_m) == 16
    assert all(
        expert_id == -1 or 0 <= expert_id < workload.experts_per_rank
        for row in workload.rank0_local_topk_ids
        for expert_id in row
    )


def test_global_token_power_law_workload_does_not_expand_tokens_by_ep():
    workload = build_dsv4_power_law_megamoe_workload_from_global_tokens(
        num_global_tokens=17,
        routed_num_experts=8,
        routed_topk=2,
        moe_ep_size=4,
        alpha=1.2,
        seed=9,
    )

    assert workload.num_global_tokens == 17
    assert workload.num_tokens_per_rank == 5
    assert sum(len(rank_rows) for rank_rows in workload.routed_topk_ids_by_src_rank) == 17
    assert sum(sum(row) for row in workload.route_matrix) == 17 * 2
    assert workload.routed_rank_loads[0] == max(workload.routed_rank_loads)


def test_global_token_uniform_workload_keeps_global_token_count():
    workload = build_dsv4_uniform_megamoe_workload_from_global_tokens(
        num_global_tokens=17,
        routed_num_experts=8,
        routed_topk=2,
        moe_ep_size=4,
    )

    assert workload.num_global_tokens == 17
    assert workload.num_tokens_per_rank == 5
    assert sum(len(rank_rows) for rank_rows in workload.routed_topk_ids_by_src_rank) == 17
    assert sum(sum(row) for row in workload.route_matrix) == 17 * 2


def test_power_law_workload_route_matrix_preserves_all_source_rank_assignments():
    workload = build_dsv4_power_law_megamoe_workload(
        num_tokens_per_rank=5,
        routed_num_experts=8,
        routed_topk=3,
        moe_ep_size=4,
        alpha=0.8,
        seed=11,
    )

    assert sum(sum(row) for row in workload.route_matrix) == 5 * 4 * 3
    assert all(sum(row) == 5 * 3 for row in workload.route_matrix)


def test_power_law_ep1_workload_has_zero_remote_traffic():
    workload = build_dsv4_power_law_megamoe_workload(
        num_tokens_per_rank=8,
        routed_num_experts=8,
        routed_topk=2,
        moe_ep_size=1,
        alpha=1.2,
        hidden_size=4096,
        seed=3,
    )

    assert workload.route_matrix == ((16,),)
    assert workload.traffic is not None
    assert workload.traffic.total_remote_edges == 0
    assert workload.traffic.bottleneck_primary_bytes == 0


def test_fused_shared_experts_do_not_add_inter_rank_routes():
    workload = build_dsv4_power_law_megamoe_workload(
        num_tokens_per_rank=6,
        routed_num_experts=8,
        routed_topk=2,
        moe_ep_size=4,
        alpha=1.1,
        num_fused_shared_experts=1,
        seed=5,
    )

    assert workload.mega_topk == 3
    assert sum(sum(row) for row in workload.route_matrix) == 6 * 4 * 2
    assert all(len(row) == 3 for rank_rows in workload.mega_topk_ids_by_src_rank for row in rank_rows)
    assert all(
        row[-1] == workload.routed_num_experts
        for rank_rows in workload.mega_topk_ids_by_src_rank
        for row in rank_rows
    )
    assert any(workload.experts_per_rank in row for row in workload.rank0_local_topk_ids)
    assert len(workload.rank0_masked_m) == workload.experts_per_rank + 1

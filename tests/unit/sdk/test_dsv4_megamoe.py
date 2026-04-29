# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from aiconfigurator.sdk.dsv4_megamoe import (
    DSV4_MEGAMOE_EP8_B200_RANDOM_EFFECTIVE_BW_MODEL,
    MegaMoEEffectiveBandwidthModel,
    build_dsv4_power_law_megamoe_workload,
    build_dsv4_power_law_megamoe_workload_from_global_tokens,
    build_dsv4_uniform_megamoe_workload,
    build_dsv4_uniform_megamoe_workload_from_global_tokens,
    build_route_matrix,
    compose_megamoe_routed_latency_ms,
    dsv4_megamoe_ep8_b200_random_remote_bandwidth_bps,
    estimate_dsv4_megamoe_ep8_random_calibrated_communication_ms,
    estimate_megamoe_communication_from_effective_bandwidth_model_ms,
    estimate_megamoe_communication_ms,
    estimate_megamoe_traffic,
    owner_rank_for_expert,
    route_matrix_from_flat_assignments,
)


def test_owner_rank_for_expert_uses_contiguous_expert_ownership():
    assert owner_rank_for_expert(0, num_experts=8, moe_ep_size=4) == 0
    assert owner_rank_for_expert(1, num_experts=8, moe_ep_size=4) == 0
    assert owner_rank_for_expert(2, num_experts=8, moe_ep_size=4) == 1
    assert owner_rank_for_expert(7, num_experts=8, moe_ep_size=4) == 3


def test_owner_rank_rejects_non_divisible_experts():
    with pytest.raises(ValueError, match="divisible"):
        owner_rank_for_expert(0, num_experts=10, moe_ep_size=4)


def test_build_route_matrix_from_topk_assignments():
    routes = build_route_matrix(
        [
            [[0, 1], [2, 3]],  # src 0 -> rank 0 twice, rank 1 twice
            [[4, 7], [-1, 5]],  # src 1 -> rank 2 twice, rank 3 once
            [],
            [],
        ],
        num_experts=8,
        moe_ep_size=4,
    )

    assert routes == (
        (2, 2, 0, 0),
        (0, 0, 2, 1),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
    )


def test_all_local_routing_has_zero_remote_primary_bytes():
    traffic = estimate_megamoe_traffic(
        [
            [3, 0],
            [0, 5],
        ],
        hidden_size=4096,
    )

    assert traffic.total_remote_edges == 0
    assert traffic.total_primary_bytes == 0
    assert traffic.bottleneck_primary_bytes == 0
    assert traffic.owner_primary_bytes == (0, 0)
    assert traffic.endpoint_primary_bytes == (0, 0)


def test_remote_routing_uses_deepgemm_primary_byte_convention():
    hidden = 4096
    traffic = estimate_megamoe_traffic(
        [
            [0, 2],
            [2, 0],
        ],
        hidden_size=hidden,
    )

    bytes_per_selection = 3 * hidden
    assert traffic.primary_bytes_per_remote_selection == bytes_per_selection
    assert traffic.total_remote_edges == 4
    assert traffic.owner_remote_edges == (2, 2)
    assert traffic.endpoint_remote_edges == (2, 2)
    assert traffic.total_primary_bytes == 4 * bytes_per_selection
    assert traffic.owner_primary_bytes == (2 * bytes_per_selection, 2 * bytes_per_selection)
    assert traffic.endpoint_primary_bytes == (2 * bytes_per_selection, 2 * bytes_per_selection)
    assert traffic.bottleneck_primary_bytes == 2 * bytes_per_selection
    assert traffic.bottleneck_endpoint_primary_bytes == 2 * bytes_per_selection


def test_deepgemm_num_recv_tokens_interpretation_matches_owner_bottleneck():
    hidden = 7168
    num_recv_tokens = 5
    traffic = estimate_megamoe_traffic(
        [
            [0, num_recv_tokens],
            [0, 0],
        ],
        hidden_size=hidden,
    )

    assert traffic.owner_primary_bytes[1] == num_recv_tokens * hidden * 3
    assert traffic.bottleneck_primary_bytes == num_recv_tokens * hidden * 3
    assert traffic.bottleneck_owner_rank == 1


def test_active_owner_bottleneck_is_reported_separately_from_source_endpoint():
    hidden = 1024
    traffic = estimate_megamoe_traffic(
        [
            [0, 5, 0],
            [0, 0, 0],
            [0, 5, 0],
        ],
        hidden_size=hidden,
    )

    assert traffic.owner_remote_edges == (0, 10, 0)
    assert traffic.endpoint_remote_edges == (5, 0, 5)
    assert traffic.bottleneck_owner_rank == 1
    assert traffic.bottleneck_endpoint_rank == 0
    assert traffic.bottleneck_primary_bytes == 10 * 3 * hidden
    assert traffic.bottleneck_endpoint_primary_bytes == 5 * 3 * hidden


def test_metadata_bytes_are_reported_separately_from_primary_bytes():
    traffic = estimate_megamoe_traffic(
        [
            [0, 1],
            [0, 0],
        ],
        hidden_size=4096,
        quant_group_size=32,
    )

    assert traffic.primary_bytes_per_remote_selection == 3 * 4096
    assert traffic.metadata_bytes_per_remote_selection == 4096 // 32 + 8
    assert traffic.total_primary_bytes == 3 * 4096
    assert traffic.total_metadata_bytes == 4096 // 32 + 8


def test_communication_estimator_uses_bottleneck_owner_bytes_and_barriers():
    comm = estimate_megamoe_communication_ms(
        [
            [0, 4],
            [1, 0],
        ],
        hidden_size=1024,
        effective_nvlink_bandwidth_bps=1_000_000_000,
        nvl_barrier_latency_us=3.0,
        nvl_barrier_count=2,
    )

    expected_data_bytes = 4 * 3 * 1024
    assert comm.data_bytes == expected_data_bytes
    assert math.isclose(comm.data_ms, expected_data_bytes / 1_000_000_000 * 1000.0)
    assert math.isclose(comm.fixed_overlappable_latency_ms, 0.0)
    assert math.isclose(comm.barrier_ms, 0.006)
    assert math.isclose(comm.cleanup_barrier_ms, 0.003)
    assert math.isclose(comm.overlappable_ms, comm.data_ms + comm.barrier_ms)
    assert math.isclose(comm.tail_ms, comm.cleanup_barrier_ms)
    assert math.isclose(comm.unoverlapped_ms, comm.overlappable_ms + comm.tail_ms)
    assert math.isclose(comm.total_ms, comm.unoverlapped_ms)
    assert comm.bottleneck_rank == 1


def test_communication_estimator_can_add_fixed_overlappable_latency():
    comm = estimate_megamoe_communication_ms(
        [[0, 1], [0, 0]],
        hidden_size=1024,
        effective_nvlink_bandwidth_bps=3_072_000_000,
        nvl_barrier_latency_us=0.0,
        fixed_overlappable_latency_ms=0.05,
    )

    assert math.isclose(comm.data_ms, 0.001)
    assert math.isclose(comm.fixed_overlappable_latency_ms, 0.05)
    assert math.isclose(comm.overlappable_ms, 0.051)


def test_communication_estimator_can_include_metadata_in_data_bytes():
    primary_only = estimate_megamoe_communication_ms(
        [[0, 1], [0, 0]],
        hidden_size=1024,
        effective_nvlink_bandwidth_bps=1_000_000_000,
        nvl_barrier_latency_us=0.0,
        include_metadata=False,
    )
    with_metadata = estimate_megamoe_communication_ms(
        [[0, 1], [0, 0]],
        hidden_size=1024,
        effective_nvlink_bandwidth_bps=1_000_000_000,
        nvl_barrier_latency_us=0.0,
        include_metadata=True,
    )

    assert with_metadata.data_bytes == (
        primary_only.data_bytes + primary_only.traffic.metadata_bytes_per_remote_selection
    )


def test_overlap_composer_uses_max_not_sum():
    assert compose_megamoe_routed_latency_ms(
        local_routing_prep_ms=0.2,
        core_compute_ms=1.5,
        comm_ms=1.0,
        sync_tail_ms=0.1,
    ) == 1.8
    assert compose_megamoe_routed_latency_ms(
        local_routing_prep_ms=0.2,
        core_compute_ms=1.0,
        comm_ms=1.5,
        sync_tail_ms=0.1,
    ) == 1.8


def test_overlap_composer_uses_comm_estimate_hot_path_and_tail():
    comm = estimate_megamoe_communication_ms(
        [[0, 1], [0, 0]],
        hidden_size=1024,
        effective_nvlink_bandwidth_bps=3_072_000_000,
        nvl_barrier_latency_us=3.0,
        nvl_barrier_count=2,
        cleanup_barrier_count=1,
    )

    assert math.isclose(comm.data_ms, 0.001)
    assert math.isclose(comm.overlappable_ms, 0.007)
    assert math.isclose(comm.tail_ms, 0.003)
    assert math.isclose(
        compose_megamoe_routed_latency_ms(
            local_routing_prep_ms=0.2,
            core_compute_ms=0.005,
            comm_estimate=comm,
            sync_tail_ms=0.1,
        ),
        0.310,
    )


def test_random_ep8_b200_bandwidth_curve_uses_log_log_interpolation():
    exact = dsv4_megamoe_ep8_b200_random_remote_bandwidth_bps(16)
    model_exact = DSV4_MEGAMOE_EP8_B200_RANDOM_EFFECTIVE_BW_MODEL.raw_bandwidth_bps(16)
    left = dsv4_megamoe_ep8_b200_random_remote_bandwidth_bps(16)
    middle = dsv4_megamoe_ep8_b200_random_remote_bandwidth_bps(24)
    right = dsv4_megamoe_ep8_b200_random_remote_bandwidth_bps(32)

    assert math.isclose(exact, 5.383984501287102e9)
    assert math.isclose(model_exact, exact)
    assert left < middle < right


def test_effective_bandwidth_model_estimator_is_generic():
    model = MegaMoEEffectiveBandwidthModel(
        name="test-ep2",
        ep_size=2,
        bandwidth_points_gbps=((1, 1.0), (4, 4.0)),
        bandwidth_scale=2.0,
        fixed_overlappable_latency_ms=0.1,
    )

    comm = estimate_megamoe_communication_from_effective_bandwidth_model_ms(
        [[0, 4], [0, 0]],
        hidden_size=1024,
        num_tokens_per_rank=1,
        bandwidth_model=model,
    )

    assert math.isclose(comm.effective_nvlink_bandwidth_bps, 2_000_000_000.0)
    assert math.isclose(comm.data_ms, 4 * 3 * 1024 / 2_000_000_000.0 * 1000.0)
    assert math.isclose(comm.overlappable_ms, comm.data_ms + 0.1)


def test_random_ep8_b200_calibrated_communication_uses_scaled_curve_and_fixed_latency():
    routes = [[0 for _ in range(8)] for _ in range(8)]
    routes[0][1] = 10

    comm = estimate_dsv4_megamoe_ep8_random_calibrated_communication_ms(
        routes,
        hidden_size=1024,
        num_tokens_per_rank=16,
        bandwidth_scale=2.0,
        fixed_overlappable_latency_ms=0.123,
    )

    expected_data_bytes = 10 * 3 * 1024
    expected_bw = 2.0 * dsv4_megamoe_ep8_b200_random_remote_bandwidth_bps(16)
    assert comm.data_bytes == expected_data_bytes
    assert math.isclose(comm.effective_nvlink_bandwidth_bps, expected_bw)
    assert math.isclose(comm.data_ms, expected_data_bytes / expected_bw * 1000.0)
    assert math.isclose(comm.overlappable_ms, comm.data_ms + 0.123)
    assert comm.tail_ms == 0.0


def test_random_ep8_b200_calibrated_communication_requires_ep8_routes():
    with pytest.raises(ValueError, match="8x8"):
        estimate_dsv4_megamoe_ep8_random_calibrated_communication_ms(
            [[0, 1], [0, 0]],
            hidden_size=1024,
            num_tokens_per_rank=16,
        )


def test_route_matrix_from_flat_assignments_ignores_masked_experts():
    routes = route_matrix_from_flat_assignments(
        [
            (0, 0),
            (0, 3),
            (1, 1),
            (1, -1),
        ],
        num_experts=4,
        moe_ep_size=2,
    )

    assert routes == (
        (1, 1),
        (1, 0),
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

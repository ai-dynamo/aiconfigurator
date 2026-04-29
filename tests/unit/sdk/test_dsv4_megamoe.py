# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from aiconfigurator.sdk.operations import (
    Dsv4MegaMoEDispatch,
    build_route_matrix,
    compose_megamoe_routed_latency_ms,
    estimate_megamoe_communication_from_comm_path_model_ms,
    estimate_megamoe_communication_from_effective_bandwidth_model_ms,
    estimate_megamoe_communication_from_perf_database_ms,
    estimate_megamoe_communication_ms,
    estimate_megamoe_traffic,
    owner_rank_for_expert,
    route_matrix_from_flat_assignments,
)
from aiconfigurator.sdk.perf_database import MegaMoECommPathModel, MegaMoEEffectiveBandwidthModel


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


def test_effective_bandwidth_model_uses_log_log_interpolation():
    model = MegaMoEEffectiveBandwidthModel(
        name="test-ep2",
        ep_size=2,
        bandwidth_points_gbps=((16, 4.0), (32, 16.0)),
        bandwidth_scale=1.0,
        fixed_overlappable_latency_ms=0.0,
    )

    exact = model.raw_bandwidth_bps(16)
    left = model.raw_bandwidth_bps(16)
    middle = model.raw_bandwidth_bps(24)
    right = model.raw_bandwidth_bps(32)

    assert math.isclose(exact, 4.0e9)
    assert left < middle < right


def test_effective_bandwidth_model_estimator_returns_traffic_path_latency():
    model = MegaMoEEffectiveBandwidthModel(
        name="test-ep2",
        ep_size=2,
        bandwidth_points_gbps=((1, 1.0), (4, 4.0)),
        bandwidth_scale=1.0,
        fixed_overlappable_latency_ms=0.0,
    )

    estimate = estimate_megamoe_communication_from_effective_bandwidth_model_ms(
        [[0, 4], [0, 0]],
        hidden_size=1024,
        num_tokens_per_rank=1,
        bandwidth_model=model,
    )

    expected_data_ms = 4 * 3 * 1024 / 1.0e9 * 1000.0
    assert estimate.data_bytes == 4 * 3 * 1024
    assert math.isclose(estimate.effective_nvlink_bandwidth_bps, 1.0e9)
    assert math.isclose(estimate.data_ms, expected_data_ms)
    assert math.isclose(estimate.overlappable_ms, expected_data_ms)


def test_effective_bandwidth_model_estimator_rejects_calibrated_effective_bw():
    routes = [[0 for _ in range(8)] for _ in range(8)]
    routes[0][1] = 10
    model = MegaMoEEffectiveBandwidthModel(
        name="test-ep8",
        ep_size=8,
        bandwidth_points_gbps=((16, 4.0), (32, 8.0)),
        bandwidth_scale=2.0,
        fixed_overlappable_latency_ms=0.0,
    )

    with pytest.raises(ValueError, match="bandwidth_scale must be 1.0"):
        estimate_megamoe_communication_from_effective_bandwidth_model_ms(
            routes,
            hidden_size=1024,
            num_tokens_per_rank=16,
            bandwidth_model=model,
        )

    model = MegaMoEEffectiveBandwidthModel(
        name="test-ep8",
        ep_size=8,
        bandwidth_points_gbps=((16, 4.0), (32, 8.0)),
        bandwidth_scale=1.0,
        fixed_overlappable_latency_ms=0.1,
    )
    with pytest.raises(ValueError, match="fixed_overlappable_latency_ms must be 0.0"):
        estimate_megamoe_communication_from_effective_bandwidth_model_ms(
            routes,
            hidden_size=1024,
            num_tokens_per_rank=16,
            bandwidth_model=model,
        )


def test_effective_bandwidth_model_estimator_requires_matching_ep_routes():
    model = MegaMoEEffectiveBandwidthModel(
        name="test-ep8",
        ep_size=8,
        bandwidth_points_gbps=((16, 4.0), (32, 8.0)),
        bandwidth_scale=1.0,
        fixed_overlappable_latency_ms=0.0,
    )
    with pytest.raises(ValueError, match="8x8"):
        estimate_megamoe_communication_from_effective_bandwidth_model_ms(
            [[0, 1], [0, 0]],
            hidden_size=1024,
            num_tokens_per_rank=16,
            bandwidth_model=model,
        )


def test_comm_path_model_estimator_returns_measured_overlap_terms():
    model = MegaMoECommPathModel(
        name="test-ep2",
        ep_size=2,
        comm_path_points_ms=((8, 0.15), (16, 0.25)),
        tail_points_ms=((8, 0.01), (16, 0.02)),
        comm_plus_tail_points_ms=((8, 0.16), (16, 0.27)),
        target_fused_points_ms=((8, 0.35), (16, 0.45)),
    )

    estimate = estimate_megamoe_communication_from_comm_path_model_ms(
        [[0, 4], [0, 0]],
        hidden_size=1024,
        num_tokens_per_rank=8,
        comm_path_model=model,
    )

    assert estimate.data_bytes == 4 * 3 * 1024
    assert math.isclose(estimate.overlappable_ms, 0.15)
    assert math.isclose(estimate.tail_ms, 0.01)
    assert math.isclose(estimate.total_ms, 0.16)


def test_perf_database_estimator_queries_comm_path_model():
    class FakeDatabase:
        def __init__(self):
            self.calls = []

        def query_dsv4_megamoe_comm_path_model(self, **kwargs):
            self.calls.append(kwargs)
            return MegaMoECommPathModel(
                name="db-model",
                ep_size=2,
                comm_path_points_ms=((8, 0.15), (16, 0.25)),
                tail_points_ms=((8, 0.01), (16, 0.02)),
                comm_plus_tail_points_ms=((8, 0.16), (16, 0.27)),
                target_fused_points_ms=((8, 0.35), (16, 0.45)),
            )

    database = FakeDatabase()
    estimate = estimate_megamoe_communication_from_perf_database_ms(
        [[0, 4], [0, 0]],
        database=database,
        hidden_size=1024,
        inter_size=512,
        topk=6,
        num_experts=16,
        moe_ep_size=2,
        routing_mode="power-law",
        power_law_alpha=1.01,
        num_tokens_per_rank=8,
    )

    assert database.calls == [
        {
            "hidden_size": 1024,
            "inter_size": 512,
            "topk": 6,
            "num_experts": 16,
            "moe_ep_size": 2,
            "routing_mode": "power-law",
            "power_law_alpha": 1.01,
            "kernel_source": "DeepGEMM_fp8_fp4_mega_moe",
        }
    ]
    assert math.isclose(estimate.overlappable_ms, 0.15)
    assert math.isclose(estimate.tail_ms, 0.01)


def test_dsv4_megamoe_dispatch_op_queries_comm_path_model():
    class FakeDatabase:
        def __init__(self):
            self.calls = []

        def query_dsv4_megamoe_comm_path_model(self, **kwargs):
            self.calls.append(kwargs)
            return MegaMoECommPathModel(
                name="db-model",
                ep_size=2,
                comm_path_points_ms=((8, 0.15), (16, 0.25)),
                tail_points_ms=((8, 0.01), (16, 0.02)),
                comm_plus_tail_points_ms=((8, 0.16), (16, 0.27)),
                target_fused_points_ms=((8, 0.35), (16, 0.45)),
            )

    op = Dsv4MegaMoEDispatch(
        "dsv4_megamoe_comm",
        1.0,
        hidden_size=1024,
        inter_size=512,
        topk=6,
        num_experts=16,
        moe_ep_size=2,
        routing_mode="power-law",
        power_law_alpha=1.01,
    )
    database = FakeDatabase()
    result = op.query(database, x=8, route_matrix=[[0, 4], [0, 0]])

    assert database.calls == [
        {
            "hidden_size": 1024,
            "inter_size": 512,
            "topk": 6,
            "num_experts": 16,
            "moe_ep_size": 2,
            "routing_mode": "power-law",
            "power_law_alpha": 1.01,
            "kernel_source": "DeepGEMM_fp8_fp4_mega_moe",
        }
    ]
    assert math.isclose(float(result), 0.15)
    assert result.energy == 0.0


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

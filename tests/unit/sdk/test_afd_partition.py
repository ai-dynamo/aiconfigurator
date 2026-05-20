# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AFD operation partitioning."""

import pytest

from aiconfigurator.sdk import operations
from aiconfigurator.sdk.afd_partition import AFDPartitionError, build_afd_ops_partition

pytestmark = pytest.mark.unit


class _Model:
    def __init__(self, *, context_ops=None, generation_ops=None) -> None:
        self.context_ops = context_ops or []
        self.generation_ops = generation_ops or []


class _NamedOp(operations.Operation):
    def __init__(self, name: str) -> None:
        super().__init__(name, 1.0)

    def query(self, database, **kwargs):
        raise NotImplementedError

    def get_weights(self, **kwargs):
        return 0.0


def _names(op_list):
    return [op._name for op in op_list]


def test_build_afd_ops_partition_dense_generation_path():
    model = _Model(
        generation_ops=[
            _NamedOp("generation_embedding"),
            _NamedOp("generation_add_norm_1"),
            _NamedOp("generation_qkv_gemm"),
            _NamedOp("generation_attention"),
            _NamedOp("generation_proj_gemm"),
            operations.CustomAllReduce("generation_ar_1", 1, 4096, 4),
            _NamedOp("generation_add_norm_2"),
            _NamedOp("generation_ffn1_gemm"),
            _NamedOp("generation_act"),
            _NamedOp("generation_ffn2_gemm"),
            _NamedOp("generation_logits_gemm"),
            operations.P2P("generation_p2p", 1, 4096, 2),
        ]
    )

    partition = build_afd_ops_partition(model, phase="generation")

    assert partition.phase == "generation"
    assert _names(partition.attn_ops) == [
        "generation_embedding",
        "generation_add_norm_1",
        "generation_qkv_gemm",
        "generation_attention",
        "generation_proj_gemm",
        "generation_add_norm_2",
        "generation_logits_gemm",
    ]
    assert _names(partition.ffn_ops) == [
        "generation_ffn1_gemm",
        "generation_act",
        "generation_ffn2_gemm",
    ]
    assert _names(partition.boundary_ops) == ["generation_add_norm_2", "generation_logits_gemm"]
    assert _names(partition.skipped_ops) == ["generation_ar_1", "generation_p2p"]


def test_build_afd_ops_partition_moe_overlap_stays_atomic_on_f_worker():
    routed_ops = [
        _NamedOp("generation_router_gemm"),
        _NamedOp("generation_moe_pre_dispatch"),
        _NamedOp("generation_moe"),
        _NamedOp("generation_moe_post_dispatch"),
    ]
    shared_ops = [
        _NamedOp("generation_shared_gate_up_gemm"),
        _NamedOp("generation_shared_act_gate"),
        _NamedOp("generation_shared_ffn2_gemm"),
    ]
    overlap = operations.OverlapOp("generation_moe_dispatch_overlap", group_a=routed_ops, group_b=shared_ops)
    model = _Model(
        generation_ops=[
            _NamedOp("generation_add_norm_2"),
            overlap,
            _NamedOp("generation_moe_reduce_add"),
        ]
    )

    partition = build_afd_ops_partition(model, phase="generation")

    assert _names(partition.attn_ops) == ["generation_add_norm_2", "generation_moe_reduce_add"]
    assert partition.ffn_ops == [overlap]
    assert _names(partition.boundary_ops) == ["generation_add_norm_2", "generation_moe_reduce_add"]
    assert all(inner not in partition.attn_ops + partition.ffn_ops for inner in routed_ops + shared_ops)


def test_build_afd_ops_partition_attention_overlap_stays_atomic_on_a_worker():
    overlap = operations.OverlapOp(
        "generation_bmm_rope_overlap",
        group_a=[_NamedOp("generation_bmm_pre")],
        group_b=[_NamedOp("generation_rope_kvcache")],
    )
    model = _Model(generation_ops=[overlap])

    partition = build_afd_ops_partition(model, phase="generation")

    assert partition.attn_ops == [overlap]
    assert partition.ffn_ops == []


def test_build_afd_ops_partition_rejects_overlap_spanning_boundary():
    overlap = operations.OverlapOp(
        "generation_future_overlap",
        group_a=[_NamedOp("generation_attention")],
        group_b=[_NamedOp("generation_moe")],
    )
    model = _Model(generation_ops=[overlap])

    with pytest.raises(AFDPartitionError, match="spans A/F boundaries"):
        build_afd_ops_partition(model, phase="generation")


def test_build_afd_ops_partition_context_path():
    model = _Model(
        context_ops=[
            _NamedOp("context_embedding"),
            _NamedOp("context_qkv_gemm"),
            _NamedOp("context_attention"),
            _NamedOp("context_proj_gemm"),
            _NamedOp("context_add_norm_2"),
            _NamedOp("context_gate_ffn1_gemm"),
            _NamedOp("context_act_gate"),
            _NamedOp("context_ffn2_gemm"),
            operations.P2P("context_p2p", 1, 4096, 2),
        ]
    )

    partition = build_afd_ops_partition(model, phase="context")

    assert partition.phase == "context"
    assert _names(partition.attn_ops) == [
        "context_embedding",
        "context_qkv_gemm",
        "context_attention",
        "context_proj_gemm",
        "context_add_norm_2",
    ]
    assert _names(partition.ffn_ops) == [
        "context_gate_ffn1_gemm",
        "context_act_gate",
        "context_ffn2_gemm",
    ]
    assert _names(partition.boundary_ops) == ["context_add_norm_2"]
    assert _names(partition.skipped_ops) == ["context_p2p"]


def test_build_afd_ops_partition_boundary_placement_can_be_overridden():
    model = _Model(
        generation_ops=[
            _NamedOp("generation_add_norm_2"),
            _NamedOp("generation_logits_gemm"),
            _NamedOp("generation_moe_reduce_add"),
        ]
    )

    partition = build_afd_ops_partition(model, phase="generation", boundary_on_attn=False)

    assert partition.attn_ops == []
    assert _names(partition.ffn_ops) == [
        "generation_add_norm_2",
        "generation_logits_gemm",
        "generation_moe_reduce_add",
    ]
    assert _names(partition.boundary_ops) == [
        "generation_add_norm_2",
        "generation_logits_gemm",
        "generation_moe_reduce_add",
    ]


def test_build_afd_ops_partition_skips_model_internal_dispatch_ops():
    model = _Model(
        generation_ops=[
            _NamedOp("generation_moe_pre_dispatch"),
            _NamedOp("generation_moe_post_dispatch"),
        ]
    )

    partition = build_afd_ops_partition(model, phase="generation")

    assert partition.attn_ops == []
    assert partition.ffn_ops == []
    assert _names(partition.skipped_ops) == ["generation_moe_pre_dispatch", "generation_moe_post_dispatch"]


def test_build_afd_ops_partition_rejects_unknown_ops_by_default():
    model = _Model(generation_ops=[_NamedOp("generation_unknown_kernel")])

    with pytest.raises(AFDPartitionError, match="Cannot classify op"):
        build_afd_ops_partition(model, phase="generation")


def test_build_afd_ops_partition_rejects_unclassifiable_mamba_ops_by_default():
    model = _Model(generation_ops=[_NamedOp("generation_mamba_in_proj_gemm")])

    with pytest.raises(AFDPartitionError, match="cannot safely classify Mamba"):
        build_afd_ops_partition(model, phase="generation")


def test_build_afd_ops_partition_rejects_unclassifiable_gdn_ops_by_default():
    model = _Model(generation_ops=[_NamedOp("generation_gdn_chunk_gated_delta_rule")])

    with pytest.raises(AFDPartitionError, match="cannot safely classify GDN"):
        build_afd_ops_partition(model, phase="generation")


def test_build_afd_ops_partition_rejects_unclassifiable_ops_even_when_unknown_allowed():
    model = _Model(generation_ops=[_NamedOp("generation_mamba_ssm_kernel")])

    with pytest.raises(AFDPartitionError, match="not covered by the current attention/FFN partition rules"):
        build_afd_ops_partition(model, phase="generation", allow_unknown_ops=True, unknown_side="ffn")


def test_build_afd_ops_partition_can_allow_unknown_ops():
    model = _Model(generation_ops=[_NamedOp("generation_future_kernel")])

    partition = build_afd_ops_partition(model, phase="generation", allow_unknown_ops=True, unknown_side="ffn")

    assert partition.attn_ops == []
    assert _names(partition.ffn_ops) == ["generation_future_kernel"]

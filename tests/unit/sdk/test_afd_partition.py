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


# ---------------------------------------------------------------------------
# Classifier-consistency regressions: HF-style FFN ``proj_gemm`` names
# stay on F, canonical attention ``proj_gemm`` names stay on A, and the
# OverlapOp wrappers honor the same attn -> ffn marker order as bare ops.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ffn_proj_name",
    [
        "generation_down_proj_gemm",
        "generation_up_proj_gemm",
        "generation_gate_proj_gemm",
        # Phase prefix may be absent on HF-naming ports.
        "down_proj_gemm",
        "up_proj_gemm",
        "gate_proj_gemm",
    ],
)
def test_build_afd_ops_partition_hf_style_ffn_proj_gemm_lands_on_ffn(ffn_proj_name):
    """HF-style FFN names containing ``proj_gemm`` must classify as FFN, not Attention.

    The bare ``proj_gemm`` substring is the attention out-projection
    marker; without an explicit exclusion guard ``down_proj_gemm`` /
    ``up_proj_gemm`` / ``gate_proj_gemm`` would match
    ``_is_attention_side_op`` first (attn precedes ffn in the unified
    classifier) and the FFN GEMM cost would be silently routed into the
    A-pool latency. This pins that the guard plus the explicit FFN
    markers route them to F.
    """
    model = _Model(generation_ops=[_NamedOp(ffn_proj_name)])

    partition = build_afd_ops_partition(model, phase="generation")

    assert _names(partition.attn_ops) == [], f"{ffn_proj_name} must not classify as attn-side"
    assert _names(partition.ffn_ops) == [ffn_proj_name]


def test_build_afd_ops_partition_canonical_attn_proj_gemm_still_attn():
    """The canonical attention out-projection ``<phase>_proj_gemm`` must stay on the A-pool.

    Every model under ``models/`` (deepseek, llama, gpt, moe, hybrid_moe,
    qwen35, nemotron) uses ``<phase>_proj_gemm`` for the attention
    output projection. The FFN-style exclusion guard only filters
    ``down_/up_/gate_proj_gemm``; the canonical name must still route
    to A-pool.
    """
    model = _Model(
        generation_ops=[
            _NamedOp("generation_attention"),
            _NamedOp("generation_proj_gemm"),
            _NamedOp("generation_global_proj_gemm"),
            _NamedOp("generation_swa_proj_gemm"),
        ]
    )

    partition = build_afd_ops_partition(model, phase="generation")

    assert _names(partition.attn_ops) == [
        "generation_attention",
        "generation_proj_gemm",
        "generation_global_proj_gemm",
        "generation_swa_proj_gemm",
    ]
    assert partition.ffn_ops == []


def test_build_afd_ops_partition_overlap_no_inner_uses_unified_order():
    """OverlapOp with no inner ops must classify by the same attn -> ffn order as bare ops.

    Names can hit both attn and ffn substrings (e.g.
    ``moe_attention_overlap`` -- contains ``moe`` AND ``attention``).
    With ``_classify_op`` / ``_classify_overlap_op`` (no-inner branch) /
    ``_classify_inner_overlap_op`` all funneled through
    ``_classify_by_markers``, such a name must resolve to attn from
    every callsite -- if any callsite reversed the order it would
    classify the same name into a different pool.
    """
    # An OverlapOp with no inner ops, named to hit both attn ("attention")
    # and ffn ("moe") substrings. Attn precedes ffn in the unified order.
    overlap_no_inner = operations.OverlapOp("generation_moe_attention_overlap", group_a=[], group_b=[])
    standalone = _NamedOp("generation_moe_attention_overlap")
    model = _Model(generation_ops=[overlap_no_inner, standalone])

    partition = build_afd_ops_partition(model, phase="generation")

    # Both forms (OverlapOp-no-inner + standalone op with the same name)
    # land on attn -- the unified order makes the two callsites agree
    # by construction; if either callsite reversed the order, the
    # OverlapOp would go to ffn and the standalone op to attn,
    # diverging silently.
    assert _names(partition.attn_ops) == [
        "generation_moe_attention_overlap",
        "generation_moe_attention_overlap",
    ]
    assert partition.ffn_ops == []


def test_build_afd_ops_partition_inner_overlap_uses_unified_order():
    """An OverlapOp whose inner ops are HF-style FFN GEMMs must settle on F.

    The inner-op classifier shares ``_classify_by_markers`` with the
    bare-op path, so the FFN-style ``proj_gemm`` exclusion guard
    applies inside ``OverlapOp`` too. Without it, ``down_proj_gemm``
    would hit the bare ``proj_gemm`` attn marker first; combined with
    the OverlapOp inner sides being unanimous, the whole OverlapOp
    would have landed on attn even though every inner op is an FFN
    GEMM.
    """
    overlap = operations.OverlapOp(
        "generation_hf_ffn_overlap",
        group_a=[_NamedOp("generation_gate_proj_gemm"), _NamedOp("generation_up_proj_gemm")],
        group_b=[_NamedOp("generation_down_proj_gemm")],
    )
    model = _Model(generation_ops=[overlap])

    partition = build_afd_ops_partition(model, phase="generation")

    assert partition.attn_ops == []
    assert partition.ffn_ops == [overlap]

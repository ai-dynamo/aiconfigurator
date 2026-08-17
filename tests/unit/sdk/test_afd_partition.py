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


def _named_op(name: str):
    """Minimal engine-backed leaf: the partitioner classifies ops by NAME
    only, and Rust composites require engine-backed children — a trivial
    ElementWise carries the name without any other behavior."""
    return operations.ElementWise(name, 1.0, 1, 1)


def _names(op_list):
    return [op._name for op in op_list]


def test_build_afd_ops_partition_dense_generation_path():
    model = _Model(
        generation_ops=[
            _named_op("generation_embedding"),
            _named_op("generation_add_norm_1"),
            _named_op("generation_qkv_gemm"),
            _named_op("generation_attention"),
            _named_op("generation_proj_gemm"),
            operations.CustomAllReduce("generation_ar_1", 1, 4096, 4),
            _named_op("generation_add_norm_2"),
            _named_op("generation_ffn1_gemm"),
            _named_op("generation_act"),
            _named_op("generation_ffn2_gemm"),
            _named_op("generation_logits_gemm"),
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
        _named_op("generation_router_gemm"),
        _named_op("generation_moe_pre_dispatch"),
        _named_op("generation_moe"),
        _named_op("generation_moe_post_dispatch"),
    ]
    shared_ops = [
        _named_op("generation_shared_gate_up_gemm"),
        _named_op("generation_shared_act_gate"),
        _named_op("generation_shared_ffn2_gemm"),
    ]
    overlap = operations.OverlapOp("generation_moe_dispatch_overlap", group_a=routed_ops, group_b=shared_ops)
    model = _Model(
        generation_ops=[
            _named_op("generation_add_norm_2"),
            overlap,
            _named_op("generation_moe_reduce_add"),
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
        group_a=[_named_op("generation_bmm_pre")],
        group_b=[_named_op("generation_rope_kvcache")],
    )
    model = _Model(generation_ops=[overlap])

    partition = build_afd_ops_partition(model, phase="generation")

    assert partition.attn_ops == [overlap]
    assert partition.ffn_ops == []


def test_build_afd_ops_partition_rejects_overlap_spanning_boundary():
    overlap = operations.OverlapOp(
        "generation_future_overlap",
        group_a=[_named_op("generation_attention")],
        group_b=[_named_op("generation_moe")],
    )
    model = _Model(generation_ops=[overlap])

    with pytest.raises(AFDPartitionError, match="spans A/F boundaries"):
        build_afd_ops_partition(model, phase="generation")


def test_build_afd_ops_partition_context_path():
    model = _Model(
        context_ops=[
            _named_op("context_embedding"),
            _named_op("context_qkv_gemm"),
            _named_op("context_attention"),
            _named_op("context_proj_gemm"),
            _named_op("context_add_norm_2"),
            _named_op("context_gate_ffn1_gemm"),
            _named_op("context_act_gate"),
            _named_op("context_ffn2_gemm"),
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
            _named_op("generation_add_norm_2"),
            _named_op("generation_logits_gemm"),
            _named_op("generation_moe_reduce_add"),
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
            _named_op("generation_moe_pre_dispatch"),
            _named_op("generation_moe_post_dispatch"),
        ]
    )

    partition = build_afd_ops_partition(model, phase="generation")

    assert partition.attn_ops == []
    assert partition.ffn_ops == []
    assert _names(partition.skipped_ops) == ["generation_moe_pre_dispatch", "generation_moe_post_dispatch"]


def test_build_afd_ops_partition_rejects_unknown_ops_by_default():
    model = _Model(generation_ops=[_named_op("generation_unknown_kernel")])

    with pytest.raises(AFDPartitionError, match="Cannot classify op"):
        build_afd_ops_partition(model, phase="generation")


def test_build_afd_ops_partition_rejects_unclassifiable_mamba_ops_by_default():
    model = _Model(generation_ops=[_named_op("generation_mamba_in_proj_gemm")])

    with pytest.raises(AFDPartitionError, match="cannot safely classify Mamba"):
        build_afd_ops_partition(model, phase="generation")


def test_build_afd_ops_partition_rejects_unclassifiable_gdn_ops_by_default():
    model = _Model(generation_ops=[_named_op("generation_gdn_chunk_gated_delta_rule")])

    with pytest.raises(AFDPartitionError, match="cannot safely classify GDN"):
        build_afd_ops_partition(model, phase="generation")


def test_build_afd_ops_partition_rejects_unclassifiable_ops_even_when_unknown_allowed():
    model = _Model(generation_ops=[_named_op("generation_mamba_ssm_kernel")])

    with pytest.raises(AFDPartitionError, match="not covered by the current attention/FFN partition rules"):
        build_afd_ops_partition(model, phase="generation", allow_unknown_ops=True, unknown_side="ffn")


def test_build_afd_ops_partition_can_allow_unknown_ops():
    model = _Model(generation_ops=[_named_op("generation_future_kernel")])

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
    model = _Model(generation_ops=[_named_op(ffn_proj_name)])

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
            _named_op("generation_attention"),
            _named_op("generation_proj_gemm"),
            _named_op("generation_global_proj_gemm"),
            _named_op("generation_swa_proj_gemm"),
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
    standalone = _named_op("generation_moe_attention_overlap")
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
        group_a=[_named_op("generation_gate_proj_gemm"), _named_op("generation_up_proj_gemm")],
        group_b=[_named_op("generation_down_proj_gemm")],
    )
    model = _Model(generation_ops=[overlap])

    partition = build_afd_ops_partition(model, phase="generation")

    assert partition.attn_ops == []
    assert partition.ffn_ops == [overlap]


def _moe_layer_ops():
    return [
        _named_op("generation_qkv_gemm"),
        _named_op("generation_attention"),
        _named_op("generation_moe_router_gemm"),
        _named_op("generation_moe_gemm"),
    ]


def test_router_stays_on_ffn_by_default():
    partition = build_afd_ops_partition(_Model(generation_ops=_moe_layer_ops()), phase="generation")

    assert "generation_moe_router_gemm" in _names(partition.ffn_ops)
    assert "generation_moe_router_gemm" not in _names(partition.attn_ops)


def test_router_on_attn_moves_only_the_router():
    """``router_on_attn=True`` matches the FastAFD boundary.

    Routing happens on the attention worker and routed tokens cross the pool
    boundary. Only the router GEMM changes pool; the expert GEMMs stay on F.
    """
    partition = build_afd_ops_partition(
        _Model(generation_ops=_moe_layer_ops()),
        phase="generation",
        router_on_attn=True,
    )

    assert "generation_moe_router_gemm" in _names(partition.attn_ops)
    assert "generation_moe_router_gemm" not in _names(partition.ffn_ops)
    # Everything else is unmoved.
    assert "generation_moe_gemm" in _names(partition.ffn_ops)
    assert "generation_attention" in _names(partition.attn_ops)


@pytest.mark.parametrize("router_on_attn", [False, True])
def test_router_placement_does_not_change_the_op_set(router_on_attn):
    """The knob repartitions; it never adds or drops work.

    The cross-pool transfer volume is driven by ``p_send`` combinatorics and
    is identical under both placements, so the union of both pools must be
    invariant.
    """
    partition = build_afd_ops_partition(
        _Model(generation_ops=_moe_layer_ops()),
        phase="generation",
        router_on_attn=router_on_attn,
    )

    assert set(_names(partition.attn_ops)) | set(_names(partition.ffn_ops)) == {
        "generation_qkv_gemm",
        "generation_attention",
        "generation_moe_router_gemm",
        "generation_moe_gemm",
    }


def test_router_placement_applies_inside_overlap_ops():
    """An all-router OverlapOp must follow the knob, not the FFN markers."""
    overlap = operations.OverlapOp(
        "generation_router_overlap",
        group_a=[_named_op("generation_moe_router_gemm")],
        group_b=[_named_op("generation_moe_router_aux")],
    )
    model = _Model(generation_ops=[overlap])

    on_ffn = build_afd_ops_partition(model, phase="generation")
    on_attn = build_afd_ops_partition(model, phase="generation", router_on_attn=True)

    assert on_ffn.ffn_ops == [overlap] and on_ffn.attn_ops == []
    assert on_attn.attn_ops == [overlap] and on_attn.ffn_ops == []


def test_router_placement_leaves_the_transfer_volume_model_alone():
    """``router_on_attn`` repartitions compute; it never changes traffic.

    Both placements move the same bytes across the boundary -- FastAFD sends
    routed tokens, aic sends un-routed hidden states, and either way the model
    is the ``p_send`` combinatorics over ``num_experts`` / ``topk`` /
    ``num_f_nodes``. The structural guarantee is that the knob never reaches
    the transfer op, so it *cannot* perturb the volume: assert that directly
    rather than comparing two runs that were never going to differ.
    """
    import inspect

    from aiconfigurator.sdk.operations import AFDTransfer
    from aiconfigurator.sdk.operations.afd_transfer import _afd_send_prob

    transfer_params = set(inspect.signature(AFDTransfer.__init__).parameters)
    assert "router_on_attn" not in transfer_params, (
        "router placement must not be an AFDTransfer input; if it becomes one, "
        "the volume model is no longer placement-independent"
    )

    send_prob_params = set(inspect.signature(_afd_send_prob).parameters)
    assert send_prob_params == {"num_experts", "topk", "num_f_nodes"}, (
        f"p_send must depend only on the MoE shape and F-node count, got {sorted(send_prob_params)}"
    )


def test_router_placement_moves_exactly_one_op_between_pools():
    """Quantify the repartition: F loses the router, A gains it, nothing else."""
    ops = _moe_layer_ops()
    on_ffn = build_afd_ops_partition(_Model(generation_ops=ops), phase="generation")
    on_attn = build_afd_ops_partition(_Model(generation_ops=_moe_layer_ops()), phase="generation", router_on_attn=True)

    moved_to_attn = set(_names(on_attn.attn_ops)) - set(_names(on_ffn.attn_ops))
    left_ffn = set(_names(on_ffn.ffn_ops)) - set(_names(on_attn.ffn_ops))

    assert moved_to_attn == {"generation_moe_router_gemm"}
    assert left_ffn == {"generation_moe_router_gemm"}
    assert len(on_attn.attn_ops) == len(on_ffn.attn_ops) + 1
    assert len(on_attn.ffn_ops) == len(on_ffn.ffn_ops) - 1

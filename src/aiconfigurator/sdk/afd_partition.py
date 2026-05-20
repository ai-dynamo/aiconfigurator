# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AFD model operation partitioning utilities.

Splits a model's ``context_ops`` / ``generation_ops`` lists into two
parallel groups -- one consumed by the A-worker (attention) pool and
one by the F-worker (FFN / MoE) pool -- so each pool's per-step
latency can be estimated independently. The AFD session wraps the
partitioned lists with cross-pool transfer and intra-pool collective
ops (see ``AFDTransfer``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from aiconfigurator.sdk import operations

AFDPhase = Literal["context", "generation"]
AFDSide = Literal["attn", "ffn", "boundary", "skip"]


class AFDPartitionError(ValueError):
    """Raised when an op sequence cannot be safely partitioned for AFD."""


@dataclass
class AFDOpsPartition:
    """Partitioned AFD operation lists.

    ``boundary_ops`` are also appended to either ``attn_ops`` or ``ffn_ops``
    according to ``boundary_on_attn``. ``skipped_ops`` records model-internal
    communication/dispatch ops that are intentionally not counted in either
    worker pool because AFD models them outside the original model op list.
    """

    attn_ops: list[operations.Operation] = field(default_factory=list)
    ffn_ops: list[operations.Operation] = field(default_factory=list)
    boundary_ops: list[operations.Operation] = field(default_factory=list)
    skipped_ops: list[operations.Operation] = field(default_factory=list)
    phase: AFDPhase = "generation"


def build_afd_ops_partition(
    model,
    phase: AFDPhase = "generation",
    *,
    boundary_on_attn: bool = True,
    allow_unknown_ops: bool = False,
    unknown_side: Literal["attn", "ffn"] = "attn",
) -> AFDOpsPartition:
    """Partition model ops into Attention-side and FFN-side groups for AFD.

    Args:
        model: Model instance exposing ``context_ops`` and ``generation_ops``.
        phase: ``"context"`` or ``"generation"``.
        boundary_on_attn: Assign boundary ops to A-worker when true; otherwise F-worker.
        allow_unknown_ops: If false, unclassified ops raise ``AFDPartitionError``.
        unknown_side: Destination for unknown ops when ``allow_unknown_ops`` is true.
    """

    model_phase = _validate_phase(phase)
    op_sequence = model.context_ops if model_phase == "context" else model.generation_ops
    partition = AFDOpsPartition(phase=model_phase)

    for op in op_sequence:
        side = _classify_op(op, allow_unknown_ops=allow_unknown_ops, unknown_side=unknown_side)
        _append_partition_op(partition, op, side, boundary_on_attn=boundary_on_attn)

    return partition


def _append_partition_op(
    partition: AFDOpsPartition,
    op: operations.Operation,
    side: AFDSide,
    *,
    boundary_on_attn: bool,
) -> None:
    if side == "attn":
        partition.attn_ops.append(op)
    elif side == "ffn":
        partition.ffn_ops.append(op)
    elif side == "boundary":
        partition.boundary_ops.append(op)
        if boundary_on_attn:
            partition.attn_ops.append(op)
        else:
            partition.ffn_ops.append(op)
    elif side == "skip":
        partition.skipped_ops.append(op)


def _classify_op(
    op: operations.Operation,
    *,
    allow_unknown_ops: bool,
    unknown_side: Literal["attn", "ffn"],
) -> AFDSide:
    name = _op_name(op)

    if isinstance(op, operations.OverlapOp):
        return _classify_overlap_op(op, allow_unknown_ops=allow_unknown_ops, unknown_side=unknown_side)
    if _is_skipped_model_internal_op(op, name):
        return "skip"
    _raise_if_unclassifiable_layer_family(op)
    if _is_boundary_op(name):
        return "boundary"
    if _is_attention_side_op(name):
        return "attn"
    if _is_ffn_side_op(name):
        return "ffn"

    return _unknown_or_default(op, allow_unknown_ops=allow_unknown_ops, unknown_side=unknown_side)


def _classify_overlap_op(
    op: operations.OverlapOp,
    *,
    allow_unknown_ops: bool,
    unknown_side: Literal["attn", "ffn"],
) -> AFDSide:
    name = _op_name(op)
    inner_ops = list(getattr(op, "_group_a", [])) + list(getattr(op, "_group_b", []))
    if inner_ops:
        inner_sides = {
            _classify_inner_overlap_op(inner_op, allow_unknown_ops=allow_unknown_ops, unknown_side=unknown_side)
            for inner_op in inner_ops
        }
        inner_sides.discard("skip")
        if len(inner_sides) == 1:
            return inner_sides.pop()
        if not inner_sides:
            return _unknown_or_default(op, allow_unknown_ops=allow_unknown_ops, unknown_side=unknown_side)

        raise AFDPartitionError(f"OverlapOp '{name}' spans A/F boundaries and cannot be kept atomic.")

    if _is_skipped_model_internal_op(op, name):
        return "skip"
    _raise_if_unclassifiable_layer_family(op)
    if _is_boundary_op(name):
        return "boundary"
    if _is_ffn_side_op(name):
        return "ffn"
    if _is_attention_side_op(name):
        return "attn"
    return _unknown_or_default(op, allow_unknown_ops=allow_unknown_ops, unknown_side=unknown_side)


def _classify_inner_overlap_op(
    op: operations.Operation,
    *,
    allow_unknown_ops: bool,
    unknown_side: Literal["attn", "ffn"],
) -> AFDSide:
    if isinstance(op, operations.OverlapOp):
        return _classify_overlap_op(op, allow_unknown_ops=allow_unknown_ops, unknown_side=unknown_side)

    name = _op_name(op)
    if _is_skipped_model_internal_op(op, name):
        return "skip"
    _raise_if_unclassifiable_layer_family(op)
    if _is_boundary_op(name):
        return "boundary"
    if _is_attention_side_op(name):
        return "attn"
    if _is_ffn_side_op(name):
        return "ffn"
    return _unknown_or_default(op, allow_unknown_ops=allow_unknown_ops, unknown_side=unknown_side)


def _validate_phase(phase: str) -> AFDPhase:
    if phase == "context":
        return "context"
    if phase == "generation":
        return "generation"
    raise AFDPartitionError(f"build_afd_ops_partition: phase must be 'context' or 'generation', got {phase!r}")


def _op_name(op: operations.Operation) -> str:
    return str(getattr(op, "_name", op.__class__.__name__)).lower()


def _unknown_or_default(
    op: operations.Operation,
    *,
    allow_unknown_ops: bool,
    unknown_side: Literal["attn", "ffn"],
) -> AFDSide:
    if allow_unknown_ops:
        return unknown_side
    raise AFDPartitionError(f"Cannot classify op '{_op_name(op)}' for AFD partitioning.")


def _is_skipped_model_internal_op(op: operations.Operation, name: str) -> bool:
    # TODO(afd, Phase-2): when an ``MoEDispatch`` op (name contains ``"dispatch"``)
    # appears inside an ``OverlapOp`` (e.g. ``generation_moe_overlap``), this
    # skip-list only excludes it from the overlap *classification vote* --
    # its cost is still folded into the OverlapOp's F-pool latency via
    # ``OverlapOp.query()`` and stays invisible to ``AFDTransfer`` /
    # ``_pipeline_tcycle``. That hides the MoE EP all-to-all from the AFD
    # pipeline overlap math, so contention between the MoE all-to-all and
    # the cross-pool A<->F transfer on the same NIC fabric is silently
    # dropped. Surface ``MoEDispatch`` cost as an additional contribution
    # to ``t_c_layer`` (or split the OverlapOp into compute + dispatch
    # sub-stages) when AFD is active, instead of folding it into ``t_f``.
    if isinstance(op, (operations.CustomAllReduce, operations.P2P, operations.NCCL)):
        return True

    return any(
        marker in name
        for marker in (
            "_p2p",
            "_ar",
            "all_reduce",
            "allreduce",
            "tp_all_gather",
            "tp_reduce_scatter",
            "dispatch",
        )
    )


def _raise_if_unclassifiable_layer_family(op: operations.Operation) -> None:
    name = _op_name(op)
    family = _unclassifiable_layer_family(name)
    if family is None:
        return

    raise AFDPartitionError(
        f"AFD op partitioner cannot safely classify {family} op '{name}'. "
        f"{family} layers are not covered by the current attention/FFN partition rules. "
        "Add an explicit AFD partitioning rule before running estimate-mode afd "
        "for this model."
    )


def _unclassifiable_layer_family(name: str) -> str | None:
    if "mamba" in name:
        return "Mamba"
    if "gdn" in name:
        return "GDN"
    return None


def _is_boundary_op(name: str) -> bool:
    return any(
        marker in name
        for marker in (
            "logits_gemm",
            "reduce_add",
            "add_norm_2",
            "_ffn_norm",
            "_moe_norm",
            "_mlp_norm",
            "_dense_ffn_norm",
        )
    ) or name.endswith("_combine")


def _is_attention_side_op(name: str) -> bool:
    return any(
        marker in name
        for marker in (
            "embedding",
            "add_norm_1",
            "_attn_norm",
            "attention",
            "qkv",
            "q_a_layernorm",
            "q_b_proj",
            "kv_b_proj",
            "downscale_gemm",
            "mla",
            "bmm",
            "rope",
            "proj_gemm",
        )
    )


def _is_ffn_side_op(name: str) -> bool:
    # TODO(afd, Phase-2): the ``"shared"`` marker pins shared-expert ops to
    # the F-Worker, but the optimal placement is topology-dependent. From a
    # comm standpoint shared belongs on A (every token is already there, no
    # cross-pool replication needed); from a compute standpoint it belongs
    # on F (scale-out across F GPUs and TP-shard the GEMMs). Add a knob --
    # ``AFDConfig.shared_on_attn`` parallel to ``boundary_on_attn`` -- so
    # callers can flip the side without editing this classifier. Note that
    # moving shared to A also requires AFDTransfer to drop the redundant
    # shared-replication term added by the companion TODO in ``operations.py``.
    return any(
        marker in name
        for marker in (
            "router",
            "moe",
            "ffn",
            "mlp",
            "shared",
            "expert",
            "act_gate",
            "_act",
            "gate_up",
            "gate_ffn",
            "up_gemm",
            "down_gemm",
            "relu",
        )
    )

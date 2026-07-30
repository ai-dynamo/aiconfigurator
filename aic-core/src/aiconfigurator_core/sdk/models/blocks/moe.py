# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MoE block shape descriptor.

:class:`MoEBlockShape` captures the checkpoint-level geometry of a model's MoE
block: the expert GEMM dimensions, routing width, shared-expert count, and how
many transformer layers carry an MoE FFN. It is derived from the
``_get_model_info`` dict (HF ``config.json`` parse + the derived
``num_shared_experts`` / ``num_moe_layers`` fields) and consumed by the generic
MoE-block builder.
"""

from __future__ import annotations

from dataclasses import dataclass

from aiconfigurator_core.sdk.models.helpers import check_is_moe


@dataclass(frozen=True)
class MoEBlockShape:
    """Checkpoint-level shape of a model's MoE block(s).

    Attributes:
        hidden_size: Model hidden size (MoE GEMM K/N dimension).
        moe_inter_size: Per-expert FFN intermediate size.
        topk: Number of routed experts activated per token.
        num_experts: Total routed-expert count.
        num_shared_experts: Shared (always-active) expert count; 0 when the
            checkpoint has none.
        num_moe_layers: Number of transformer layers that carry an MoE block.
        is_gated: Whether the expert FFN is gated (SwiGLU-style gate+up).
    """

    hidden_size: int
    moe_inter_size: int
    topk: int
    num_experts: int
    num_shared_experts: int  # 0 when absent
    num_moe_layers: int  # layers that carry an MoE block
    is_gated: bool = True

    @classmethod
    def from_model_info(cls, model_info: dict) -> MoEBlockShape:
        """Build the shape from a ``_get_model_info`` dict.

        Raises:
            ValueError: If the model is not a MoE model (same signal as
                :func:`check_is_moe`).
        """
        if not check_is_moe(model_info.get("model_path", ""), model_info=model_info):
            raise ValueError(f"Model with architecture {model_info.get('architecture')!r} is not a MoE model")
        return cls(
            hidden_size=model_info["hidden_size"],
            moe_inter_size=model_info["moe_inter_size"],
            topk=model_info["topk"],
            num_experts=model_info["num_experts"],
            num_shared_experts=model_info["num_shared_experts"],
            num_moe_layers=model_info["num_moe_layers"],
        )

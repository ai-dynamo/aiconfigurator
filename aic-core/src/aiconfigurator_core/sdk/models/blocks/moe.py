# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MoE block shape descriptor and the generic MoE-block builder.

:class:`MoEBlockShape` captures the checkpoint-level geometry of a model's MoE
block: the expert GEMM dimensions, routing width, shared-expert count, and how
many transformer layers carry an MoE FFN. It is derived from the
``_get_model_info`` dict (HF ``config.json`` parse + the derived
``num_shared_experts`` / ``num_moe_layers`` fields) and consumed by the generic
MoE-block builder.

:func:`build_moe_block_ops` is the one place MoE blocks are wired: model
classes keep attention/dense wiring and hand the shape (plus their model-owned
workload-distribution string and scale factor) to the builder, which emits
router GEMM, shared-expert GEMMs, and the dispatch/compute/combine ops.
Family/framework/system-specific deviations register through
:func:`register_moe_block` (the G3 escape hatch) instead of new model classes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import aiconfigurator_core.sdk.operations as ops
from aiconfigurator_core.sdk import common
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


# ---------------------------------------------------------------------------
# Builder specialization registry (the G3 escape hatch)
# ---------------------------------------------------------------------------

#: Registered builder variants keyed by ``(family, framework, system)`` where
#: ``"*"`` is a per-position wildcard. Module-level state: tests that register
#: variants must snapshot/restore this dict (fixture pattern in
#: ``tests/unit/sdk/models/test_moe_block_builder.py``).
_MOE_BLOCK_REGISTRY: dict[tuple[str, str, str], Callable] = {}


def register_moe_block(family: str = "*", framework: str = "*", system: str = "*") -> Callable:
    """Register a MoE-block builder variant for ``(family, framework, system)``.

    The decorated function is called as ``fn(default, **ctx)`` where
    ``default`` is a zero-argument continuation returning a fresh copy of the
    generic pipeline's ops (compose with it rather than reimplementing) and
    ``ctx`` carries the full :func:`build_moe_block_ops` parameter set:
    ``prefix``/``shape``/``cfg``/``quant_mode``/``workload_distribution``/
    ``scale_factor``/``backend_name``/``inference_phase``/``attn_cp_size``/
    ``gpus_per_node``. It must return the block's op list.
    """

    def _decorator(fn: Callable) -> Callable:
        _MOE_BLOCK_REGISTRY[(family, framework, system)] = fn
        return fn

    return _decorator


def _match_rank(key: tuple[str, str, str], query: tuple[str | None, str | None, str | None]) -> int:
    """Specificity of ``key`` against ``query``; -1 when the key does not match.

    Exact beats wildcard per position with left-to-right priority
    family > framework > system, encoded as the 3-bit number
    ``(family_exact, framework_exact, system_exact)``. Ties are impossible:
    for a fixed query the exact positions determine the key, so two distinct
    matching keys always differ in rank. Unknown query values (``None``)
    match only wildcard positions.
    """
    rank = 0
    for bit, want, have in zip((4, 2, 1), key, query, strict=True):
        if want == "*":
            continue
        if want != have:
            return -1
        rank += bit
    return rank


def _select_moe_block_variant(family: str | None, framework: str | None, system: str | None) -> Callable | None:
    """Most-specific-wins lookup; ``None`` when no registered variant matches."""
    query = (family, framework, system)
    best_fn = None
    best_rank = -1
    for key, fn in _MOE_BLOCK_REGISTRY.items():
        rank = _match_rank(key, query)
        if rank > best_rank:
            best_rank = rank
            best_fn = fn
    return best_fn


# ---------------------------------------------------------------------------
# Generic MoE-block builder
# ---------------------------------------------------------------------------


def build_moe_block_ops(
    prefix: str,  # "context" | "generation"
    shape: MoEBlockShape,
    cfg,  # ModelConfig
    quant_mode,  # common.MoEQuantMode
    workload_distribution: str,  # model-owned alpha string, e.g. "power_law_1.01"
    *,
    scale_factor: float,  # num layers x mtp factor — model-owned (NOT shape.num_moe_layers)
    backend_name: str,  # "sglang" | "vllm" | "trtllm"
    inference_phase: str,  # "context" | "generation"
    attn_cp_size: int = 1,
    gpus_per_node: int = 8,
) -> list:
    """Build the MoE-block op list: router, shared experts, dispatch/compute/combine.

    ``scale_factor`` is deliberately caller-supplied: legacy model classes
    scale their MoE ops by their OWN layer count (e.g. DeepSeek uses all 61
    layers, not the 58 MoE-true ``shape.num_moe_layers``) and gate parity
    depends on passing that legacy value through unchanged.

    Dispatches to a :func:`register_moe_block` variant when one matches
    ``(family, framework, system)``; the family/system query values are read
    from optional ``cfg`` attributes (``model_family`` / ``system``) — absent
    attributes match only wildcard registrations.
    """
    ctx = {
        "prefix": prefix,
        "shape": shape,
        "cfg": cfg,
        "quant_mode": quant_mode,
        "workload_distribution": workload_distribution,
        "scale_factor": scale_factor,
        "backend_name": backend_name,
        "inference_phase": inference_phase,
        "attn_cp_size": attn_cp_size,
        "gpus_per_node": gpus_per_node,
    }

    def default() -> list:
        return _default_moe_block_ops(**ctx)

    variant = _select_moe_block_variant(
        family=getattr(cfg, "model_family", None),
        framework=backend_name,
        system=getattr(cfg, "system", None),
    )
    if variant is None:
        return default()
    return variant(default, **ctx)


def _default_moe_block_ops(
    prefix: str,
    shape: MoEBlockShape,
    cfg,
    quant_mode,
    workload_distribution: str,
    scale_factor: float,
    backend_name: str,
    inference_phase: str,
    attn_cp_size: int,
    gpus_per_node: int,
) -> list:
    """The generic pipeline: verbatim transcription of the legacy fused sites.

    Context-phase CP kwargs mirror the legacy sites exactly: token-major ops
    get ``seq_split``, dispatches get ``attn_cp_size``. Generation is not
    CP-modeled (the legacy sites pass neither kwarg there).
    """
    is_context = inference_phase == "context"
    seq_split_kwargs = {"seq_split": attn_cp_size} if is_context else {}
    dispatch_cp_kwargs = {"attn_cp_size": attn_cp_size} if is_context else {}

    # Router GEMM: hidden_size -> num_experts, always emitted (spec section 4.4.4).
    # Transcribed from MOEModel.__init__ (models/moe.py:181-192 context,
    # :272-282 generation).
    block_ops = [
        ops.GEMM(
            f"{prefix}_router_gemm",
            scale_factor,
            shape.num_experts,
            shape.hidden_size,
            common.GEMMQuantMode.bfloat16,
            **seq_split_kwargs,
        )
    ]

    # Shared experts: gate+up fused into one GEMM (matches TRT-LLM GatedMLP),
    # replicated per rank under ADP. Transcribed from DeepSeekModel.__init__
    # (models/deepseek.py:219-246 context, :445-467 generation), which sizes
    # ``2 * moe_inter_size // tp`` with exactly one shared expert; the generic
    # form scales the intermediate size by ``num_shared_experts``.
    if shape.num_shared_experts > 0:
        shared_inter_size = shape.num_shared_experts * shape.moe_inter_size
        block_ops.extend(
            [
                ops.GEMM(
                    f"{prefix}_shared_gate_up_gemm",
                    scale_factor,
                    2 * shared_inter_size // cfg.tp_size,
                    shape.hidden_size,
                    cfg.gemm_quant_mode,
                    **seq_split_kwargs,
                ),
                ops.ElementWise(
                    f"{prefix}_shared_act_gate",
                    scale_factor,
                    2 * shared_inter_size // cfg.tp_size,
                    shared_inter_size // cfg.tp_size,
                    0.8,
                    **seq_split_kwargs,
                ),
                ops.GEMM(
                    f"{prefix}_shared_ffn2_gemm",
                    scale_factor,
                    shape.hidden_size,
                    shared_inter_size // cfg.tp_size,
                    cfg.gemm_quant_mode,
                    **seq_split_kwargs,
                ),
            ]
        )

    # Large-EP seam: a per-phase comm backend on cfg selects the MoEAllToAll +
    # EPMoE emission. ``moe_comm_backend`` (dict[str, str] | None) does not
    # exist on ModelConfig yet — hence getattr; absent/uncovered phase means
    # the fused path below.
    comm_backend = (getattr(cfg, "moe_comm_backend", None) or {}).get(inference_phase)
    if comm_backend:
        raise NotImplementedError("large-EP emission lands in the next commit")

    # Fused/small-EP path: dispatch tokens to experts, moe calc and get tokens
    # back. Transcribed from MOEModel.__init__ (models/moe.py:195-237 context,
    # :285-325 generation) — argument lists value-identical.
    block_ops.extend(
        [
            ops.MoEDispatch(
                f"{prefix}_moe_pre_dispatch",
                scale_factor,
                shape.hidden_size,
                shape.topk,
                shape.num_experts,
                cfg.moe_tp_size,
                cfg.moe_ep_size,
                cfg.attention_dp_size,
                True,
                quant_mode=quant_mode,
                **dispatch_cp_kwargs,
            ),
            ops.MoE(
                f"{prefix}_moe",
                scale_factor,
                shape.hidden_size,
                shape.moe_inter_size,
                shape.topk,
                shape.num_experts,
                cfg.moe_tp_size,
                cfg.moe_ep_size,
                quant_mode,
                workload_distribution,
                cfg.attention_dp_size,
            ),
            ops.MoEDispatch(
                f"{prefix}_moe_post_dispatch",
                scale_factor,
                shape.hidden_size,
                shape.topk,
                shape.num_experts,
                cfg.moe_tp_size,
                cfg.moe_ep_size,
                cfg.attention_dp_size,
                False,
                quant_mode=quant_mode,
                **dispatch_cp_kwargs,
            ),
        ]
    )
    return block_ops

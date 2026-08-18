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

:data:`LARGE_EP_READY_FAMILIES` names the model families whose classes are
wired for the large-EP emission below — the enumerator's assignment gate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import aiconfigurator_core.sdk.operations as ops
from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.models.helpers import check_is_moe
from aiconfigurator_core.sdk.operations.moe_comm import MOE_A2A_BACKENDS, nodes_for

#: Model families whose classes construct a large-EP graph when the enumerator
#: sets ``ModelConfig.moe_comm_backend``. The enumerator must never assign a
#: comm backend outside this set: HYBRIDMOE / MINIMAXM3 raise on it by design
#: (their shared-expert wiring is not reproduced by the large-EP branch), and
#: DEEPSEEKV4 owns its own MegaMoE path. QWEN3VL_MOE is deliberately excluded
#: even though it rides MOEModel (``backend_name`` now threads through —
#: MoEDispatch's comm flavor is a constructor field — but enabling the
#: large-EP branch for it remains a documented follow-up).
LARGE_EP_READY_FAMILIES = frozenset({"MOE", "DEEPSEEK", "DEEPSEEKV32", "KIMIK25"})


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
    ``ctx`` carries the :func:`build_moe_block_ops` parameter set:
    ``prefix``/``shape``/``cfg``/``quant_mode``/``workload_distribution``/
    ``scale_factor``/``backend_name``/``inference_phase``/``model_family``/
    ``attn_cp_size``/``gpus_per_node``/``shared_gemm_quant_mode``
    (``dispatch_quant_mode`` deliberately rides the ``default`` continuation
    instead — the ctx key set is a pinned contract). It must return the
    block's op list.

    Raises:
        ValueError: On a duplicate ``(family, framework, system)`` key — a
            collision is a wiring bug, never a legitimate override.
    """

    def _decorator(fn: Callable) -> Callable:
        key = (family, framework, system)
        if key in _MOE_BLOCK_REGISTRY:
            raise ValueError(f"duplicate MoE-block variant registration for {key}")
        _MOE_BLOCK_REGISTRY[key] = fn
        return fn

    return _decorator


def _match_rank(key: tuple[str, str, str], query: tuple[str | None, str | None, str | None]) -> int:
    """Specificity of ``key`` against ``query``; -1 when the key does not match.

    Exact beats wildcard per position with left-to-right priority
    family > framework > system, encoded as the 3-bit number
    ``(family_exact, framework_exact, system_exact)``. Ties are impossible:
    for a fixed query the exact positions determine the key, so two distinct
    matching keys always differ in rank. Unknown query values (``None``, or
    the ``"*"`` default of ``model_family``) match only wildcard positions.
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

#: Default for ``build_moe_block_ops(dispatch_quant_mode=...)``: forward the
#: block's ``quant_mode`` to the fused MoEDispatch ops (what every legacy fused
#: site does). A distinct sentinel rather than ``None`` because ``None`` is a
#: meaningful override — the hybrid family's dispatches are quant-agnostic.
_DISPATCH_QUANT_FORWARD = object()


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
    model_family: str = "*",  # registry family axis; "*" matches only wildcard registrations
    attn_cp_size: int = 1,
    gpus_per_node: int | None = None,  # node width (hardware fact) — required by the large-EP emission
    shared_gemm_quant_mode=None,  # common.GEMMQuantMode | None
    dispatch_quant_mode=_DISPATCH_QUANT_FORWARD,  # common.MoEQuantMode | None; unset forwards ``quant_mode``
) -> list:
    """Build the MoE-block op list: router, shared experts, dispatch/compute/combine.

    ``scale_factor`` is deliberately caller-supplied: legacy model classes
    scale their MoE ops by their OWN layer count (e.g. DeepSeek uses all 61
    layers, not the 58 MoE-true ``shape.num_moe_layers``) and gate parity
    depends on passing that legacy value through unchanged.

    ``gpus_per_node`` has no default: the node width is a hardware fact, and a
    guessed value would silently mis-price cross-node all-to-all, so the
    large-EP branch raises when a comm backend fires without it (the fused
    path never reads it — fused-only callers may omit it).

    ``shared_gemm_quant_mode`` overrides ``cfg.gemm_quant_mode`` for the
    shared-expert GEMMs only (``None`` keeps them on the model-wide mode).
    Checkpoints may exclude the shared experts from quantization while the
    routed experts stay quantized — e.g. ``nvidia/GLM-5.2-NVFP4`` ignores
    every ``mlp.shared_experts*`` module, which the DeepSeekV32 family reads
    into ``extra_params["dsa_shared_expert_quant_mode"]``.

    ``dispatch_quant_mode`` overrides the quant mode of the FUSED path's
    MoEDispatch ops only; when unset they forward ``quant_mode``. ``None`` is
    a meaningful override (quant-agnostic dispatches — the hybrid family's
    legacy spans never carried one), hence the non-``None`` sentinel default.
    The large-EP branch keys its comm dtypes off ``quant_mode`` and ignores
    this parameter.

    Dispatches to a :func:`register_moe_block` variant when one matches
    ``(model_family, backend_name, system)``; the system query value is read
    from the optional ``cfg.system`` attribute — an absent attribute matches
    only wildcard registrations.
    """
    assert prefix == inference_phase, (
        f"prefix {prefix!r} must equal the inference_phase-derived prefix {inference_phase!r} "
        "(context->'context', generation->'generation'); a mismatch is a caller bug"
    )
    if gpus_per_node is None:
        if getattr(cfg, "moe_comm_backend", None):
            # Topology has no safe default (this PR's own rule): resolve the
            # omitted argument from the config's validated value instead of a
            # silent eight-GPU assumption — a GB200-style
            # cfg.num_gpus_per_node=4 at EP16 is a four-node all-to-all, not
            # two. Raises when the config carries no value either.
            from aiconfigurator_core.sdk.models.helpers import large_ep_gpus_per_node

            gpus_per_node = large_ep_gpus_per_node(cfg)
        else:
            # Fused block: no all-to-all is emitted, the coordinate is unused.
            gpus_per_node = 0
    ctx = {
        "prefix": prefix,
        "shape": shape,
        "cfg": cfg,
        "quant_mode": quant_mode,
        "workload_distribution": workload_distribution,
        "scale_factor": scale_factor,
        "backend_name": backend_name,
        "inference_phase": inference_phase,
        "model_family": model_family,
        "attn_cp_size": attn_cp_size,
        "gpus_per_node": gpus_per_node,
        "shared_gemm_quant_mode": shared_gemm_quant_mode,
    }

    def default() -> list:
        # ``dispatch_quant_mode`` rides the continuation, NOT ctx: the ctx key
        # set is a pinned variant contract (test_variant_receives_full_ctx).
        return _default_moe_block_ops(dispatch_quant_mode=dispatch_quant_mode, **ctx)

    variant = _select_moe_block_variant(
        family=model_family,
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
    model_family: str,
    attn_cp_size: int,
    gpus_per_node: int | None,
    shared_gemm_quant_mode=None,
    dispatch_quant_mode=_DISPATCH_QUANT_FORWARD,
) -> list:
    """The generic pipeline: verbatim transcription of the legacy fused sites.

    Context-phase CP kwargs mirror the legacy sites exactly: token-major ops
    get ``seq_split``, dispatches get ``attn_cp_size``. Generation is not
    CP-modeled (the legacy sites pass neither kwarg there).
    """
    is_context = inference_phase == "context"
    seq_split_kwargs = {"seq_split": attn_cp_size} if is_context else {}
    dispatch_cp_kwargs = {"attn_cp_size": attn_cp_size} if is_context else {}

    # Large-EP branch: a per-phase comm backend on cfg selects the
    # MoEAllToAll + MoEExpertCompute emission (with its own shared-expert flavor).
    # ``moe_comm_backend`` (dict[str, str] | None) is read with getattr on
    # purpose: the builder duck-types ``cfg`` (same contract as the optional
    # ``cfg.system`` registry axis), so lightweight config doubles without the
    # attribute keep working; absent/uncovered phase means the fused path below.
    comm_backend = (getattr(cfg, "moe_comm_backend", None) or {}).get(inference_phase)
    if comm_backend:
        return _large_ep_block_ops(
            comm_backend,
            prefix=prefix,
            shape=shape,
            cfg=cfg,
            quant_mode=quant_mode,
            workload_distribution=workload_distribution,
            scale_factor=scale_factor,
            backend_name=backend_name,
            inference_phase=inference_phase,
            attn_cp_size=attn_cp_size,
            gpus_per_node=gpus_per_node,
            shared_gemm_quant_mode=shared_gemm_quant_mode,
        )

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
        shared_quant_mode = cfg.gemm_quant_mode if shared_gemm_quant_mode is None else shared_gemm_quant_mode
        block_ops.extend(
            [
                ops.GEMM(
                    f"{prefix}_shared_gate_up_gemm",
                    scale_factor,
                    2 * shared_inter_size // cfg.tp_size,
                    shape.hidden_size,
                    shared_quant_mode,
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
                    shared_quant_mode,
                    **seq_split_kwargs,
                ),
            ]
        )

    # Fused/small-EP path: dispatch tokens to experts, moe calc and get tokens
    # back. Transcribed from MOEModel.__init__ (models/moe.py:195-237 context,
    # :285-325 generation) — argument lists value-identical. The dispatches
    # forward the block quant mode unless the caller overrode it (``None`` =
    # quant-agnostic, the hybrid family's legacy dispatch flavor).
    if dispatch_quant_mode is _DISPATCH_QUANT_FORWARD:
        dispatch_quant_mode = quant_mode
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
                quant_mode=dispatch_quant_mode,
                **dispatch_cp_kwargs,
                backend=backend_name,
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
                quant_mode=dispatch_quant_mode,
                **dispatch_cp_kwargs,
                backend=backend_name,
            ),
        ]
    )
    return block_ops


# ---------------------------------------------------------------------------
# Large-EP emission (cfg.moe_comm_backend selects a MoEAllToAll/MoEExpertCompute graph)
# ---------------------------------------------------------------------------


def _dispatch_dtype(comm_backend: str, quant_mode) -> str:
    """Comm-table dtype key for the prepare/dispatch phases.

    DeepEP rows have no dtype axis — the adapted tables key everything under
    ``"default"`` (moe_comm.py ``_adapt_legacy_deepep``). The trtllm nvlink
    rows key the run's ``moe_dtype`` string, i.e. the ``MoEQuantMode`` member
    name (``_adapt_legacy_trtllm_alltoall`` passes the parquet string through
    and the legacy loader spells it via ``MoEQuantMode[...]``); ``fp8_block``
    resolves to the ``fp8`` rows at query time — the same behavioral aliasing
    the legacy ``_normalize_quant_mode_for_table`` applied.
    """
    if comm_backend.startswith("deepep"):
        return "default"
    return quant_mode.name


def _combine_dtype(comm_backend: str, quant_mode, inference_phase: str) -> str:
    """Comm-table dtype key for the combine phase.

    DeepEP: ``"default"`` (no dtype axis). nvlink: the adapted tables pin the
    low-precision combine kernel under ``"fp4"``; the legacy graph enables it
    only in GENERATION for nvfp4 runs (``use_low_precision_combine=
    (moe_quant_mode == nvfp4)``, deepseek.py:1005-1011) while the context
    post_dispatch site (deepseek.py:798-812) never passes the flag — context
    combine stays on the standard rows keyed by the run dtype.
    """
    if comm_backend.startswith("deepep"):
        return "default"
    if inference_phase == "generation" and quant_mode == common.MoEQuantMode.nvfp4:
        return "fp4"
    return quant_mode.name


def _large_ep_shared_expert_ops(
    prefix: str,
    shape: MoEBlockShape,
    cfg,
    scale_factor: float,
    backend_name: str,
    is_context: bool,
    attn_cp_size: int,
    shared_gemm_quant_mode=None,
) -> list:
    """Shared experts under a large-EP comm backend: FULL weights, no ÷tp.

    Both legacy wideEP graphs replicate the whole shared expert per rank
    (ADP mode, ``shared_tp_size=1``):

    - trtllm (deepseek.py:720-744 context, :940-962 generation): the fused
      ``{prefix}_shared_*`` names at full ``2 * moe_inter_size``, no token
      scaling and no CP kwargs (TrtllmWideEP rejects CP).
    - sglang/vllm deepep (deepseek.py:1174-1204 context, :1294-1318
      generation): the ``{prefix}_gate_ffn1_gemm`` / ``{prefix}_act_gate`` /
      ``{prefix}_ffn2_gemm`` names — full size, with the CONTEXT triplet
      carrying ``scale_num_tokens=tp_size`` (attention TP shards the token
      stream) plus ``seq_split``; the generation site passes neither.
    """
    if shape.num_shared_experts == 0:
        return []
    shared_inter_size = shape.num_shared_experts * shape.moe_inter_size
    shared_quant_mode = cfg.gemm_quant_mode if shared_gemm_quant_mode is None else shared_gemm_quant_mode
    if backend_name == "trtllm":
        names = (f"{prefix}_shared_gate_up_gemm", f"{prefix}_shared_act_gate", f"{prefix}_shared_ffn2_gemm")
        token_kwargs = {}
    else:
        names = (f"{prefix}_gate_ffn1_gemm", f"{prefix}_act_gate", f"{prefix}_ffn2_gemm")
        token_kwargs = {"scale_num_tokens": cfg.tp_size, "seq_split": attn_cp_size} if is_context else {}
    return [
        ops.GEMM(
            names[0],
            scale_factor,
            2 * shared_inter_size,
            shape.hidden_size,
            shared_quant_mode,
            **token_kwargs,
        ),
        ops.ElementWise(
            names[1],
            scale_factor,
            2 * shared_inter_size,
            shared_inter_size,
            0.8,
            **token_kwargs,
        ),
        ops.GEMM(
            names[2],
            scale_factor,
            shape.hidden_size,
            shared_inter_size,
            shared_quant_mode,
            **token_kwargs,
        ),
    ]


def _large_ep_block_ops(
    comm_backend: str,
    *,
    prefix: str,
    shape: MoEBlockShape,
    cfg,
    quant_mode,
    workload_distribution: str,
    scale_factor: float,
    backend_name: str,
    inference_phase: str,
    attn_cp_size: int,
    gpus_per_node: int | None,
    shared_gemm_quant_mode=None,
) -> list:
    """Large-EP branch: router + shared experts + A2A dispatch/MoEExpertCompute/combine.

    Fidelity notes (each transcribed from the legacy wideEP graphs):

    - ``node_num = nodes_for(moe_ep * moe_tp, gpus_per_node)`` (A5): both
      legacy classes derive the comm node span from the whole MoE group,
      which coincides only under pp=1 configs.
    - ``attention_tp_size=cfg.tp_size`` only for deepep backends in CONTEXT:
      the legacy dispatch sites pass ``scale_num_tokens=tp_size`` at the
      context sites only (deepseek.py:1206-1227, models/moe.py context site);
      every generation site uses the default divisor 1 (deepseek.py:
      1320-1340, models/moe.py generation site). The legacy trtllm alltoall
      queried undivided tokens -> nvlink passes 1 in both phases.
    - ``sms`` rides only ``deepep_ht`` (the legacy normal-mode table keys an
      SM budget; LL and nvlink rows carry none).
    - ``enable_eplb`` reaches MoEExpertCompute only for deepep backends: the sglang MoE
      query corrects prefill tokens by 0.8 under EPLB, while trtllm EPLB
      rides the ``_eplb`` workload-distribution suffix instead.
    - trtllm structure: when the shape has shared experts, a trailing
      ``{prefix}_moe_reduce_add`` ElementWise (deepseek.py:816-824 context,
      :1022-1032 generation — it models the routed-topk + SHARED add) and,
      in generation, the routed/shared OverlapOp (deepseek.py:1014-1020);
      shared-less shapes stay flat with neither.
    """
    if gpus_per_node is None:
        raise ValueError(
            f"moe_comm_backend {comm_backend!r} selected the large-EP emission but gpus_per_node "
            "was not provided — the node width is a hardware fact with no safe default (a guessed "
            "value silently mis-prices cross-node all-to-all; see models.helpers.large_ep_gpus_per_node)"
        )
    is_context = inference_phase == "context"
    seq_split_kwargs = {"seq_split": attn_cp_size} if is_context else {}
    is_deepep = comm_backend.startswith("deepep")

    spec = MOE_A2A_BACKENDS[comm_backend]
    node_num = nodes_for(cfg.moe_ep_size * cfg.moe_tp_size, gpus_per_node)  # A5
    a2a_kwargs = {
        "comm_backend": comm_backend,
        "hidden_size": shape.hidden_size,
        "topk": shape.topk,
        "num_experts": shape.num_experts,
        "moe_ep_size": cfg.moe_ep_size,
        "node_num": node_num,
        "sms": cfg.sms if comm_backend == "deepep_ht" else 0,
        # DeepEP context receives per-attention-rank tokens. Context
        # parallelism contributes to that attention width just like TP does;
        # generation remains unsharded here.
        "attention_tp_size": cfg.tp_size * cfg.cp_size if is_deepep and is_context else 1,
    }

    # Routed path: router GEMM (spec section 4.4.4 — always emitted here; the
    # DeepSeek-sglang registered variants strip it for legacy fidelity), then
    # prepare (when the backend declares it), dispatch, expert compute, combine.
    routed_ops = [
        ops.GEMM(
            f"{prefix}_router_gemm",
            scale_factor,
            shape.num_experts,
            shape.hidden_size,
            common.GEMMQuantMode.bfloat16,
            **seq_split_kwargs,
        )
    ]
    for comm_phase in spec.comm_phases[:-1]:  # prepare (if declared), dispatch
        routed_ops.append(
            ops.MoEAllToAll(
                f"{prefix}_moe_{comm_phase}",
                scale_factor,
                phase=comm_phase,
                comm_dtype=_dispatch_dtype(comm_backend, quant_mode),
                **a2a_kwargs,
            )
        )
    routed_ops.append(
        ops.MoEExpertCompute(
            f"{prefix}_moe",
            scale_factor,
            hidden_size=shape.hidden_size,
            inter_size=shape.moe_inter_size,
            topk=shape.topk,
            num_experts=shape.num_experts,
            moe_ep_size=cfg.moe_ep_size,
            quant_mode=quant_mode,
            workload_distribution=workload_distribution,
            attention_dp_size=cfg.attention_dp_size,
            inference_phase=inference_phase,
            num_slots=cfg.wideep_num_slots or None,
            is_gated=shape.is_gated,
            enable_eplb=cfg.enable_eplb and is_deepep,
        )
    )
    routed_ops.append(
        ops.MoEAllToAll(
            f"{prefix}_moe_combine",
            scale_factor,
            phase="combine",
            comm_dtype=_combine_dtype(comm_backend, quant_mode, inference_phase),
            **a2a_kwargs,
        )
    )

    shared_ops = _large_ep_shared_expert_ops(
        prefix, shape, cfg, scale_factor, backend_name, is_context, attn_cp_size, shared_gemm_quant_mode
    )

    if backend_name == "trtllm" and shared_ops:
        # moe_reduce_add_shared_output: sum routed output over top_k + add
        # shared output (deepseek.py:816-824, :1022-1032) — only meaningful
        # when there IS a shared output to add.
        reduce_add = ops.ElementWise(
            f"{prefix}_moe_reduce_add", scale_factor, 2 * shape.hidden_size, shape.hidden_size, 0.8
        )
        if not is_context:
            # Generation overlaps shared/routed on parallel streams (CUDA
            # Graph, deepseek.py:1014-1020); context runs sequentially.
            return [ops.OverlapOp(f"{prefix}_moe_overlap", group_a=routed_ops, group_b=shared_ops), reduce_add]
        return [routed_ops[0], *shared_ops, *routed_ops[1:], reduce_add]
    return [routed_ops[0], *shared_ops, *routed_ops[1:]]


# ---------------------------------------------------------------------------
# Registered variants shipped with the builder
# ---------------------------------------------------------------------------


@register_moe_block(family="DEEPSEEK", framework="sglang")
def _deepseek_sglang_moe_block(default, *, prefix, cfg, inference_phase, **_ctx):
    """DeepSeek-on-sglang router fidelity (A3): strip the router under deepep.

    The legacy sglang wideEP DeepSeek graphs (WideEPDeepSeekModel /
    WideEPDeepSeekV32Model) never wire a router GEMM on the deepep path,
    while the generic pipeline always emits one (spec section 4.4.4). Any
    other phase/backend combination returns ``default()`` unchanged.
    """
    block_ops = default()
    comm_backend = (getattr(cfg, "moe_comm_backend", None) or {}).get(inference_phase)
    if comm_backend and comm_backend.startswith("deepep"):
        return [op for op in block_ops if op._name != f"{prefix}_router_gemm"]
    return block_ops


register_moe_block(family="DEEPSEEKV32", framework="sglang")(_deepseek_sglang_moe_block)

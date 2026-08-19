# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MoE-block wiring goldens for the fused-only families (Task 7).

``HybridMoEModel`` and ``MiniMaxM3Model`` emit their MoE blocks through
``build_moe_block_ops``. Zero-regression contract against the legacy
hand-wired spans (verified mechanically against the ac6262c archive), with
EXACTLY one ledgered delta:

- A8 (``router-gemm-hybrid-moe``): the legacy hybrid span gated its router
  GEMM on ``num_experts >= 128``; the builder always emits it (spec section
  4.4.4). Llama-4-Scout-class models (<128 experts) gain one router GEMM per
  MoE layer type and phase. MiMo-V2-Flash (256) and Llama-4-Maverick (128)
  already emitted it, so their graphs are unchanged.

Family fidelity pins that must NOT drift:

- hybrid dispatch ops stay quant-agnostic (``_quant_mode is None``): the
  legacy span never forwarded ``quant_mode`` while the builder's fused path
  always does, so the family resets it on the returned ops.
- hybrid CP stays post-hoc: ``set_hybrid_config`` mutates the returned ops
  (``_attn_cp_size`` / ``_is_context`` / ``_seq_split``) after emission.
- minimax keeps its shared triplet hand-wired (context: BEFORE the router;
  generation: the OverlapOp's group_b) and its dispatches carry the run's
  MoE quant mode.
"""

from __future__ import annotations

import pytest

import aiconfigurator_core.sdk.operations as ops
from aiconfigurator_core.sdk import config
from aiconfigurator_core.sdk.models import get_model
from aiconfigurator_core.sdk.models.blocks import moe as blocks_moe

pytestmark = pytest.mark.unit

MIMO = "XiaomiMiMo/MiMo-V2-Flash"  # 256 experts, 48 layers (24 global_moe / 23 swa_moe / 1 swa_dense)
SCOUT = "meta-llama/Llama-4-Scout-17B-16E-Instruct"  # 16 experts (<128): the A8 model
MAVERICK = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"  # 128 experts (>=128 gate boundary)
M3 = "MiniMaxAI/MiniMax-M3"  # 128 experts + 1 shared expert, 60 layers


def _build(model_path, backend="sglang", **cfg_kwargs):
    cfg_kwargs.setdefault("tp_size", 8)
    cfg_kwargs.setdefault("pp_size", 1)
    cfg_kwargs.setdefault("moe_tp_size", 1)
    cfg_kwargs.setdefault("moe_ep_size", 8)
    cfg_kwargs.setdefault("attention_dp_size", 1)
    return get_model(model_path, config.ModelConfig(**cfg_kwargs), backend)


def _names(op_list):
    return [op._name for op in op_list]


def _op(op_list, name):
    """The single top-level op with this name."""
    (found,) = [op for op in op_list if op._name == name]
    return found


def _dispatches(op_list):
    # Composite getters (``overlap._group_a``) re-wrap children as the Rust
    # base classes, so match by class name instead of the Python shell type.
    return [op for op in op_list if type(op).__name__ == "MoEDispatch"]


class TestHybridMoEBlocksViaBuilder:
    def test_scout_gains_router_gemms_a8(self):
        """A8: <128-expert hybrid models now emit one router GEMM per MoE span."""
        model = _build(SCOUT)
        for phase, op_list in (("context", model.context_ops), ("generation", model.generation_ops)):
            for layer_type in ("global", "swa"):
                router = _op(op_list, f"{phase}_{layer_type}_router_gemm")
                assert isinstance(router, ops.GEMM)
                assert router._n == 16  # num_experts
                assert router._k == 5120  # hidden_size
                assert router._quant_mode.name == "bfloat16"
                assert router._scale_factor == 24  # 24 global + 24 swa MoE layers, mtp factor 1.0

    def test_scout_context_sequence_golden(self):
        """Full ordered context emission for the A8 model (routers inserted in place)."""
        model = _build(SCOUT)
        assert _names(model.context_ops) == [
            "context_embedding",
            "context_global_attn_norm",
            "context_global_qkv_gemm",
            "context_attention",
            "context_global_proj_gemm",
            "context_global_moe_norm",
            "context_global_router_gemm",  # A8
            "context_global_moe_pre_dispatch",
            "context_global_moe",
            "context_global_moe_post_dispatch",
            "context_swa_attn_norm",
            "context_swa_qkv_gemm",
            "context_attention",
            "context_swa_proj_gemm",
            "context_swa_moe_norm",
            "context_swa_router_gemm",  # A8
            "context_swa_moe_pre_dispatch",
            "context_swa_moe",
            "context_swa_moe_post_dispatch",
            "context_logits_gemm",
            "context_p2p",
        ]

    def test_mimo_context_sequence_unchanged(self):
        """>=128-expert models already emitted the router: byte-identical emission."""
        model = _build(MIMO)
        assert _names(model.context_ops) == [
            "context_embedding",
            "context_global_attn_norm",
            "context_global_qkv_gemm",
            "context_attention",
            "context_global_proj_gemm",
            "context_global_moe_norm",
            "context_global_router_gemm",
            "context_global_moe_pre_dispatch",
            "context_global_moe",
            "context_global_moe_post_dispatch",
            "context_swa_attn_norm",
            "context_swa_qkv_gemm",
            "context_attention",
            "context_swa_proj_gemm",
            "context_swa_moe_norm",
            "context_swa_router_gemm",
            "context_swa_moe_pre_dispatch",
            "context_swa_moe",
            "context_swa_moe_post_dispatch",
            "context_swa_dense_attn_norm",
            "context_swa_dense_qkv_gemm",
            "context_attention",
            "context_swa_dense_proj_gemm",
            "context_swa_dense_ffn_norm",
            "context_swa_dense_gate_up_gemm",
            "context_swa_dense_act",
            "context_swa_dense_down_gemm",
            "context_logits_gemm",
            "context_p2p",
        ]

    def test_maverick_boundary_router_only_on_moe_span(self):
        """128 experts sits ON the legacy gate boundary: router emitted before and after;
        the dense span gains nothing."""
        model = _build(MAVERICK)
        names = _names(model.context_ops)
        assert "context_global_router_gemm" in names
        assert not any("dense" in n and "router" in n for n in names)

    def test_dispatch_ops_stay_quant_agnostic(self):
        """Legacy hybrid dispatches never carried quant_mode; the builder's fused
        path always forwards it, so the family resets it on the returned ops
        (quant-aware dispatch modeling is a deliberate follow-up, not a rewire
        side effect)."""
        model = _build(MIMO)
        dispatches = _dispatches(model.context_ops) + _dispatches(model.generation_ops)
        assert len(dispatches) == 8  # (global + swa) x (pre + post) x (context + generation)
        assert all(op._quant_mode is None for op in dispatches)

    def test_moe_op_state_unchanged(self):
        model = _build(SCOUT)
        moe = _op(model.context_ops, "context_global_moe")
        assert isinstance(moe, ops.MoE)
        assert moe._inter_size == 8192
        assert moe._topk == 1
        assert moe._num_experts == 16
        assert moe._moe_tp_size == 1 and moe._moe_ep_size == 8 and moe._attention_dp_size == 1
        assert moe._quant_mode == model.config.moe_quant_mode
        assert moe._workload_distribution == "power_law_1.01"

    def test_cp_mutations_land_on_builder_ops(self):
        """The post-hoc CP wiring mutates the builder-returned ops: prefill router/MoE
        get seq_split, dispatches get attn_cp_size; decode dispatches are re-routed
        to the decode-CP comm path (attn_cp_size + is_context=False)."""
        model = _build(MIMO, tp_size=1, moe_ep_size=2, attention_dp_size=1, cp_size=2)
        assert _op(model.context_ops, "context_global_router_gemm")._seq_split == 2
        assert _op(model.context_ops, "context_global_moe")._seq_split == 2
        context_dispatches = _dispatches(model.context_ops)
        assert len(context_dispatches) == 4
        assert all(op._attn_cp_size == 2 for op in context_dispatches)
        assert all(op._is_context for op in context_dispatches)
        generation_dispatches = _dispatches(model.generation_ops)
        assert len(generation_dispatches) == 4
        assert all(op._attn_cp_size == 2 for op in generation_dispatches)
        assert all(not op._is_context for op in generation_dispatches)

    def test_moe_comm_backend_is_rejected_loudly(self):
        """The builder fires its large-EP emission off cfg.moe_comm_backend
        internally; this family has no large-EP wiring (no node-width
        resolution, fused-only builder calls), so a config carrying a comm
        backend must fail construction instead of silently mis-modeling."""
        with pytest.raises(ValueError, match="large-EP is not wired for the HYBRIDMOE family"):
            _build(SCOUT, moe_comm_backend={"context": "deepep_ht", "generation": "deepep_ll"})

    def test_fused_construction_without_comm_backend_still_works(self):
        model = _build(SCOUT, moe_comm_backend=None)
        assert "context_global_moe_pre_dispatch" in _names(model.context_ops)

    def test_builder_receives_family_and_backend(self, monkeypatch):
        """The builder call passes model_family/backend_name so registered
        variants can target this family."""
        calls = []
        original = blocks_moe.build_moe_block_ops

        def probe(*args, **kwargs):
            calls.append(kwargs)
            return original(*args, **kwargs)

        from aiconfigurator_core.sdk.models import hybrid_moe

        monkeypatch.setattr(hybrid_moe, "build_moe_block_ops", probe)
        _build(SCOUT)
        assert len(calls) == 4  # (global + swa) x (context + generation)
        assert all(kw["model_family"] == "HYBRIDMOE" for kw in calls)
        assert all(kw["backend_name"] == "sglang" for kw in calls)
        assert sorted(kw["scale_factor"] for kw in calls) == [24, 24, 24, 24]
        assert all("gpus_per_node" not in kw for kw in calls)  # fused-only family
        assert all("attn_cp_size" not in kw for kw in calls)  # CP stays post-hoc


class TestMiniMaxM3MoEBlockViaBuilder:
    def test_context_sequence_golden(self):
        """Shared triplet stays hand-wired BEFORE the builder block (legacy order)."""
        model = _build(M3)
        assert _names(model.context_ops) == [
            "context_embedding",
            "context_add_norm_1",
            "context_attention",
            "context_add_norm_2",
            "context_shared_gate_up_gemm",
            "context_shared_act_gate",
            "context_shared_ffn2_gemm",
            "context_router_gemm",
            "context_moe_pre_dispatch",
            "context_moe",
            "context_moe_post_dispatch",
            "context_logits_gemm",
            "context_p2p",
        ]

    def test_generation_overlap_groups_golden(self):
        """group_a is exactly the builder's routed block; group_b the hand-wired shared triplet."""
        model = _build(M3)
        overlap = _op(model.generation_ops, "generation_moe_overlap")
        assert isinstance(overlap, ops.OverlapOp)
        assert _names(overlap._group_a) == [
            "generation_router_gemm",
            "generation_moe_pre_dispatch",
            "generation_moe",
            "generation_moe_post_dispatch",
        ]
        assert _names(overlap._group_b) == [
            "generation_shared_gate_up_gemm",
            "generation_shared_act_gate",
            "generation_shared_ffn2_gemm",
        ]

    def test_dispatch_quant_mode_follows_moe_quant(self):
        """Unlike hybrid, this family always forwarded quant_mode to its dispatches."""
        model = _build(M3)
        overlap = _op(model.generation_ops, "generation_moe_overlap")
        dispatches = _dispatches(model.context_ops) + _dispatches(overlap._group_a)
        assert len(dispatches) == 4
        assert all(op._quant_mode == model.config.moe_quant_mode for op in dispatches)

    def test_router_unconditional_and_shared_sizing(self):
        model = _build(M3)
        router = _op(model.context_ops, "context_router_gemm")
        assert router._n == 128 and router._k == 6144
        assert router._scale_factor == 60
        gate_up = _op(model.context_ops, "context_shared_gate_up_gemm")
        # one shared expert, TP-sharded (2 * moe_inter_size // tp_size)
        assert gate_up._n == 2 * 3072 // 8
        assert gate_up._k == 6144
        ffn2 = _op(model.context_ops, "context_shared_ffn2_gemm")
        assert ffn2._n == 6144 and ffn2._k == 3072 // 8

    def test_moe_comm_backend_is_rejected_loudly(self):
        """A comm-backend-carrying config would fire the builder's large-EP
        emission with NO shared experts (this family passes
        num_shared_experts=0 and hand-wires the triplet) — fail loudly."""
        with pytest.raises(ValueError, match="large-EP is not wired for the MINIMAXM3 family"):
            _build(M3, moe_comm_backend={"context": "deepep_ht", "generation": "deepep_ll"})

    def test_fused_construction_without_comm_backend_still_works(self):
        model = _build(M3, moe_comm_backend=None)
        assert "context_moe_pre_dispatch" in _names(model.context_ops)

    def test_builder_receives_family_and_backend(self, monkeypatch):
        calls = []
        original = blocks_moe.build_moe_block_ops

        def probe(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        from aiconfigurator_core.sdk.models import minimax_m3

        monkeypatch.setattr(minimax_m3, "build_moe_block_ops", probe)
        _build(M3, backend="trtllm")
        assert len(calls) == 2  # context + generation
        assert all(kw["model_family"] == "MINIMAXM3" for _, kw in calls)
        assert all(kw["backend_name"] == "trtllm" for _, kw in calls)
        # the shared triplet stays hand-wired: the builder must not emit one
        assert all(args[1].num_shared_experts == 0 for args, _ in calls)
        assert sorted(kw["scale_factor"] for _, kw in calls) == [60, 60]
        assert all("gpus_per_node" not in kw for _, kw in calls)  # fused-only family

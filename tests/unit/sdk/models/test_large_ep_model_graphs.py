# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-level op-graph goldens for the large-EP wiring (Task 6).

The five wideEP/EP model classes (``WideEPDeepSeekModel``,
``TrtllmWideEPDeepSeekModel``, ``WideEPDeepSeekV32Model``,
``TrtllmWideEPDeepSeekV32Model``, ``SGLangEPMOEModel``) are gone; the
surviving family classes emit the same graphs when the enumerator sets
``ModelConfig.moe_comm_backend``. Every RECORDED expectation below was
captured by instantiating those legacy classes at commit 8372e60 (the
commit immediately before this task) with the configs spelled here, then
mapped through A6:

- sglang: legacy ``{p}_moe_pre_dispatch`` (one op whose deepep row sums the
  dispatch+combine legs) == ``{p}_moe_dispatch`` + ``{p}_moe_combine``;
  the legacy graph has no post_dispatch op.
- trtllm: legacy ``{p}_moe_pre_dispatch`` (prepare+dispatch legs) ==
  ``{p}_moe_prepare`` + ``{p}_moe_dispatch``; legacy
  ``{p}_moe_post_dispatch`` == ``{p}_moe_combine``.

Zero-regression contract: outside the MoE block every op is byte-identical
to the deleted classes' (verified mechanically against the 8372e60 archive
over 15 configs — see the task-6 report, Fix round 1). The one ledgered
ordering difference is that the builder emits the trtllm router GEMM BEFORE
the shared triplet where the legacy graph emitted it after; the op set and
per-op state are unchanged and the phase cost is a sum.

Value equivalence of the MoE block itself (comm + compute latencies against
the surviving legacy query methods) is covered by
``test_moe_block_builder_large_ep.py``; this file pins the model-level graph.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

import aiconfigurator_core.sdk.operations as ops
from aiconfigurator_core.sdk import common, config
from aiconfigurator_core.sdk.models import attention_op_keys, get_model

pytestmark = pytest.mark.unit

DSR1 = "deepseek-ai/DeepSeek-R1"
DSV32 = "deepseek-ai/DeepSeek-V3.2-Exp"
QWEN3 = "Qwen/Qwen3-235B-A22B"

SGLANG_COMM = {"context": "deepep_ht", "generation": "deepep_ll"}
TRTLLM_COMM = {"context": "nvlink_two_sided", "generation": "nvlink_two_sided"}

# Recorded from the legacy classes at 8372e60 (nextn=0 -> mtp factor 1.0).
DS_LAYERS = 61
QWEN3_LAYERS = 94
PDL_FACTOR = 0.9  # TrtllmWideEPDeepSeek{,V32}Model._pdl_factor


def _names(op_list):
    return [op._name for op in op_list]


def _op(op_list, name):
    """The single top-level op with this name."""
    (found,) = [op for op in op_list if op._name == name]
    return found


def _build(model_path, backend, **cfg_kwargs):
    cfg = config.ModelConfig(**cfg_kwargs)
    return get_model(model_path, cfg, backend)


def _deepseek_sglang(tp_size=1, attention_dp_size=32, **extra):
    return _build(
        DSR1,
        "sglang",
        tp_size=tp_size,
        moe_tp_size=1,
        moe_ep_size=32,
        attention_dp_size=attention_dp_size,
        moe_backend="deepep_moe",
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        moe_quant_mode=common.MoEQuantMode.fp8_block,
        moe_comm_backend=dict(SGLANG_COMM),
        **extra,
    )


def _deepseek_trtllm(**extra):
    return _build(
        DSR1,
        "trtllm",
        tp_size=1,
        moe_tp_size=1,
        moe_ep_size=16,
        attention_dp_size=16,
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        moe_quant_mode=common.MoEQuantMode.nvfp4,
        moe_comm_backend=dict(TRTLLM_COMM),
        num_gpus_per_node=4,
        **extra,
    )


def _v32_sglang(tp_size=1, attention_dp_size=32, **extra):
    return _build(
        DSV32,
        "sglang",
        tp_size=tp_size,
        moe_tp_size=1,
        moe_ep_size=32,
        attention_dp_size=attention_dp_size,
        moe_backend="deepep_moe",
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        moe_quant_mode=common.MoEQuantMode.fp8_block,
        moe_comm_backend=dict(SGLANG_COMM),
        **extra,
    )


def _v32_trtllm(**extra):
    return _build(
        DSV32,
        "trtllm",
        tp_size=1,
        moe_tp_size=1,
        moe_ep_size=16,
        attention_dp_size=16,
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        moe_quant_mode=common.MoEQuantMode.nvfp4,
        moe_comm_backend=dict(TRTLLM_COMM),
        num_gpus_per_node=4,
        **extra,
    )


def _moe_sglang(**extra):
    return _build(
        QWEN3,
        "sglang",
        tp_size=1,
        moe_tp_size=1,
        moe_ep_size=8,
        attention_dp_size=8,
        moe_backend="deepep_moe",
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        moe_quant_mode=common.MoEQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        **extra,
    )


# ---------------------------------------------------------------------------
# (a) DeepSeek on sglang, large EP (EP32, h200-like)
# ---------------------------------------------------------------------------


class TestDeepSeekSglangLargeEP:
    # RECORDED from WideEPDeepSeekModel @ 8372e60: the deepep graph has no
    # add_norm_2 / logits_gemm / P2P, and its MoE block carries no router.
    CONTEXT: ClassVar[list[str]] = [
        "context_qkv_a_proj_gemm",
        "context_embedding",
        "context_add_norm_1",
        "context_downscale_gemm",
        "context_attention",
        "context_gate_ffn1_gemm",
        "context_act_gate",
        "context_ffn2_gemm",
        "context_moe_dispatch",  # A6: legacy context_moe_pre_dispatch
        "context_moe",
        "context_moe_combine",  # A6: legacy context_moe_pre_dispatch
    ]
    GENERATION: ClassVar[list[str]] = [
        "generation_qkv_a_proj_gemm",
        "generation_embedding",
        "generation_add_norm_1",
        "generation_downscale_gemm",
        "generation_attention",
        "generation_gate_ffn1_gemm",
        "generation_act_gate",
        "generation_ffn2_gemm",
        "generation_moe_dispatch",
        "generation_moe",
        "generation_moe_combine",
    ]

    def test_context_and_generation_graphs(self):
        model = _deepseek_sglang()
        assert _names(model.context_ops) == self.CONTEXT
        assert _names(model.generation_ops) == self.GENERATION
        assert isinstance(model.context_ops[4], ops.WideEPContextMLA)
        assert isinstance(model.generation_ops[4], ops.WideEPGenerationMLA)
        # qkv_a_proj is replicated on every rank: attention TP shards the token
        # stream, so its token count scales by tp (deepseek.py:1109-1121).
        assert model.context_ops[0]._scale_num_tokens == 1
        assert model.generation_ops[0]._scale_num_tokens == 1

    def test_moe_block_ops_and_scales(self):
        model = _deepseek_sglang()
        dispatch, moe, combine = model.context_ops[-3:]
        assert isinstance(dispatch, ops.MoEAllToAll) and dispatch._phase == "dispatch"
        assert isinstance(combine, ops.MoEAllToAll) and combine._phase == "combine"
        assert dispatch._comm_backend == combine._comm_backend == "deepep_ht"
        assert dispatch._node_num == 4  # nodes_for(32 * 1, 8)
        assert dispatch._attention_tp_size == 1  # cfg.tp_size, context only
        assert isinstance(moe, ops.EPMoE)
        assert moe._workload_distribution == "power_law_1.01"
        assert moe._enable_eplb is False
        assert [op._scale_factor for op in model.context_ops[-3:]] == [DS_LAYERS] * 3

        gen_dispatch, gen_moe, gen_combine = model.generation_ops[-3:]
        assert gen_dispatch._comm_backend == gen_combine._comm_backend == "deepep_ll"
        assert gen_dispatch._attention_tp_size == 1  # generation never divides
        assert gen_moe._workload_distribution == "power_law_1.01"
        assert [op._scale_factor for op in model.generation_ops[-3:]] == [float(DS_LAYERS)] * 3

    def test_tp2_adds_nccl_allgather_and_reduce_scatter(self):
        model = _deepseek_sglang(tp_size=2, attention_dp_size=16)
        assert _names(model.context_ops) == [
            "context_qkv_a_proj_gemm",
            "context_tp_all_gather",
            "context_embedding",
            "context_add_norm_1",
            "context_downscale_gemm",
            "context_attention",
            "context_tp_reduce_scatter",
            "context_gate_ffn1_gemm",
            "context_act_gate",
            "context_ffn2_gemm",
            "context_moe_dispatch",
            "context_moe",
            "context_moe_combine",
        ]
        # No TP collectives in the legacy generation graph.
        assert _names(model.generation_ops) == self.GENERATION
        assert model.context_ops[0]._scale_num_tokens == 2
        # deepep context dispatch divides tokens by the attention TP width;
        # generation does not (models/blocks/moe.py, Task 5 fix).
        assert model.context_ops[-3]._attention_tp_size == 2
        assert model.generation_ops[-3]._attention_tp_size == 1

    def test_eplb_flattens_the_prefill_distribution_only(self):
        model = _deepseek_sglang(enable_eplb=True)
        assert model.context_ops[-2]._workload_distribution == "power_law_0.6"
        assert model.context_ops[-2]._enable_eplb is True
        # The legacy class passed enable_eplb=False in decode; EPMoE gates the
        # 0.8 token correction on inference_phase == "context", so the decode
        # op's flag is inert (deepseek.py:1359 @ 8372e60 vs moe_comm.py:1295).
        assert model.generation_ops[-2]._workload_distribution == "power_law_1.01"
        assert model.generation_ops[-2]._inference_phase == "generation"


# ---------------------------------------------------------------------------
# (b) DeepSeek on trtllm, large EP (EP16, gb200-like)
# ---------------------------------------------------------------------------


class TestDeepSeekTrtllmLargeEP:
    # RECORDED from TrtllmWideEPDeepSeekModel @ 8372e60 under the A6 mapping:
    # the GRANULAR MLA stack (q_a_layernorm, no mla_block FallbackOp, the
    # bmm/rope overlap) that the ``context_mla_granular`` / ``generation_mla``
    # capability keys feed.
    CONTEXT: ClassVar[list[str]] = [
        "context_embedding",
        "context_add_norm_1",
        "context_downscale_gemm",
        "context_q_a_layernorm",
        "context_q_b_proj_gemm",
        "context_kv_b_proj_gemm",
        "context_attention",
        "context_proj_gemm",
        "context_add_norm_2",
        "context_router_gemm",
        "context_shared_gate_up_gemm",
        "context_shared_act_gate",
        "context_shared_ffn2_gemm",
        "context_moe_prepare",  # A6: legacy context_moe_pre_dispatch
        "context_moe_dispatch",  # A6: legacy context_moe_pre_dispatch
        "context_moe",
        "context_moe_combine",  # A6: legacy context_moe_post_dispatch
        "context_moe_reduce_add",
        "context_logits_gemm",
        "context_p2p",
    ]
    GENERATION: ClassVar[list[str]] = [
        "generation_embedding",
        "generation_add_norm_1",
        "generation_downscale_gemm",
        "generation_q_a_layernorm",
        "generation_q_b_proj_gemm",
        "generation_bmm_rope_overlap",
        "generation_attention",
        "generation_bmm_post",
        "generation_proj_gemm",
        "generation_add_norm_2",
        "generation_moe_overlap",
        "generation_moe_reduce_add",
        "generation_logits_gemm",
        "generation_p2p",
    ]

    def test_context_and_generation_graphs(self):
        model = _deepseek_trtllm()
        assert _names(model.context_ops) == self.CONTEXT
        assert _names(model.generation_ops) == self.GENERATION
        # The granular stack, not the fused mla_block FallbackOp.
        assert isinstance(_op(model.context_ops, "context_attention"), ops.ContextMLA)
        assert isinstance(_op(model.generation_ops, "generation_attention"), ops.GenerationMLA)
        bmm_rope = _op(model.generation_ops, "generation_bmm_rope_overlap")
        assert _names(bmm_rope._group_a) == ["generation_bmm_pre"]
        assert _names(bmm_rope._group_b) == ["generation_rope_kvcache"]

    def test_generation_overlaps_routed_and_shared(self):
        model = _deepseek_trtllm()
        overlap = _op(model.generation_ops, "generation_moe_overlap")
        assert isinstance(overlap, ops.OverlapOp)
        assert _names(overlap._group_a) == [
            "generation_router_gemm",
            "generation_moe_prepare",
            "generation_moe_dispatch",
            "generation_moe",
            "generation_moe_combine",
        ]
        assert _names(overlap._group_b) == [
            "generation_shared_gate_up_gemm",
            "generation_shared_act_gate",
            "generation_shared_ffn2_gemm",
        ]

    def test_legacy_pdl_factor_scales_the_whole_decode_stack(self):
        model = _deepseek_trtllm()
        assert _op(model.context_ops, "context_moe")._scale_factor == DS_LAYERS
        # Attention AND MoE carry the PDL discount (the legacy class scaled
        # every decode layer op by it); embedding/logits stay at the mtp scale.
        for name in (
            "generation_add_norm_1",
            "generation_downscale_gemm",
            "generation_q_a_layernorm",
            "generation_q_b_proj_gemm",
            "generation_attention",
            "generation_bmm_post",
            "generation_proj_gemm",
            "generation_add_norm_2",
            "generation_moe_reduce_add",
        ):
            assert _op(model.generation_ops, name)._scale_factor == DS_LAYERS * PDL_FACTOR, name
        assert _op(model.generation_ops, "generation_embedding")._scale_factor == 1.0
        assert _op(model.generation_ops, "generation_logits_gemm")._scale_factor == 1.0
        overlap = _op(model.generation_ops, "generation_moe_overlap")
        assert all(op._scale_factor == DS_LAYERS * PDL_FACTOR for op in overlap._group_a)
        assert all(op._scale_factor == DS_LAYERS * PDL_FACTOR for op in overlap._group_b)

    def test_a2a_axes_and_nvfp4_combine_dtype(self):
        model = _deepseek_trtllm()
        prepare = _op(model.context_ops, "context_moe_prepare")
        dispatch = _op(model.context_ops, "context_moe_dispatch")
        moe = _op(model.context_ops, "context_moe")
        combine = _op(model.context_ops, "context_moe_combine")
        for a2a in (prepare, dispatch, combine):
            assert isinstance(a2a, ops.MoEAllToAll)
            assert a2a._comm_backend == "nvlink_two_sided"
            assert a2a._node_num == 4  # nodes_for(16 * 1, num_gpus_per_node=4)
            assert a2a._attention_tp_size == 1  # trtllm alltoall gets undivided tokens
        assert combine._comm_dtype == "nvfp4"  # context keeps the standard rows
        assert isinstance(moe, ops.EPMoE)
        gen_combine = _op(model.generation_ops, "generation_moe_overlap")._group_a[-1]
        assert gen_combine._comm_dtype == "fp4"  # generation low-precision combine

    @pytest.mark.parametrize(
        ("enable_eplb", "expected"),
        [(False, "power_law_1.01"), (True, "power_law_1.01_eplb")],
    )
    def test_eplb_rides_the_distribution_suffix(self, enable_eplb, expected):
        model = _deepseek_trtllm(enable_eplb=enable_eplb)
        moe = _op(model.context_ops, "context_moe")
        assert moe._workload_distribution == expected
        assert moe._enable_eplb is False  # trtllm never uses the deepep 0.8 correction

    def test_num_slots_flows_into_the_ep_moe_op(self):
        model = _deepseek_trtllm(enable_eplb=True, wideep_num_slots=288)
        assert _op(model.context_ops, "context_moe")._num_slots == 288


class TestTrtllmLargeEPValidation:
    """Transcribed from TrtllmWideEPDeepSeekModel @ 8372e60 (deepseek.py:638-678)."""

    def test_requires_attention_dp(self):
        with pytest.raises(ValueError, match="attention_dp_size > 1"):
            _build(
                DSR1,
                "trtllm",
                tp_size=1,
                moe_tp_size=1,
                moe_ep_size=1,
                attention_dp_size=1,
                gemm_quant_mode=common.GEMMQuantMode.fp8_block,
                moe_quant_mode=common.MoEQuantMode.nvfp4,
                moe_comm_backend=dict(TRTLLM_COMM),
            )

    def test_requires_ep_size_above_one(self):
        with pytest.raises(ValueError, match="moe_ep_size > 1"):
            _build(
                DSR1,
                "trtllm",
                tp_size=1,
                moe_tp_size=4,
                moe_ep_size=1,
                attention_dp_size=4,
                gemm_quant_mode=common.GEMMQuantMode.fp8_block,
                moe_quant_mode=common.MoEQuantMode.nvfp4,
                moe_comm_backend=dict(TRTLLM_COMM),
            )

    def test_rejects_redundant_slots_without_eplb(self):
        with pytest.raises(ValueError, match="must equal"):
            _deepseek_trtllm(enable_eplb=False, wideep_num_slots=288)

    def test_rejects_slots_below_num_experts(self):
        with pytest.raises(ValueError, match="must be >="):
            _deepseek_trtllm(enable_eplb=True, wideep_num_slots=128)

    def test_warns_when_ep_size_not_above_topk(self, caplog):
        with caplog.at_level("WARNING"):
            _build(
                DSR1,
                "trtllm",
                tp_size=1,
                moe_tp_size=1,
                moe_ep_size=8,
                attention_dp_size=8,
                gemm_quant_mode=common.GEMMQuantMode.fp8_block,
                moe_quant_mode=common.MoEQuantMode.nvfp4,
                moe_comm_backend=dict(TRTLLM_COMM),
            )
        assert "AlltoAll communication will be disabled" in caplog.text

    def test_fused_configs_skip_the_validation(self):
        # No moe_comm_backend -> fused path, no wideEP constraints.
        model = _build(
            DSR1,
            "trtllm",
            tp_size=8,
            moe_tp_size=1,
            moe_ep_size=8,
            attention_dp_size=1,
            gemm_quant_mode=common.GEMMQuantMode.fp8_block,
            moe_quant_mode=common.MoEQuantMode.nvfp4,
        )
        assert "context_moe_post_dispatch" in _names(model.context_ops)


# ---------------------------------------------------------------------------
# (c) DeepSeek-V3.2 / GLM DSA family
# ---------------------------------------------------------------------------


class TestDeepSeekV32LargeEP:
    # RECORDED from WideEPDeepSeekV32Model @ 8372e60.
    SGLANG_CONTEXT: ClassVar[list[str]] = [
        "context_attention",
        "context_gate_ffn1_gemm",
        "context_act_gate",
        "context_ffn2_gemm",
        "context_moe_dispatch",
        "context_moe",
        "context_moe_combine",
    ]
    SGLANG_GENERATION: ClassVar[list[str]] = [
        "generation_attention",
        "generation_gate_ffn1_gemm",
        "generation_act_gate",
        "generation_ffn2_gemm",
        "generation_moe_dispatch",
        "generation_moe",
        "generation_moe_combine",
    ]

    def test_sglang_graphs(self):
        model = _v32_sglang()
        assert _names(model.context_ops) == self.SGLANG_CONTEXT
        assert _names(model.generation_ops) == self.SGLANG_GENERATION
        assert isinstance(model.context_ops[0], ops.ContextDSAModule)
        assert isinstance(model.generation_ops[0], ops.GenerationDSAModule)
        assert model.context_ops[-2]._workload_distribution == "power_law_1.01"

    def test_sglang_tp2_adds_nccl_collectives(self):
        model = _v32_sglang(tp_size=2, attention_dp_size=16)
        assert _names(model.context_ops)[:3] == [
            "context_tp_all_gather",
            "context_attention",
            "context_tp_reduce_scatter",
        ]
        assert _names(model.generation_ops) == self.SGLANG_GENERATION

    def test_sglang_eplb_flattens_prefill(self):
        model = _v32_sglang(enable_eplb=True)
        assert model.context_ops[-2]._workload_distribution == "power_law_0.6"
        assert model.generation_ops[-2]._workload_distribution == "power_law_1.01"

    def test_trtllm_graphs(self):
        model = _v32_trtllm()
        assert _names(model.context_ops) == [
            "context_embedding",
            "context_add_norm_1",
            "context_attention",
            "context_add_norm_2",
            "context_router_gemm",
            "context_shared_gate_up_gemm",
            "context_shared_act_gate",
            "context_shared_ffn2_gemm",
            "context_moe_prepare",
            "context_moe_dispatch",
            "context_moe",
            "context_moe_combine",
            "context_moe_reduce_add",
            "context_logits_gemm",
            "context_p2p",
        ]
        assert _names(model.generation_ops) == [
            "generation_embedding",
            "generation_add_norm_1",
            "generation_attention",
            "generation_add_norm_2",
            "generation_moe_overlap",
            "generation_moe_reduce_add",
            "generation_logits_gemm",
            "generation_p2p",
        ]

    def test_trtllm_pdl_factor_and_eplb_suffix(self):
        model = _v32_trtllm(enable_eplb=True)
        assert _op(model.context_ops, "context_moe")._workload_distribution == "power_law_1.01_eplb"
        overlap = _op(model.generation_ops, "generation_moe_overlap")
        assert all(op._scale_factor == DS_LAYERS * PDL_FACTOR for op in overlap._group_a)
        # The legacy class scaled the whole decode stack by the PDL factor.
        for name in ("generation_add_norm_1", "generation_attention", "generation_add_norm_2"):
            assert _op(model.generation_ops, name)._scale_factor == DS_LAYERS * PDL_FACTOR, name

    def test_trtllm_validation_applies(self):
        with pytest.raises(ValueError, match="must equal"):
            _v32_trtllm(enable_eplb=False, wideep_num_slots=288)


class TestGLMSharedExpertQuantMode:
    """GLM-5.2-NVFP4 excludes ``mlp.shared_experts*`` from NVFP4.

    The legacy shared-expert dtype is asymmetric and is reproduced as such:
    the trtllm class sized its shared GEMMs with
    ``_dsa_shared_expert_quant_mode`` (bf16 here) while the sglang class used
    the plain ``gemm_quant_mode`` (nvfp4). Weights recorded from the legacy
    classes at 8372e60.
    """

    GLM = "nvidia/GLM-5.2-NVFP4"
    LEGACY_BF16_GATE_UP_WEIGHTS = 50331648
    LEGACY_NVFP4_GATE_UP_WEIGHTS = 14155776.0

    def _build(self, backend, comm):
        return _build(
            self.GLM,
            backend,
            tp_size=1,
            moe_tp_size=1,
            moe_ep_size=16,
            attention_dp_size=16,
            moe_backend="deepep_moe" if backend == "sglang" else None,
            moe_comm_backend=dict(comm),
            num_gpus_per_node=4,
        )

    def test_trtllm_keeps_the_shared_experts_unquantized(self):
        model = self._build("trtllm", TRTLLM_COMM)
        gate_up = _op(model.context_ops, "context_shared_gate_up_gemm")
        assert gate_up._quant_mode is common.GEMMQuantMode.bfloat16
        assert gate_up._weights == self.LEGACY_BF16_GATE_UP_WEIGHTS
        assert _op(model.context_ops, "context_shared_ffn2_gemm")._quant_mode is common.GEMMQuantMode.bfloat16
        # The routed experts stay on the model-wide MoE mode.
        assert _op(model.context_ops, "context_moe")._quant_mode is common.MoEQuantMode.nvfp4

    def test_sglang_uses_the_model_wide_gemm_mode(self):
        model = self._build("sglang", SGLANG_COMM)
        gate_up = _op(model.context_ops, "context_gate_ffn1_gemm")
        assert gate_up._quant_mode is common.GEMMQuantMode.nvfp4
        assert gate_up._weights == self.LEGACY_NVFP4_GATE_UP_WEIGHTS


# ---------------------------------------------------------------------------
# (d) Traditional MoE family (ex-SGLangEPMOEModel)
# ---------------------------------------------------------------------------


class TestMOEModelLargeEP:
    # RECORDED from SGLangEPMOEModel @ 8372e60 under the A6 mapping. The legacy
    # large-EP graph replicates the embedding (no vocab//tp shard) and ends at
    # the logits gemm — no ``{p}_embedding_ar``, no P2P.
    CONTEXT: ClassVar[list[str]] = [
        "context_embedding",
        "context_add_norm_1",
        "context_qkv_gemm",
        "context_attention",
        "context_proj_gemm",
        "context_add_norm_2",
        "context_router_gemm",
        "context_moe_dispatch",
        "context_moe",
        "context_moe_combine",
    ]
    GENERATION: ClassVar[list[str]] = [
        "generation_embedding",
        "generation_add_norm_1",
        "generation_qkv_gemm",
        "generation_attention",
        "generation_proj_gemm",
        "generation_add_norm_2",
        "generation_router_gemm",
        "generation_moe_dispatch",
        "generation_moe",
        "generation_moe_combine",
        "generation_logits_gemm",
    ]

    def test_graphs_and_backends(self):
        model = _moe_sglang(moe_comm_backend=dict(SGLANG_COMM))
        assert _names(model.context_ops) == self.CONTEXT
        assert _names(model.generation_ops) == self.GENERATION
        assert model.context_ops[7]._comm_backend == "deepep_ht"
        assert model.generation_ops[7]._comm_backend == "deepep_ll"
        assert isinstance(model.context_ops[8], ops.EPMoE)
        assert model.context_ops[8]._scale_factor == QWEN3_LAYERS
        assert model.generation_ops[8]._scale_factor == float(QWEN3_LAYERS)
        # Embedding replicated, not vocab-sharded (legacy large-EP form).
        fused = _moe_sglang()
        assert model.context_ops[0]._row_size == _op(fused.context_ops, "context_embedding")._row_size

    def test_distributions_use_the_moe_family_alpha(self):
        model = _moe_sglang(moe_comm_backend=dict(SGLANG_COMM))
        assert model.context_ops[8]._workload_distribution == "power_law_1.2"
        assert model.generation_ops[8]._workload_distribution == "power_law_1.2"

    def test_eplb_flattens_prefill_only(self):
        model = _moe_sglang(moe_comm_backend=dict(SGLANG_COMM), enable_eplb=True)
        assert model.context_ops[8]._workload_distribution == "power_law_0.6"
        assert model.context_ops[8]._enable_eplb is True
        # Decode keeps the family alpha; the EPLB token correction is
        # prefill-only inside EPMoE (moe_comm.py:1295).
        assert model.generation_ops[8]._workload_distribution == "power_law_1.2"
        assert model.generation_ops[8]._inference_phase == "generation"

    def test_deepep_backend_alone_no_longer_switches_the_graph(self):
        # moe_backend=deepep_moe without moe_comm_backend is the FUSED graph
        # now: only the enumerator's per-phase comm backend selects large EP.
        model = _moe_sglang()
        assert "context_moe_post_dispatch" in _names(model.context_ops)
        assert "context_moe_dispatch" not in _names(model.context_ops)


# ---------------------------------------------------------------------------
# (e) Fused goldens — RECORDED at 8372e60, must be unchanged by this task
# ---------------------------------------------------------------------------


class TestFusedGraphsUnchanged:
    DS_CONTEXT: ClassVar[list[str]] = [
        "context_embedding",
        "context_add_norm_1",
        "context_mla_block",
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
    DS_GENERATION: ClassVar[list[str]] = [
        "generation_embedding",
        "generation_add_norm_1",
        "generation_mla_block",
        "generation_add_norm_2",
        "generation_moe_overlap",
        "generation_logits_gemm",
        "generation_p2p",
    ]

    @pytest.mark.parametrize("backend", ["sglang", "trtllm"])
    def test_deepseek_fused(self, backend):
        model = _build(
            DSR1,
            backend,
            tp_size=8,
            moe_tp_size=1,
            moe_ep_size=8,
            attention_dp_size=1,
            gemm_quant_mode=common.GEMMQuantMode.fp8_block,
            moe_quant_mode=common.MoEQuantMode.fp8_block,
        )
        assert _names(model.context_ops) == self.DS_CONTEXT
        assert _names(model.generation_ops) == self.DS_GENERATION
        assert model.context_ops[9]._workload_distribution == "power_law_1.01"
        overlap = model.generation_ops[4]
        assert _names(overlap._group_a) == [
            "generation_router_gemm",
            "generation_moe_pre_dispatch",
            "generation_moe",
            "generation_moe_post_dispatch",
        ]

    def test_deepseek_v32_fused(self):
        model = _build(
            DSV32,
            "sglang",
            tp_size=8,
            moe_tp_size=1,
            moe_ep_size=8,
            attention_dp_size=1,
            gemm_quant_mode=common.GEMMQuantMode.fp8_block,
            moe_quant_mode=common.MoEQuantMode.fp8_block,
        )
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
        assert _names(model.generation_ops) == [
            "generation_embedding",
            "generation_add_norm_1",
            "generation_attention",
            "generation_add_norm_2",
            "generation_moe_overlap",
            "generation_logits_gemm",
            "generation_p2p",
        ]

    def test_moe_family_fused(self):
        model = _moe_sglang()
        assert _names(model.context_ops) == [
            "context_embedding",
            "context_add_norm_1",
            "context_qkv_gemm",
            "context_attention",
            "context_proj_gemm",
            "context_add_norm_2",
            "context_router_gemm",
            "context_moe_pre_dispatch",
            "context_moe",
            "context_moe_post_dispatch",
            "context_embedding_ar",
            "context_p2p",
        ]
        assert _names(model.generation_ops) == [
            "generation_embedding",
            "generation_add_norm_1",
            "generation_qkv_gemm",
            "generation_attention",
            "generation_proj_gemm",
            "generation_add_norm_2",
            "generation_router_gemm",
            "generation_moe_pre_dispatch",
            "generation_moe",
            "generation_moe_post_dispatch",
            "generation_logits_gemm",
            "generation_embedding_ar",
            "generation_p2p",
        ]
        assert model.context_ops[8]._workload_distribution == "power_law_1.2"


# ---------------------------------------------------------------------------
# (f) attention_op_keys: the wideep flag became the large-EP selector
# ---------------------------------------------------------------------------


class TestAttentionOpKeys:
    @pytest.mark.parametrize("family", ["DEEPSEEK", "KIMIK25"])
    def test_sglang_large_ep_uses_the_wideep_mla_tables(self, family):
        assert attention_op_keys(family, "sglang", is_large_ep=True) == (
            "wideep_context_mla",
            "wideep_generation_mla",
        )
        assert attention_op_keys(family, "sglang") == ("context_mla", "generation_mla")

    def test_trtllm_large_ep_uses_the_granular_context_slice(self):
        assert attention_op_keys("DEEPSEEK", "trtllm", is_large_ep=True) == (
            "context_mla_granular",
            "generation_mla",
        )

    def test_other_families_are_unaffected(self):
        assert attention_op_keys("DEEPSEEKV32", "sglang", is_large_ep=True) == (
            "dsa_context_module",
            "dsa_generation_module",
        )
        assert attention_op_keys("MOE", "sglang", is_large_ep=True) == (
            "context_attention",
            "generation_attention",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

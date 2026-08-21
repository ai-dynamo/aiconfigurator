# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 hybrid GDN + full-attention LM modeling contracts."""

import json

import pytest

from aiconfigurator.sdk import common, models
from aiconfigurator.sdk import config as sdk_config
from aiconfigurator.sdk.operations import CustomAllReduce, OverlapOp

pytestmark = pytest.mark.unit


def _model_config(
    tp_size=2,
    *,
    moe_tp_size=None,
    moe_ep_size=1,
    attention_dp_size=1,
    moe_backend=None,
    enable_encoder_dp=True,
):
    return sdk_config.ModelConfig(
        tp_size=tp_size,
        pp_size=1,
        moe_tp_size=tp_size if moe_tp_size is None else moe_tp_size,
        moe_ep_size=moe_ep_size,
        attention_dp_size=attention_dp_size,
        moe_backend=moe_backend,
        enable_encoder_dp=enable_encoder_dp,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
    )


def _flatten_ops(phase_ops):
    for op in phase_ops:
        if isinstance(op, OverlapOp):
            yield from op._group_a
            yield from op._group_b
        else:
            yield op


@pytest.mark.parametrize(
    ("model_name", "expected_out_hidden", "is_moe"),
    [
        ("Qwen/Qwen3.5-27B", 5120, False),
        ("Qwen/Qwen3.5-35B-A3B", 2048, True),
        ("Qwen/Qwen3.5-122B-A10B", 3072, True),
        ("Qwen/Qwen3.5-397B-A17B", 4096, True),
    ],
)
def test_qwen35_dense_and_moe_variants_preserve_language_and_build_vision_encoder(
    model_name, expected_out_hidden, is_moe
):
    model = models.get_model(model_name, _model_config(tp_size=2), "vllm")

    assert isinstance(model.extra_params, common.Qwen35Config)
    assert (model.extra_params.num_experts > 0) is is_moe
    assert model.encoder_config is model.extra_params.vision_config
    assert model.encoder_config.out_hidden_size == expected_out_hidden
    assert model.encoder_config.projector_dims == ((4608, 4608), (4608, expected_out_hidden))
    assert model.encoder_config.projector_n_instances == 1
    assert model.encoder_config.deepstack_visual_indexes == ()
    assert {"context_gdn_scan", "context_attention"} <= {op._name for op in model.context_ops}
    assert {"generation_gdn_recurrence", "generation_attention"} <= {op._name for op in model.generation_ops}
    assert {
        "encoder_patch_embed_gemm",
        "encoder_position_embed",
        "encoder_attention",
        "encoder_merger_norm",
        "encoder_projector_fc1_gemm",
        "encoder_rope_apply",
    } <= {op._name for op in model.encoder_ops}
    position_embed = next(op for op in model.encoder_ops if op._name == "encoder_position_embed")
    position_spec = json.loads(position_embed._spec_json())["Elementwise"]
    assert position_spec["bytes_per_token"] == 2 * (2 * 1152 + 1152)
    merger_norm = next(op for op in model.encoder_ops if op._name == "encoder_merger_norm")
    merger_spec = json.loads(merger_norm._spec_json())["Elementwise"]
    assert merger_spec["bytes_per_token"] == 2 * (1152 + 1152)


def test_qwen35_encoder_dp_replicates_vit_and_gathers_projected_tokens():
    model = models.get_model(
        "Qwen/Qwen3.5-27B",
        _model_config(tp_size=4, enable_encoder_dp=True),
        "vllm",
    )
    encoder_ops = {op._name: op for op in model.encoder_ops}

    assert encoder_ops["encoder_attention"]._n == 16
    assert encoder_ops["encoder_qkv_gemm"]._n == 3 * 1152
    assert "encoder_dp_all_gather" in encoder_ops


def test_qwen35_legacy_encoder_tp_shards_vit_and_projector():
    model = models.get_model(
        "Qwen/Qwen3.5-27B",
        _model_config(tp_size=4, enable_encoder_dp=False),
        "vllm",
    )
    encoder_ops = {op._name: op for op in model.encoder_ops}

    assert encoder_ops["encoder_attention"]._n == 4
    assert encoder_ops["encoder_qkv_gemm"]._n == 3 * 1152 // 4
    assert encoder_ops["encoder_projector_fc0_gemm"]._n == 4608 // 4
    assert encoder_ops["encoder_projector_fc1_gemm"]._k == 4608 // 4
    assert "encoder_dp_all_gather" not in encoder_ops


def test_bundled_qwen35_122b_is_not_added_to_default_support_matrix_roster():
    assert "Qwen/Qwen3.5-122B-A10B" not in common.DefaultHFModels
    assert "Qwen/Qwen3.5-122B-A10B" not in common.SupportMatrixHFModels


@pytest.mark.parametrize(
    (
        "model_name",
        "expected_k_heads",
        "expected_v_heads",
        "expected_in_proj_n",
        "expected_ba_n",
        "expected_out_proj_k",
    ),
    [
        ("Qwen/Qwen3.5-27B", 4, 12, 4096, 24, 1536),
        ("Qwen/Qwen3.5-35B-A3B", 4, 8, 3072, 16, 1024),
    ],
)
def test_qwen35_tp4_gdn_uses_local_heads_without_resharding_projection_gemms(
    model_name, expected_k_heads, expected_v_heads, expected_in_proj_n, expected_ba_n, expected_out_proj_k
):
    """GDN lookup heads are TP-local; qkvz and ba are separate per-rank GEMMs."""
    model = models.get_model(model_name, _model_config(tp_size=4), "sglang")

    context_ops = {op._name: op for op in model.context_ops}
    generation_ops = {op._name: op for op in model.generation_ops}

    for op_name in ("context_gdn_conv1d", "context_gdn_scan"):
        assert context_ops[op_name]._num_k_heads == expected_k_heads
        assert context_ops[op_name]._num_v_heads == expected_v_heads
    for op_name in ("generation_gdn_conv1d", "generation_gdn_recurrence"):
        assert generation_ops[op_name]._num_k_heads == expected_k_heads
        assert generation_ops[op_name]._num_v_heads == expected_v_heads

    assert context_ops["context_gdn_in_proj_gemm"]._n == expected_in_proj_n
    assert context_ops["context_gdn_in_proj_ba_gemm"]._n == expected_ba_n
    assert context_ops["context_gdn_out_proj_gemm"]._k == expected_out_proj_k
    assert generation_ops["generation_gdn_in_proj_gemm"]._n == expected_in_proj_n
    assert generation_ops["generation_gdn_in_proj_ba_gemm"]._n == expected_ba_n
    assert generation_ops["generation_gdn_out_proj_gemm"]._k == expected_out_proj_k


def test_qwen35_rejects_tensor_parallel_size_that_cannot_shard_gdn_heads():
    with pytest.raises(ValueError, match="GDN head counts must both be divisible"):
        models.get_model("Qwen/Qwen3.5-27B", _model_config(tp_size=3), "sglang")


def test_qwen35_rejects_megamoe_backend():
    """MegaMoE is a DeepSeek-V4-only sglang module; modeling it here would
    double-count the attention AR through the non-DeepEP dispatch branch."""
    with pytest.raises(ValueError, match="megamoe"):
        models.get_model("Qwen/Qwen3.5-35B-A3B", _model_config(tp_size=4, moe_backend="megamoe"), "sglang")


@pytest.mark.parametrize(
    "model_config_kwargs",
    [
        {"tp_size": 8},  # pure TP (moe_tp follows tp)
        {"tp_size": 8, "moe_tp_size": 1, "moe_ep_size": 8},  # attention TP + EP
    ],
)
def test_qwen35_moe_prices_comm_through_dispatch_pair_for_all_topologies(model_config_kwargs):
    """Every topology emits the same dispatch chain (serial in context, routed
    and shared experts overlapped in generation); layout-specific collectives
    are resolved inside MoEDispatch, leaving one attention-side AR per layer
    plus embedding."""
    model = models.get_model("Qwen/Qwen3.5-35B-A3B", _model_config(**model_config_kwargs), "vllm")

    context_names = [op._name for op in model.context_ops]
    assert not any(name.endswith("_moe_final_ar") for name in context_names)
    for prefix in ("context_gdn", "context_full"):
        expected_order = [
            f"{prefix}_router_gemm",
            f"{prefix}_moe_pre_dispatch",
            f"{prefix}_moe",
            f"{prefix}_moe_post_dispatch",
            f"{prefix}_shared_expert_gate_gemm",
            f"{prefix}_shared_gate_up_gemm",
            f"{prefix}_shared_act_gate",
            f"{prefix}_shared_down_gemm",
            f"{prefix}_shared_expert_gate_mul",
            f"{prefix}_shared_merge",
        ]
        indices = [context_names.index(name) for name in expected_order]
        assert indices == sorted(indices)

    # Generation runs routed and shared experts on parallel CUDA streams
    # (OverlapOp); the merge and the post collective (all-reduce of the
    # merged sum) are serial after the join.
    generation_names = [op._name for op in model.generation_ops]
    generation_ops = {op._name: op for op in model.generation_ops}
    for prefix in ("generation_gdn", "generation_full"):
        overlap = generation_ops[f"{prefix}_moe_overlap"]
        assert [op._name for op in overlap._group_a] == [
            f"{prefix}_router_gemm",
            f"{prefix}_moe_pre_dispatch",
            f"{prefix}_moe",
        ]
        assert [op._name for op in overlap._group_b] == [
            f"{prefix}_shared_expert_gate_gemm",
            f"{prefix}_shared_gate_up_gemm",
            f"{prefix}_shared_act_gate",
            f"{prefix}_shared_down_gemm",
            f"{prefix}_shared_expert_gate_mul",
        ]
        assert generation_names.index(f"{prefix}_shared_merge") < generation_names.index(f"{prefix}_moe_post_dispatch")

    # Explicit CustomAllReduce ops: 40 attention-side + 1 embedding.
    for phase_ops in (model.context_ops, model.generation_ops):
        allreduce_ops = [op for op in phase_ops if isinstance(op, CustomAllReduce)]
        assert sum(op._scale_factor for op in allreduce_ops) == 41


def test_qwen35_shared_expert_scalar_gate_uses_true_output_width():
    """The runtime ReplicatedLinear scalar gate is hidden_size -> 1."""
    model = models.get_model(
        "Qwen/Qwen3.5-397B-A17B",
        _model_config(tp_size=8, moe_tp_size=1, moe_ep_size=8),
        "vllm",
    )

    for phase_ops in (model.context_ops, model.generation_ops):
        scalar_gates = [op for op in _flatten_ops(phase_ops) if op._name.endswith("_shared_expert_gate_gemm")]
        assert len(scalar_gates) == 2
        assert {op._n for op in scalar_gates} == {1}


def test_qwen35_sglang_standard_dispatcher_omits_nonexistent_pre_dispatch():
    """SGLang StandardDispatcher has no collective before routed experts; its
    CUDA-graph decode overlaps shared and routed experts, the scalar gate is
    fused into the post-join merge kernel, and the post all-reduce is serial
    (outside the overlap)."""
    model = models.get_model(
        "Qwen/Qwen3.5-397B-A17B",
        _model_config(tp_size=8, moe_tp_size=1, moe_ep_size=8),
        "sglang",
    )

    for phase, phase_ops in (("context", model.context_ops), ("generation", model.generation_ops)):
        op_names = [op._name for op in _flatten_ops(phase_ops)]
        assert any(name.endswith(("_gdn_ar", "_full_ar")) for name in op_names)
        for prefix in (f"{phase}_gdn", f"{phase}_full"):
            assert f"{prefix}_moe_pre_dispatch" not in op_names
            assert f"{prefix}_moe" in op_names
            assert f"{prefix}_moe_post_dispatch" in op_names
            assert f"{prefix}_shared_expert_gate_gemm" not in op_names
            assert f"{prefix}_shared_expert_gate_mul" not in op_names
            assert f"{prefix}_shared_merge" in op_names
    for op in model.generation_ops:
        if isinstance(op, OverlapOp):
            group_names = [inner._name for inner in _flatten_ops([op])]
            assert not any(name.endswith("_moe_post_dispatch") for name in group_names)
    assert any(isinstance(op, OverlapOp) for op in model.generation_ops)


def test_qwen35_sglang_deepep_prices_one_dispatch_and_replicates_shared_expert():
    """DeepEP rows hold the full dispatch+combine round trip, so one dispatch
    op prices it; sglang DeepEP replicates the shared expert (tp_size=1)
    instead of TP-sharding it."""
    model = models.get_model(
        "Qwen/Qwen3.5-35B-A3B",
        _model_config(tp_size=8, moe_tp_size=1, moe_ep_size=8, moe_backend="deepep_moe"),
        "sglang",
    )
    cfg = model.extra_params

    for phase, phase_ops in (("context", model.context_ops), ("generation", model.generation_ops)):
        op_names = [op._name for op in phase_ops]
        # DeepEP scatters: the attn-TP reduction is NOT folded into a gather.
        assert any(name.endswith(("_gdn_ar", "_full_ar")) for name in op_names)
        for prefix in (f"{phase}_gdn", f"{phase}_full"):
            assert f"{prefix}_moe_pre_dispatch" in op_names
            assert f"{prefix}_moe_post_dispatch" not in op_names
        gate_ups = [op for op in phase_ops if op._name.endswith("_shared_gate_up_gemm")]
        assert {op._n for op in gate_ups} == {2 * cfg.shared_expert_inter_size}
        # Context runs the shared expert on the attn-TP scattered token slice.
        expected_scale = 8 if phase == "context" else 1
        assert {op._scale_num_tokens for op in gate_ups} == {expected_scale}
    # DeepEP keeps shared and routed experts serial in generation.
    assert not any(isinstance(op, OverlapOp) for op in model.generation_ops)


def test_qwen35_sglang_default_moe_keeps_pre_dispatch_under_attention_dp():
    """With DP attention the LayerCommunicator pre-MLP gather is real and
    priced; the attn-TP partial-sum reduction folds into that gather, so the
    per-layer attention AR disappears."""
    model = models.get_model(
        "Qwen/Qwen3.5-35B-A3B",
        _model_config(tp_size=4, moe_tp_size=1, moe_ep_size=8, attention_dp_size=2),
        "sglang",
    )

    for phase_ops in (model.context_ops, model.generation_ops):
        op_names = [op._name for op in _flatten_ops(phase_ops)]
        assert any(name.endswith("_moe_pre_dispatch") for name in op_names)
        assert not any(name.endswith(("_gdn_ar", "_full_ar")) for name in op_names)


def test_qwen35_memory_charges_kv_on_full_layers_and_constant_gdn_state():
    """35B-A3B at tp4: 10 full layers hold per-token KV; 30 GDN layers hold a
    constant per-request state (fp32 SSM + bf16 conv window), TP-sharded."""
    model = models.get_model("Qwen/Qwen3.5-35B-A3B", _model_config(tp_size=4), "vllm")

    assert model.get_kvcache_elements_per_token() == 10 * 2 * 1 * 256
    expected_state = 30 * ((32 // 4) * 128 * 128 * 4 + (2 * 16 * 128 + 32 * 128) // 4 * 3 * 2)
    assert model._gdn_state_bytes_per_request() == expected_state
    per_token_bytes = 2 * model.get_kvcache_elements_per_token()
    assert model.get_kvcache_bytes_per_sequence(4096) == 4096 * per_token_bytes + expected_state
    assert model.get_kvcache_max_tokens(expected_state + 100 * per_token_bytes) == 100

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 hybrid GDN + full-attention LM modeling contracts."""

import pytest

from aiconfigurator.sdk import common, models
from aiconfigurator.sdk import config as sdk_config
from aiconfigurator.sdk.operations import CustomAllReduce

pytestmark = pytest.mark.unit


def _model_config(tp_size=2, *, moe_tp_size=None, moe_ep_size=1, attention_dp_size=1):
    return sdk_config.ModelConfig(
        tp_size=tp_size,
        pp_size=1,
        moe_tp_size=tp_size if moe_tp_size is None else moe_tp_size,
        moe_ep_size=moe_ep_size,
        attention_dp_size=attention_dp_size,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
    )


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


@pytest.mark.parametrize(
    "model_config_kwargs",
    [
        {"tp_size": 8},  # pure TP (moe_tp follows tp)
        {"tp_size": 8, "moe_tp_size": 1, "moe_ep_size": 8},  # attention TP + EP
    ],
)
def test_qwen35_moe_prices_comm_through_dispatch_pair_for_all_topologies(model_config_kwargs):
    """Every topology emits the same pre/MoE/post dispatch chain; layout-specific
    collectives are resolved inside MoEDispatch, leaving one attention-side AR
    per layer plus embedding."""
    model = models.get_model("Qwen/Qwen3.5-35B-A3B", _model_config(**model_config_kwargs), "vllm")
    for phase, phase_ops in (("context", model.context_ops), ("generation", model.generation_ops)):
        op_names = [op._name for op in phase_ops]

        assert not any(name.endswith("_moe_final_ar") for name in op_names)
        for prefix in (f"{phase}_gdn", f"{phase}_full"):
            expected_order = [
                f"{prefix}_router_gemm",
                f"{prefix}_moe_pre_dispatch",
                f"{prefix}_moe",
                f"{prefix}_moe_post_dispatch",
                f"{prefix}_shared_expert_gate_gemm",
                f"{prefix}_shared_gate_up_gemm",
                f"{prefix}_shared_act_gate",
                f"{prefix}_shared_down_gemm",
            ]
            indices = [op_names.index(name) for name in expected_order]
            assert indices == sorted(indices)

        # Explicit CustomAllReduce ops: 40 attention-side + 1 embedding.
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
        scalar_gates = [
            op
            for op in phase_ops
            if op._name.endswith("_shared_expert_gate_gemm")
        ]
        assert len(scalar_gates) == 2
        assert {op._n for op in scalar_gates} == {1}


def test_qwen35_sglang_standard_dispatcher_omits_nonexistent_pre_dispatch():
    """SGLang StandardDispatcher has no collective before routed experts."""
    model = models.get_model(
        "Qwen/Qwen3.5-397B-A17B",
        _model_config(tp_size=8, moe_tp_size=1, moe_ep_size=8),
        "sglang",
    )

    for phase, phase_ops in (("context", model.context_ops), ("generation", model.generation_ops)):
        op_names = [op._name for op in phase_ops]
        for prefix in (f"{phase}_gdn", f"{phase}_full"):
            assert f"{prefix}_moe_pre_dispatch" not in op_names
            assert f"{prefix}_moe" in op_names
            assert f"{prefix}_moe_post_dispatch" in op_names

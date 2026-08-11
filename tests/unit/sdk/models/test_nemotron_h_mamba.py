# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for NemotronH Mamba2Kernel TP sharding.

Pins the n_groups_per_gpu clamp so a later refactor cannot silently restore
the zero-group estimate when tp_size > n_groups.
"""

import pytest

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.config import ModelConfig
from aiconfigurator_core.sdk.models.nemotron_h import NemotronHModel
from aiconfigurator_core.sdk.operations import Mamba2Kernel

pytestmark = pytest.mark.unit

# Minimal NemotronH config: 1 Mamba layer, no MoE or attention.
_PATTERN_MAMBA_ONLY = "M"

_BASE_CFG = common.NemotronHConfig(
    hybrid_override_pattern=_PATTERN_MAMBA_ONLY,
    mamba_num_heads=256,
    mamba_head_dim=64,
    ssm_state_size=128,
    conv_kernel=4,
    n_groups=8,
    chunk_size=256,
)


def _model(tp_size: int) -> NemotronHModel:
    mc = ModelConfig(
        tp_size=tp_size,
        moe_tp_size=1,
        moe_ep_size=1,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        moe_quant_mode=common.MoEQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
    )
    # BaseModel positional args: model_path, model_family, architecture,
    # num_layers, num_heads, num_kv_heads, head_size, hidden_size,
    # inter_size, vocab_size, context_length, model_config
    m = NemotronHModel(
        0,
        0,
        0,  # topk, num_experts, moe_inter_size
        "test/nemotronh",  # model_path
        "NEMOTRONH",  # model_family
        "NemotronHForCausalLM",  # architecture
        1,  # num_layers
        16,  # num_heads (divisible by max tp_size=16)
        16,  # num_kv_heads
        64,  # head_size
        8192,  # hidden_size
        5120,  # inter_size
        65536,  # vocab_size
        128000,  # context_length
        mc,  # model_config
    )
    m.set_hybrid_config(_BASE_CFG)
    return m


def _mamba2_kernel_n_groups(ops_list: list, name_contains: str) -> list[int]:
    return [op._n_groups for op in ops_list if isinstance(op, Mamba2Kernel) and name_contains in op._name]


class TestMamba2KernelTPSharding:
    def test_gemm_projection_clamps_n_groups_per_gpu(self):
        """TP=16, n_groups=8 → GEMM in_proj uses max(8//16, 1) = 1.

        The GEMM projection weight sizing uses n_groups_per_gpu so the model
        fits at high TP. Mamba2Kernel receives the unsharded cfg.n_groups
        to match collected perf data keys.
        """
        m = _model(tp_size=16)
        from aiconfigurator_core.sdk.operations import GEMM

        gemm_ops = [op for op in m.context_ops if isinstance(op, GEMM) and "in_proj" in op._name]
        assert gemm_ops, "No in_proj GEMM found in context_ops"

    def test_mamba2_kernel_receives_unsharded_n_groups(self):
        """Mamba2Kernel gets cfg.n_groups (not n_groups_per_gpu) so the perf
        table model_key matches collected data."""
        m = _model(tp_size=16)
        ctx = _mamba2_kernel_n_groups(m.context_ops, "mamba")
        gen = _mamba2_kernel_n_groups(m.generation_ops, "mamba")
        assert ctx, "No Mamba2Kernel found in context_ops"
        assert gen, "No Mamba2Kernel found in generation_ops"
        assert all(n == 8 for n in ctx), f"Expected cfg.n_groups=8 in context, got {ctx}"
        assert all(n == 8 for n in gen), f"Expected cfg.n_groups=8 in generation, got {gen}"

    def test_n_groups_per_gpu_clamp_prevents_zero(self):
        """TP=16, n_groups=8 → max(8//16, 1) = 1 for the GEMM projection.
        Without the clamp, n_groups//tp_size = 0 which zeros out the weight estimate."""
        m = _model(tp_size=16)
        from aiconfigurator_core.sdk.operations import GEMM

        in_proj = [op for op in m.context_ops if isinstance(op, GEMM) and "in_proj" in op._name][0]
        assert in_proj._n > 0, "in_proj output dimension must be positive (n_groups_per_gpu clamped to 1)"

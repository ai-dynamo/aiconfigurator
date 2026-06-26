# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dense / sweep context-parallelism (CP) support.

Covers the parts added on top of the GLM-5 DSA CP path (see
``test_cp_dsa_modeling.py`` for the sparse-attention modeling):

- ``ModelConfig`` CP geometry: ``total_gpus_per_worker`` / ``attn_width``.
- ``enumerate_parallel_config`` CP sweep (6-tuple, width match, sglang guard).
- ``ContextAttention`` zigzag chunk math for balanced per-rank prefill work.
- ``BaseModel._cp_kv_memory_divisor`` (CP keeps full KV per rank; divisor 1).
- ``supports_cp`` capability matrix.
- gemma4 / hybrid_moe heterogeneous-KV CP wiring (per-type all-gather).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import aiconfigurator.sdk.common as common
import aiconfigurator.sdk.operations as ops
from aiconfigurator.sdk import config as cfgmod
from aiconfigurator.sdk.performance_result import PerformanceResult
from aiconfigurator.sdk.utils import enumerate_parallel_config

pytestmark = pytest.mark.unit


def _mkcfg(cp, ep=1):
    return cfgmod.ModelConfig(
        tp_size=1,
        pp_size=1,
        attention_dp_size=1,
        cp_size=cp,
        cp_style="allgather" if cp > 1 else "none",
        moe_tp_size=1,
        moe_ep_size=ep,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        moe_quant_mode=common.MoEQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        comm_quant_mode=common.CommQuantMode.half,
    )


# --------------------------------------------------------------------------
# ModelConfig CP geometry
# --------------------------------------------------------------------------


def test_total_gpus_per_worker_folds_cp():
    cfg = cfgmod.ModelConfig(tp_size=2, pp_size=2, attention_dp_size=1, cp_size=4)
    assert cfg.total_gpus_per_worker == 2 * 2 * 1 * 4
    assert cfg.attn_width == 2 * 4 * 1


def test_cp1_geometry_unchanged():
    cfg = cfgmod.ModelConfig(tp_size=4, pp_size=1, attention_dp_size=1, cp_size=1)
    assert cfg.total_gpus_per_worker == 4
    assert cfg.attn_width == 4


def test_moe_width_validation_includes_cp():
    # attn_width = tp*cp*dp = 1*8*1 must equal moe_tp*moe_ep = 1*8.
    cfg = cfgmod.ModelConfig(tp_size=1, attention_dp_size=1, cp_size=8, moe_tp_size=1, moe_ep_size=8)
    assert cfg.attn_width == cfg.moe_tp_size * cfg.moe_ep_size


def test_model_width_assert_includes_cp():
    # MoE model __init__ asserts tp*cp*dp == moe_tp*moe_ep; cp=4 vs moe_ep=8 -> mismatch.
    from aiconfigurator.sdk.models.gemma4 import Gemma4MixModel

    with pytest.raises(AssertionError):
        Gemma4MixModel(
            8,
            128,
            1024,
            "g",
            "GEMMA4MIX",
            "Gemma4MixForCausalLM",
            2,
            16,
            4,
            128,
            2048,
            8192,
            256000,
            8192,
            _mkcfg(4, ep=8),
            {},
        )


# --------------------------------------------------------------------------
# enumerate_parallel_config CP sweep
# --------------------------------------------------------------------------


def test_enumerate_non_cp_is_six_tuple_cp_one():
    r = enumerate_parallel_config(
        num_gpu_list=[1, 2, 4, 8],
        tp_list=[1, 2, 4, 8],
        pp_list=[1],
        is_moe=False,
        backend=common.BackendName.sglang,
    )
    assert r and all(len(c) == 6 and c[5] == 1 for c in r)


def test_default_cp_list_auto_sweep_policy():
    # Capability-derived: any family whose class supports_cp on sglang auto-sweeps.
    from aiconfigurator.sdk.task_v2 import _default_cp_list_for

    for fam in ("DEEPSEEKV32", "DEEPSEEKV4", "LLAMA", "GPT", "MOE", "GEMMA4MIX", "HYBRIDMOE"):
        assert _default_cp_list_for(fam, "sglang") == [1, 2, 4, 8], fam
    # CP only on sglang -> non-sglang backends never sweep.
    assert _default_cp_list_for("DEEPSEEKV32", "trtllm") == [1]
    assert _default_cp_list_for("LLAMA", "trtllm") == [1]
    # Unknown / non-CP family falls back to [1].
    assert _default_cp_list_for("NOT_A_REAL_FAMILY", "sglang") == [1]


def test_enumerate_dense_cp_sweep():
    r = enumerate_parallel_config(
        num_gpu_list=[8],
        tp_list=[1, 8],
        pp_list=[1],
        cp_list=[1, 8],
        is_moe=False,
        backend=common.BackendName.sglang,
    )
    assert [8, 1, 1, 1, 1, 1] in r  # pure TP
    assert [1, 1, 1, 1, 1, 8] in r  # pure CP


def test_enumerate_moe_cp_width_match():
    # tp*cp*dp == moe_tp*moe_ep : tp1 cp8 dp1 == 1*8.
    r = enumerate_parallel_config(
        num_gpu_list=[8],
        tp_list=[1],
        pp_list=[1],
        dp_list=[1],
        moe_tp_list=[1],
        moe_ep_list=[8],
        cp_list=[1, 8],
        is_moe=True,
        backend=common.BackendName.sglang,
        enable_wideep=True,
    )
    assert [1, 1, 1, 1, 8, 8] in r
    assert [1, 1, 1, 1, 8, 1] not in r  # cp1 -> attn_width 1 != moe_ep 8


def test_enumerate_cp_rejected_on_non_sglang():
    with pytest.raises(ValueError, match="CP is only supported on sglang"):
        enumerate_parallel_config(
            num_gpu_list=[8],
            tp_list=[8],
            pp_list=[1],
            cp_list=[1, 8],
            backend=common.BackendName.trtllm,
        )


# --------------------------------------------------------------------------
# ContextAttention zigzag chunk math
# --------------------------------------------------------------------------


def _recording_db():
    db = MagicMock()
    db.query_context_attention.return_value = PerformanceResult(1.0, energy=0.0, source="silicon")
    db.query_mem_op.return_value = PerformanceResult(0.0, energy=0.0, source="silicon")
    return db


def test_context_attention_cp1_single_full_chunk():
    db = _recording_db()
    op = ops.ContextAttention("a", 1, 8, 8, common.KVCacheQuantMode.bfloat16, common.FMHAQuantMode.bfloat16, cp_size=1)
    op.query(db, batch_size=1, s=8000, prefix=0)
    calls = db.query_context_attention.call_args_list
    assert len(calls) == 1
    assert (calls[0].args[1], calls[0].args[2]) == (8000, 0)


def test_context_attention_zigzag_two_balanced_chunks():
    db = _recording_db()
    op = ops.ContextAttention("a", 1, 8, 8, common.KVCacheQuantMode.bfloat16, common.FMHAQuantMode.bfloat16, cp_size=2)
    op.query(db, batch_size=1, s=8000, prefix=0)
    calls = db.query_context_attention.call_args_list
    # c = ceil(8000 / (2*cp)) = ceil(8000/4) = 2000.
    # chunk_prev attends [0, prefix+c); chunk_next attends [0, prefix+isl).
    sp = sorted((c.args[1], c.args[2]) for c in calls)
    assert len(calls) == 2
    assert sp == [(2000, 0), (2000, 6000)]


# --------------------------------------------------------------------------
# KV memory divisor + supports_cp
# --------------------------------------------------------------------------


def test_cp_kv_memory_divisor():
    from aiconfigurator.sdk.models.llama import LLAMAModel

    def _m(cp):
        return LLAMAModel("l", "LLAMA", "LlamaForCausalLM", 4, 32, 8, 128, 4096, 14336, 128256, 131072, _mkcfg(cp), {})

    # CP gives no per-rank KV-memory savings for any family: each rank holds the
    # FULL KV (dense gathers+replicates; MLA/DSA decode reads full resident KV
    # since decode does not run CP). Verified vs sglang v0.5.13. -> divisor 1.
    assert _m(1)._cp_kv_memory_divisor() == 1
    assert _m(8)._cp_kv_memory_divisor() == 1
    dsa = _m(8)
    dsa.model_family = "DEEPSEEKV32"
    assert dsa._cp_kv_memory_divisor() == 1


def test_supports_cp_matrix():
    from aiconfigurator.sdk.models.deepseek import DeepSeekModel
    from aiconfigurator.sdk.models.gemma4 import Gemma4MixModel
    from aiconfigurator.sdk.models.gpt import GPTModel
    from aiconfigurator.sdk.models.hybrid_moe import HybridMoEModel
    from aiconfigurator.sdk.models.llama import LLAMAModel

    for cls in (LLAMAModel, GPTModel, DeepSeekModel, Gemma4MixModel, HybridMoEModel):
        assert cls.supports_cp("sglang") is True, cls.__name__
        assert cls.supports_cp("trtllm") is False, cls.__name__


# --------------------------------------------------------------------------
# gemma4 / hybrid_moe heterogeneous-KV CP wiring
# --------------------------------------------------------------------------


def test_gemma4_cp_per_type_allgather_and_zigzag():
    from aiconfigurator.sdk.models.gemma4 import Gemma4MixModel

    m = Gemma4MixModel(
        8,
        128,
        1024,
        "g",
        "GEMMA4MIX",
        "Gemma4MixForCausalLM",
        2,
        16,
        4,
        128,
        2048,
        8192,
        256000,
        8192,
        _mkcfg(8, ep=8),
        {},
    )
    m.set_gemma4_config(
        common.Gemma4MixConfig(
            layer_types=["sliding_attention", "full_attention"],
            swa_num_kv_heads=4,
            swa_head_dim=128,
            global_num_kv_heads=4,
            global_head_dim=128,
            sliding_window_size=1024,
            attention_k_eq_v=False,
        )
    )
    names = [o._name for o in m.context_ops]
    assert "context_cp_all_gather_swa" in names
    assert "context_cp_all_gather_global" in names
    attn = [o for o in m.context_ops if o._name == "context_attention"]
    assert attn and all(o._cp_size == 8 for o in attn)
    md = [o for o in m.context_ops if isinstance(o, ops.MoEDispatch)]
    assert md and all(o._attn_cp_size == 8 for o in md)


def test_hybrid_moe_cp_per_type_allgather():
    from aiconfigurator.sdk.models.hybrid_moe import HybridMoEModel

    m = HybridMoEModel(
        8,
        128,
        1024,
        "h",
        "HYBRIDMOE",
        "HybridMoEForCausalLM",
        4,
        16,
        4,
        128,
        2048,
        8192,
        151936,
        40960,
        _mkcfg(8, ep=8),
        {},
    )
    m.set_hybrid_config(
        common.HybridMoEConfig(
            attn_layer_pattern=[1, 0, 0, 1],
            moe_layer_freq=[1, 1, 0, 0],
            swa_num_kv_heads=4,
            swa_head_dim=128,
            swa_v_head_dim=128,
            global_v_head_dim=128,
            sliding_window_size=1024,
            dense_inter_size=8192,
        )
    )
    names = [o._name for o in m.context_ops]
    assert "context_cp_all_gather_global" in names
    assert "context_cp_all_gather_swa" in names
    attn = [o for o in m.context_ops if o._name == "context_attention"]
    assert attn and all(o._cp_size == 8 for o in attn)

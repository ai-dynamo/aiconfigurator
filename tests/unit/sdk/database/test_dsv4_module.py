# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.models import get_model

pytestmark = pytest.mark.unit


def _dsv4_attn_kwargs(compress_ratio: int) -> dict:
    return {
        "b": 2,
        "s": 256,
        "prefix": 0,
        "num_heads": 16,
        "hidden_size": 7168,
        "q_lora_rank": 1536,
        "o_lora_rank": 1024,
        "head_dim": 512,
        "rope_head_dim": 64,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 1024,
        "window_size": 128,
        "compress_ratio": compress_ratio,
        "o_groups": 2,
        "kvcache_quant_mode": common.KVCacheQuantMode.fp8,
        "fmha_quant_mode": common.FMHAQuantMode.bfloat16,
        "gemm_quant_mode": common.GEMMQuantMode.fp8_block,
    }


class TestDSV4MHCModule:
    def test_mhc_sol_and_hybrid_return_positive(self, comprehensive_perf_db):
        for mode in (common.DatabaseMode.SOL, common.DatabaseMode.HYBRID):
            result = comprehensive_perf_db.query_dsv4_mhc_module(
                num_tokens=512,
                hidden_size=7168,
                hc_mult=4,
                sinkhorn_iters=20,
                op="pre",
                quant_mode=common.GEMMQuantMode.bfloat16,
                database_mode=mode,
            )
            assert float(result) > 0

    def test_mhc_sol_full_shape(self, comprehensive_perf_db):
        result = comprehensive_perf_db.query_dsv4_mhc_module(
            num_tokens=512,
            hidden_size=7168,
            hc_mult=4,
            sinkhorn_iters=20,
            op="both",
            quant_mode=common.GEMMQuantMode.bfloat16,
            database_mode=common.DatabaseMode.SOL_FULL,
        )
        assert len(result) == 3
        sol_time, sol_math, sol_mem = result
        assert math.isclose(sol_time, max(sol_math, sol_mem), rel_tol=1e-6)


class TestDSV4AttentionModule:
    @pytest.mark.parametrize("compress_ratio", [0, 4, 128])
    def test_context_sol_returns_positive_for_all_attention_kinds(self, comprehensive_perf_db, compress_ratio):
        result = comprehensive_perf_db.query_context_dsv4_attention_module(
            **_dsv4_attn_kwargs(compress_ratio),
            database_mode=common.DatabaseMode.SOL,
        )
        assert float(result) > 0

    @pytest.mark.parametrize("compress_ratio", [0, 4, 128])
    def test_generation_hybrid_falls_back_for_all_attention_kinds(self, comprehensive_perf_db, compress_ratio):
        kwargs = _dsv4_attn_kwargs(compress_ratio)
        kwargs.pop("prefix")
        kwargs["s"] = 4096
        result = comprehensive_perf_db.query_generation_dsv4_attention_module(
            **kwargs,
            database_mode=common.DatabaseMode.HYBRID,
        )
        assert float(result) > 0

    def test_csa_index_topk_changes_sol(self, comprehensive_perf_db):
        base = _dsv4_attn_kwargs(4)
        low_topk = comprehensive_perf_db.query_context_dsv4_attention_module(
            **{**base, "index_topk": 128, "s": 4096},
            database_mode=common.DatabaseMode.SOL,
        )
        high_topk = comprehensive_perf_db.query_context_dsv4_attention_module(
            **{**base, "index_topk": 1024, "s": 4096},
            database_mode=common.DatabaseMode.SOL,
        )
        assert high_topk > low_topk


def test_deepseek_v4_static_sol_and_hybrid_run_end_to_end(comprehensive_perf_db):
    comprehensive_perf_db.system_spec["gpu"]["mem_capacity"] = 288400343040
    comprehensive_perf_db.system_spec["misc"]["nccl_mem"] = {1: 0, 2: 0, 4: 0, 8: 0}
    comprehensive_perf_db.system_spec["misc"]["other_mem"] = 0
    model_config = config.ModelConfig(
        tp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        nextn=1,
        nextn_accept_rates=[0.85, 0.3, 0.0, 0.0, 0.0],
        overwrite_num_layers=2,
    )
    model = get_model("sgl-project/DeepSeek-V4-Flash-FP8", model_config, backend_name="trtllm")
    backend = TRTLLMBackend()
    runtime = RuntimeConfig(batch_size=1, beam_width=1, isl=128, osl=4, prefix=0)

    for mode in (common.DatabaseMode.SOL, common.DatabaseMode.HYBRID):
        comprehensive_perf_db.set_default_database_mode(mode)
        summary = backend.run_static(model, comprehensive_perf_db, runtime, mode="static", stride=1)
        assert sum(summary.get_context_latency_dict().values()) > 0
        assert sum(summary.get_generation_latency_dict().values()) > 0

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import (
    load_context_deepseek_v4_attention_module_data,
    load_deepseek_v4_mhc_module_data,
    load_generation_deepseek_v4_attention_module_data,
)

pytestmark = pytest.mark.unit


def _deepseek_v4_attn_kwargs(compress_ratio: int) -> dict:
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


def test_deepseek_v4_module_loaders_accept_collector_filenames(tmp_path):
    mhc_file = tmp_path / "dsv4_mhc_module_perf.txt"
    mhc_file.write_text(
        "framework,version,device,op_name,kernel_source,model,op,num_tokens,hc_mult,hidden_size,latency,power\n"
        "SGLang,0.0,gpu,dsv4_mhc_pre_module,sglang_mhc,m,pre,512,4,7168,0.25,100\n",
        encoding="utf-8",
    )
    context_file = tmp_path / "dsv4_csa_context_module_perf.txt"
    context_file.write_text(
        "framework,version,device,op_name,kernel_source,model,architecture,mla_dtype,kv_cache_dtype,gemm_type,"
        "num_heads,batch_size,isl,tp_size,step,compress_ratio,latency,power\n"
        "SGLang,0.0,gpu,dsv4_csa_context_module,compressed_flashmla,m,DeepseekV4ForCausalLM,bfloat16,"
        "fp8_e4m3,bfloat16,128,2,256,1,0,4,0.50,100\n",
        encoding="utf-8",
    )
    generation_file = tmp_path / "dsv4_hca_generation_module_perf.txt"
    generation_file.write_text(
        "framework,version,device,op_name,kernel_source,model,architecture,mla_dtype,kv_cache_dtype,gemm_type,"
        "num_heads,batch_size,isl,tp_size,step,compress_ratio,latency,power\n"
        "SGLang,0.0,gpu,dsv4_hca_generation_module,compressed_flashmla,m,DeepseekV4ForCausalLM,bfloat16,"
        "fp8_e4m3,bfloat16,128,2,1,1,4096,128,0.75,100\n",
        encoding="utf-8",
    )

    mhc = load_deepseek_v4_mhc_module_data(str(tmp_path / "deepseek_v4_mhc_module_perf.txt"))
    context = load_context_deepseek_v4_attention_module_data(str(tmp_path / "deepseek_v4_context_module_perf.txt"))
    generation = load_generation_deepseek_v4_attention_module_data(
        str(tmp_path / "deepseek_v4_generation_module_perf.txt")
    )

    assert mhc[common.GEMMQuantMode.bfloat16]["pre"][4][7168][512]["latency"] == 0.25
    assert (
        context[common.FMHAQuantMode.bfloat16][common.KVCacheQuantMode.fp8][common.GEMMQuantMode.bfloat16][
            "DeepseekV4ForCausalLM"
        ][4][128][256][2]["latency"]
        == 0.50
    )
    assert (
        generation[common.KVCacheQuantMode.fp8][common.GEMMQuantMode.bfloat16]["DeepseekV4ForCausalLM"][128][128][2][
            4097
        ]["latency"]
        == 0.75
    )


class TestDeepSeekV4MHCModule:
    def test_mhc_sol_and_hybrid_return_positive(self, comprehensive_perf_db):
        for mode in (common.DatabaseMode.SOL, common.DatabaseMode.HYBRID):
            result = comprehensive_perf_db.query_deepseek_v4_mhc_module(
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
        result = comprehensive_perf_db.query_deepseek_v4_mhc_module(
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


class TestDeepSeekV4AttentionModule:
    @pytest.mark.parametrize("compress_ratio", [0, 4, 128])
    def test_context_sol_returns_positive_for_all_attention_kinds(self, comprehensive_perf_db, compress_ratio):
        result = comprehensive_perf_db.query_context_deepseek_v4_attention_module(
            **_deepseek_v4_attn_kwargs(compress_ratio),
            database_mode=common.DatabaseMode.SOL,
        )
        assert float(result) > 0

    @pytest.mark.parametrize("compress_ratio", [0, 4, 128])
    def test_generation_hybrid_falls_back_for_all_attention_kinds(self, comprehensive_perf_db, compress_ratio):
        kwargs = _deepseek_v4_attn_kwargs(compress_ratio)
        kwargs.pop("prefix")
        kwargs["s"] = 4096
        result = comprehensive_perf_db.query_generation_deepseek_v4_attention_module(
            **kwargs,
            database_mode=common.DatabaseMode.HYBRID,
        )
        assert float(result) > 0

    def test_csa_topk_changes_attention_workload(self, comprehensive_perf_db):
        base = _deepseek_v4_attn_kwargs(4)
        low_topk = comprehensive_perf_db.query_context_deepseek_v4_attention_module(
            **{**base, "index_topk": 128, "s": 4096},
            database_mode=common.DatabaseMode.SOL,
        )
        high_topk = comprehensive_perf_db.query_context_deepseek_v4_attention_module(
            **{**base, "index_topk": 1024, "s": 4096},
            database_mode=common.DatabaseMode.SOL,
        )
        assert high_topk > low_topk

    @pytest.mark.parametrize("compress_ratio", [0, 4, 128])
    def test_context_prefix_changes_sol_for_all_attention_kinds(self, comprehensive_perf_db, compress_ratio):
        base = {**_deepseek_v4_attn_kwargs(compress_ratio), "s": 512}
        no_prefix = comprehensive_perf_db.query_context_deepseek_v4_attention_module(
            **base,
            database_mode=common.DatabaseMode.SOL,
        )
        with_prefix = comprehensive_perf_db.query_context_deepseek_v4_attention_module(
            **{**base, "prefix": 1024},
            database_mode=common.DatabaseMode.SOL,
        )
        assert with_prefix > no_prefix

    @pytest.mark.parametrize("compress_ratio", [0, 4, 128])
    def test_context_sol_increases_with_sequence_length(self, comprehensive_perf_db, compress_ratio):
        base = _deepseek_v4_attn_kwargs(compress_ratio)
        short = comprehensive_perf_db.query_context_deepseek_v4_attention_module(
            **{**base, "s": 256},
            database_mode=common.DatabaseMode.SOL,
        )
        long = comprehensive_perf_db.query_context_deepseek_v4_attention_module(
            **{**base, "s": 4096},
            database_mode=common.DatabaseMode.SOL,
        )
        assert long > short

    def test_csa_indexer_logits_scale_with_compressed_length_not_topk_only(self, comprehensive_perf_db):
        base = {**_deepseek_v4_attn_kwargs(4), "s": 4096}
        short_cache = comprehensive_perf_db.query_context_deepseek_v4_attention_module(
            **{**base, "prefix": 0, "index_topk": 16},
            database_mode=common.DatabaseMode.SOL_FULL,
        )
        long_cache = comprehensive_perf_db.query_context_deepseek_v4_attention_module(
            **{**base, "prefix": 4096, "index_topk": 16},
            database_mode=common.DatabaseMode.SOL_FULL,
        )
        assert long_cache[1] > short_cache[1]

    def test_kvcache_quant_changes_sol_memory(self, comprehensive_perf_db):
        base = {**_deepseek_v4_attn_kwargs(128), "s": 4096, "prefix": 4096}
        bf16 = comprehensive_perf_db.query_context_deepseek_v4_attention_module(
            **{**base, "kvcache_quant_mode": common.KVCacheQuantMode.bfloat16},
            database_mode=common.DatabaseMode.SOL_FULL,
        )
        fp8 = comprehensive_perf_db.query_context_deepseek_v4_attention_module(
            **{**base, "kvcache_quant_mode": common.KVCacheQuantMode.fp8},
            database_mode=common.DatabaseMode.SOL_FULL,
        )
        assert fp8[2] < bf16[2]

    def test_gemm_quant_changes_sol_math_and_memory(self, comprehensive_perf_db):
        base = {**_deepseek_v4_attn_kwargs(4), "s": 4096, "prefix": 1024}
        bf16 = comprehensive_perf_db.query_context_deepseek_v4_attention_module(
            **{**base, "gemm_quant_mode": common.GEMMQuantMode.bfloat16},
            database_mode=common.DatabaseMode.SOL_FULL,
        )
        fp8 = comprehensive_perf_db.query_context_deepseek_v4_attention_module(
            **{**base, "gemm_quant_mode": common.GEMMQuantMode.fp8_block},
            database_mode=common.DatabaseMode.SOL_FULL,
        )
        assert fp8[1] < bf16[1]
        assert fp8[2] < bf16[2]


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

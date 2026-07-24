# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk import operations as ops
from aiconfigurator.sdk.backends.sglang_backend import SGLANGBackend
from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.operations.dsv4 import (
    _deep_merge_dsv4_dicts,
)
from aiconfigurator.sdk.perf_database import (
    LoadedOpData,
    load_mhc_module_data,
)

pytestmark = pytest.mark.unit


def _deepseek_v4_attn_kwargs(compress_ratio: int) -> dict:
    return {
        "b": 2,
        "s": 256,
        "prefix": 0,
        "num_heads": 16,
        "native_heads": 128,
        "tp_size": 8,
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


def _deepseek_v4_value(latency: float) -> dict[str, float]:
    return {"latency": latency, "power": 10.0, "energy": latency * 10.0}


def _write_mhc_perf(path, rows: list[str]) -> str:
    header = "framework,version,device,op_name,kernel_source,architecture,num_tokens,hc_mult,hidden_size,latency"
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return str(path)


def _context_deepseek_v4_data(compress_ratio: int, attn_dict: dict, native_heads: int = 128) -> dict:
    return {
        common.FMHAQuantMode.bfloat16: {
            common.KVCacheQuantMode.fp8: {
                common.GEMMQuantMode.fp8_block: {
                    native_heads: {
                        compress_ratio: attn_dict,
                    },
                },
            },
        },
    }


def _generation_deepseek_v4_data(compress_ratio: int, attn_dict: dict, native_heads: int = 128) -> dict:
    return {
        common.KVCacheQuantMode.fp8: {
            common.GEMMQuantMode.fp8_block: {
                native_heads: {
                    compress_ratio: attn_dict,
                },
            },
        },
    }


def _dsv4_sampled_batch_caps_grid() -> dict:
    """Mock the real DSV4 sampled shape: b=1/2/4/8 with shrinking max s."""
    return {
        8: {
            1024: {
                1: _deepseek_v4_value(1.00),
                2: _deepseek_v4_value(3.00),
                4: _deepseek_v4_value(6.00),
                8: _deepseek_v4_value(12.00),
            },
            2048: {
                1: _deepseek_v4_value(2.00),
                2: _deepseek_v4_value(4.80),
                4: _deepseek_v4_value(8.00),
            },
            4096: {
                1: _deepseek_v4_value(3.00),
                2: _deepseek_v4_value(5.80),
            },
            8192: {
                1: _deepseek_v4_value(4.00),
            },
        }
    }


def _dsv4_generation_sampled_grid() -> dict:
    """Generation shape is [tp][b][s_total]; collector's minimum s_total is 2."""
    return {
        8: {
            1: {
                2: _deepseek_v4_value(0.20),
                5: _deepseek_v4_value(0.50),
            },
            2: {
                2: _deepseek_v4_value(0.40),
                5: _deepseek_v4_value(1.00),
            },
        }
    }


def _dsv4_sparse_kernel_grid(lat_without_prefix: float = 0.02, lat_with_prefix: float = 0.05) -> dict:
    return {
        128: {
            1: {
                0: {
                    54.0: {1: {"latency": lat_without_prefix}},
                },
                2816.0: {
                    54.0: {1: {"latency": lat_with_prefix}},
                },
            }
        }
    }


def test_mhc_module_loader_returns_none_for_missing_file(tmp_path):
    assert load_mhc_module_data(str(tmp_path / "mhc_module_perf.txt")) is None


class TestDeepSeekV4MHCModule:
    def test_mhc_empirical_raises_without_data(self, comprehensive_perf_db):
        from aiconfigurator.sdk.errors import EmpiricalNotImplementedError

        with pytest.raises(EmpiricalNotImplementedError):
            comprehensive_perf_db.query_mhc_module(
                num_tokens=512,
                hidden_size=7168,
                hc_mult=4,
                sinkhorn_iters=20,
                op="pre",
                quant_mode=common.GEMMQuantMode.bfloat16,
                database_mode=common.DatabaseMode.EMPIRICAL,
            )

    def test_mhc_sol_full_shape(self, comprehensive_perf_db):
        result = comprehensive_perf_db.query_mhc_module(
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
        assert sol_time > 0
        assert math.isclose(sol_time, max(sol_math, sol_mem), rel_tol=1e-6)

    def test_mhc_weight_memory_uses_quant_mode(self, comprehensive_perf_db):
        bf16_op = ops.DeepSeekV4MHCModule(
            "mhc",
            1,
            "pre",
            7168,
            4,
            20,
            common.GEMMQuantMode.bfloat16,
        )
        fp8_op = ops.DeepSeekV4MHCModule(
            "mhc",
            1,
            "pre",
            7168,
            4,
            20,
            common.GEMMQuantMode.fp8_block,
        )
        assert fp8_op.get_weights() == pytest.approx(bf16_op.get_weights() / 2)

        bf16_sol = comprehensive_perf_db.query_mhc_module(
            num_tokens=512,
            hidden_size=7168,
            hc_mult=4,
            sinkhorn_iters=20,
            op="pre",
            quant_mode=common.GEMMQuantMode.bfloat16,
            database_mode=common.DatabaseMode.SOL_FULL,
        )
        fp8_sol = comprehensive_perf_db.query_mhc_module(
            num_tokens=512,
            hidden_size=7168,
            hc_mult=4,
            sinkhorn_iters=20,
            op="pre",
            quant_mode=common.GEMMQuantMode.fp8_block,
            database_mode=common.DatabaseMode.SOL_FULL,
        )
        assert fp8_sol[2] < bf16_sol[2]

    def test_mhc_loader_and_query_use_shape(self, mutable_comprehensive_perf_db, tmp_path):
        path = _write_mhc_perf(
            tmp_path / "mhc_module_perf.txt",
            [
                "VLLM,test,H20,pre,mhc,DeepseekV4ForCausalLM,512,4,4096,1.5",
                "VLLM,test,H20,pre,mhc,DeepseekV4ForCausalLM,512,4,7168,2.5",
            ],
        )
        db = mutable_comprehensive_perf_db
        db._mhc_module_data = load_mhc_module_data(path)

        result = db.query_mhc_module(
            num_tokens=512,
            hidden_size=7168,
            hc_mult=4,
            sinkhorn_iters=20,
            op="pre",
            quant_mode=common.GEMMQuantMode.bfloat16,
            database_mode=common.DatabaseMode.SILICON,
        )

        assert float(result) == pytest.approx(2.5)


class TestDeepSeekV4AttentionModule:
    def test_generation_uses_pre_decode_kv_length(self, comprehensive_perf_db):
        base = _deepseek_v4_attn_kwargs(4)
        base.pop("prefix")
        current = comprehensive_perf_db.query_generation_deepseek_v4_attention_module(
            **{**base, "s": 512},
            database_mode=common.DatabaseMode.SOL_FULL,
        )
        next_step = comprehensive_perf_db.query_generation_deepseek_v4_attention_module(
            **{**base, "s": 513},
            database_mode=common.DatabaseMode.SOL_FULL,
        )

        assert next_step[1] > current[1]

    def test_generation_silicon_below_min_sampled_s_total_holds_boundary_util(self, mutable_comprehensive_perf_db):
        """b=1, s_total=1 sits below the min sampled s_total=2: the engine holds
        the boundary util and lets the decode SOL carry the (tiny) difference,
        instead of the legacy raw-linear downward extrapolation (which halved
        the latency straight through the launch-overhead floor)."""
        db = mutable_comprehensive_perf_db
        # SCHEME A silicon data is {head}{cr}{b}{s_total} — no tp level. The shared
        # The shared grid fixture is {tp}{b}{s_total};
        # strip the tp wrapper so it lands as {b}{s_total} under {head}{cr}.
        mock_grid = _dsv4_generation_sampled_grid()[8]
        db._generation_deepseek_v4_attention_module_data = LoadedOpData(
            _generation_deepseek_v4_data(4, mock_grid),
            common.PerfDataFilename.dsv4_csa_generation_module,
            "mock_dsv4_generation_module_tp8",
        )
        kwargs = _deepseek_v4_attn_kwargs(4)
        kwargs.pop("prefix")

        result = db.query_generation_deepseek_v4_attention_module(
            **{
                **kwargs,
                "b": 1,
                "s": 1,
                "num_heads": 8,
            },
            database_mode=common.DatabaseMode.SILICON,
        )

        def sol(b, s_total):
            return float(
                db.query_generation_deepseek_v4_attention_module(
                    **{**kwargs, "b": b, "s": s_total, "num_heads": 8},
                    database_mode=common.DatabaseMode.SOL,
                )
            )

        expected = 0.20 * sol(1, 1) / sol(1, 2)  # boundary util held at s_total=2
        assert float(result) == pytest.approx(expected)
        assert result.energy == pytest.approx(expected * 10.0)

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

    def test_attention_sol_covers_flash_and_pro_shapes(self, comprehensive_perf_db):
        common_kwargs = {
            "b": 2,
            "s": 4096,
            "prefix": 1024,
            "tp_size": 8,
            "head_dim": 512,
            "rope_head_dim": 64,
            "index_n_heads": 64,
            "index_head_dim": 128,
            "window_size": 128,
            "compress_ratio": 4,
            "kvcache_quant_mode": common.KVCacheQuantMode.fp8,
            "fmha_quant_mode": common.FMHAQuantMode.bfloat16,
            "gemm_quant_mode": common.GEMMQuantMode.fp8_block,
        }
        shapes = {
            "flash": {
                "num_heads": 8,
                "native_heads": 64,
                "hidden_size": 4096,
                "q_lora_rank": 1024,
                "o_lora_rank": 1024,
                "index_topk": 512,
                "o_groups": 1,
            },
            "pro": {
                "num_heads": 16,
                "native_heads": 128,
                "hidden_size": 7168,
                "q_lora_rank": 1536,
                "o_lora_rank": 1024,
                "index_topk": 1024,
                "o_groups": 2,
            },
        }

        context_results = {}
        generation_results = {}
        for name, shape in shapes.items():
            context_result = comprehensive_perf_db.query_context_deepseek_v4_attention_module(
                **{**common_kwargs, **shape},
                database_mode=common.DatabaseMode.SOL_FULL,
            )
            generation_kwargs = {**common_kwargs, **shape}
            generation_kwargs.pop("prefix")
            generation_result = comprehensive_perf_db.query_generation_deepseek_v4_attention_module(
                **generation_kwargs,
                database_mode=common.DatabaseMode.SOL_FULL,
            )

            assert context_result[0] == max(context_result[1], context_result[2])
            assert generation_result[0] == max(generation_result[1], generation_result[2])
            assert context_result[0] > 0
            assert generation_result[0] > 0
            context_results[name] = context_result
            generation_results[name] = generation_result

        assert context_results["pro"][1] > context_results["flash"][1]
        assert generation_results["pro"][1] > generation_results["flash"][1]

    def test_csa_context_silicon_reads_prefix_resolved_table(self, mutable_comprehensive_perf_db):
        # SCHEME A reads the prefix-resolved silicon table {head}{cr}{prefix}{s}{b}
        # directly (the topK regime change is modeled by the topK-calib DELTA, not
        # a separate raw same-regime piecewise pass). Query (prefix=0, s=4097):
        # c4_len = 4097//4 = 1024 <= index_topk, so the topK DELTA is 0 and the
        # table value at s=4097 is returned unchanged.
        attn_dict = {
            0: {
                4096: {2: _deepseek_v4_value(20.0)},
                4097: {2: _deepseek_v4_value(21.0)},
                8192: {2: _deepseek_v4_value(80.0)},
                12288: {2: _deepseek_v4_value(100.0)},
            }
        }
        db = mutable_comprehensive_perf_db
        db._context_deepseek_v4_attention_module_data = LoadedOpData(
            _context_deepseek_v4_data(4, attn_dict, native_heads=16),
            common.PerfDataFilename.dsv4_csa_context_module,
            "models",
        )
        db._raw_context_deepseek_v4_attention_module_data = None

        base = _deepseek_v4_attn_kwargs(4)
        result = db.query_context_deepseek_v4_attention_module(
            **{**base, "s": 4097, "prefix": 0},
            database_mode=common.DatabaseMode.SILICON,
        )

        assert float(result) == pytest.approx(21.0)
        assert result.energy == pytest.approx(21.0 * 10.0)

    def test_generation_kv_bytes_independent_of_num_heads(self, comprehensive_perf_db):
        """DeepSeek-V4 KV cache stores one ``head_dim``-sized vector per token,
        shared across all attention heads (MLA / MQA-equivalent layout).

        Therefore the KV-traffic component of the attention SOL ``sol_mem`` term
        must NOT scale with ``num_heads``: scaling ``num_heads`` only changes the
        compute (sol_math) and the projection-related weight/activation bytes.

        Regression test for the bug where ``kv_cache_bytes`` was multiplied by
        ``num_heads``, which produced unrealistically large ``sol_mem`` values
        (often >100 ms per decode step at moderate batch sizes). The inflated
        SOL caused HYBRID/silicon latency to fall *below* SOL latency in the
        ``aiconfigurator cli estimate ... --detail all`` "Latency Summary"
        report -- a physical impossibility, since SOL is the per-op roofline
        lower bound and cannot exceed any silicon-measured execution time.
        """
        base = _deepseek_v4_attn_kwargs(4)
        # Decode-mode shape: large batch, kv_len = s - 1, KV traffic dominates sol_mem.
        kwargs = {
            **base,
            "b": 256,
            "s": 8192,
            "num_heads": 16,
            "index_topk": 1024,
        }
        kwargs.pop("prefix")

        # SOL_FULL returns a tuple-like (max(sol_math, sol_mem), sol_math, sol_mem).
        small = comprehensive_perf_db.query_generation_deepseek_v4_attention_module(
            **kwargs,
            database_mode=common.DatabaseMode.SOL_FULL,
        )
        large = comprehensive_perf_db.query_generation_deepseek_v4_attention_module(
            **{**kwargs, "num_heads": 128},
            database_mode=common.DatabaseMode.SOL_FULL,
        )

        # sol_math (index 1) must scale with num_heads: more compute per token.
        assert large[1] > small[1]
        # KV-cache traffic (a major component of sol_mem at this batch) must NOT
        # scale with num_heads -- sol_mem should be much closer to the small case
        # than the 128/16 = 8x ratio that the buggy formula would produce. We
        # leave headroom for projection/activation/weight scaling that DOES
        # legitimately depend on num_heads (Q projection output is num_heads x
        # head_dim per token).
        ratio = float(large[2]) / float(small[2])
        assert ratio < 4.0, (
            f"sol_mem scaled by {ratio:.2f}x when num_heads went 16→128. KV "
            f"cache bytes should be MQA-style (head_dim only), independent of "
            f"num_heads."
        )

    def test_context_silicon_resolves_rank_local_head_bucket(self, mutable_comprehensive_perf_db):
        # SCHEME A: the head axis is the rank-local head count (native // tp), in
        # line with the universal attention convention (per-rank heads, no tp
        # axis). A Pro query at tp=8 (native 128 -> num_heads=16) must resolve the
        # 16-head bucket, not a smaller local-head bucket. cr=4 / prefix=0 ->
        # c4_len=64 <= index_topk, so the topK DELTA is 0 and the raw latency is
        # returned unchanged. Data is prefix-resolved: {head}{cr}{prefix}{s}{b}.
        db = mutable_comprehensive_perf_db
        data = _context_deepseek_v4_data(
            4,
            {0: {256: {2: _deepseek_v4_value(11.0)}}},
            native_heads=8,
        )
        pro_data = _context_deepseek_v4_data(
            4,
            {0: {256: {2: _deepseek_v4_value(22.0)}}},
            native_heads=16,
        )
        _deep_merge_dsv4_dicts(data, pro_data)
        db._context_deepseek_v4_attention_module_data = LoadedOpData(
            data,
            common.PerfDataFilename.dsv4_csa_context_module,
            "models",
        )
        db._raw_context_deepseek_v4_attention_module_data = None

        result = db.query_context_deepseek_v4_attention_module(
            **_deepseek_v4_attn_kwargs(4),
            database_mode=common.DatabaseMode.SILICON,
        )

        assert float(result) == pytest.approx(22.0)

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

    def test_context_silicon_handles_bs1_s54_prefix2816_single_attn_module(self, mutable_comprehensive_perf_db):
        """Full-query regression for the single bs=1, isl=54, prefix=2816 attention module."""
        db = mutable_comprehensive_perf_db
        module_grid = _dsv4_sampled_batch_caps_grid()
        db._context_deepseek_v4_attention_module_data = LoadedOpData(
            _context_deepseek_v4_data(4, module_grid),
            common.PerfDataFilename.dsv4_csa_context_module,
            "mock_dsv4_context_module_tp8",
        )
        db._raw_context_deepseek_v4_attention_module_data = LoadedOpData(
            _context_deepseek_v4_data(4, module_grid),
            common.PerfDataFilename.dsv4_csa_context_module,
            "mock_raw_dsv4_context_module_tp8",
        )
        sparse_grid = _dsv4_sparse_kernel_grid()
        db._dsv4_sparse_kernel_data = {
            "paged_mqa_logits": LoadedOpData(
                sparse_grid,
                common.PerfDataFilename.dsv4_paged_mqa_logits_module,
                "mock_dsv4_paged_mqa_logits_module",
            ),
        }

        result = db.query_context_deepseek_v4_attention_module(
            **{
                **_deepseek_v4_attn_kwargs(4),
                "b": 1,
                "s": 54,
                "prefix": 2816,
                "num_heads": 8,
            },
            database_mode=common.DatabaseMode.SILICON,
        )

        assert float(result) > 0
        assert result.energy >= 0

    def test_context_silicon_uses_prefix_anchor_without_topk_calib(self, mutable_comprehensive_perf_db):
        """SCHEME A correction is the topK-calib DELTA (flat - top_last), not the
        old paged_mqa_logits sparse-kernel delta. When the topK calib is absent,
        the prefix CSA query returns the measured module latency UNCORRECTED
        (DELTA = 0) instead of raising — the prefix-resolved table already carries
        the prefix in its leading axis, so there is no s+prefix double-count."""
        db = mutable_comprehensive_perf_db
        # prefix-resolved {head}{cr}{prefix}{s}{b}; prefix=8192/s=54 -> c4_len=2061
        # > index_topk, so a correction WOULD apply if a calib were loaded.
        db._context_deepseek_v4_attention_module_data = LoadedOpData(
            _context_deepseek_v4_data(
                4,
                {
                    0: {54: {1: _deepseek_v4_value(2.0)}},
                    8192: {54: {1: _deepseek_v4_value(5.0)}},
                },
                native_heads=16,
            ),
            common.PerfDataFilename.dsv4_csa_context_module,
            "models",
        )
        db._raw_context_deepseek_v4_attention_module_data = None
        db._dsv4_csa_topk_calib = None  # no topK calibration loaded

        base = {**_deepseek_v4_attn_kwargs(4), "b": 1, "s": 54, "num_heads": 16}
        prefix0 = db.query_context_deepseek_v4_attention_module(
            **{**base, "prefix": 0},
            database_mode=common.DatabaseMode.SILICON,
        )
        prefix8192 = db.query_context_deepseek_v4_attention_module(
            **{**base, "prefix": 8192},
            database_mode=common.DatabaseMode.SILICON,
        )

        assert float(prefix0) == pytest.approx(2.0)
        assert float(prefix8192) == pytest.approx(5.0)

    def test_context_silicon_handles_b3_s2682_prefix0_num_heads8_from_sampled_batches(
        self, mutable_comprehensive_perf_db
    ):
        """Full-query regression for sampled b=2/b=4 data and query b=3."""
        db = mutable_comprehensive_perf_db
        module_grid = _dsv4_sampled_batch_caps_grid()
        db._context_deepseek_v4_attention_module_data = LoadedOpData(
            _context_deepseek_v4_data(4, module_grid),
            common.PerfDataFilename.dsv4_csa_context_module,
            "mock_dsv4_context_module_tp8",
        )
        db._raw_context_deepseek_v4_attention_module_data = LoadedOpData(
            _context_deepseek_v4_data(4, module_grid),
            common.PerfDataFilename.dsv4_csa_context_module,
            "mock_raw_dsv4_context_module_tp8",
        )

        result = db.query_context_deepseek_v4_attention_module(
            **{
                **_deepseek_v4_attn_kwargs(4),
                "b": 3,
                "s": 2682,
                "prefix": 0,
                "num_heads": 8,
            },
            database_mode=common.DatabaseMode.SILICON,
        )

        assert float(result) > 0
        assert result.energy >= 0


def test_deepseek_v4_static_sol_runs_end_to_end(mutable_comprehensive_perf_db):
    db = mutable_comprehensive_perf_db
    db.system_spec["gpu"]["mem_capacity"] = 288400343040
    db.system_spec["misc"]["nccl_mem"] = {1: 0, 2: 0, 4: 0, 8: 0}
    db.system_spec["misc"]["other_mem"] = 0
    model_config = config.ModelConfig(
        tp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        nextn=1,
        overwrite_num_layers=2,
    )
    model = get_model("sgl-project/DeepSeek-V4-Flash-FP8", model_config, backend_name="trtllm")
    backend = TRTLLMBackend()
    runtime = RuntimeConfig(batch_size=1, beam_width=1, isl=128, osl=4, prefix=0)

    db.set_default_database_mode(common.DatabaseMode.SOL)
    summary = backend.run_static(model, db, runtime, mode="static", stride=1)
    assert sum(summary.get_context_latency_dict().values()) > 0
    assert sum(summary.get_generation_latency_dict().values()) > 0


def test_sglang_deepseek_v4_pro_moe_workspace_uses_residual_hidden_size(mutable_comprehensive_perf_db):
    db = mutable_comprehensive_perf_db
    db.system_spec["gpu"]["mem_capacity"] = 198674743296  # GB200 189471 MiB
    db.system_spec["misc"]["nccl_mem"] = {1: 0, 2: 358612992, 4: 411041792, 8: 411041792}
    db.system_spec["misc"]["other_mem"] = 3758096384

    model_config = config.ModelConfig(
        tp_size=1,
        pp_size=1,
        attention_dp_size=8,
        moe_tp_size=1,
        moe_ep_size=8,
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        moe_quant_mode=common.MoEQuantMode.w4a8_mxfp4_mxfp8,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        comm_quant_mode=common.CommQuantMode.half,
        moe_backend="megamoe",
        nextn=0,
    )
    model = get_model("deepseek-ai/DeepSeek-V4-Pro", model_config, backend_name="sglang")

    memory = SGLANGBackend()._get_memory_usage(
        model,
        db,
        batch_size=1,
        beam_width=1,
        isl=8192,
        osl=1024,
    )

    num_tokens = 8192
    attention_width = model._num_heads * model._head_size
    residual_width = model._hidden_size
    assert model.activation_hidden_size == residual_width
    assert attention_width > residual_width

    tp_activation_factor = 28
    attention_workspace = 2 * num_tokens * attention_width * tp_activation_factor
    moe_scale_workspace = (
        num_tokens
        * residual_width
        * model.config.attention_dp_size
        * model._num_experts
        * model._topk
        / model.config.moe_ep_size
        / 128
        * 4
    )
    expected_activation_gib = (attention_workspace + moe_scale_workspace) * 1.15 / (1 << 30)

    assert memory["activations"] == pytest.approx(expected_activation_gib)

    old_moe_scale_workspace = (
        num_tokens
        * attention_width
        * model.config.attention_dp_size
        * model._num_experts
        * model._topk
        / model.config.moe_ep_size
        / 128
        * 4
    )
    old_activation_gib = (attention_workspace + old_moe_scale_workspace) * 1.15 / (1 << 30)
    assert memory["activations"] < old_activation_gib

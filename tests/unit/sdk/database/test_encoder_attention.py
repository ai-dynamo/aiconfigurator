# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import EncoderAttention

pytestmark = pytest.mark.unit


class TestQueryEncoderAttention:
    """Test cases for query_encoder_attention method (non-causal, MHA, no KV cache)."""

    def test_sol_mode_no_causal_factor(self, comprehensive_perf_db):
        """SOL FLOPs for encoder is full N^2 (no /2 for causality)."""
        b, s, n, h = 2, 64, 16, 72
        fmha = common.FMHAQuantMode.bfloat16

        result = comprehensive_perf_db.query_encoder_attention(b, s, n, h, fmha, database_mode=common.DatabaseMode.SOL)

        # Non-causal full N^2: ops = 2*b*s*s*n*h*2 (no /2)
        ops = 2 * b * s * s * n * h * 2
        mem_bytes = 2 * b * (3 * n * s * h + n * s * h)
        sol_math = ops / comprehensive_perf_db.system_spec["gpu"]["bfloat16_tc_flops"] * 1000 / fmha.value.compute
        sol_mem = mem_bytes / comprehensive_perf_db.system_spec["gpu"]["mem_bw"] * 1000
        expected = max(sol_math, sol_mem)

        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_sol_is_twice_causal_flops(self, comprehensive_perf_db):
        """Encoder SOL FLOPs should be 2x the LLM causal SOL FLOPs (compute-bound regime)."""
        b, s, n, h = 1, 4096, 32, 128  # large s, compute-bound
        fmha = common.FMHAQuantMode.bfloat16
        kv = common.KVCacheQuantMode.bfloat16

        encoder_sol_math = comprehensive_perf_db.query_encoder_attention(
            b, s, n, h, fmha, database_mode=common.DatabaseMode.SOL_FULL
        )[1]
        # context_attention (causal) at same config: prefix=0
        causal_sol_math = comprehensive_perf_db.query_context_attention(
            b, s, 0, n, n, kv, fmha, database_mode=common.DatabaseMode.SOL_FULL
        )[1]

        # encoder is non-causal -> exactly 2x compute of causal
        assert math.isclose(encoder_sol_math, 2 * causal_sol_math, rel_tol=1e-6)

    def test_silicon_mode_lookup(self, comprehensive_perf_db):
        """SILICON mode looks up [fmha][head_size][n][s][b]."""
        b, s, n, h = 2, 64, 16, 72
        fmha = common.FMHAQuantMode.bfloat16

        result = comprehensive_perf_db.query_encoder_attention(
            b, s, n, h, fmha, database_mode=common.DatabaseMode.SILICON
        )
        expected = comprehensive_perf_db._encoder_attention_data[fmha][h][n][s][b]["latency"]
        assert math.isclose(result, expected, rel_tol=1e-6)


class TestEncoderAttentionOp:
    """Test cases for EncoderAttention op class."""

    def test_query_matches_database(self, comprehensive_perf_db):
        """EncoderAttention.query() returns query_encoder_attention latency (no RoPE)."""
        op = EncoderAttention(
            "encoder_attention",
            scale_factor=1.0,
            num_heads=16,
            head_size=72,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            partial_rotary_factor=0.0,  # disable RoPE for clean comparison
        )
        result = op.query(comprehensive_perf_db, batch_size=2, s=64)
        expected = comprehensive_perf_db.query_encoder_attention(2, 64, 16, 72, common.FMHAQuantMode.bfloat16)
        assert math.isclose(float(result), float(expected), rel_tol=1e-6)

    def test_scale_factor_multiplies_latency(self, comprehensive_perf_db):
        """scale_factor multiplies the per-layer latency (e.g., depth for ViT block stack)."""
        depth = 32
        op_single = EncoderAttention("encoder_attention", 1.0, 16, 72, partial_rotary_factor=0.0)
        op_depth = EncoderAttention("encoder_attention", depth, 16, 72, partial_rotary_factor=0.0)
        r1 = op_single.query(comprehensive_perf_db, batch_size=2, s=64)
        rd = op_depth.query(comprehensive_perf_db, batch_size=2, s=64)
        assert math.isclose(float(rd), float(r1) * depth, rel_tol=1e-6)

    def test_partial_rope_adds_latency(self, comprehensive_perf_db):
        """partial_rotary_factor > 0 adds RoPE mem-op latency on top of attention."""
        kwargs = dict(scale_factor=1.0, num_heads=16, head_size=72)
        op_no_rope = EncoderAttention("no_rope", partial_rotary_factor=0.0, **kwargs)
        op_full_rope = EncoderAttention("full_rope", partial_rotary_factor=1.0, **kwargs)
        op_half_rope = EncoderAttention("half_rope", partial_rotary_factor=0.5, **kwargs)

        r_none = float(op_no_rope.query(comprehensive_perf_db, batch_size=2, s=64))
        r_full = float(op_full_rope.query(comprehensive_perf_db, batch_size=2, s=64))
        r_half = float(op_half_rope.query(comprehensive_perf_db, batch_size=2, s=64))

        # half RoPE adds half of full RoPE's extra latency
        rope_full = r_full - r_none
        rope_half = r_half - r_none
        assert rope_full > 0
        assert math.isclose(rope_half, 0.5 * rope_full, rel_tol=1e-6)

    def test_get_weights_returns_zero(self, comprehensive_perf_db):
        """EncoderAttention has no weights (attention is a pure compute op)."""
        op = EncoderAttention("encoder_attention", 1.0, 16, 72)
        assert op.get_weights() == 0.0

    @pytest.mark.parametrize("bad_mode", [common.FMHAQuantMode.fp8, common.FMHAQuantMode.fp8_block])
    def test_init_rejects_non_bfloat16_quant_mode(self, bad_mode):
        """Only bfloat16 has encoder perf data; other modes must fail fast."""
        with pytest.raises(ValueError, match="bfloat16"):
            EncoderAttention("encoder_attention", 1.0, 16, 72, fmha_quant_mode=bad_mode)

    @pytest.mark.parametrize("bad_factor", [-0.1, 1.1, 2.0])
    def test_init_rejects_out_of_range_partial_rotary_factor(self, bad_factor):
        """partial_rotary_factor must lie in [0.0, 1.0]."""
        with pytest.raises(ValueError, match=r"partial_rotary_factor"):
            EncoderAttention("encoder_attention", 1.0, 16, 72, partial_rotary_factor=bad_factor)

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import EncoderAttention

pytestmark = pytest.mark.unit


# Retired with #1357 PR-5: the encoder-attention query math this file pinned
# on the synthetic fixture (non-causal SOL FLOPs, the 2x-causal relation,
# silicon table lookups, op-vs-facade equality, scale-factor and partial-RoPE
# latency composition) moved to the compiled engine and is anchored by
# the frozen parity
# goldens. Construction contracts stay below.


class TestEncoderAttentionOp:
    """Test cases for EncoderAttention op class."""

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

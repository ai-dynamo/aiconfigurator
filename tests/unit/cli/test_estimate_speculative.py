# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cli_estimate speculative-block wiring (agg/static paths).

Uses the real Qwen3-8B model + h100_sxm vLLM tables with the eagle3 scheme
(chain k3) — the same configuration the Phase-0/1 campaigns validated —
so the assertions exercise the full resolve -> attach -> price -> project
chain, not mocks.
"""

from __future__ import annotations

import pytest

from aiconfigurator.cli.api import cli_estimate

pytestmark = pytest.mark.unit

EAGLE3_CONFIG = {
    "model_type": "llama",
    "num_hidden_layers": 1,
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "draft_vocab_size": 32000,
    "sliding_window": None,
    "use_sliding_window": False,
}
SPEC_BLOCK = {
    "method": "eagle3",
    "params": {"num_speculative_tokens": 3},
    "draft_config": EAGLE3_CONFIG,
    "accepted_tokens": 1.8,  # measured gsm8k chain E+1 = 2.80
}
COMMON = dict(
    model_path="Qwen/Qwen3-8B",
    system_name="h100_sxm",
    backend_name="vllm",
    backend_version="0.24.0",
    isl=64,
    osl=261,
    batch_size=8,
    gemm_quant_mode="bfloat16",
    kvcache_quant_mode="bfloat16",
    fmha_quant_mode="bfloat16",
)


class TestEstimateSpeculativeBlock:
    def test_agg_scheme_folds_acceptance(self):
        baseline = cli_estimate(mode="agg", **COMMON)
        spec = cli_estimate(mode="agg", speculative=SPEC_BLOCK, **COMMON)
        # verify-width rounds cost more than an AR step but commit 2.8
        # tokens: TPOT must land strictly between round/1 and baseline.
        assert 0 < spec.tpot < baseline.tpot
        # progress fold is 1 + accepted = 2.8: the speedup cannot exceed it.
        assert baseline.tpot / spec.tpot < 2.8
        # accepted_tokens must actually drive the projection: the same
        # scheme at zero acceptance pays the verify round for one token
        # per step and must be strictly slower.
        zero = cli_estimate(mode="agg", speculative={**SPEC_BLOCK, "accepted_tokens": 0.0}, **COMMON)
        assert spec.tpot < zero.tpot

    def test_static_gen_scheme_projection(self):
        baseline = cli_estimate(mode="static_gen", **COMMON)
        spec = cli_estimate(mode="static_gen", speculative=SPEC_BLOCK, **COMMON)
        assert 0 < spec.tpot < baseline.tpot

    def test_mtp_sugar_still_desugars_to_nextn(self):
        # mtp inside the block must be EXACTLY the legacy pair: a wrong
        # acceptance mapping or a silent AR fallback would still produce a
        # positive tpot, so compare against the legacy nextn output.
        explicit = cli_estimate(
            mode="static_gen",
            speculative={"method": "mtp", "params": {"depth": 1}, "accepted_tokens": 0.7},
            **COMMON,
        )
        legacy = cli_estimate(mode="static_gen", nextn=1, nextn_accepted=0.7, **COMMON)
        baseline = cli_estimate(mode="static_gen", **COMMON)
        assert explicit.tpot == pytest.approx(legacy.tpot)
        assert explicit.tpot != pytest.approx(baseline.tpot)  # not an AR fallback

    def test_mtp_block_still_valid_for_disagg(self):
        # The disagg rejection below covers SCHEME methods only: mtp
        # desugars to the legacy nextn pair and must keep working outside
        # agg/static — pin it against the equivalent legacy configuration.
        disagg_args = dict(
            prefill_tp_size=1,
            prefill_pp_size=1,
            prefill_batch_size=1,
            prefill_num_workers=1,
            decode_tp_size=1,
            decode_pp_size=1,
            decode_batch_size=8,
            decode_num_workers=1,
        )
        explicit = cli_estimate(
            mode="disagg",
            speculative={"method": "mtp", "params": {"depth": 1}, "accepted_tokens": 0.7},
            **disagg_args,
            **COMMON,
        )
        legacy = cli_estimate(mode="disagg", nextn=1, nextn_accepted=0.7, **disagg_args, **COMMON)
        assert explicit.tpot == pytest.approx(legacy.tpot)

    def test_scheme_rejected_for_disagg(self):
        with pytest.raises(NotImplementedError, match="agg/static"):
            cli_estimate(
                mode="disagg",
                speculative=SPEC_BLOCK,
                prefill_tp_size=1,
                prefill_pp_size=1,
                prefill_batch_size=1,
                prefill_num_workers=1,
                decode_tp_size=1,
                decode_pp_size=1,
                decode_batch_size=8,
                decode_num_workers=1,
                **COMMON,
            )

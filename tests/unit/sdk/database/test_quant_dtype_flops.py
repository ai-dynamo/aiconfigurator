# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Op-level guards for per-dtype tc_flops resolution (issue #1398).

The shared fixtures deliberately pin ``int8/fp8/fp4_tc_flops`` at the
historical 1/2/4 ratios of bf16 so pre-existing numeric assertions stay
valid — which also means those tests cannot distinguish the new per-dtype
resolution from the old ``bfloat16_tc_flops * compute_factor`` scaling.
The tests here break the ratio on purpose: they monkeypatch a divergent
per-dtype entry (mirroring b300, where ``fp4_tc_flops`` is 14 PFLOPS versus
``bf16*4`` = 9) and assert the SOL math follows the YAML entry, so any
regression back to compute-factor scaling fails loudly at the op level.

Also covers the eager-resolution contract: a missing ``*_tc_flops`` entry
rejects the query at entry in every database mode — including SILICON, where
an exact table hit never invokes the SOL closure — keeping the Python engine
aligned with Rust, which resolves flops with ``?`` before the table lookup.
"""

from __future__ import annotations

import math

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.errors import MissingSystemFlopsError


class TestDivergentRatioSol:
    """SOL math must read the per-dtype YAML entry, not scale bf16."""

    def test_moe_sol_uses_fp4_entry_not_bf16_ratio(self, comprehensive_perf_db, monkeypatch):
        gpu = comprehensive_perf_db.system_spec["gpu"]
        # 10x bf16 — far from the fixture's 4x — so a compute-factor
        # regression produces a 2.5x-wrong sol_math.
        monkeypatch.setitem(gpu, "fp4_tc_flops", gpu["bfloat16_tc_flops"] * 10)

        num_tokens, hidden_size, inter_size = 512, 4096, 16384
        topk, num_experts = 2, 8
        _, sol_math, _ = comprehensive_perf_db.query_moe(
            num_tokens,
            hidden_size,
            inter_size,
            topk,
            num_experts,
            1,
            1,
            common.MoEQuantMode.nvfp4,
            "uniform",
            database_mode=common.DatabaseMode.SOL_FULL,
        )

        ops = num_tokens * topk * hidden_size * inter_size * 3 * 2
        assert math.isclose(sol_math, ops / gpu["fp4_tc_flops"] * 1000, rel_tol=1e-9)
        legacy_math = ops / (gpu["bfloat16_tc_flops"] * common.MoEQuantMode.nvfp4.value.compute) * 1000
        assert not math.isclose(sol_math, legacy_math, rel_tol=1e-2)

    def test_generation_attention_sol_uses_fp8_entry(self, comprehensive_perf_db, monkeypatch):
        gpu = comprehensive_perf_db.system_spec["gpu"]
        monkeypatch.setitem(gpu, "fp8_tc_flops", gpu["bfloat16_tc_flops"] * 5)

        b, s, n, n_kv = 4, 4096, 32, 8
        _, sol_math, _ = comprehensive_perf_db.query_generation_attention(
            b,
            s,
            n,
            n_kv,
            common.KVCacheQuantMode.fp8,
            database_mode=common.DatabaseMode.SOL_FULL,
        )

        ops = 2 * b * n * 128 * 2 * (s - 1)
        assert math.isclose(sol_math, ops / gpu["fp8_tc_flops"] * 1000, rel_tol=1e-9)


class TestEagerResolution:
    """A missing entry rejects the query at entry in EVERY mode (Rust parity)."""

    @pytest.mark.parametrize(
        "database_mode",
        [common.DatabaseMode.SOL, common.DatabaseMode.HYBRID, common.DatabaseMode.SILICON],
    )
    def test_moe_missing_fp4_entry_rejected_in_all_modes(self, comprehensive_perf_db, monkeypatch, database_mode):
        gpu = comprehensive_perf_db.system_spec["gpu"]
        monkeypatch.delitem(gpu, "fp4_tc_flops", raising=False)

        with pytest.raises(MissingSystemFlopsError, match="fp4_tc_flops"):
            comprehensive_perf_db.query_moe(
                8,
                1024,
                4096,
                2,
                8,
                1,
                1,
                common.MoEQuantMode.nvfp4,
                "uniform",
                database_mode=database_mode,
            )

    def test_wideep_generation_mla_validates_derived_dtype_not_label(self, comprehensive_perf_db, monkeypatch):
        """The generation SOL dtype is re-derived from the kv-cache dtype; the
        fmha label is inert. The eager check must follow the derivation: an
        fp8 KV with a bf16 label needs fp8_tc_flops (no exact-hit bypass),
        while a bf16 KV with an fp8 label must not be falsely rejected."""
        gpu = comprehensive_perf_db.system_spec["gpu"]
        monkeypatch.delitem(gpu, "fp8_tc_flops", raising=False)

        with pytest.raises(MissingSystemFlopsError, match="fp8_tc_flops"):
            comprehensive_perf_db.query_wideep_generation_mla(
                4,
                1024,
                8,
                common.KVCacheQuantMode.fp8,
                common.FMHAQuantMode.bfloat16,
                database_mode=common.DatabaseMode.SOL,
            )

        result = comprehensive_perf_db.query_wideep_generation_mla(
            4,
            1024,
            8,
            common.KVCacheQuantMode.bfloat16,
            common.FMHAQuantMode.fp8,
            database_mode=common.DatabaseMode.SOL,
        )
        assert float(result) > 0

    def test_pre_sm89_fp8_kv_derives_bf16_pipeline(self, comprehensive_perf_db, monkeypatch):
        """Ampere-class hardware has no fp8 tensor cores: the decode kernel
        dequantizes fp8 KV and runs the MMA on the bf16 pipeline, so fp8-kv
        generation queries must neither demand fp8_tc_flops nor price the
        fp8 rate (a100 ships 2,534 measured fp8-kv generation-MLA rows
        collected exactly that way)."""
        gpu = comprehensive_perf_db.system_spec["gpu"]
        monkeypatch.setitem(gpu, "sm_version", 80)
        monkeypatch.delitem(gpu, "fp8_tc_flops", raising=False)

        # Distinct shape from the fp8-entry test above: these PerfDatabase
        # query methods are lru_cached, and identical args would replay the
        # earlier fp8-pipeline result instead of re-deriving under sm=80.
        _, sol_math, _ = comprehensive_perf_db.query_generation_attention(
            2,
            2048,
            32,
            8,
            common.KVCacheQuantMode.fp8,
            database_mode=common.DatabaseMode.SOL_FULL,
        )
        ops = 2 * 2 * 32 * 128 * 2 * (2048 - 1)
        assert math.isclose(sol_math, ops / gpu["bfloat16_tc_flops"] * 1000, rel_tol=1e-9)

    def test_sm89_plus_fp8_kv_still_requires_fp8_entry(self, comprehensive_perf_db, monkeypatch):
        gpu = comprehensive_perf_db.system_spec["gpu"]
        monkeypatch.setitem(gpu, "sm_version", 90)
        monkeypatch.delitem(gpu, "fp8_tc_flops", raising=False)

        with pytest.raises(MissingSystemFlopsError, match="fp8_tc_flops"):
            comprehensive_perf_db.query_generation_attention(
                2, 1024, 32, 8, common.KVCacheQuantMode.fp8, database_mode=common.DatabaseMode.SOL
            )

    def test_gemm_missing_fp8_entry_rejected_in_silicon_mode(self, stub_perf_db, monkeypatch):
        """SILICON specifically: an exact table hit must not bypass the check."""
        gpu = stub_perf_db.system_spec["gpu"]
        monkeypatch.delitem(gpu, "fp8_tc_flops", raising=False)

        with pytest.raises(MissingSystemFlopsError, match="fp8_tc_flops"):
            stub_perf_db.query_gemm(16, 1024, 1024, common.GEMMQuantMode.fp8, database_mode=common.DatabaseMode.SILICON)

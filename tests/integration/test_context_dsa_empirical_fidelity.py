# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-data regressions for Context DSA ragged utilization interpolation.

These points are the former prefill tails from the expanded fidelity matrix.
They require the B200/SGLang performance database from Git LFS, so they belong
to the build/integration suite rather than the data-independent unit suite.
"""

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import util_empirical
from aiconfigurator.sdk.operations.dsa import ContextDSAModule, GenerationDSAModule
from aiconfigurator.sdk.perf_database import get_database

pytestmark = [pytest.mark.integration, pytest.mark.build]


@pytest.mark.parametrize(
    ("num_heads", "sequence", "kv_mode", "gemm_mode", "architecture", "expected_silicon", "expected_empirical"),
    [
        (
            8,
            14995,
            common.KVCacheQuantMode.fp8,
            common.GEMMQuantMode.bfloat16,
            "GlmMoeDsaForCausalLM",
            9.933732262398413,
            9.552782289319444,
        ),
        (
            16,
            2704,
            common.KVCacheQuantMode.fp8,
            common.GEMMQuantMode.fp8_block,
            "DeepseekV32ForCausalLM",
            1.996558923832109,
            1.6489018500570642,
        ),
        (
            64,
            2704,
            common.KVCacheQuantMode.fp8,
            common.GEMMQuantMode.fp8_block,
            "DeepseekV32ForCausalLM",
            2.6482720580148342,
            2.146270787994807,
        ),
    ],
)
def test_b200_sglang_context_dsa_prefill_tails(
    num_heads,
    sequence,
    kv_mode,
    gemm_mode,
    architecture,
    expected_silicon,
    expected_empirical,
):
    ContextDSAModule.clear_cache()
    util_empirical.clear_grid_cache()
    database = get_database("b200_sxm", "sglang", "0.5.10", database_mode="SILICON")
    query = {
        "b": 3,
        "s": sequence,
        "prefix": 0,
        "num_heads": num_heads,
        "kvcache_quant_mode": kv_mode,
        "fmha_quant_mode": common.FMHAQuantMode.bfloat16,
        "gemm_quant_mode": gemm_mode,
        "architecture": architecture,
    }

    silicon = float(database.query_context_dsa_module(**query, database_mode=common.DatabaseMode.SILICON))
    empirical = float(database.query_context_dsa_module(**query, database_mode=common.DatabaseMode.EMPIRICAL))

    assert silicon == pytest.approx(expected_silicon)
    assert empirical == pytest.approx(expected_empirical)
    if architecture == "GlmMoeDsaForCausalLM":
        assert abs(empirical / silicon - 1.0) < 0.05
    else:
        assert abs(empirical / silicon - 1.0) < 0.20


def test_b300_vllm_active_dsa_rows_outrank_shared_fallback():
    """Active VLLM measurements must survive overlapping TRT-LLM fallback rows."""
    ContextDSAModule.clear_cache()
    GenerationDSAModule.clear_cache()
    util_empirical.clear_grid_cache()
    database = get_database("b300_sxm", "vllm", "0.19.0", database_mode="EMPIRICAL")

    context = float(
        database.query_context_dsa_module(
            b=7,
            s=1003,
            prefix=0,
            num_heads=16,
            kvcache_quant_mode=common.KVCacheQuantMode.fp8,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.fp8_block,
            architecture="DeepseekV32ForCausalLM",
            database_mode=common.DatabaseMode.EMPIRICAL,
        )
    )
    generation = float(
        database.query_generation_dsa_module(
            b=7,
            s=1003,
            num_heads=16,
            kv_cache_dtype=common.KVCacheQuantMode.fp8,
            gemm_quant_mode=common.GEMMQuantMode.fp8_block,
            architecture="DeepseekV32ForCausalLM",
            database_mode=common.DatabaseMode.EMPIRICAL,
        )
    )

    # Before first-wins merging, later shared fallback rows produced roughly
    # 1.510 ms context and 0.0875 ms generation at this point.
    assert context == pytest.approx(4.394897400594101)
    assert generation == pytest.approx(0.19692560748875187)

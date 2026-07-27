# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-value pins for the generation MLA module table on fp8-KV decode.

Dropping the degenerate fmha axis makes the module table live for
fp8-checkpoint DeepSeek-V3/R1/Kimi decode (previously the fp8 fmha label
missed the fmha-keyed table and FallbackOp silently composed the granular
path).  That moves decode predictions by tens of percent on every system
shipping mla_generation_module data, so these pins hold the module-table
numbers still: any drift here is a data or interpolation change, not noise.
Values minted on the PR branch via ``get_database`` (production loading,
shared layer included) in SILICON mode.
"""

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import get_database

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("system", "b", "s", "num_heads", "gemm", "expected_ms"),
    [
        ("gb200", 8, 4097, 128, common.GEMMQuantMode.bfloat16, 0.1536000000000000),
        ("gb200", 8, 3000, 128, common.GEMMQuantMode.bfloat16, 0.1519930664062500),
        ("gb200", 64, 4096, 128, common.GEMMQuantMode.bfloat16, 0.1999953125000000),
        ("h200_sxm", 64, 4096, 16, common.GEMMQuantMode.fp8_block, 0.1904882812500000),
    ],
)
def test_generation_mla_module_fp8_kv_exact_values(system, b, s, num_heads, gemm, expected_ms):
    db = get_database(system, "trtllm", "1.3.0rc10")
    db.set_default_database_mode(common.DatabaseMode.SILICON)
    result = db.query_generation_mla_module(
        b=b,
        s=s,
        num_heads=num_heads,
        kv_cache_dtype=common.KVCacheQuantMode.fp8,
        gemm_quant_mode=gemm,
    )
    assert float(result) == pytest.approx(expected_ms, rel=1e-9)

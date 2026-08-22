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
        # Re-minted after the shared-layer first-wins fix: these keys have no
        # row in the trtllm primary and fill from cross-backend vLLM siblings;
        # last-wins accidentally let the OLDEST framework version (0.19.0) win,
        # first-wins restores the design's newest-first priority (0.24.0).
        ("gb200", 8, 4097, 128, common.GEMMQuantMode.bfloat16, 0.1530000000000000),
        ("gb200", 8, 3000, 128, common.GEMMQuantMode.bfloat16, 0.1528928710937500),
        ("gb200", 64, 4096, 128, common.GEMMQuantMode.bfloat16, 0.2011936523437500),
        # Re-minted after the shared-layer first-wins fix: the old pin
        # (0.19048828125) captured a 1.2.0rc5 sibling row that silently
        # overrode 1.3.0rc10's own row at this key (last-wins loader bug).
        ("h200_sxm", 64, 4096, 16, common.GEMMQuantMode.fp8_block, 0.1146884765625000),
    ],
)
def test_generation_mla_module_fp8_kv_exact_values(system, b, s, num_heads, gemm, expected_ms):
    from aiconfigurator_core.sdk.engine import _evaluate_single_op
    from aiconfigurator_core.sdk.operations.mla import MLAModule

    db = get_database(system, "trtllm", "1.3.0rc10", allow_unlisted_version=True)
    db.set_default_database_mode(common.DatabaseMode.SILICON)
    # The retired query_generation_mla_module shim's exact twin (bfloat16
    # FMHA is ignored by the decode table).
    op = MLAModule(
        "generation_mla_module_query",
        1.0,
        False,
        num_heads,
        common.KVCacheQuantMode.fp8,
        common.FMHAQuantMode.bfloat16,
        gemm,
    )
    result = _evaluate_single_op(db, op, is_context=False, batch_size=int(b), s=int(s))
    assert float(result) == pytest.approx(expected_ms, rel=1e-9)

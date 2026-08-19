# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIC-1747: a DSA perf table with no ``*_skip_indexer`` rows must not kill the model.

GLM-5.2 shares one topk index across a layer group and amortizes
``per_layer = w*full + (1-w)*skip`` (w = 21/78 = 0.2692) off directly-collected
skip-indexer rows. Those rows exist only in the sglang collector's 0.5.14 output
(PR #1556 collected h200/h100/b200; 0.5.9-0.5.12 tables have none anywhere), so
the skip query used to hard-fail every GLM-5.2 sweep point on skip-less tables.

Policy (mirrors ``models/deepseek_v32.py``'s no-skip-producer gate): the engine
operator degrades the amortization to all-full instead of erroring, and stays
byte-identical wherever the skip rows exist. The per-call Python query stack and
the legacy Python parquet loaders are gone (PR-5/PR-6/PR-7), so ALL behavioral
regressions — degrade-to-all-full with exact pins, mixed amortization preserved
with a blend identity, SOL-mode blend retention, the full/skip ``op_name``
split — live in the Rust operator tests (``operators/dsa.rs``). This file keeps
the one Python surface left: the shipped table-view availability the engine
degradation keys off.
"""

import pytest

from aiconfigurator.sdk.operations.dsa import ContextDSAModule, GenerationDSAModule

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("system", "version", "expect_skip_rows"),
    [
        # 0.5.9-0.5.12-era tables have no skip rows anywhere (the AIC-1747
        # probe collection covered 0.5.14 only) — this is the gap the
        # degradation now exists for. Anchored to tables no in-flight data PR
        # touches, matching the Rust twin's test anchors.
        ("h200_sxm", "0.5.10", False),
        ("b200_sxm", "0.5.14", True),
        ("gb200", "0.5.14", True),
    ],
)
def test_shipped_sglang_skip_row_availability(system, version, expect_skip_rows):
    """Pins the shipped skip-row availability the engine degradation keys off."""
    from aiconfigurator_core.sdk.perf_database import get_database

    db = get_database(system, "sglang", version)
    ContextDSAModule.load_data(db)
    GenerationDSAModule.load_data(db)

    assert bool(db._context_dsa_module_skip_data) is expect_skip_rows
    assert bool(db._generation_dsa_module_skip_data) is expect_skip_rows
    # Either way the FULL table loads — the model must never die at load.
    assert db._context_dsa_module_data
    assert db._generation_dsa_module_data

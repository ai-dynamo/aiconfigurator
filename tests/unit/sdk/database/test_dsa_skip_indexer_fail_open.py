# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIC-1747: a DSA perf table with no ``*_skip_indexer`` rows must not kill the model.

GLM-5.2 shares one topk index across a layer group and amortizes
``per_layer = w*full + (1-w)*skip`` (w = 21/78 = 0.2692) off directly-collected
skip-indexer rows. Those rows exist only in the sglang collector's 0.5.14 output
(PR #1556 collected h200/h100/b200; 0.5.9-0.5.12 tables have none anywhere), so
the skip query used to hard-fail every GLM-5.2 sweep point on skip-less tables.

Policy (mirrors ``models/deepseek_v32.py``'s no-skip-producer gate): degrade the
amortization to all-full instead of erroring, and stay byte-identical wherever
the skip rows DO exist. Since the per-call Python query stack was retired behind
engine-routed shims (PR-5/PR-6), the amortization and its degradation execute in
the compiled engine — the behavioral regressions (degrade-to-all-full, mixed
amortization preserved, SOL-mode blend retention, exact twin pins) live in the
Rust operator tests (``operators/dsa.rs``). This file keeps the Python-side
surfaces that still exist: the parquet loaders' full/skip split and the shipped
table-view availability the degradation keys off.
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aiconfigurator.sdk.operations.dsa import (
    ContextDSAModule,
    GenerationDSAModule,
    load_context_dsa_module_data,
    load_generation_dsa_module_data,
)

pytestmark = pytest.mark.unit

GLM5_ARCHITECTURE = "GlmMoeDsaForCausalLM"

FULL_LATENCY = 10.0


def _write_dsa_parquet(path, op_names, latency=FULL_LATENCY):
    """Minimal DSA module table in the shipped sglang schema.

    ``op_names`` selects which variants are present; passing only
    ``dsa_context_module`` reproduces a skip-less table (the pre-#1556 h200
    state, still the live state for sglang 0.5.9-0.5.12).
    """
    rows = [{"op_name": name} for name in op_names]
    table = pa.table(
        {
            "framework": ["sglang"] * len(rows),
            "version": ["0.5.14"] * len(rows),
            "device": ["h200_sxm"] * len(rows),
            "op_name": [r["op_name"] for r in rows],
            "kernel_source": ["sglang_dsa_indexer_trtllm"] * len(rows),
            "model": ["zai-org/GLM-5.2-FP8"] * len(rows),
            "architecture": [GLM5_ARCHITECTURE] * len(rows),
            "mla_dtype": ["bfloat16"] * len(rows),
            "kv_cache_dtype": ["fp8"] * len(rows),
            "gemm_type": ["fp8_block"] * len(rows),
            "num_heads": [64] * len(rows),
            "batch_size": [1] * len(rows),
            "isl": [4096] * len(rows),
            "tp_size": [1] * len(rows),
            "step": [0] * len(rows),
            "latency": [latency] * len(rows),
        }
    )
    pq.write_table(table, path)
    return str(path)


# ───────────────────────────────────────────────────────────────────────
# Loader: the full-only table loads, and its skip variant is empty (no raise)
# ───────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("loader", "op_name"),
    [
        (load_context_dsa_module_data, "dsa_context_module"),
        (load_generation_dsa_module_data, "dsa_generation_module"),
    ],
)
def test_full_only_table_loads_and_skip_variant_is_empty(tmp_path, loader, op_name):
    src = _write_dsa_parquet(tmp_path / f"{op_name}_perf.parquet", [op_name])

    full = loader(src, op_kind="full")
    assert full, "full-indexer rows must load"

    # The skip variant must not raise — it simply yields no rows.
    skip = loader(src, op_kind="skip")
    assert not skip


def test_both_variants_still_split_by_op_name(tmp_path):
    src = _write_dsa_parquet(
        tmp_path / "dsa_context_module_perf.parquet",
        ["dsa_context_module", "dsa_context_module_skip_indexer"],
    )
    assert load_context_dsa_module_data(src, op_kind="full")
    assert load_context_dsa_module_data(src, op_kind="skip")


# ───────────────────────────────────────────────────────────────────────
# The shipped data the degradation exists for
# ───────────────────────────────────────────────────────────────────────


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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loaded-table pins for the GDN cross-backend donor fill.

The perf_data_reuse_manifest regen (f869f616d) made three gdn_perf
kernel_sources cross-backend inheritable (`causal_conv1d_fn`,
`chunk_gated_delta_rule`, `fused_recurrent_gated_delta_rule_packed_decode`).

The pinned values below are the CURRENT truth after (a) the 2026-08 GDN
re-collection (TP-local grids, retimed collector domain — a wholesale
parquet replacement) and (b) first-wins shared-layer merging: a framework's
own persisted rows shield their coordinates from every donor, so
cross-backend donors serve only as gap fill for shapes the own sources do
not cover. Each test pins one donor-served coordinate and one own-row
coordinate on the RAW loaded table (``database._gdn_data`` — the per-call
``_query_gdn_table`` observation retired to the compiled engine with #1357
PR-5; lane-precedence QUERY semantics are anchored by the frozen parity
goldens). If any of these move (manifest change, new data, loader change),
that shift must be a conscious act.
"""

import pytest

from aiconfigurator.sdk.perf_database import get_database
from aiconfigurator_core.sdk.operations.mamba import GDNKernel

pytestmark = pytest.mark.unit


def _gdn_context_lane(db, kernel_source, model_key):
    GDNKernel.load_data(db)
    return db._gdn_data[kernel_source]["context"][model_key]


def test_b200_sglang_gdn_conv_donor_fill_is_pinned():
    db = get_database("b200_sxm", "sglang", "0.5.14")
    key = (2048, 16, 128, 32, 128, 4)  # Qwen3.5-family GDN shard
    lane = _gdn_context_lane(db, "causal_conv1d_fn", key)
    # Donor-served coordinate: the sglang grid stops at the collection token
    # budget, so batch=32 x seq=16384 exists ONLY via the cross-backend fill.
    # The pin makes that graft visible and frozen.
    assert lane[32][16384]["latency"] == pytest.approx(6.3917822265624995, rel=1e-9)
    # Own-row coordinate (sglang's own measurement): first-wins must keep
    # shielding it from every donor.
    assert lane[32][4096]["latency"] == pytest.approx(1.46670654296875, rel=1e-9)
    channels = {record["channel"] for record in db.data_provenance.get("gdn_perf.parquet", []) if record["exists"]}
    assert "cross_backend" in channels  # the donor sources are admitted


def test_h200_vllm_gdn_chunk_own_physical_lane_rows_are_pinned():
    db = get_database("h200_sxm", "vllm", "0.24.0")
    key = (5120, 16, 128, 48, 128, 4)  # Qwen3.5-27B / Nemotron-H-family GDN shard
    # vLLM 0.24's own chunk_gated_delta_rule_flashinfer physical rows: the
    # engine's own-physical-lane precedence serves THESE at query time, not
    # the logical `chunk_gated_delta_rule` lane, which after the shared-layer
    # merge holds sglang/trtllm donor rows and older-version vllm rows.
    phys = _gdn_context_lane(db, "chunk_gated_delta_rule_flashinfer", key)
    assert phys[16][4096]["latency"] == pytest.approx(1.3013623046875, rel=1e-9)
    assert phys[8][4096]["latency"] == pytest.approx(0.640457305908203, rel=1e-9)
    # The logical lane really does cover the same coordinate with a DIFFERENT
    # (donor) value — the precedence question the engine answers is live.
    logical = _gdn_context_lane(db, "chunk_gated_delta_rule", key)
    assert logical[16][4096]["latency"] != pytest.approx(1.3013623046875, rel=1e-9)
    channels = {record["channel"] for record in db.data_provenance.get("gdn_perf.parquet", []) if record["exists"]}
    assert "cross_backend" in channels

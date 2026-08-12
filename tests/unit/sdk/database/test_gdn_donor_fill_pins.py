# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query-level pins for the GDN cross-backend donor fill.

The op_kernel_source_manifest regen (f869f616d) made three gdn_perf
kernel_sources cross-backend inheritable (`causal_conv1d_fn`,
`chunk_gated_delta_rule`, `fused_recurrent_gated_delta_rule_packed_decode`).

The pinned values below are the CURRENT truth after (a) the 2026-08 GDN
re-collection (TP-local grids, retimed collector domain — a wholesale
parquet replacement) and (b) the own-physical-lane precedence rule: a
framework's own persisted kernel rows (e.g. vLLM 0.24's
chunk_gated_delta_rule_*) beat logical-lane rows, so cross-backend donors
serve only as gap fill for shapes the own lanes do not cover. Each test
pins one donor-served coordinate and one own-row coordinate; every query
is an exact grid hit, so perf_interp returns the raw leaf. If any of
these move (manifest change, new data, loader change), that shift must
be a conscious act.
"""

import pytest

from aiconfigurator.sdk.perf_database import get_database
from aiconfigurator_core.sdk.operations.mamba import GDNKernel

pytestmark = pytest.mark.unit


def _query_gdn_context(db, kernel_source, batch, seq, model_key):
    d_model, n_k, k_dim, n_v, v_dim, d_conv = model_key
    return float(
        GDNKernel._query_gdn_table(
            db,
            phase="context",
            kernel_source=kernel_source,
            batch_size=batch,
            seq_len=seq,
            d_model=d_model,
            num_k_heads=n_k,
            head_k_dim=k_dim,
            num_v_heads=n_v,
            head_v_dim=v_dim,
            d_conv=d_conv,
        )
    )


def test_b200_sglang_gdn_conv_donor_fill_is_pinned():
    db = get_database("b200_sxm", "sglang", "0.5.14")
    key = (2048, 16, 128, 32, 128, 4)  # Qwen3.5-family GDN shard
    # Donor-served coordinate: the sglang grid stops at the collection token
    # budget, so batch=32 x seq=16384 exists ONLY via the cross-backend fill.
    # The pin makes that graft visible and frozen.
    assert _query_gdn_context(db, "causal_conv1d_fn", 32, 16384, key) == pytest.approx(6.3917822265624995, rel=1e-9)
    # Own-row coordinate (sglang's own measurement): first-wins must keep
    # shielding it from every donor.
    assert _query_gdn_context(db, "causal_conv1d_fn", 32, 4096, key) == pytest.approx(1.46670654296875, rel=1e-9)
    channels = {record["channel"] for record in db.data_provenance.get("gdn_perf.parquet", []) if record["exists"]}
    assert "cross_backend" in channels  # the donor sources are admitted


def test_h200_vllm_gdn_chunk_own_physical_lane_precedence_is_pinned():
    db = get_database("h200_sxm", "vllm", "0.24.0")
    key = (5120, 16, 128, 48, 128, 4)  # Qwen3.5-27B / Nemotron-H-family GDN shard
    # Both coordinates are served by vLLM's own chunk_gated_delta_rule_flashinfer
    # rows: the physical lane beats the logical `chunk_gated_delta_rule` lane,
    # which after the shared-layer merge holds sglang/trtllm donor rows and
    # older-version vllm rows.
    assert _query_gdn_context(db, "chunk_gated_delta_rule", 16, 4096, key) == pytest.approx(1.3013623046875, rel=1e-9)
    assert _query_gdn_context(db, "chunk_gated_delta_rule", 8, 4096, key) == pytest.approx(0.640457305908203, rel=1e-9)
    channels = {record["channel"] for record in db.data_provenance.get("gdn_perf.parquet", []) if record["exists"]}
    assert "cross_backend" in channels

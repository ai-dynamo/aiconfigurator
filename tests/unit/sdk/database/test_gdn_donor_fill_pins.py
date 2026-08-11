# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query-level pins for the GDN cross-backend donor fill.

The perf_data_reuse_manifest regen (f869f616d) made three gdn_perf
kernel_sources cross-backend inheritable (`causal_conv1d_fn`,
`chunk_gated_delta_rule`, `fused_recurrent_gated_delta_rule_packed_decode`).
Audit result (2026-08-04): donors only EXTEND grids — zero pre-existing
leaf values changed on the audited combos (first-wins keeps own rows) —
but the extensions are real: +48 leaf points on b200_sxm/sglang/0.5.14 and
+43 on h200_sxm/vllm/0.24.0, all context-phase boundary coordinates for
Qwen3.5/Nemotron-H-family shards.

The pinned values below are the CURRENT post-regen truth and INCLUDE
cross-backend donor fill as of f869f616d. Each test pins one donor-only
coordinate (absent before the regen) and one own-row coordinate (proving
first-wins still shields the active backend's measurements). Every query
is an exact grid hit, so perf_interp returns the raw leaf — no
interpolation noise. If any of these move (manifest change, new data,
loader change), that shift must be a conscious act.
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
    # Donor-only coordinate: (batch=32, seq=16384) exists ONLY via the
    # trtllm/vllm cross-backend fill (absent under the pre-regen manifest).
    # NOTE the regime jump vs the own row below — the donor lane's conv is
    # far slower per token; the pin makes that graft visible and frozen.
    assert _query_gdn_context(db, "causal_conv1d_fn", 32, 16384, key) == pytest.approx(11.181324768, rel=1e-9)
    # Own-row coordinate (pre-existing sglang measurement): first-wins must
    # keep shielding it from every donor.
    assert _query_gdn_context(db, "causal_conv1d_fn", 16, 16384, key) == pytest.approx(0.700073624, rel=1e-9)
    channels = {record["channel"] for record in db.data_provenance.get("gdn_perf.parquet", []) if record["exists"]}
    assert "cross_backend" in channels  # the donor sources are admitted


def test_h200_vllm_gdn_chunk_donor_fill_is_pinned():
    db = get_database("h200_sxm", "vllm", "0.24.0")
    key = (5120, 16, 128, 48, 128, 4)  # Nemotron-H-family GDN shard
    # Donor-only coordinate: the whole batch=16 context row for this shard
    # comes from the sglang/trtllm fill (absent under the pre-regen manifest).
    assert _query_gdn_context(db, "chunk_gated_delta_rule", 16, 4096, key) == pytest.approx(6.033555222, rel=1e-9)
    # Own-row coordinate (pre-existing via the vllm same-backend version
    # chain; identical before and after the regen), first-wins-shielded.
    assert _query_gdn_context(db, "chunk_gated_delta_rule", 8, 4096, key) == pytest.approx(2.960513953006629, rel=1e-9)
    channels = {record["channel"] for record in db.data_provenance.get("gdn_perf.parquet", []) if record["exists"]}
    assert "cross_backend" in channels

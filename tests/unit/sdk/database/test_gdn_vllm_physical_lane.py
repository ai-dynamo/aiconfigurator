# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GDN context-scan lane preference: vLLM's own physical rows (AIC-1782 V5b).

The vLLM 0.24.0 and 0.27.1 collectors persist the EXECUTED context-scan
label (``chunk_gated_delta_rule_flashinfer`` on SM100 systems) while the
modeling layer queries the logical ``chunk_gated_delta_rule``; after the
shared-layer merge the logical lane holds cross-backend donor rows
(perf_data_reuse_manifest tier:shared), so ``query_gdn``'s physical-lane
alias walk (``perf_database/state_space.rs``) must serve the own physical
rows first. The AIC-1782 V5a smoke caught vllm/0.27.1 outside the (then
0.24.0-exact) gate: the Qwen3.8-Max context scan priced from sglang 0.5.17
donor rows instead of the landed vllm flashinfer silicon.

Same real-data engine-query pattern as ``test_gdn_flashinfer_lane.py``
(``Operation._engine_query`` — the sanctioned single-op plumbing — plus
raw-table pins in the ``test_gdn_donor_fill_pins.py`` convention). The Rust
unit tests in ``perf_database/state_space.rs``
(``vllm_0271_gdn_own_physical_lane_wins_over_logical_donor_lane``,
``vllm_024_gdn_own_physical_lane_wins_over_logical_lane``,
``gdn_physical_aliases_are_version_gated``) cover the same claims with
synthetic in-memory tables; this file is the real-data-grounded companion.
"""

import pytest

from aiconfigurator.sdk.perf_database import get_database
from aiconfigurator_core.sdk.operations.mamba import GDNKernel

pytestmark = pytest.mark.unit


def _gdn_context_op(kernel_source, model_key):
    d_model, num_k_heads, head_k_dim, num_v_heads, head_v_dim, d_conv = model_key
    return GDNKernel(
        "gdn", 1.0, kernel_source, "context", d_model, num_k_heads, head_k_dim, num_v_heads, head_v_dim, d_conv
    )


def _raw_context_latency(db, kernel_source, model_key, batch, seq_len):
    GDNKernel.load_data(db)
    return db._gdn_data[kernel_source]["context"][model_key][batch][seq_len]["latency"]


def test_query_gdn_vllm_0271_context_scan_prefers_own_flashinfer_rows():
    """The V5b regression on the shipped tables: both lanes cover the
    Qwen3.8-Max tp16 shard with distinct measured values, so a logical-lane
    query resolving to the flashinfer value is a genuine precedence proof."""
    db = get_database("gb300", "vllm", "0.27.1")
    model_key = (8192, 1, 128, 8, 128, 4)  # Qwen3.8-Max GDN tp16 shard

    # Raw-table pins: the donor-served logical lane (sglang 0.5.17 rows via
    # the cross-backend fill — the value the V5a smoke saw priced) and vLLM's
    # own persisted physical lane.
    donor_raw = _raw_context_latency(db, "chunk_gated_delta_rule", model_key, 1, 1024)
    own_raw = _raw_context_latency(db, "chunk_gated_delta_rule_flashinfer", model_key, 1, 1024)
    assert donor_raw == pytest.approx(0.06730688095092774, rel=1e-9)
    assert own_raw == pytest.approx(0.05215519905090332, rel=1e-9)

    # The precedence claim: querying the logical kernel_source must resolve
    # the own physical row, not the donor.
    result = _gdn_context_op("chunk_gated_delta_rule", model_key)._engine_query(db, batch_size=1, s=1024)
    assert result.source == "silicon"
    assert float(result) == pytest.approx(own_raw, rel=1e-9)
    assert float(result) != pytest.approx(donor_raw, rel=1e-9)


def test_query_gdn_vllm_0240_context_scan_still_resolves_own_flashinfer_rows():
    """Version-gate widening must not disturb 0.24.0: the pre-existing alias
    walk keeps serving vLLM 0.24.0's own physical rows (Qwen3.5-397B tp8
    shard), not the cross-backend donor covering the same coordinate."""
    db = get_database("gb300", "vllm", "0.24.0")
    model_key = (5120, 2, 128, 6, 128, 4)  # Qwen3.5-397B GDN tp8 shard

    donor_raw = _raw_context_latency(db, "chunk_gated_delta_rule", model_key, 1, 1024)
    own_raw = _raw_context_latency(db, "chunk_gated_delta_rule_flashinfer", model_key, 1, 1024)
    assert donor_raw == pytest.approx(0.06710527896881104, rel=1e-9)
    assert own_raw == pytest.approx(0.05393343925476074, rel=1e-9)

    result = _gdn_context_op("chunk_gated_delta_rule", model_key)._engine_query(db, batch_size=1, s=1024)
    assert result.source == "silicon"
    assert float(result) == pytest.approx(own_raw, rel=1e-9)
    assert float(result) != pytest.approx(donor_raw, rel=1e-9)

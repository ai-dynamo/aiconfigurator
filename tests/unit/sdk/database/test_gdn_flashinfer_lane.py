# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GDN decode lane preference: FlashInfer bf16-state lane on SM100+ sglang
(AIC-1745, Task 3).

Task 2 (collector) added ``flashinfer_gated_delta_rule_decode`` as a SIBLING
decode row alongside the existing fla/triton fp32-state lane
(``fused_sigmoid_gating_delta_rule_update``) on SM100+ sglang. This module
covers the modeling side: ``GDNKernel._query_gdn_table`` (mamba.py) must
(a) prefer the FlashInfer lane's own rows over the fla lane when both cover
a shape on SM100+ sglang, degrading to the fla lane exactly as before when
the FlashInfer lane is absent, and (b) size the recurrent state at 2 bytes
(bf16) for the FlashInfer lane vs. 4 bytes (fp32) for fla lanes in
``get_sol``.

Fixture style follows ``test_gdn_kernel_alias.py`` (#1503's own-physical-lane
precedence tests) and ``test_gdn_donor_fill_pins.py`` (direct
``GDNKernel._query_gdn_table`` calls against a real system spec).
"""

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import LoadedOpData, get_database
from aiconfigurator_core.sdk.operations.mamba import GDNKernel

pytestmark = pytest.mark.unit

# Same shape convention as test_gdn_kernel_alias.py: (d_model, num_k_heads,
# head_k_dim, num_v_heads, head_v_dim, d_conv).
MODEL_KEY = (2048, 16, 128, 32, 128, 4)
MODEL_SHAPE = {
    "d_model": 2048,
    "num_k_heads": 16,
    "head_k_dim": 128,
    "num_v_heads": 32,
    "head_v_dim": 128,
    "d_conv": 4,
}


def _generation_table(latency):
    return {1: {"latency": latency, "power": 0.0, "energy": 0.0}}


def _query_gdn_generation(db, kernel_source, batch, model_key, d_model=None):
    d, n_k, k_dim, n_v, v_dim, d_conv = model_key
    return float(
        GDNKernel._query_gdn_table(
            db,
            phase="generation",
            kernel_source=kernel_source,
            batch_size=batch,
            seq_len=None,
            d_model=d if d_model is None else d_model,
            num_k_heads=n_k,
            head_k_dim=k_dim,
            num_v_heads=n_v,
            head_v_dim=v_dim,
            d_conv=d_conv,
        )
    )


@pytest.fixture
def sglang_sm100_gdn_db(stub_perf_db):
    stub_perf_db.backend = "sglang"
    stub_perf_db.system_spec["gpu"]["sm_version"] = 103
    return stub_perf_db


# ---------------------------------------------------------------------------
# Precedence: SM100+ sglang decode prefers the FlashInfer lane when present.
# ---------------------------------------------------------------------------


def test_query_gdn_sglang_sm100_prefers_flashinfer_decode_lane(sglang_sm100_gdn_db):
    sglang_sm100_gdn_db._gdn_data = LoadedOpData(
        {
            "fused_sigmoid_gating_delta_rule_update": {"generation": {MODEL_KEY: _generation_table(4.0)}},
            "flashinfer_gated_delta_rule_decode": {"generation": {MODEL_KEY: _generation_table(2.1)}},
        },
        common.PerfDataFilename.gdn,
        "gdn_perf.txt",
    )

    result = sglang_sm100_gdn_db.query_gdn(
        phase="generation",
        kernel_source="fused_sigmoid_gating_delta_rule_update",
        batch_size=1,
        seq_len=None,
        **MODEL_SHAPE,
    )

    # The FlashInfer lane's own row wins, not the fla lane's.
    assert float(result) == pytest.approx(2.1)
    assert result.source == "silicon"


def test_query_gdn_sglang_sm90_keeps_fla_lane_first(sglang_sm100_gdn_db):
    """Hopper unchanged: the fla lane wins even when a flashinfer row exists
    (serving never runs the FlashInfer decode kernel on SM90, so the
    modeling layer must not reach for it there even if one is present)."""
    sglang_sm100_gdn_db.system_spec["gpu"]["sm_version"] = 90
    sglang_sm100_gdn_db._gdn_data = LoadedOpData(
        {
            "fused_sigmoid_gating_delta_rule_update": {"generation": {MODEL_KEY: _generation_table(4.0)}},
            "flashinfer_gated_delta_rule_decode": {"generation": {MODEL_KEY: _generation_table(2.1)}},
        },
        common.PerfDataFilename.gdn,
        "gdn_perf.txt",
    )

    result = sglang_sm100_gdn_db.query_gdn(
        phase="generation",
        kernel_source="fused_sigmoid_gating_delta_rule_update",
        batch_size=1,
        seq_len=None,
        **MODEL_SHAPE,
    )

    assert float(result) == pytest.approx(4.0)
    assert result.source == "silicon"


def test_query_gdn_sglang_sm100_falls_back_to_fla_when_flashinfer_lane_absent(sglang_sm100_gdn_db):
    """Tables collected before AIC-1745's collector change (Task 2) carry no
    flashinfer_gated_delta_rule_decode rows: degrade to the fla lane exactly
    as before, never a hard failure."""
    sglang_sm100_gdn_db._gdn_data = LoadedOpData(
        {"fused_sigmoid_gating_delta_rule_update": {"generation": {MODEL_KEY: _generation_table(4.0)}}},
        common.PerfDataFilename.gdn,
        "gdn_perf.txt",
    )

    result = sglang_sm100_gdn_db.query_gdn(
        phase="generation",
        kernel_source="fused_sigmoid_gating_delta_rule_update",
        batch_size=1,
        seq_len=None,
        **MODEL_SHAPE,
    )

    assert float(result) == pytest.approx(4.0)
    assert result.source == "silicon"


def test_query_gdn_vllm_decode_alias_unaffected_by_sglang_branch(stub_perf_db):
    """The new sglang elif is a sibling of the vllm-0.24.0 branch, not a
    replacement: vllm's own decode alias still resolves independent of
    sm_version (regression guard for the added elif's placement)."""
    stub_perf_db.backend = "vllm"
    stub_perf_db.version = "0.24.0"
    stub_perf_db.system_spec["gpu"]["sm_version"] = 103
    stub_perf_db._gdn_data = LoadedOpData(
        {"fused_recurrent_gated_delta_rule_packed_decode": {"generation": {MODEL_KEY: _generation_table(3.5)}}},
        common.PerfDataFilename.gdn,
        "gdn_perf.txt",
    )

    result = stub_perf_db.query_gdn(
        phase="generation",
        kernel_source="fused_sigmoid_gating_delta_rule_update",
        batch_size=1,
        seq_len=None,
        **MODEL_SHAPE,
    )

    assert float(result) == pytest.approx(3.5)
    assert result.source == "silicon"


# ---------------------------------------------------------------------------
# get_sol: per-lane state bytes (2 for flashinfer, 4 for fla).
# ---------------------------------------------------------------------------


def test_get_sol_flashinfer_lane_state_bytes_ratio_matches_closed_form(stub_perf_db):
    """Pure-SOL fallback (no gdn_data loaded -> get_sol()[0] path): the
    FlashInfer lane's bf16 (2-byte) state must be strictly cheaper than the
    fla lane's fp32 (4-byte) state at the same shape, by exactly the
    closed-form bytes ratio -- the q/k/v activation terms (2-byte bf16) are
    identical for both lanes; only the state read+write terms differ.
    """
    d_model, num_k_heads, head_k_dim, num_v_heads, head_v_dim, d_conv = MODEL_KEY
    batch = 4

    sol_flashinfer = _query_gdn_generation(stub_perf_db, "flashinfer_gated_delta_rule_decode", batch, MODEL_KEY)
    sol_fla = _query_gdn_generation(stub_perf_db, "fused_sigmoid_gating_delta_rule_update", batch, MODEL_KEY)

    assert sol_flashinfer > 0
    assert sol_flashinfer < sol_fla

    activation_bytes = batch * (2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim) * 2  # read
    activation_bytes += batch * num_v_heads * head_v_dim * 2  # write
    state_size = num_v_heads * head_k_dim * head_v_dim
    total_flashinfer = activation_bytes + 2 * (state_size * 2 * batch)  # read + write state terms
    total_fla = activation_bytes + 2 * (state_size * 4 * batch)
    expected_ratio = total_flashinfer / total_fla

    assert sol_flashinfer / sol_fla == pytest.approx(expected_ratio, rel=1e-9)


def test_get_sol_context_scan_state_bytes_unchanged_by_flashinfer_lane(stub_perf_db):
    """The FlashInfer lane is decode-only: chunk_gated_delta_rule (context)
    keeps its 4-byte fp32 state regardless, verified via the closed form
    (a regression guard that the decode-branch edit didn't leak into
    context)."""
    d_model, num_k_heads, head_k_dim, num_v_heads, head_v_dim, d_conv = MODEL_KEY
    batch, seq = 4, 128

    sol_context = float(
        GDNKernel._query_gdn_table(
            stub_perf_db,
            phase="context",
            kernel_source="chunk_gated_delta_rule",
            batch_size=batch,
            seq_len=seq,
            d_model=d_model,
            num_k_heads=num_k_heads,
            head_k_dim=head_k_dim,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            d_conv=d_conv,
        )
    )

    x = batch * seq
    chunk_size = 64
    state_size = num_v_heads * head_k_dim * head_v_dim
    num_chunks = seq // chunk_size
    h_chunks_bytes = num_chunks * state_size * 2 * batch
    read_bytes = (
        x * (2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim) * 2 + state_size * 4 * batch + h_chunks_bytes
    )
    write_bytes = x * num_v_heads * head_v_dim * 2 + state_size * 4 * batch + h_chunks_bytes
    expected_ms = (read_bytes + write_bytes) / stub_perf_db.system_spec["gpu"]["mem_bw"] * 1000

    assert sol_context == pytest.approx(expected_ms, rel=1e-9)


# ---------------------------------------------------------------------------
# Step 3: census anchor -- Qwen3.5-397B dims, tp4 shard, bs=1, gb300 spec.
# ---------------------------------------------------------------------------


def test_flashinfer_lane_sol_census_anchor_qwen35_397b_tp4_gb300():
    """Qwen3.5-397B GDN dims from the model config log (num_k_heads=16,
    head_k_dim=128, num_v_heads=64, head_v_dim=128), sharded tp4 the way
    Qwen35Model._build_generation_ops does (``gdn_nk_per_tp = nk // tp``,
    ``gdn_nv_per_tp = nv // tp`` -- head COUNTS only, head dims unchanged),
    bs=1, real gb300 spec (no flashinfer_gated_delta_rule_decode rows exist
    in the packaged gb300/sglang/0.5.14 table yet -- Task 2 added the
    collector lane but no GPU was available to re-collect -- so this is a
    pure-SOL fallback query, exactly like the collector-blind-spot case
    ``test_query_gdn_sglang_sm100_falls_back_to_fla_when_flashinfer_lane_absent``
    covers above).
    """
    db = get_database("gb300", "sglang", "0.5.14")
    assert db.system_spec["gpu"]["sm_version"] >= 100

    tp = 4
    num_k_heads_full, head_k_dim, num_v_heads_full, head_v_dim = 16, 128, 64, 128
    num_k_heads = num_k_heads_full // tp
    num_v_heads = num_v_heads_full // tp
    batch = 1

    sol_flashinfer = _query_gdn_generation(
        db,
        "flashinfer_gated_delta_rule_decode",
        batch,
        (8192, num_k_heads, head_k_dim, num_v_heads, head_v_dim, 4),
    )
    sol_fla = _query_gdn_generation(
        db,
        "fused_sigmoid_gating_delta_rule_update",
        batch,
        (8192, num_k_heads, head_k_dim, num_v_heads, head_v_dim, 4),
    )

    # Unambiguous, parameter-free properties: the FlashInfer lane is
    # strictly cheaper than the fla lane at this exact shape, and both are
    # a genuine SOL fallback (no real flashinfer_gated_delta_rule_decode
    # silicon exists yet for gb300/sglang/0.5.14).
    assert 0 < sol_flashinfer < sol_fla

    # Closed-form pin at these exact dims/batch/mem_bw (self-consistent with
    # get_sol's formula -- see test_get_sol_flashinfer_lane_state_bytes_ratio
    # _matches_closed_form above for the same derivation technique).
    #
    # NOTE (flagged for human review, see task-3-report.md "concerns"): at
    # the plan's literal parameters (tp4-sharded head COUNTS, bs=1) this
    # closed form evaluates to ~0.132 us, not the brief's stated
    # [1.5, 3.5] us band (census/measured ~2.1 us). Reproducing ~2.1 us
    # from this formula requires either unsharded head counts with batch=4,
    # or sharded head counts with batch=16 (both verified by brute-force
    # search) -- i.e. a (num_v_heads_used * batch) product of 256, not the
    # literal tp4/bs=1 reading's product of 16. The pin below documents the
    # literal-parameter reality rather than silently substituting different
    # parameters to hit the stated band.
    mem_bw = db.system_spec["gpu"]["mem_bw"]
    activation_bytes = batch * (2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim) * 2
    activation_bytes += batch * num_v_heads * head_v_dim * 2
    state_size = num_v_heads * head_k_dim * head_v_dim
    total_flashinfer = activation_bytes + 2 * (state_size * 2 * batch)
    expected_us = total_flashinfer / mem_bw * 1000 * 1000

    assert sol_flashinfer * 1000 == pytest.approx(expected_us, rel=1e-9)

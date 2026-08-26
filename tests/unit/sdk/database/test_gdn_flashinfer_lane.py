# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GDN decode lane preference: FlashInfer bf16-state lane on SM-major-10
sglang (AIC-1745, Task 3).

Task 2 (collector) added ``flashinfer_gated_delta_rule_decode`` as a SIBLING
decode row alongside the existing fla/triton fp32-state lane
(``fused_sigmoid_gating_delta_rule_update``) on SM100+ sglang. This module
covers the modeling side, mirroring serving's exact predicate
(``_handle_linear_attn_backend``, server_args.py:4884-4915 @ pinned v0.5.14:
``is_sm100_supported()`` — capability major EXACTLY 10 — AND
``mamba_ssm_dtype == "bfloat16"``): the compiled engine must
(a) prefer the FlashInfer lane's own rows over the fla lane when both cover
a shape AND the model's SSM state dtype is bfloat16, degrading to the fla
lane exactly as before when the FlashInfer lane is absent, (b) keep the fla
lane for float32-state models (every bundled Qwen3.5/3.6 config pins
``mamba_ssm_dtype: float32``), and (c) size the recurrent state at 2 bytes
(bf16) for the FlashInfer lane vs. 4 bytes (fp32) for fla lanes in the SOL
formula.

Rebase note (post pyo3 op-unification, #1552/#1555/#1566): the Python
per-call query stack (``GDNKernel._query_gdn_table``, ``PerfDatabase.query_gdn``)
is retired — lane selection AND the SOL byte-count formula now live solely in
the Rust engine (``operators/mamba.rs::GdnOp``,
``perf_database/state_space.rs::StateSpaceTable::query_gdn``; see
``.claude/rules/rust-core/parity.md`` Rule 2 — Python may not own `_query_*`/
`get_sol` estimation math). This module now exercises that engine through the
sanctioned ``Operation._engine_query`` shim (see ``test_msa.py``,
``test_deepseek_v4_module.py`` for the same pattern) against REAL on-disk
tables — ``PerfDatabase._gdn_data`` is a raw loaded-table cache the engine
never reads for a query, so a synthetic injected table (the pre-rebase
version of this file) is invisible to it and no longer proves anything about
query-time behavior.

The Rust unit tests in ``operators/mamba.rs`` and
``perf_database/state_space.rs`` (``sglang_sm100_gdn_prefers_flashinfer_decode_lane_for_bf16_state``,
``sglang_sm100_gdn_keeps_fla_lane_for_fp32_state_even_if_flashinfer_present``,
``sglang_sm100_gdn_falls_back_to_fla_lane_when_flashinfer_absent``,
``sglang_sm90_gdn_never_selects_flashinfer_lane_even_if_present``,
``sglang_sm120_gdn_never_selects_flashinfer_lane_regardless_of_dtype``,
``gdn_flashinfer_lane_sol_matches_python_bs128_census_anchor``) cover the
same claims with synthetic in-memory tables (including the strict "SM90/SM120
ignores a present flashinfer row" edge cases real data cannot exercise, since
serving never collects those combinations) — this file is the Python-level,
real-data-grounded companion, not a duplicate.

Fixture style follows ``test_gdn_donor_fill_pins.py`` (direct ``get_database``
+ ``GDNKernel.load_data`` for raw-table pins) and ``test_msa.py``
(``op._engine_query`` for query-time behavior)."""

import pytest

from aiconfigurator.sdk.perf_database import get_database
from aiconfigurator_core.sdk.operations.mamba import GDNKernel

pytestmark = pytest.mark.unit

# Arbitrary synthetic shape for the pure-SOL formula tests below (no data
# lookup involved — see ``database_mode="SOL"``), same convention as
# ``test_gdn_kernel_alias.py``'s successor: (d_model, num_k_heads,
# head_k_dim, num_v_heads, head_v_dim, d_conv).
MODEL_KEY = (2048, 16, 128, 32, 128, 4)


def _gdn_op(kernel_source, phase, model_key, mamba_ssm_dtype="float32"):
    d_model, num_k_heads, head_k_dim, num_v_heads, head_v_dim, d_conv = model_key
    return GDNKernel(
        "gdn",
        1.0,
        kernel_source,
        phase,
        d_model,
        num_k_heads,
        head_k_dim,
        num_v_heads,
        head_v_dim,
        d_conv,
        mamba_ssm_dtype=mamba_ssm_dtype,
    )


def _query_gdn_generation(db, kernel_source, model_key, batch, d_model=None, mamba_ssm_dtype="float32"):
    """Query the GDN op's generation phase through the compiled engine
    (``Operation._engine_query`` — the permanent internal single-op plumbing,
    see ``test_msa.py``). Returns ``(latency_ms, source)``."""
    key = model_key if d_model is None else (d_model, *model_key[1:])
    op = _gdn_op(kernel_source, "generation", key, mamba_ssm_dtype=mamba_ssm_dtype)
    result = op._engine_query(db, batch_size=batch, s=1)
    return float(result), result.source


def _query_gdn_context(db, kernel_source, model_key, batch, seq_len, d_model=None):
    key = model_key if d_model is None else (d_model, *model_key[1:])
    op = _gdn_op(kernel_source, "context", key)
    result = op._engine_query(db, batch_size=batch, s=seq_len)
    return float(result), result.source


# ---------------------------------------------------------------------------
# Precedence: SM-major-10 sglang decode prefers the FlashInfer lane when
# present AND the model's SSM state dtype is bfloat16 (serving's exact
# predicate). Real on-disk tables (AIC-1745's gb300/gb200/b200_sxm
# sglang/0.5.14 rows): both lanes cover the Qwen3.5-397B tp4 shard shape,
# with distinct measured values, so a query resolving to the FlashInfer
# value (not the fla value) is a genuine query-time proof, not a fixture
# artifact.
# ---------------------------------------------------------------------------


def test_query_gdn_sglang_sm100_prefers_flashinfer_decode_lane_for_bf16_state():
    db = get_database("gb300", "sglang", "0.5.14")
    assert db.system_spec["gpu"]["sm_version"] >= 100
    model_key = (4096, 4, 128, 16, 128, 4)  # Qwen3.5-397B GDN tp4 shard

    # Raw-table pin (test_gdn_donor_fill_pins.py convention): the fla lane's
    # OWN row at this shape, read directly off the loaded table. A QUERY for
    # this kernel_source is exactly what resolves to the FlashInfer value
    # below instead — the raw table is the only way to see this number.
    GDNKernel.load_data(db)
    fla_raw = db._gdn_data["fused_sigmoid_gating_delta_rule_update"]["generation"][model_key][1]["latency"]
    assert fla_raw == pytest.approx(0.004236159920692444, rel=1e-9)

    # Own-lane query pin: the FlashInfer kernel_source resolves its own row.
    fi_latency, fi_source = _query_gdn_generation(db, "flashinfer_gated_delta_rule_decode", model_key, batch=1)
    assert fi_source == "silicon"
    assert fi_latency == pytest.approx(0.0023209600150585173, rel=1e-9)
    assert fi_latency < fla_raw  # bf16 (2-byte) state genuinely cheaper than fp32 (4-byte)

    # The precedence claim: querying the FLA kernel_source on SM-major-10
    # sglang with a bf16-state model must resolve the FlashInfer lane's row,
    # not its own raw value.
    resolved_latency, resolved_source = _query_gdn_generation(
        db, "fused_sigmoid_gating_delta_rule_update", model_key, batch=1, mamba_ssm_dtype="bfloat16"
    )
    assert resolved_source == "silicon"
    assert resolved_latency == pytest.approx(fi_latency, rel=1e-9)
    assert resolved_latency != pytest.approx(fla_raw, rel=1e-9)


def test_query_gdn_sglang_sm100_keeps_fla_lane_for_fp32_state():
    """Every bundled Qwen3.5/3.6 config pins mamba_ssm_dtype=float32, and
    serving auto-selects the FlashInfer decode backend only for bfloat16
    state (server_args.py:4884-4915 @0.5.14) — so the DEFAULT query must
    resolve the fla lane's own row even though the FlashInfer sibling row
    covers the same shape in this real table."""
    db = get_database("gb300", "sglang", "0.5.14")
    assert db.system_spec["gpu"]["sm_version"] >= 100
    model_key = (4096, 4, 128, 16, 128, 4)  # Qwen3.5-397B GDN tp4 shard

    GDNKernel.load_data(db)
    fla_raw = db._gdn_data["fused_sigmoid_gating_delta_rule_update"]["generation"][model_key][1]["latency"]

    resolved_latency, resolved_source = _query_gdn_generation(
        db, "fused_sigmoid_gating_delta_rule_update", model_key, batch=1
    )
    assert resolved_source == "silicon"
    assert resolved_latency == pytest.approx(fla_raw, rel=1e-9)
    assert resolved_latency == pytest.approx(0.004236159920692444, rel=1e-9)


def test_query_gdn_sglang_sm90_keeps_fla_lane_first():
    """Hopper unchanged: serving never runs the FlashInfer decode kernel on
    SM90, so real SM90 sglang tables carry no flashinfer_gated_delta_rule_decode
    rows at all and the query resolves the fla lane directly (the strict
    "even if a flashinfer row were present" edge is real-data-unreachable;
    the Rust test ``sglang_sm90_gdn_never_selects_flashinfer_lane_even_if_present``
    covers it with a synthetic table)."""
    db = get_database("h200_sxm", "sglang", "0.5.14")
    assert db.system_spec["gpu"]["sm_version"] < 100
    model_key = (1024, 4, 128, 4, 128, 4)

    GDNKernel.load_data(db)
    assert "flashinfer_gated_delta_rule_decode" not in db._gdn_data  # genuinely absent on SM90

    latency, source = _query_gdn_generation(db, "fused_sigmoid_gating_delta_rule_update", model_key, batch=1)
    assert source == "silicon"
    assert latency > 0


def test_query_gdn_sglang_sm100_falls_back_to_fla_when_flashinfer_lane_absent():
    """Tables collected before AIC-1745's collector change carry no
    flashinfer_gated_delta_rule_decode rows: degrade to the fla lane exactly
    as before, never a hard failure. b200_sxm/sglang/0.5.10 is real,
    genuinely SM100+, and genuinely predates the collector change (unlike
    0.5.14/0.5.16, which now carry the FlashInfer lane on every SM100+
    system checked)."""
    db = get_database("b200_sxm", "sglang", "0.5.10")
    assert db.system_spec["gpu"]["sm_version"] >= 100
    model_key = (1024, 4, 128, 4, 128, 4)

    GDNKernel.load_data(db)
    assert "flashinfer_gated_delta_rule_decode" not in db._gdn_data  # pre-AIC-1745 collector data

    # bf16 state so the alias branch actually fires (fp32 would bypass it);
    # the empty alias slot must degrade to the fla lane, not hard-fail.
    latency, source = _query_gdn_generation(
        db, "fused_sigmoid_gating_delta_rule_update", model_key, batch=1, mamba_ssm_dtype="bfloat16"
    )
    assert source == "silicon"
    assert latency > 0


def test_query_gdn_vllm_generation_unaffected_by_sglang_branch_even_at_sm100():
    """The new sglang elif is a sibling of the vllm-0.24.0 branch, not a
    replacement: vllm's own generation row still resolves independent of
    sm_version (regression guard for the added elif's placement). gb300 is
    sm_version=103 — AT the exact SM100+ threshold that gates the new sglang
    branch — deliberately chosen so the only thing keeping the sglang elif
    from firing is the ``backend == "sglang"`` check, not sm_version."""
    db = get_database("gb300", "vllm", "0.24.0")
    assert db.system_spec["gpu"]["sm_version"] >= 100
    model_key = (1024, 2, 128, 2, 128, 4)

    latency, source = _query_gdn_generation(db, "fused_sigmoid_gating_delta_rule_update", model_key, batch=1)
    assert source == "silicon"
    assert latency == pytest.approx(0.004360319972038269, rel=1e-9)


# ---------------------------------------------------------------------------
# get_sol: per-lane state bytes (2 for flashinfer, 4 for fla). Pure formula —
# an off-key synthetic d_model (8192, same convention as the census anchors
# below) keeps the query off every real collected row regardless of system,
# guaranteeing the pure-SOL fallback path (real gb300/sglang/0.5.14 spec for
# mem_bw etc., matching the census anchors' technique). MODEL_KEY's real
# (num_k_heads, head_k_dim, num_v_heads, head_v_dim, d_conv) DO collide with
# real gb300 rows at its own d_model=2048 -- confirmed empirically -- so the
# override is required, not optional.
# ---------------------------------------------------------------------------


def test_get_sol_flashinfer_lane_state_bytes_ratio_matches_closed_form():
    """The FlashInfer lane's bf16 (2-byte) state must be strictly cheaper
    than the fla lane's fp32 (4-byte) state at the same shape, by exactly
    the closed-form bytes ratio -- the q/k/v activation terms (2-byte bf16)
    are identical for both lanes; only the state read+write terms differ."""
    db = get_database("gb300", "sglang", "0.5.14")
    _, num_k_heads, head_k_dim, num_v_heads, head_v_dim, d_conv = MODEL_KEY
    batch = 4

    sol_flashinfer, src_fi = _query_gdn_generation(
        db, "flashinfer_gated_delta_rule_decode", MODEL_KEY, batch, d_model=8192
    )
    sol_fla, src_fla = _query_gdn_generation(
        db, "fused_sigmoid_gating_delta_rule_update", MODEL_KEY, batch, d_model=8192
    )
    assert src_fi == "sol"
    assert src_fla == "sol"

    assert sol_flashinfer > 0
    assert sol_flashinfer < sol_fla

    activation_bytes = batch * (2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim) * 2  # read
    activation_bytes += batch * num_v_heads * head_v_dim * 2  # write
    state_size = num_v_heads * head_k_dim * head_v_dim
    total_flashinfer = activation_bytes + 2 * (state_size * 2 * batch)  # read + write state terms
    total_fla = activation_bytes + 2 * (state_size * 4 * batch)
    expected_ratio = total_flashinfer / total_fla

    assert sol_flashinfer / sol_fla == pytest.approx(expected_ratio, rel=1e-9)


def test_get_sol_context_scan_state_bytes_unchanged_by_flashinfer_lane():
    """The FlashInfer lane is decode-only: chunk_gated_delta_rule (context)
    keeps its 4-byte fp32 state regardless, verified via the closed form
    (a regression guard that the decode-branch edit didn't leak into
    context)."""
    db = get_database("gb300", "sglang", "0.5.14")
    _, num_k_heads, head_k_dim, num_v_heads, head_v_dim, d_conv = MODEL_KEY
    batch, seq = 4, 128

    sol_context, source = _query_gdn_context(db, "chunk_gated_delta_rule", MODEL_KEY, batch, seq, d_model=8192)
    assert source == "sol"

    x = batch * seq
    chunk_size = 64
    state_size = num_v_heads * head_k_dim * head_v_dim
    num_chunks = seq // chunk_size
    h_chunks_bytes = num_chunks * state_size * 2 * batch
    read_bytes = (
        x * (2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim) * 2 + state_size * 4 * batch + h_chunks_bytes
    )
    write_bytes = x * num_v_heads * head_v_dim * 2 + state_size * 4 * batch + h_chunks_bytes
    expected_ms = (read_bytes + write_bytes) / db.system_spec["gpu"]["mem_bw"] * 1000

    assert sol_context == pytest.approx(expected_ms, rel=1e-9)


# ---------------------------------------------------------------------------
# Census anchor -- Qwen3.5-397B dims, tp4 shard, real gb300 spec. The 8192
# d_model below is a deliberate synthetic stand-in (397B's real value is
# 4096): flashinfer decode silicon rows are keyed at d_model=4096 in the
# packaged gb300/sglang/0.5.14 table, so the off-key d_model keeps the query
# off those shipped rows and pins the pure-SOL closed form itself.
# ---------------------------------------------------------------------------


def test_flashinfer_lane_sol_census_anchor_qwen35_397b_tp4_gb300():
    """Qwen3.5-397B GDN dims from the model config log (num_k_heads=16,
    head_k_dim=128, num_v_heads=64, head_v_dim=128), sharded tp4 the way
    Qwen35Model._build_generation_ops does (``gdn_nk_per_tp = nk // tp``,
    ``gdn_nv_per_tp = nv // tp`` -- head COUNTS only, head dims unchanged),
    bs=1, real gb300 spec."""
    db = get_database("gb300", "sglang", "0.5.14")
    assert db.system_spec["gpu"]["sm_version"] >= 100

    tp = 4
    num_k_heads_full, head_k_dim, num_v_heads_full, head_v_dim = 16, 128, 64, 128
    num_k_heads = num_k_heads_full // tp
    num_v_heads = num_v_heads_full // tp
    batch = 1
    model_key = (8192, num_k_heads, head_k_dim, num_v_heads, head_v_dim, 4)

    sol_flashinfer, src_fi = _query_gdn_generation(db, "flashinfer_gated_delta_rule_decode", model_key, batch)
    sol_fla, _ = _query_gdn_generation(db, "fused_sigmoid_gating_delta_rule_update", model_key, batch)

    # Unambiguous, parameter-free properties: the FlashInfer lane is
    # strictly cheaper than the fla lane at this exact shape, and both are
    # a pure-SOL fallback by construction: the off-key d_model=8192 above
    # keeps the query away from the real d_model=4096 silicon rows.
    assert src_fi == "sol"
    assert 0 < sol_flashinfer < sol_fla

    # Closed-form pin at these exact dims/batch/mem_bw (self-consistent with
    # the SOL formula -- see
    # test_get_sol_flashinfer_lane_state_bytes_ratio_matches_closed_form
    # above for the same derivation technique).
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


def test_flashinfer_lane_sol_matches_census_at_bs128():
    """Census anchor (L3 audit, gb300): measured flashinfer GDN decode at bs=128 is
    ~20.9 us/layer; the bf16-state SOL must land at ~80% of that. Guards the
    state-bytes term at a batch size where the kernel is genuinely memory-bound
    (bs=1 is launch-floor territory where SOL is far below measured). Rust
    twin: ``operators::mamba::tests::gdn_flashinfer_lane_sol_matches_python_bs128_census_anchor``.
    """
    db = get_database("gb300", "sglang", "0.5.14")
    assert db.system_spec["gpu"]["sm_version"] >= 100

    tp = 4
    num_k_heads_full, head_k_dim, num_v_heads_full, head_v_dim = 16, 128, 64, 128
    num_k_heads = num_k_heads_full // tp
    num_v_heads = num_v_heads_full // tp
    batch = 128
    # Off-key synthetic d_model (see the bs=1 census anchor above); keeps
    # this query on the pure-SOL path regardless of table rows.
    model_key = (8192, num_k_heads, head_k_dim, num_v_heads, head_v_dim, 4)

    sol_flashinfer_ms, source = _query_gdn_generation(db, "flashinfer_gated_delta_rule_decode", model_key, batch)
    assert source == "sol"

    sol_us = sol_flashinfer_ms * 1000
    assert 12.0 <= sol_us <= 22.0

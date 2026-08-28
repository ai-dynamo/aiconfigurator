# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GDN dtype routing and SOL contracts.

Packaged GDN rows have no state-dtype key and were collected with FP32 state
for state-sensitive kernels. Repository collection therefore rejects
non-FP32 cases. The compiled consumer may reuse causal-convolution rows for
any dtype, but uses state-sensitive empirical rows only for FP32. Its sole
exception mirrors SGLang serving's exact predicate: on compute-capability
major 10 with BF16 state, a logical generation recurrence may use an exact
FlashInfer physical alias. If that alias is absent, the query falls to
dtype-aware SOL instead of using the untyped FLA row.

Rebase note (post pyo3 op-unification, #1552/#1555/#1566): lane selection and
the SOL byte-count formula now live solely in the Rust engine
(``operators/mamba.rs::GdnOp``,
``perf_database/state_space.rs::StateSpaceTable::query_gdn``). This module
exercises real on-disk table absence/default routing and pure-SOL formulas
through the sanctioned ``Operation._engine_query`` shim. Hypothetical BF16
FlashInfer presence uses explicit Rust in-memory fixtures; no packaged table
is claimed to contain such a row.

The Rust unit tests in ``operators/mamba.rs`` and
``perf_database/state_space.rs`` (``sglang_sm100_gdn_prefers_flashinfer_decode_lane_for_bf16_state``,
``sglang_sm100_gdn_keeps_fla_lane_for_fp32_state_even_if_flashinfer_present``,
``sglang_sm100_gdn_misses_when_required_flashinfer_alias_is_absent``,
``sglang_sm90_gdn_uses_fla_rows_only_for_fp32_state``,
``sglang_sm120_gdn_uses_fla_rows_only_for_fp32_state``,
``gdn_flashinfer_lane_sol_matches_python_bs128_census_anchor``) cover the
routing claims with explicit synthetic in-memory tables. This file is the
Python-level real-data/SOL companion, not a duplicate.

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


def _query_gdn_context(
    db,
    kernel_source,
    model_key,
    batch,
    seq_len,
    d_model=None,
    mamba_ssm_dtype="float32",
):
    key = model_key if d_model is None else (d_model, *model_key[1:])
    op = _gdn_op(kernel_source, "context", key, mamba_ssm_dtype=mamba_ssm_dtype)
    result = op._engine_query(db, batch_size=batch, s=seq_len)
    return float(result), result.source


# ---------------------------------------------------------------------------
# Current packaged tables contain no non-serving FlashInfer rows. Hypothetical
# bf16 routing is covered by explicit in-memory fixtures in state_space.rs.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("system", ("b200_sxm", "b300_sxm", "gb200", "gb300"))
def test_current_sglang_0514_tables_have_no_flashinfer_decode_rows(system):
    db = get_database(system, "sglang", "0.5.14")
    assert 100 <= db.system_spec["gpu"]["sm_version"] < 110

    GDNKernel.load_data(db)

    assert "flashinfer_gated_delta_rule_decode" not in db._gdn_data


def test_query_gdn_sglang_sm100_keeps_fla_lane_for_fp32_state():
    """Every bundled Qwen3.5/3.6 config pins mamba_ssm_dtype=float32, and
    serving auto-selects the FlashInfer decode backend only for bfloat16
    state (server_args.py:4884-4915 @0.5.14) — so the default query resolves
    the fla lane's own row in the current honest table."""
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


def test_query_gdn_sglang_sm100_missing_bf16_flashinfer_alias_uses_sol():
    db = get_database("gb300", "sglang", "0.5.14")
    assert db.system_spec["gpu"]["sm_version"] == 103
    model_key = (4096, 4, 128, 16, 128, 4)

    GDNKernel.load_data(db)
    assert "flashinfer_gated_delta_rule_decode" not in db._gdn_data

    latency, source = _query_gdn_generation(
        db,
        "fused_sigmoid_gating_delta_rule_update",
        model_key,
        batch=1,
        mamba_ssm_dtype="bfloat16",
    )

    assert source == "sol"
    assert latency > 0


@pytest.mark.parametrize(
    ("system", "expected_sm"),
    [("h200_sxm", 90), ("rtx_pro_6000_server", 120)],
)
def test_query_gdn_non_sm10_fla_rows_require_fp32_state(system, expected_sm):
    db = get_database(system, "sglang", "0.5.14")
    assert db.system_spec["gpu"]["sm_version"] == expected_sm
    model_key = (1024, 4, 128, 4, 128, 4)

    GDNKernel.load_data(db)
    assert "flashinfer_gated_delta_rule_decode" not in db._gdn_data

    fp32_latency, fp32_source = _query_gdn_generation(
        db,
        "fused_sigmoid_gating_delta_rule_update",
        model_key,
        batch=1,
        mamba_ssm_dtype="float32",
    )
    assert fp32_source == "silicon"
    assert fp32_latency > 0

    for dtype in ("bfloat16", "float16"):
        latency, source = _query_gdn_generation(
            db,
            "fused_sigmoid_gating_delta_rule_update",
            model_key,
            batch=1,
            mamba_ssm_dtype=dtype,
        )
        assert source == "sol"
        assert latency > 0


def test_query_gdn_causal_conv_rows_are_available_for_bf16_state():
    db = get_database("h200_sxm", "sglang", "0.5.14")
    model_key = (1024, 4, 128, 4, 128, 4)

    context_latency, context_source = _query_gdn_context(
        db,
        "causal_conv1d_fn",
        model_key,
        batch=1,
        seq_len=128,
        mamba_ssm_dtype="bfloat16",
    )
    generation_latency, generation_source = _query_gdn_generation(
        db,
        "causal_conv1d_update",
        model_key,
        batch=1,
        mamba_ssm_dtype="bfloat16",
    )

    assert context_source == generation_source == "silicon"
    assert context_latency > 0
    assert generation_latency > 0


def test_query_gdn_vllm_generation_unaffected_by_sglang_major10_branch():
    """The sglang branch is parallel to the vllm-0.24.0 branch, not a
    replacement: vllm's own generation row still resolves independent of
    sm_version (regression guard for the added elif's placement). gb300 is
    sm_version=103, inside compute-capability major 10; the only thing keeping
    the sglang branch from firing is the ``backend == "sglang"`` check, not
    sm_version."""
    db = get_database("gb300", "vllm", "0.24.0")
    assert db.system_spec["gpu"]["sm_version"] >= 100
    model_key = (1024, 2, 128, 2, 128, 4)

    latency, source = _query_gdn_generation(db, "fused_sigmoid_gating_delta_rule_update", model_key, batch=1)
    assert source == "silicon"
    assert latency > 0


# ---------------------------------------------------------------------------
# get_sol: state bytes follow mamba_ssm_dtype for context/FLA; explicit
# FlashInfer remains two bytes. Pure formula —
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
        db,
        "flashinfer_gated_delta_rule_decode",
        MODEL_KEY,
        batch,
        d_model=8192,
        mamba_ssm_dtype="bfloat16",
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


def test_get_sol_fla_recurrence_state_bytes_follow_model_dtype():
    db = get_database("gb300", "sglang", "0.5.14")
    _, num_k_heads, head_k_dim, num_v_heads, head_v_dim, d_conv = MODEL_KEY
    batch = 4

    results = {
        dtype: _query_gdn_generation(
            db,
            "fused_sigmoid_gating_delta_rule_update",
            MODEL_KEY,
            batch,
            d_model=8192,
            mamba_ssm_dtype=dtype,
        )
        for dtype in ("float32", "bfloat16", "float16")
    }
    assert {source for _, source in results.values()} == {"sol"}

    activation_bytes = batch * (2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim) * 2
    activation_bytes += batch * num_v_heads * head_v_dim * 2
    state_size = num_v_heads * head_k_dim * head_v_dim
    total_fp32 = activation_bytes + 2 * state_size * 4 * batch
    total_16bit = activation_bytes + 2 * state_size * 2 * batch
    sol_fp32 = results["float32"][0]
    sol_bf16 = results["bfloat16"][0]
    sol_fp16 = results["float16"][0]

    assert sol_bf16 == pytest.approx(sol_fp16, rel=1e-9)
    assert sol_bf16 / sol_fp32 == pytest.approx(total_16bit / total_fp32, rel=1e-9)


def test_get_sol_context_scan_state_bytes_follow_model_dtype():
    """Context scan SOL uses the configured recurrent-state width: two bytes
    for BF16 and four for FP32, with identical BF16 activation/chunk terms."""
    db = get_database("gb300", "sglang", "0.5.14")
    _, num_k_heads, head_k_dim, num_v_heads, head_v_dim, d_conv = MODEL_KEY
    batch, seq = 4, 128

    sol_fp32, source_fp32 = _query_gdn_context(
        db,
        "chunk_gated_delta_rule",
        MODEL_KEY,
        batch,
        seq,
        d_model=8192,
        mamba_ssm_dtype="float32",
    )
    sol_bf16, source_bf16 = _query_gdn_context(
        db,
        "chunk_gated_delta_rule",
        MODEL_KEY,
        batch,
        seq,
        d_model=8192,
        mamba_ssm_dtype="bfloat16",
    )
    assert source_fp32 == source_bf16 == "sol"

    x = batch * seq
    chunk_size = 64
    state_size = num_v_heads * head_k_dim * head_v_dim
    num_chunks = seq // chunk_size
    h_chunks_bytes = num_chunks * state_size * 2 * batch
    activation_and_chunks = x * (2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim) * 2
    activation_and_chunks += x * num_v_heads * head_v_dim * 2 + 2 * h_chunks_bytes
    total_fp32 = activation_and_chunks + 2 * state_size * 4 * batch
    total_bf16 = activation_and_chunks + 2 * state_size * 2 * batch

    assert sol_fp32 == pytest.approx(total_fp32 / db.system_spec["gpu"]["mem_bw"] * 1000, rel=1e-9)
    assert sol_bf16 == pytest.approx(total_bf16 / db.system_spec["gpu"]["mem_bw"] * 1000, rel=1e-9)
    assert sol_bf16 / sol_fp32 == pytest.approx(total_bf16 / total_fp32, rel=1e-9)


# ---------------------------------------------------------------------------
# Census anchor -- Qwen3.5-397B dims, tp4 shard, real gb300 spec. The 8192
# d_model below is a deliberate synthetic stand-in (397B's real value is
# 4096). The current table contains no FlashInfer decode rows; the off-key
# d_model also keeps this a pure-SOL check if serving-true data is added later.
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

    sol_flashinfer, src_fi = _query_gdn_generation(
        db,
        "flashinfer_gated_delta_rule_decode",
        model_key,
        batch,
        mamba_ssm_dtype="bfloat16",
    )
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

    sol_flashinfer_ms, source = _query_gdn_generation(
        db,
        "flashinfer_gated_delta_rule_decode",
        model_key,
        batch,
        mamba_ssm_dtype="bfloat16",
    )
    assert source == "sol"

    sol_us = sol_flashinfer_ms * 1000
    assert 12.0 <= sol_us <= 22.0

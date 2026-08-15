# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L1 query-equivalence gate: ``query_moe_expert_compute`` vs the legacy compute tables.

Shipped-data sweeps on real databases. The legacy facades this file compared
against (``query_moe`` with ``moe_backend="deepep_moe"`` and
``query_wideep_moe_compute``'s own Python walk) retired with #1357 PR-5 —
the sglang deepep_moe compute path outright (AIC-1601) — so the legacy side
of each comparison is the RAW loaded table row itself, probed at EXACT
collected token points (min / median / max per slice; off-grid interpolation
and the beyond-range util-hold — including the is_gated num_gemms SOL shape
— retired to the compiled engine and are anchored by the frozen parity
goldens). The engine-routed ``query_moe_expert_compute`` must reproduce the
raw rows at rel <= 1e-9:

- sglang h200_sxm 0.5.6.post2: every ``wideep_context_moe`` /
  ``wideep_generation_moe`` slice under the matching phase.
- trtllm gb200 1.3.0rc10: every ``wideep_moe`` slice (all num_slots
  {256, 288, 384}, all distributions incl. ``_eplb``). The legacy table has
  no phase split, so BOTH unified phases must return bit-identical values.
  NO carve-outs.

Both sweeps also probe an UNCOLLECTED distribution (``"power_law"`` — the
production default request string, absent from both shipped tables) once per
slice: on sglang this pins the ->"uniform" fallback (raw uniform row as the
expectation); on gb200 (whose table has no "uniform") it pins the
->first-available fallback (raw first-collected-distribution row).
"""

import itertools
import os
from pathlib import Path

import pytest

from aiconfigurator_core.sdk.operations.base import resolve_op_data_path
from aiconfigurator_core.sdk.operations.moe import (
    load_wideep_context_moe_data,
    load_wideep_generation_moe_data,
    load_wideep_moe_compute_data,
)
from aiconfigurator_core.sdk.perf_database import get_database

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SYSTEMS_DATA_ROOT = REPO_ROOT / "aic-core" / "src" / "aiconfigurator_core" / "systems" / "data"

SGLANG_CONTEXT_PATH = resolve_op_data_path(
    str(SYSTEMS_DATA_ROOT / "h200_sxm"), "sglang", "0.5.6.post2", "wideep_context_moe_perf.parquet"
)
SGLANG_GENERATION_PATH = resolve_op_data_path(
    str(SYSTEMS_DATA_ROOT / "h200_sxm"), "sglang", "0.5.6.post2", "wideep_generation_moe_perf.parquet"
)
TRTLLM_WIDEEP_PATH = resolve_op_data_path(
    str(SYSTEMS_DATA_ROOT / "gb200"), "trtllm", "1.3.0rc10", "wideep_moe_perf.parquet"
)

REL_TOL = 1e-9

# Uncollected in both shipped tables (asserted in each sweep): exercises the
# distribution-fallback path on real data.
UNCOLLECTED_DIST = "power_law"


def _iter_slices(nested, depth):
    """Yield ``(key_tuple, node)`` for every node ``depth`` dict levels down."""
    if depth == 0:
        yield (), nested
        return
    for key, sub in nested.items():
        for rest, node in _iter_slices(sub, depth - 1):
            yield (key, *rest), node


def _exact_token_probes(token_keys):
    """Three exact collected points per slice: min / median / max."""
    keys = sorted(token_keys)
    return sorted({keys[0], keys[len(keys) // 2], keys[-1]})


# ---------------------------------------------------------------------------
# (a) sglang h200_sxm 0.5.6.post2 — WideEP context + generation MoE tables
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (os.path.exists(SGLANG_CONTEXT_PATH) and os.path.exists(SGLANG_GENERATION_PATH)),
    reason="shipped h200_sxm sglang 0.5.6.post2 wideep MoE parquets not present",
)
def test_l1_sglang_wideep_moe_query_equivalence():
    db = get_database("h200_sxm", "sglang", "0.5.6.post2")
    assert db is not None

    # Legacy tables: [quant][dist][topk][experts][hidden][inter][tp][ep] -> {tokens: leaf (ms)}.
    for phase, path, loader in (
        ("context", SGLANG_CONTEXT_PATH, load_wideep_context_moe_data),
        ("generation", SGLANG_GENERATION_PATH, load_wideep_generation_moe_data),
    ):
        legacy_table = loader(path)
        assert legacy_table
        assert all(UNCOLLECTED_DIST not in legacy_table[quant] for quant in legacy_table)
        comparisons = 0
        for (quant, dist, topk, experts, hidden, inter, tp, ep), tokens in _iter_slices(legacy_table, 8):
            for tok in _exact_token_probes(tokens):
                # num_slots = num_experts: the legacy sglang tables have no
                # EPLB redundancy axis (spec §4.2 adapter pin).
                unified = db.query_moe_expert_compute(
                    "deepep_moe", quant, dist, phase, topk, experts, experts, hidden, inter, tp, ep, tok
                )
                context = f"sglang wideep {phase} {quant.name} {dist} {topk=} {experts=} {ep=} {tok=}"
                assert float(unified) == pytest.approx(tokens[tok]["latency"], rel=REL_TOL), context
                assert unified.energy == pytest.approx(tokens[tok]["energy"], rel=REL_TOL), context
                comparisons += 1
            # One uncollected probe pinning the fallback path (legacy resolved
            # "power_law" -> "uniform"; expectation = the raw uniform row).
            uniform_slice = (
                legacy_table[quant].get("uniform", {}).get(topk, {}).get(experts, {}).get(hidden, {}).get(inter, {})
            )
            uniform_tokens = uniform_slice.get(tp, {}).get(ep, {})
            tok = min(tokens)
            if tok in uniform_tokens:
                fallback = db.query_moe_expert_compute(
                    "deepep_moe", quant, UNCOLLECTED_DIST, phase, topk, experts, experts, hidden, inter, tp, ep, tok
                )
                assert float(fallback) == pytest.approx(uniform_tokens[tok]["latency"], rel=REL_TOL)
                comparisons += 1
        assert comparisons > 50, f"sglang wideep {phase} sweep too small: {comparisons}"


# ---------------------------------------------------------------------------
# (b) trtllm gb200 1.3.0rc10 — WideEP MoE compute table
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.path.exists(TRTLLM_WIDEEP_PATH),
    reason="shipped gb200 trtllm 1.3.0rc10 wideep_moe parquet not present",
)
def test_l1_trtllm_wideep_moe_compute_query_equivalence():
    db = get_database("gb200", "trtllm", "1.3.0rc10")
    assert db is not None

    # Legacy table: [kernel][quant][dist][topk][experts][hidden][inter][slots][tp][ep] -> {tokens: leaf (ms)}.
    legacy_table = load_wideep_moe_compute_data(TRTLLM_WIDEEP_PATH)
    assert legacy_table

    assert all(
        UNCOLLECTED_DIST not in legacy_table[kernel][quant] and "uniform" not in legacy_table[kernel][quant]
        for kernel in legacy_table
        for quant in legacy_table[kernel]
    )

    comparisons = 0
    slots_seen: set[int] = set()
    dists_seen: set[str] = set()
    for (kernel, quant, dist, topk, experts, hidden, inter, slots, tp, ep), tokens in _iter_slices(legacy_table, 10):
        slots_seen.add(slots)
        dists_seen.add(dist)
        probes = _exact_token_probes(tokens)
        for i, tok in enumerate(probes):
            context = f"trtllm wideep {kernel} {quant.name} {dist} {slots=} {experts=} {hidden=} {ep=} {tok=}"
            unified_ctx = db.query_moe_expert_compute(
                kernel, quant, dist, "context", topk, experts, slots, hidden, inter, tp, ep, tok
            )
            assert float(unified_ctx) == pytest.approx(tokens[tok]["latency"], rel=REL_TOL), context
            if i == 0:
                # The legacy table has no context/generation split: both unified
                # phases carry the same rows and must return identical values.
                unified_gen = db.query_moe_expert_compute(
                    kernel, quant, dist, "generation", topk, experts, slots, hidden, inter, tp, ep, tok
                )
                assert float(unified_ctx) == float(unified_gen), context
                assert unified_ctx.energy == unified_gen.energy, context
            comparisons += 1
        # One uncollected probe pinning the fallback path: with no "uniform"
        # collected, the unified query answers with the FIRST available
        # distribution in table insertion order.
        first_dist = next(iter(legacy_table[kernel][quant]))
        # .get() chain: the table is a nested defaultdict and _iter_slices holds
        # live iterators over these levels — direct indexing on a missing key
        # would vivify entries mid-iteration.
        first_tokens = (
            legacy_table[kernel][quant]
            .get(first_dist, {})
            .get(topk, {})
            .get(experts, {})
            .get(hidden, {})
            .get(inter, {})
            .get(slots, {})
            .get(tp, {})
            .get(ep, {})
        )
        tok = min(tokens)
        if tok in first_tokens:
            fallback = db.query_moe_expert_compute(
                kernel, quant, UNCOLLECTED_DIST, "context", topk, experts, slots, hidden, inter, tp, ep, tok
            )
            assert float(fallback) == pytest.approx(first_tokens[tok]["latency"], rel=REL_TOL)
            comparisons += 1
    assert comparisons > 300, f"trtllm wideep sweep too small: {comparisons}"
    # The sweep genuinely covered the EPLB axes.
    assert {256, 288, 384} <= slots_seen
    assert any(dist.endswith("_eplb") for dist in dists_seen)


# ---------------------------------------------------------------------------
# (c) Review follow-ups: eplb / per-call quant override parity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.path.exists(SGLANG_CONTEXT_PATH),
    reason="shipped h200 sglang 0.5.6.post2 wideep parquets not present",
)
def test_l1_sglang_context_eplb_token_correction_equivalence():
    # Legacy applied int(tokens * 0.8) for context EPLB before the table walk
    # (the retired moe.py); the unified query must reproduce it. Probed
    # raw-derivably: pick a collected token t0 (t0 % 4 == 0) and query
    # tok = t0 * 5 / 4, so the corrected walk lands EXACTLY on the raw t0 row.
    db = get_database("h200_sxm", "sglang", "0.5.6.post2")
    legacy_table = load_wideep_context_moe_data(SGLANG_CONTEXT_PATH)
    comparisons = 0
    for (quant, dist, topk, experts, hidden, inter, tp, ep), tokens in itertools.islice(
        _iter_slices(legacy_table, 8), 6
    ):
        anchors = [t for t in sorted(tokens) if t % 4 == 0][:2]
        for t0 in anchors:
            tok = t0 * 5 // 4  # int(tok * 0.8) == t0
            unified = db.query_moe_expert_compute(
                "deepep_moe",
                quant,
                dist,
                "context",
                topk,
                experts,
                experts,
                hidden,
                inter,
                tp,
                ep,
                tok,
                enable_eplb=True,
            )
            context = f"eplb ctx {quant.name} {dist} {topk=} {experts=} {ep=} {tok=} (anchor {t0})"
            assert float(unified) == pytest.approx(tokens[t0]["latency"], rel=REL_TOL), context
            comparisons += 1
    assert comparisons >= 8, f"eplb sweep too small: {comparisons}"
    # Generation is NOT corrected — one probe pinning the asymmetry.
    (quant, dist, topk, experts, hidden, inter, tp, ep), tokens = next(_iter_slices(legacy_table, 8))
    with_eplb = db.query_moe_expert_compute(
        "deepep_moe",
        quant,
        dist,
        "generation",
        topk,
        experts,
        experts,
        hidden,
        inter,
        tp,
        ep,
        min(tokens),
        enable_eplb=True,
    )
    without = db.query_moe_expert_compute(
        "deepep_moe", quant, dist, "generation", topk, experts, experts, hidden, inter, tp, ep, min(tokens)
    )
    assert float(with_eplb) == float(without)


# The non-gated OVERFLOW equivalence probe (is_gated=False shaping the
# beyond-range num_gemms SOL) retired with the legacy facade: beyond-range
# util-holds are engine math with no raw-row expectation, anchored by the
# frozen parity goldens.


@pytest.mark.skipif(
    not os.path.exists(SGLANG_CONTEXT_PATH),
    reason="shipped h200 sglang 0.5.6.post2 wideep parquets not present",
)
def test_moe_expert_compute_per_call_quant_mode_override_reaches_the_walk():
    # Legacy expert-compute ops honor kwargs.get("quant_mode"); the op-level
    # override must reach both kernel resolution and the table walk. Construct
    # with an uncollected ctor mode and query with the collected one: only the
    # override can make the walk succeed.
    from aiconfigurator_core.sdk import common
    from aiconfigurator_core.sdk.errors import PerfDataNotAvailableError
    from aiconfigurator_core.sdk.operations.moe_comm import MoEExpertCompute

    db = get_database("h200_sxm", "sglang", "0.5.6.post2")
    legacy_table = load_wideep_context_moe_data(SGLANG_CONTEXT_PATH)
    (quant, dist, topk, experts, hidden, inter, tp, ep), tokens = next(_iter_slices(legacy_table, 8))
    op = MoEExpertCompute(
        "review_probe",
        1.0,
        hidden_size=hidden,
        inter_size=inter,
        topk=topk,
        num_experts=experts,
        moe_ep_size=ep,
        quant_mode=common.MoEQuantMode.nvfp4,  # uncollected on this table
        workload_distribution=dist,
        attention_dp_size=1,
        inference_phase="context",
    )
    with pytest.raises(PerfDataNotAvailableError):
        op.query(db, x=min(tokens))  # ctor mode alone must miss
    overridden = op.query(db, x=min(tokens), quant_mode=quant)  # override hits
    direct = db.query_moe_expert_compute(
        "deepep_moe", quant, dist, "context", topk, experts, experts, hidden, inter, tp, ep, min(tokens)
    )
    assert float(overridden) == pytest.approx(float(direct), rel=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

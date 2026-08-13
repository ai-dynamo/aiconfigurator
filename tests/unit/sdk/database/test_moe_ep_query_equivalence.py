# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L1 query-equivalence gate: ``query_moe_expert_compute`` vs the legacy compute queries.

Shipped-data sweeps on real databases. For EVERY slice of the legacy tables
and a token-probe set spanning exact hits, in-range interpolation, and the
beyond-range util-hold ({min, max, midpoints of adjacent collected points,
2 x max}), the unified query must reproduce the legacy query at rel <= 1e-9:

- sglang h200_sxm 0.5.6.post2: every ``wideep_context_moe`` /
  ``wideep_generation_moe`` slice == ``query_moe`` with
  ``moe_backend="deepep_moe"`` and the matching ``is_context``. The legacy
  eplb x0.8 token correction applies only when ``enable_eplb=True`` is
  passed; it is NOT passed here (raw-token semantics on both sides).
- trtllm gb200 1.3.0rc10: every ``wideep_moe`` slice (all num_slots
  {256, 288, 384}, all distributions incl. ``_eplb``) ==
  ``query_wideep_moe_compute`` (which auto-selects the same sole collected
  kernel). The legacy table has no phase split, so BOTH unified phases must
  return bit-identical values. NO carve-outs.

Both sweeps also probe an UNCOLLECTED distribution (``"power_law"`` — the
production default request string, absent from both shipped tables) once per
slice: on sglang this pins the ->"uniform" fallback against ``query_moe``;
on gb200 (whose table has no "uniform") it pins the ->first-available
fallback against ``query_wideep_moe_compute``'s
``available_distributions[0]`` behavior.

The 2 x max overflow probes compare like-for-like because ``query_moe_expert_compute``
reproduces the oracle boundary util-hold exactly: the same ``perf_interp``
Grid token axis with the same wideep-MoE roofline SOL (num_slots-based; equal
to the sglang oracle's num_experts-based SOL since the sglang-adapted slices
pin num_slots == num_experts, and num_gemms = 3 matches both legacy queries'
``is_gated=True`` default).
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


def _token_probes(token_keys):
    """{min, max, midpoints of adjacent collected points, 2 x max}."""
    keys = sorted(token_keys)
    probes = {keys[0], keys[-1], 2 * keys[-1]}
    for lo, hi in itertools.pairwise(keys):
        probes.add((lo + hi) // 2)
    return sorted(probes)


def _assert_equivalent(unified, legacy, context):
    assert float(unified) == pytest.approx(float(legacy), rel=REL_TOL), context
    assert unified.energy == pytest.approx(legacy.energy, rel=REL_TOL), context


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

    # Legacy tables: [quant][dist][topk][experts][hidden][inter][tp][ep] -> {tokens: leaf}.
    for phase, path, loader, is_context in (
        ("context", SGLANG_CONTEXT_PATH, load_wideep_context_moe_data, True),
        ("generation", SGLANG_GENERATION_PATH, load_wideep_generation_moe_data, False),
    ):
        legacy_table = loader(path)
        assert legacy_table
        assert all(UNCOLLECTED_DIST not in legacy_table[quant] for quant in legacy_table)
        comparisons = 0
        for (quant, dist, topk, experts, hidden, inter, tp, ep), tokens in _iter_slices(legacy_table, 8):
            # The collected distribution plus one uncollected probe pinning
            # the fallback path (legacy resolves "power_law" -> "uniform").
            for probe_dist, probe_tokens in ((dist, _token_probes(tokens)), (UNCOLLECTED_DIST, [min(tokens)])):
                for tok in probe_tokens:
                    # num_slots = num_experts: the legacy sglang tables have no
                    # EPLB redundancy axis (spec §4.2 adapter pin).
                    unified = db.query_moe_expert_compute(
                        "deepep_moe", quant, probe_dist, phase, topk, experts, experts, hidden, inter, tp, ep, tok
                    )
                    legacy = db.query_moe(
                        num_tokens=tok,
                        hidden_size=hidden,
                        inter_size=inter,
                        topk=topk,
                        num_experts=experts,
                        moe_tp_size=tp,
                        moe_ep_size=ep,
                        quant_mode=quant,
                        workload_distribution=probe_dist,
                        is_context=is_context,
                        moe_backend="deepep_moe",
                    )
                    _assert_equivalent(
                        unified,
                        legacy,
                        f"sglang wideep {phase} {quant.name} {probe_dist} {topk=} {experts=} {ep=} {tok=}",
                    )
                    comparisons += 1
        assert comparisons > 100, f"sglang wideep {phase} sweep too small: {comparisons}"


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

    # Legacy table: [kernel][quant][dist][topk][experts][hidden][inter][slots][tp][ep] -> {tokens: leaf}.
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
        # The collected distribution plus one uncollected probe pinning the
        # fallback path: with no "uniform" collected, both the legacy oracle
        # and the unified query must answer with the FIRST available
        # distribution in table insertion order.
        for probe_dist, probe_tokens in ((dist, _token_probes(tokens)), (UNCOLLECTED_DIST, [min(tokens)])):
            for tok in probe_tokens:
                context = f"trtllm wideep {kernel} {quant.name} {probe_dist} {slots=} {experts=} {hidden=} {ep=} {tok=}"
                # The legacy table has no context/generation split: both unified
                # phases carry the same rows and must return identical values.
                unified_ctx = db.query_moe_expert_compute(
                    kernel, quant, probe_dist, "context", topk, experts, slots, hidden, inter, tp, ep, tok
                )
                unified_gen = db.query_moe_expert_compute(
                    kernel, quant, probe_dist, "generation", topk, experts, slots, hidden, inter, tp, ep, tok
                )
                assert float(unified_ctx) == float(unified_gen), context
                assert unified_ctx.energy == unified_gen.energy, context
                legacy = db.query_wideep_moe_compute(
                    num_tokens=tok,
                    hidden_size=hidden,
                    inter_size=inter,
                    topk=topk,
                    num_experts=experts,
                    num_slots=slots,
                    moe_tp_size=tp,
                    moe_ep_size=ep,
                    quant_mode=quant,
                    workload_distribution=probe_dist,
                )
                _assert_equivalent(unified_ctx, legacy, context)
                comparisons += 1
    assert comparisons > 500, f"trtllm wideep sweep too small: {comparisons}"
    # The sweep genuinely covered the EPLB axes.
    assert {256, 288, 384} <= slots_seen
    assert any(dist.endswith("_eplb") for dist in dists_seen)


# ---------------------------------------------------------------------------
# (c) Review follow-ups: eplb / is_gated / per-call quant override parity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.path.exists(SGLANG_CONTEXT_PATH),
    reason="shipped h200 sglang 0.5.6.post2 wideep parquets not present",
)
def test_l1_sglang_context_eplb_token_correction_equivalence():
    # Legacy applies int(tokens * 0.8) for context EPLB before the table walk
    # (moe.py:649); the unified query must reproduce it on every context
    # slice — including the reviewer's probe point (power_law_0.6, topk=8,
    # experts=256, hidden=7168, inter=2048, tp=1, ep=2, 32 tokens: 1.0025957
    # legacy vs 1.2833225 uncorrected).
    db = get_database("h200_sxm", "sglang", "0.5.6.post2")
    legacy_table = load_wideep_context_moe_data(SGLANG_CONTEXT_PATH)
    comparisons = 0
    for (quant, dist, topk, experts, hidden, inter, tp, ep), tokens in itertools.islice(
        _iter_slices(legacy_table, 8), 6
    ):
        for tok in _token_probes(tokens)[:3]:
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
            legacy = db.query_moe(
                num_tokens=tok,
                hidden_size=hidden,
                inter_size=inter,
                topk=topk,
                num_experts=experts,
                moe_tp_size=tp,
                moe_ep_size=ep,
                quant_mode=quant,
                workload_distribution=dist,
                is_context=True,
                moe_backend="deepep_moe",
                enable_eplb=True,
            )
            _assert_equivalent(unified, legacy, f"eplb ctx {quant.name} {dist} {topk=} {experts=} {ep=} {tok=}")
            comparisons += 1
    assert comparisons >= 12, f"eplb sweep too small: {comparisons}"
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


@pytest.mark.skipif(
    not os.path.exists(SGLANG_GENERATION_PATH),
    reason="shipped h200 sglang 0.5.6.post2 wideep parquets not present",
)
def test_l1_sglang_non_gated_overflow_equivalence():
    # is_gated shapes the beyond-range SOL (num_gemms 3 vs 2, moe.py:309).
    # Probe every slice at 2x the collected max where the util-hold rides the
    # roofline — includes the reviewer's EP32 point (0.2264363 legacy
    # non-gated vs 0.2308887 with the gemm count pinned to 3).
    db = get_database("h200_sxm", "sglang", "0.5.6.post2")
    legacy_table = load_wideep_generation_moe_data(SGLANG_GENERATION_PATH)
    comparisons = 0
    for (quant, dist, topk, experts, hidden, inter, tp, ep), tokens in itertools.islice(
        _iter_slices(legacy_table, 8), 6
    ):
        tok = 2 * max(tokens)
        unified = db.query_moe_expert_compute(
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
            tok,
            is_gated=False,
        )
        legacy = db.query_moe(
            num_tokens=tok,
            hidden_size=hidden,
            inter_size=inter,
            topk=topk,
            num_experts=experts,
            moe_tp_size=tp,
            moe_ep_size=ep,
            quant_mode=quant,
            workload_distribution=dist,
            is_context=False,
            moe_backend="deepep_moe",
            is_gated=False,
        )
        _assert_equivalent(unified, legacy, f"non-gated overflow {quant.name} {dist} {topk=} {experts=} {ep=} {tok=}")
        comparisons += 1
    assert comparisons >= 4


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

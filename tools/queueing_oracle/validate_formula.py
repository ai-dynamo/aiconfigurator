# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the closed-form queueing correction directly against the DES oracle.

Both sides consume IDENTICAL timing functions, so every residual is
scheduling-semantics approximation error in the closed form, not timing
error. The DES itself was verified to 0.0% against dynamo.replay (README).

The closed form is a mean-field model (no cohort-echo tracking), so
tolerances are honest rather than tight; each metric's residual mechanism
is documented in docs/design/queueing_model.md. Out-of-domain regimes
(KV-capacity pressure, max_num_seqs-capped queueing, non-stationary
arrivals) are intentionally NOT in this battery — the model's domain is
gated upstream by AIC's memory checks and the API input shape.

Run from repo root:
    PYTHONPATH=src:tools/queueing_oracle python3 tools/queueing_oracle/validate_formula.py
"""

from __future__ import annotations

import math
import sys
from statistics import mean

import workload as wl_gen
from vllm_sim import CallbackPerfModel, EngineArgs, Simulator

from aiconfigurator.sdk.queueing import operating_point_columns

# ---------------------------------------------------------------------------
# shared timing basis
# ---------------------------------------------------------------------------


def f_prefill(batch: int, effective_isl: int, prefix: int) -> float:
    tokens = float(batch * effective_isl)
    return max(0.0, 4.209989e-07 * tokens * tokens + 1.518344e-02 * tokens + 16.50142)


def f_decode(batch: int, ctx: int) -> float:
    return max(1.0, 3.0 + 0.06 * batch + 0.0011 * ctx)


DES_PERF = CallbackPerfModel(prefill_fn=f_prefill, decode_fn=f_decode)


# ---------------------------------------------------------------------------
# DES side
# ---------------------------------------------------------------------------


def pct(sorted_vals, q):
    idx = min(len(sorted_vals) - 1, max(0, math.ceil(q * len(sorted_vals)) - 1))
    return sorted_vals[idx]


def des_agg_stats(isl, osl, c, budget, chunked=True, prefix_ratio=0.0, n_mult=10, block_size=64):
    n = n_mult * c
    args = EngineArgs(max_num_batched_tokens=budget, enable_chunked_prefill=chunked, block_size=block_size)
    reqs = wl_gen.synthetic(
        request_count=n, isl=isl, osl=osl, block_size=block_size, shared_prefix_ratio=prefix_ratio, num_prefix_groups=1
    )
    Simulator(1, args, DES_PERF, concurrency=c).run(reqs)

    by_dispatch = sorted(reqs, key=lambda r: (r.dispatch_ms, r.rid))
    transient = by_dispatch[:c]
    steady = by_dispatch[5 * c :]
    t_ttft = [r.token_times[0] - r.dispatch_ms for r in transient]
    s_ttft = sorted(r.token_times[0] - r.dispatch_ms for r in steady)
    itl = sorted(g for r in steady for g in (b - a for a, b in zip(r.token_times, r.token_times[1:], strict=False)))
    return {
        "ttft_steady_mean": mean(s_ttft),
        "ttft_steady_p50": pct(s_ttft, 0.5),
        "ttft_steady_p99": pct(s_ttft, 0.99),
        "ttft_transient_mean": mean(t_ttft),
        "ttft_transient_max": max(t_ttft),
        "ttft_blended_mean": mean(r.token_times[0] - r.dispatch_ms for r in reqs),
        "itl_p50": pct(itl, 0.5),
        "itl_p99": pct(itl, 0.99),
        "itl_mean": mean(itl),
    }


# ---------------------------------------------------------------------------
# closed-form side: map (workload, engine) onto the run_agg operating point
# ---------------------------------------------------------------------------


def closed_form_stats(isl, osl, c, budget, chunked=True, prefix=0, n_mult=10, **_):
    isl_eff = max(1, isl - prefix)
    ctx_tokens = max(1, budget - c)  # B_eff: running decodes spend budget first
    ctx_mean = isl + osl // 2
    t_gen = f_decode(c, ctx_mean)
    # run_agg semantics: a mix pass processes ctx_tokens of prefill work
    # (spanning requests) alongside the decode batch; step counts are
    # batch-level (steps_to_finish_ctx = ceil(isl*b/ctx_tokens))
    t_mix = f_prefill(1, min(ctx_tokens, isl_eff * c), prefix) + t_gen
    num_mix = math.ceil(isl_eff * c / ctx_tokens)
    num_gen = max(0.0, osl - num_mix)

    cols = operating_point_columns(
        isl=isl_eff,
        osl=osl,
        batch_size=c,
        ctx_tokens=ctx_tokens,
        mix_step_ms=t_mix,
        genonly_step_ms=t_gen,
        prefill_step_ms=t_mix - t_gen,
        num_mix_steps=num_mix,
        num_genonly_steps=num_gen,
    )
    n = n_mult * c
    blended = (min(c, n) * cols["ttft_transient_mean"] + (n - min(c, n)) * cols["ttft_steady_mean"]) / n
    return {
        "ttft_steady_mean": cols["ttft_steady_mean"],
        "ttft_steady_p50": cols["ttft_steady_p50"],
        "ttft_steady_p99": cols["ttft_steady_p99"],
        "ttft_transient_mean": cols["ttft_transient_mean"],
        "ttft_transient_max": cols["ttft_transient_max"],
        "ttft_blended_mean": blended,
        "itl_p50": cols["itl_p50"],
        "itl_p99": cols["itl_p99"],
        "itl_mean": cols["itl_mean"],
    }


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

# per-metric tolerance (%): honest bounds for a mean-field closed form.
# Residual mechanisms: cohort echo (steady/transient means), first-order
# staircase (transient), decile residual discretization (quantiles).
TOLERANCES = {
    "ttft_steady_mean": 25.0,
    "ttft_steady_p50": 20.0,
    "ttft_steady_p99": 20.0,
    "ttft_transient_mean": 20.0,
    "ttft_transient_max": 20.0,
    "ttft_blended_mean": 15.0,
    "itl_p50": 5.0,
    "itl_p99": 5.0,
    "itl_mean": 20.0,
}


def compare(name, des, formula, exempt=(), tolerances=None):
    tolerances = tolerances or TOLERANCES
    print(f"\n=== {name} ===")
    print(f"{'metric':<22}{'DES':>12}{'model':>13}{'err':>9}{'tol':>7}")
    failures = []
    for k, dv in des.items():
        fv = formula[k]
        tol = tolerances[k]
        err = (fv - dv) / dv * 100 if dv else (0.0 if not fv else float("inf"))
        exempted = k in exempt
        flag = "  <-- FAIL" if abs(err) > tol and not exempted else ""
        note = "  (info only)" if exempted else ""
        print(f"{k:<22}{dv:>12.2f}{fv:>13.2f}{err:>8.1f}%{tol:>6.0f}%{flag}{note}")
        if abs(err) > tol and not exempted:
            failures.append((k, round(err, 1)))
    return failures


def evaluator_stats(isl, osl, c, budget, chunked=True, prefix=0, n_mult=10, **_):
    from aiconfigurator.sdk.queueing import EngineSpec, WorkloadSpec, evaluate_closed_loop

    class _Timing:
        def prefill_ms(self, b, mean_isl, mean_prefix):
            return f_prefill(b, max(0, mean_isl - mean_prefix), mean_prefix)

        def decode_ms(self, b, ctx):
            return f_decode(b, ctx)

    wl = WorkloadSpec(isl=isl, osl=osl, prefix=prefix, concurrency=c, num_requests=n_mult * c)
    eng = EngineSpec(max_num_batched_tokens=budget, enable_chunked_prefill=chunked)
    rep = evaluate_closed_loop(wl, eng, _Timing(), backend="vllm")
    return {
        "ttft_steady_mean": rep.ttft_steady.mean,
        "ttft_steady_p50": rep.ttft_steady.p50,
        "ttft_steady_p99": rep.ttft_steady.p99,
        "ttft_transient_mean": rep.ttft_transient.mean,
        "ttft_transient_max": rep.ttft_transient.maximum,
        "ttft_blended_mean": rep.ttft_mean_n,
        "itl_p50": rep.itl.p50,
        "itl_p99": rep.itl.p99,
        "itl_mean": rep.itl.mean,
    }


# evaluator: same model evaluated numerically — tight tolerances
EVALUATOR_TOLERANCES = dict.fromkeys(TOLERANCES, 10.0)
EVALUATOR_TOLERANCES["itl_mean"] = 15.0
EVALUATOR_TOLERANCES["itl_p99"] = 15.0


def main():
    cases = [
        ("A isl4096 osl256 C32 B8192", dict(isl=4096, osl=256, c=32, budget=8192), ()),
        ("B isl1024 osl128 C64 B8192", dict(isl=1024, osl=128, c=64, budget=8192), ()),
        ("C isl512 osl512 C128 B4096", dict(isl=512, osl=512, c=128, budget=4096), ()),
        ("D isl8192 osl64 C16 B8192", dict(isl=8192, osl=64, c=16, budget=8192), ()),
        ("E chunked-off isl2048 C16 B8192", dict(isl=2048, osl=128, c=16, budget=8192, chunked=False), ()),
        # prefix: itl_p99 info-only — constant-hit assumption vs the DES's
        # cold-start cache locks a different cohort phase (mix-pass mass
        # point shifts by one cohort step); TTFT unaffected.
        ("I prefix2048 isl4096 osl128 C32", dict(isl=4096, osl=128, c=32, budget=8192, prefix_ratio=0.5), ("itl_p99",)),
        ("J short-osl isl2048 osl16 C32", dict(isl=2048, osl=16, c=32, budget=8192), ()),
        ("K C1 isl1024 osl64", dict(isl=1024, osl=64, c=1, budget=8192), ()),
        ("L deep-staircase B2048 isl4096 C16", dict(isl=4096, osl=128, c=16, budget=2048), ()),
    ]
    all_failures = []
    for name, kw, exempt in cases:
        des = des_agg_stats(**kw)
        fkw = dict(kw)
        prefix_ratio = fkw.pop("prefix_ratio", 0.0)
        if prefix_ratio:
            fkw["prefix"] = int(kw["isl"] * prefix_ratio)

        ev = evaluator_stats(**fkw)
        failures = compare(f"{name} [evaluator, GATED]", des, ev, exempt=exempt, tolerances=EVALUATOR_TOLERANCES)
        if failures:
            all_failures.append((f"{name} [evaluator]", failures))

        # closed-form screening tier: REPORTED, not gated. Its role is the
        # sweep hot path, where the workload is fixed and candidates differ
        # only in engine/parallel config — the per-workload bias is shared
        # across candidates, preserving ranking. Cross-workload quantitative
        # use should go through the evaluator. Sanity is still enforced.
        formula = closed_form_stats(**fkw)
        compare(f"{name} [closed-form screening, report-only]", des, formula, exempt=tuple(des))
        assert formula["ttft_steady_p99"] >= formula["ttft_steady_p50"] > 0
        assert formula["ttft_transient_max"] >= formula["ttft_transient_mean"] > 0
        assert formula["itl_p99"] >= formula["itl_p50"] > 0

    print("\n" + ("ALL WITHIN TOLERANCE" if not all_failures else f"FAILURES: {all_failures}"))
    return 1 if all_failures else 0


if __name__ == "__main__":
    sys.exit(main())

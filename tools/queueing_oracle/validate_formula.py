# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the closed-form queueing correction directly against the DES oracle.

Both sides consume IDENTICAL timing functions, so every residual is
scheduling-semantics approximation error in the closed form, not timing
error. The DES's own scheduling semantics are audited clause-by-clause
against the vLLM v1 scheduler source (docs/design/queueing_model.md par.5).

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
    # synthetic roofline-shaped basis (launch + bandwidth-linear +
    # compute-quadratic); both sides consume it, so the constants are
    # arbitrary — they only need realistic pass-length ratios
    tokens = float(batch * effective_isl)
    return 12.0 + 0.016 * tokens + 5e-07 * tokens * tokens


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
        if not math.isfinite(dv) or not math.isfinite(fv):
            err = float("inf")  # fail closed: a NaN result must not pass the gate
        else:
            err = (fv - dv) / dv * 100 if dv else (0.0 if not fv else float("inf"))
        exempted = k in exempt
        failed = (not math.isfinite(err) or abs(err) > tol) and not exempted
        flag = "  <-- FAIL" if failed else ""
        note = "  (info only)" if exempted else ""
        print(f"{k:<22}{dv:>12.2f}{fv:>13.2f}{err:>8.1f}%{tol:>6.0f}%{flag}{note}")
        if failed:
            failures.append((k, round(err, 1)))
    return failures


def evaluator_stats(isl, osl, c, budget, chunked=True, prefix=0, n_mult=10, **_):
    from aiconfigurator.sdk.queueing import EngineSpec, WorkloadSpec, evaluate_closed_loop

    class _Timing:
        # same clamps as the DES side's CallbackPerfModel, so both paths
        # consume identical timings and residuals isolate scheduling
        def prefill_ms(self, b, mean_isl, mean_prefix):
            return max(0.0, f_prefill(b, max(0, mean_isl - mean_prefix), mean_prefix))

        def decode_ms(self, b, ctx):
            return max(1.0, f_decode(b, ctx))

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


# ---------------------------------------------------------------------------
# disagg (P/D tandem) side
# ---------------------------------------------------------------------------


def des_disagg_stats(isl, osl, c, n_prefill, n_decode, kv_bytes=0, bw=0.0, n_mult=10):
    from vllm_sim import DisaggSimulator, EngineArgs, TransferSpec

    n = n_mult * c
    reqs = wl_gen.synthetic(request_count=n, isl=isl, osl=osl, block_size=64)
    spec = TransferSpec(kv_bytes, bw, bw, bw_efficiency=0.8) if kv_bytes else None
    DisaggSimulator(
        n_prefill,
        n_decode,
        EngineArgs(worker_type="prefill"),
        EngineArgs(worker_type="decode"),
        DES_PERF,
        concurrency=c,
        transfer=spec,
    ).run(reqs)
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
        "itl_p50": pct(itl, 0.5),
        "itl_p99": pct(itl, 0.99),
        "itl_mean": mean(itl),
    }


def disagg_evaluator_stats(isl, osl, c, n_prefill, n_decode, kv_bytes=0, bw=0.0, n_mult=10):
    from aiconfigurator.sdk.queueing import DisaggSpec, EngineSpec, WorkloadSpec, evaluate_disagg

    class _Timing:
        # same clamps as the DES side's CallbackPerfModel, so both paths
        # consume identical timings and residuals isolate scheduling
        def prefill_ms(self, b, mean_isl, mean_prefix):
            return max(0.0, f_prefill(b, max(0, mean_isl - mean_prefix), mean_prefix))

        def decode_ms(self, b, ctx):
            return max(1.0, f_decode(b, ctx))

    wl = WorkloadSpec(isl=isl, osl=osl, concurrency=c, num_requests=n_mult * c)
    spec = DisaggSpec(n_prefill, n_decode, kv_bytes_per_token=kv_bytes, egress_bytes_per_s=bw, ingress_bytes_per_s=bw)
    rep = evaluate_disagg(wl, EngineSpec(), EngineSpec(), _Timing(), _Timing(), spec)
    return {
        "ttft_steady_mean": rep.ttft_steady.mean,
        "ttft_steady_p50": rep.ttft_steady.p50,
        "ttft_steady_p99": rep.ttft_steady.p99,
        "ttft_transient_mean": rep.ttft_transient.mean,
        "ttft_transient_max": rep.ttft_transient.maximum,
        "itl_p50": rep.itl.p50,
        "itl_p99": rep.itl.p99,
        "itl_mean": rep.itl.mean,
    }


# evaluator: same model evaluated numerically — tight tolerances
EVALUATOR_TOLERANCES = dict.fromkeys(TOLERANCES, 10.0)
EVALUATOR_TOLERANCES["itl_mean"] = 15.0
EVALUATOR_TOLERANCES["itl_p99"] = 15.0


def check_open_loop_disagg_family(n_prefill, n_decode, rate_rps, n, shapes, kv_bytes, bw, tolerances):
    """Open-loop disagg: same verbatim tuples through the DES (trace mode)
    and the evaluator (arrival_trace); gate on distribution stats over ALL
    requests plus a per-request TTFT join (the exact-replay contract)."""
    import random as _r

    from vllm_sim import DisaggSimulator, EngineArgs, TransferSpec

    from aiconfigurator.sdk.queueing import DisaggSpec, EngineSpec, WorkloadSpec, evaluate_disagg

    rng = _r.Random(12345)
    tuples = []
    t = 0.0
    for i in range(n):
        t += rng.expovariate(rate_rps) * 1000.0
        isl, osl = shapes[i % len(shapes)]
        tuples.append((t, isl, osl))

    reqs = wl_gen.from_tuples(tuples, block_size=64)
    DisaggSimulator(
        n_prefill,
        n_decode,
        EngineArgs(worker_type="prefill"),
        EngineArgs(worker_type="decode"),
        DES_PERF,
        concurrency=None,
        transfer=TransferSpec(kv_bytes, bw, bw, bw_efficiency=0.8),
    ).run(reqs)
    des_ttft = [r.token_times[0] - r.dispatch_ms for r in reqs]
    des_itl = sorted(g for r in reqs for g in (b - a for a, b in zip(r.token_times, r.token_times[1:], strict=False)))
    des = {
        "ttft_all_mean": mean(des_ttft),
        "ttft_all_p50": pct(sorted(des_ttft), 0.5),
        "ttft_all_p99": pct(sorted(des_ttft), 0.99),
        "itl_p50": pct(des_itl, 0.5),
        "itl_p99": pct(des_itl, 0.99),
        "itl_mean": mean(des_itl),
    }

    class _Timing:
        def prefill_ms(self, b, mean_isl, mean_prefix):
            return max(0.0, f_prefill(b, max(0, mean_isl - mean_prefix), mean_prefix))

        def decode_ms(self, b, ctx):
            return max(1.0, f_decode(b, ctx))

    mean_isl = sum(s[0] for s in shapes) // len(shapes)
    mean_osl = sum(s[1] for s in shapes) // len(shapes)
    wl = WorkloadSpec(isl=mean_isl, osl=mean_osl, request_rate=rate_rps)
    spec = DisaggSpec(n_prefill, n_decode, kv_bytes_per_token=kv_bytes, egress_bytes_per_s=bw, ingress_bytes_per_s=bw)
    rep = evaluate_disagg(
        wl,
        EngineSpec(),
        EngineSpec(),
        _Timing(),
        _Timing(),
        spec,
        arrival_trace=[(t, isl, 0, osl) for (t, isl, osl) in tuples],
        warmup_requests=0,
    )
    ev_ttft = [p["ttft_ms"] for p in rep.per_request]
    ev = {
        "ttft_all_mean": rep.ttft_steady.mean,
        "ttft_all_p50": rep.ttft_steady.p50,
        "ttft_all_p99": rep.ttft_steady.p99,
        "itl_p50": rep.itl.p50,
        "itl_p99": rep.itl.p99,
        "itl_mean": rep.itl.mean,
    }
    tol = dict.fromkeys(des, 10.0)
    tol.update({k: v for k, v in tolerances.items() if k in des})
    failures = compare("  (stats)", des, ev, tolerances=tol)

    # per-request exact-replay join: identical arrivals and shapes, so the
    # TTFT of each individual request must track its DES counterpart
    rel = sorted(abs(e - d) / max(d, 1e-9) for d, e in zip(des_ttft, ev_ttft, strict=True))
    med, p90 = pct(rel, 0.5) * 100.0, pct(rel, 0.9) * 100.0
    print(f"  per-request |dTTFT| median {med:.1f}% p90 {p90:.1f}%")
    if med > 5.0 or p90 > 20.0:
        failures = list(failures) + [f"per-request join med {med:.1f}% p90 {p90:.1f}%"]
    return failures


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

    # disagg (P/D tandem) families: the sdk tandem recursion vs the DES
    # DisaggSimulator, identical timing and TransferSpec on both sides.
    # Same-phase comparison (simultaneous initial burst): the tandem system
    # is multi-stable in cohort phase, so both sides are driven from the
    # same initial condition; phase-robust output is evaluate_disagg_mixed.
    disagg_cases = [
        ("DA 1P1D isl2048 osl64 C8", dict(isl=2048, osl=64, c=8, n_prefill=1, n_decode=1)),
        (
            "DB fan-in 2P1D isl4096 bw50G",
            dict(isl=4096, osl=64, c=16, n_prefill=2, n_decode=1, kv_bytes=100_000, bw=50e9),
        ),
        (
            "DC bw-tight 2P1D isl4096 bw1G",
            dict(isl=4096, osl=64, c=16, n_prefill=2, n_decode=1, kv_bytes=100_000, bw=1e9),
        ),
        (
            "DD 2P2D isl1024 osl128 C32",
            dict(isl=1024, osl=128, c=32, n_prefill=2, n_decode=2, kv_bytes=100_000, bw=50e9),
        ),
    ]
    disagg_tol = dict.fromkeys(
        (
            "ttft_steady_mean",
            "ttft_steady_p50",
            "ttft_steady_p99",
            "ttft_transient_mean",
            "ttft_transient_max",
        ),
        10.0,
    )
    disagg_tol.update({"itl_p50": 10.0, "itl_p99": 15.0, "itl_mean": 15.0})
    for name, kw in disagg_cases:
        des = des_disagg_stats(**kw)
        ev = disagg_evaluator_stats(**kw)
        failures = compare(f"{name} [tandem evaluator, GATED]", des, ev, tolerances=disagg_tol)
        if failures:
            all_failures.append((f"{name} [tandem]", failures))

    # open-loop disagg families: BOTH sides consume the same verbatim
    # (arrival_ms, isl, osl) tuples (DES trace mode vs evaluator
    # arrival_trace), so residuals isolate scheduling semantics — and the
    # per-request join gates the exact-replay path, not just the marginals.
    for name, kw in [
        (
            "DE open-loop 1P1D fixed isl2048",
            dict(n_prefill=1, n_decode=1, rate_rps=8.0, n=160, shapes=[(2048, 64)], kv_bytes=100_000, bw=50e9),
        ),
        (
            "DF open-loop fan-in 2P1D variable",
            dict(
                n_prefill=2,
                n_decode=1,
                rate_rps=12.0,
                n=200,
                shapes=[(512, 32), (1024, 64), (2048, 96), (4096, 128)],
                kv_bytes=100_000,
                bw=50e9,
            ),
        ),
    ]:
        failures = check_open_loop_disagg_family(**kw, tolerances=disagg_tol)
        if failures:
            all_failures.append((f"{name} [tandem open-loop]", failures))
        else:
            print(f"{name} [tandem open-loop, GATED]: within tolerance")

    print(
        "\n"
        + (
            "ALL GATED (EVALUATOR-TIER) METRICS WITHIN TOLERANCE"
            if not all_failures
            else f"EVALUATOR FAILURES: {all_failures}"
        )
    )
    return 1 if all_failures else 0


if __name__ == "__main__":
    sys.exit(main())

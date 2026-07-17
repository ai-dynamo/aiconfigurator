# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate sdk.queueing against the mocker-parity DES oracle.

Both sides consume the IDENTICAL timing functions, so every residual is
scheduler-semantics error in the formula, not timing error. The DES itself
was verified to 0.0% against dynamo mocker (see README).

Run from repo root:
    PYTHONPATH=src:tools/queueing_oracle python3 tools/queueing_oracle/validate_formula.py
"""

from __future__ import annotations

import sys
from statistics import mean

import workload as wl_gen
from vllm_sim import CallbackPerfModel, DisaggSimulator, EngineArgs, Simulator

from aiconfigurator.sdk.queueing import (
    DisaggSpec,
    EngineSpec,
    WorkloadSpec,
    estimate_disagg,
    evaluate_closed_loop,
)

# ---------------------------------------------------------------------------
# shared timing basis (mocker-polynomial-shaped prefill; decode depends on
# (batch, ctx) so both consumers can honour it exactly)
# ---------------------------------------------------------------------------


def f_prefill(batch: int, effective_isl: int, prefix: int) -> float:
    tokens = float(batch * effective_isl)
    return max(0.0, 4.209989e-07 * tokens * tokens + 1.518344e-02 * tokens + 16.50142)


def f_decode(batch: int, ctx: int) -> float:
    return max(1.0, 3.0 + 0.06 * batch + 0.0011 * ctx)


class FormulaTiming:
    def prefill_ms(self, batch_size, mean_isl, mean_prefix):
        return f_prefill(batch_size, max(0, mean_isl - mean_prefix), mean_prefix)

    def decode_ms(self, batch_size, context_len):
        return f_decode(batch_size, context_len)


DES_PERF = CallbackPerfModel(prefill_fn=f_prefill, decode_fn=f_decode)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def pct(sorted_vals, q):
    import math

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
    span = max(r.completed_ms for r in steady) - min(r.dispatch_ms for r in steady)
    thr = len(steady) / (span / 1000.0)
    return {
        "ttft_steady_mean": mean(s_ttft),
        "ttft_steady_p50": pct(s_ttft, 0.5),
        "ttft_steady_p99": pct(s_ttft, 0.99),
        "ttft_transient_mean": mean(t_ttft),
        "ttft_overall_mean": mean(r.token_times[0] - r.dispatch_ms for r in reqs),
        "itl_p50": pct(itl, 0.5),
        "itl_p99": pct(itl, 0.99),
        "itl_mean": mean(itl),
        "throughput_rps": thr,
    }


def formula_agg_stats(isl, osl, c, budget, chunked=True, prefix=0, n_mult=10):
    wl = WorkloadSpec(isl=isl, osl=osl, prefix=prefix, concurrency=c, num_requests=n_mult * c)
    eng = EngineSpec(max_num_batched_tokens=budget, enable_chunked_prefill=chunked)
    rep = evaluate_closed_loop(wl, eng, FormulaTiming(), backend="vllm")
    return {
        "ttft_steady_mean": rep.ttft_steady.mean,
        "ttft_steady_p50": rep.ttft_steady.p50,
        "ttft_steady_p99": rep.ttft_steady.p99,
        "ttft_transient_mean": rep.ttft_transient.mean,
        "ttft_overall_mean": rep.ttft_mean_n,
        "itl_p50": rep.itl.p50,
        "itl_p99": rep.itl.p99,
        "itl_mean": rep.itl.mean,
        "throughput_rps": rep.throughput_rps,
    }


def compare(name, des, formula, tol_pct):
    print(f"\n=== {name} ===")
    print(f"{'metric':<22}{'DES':>12}{'formula':>12}{'err':>9}")
    worst = 0.0
    for k, dv in des.items():
        fv = formula[k]
        err = (fv - dv) / dv * 100 if dv else float("nan")
        worst = max(worst, abs(err))
        flag = "  <-- FAIL" if abs(err) > tol_pct else ""
        print(f"{k:<22}{dv:>12.2f}{fv:>12.2f}{err:>8.1f}%{flag}")
    return worst


def main():
    cases = [
        ("A isl4096 osl256 C32 B8192", dict(isl=4096, osl=256, c=32, budget=8192)),
        ("B isl1024 osl128 C64 B8192", dict(isl=1024, osl=128, c=64, budget=8192)),
        ("C isl512 osl512 C128 B4096", dict(isl=512, osl=512, c=128, budget=4096)),
        ("D isl8192 osl64 C16 B8192", dict(isl=8192, osl=64, c=16, budget=8192)),
        ("E chunked-off isl2048 C16 B8192", dict(isl=2048, osl=128, c=16, budget=8192, chunked=False)),
    ]
    failures = []
    for name, kw in cases:
        des = des_agg_stats(**kw)
        fkw = dict(kw)
        fkw.pop("prefix_ratio", None)
        formula = formula_agg_stats(**fkw)
        worst = compare(name, des, formula, tol_pct=15.0)
        if worst > 15.0:
            failures.append((name, worst))

    # disagg case
    print("\n=== DISAGG 1P1D isl4096 osl256 C32 (pinned 64GB/s, 128KB/tok) ===")
    pargs = EngineArgs(worker_type="prefill", kv_transfer_bandwidth_gbps=64.0, kv_bytes_per_token=131072)
    dargs = EngineArgs(worker_type="decode")
    reqs = wl_gen.synthetic(request_count=640, isl=4096, osl=256, block_size=64)
    DisaggSimulator(1, 1, pargs, dargs, DES_PERF, concurrency=32).run(reqs)
    by_dispatch = sorted(reqs, key=lambda r: (r.dispatch_ms, r.rid))
    # exclude both the ramp (first 5C) and the drain tail (last 2C): the
    # closed loop degrades once the backlog empties, which is a benchmark
    # artifact, not steady-state behavior
    steady = by_dispatch[5 * 32 : -2 * 32]
    des_d = {
        "ttft_steady_mean": mean(r.token_times[0] - r.dispatch_ms for r in steady),
        "itl_mean": mean(
            g for r in steady for g in (b - a for a, b in zip(r.token_times, r.token_times[1:], strict=False))
        ),
        "throughput_rps": len(steady)
        / ((max(r.completed_ms for r in steady) - min(r.dispatch_ms for r in steady)) / 1000.0),
    }
    wl = WorkloadSpec(isl=4096, osl=256, concurrency=32, num_requests=320)
    rep = estimate_disagg(
        wl,
        EngineSpec(),
        EngineSpec(),
        FormulaTiming(),
        DisaggSpec(
            num_prefill_workers=1, num_decode_workers=1, kv_transfer_bandwidth_gbps=64.0, kv_bytes_per_token=131072
        ),
    )
    formula_d = {
        "ttft_steady_mean": rep.ttft_steady.mean,
        "itl_mean": rep.itl.mean,
        "throughput_rps": rep.throughput_rps,
    }
    worst = compare("disagg", des_d, formula_d, tol_pct=20.0)
    if worst > 20.0:
        failures.append(("disagg", worst))

    print("\n" + ("ALL WITHIN TOLERANCE" if not failures else f"FAILURES: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

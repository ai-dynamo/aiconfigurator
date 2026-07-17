# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Column-for-column parity check: DES vs a dynamo.replay report JSON.

Runs the DES on the identical synthetic workload and prints each metric
side by side with relative error. Timing model must match the replay run
(default: both polynomial).
"""

from __future__ import annotations

import argparse
import json
import math
from statistics import mean, median, pstdev

import workload
from vllm_sim import EngineArgs, PolynomialPerfModel, Simulator


def pct(sorted_vals, q):
    if not sorted_vals:
        return math.nan
    idx = min(len(sorted_vals) - 1, max(0, math.ceil(q * len(sorted_vals)) - 1))
    return sorted_vals[idx]


def stats(vals):
    if not vals:
        return {}
    sv = sorted(vals)
    return {
        "mean": mean(sv),
        "median": median(sv),
        "min": sv[0],
        "max": sv[-1],
        "p75": pct(sv, 0.75),
        "p90": pct(sv, 0.90),
        "p95": pct(sv, 0.95),
        "p99": pct(sv, 0.99),
        "std": pstdev(sv),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("replay_json")
    p.add_argument("--isl", type=int, default=4096)
    p.add_argument("--osl", type=int, default=256)
    p.add_argument("--request-count", type=int, default=200)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--num-gpu-blocks", type=int, default=16384)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--perf", choices=["poly", "aic"], default="poly")
    p.add_argument("--system", default="h200_sxm")
    p.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    p.add_argument("--backend-version", default="0.24.0")
    args = p.parse_args()

    with open(args.replay_json) as f:
        rep = json.load(f)

    if args.perf == "aic":
        from aic_adapter import AicPerfSession
        from vllm_sim import CallbackPerfModel

        aic = AicPerfSession("vllm", args.system, args.model, tp_size=1, backend_version=args.backend_version)
        perf = CallbackPerfModel(prefill_fn=aic.predict_prefill, decode_fn=lambda b, ctx: aic.predict_decode(b, ctx, 2))
    else:
        perf = PolynomialPerfModel()

    engine_args = EngineArgs(num_gpu_blocks=args.num_gpu_blocks, block_size=args.block_size)
    reqs = workload.synthetic(request_count=args.request_count, isl=args.isl, osl=args.osl, block_size=args.block_size)
    Simulator(1, engine_args, perf, concurrency=args.concurrency).run(reqs)

    done = [r for r in reqs if r.completed_ms >= 0]
    ttft = [r.token_times[0] - r.dispatch_ms for r in done]
    ttst = [r.token_times[1] - r.token_times[0] for r in done if len(r.token_times) > 1]
    itl, tpot = [], []
    for r in done:
        gaps = [b - a for a, b in zip(r.token_times, r.token_times[1:], strict=False)]
        itl.extend(gaps)
        if gaps:
            tpot.append(mean(gaps))
    e2e = [r.completed_ms - r.dispatch_ms for r in done]
    span = max(r.completed_ms for r in done) - min(r.dispatch_ms for r in done)

    des = {"ttft": stats(ttft), "ttst": stats(ttst), "itl": stats(itl), "tpot": stats(tpot), "e2e_latency": stats(e2e)}
    des_scalar = {
        "output_throughput_tok_s": sum(len(r.token_times) for r in done) / (span / 1000),
        "request_throughput_rps": len(done) / (span / 1000),
        "duration_ms": span,
    }

    print(f"{'metric':<32}{'mocker':>12}{'DES':>12}{'rel_err':>9}")
    order = ["mean", "median", "p75", "p90", "p95", "p99", "min", "max", "std"]
    keymap = {
        "mean": "mean",
        "median": "median",
        "p75": "p75",
        "p90": "p90",
        "p95": "p95",
        "p99": "p99",
        "min": "min",
        "max": "max",
        "std": "std",
    }
    for metric in ("ttft", "ttst", "itl", "tpot", "e2e_latency"):
        for s in order:
            mk = f"{keymap[s]}_{metric}_ms"
            if mk not in rep:
                continue
            mv, dv = rep[mk], des[metric].get(s, math.nan)
            err = (dv - mv) / mv * 100 if mv else math.nan
            print(f"{s + '_' + metric + '_ms':<32}{mv:>12.2f}{dv:>12.2f}{err:>8.1f}%")
        print()
    for k, dv in des_scalar.items():
        mv = rep.get(k)
        if mv:
            err = (dv - mv) / mv * 100
            print(f"{k:<32}{mv:>12.2f}{dv:>12.2f}{err:>8.1f}%")


if __name__ == "__main__":
    main()

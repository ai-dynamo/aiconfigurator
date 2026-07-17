# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI for the reference discrete-event scheduler oracle.

Examples:
  # closed-loop synthetic, mirrors dynamo.replay --replay-concurrency
  python run.py --request-count 200 --isl 4096 --osl 256 --concurrency 32

  # open-loop poisson arrivals, prefix sharing
  python run.py --request-count 500 --isl 5000 --osl 500 \
      --arrival-interval-ms 50 --poisson \
      --shared-prefix-ratio 0.5 --num-prefix-groups 8

  # mooncake trace
  python run.py --trace mooncake_trace.jsonl --trace-block-size 512

  # fidelity delta of naive token-count KV accounting
  python run.py ... --kv-mode token
"""

from __future__ import annotations

import argparse
import time

import metrics
import workload
from vllm_sim import EngineArgs, PolynomialPerfModel, Simulator


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    eng = p.add_argument_group("engine")
    eng.add_argument("--num-gpu-blocks", type=int, default=16384)
    eng.add_argument("--block-size", type=int, default=64)
    eng.add_argument("--max-num-seqs", type=int, default=256)
    eng.add_argument("--max-num-batched-tokens", type=int, default=8192)
    eng.add_argument("--no-enable-prefix-caching", action="store_true")
    eng.add_argument("--no-enable-chunked-prefill", action="store_true")
    eng.add_argument("--preemption-mode", choices=["lifo", "fifo"], default="lifo")
    eng.add_argument(
        "--kv-mode",
        choices=["block", "token"],
        default="block",
        help="'token' disables block sharing + prefix cache to measure the fidelity delta of naive accounting",
    )
    eng.add_argument("--num-workers", type=int, default=1)

    wl = p.add_argument_group("workload")
    wl.add_argument("--trace", type=str, default=None, help="mooncake-style jsonl trace; otherwise synthetic")
    wl.add_argument("--trace-block-size", type=int, default=512)
    wl.add_argument("--request-count", type=int, default=200)
    wl.add_argument("--isl", type=int, default=4096)
    wl.add_argument("--osl", type=int, default=256)
    wl.add_argument("--arrival-interval-ms", type=float, default=0.0)
    wl.add_argument("--poisson", action="store_true")
    wl.add_argument("--shared-prefix-ratio", type=float, default=0.0)
    wl.add_argument("--num-prefix-groups", type=int, default=8)
    wl.add_argument("--concurrency", type=int, default=None, help="closed-loop in-flight cap (TTFT from dispatch)")
    wl.add_argument("--seed", type=int, default=0)

    p.add_argument("--report-json", type=str, default=None)
    args = p.parse_args()

    engine_args = EngineArgs(
        num_gpu_blocks=args.num_gpu_blocks,
        block_size=args.block_size,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_prefix_caching=not args.no_enable_prefix_caching,
        enable_chunked_prefill=not args.no_enable_chunked_prefill,
        preemption_mode=args.preemption_mode,
        kv_mode=args.kv_mode,
    )

    if args.trace:
        reqs = workload.load_mooncake_trace(
            args.trace,
            engine_block_size=args.block_size,
            trace_block_size=args.trace_block_size,
            limit=args.request_count or None,
        )
    else:
        reqs = workload.synthetic(
            request_count=args.request_count,
            isl=args.isl,
            osl=args.osl,
            block_size=args.block_size,
            arrival_interval_ms=args.arrival_interval_ms,
            poisson=args.poisson,
            shared_prefix_ratio=args.shared_prefix_ratio,
            num_prefix_groups=args.num_prefix_groups,
            seed=args.seed,
        )

    sim = Simulator(args.num_workers, engine_args, PolynomialPerfModel(), concurrency=args.concurrency)
    wall0 = time.perf_counter()
    sim.run(reqs)
    wall = time.perf_counter() - wall0

    summary = metrics.summarize(reqs)
    summary["sim_wall_time_s"] = wall
    virtual = summary["duration_s"]
    summary["speedup_vs_realtime"] = virtual / wall if wall > 0 else float("inf")
    metrics.print_table(summary)
    print(f"sim wall time             : {wall:.3f} s ({summary['speedup_vs_realtime']:.0f}x real time)")
    if args.report_json:
        metrics.write_json(summary, args.report_json)
        print(f"report written            : {args.report_json}")


if __name__ == "__main__":
    main()

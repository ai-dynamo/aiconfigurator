# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-config Rust-vs-Python engine-step drift map.

Throwaway diagnostic: sweeps batch / seq / tp corners across representative
architectures (dense / MoE / MLA / hybrid), agg + disagg, and reports every
(model, mode, config) where |drift| > 1% for ttft or tpot. Single-point
cli_estimate per backend, same as the parity probe but over many configs, to
enumerate the regions the single-config probe (bs=16, isl/osl=256) masks.
"""

import logging

logging.disable(logging.CRITICAL)
from aiconfigurator.cli.api import cli_estimate

# (model, system, backend, version, family)
COMBOS = [
    ("meta-llama/Llama-3.1-8B", "gb200", "vllm", "0.19.0", "dense-sm"),
    ("meta-llama/Meta-Llama-3.1-70B", "h200_sxm", "trtllm", "1.0.0", "dense-lg"),
    ("Qwen/Qwen3-30B-A3B", "gb200", "vllm", "0.19.0", "moe"),
    ("deepseek-ai/DeepSeek-R1", "b200_sxm", "sglang", "0.5.10", "mla-moe"),
    ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "gb300", "sglang", "0.5.10", "hybrid"),
    ("openai/gpt-oss-120b", "b200_sxm", "vllm", "0.19.0", "moe-oss"),
]
SEQS = [(256, 256), (4096, 512), (512, 4096), (8192, 1024)]
TPS = [1, 4, 8]
AGG_BS = [1, 16, 64, 256]
DISAGG_BD = [(1, 1), (4, 16), (4, 64), (4, 256), (1, 256)]  # (prefill_bs, decode_bs)


def drift(a, b):
    return (b - a) / a * 100 if a else 0.0


def run(**kw):
    py = cli_estimate(engine_step_backend="python", **kw)
    ru = cli_estimate(engine_step_backend="rust", **kw)
    return drift(py.ttft, ru.ttft), drift(py.tpot, ru.tpot)


def main():
    hits = []  # (family, model, mode, desc, dt, dp)
    for model, system, backend, version, fam in COMBOS:
        base = dict(model_path=model, system_name=system, backend_name=backend, backend_version=version, prefix=0)
        for tp in TPS:
            for isl, osl in SEQS:
                # agg
                for bs in AGG_BS:
                    try:
                        dt, dp = run(
                            mode="agg",
                            isl=isl,
                            osl=osl,
                            tp_size=tp,
                            moe_tp_size=1,
                            moe_ep_size=tp,
                            batch_size=bs,
                            **base,
                        )
                        if max(abs(dt), abs(dp)) > 1.0:
                            hits.append((fam, model, "agg", f"tp={tp} isl={isl} osl={osl} bs={bs}", dt, dp))
                    except Exception:
                        pass
                # disagg
                for pbs, dbs in DISAGG_BD:
                    try:
                        dt, dp = run(
                            mode="disagg",
                            isl=isl,
                            osl=osl,
                            tp_size=tp,
                            moe_tp_size=1,
                            moe_ep_size=tp,
                            prefill_batch_size=pbs,
                            prefill_num_workers=1,
                            decode_batch_size=dbs,
                            decode_num_workers=1,
                            **base,
                        )
                        if max(abs(dt), abs(dp)) > 1.0:
                            hits.append(
                                (fam, model, "disagg", f"tp={tp} isl={isl} osl={osl} pbs={pbs} dbs={dbs}", dt, dp)
                            )
                    except Exception:
                        pass
        print(f"[done] {fam:10s} {model}", flush=True)

    print("\n==== DRIFT HITS (|ttft| or |tpot| > 1%) ====")
    for fam, model, mode, desc, dt, dp in hits:
        print(f"  {fam:10s} {mode:6s} {desc:42s} ttft {dt:+6.2f}%  tpot {dp:+6.2f}%")

    # bucket summary
    print("\n==== SUMMARY by (mode, phase, batch-bucket) ====")
    from collections import defaultdict

    def _bucket(b):
        return "bs=1" if b == 1 else ("bs<=16" if b <= 16 else ("bs<=64" if b <= 64 else "bs>=256"))

    buckets = defaultdict(int)
    for fam, model, mode, desc, dt, dp in hits:
        # Bucket each phase on ITS OWN batch: ttft is prefill (bs / pbs), tpot is
        # decode (bs / dbs). desc is "...bs=N" (agg) or "...pbs=P dbs=D" (disagg).
        toks = dict(t.split("=") for t in desc.split() if "=" in t)
        prefill_b = int(toks.get("pbs", toks.get("bs", 0)))
        decode_b = int(toks.get("dbs", toks.get("bs", 0)))
        if abs(dt) > 1.0:
            buckets[(mode, "ttft", _bucket(prefill_b))] += 1
        if abs(dp) > 1.0:
            buckets[(mode, "tpot", _bucket(decode_b))] += 1
    for k in sorted(buckets):
        print(f"  {k[0]:6s} {k[1]:4s} {k[2]:8s}: {buckets[k]} configs")
    print(f"\nTOTAL HITS: {len(hits)}")


if __name__ == "__main__":
    main()

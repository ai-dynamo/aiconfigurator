"""Does util (the 'scale') depend on quant? Measure median util per quant per op.

util = SOL/measured (db SOL vs SILICON) over a shared shape grid. If util levels
differ across quant modes, the scale must be defined per-quant (it already is,
since quant is part of the slice key).
"""

import numpy as np

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import get_database

SOL, SIL = common.DatabaseMode.SOL, common.DatabaseMode.SILICON


def med_util(label, fn, shapes):
    us = []
    for sh in shapes:
        try:
            sol, sil = fn(SOL, *sh), fn(SIL, *sh)
            if sol > 0 and sil > 0:
                us.append(sol / sil)
        except Exception:
            pass
    if us:
        us = np.array(us)
        med = np.median(us)
        # MAPE if we used ONE scalar (the median util) for the whole slice:
        single_scale_mape = np.abs(us / med - 1).mean() * 100
        print(
            f"    {label:<14} n={len(us):>3}  util median {med:.3f}  [{us.min():.3f},{us.max():.3f}]"
            f"   single-scale MAPE {single_scale_mape:5.1f}%"
        )


def main():
    db = get_database("h100_sxm", "trtllm", "1.3.0rc10")

    print("GEMM (m x n x k):")
    gm = common.GEMMQuantMode
    gemm_shapes = [(m, n, k) for m in (16, 256, 4096) for n in (4096, 8192) for k in (4096, 8192)]
    for q in [gm.bfloat16, gm.fp8, gm.fp8_block]:
        med_util(q.name, lambda mode, m, n, k, _q=q: float(db.query_gemm(m, n, k, _q, mode)), gemm_shapes)

    print("\nContextAttention (n, s, b), MHA head_dim=128:")
    kvm, fm = common.KVCacheQuantMode, common.FMHAQuantMode
    attn_shapes = [(n, s, b) for n in (8, 32) for s in (256, 1024, 4096) for b in (1, 8, 64)]
    for fq, kq in [(fm.bfloat16, kvm.bfloat16), (fm.fp8, kvm.fp8)]:
        med_util(
            f"fmha={fq.name}",
            lambda mode, n, s, b, _f=fq, _k=kq: float(
                db.query_context_attention(b, s, 0, n, n, _k, _f, mode, window_size=0, head_size=128)
            ),
            attn_shapes,
        )

    print("\nMoE (num_tokens) @ a fixed config (h=4096,i=1536,topk=8,e=128,tp1,ep1):")
    mqm = common.MoEQuantMode
    moe_shapes = [(t,) for t in (1, 8, 64, 512, 4096)]
    for q in [mqm.bfloat16, mqm.fp8]:
        med_util(
            q.name,
            lambda mode, t, _q=q: float(db.query_moe(t, 4096, 1536, 8, 128, 1, 1, _q, "power_law_1.2", mode)),
            moe_shapes,
        )


if __name__ == "__main__":
    main()

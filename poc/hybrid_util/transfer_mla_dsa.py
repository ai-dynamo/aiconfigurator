"""MLA / DSA within-op transferability across num_heads (#2, like attention).

Coordinate = workload (s, b) at prefix=0; num_heads is the transferred axis.
util(n, s, b) = SOL_op / silicon_op (both from the db: SOL vs SILICON mode).
Leave-one-num_heads-out: predict target n's silicon from the NEAREST other n's
util:  pred = SOL_op(n_t) / util_op(n_src)  over matched (s, b).
Tells us whether num_heads transfers for MLA / DSA (it does for attention near).
"""

import numpy as np

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import get_database

SOL = common.DatabaseMode.SOL
SIL = common.DatabaseMode.SILICON
BF = common.KVCacheQuantMode.bfloat16
FBF = common.FMHAQuantMode.bfloat16
GBF = common.GEMMQuantMode.bfloat16
SB = [(s, b) for s in (256, 1024, 4096) for b in (1, 4, 16, 64)]


def study(name, fn, heads):
    db_util = {}  # n -> {(s,b): (sol, sil)}
    for n in heads:
        d = {}
        for s, b in SB:
            try:
                sol, sil = fn(SOL, n, s, b), fn(SIL, n, s, b)
                if sol > 0 and sil > 0:
                    d[(s, b)] = (sol, sil)
            except Exception:
                pass
        if d:
            db_util[n] = d
    avail = sorted(db_util)
    print(f"\n== {name} == num_heads with data: {avail}")
    all_ape, all_const = [], []
    for nt in avail:
        # nearest other head count in log space
        others = [n for n in avail if n != nt]
        if not others:
            continue
        nsrc = min(others, key=lambda n: abs(np.log(n) - np.log(nt)))
        shared = set(db_util[nt]) & set(db_util[nsrc])
        ape = []
        for sh in shared:
            sol_t, sil_t = db_util[nt][sh]
            sol_s, sil_s = db_util[nsrc][sh]
            util_src = sol_s / sil_s
            pred = sol_t / util_src
            ape.append(abs(pred - sil_t) / sil_t)
        # const baseline (use the op's own scale: mla ctx 0.6, dsa 0.5)
        const_apes = [abs(sol / 0.6 - sil) / sil for sol, sil in db_util[nt].values()]
        all_ape += ape
        all_const += const_apes
        if ape:
            tr, cn = np.mean(ape) * 100, np.mean(const_apes) * 100
            print(f"  heads {nt:>3} <- {nsrc:<3} : transfer {tr:6.1f}%   (const {cn:5.0f}%)")
    if all_ape:
        mape, med, cn = np.mean(all_ape) * 100, np.median(all_ape) * 100, np.mean(all_const) * 100
        print(f"  OVERALL transfer MAPE {mape:.1f}%  median {med:.1f}%  |  const {cn:.0f}%")


def main():
    db = get_database("b200_sxm", "trtllm", "1.3.0rc10")
    study("ContextMLA", lambda m, n, s, b: float(db.query_context_mla(b, s, 0, n, BF, FBF, m)), [8, 16, 32, 64, 128])
    study(
        "ContextDSAModule",
        lambda m, n, s, b: float(
            db.query_context_dsa_module(b, s, n, BF, FBF, GBF, m, prefix=0, architecture="GlmMoeDsaForCausalLM")
        ),
        [8, 16, 32, 64, 128],
    )


if __name__ == "__main__":
    main()

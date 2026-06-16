"""Cross-OP util transfer feasibility (observation 4).

Can a NEW op with no data borrow a *similar* op's util at the same workload?
Test within the context-attention family (shared coord (num_heads, s, b),
prefix=0, bf16): ContextAttention (MHA), ContextMLA, ContextDSAModule.

util_op = SOL_op / silicon_op   (both pulled from the db: SOL vs SILICON mode)
transfer A->B:  pred_B = SOL_B / util_A = SOL_B * silicon_A / SOL_A
If util is ~op-invariant, transfer is tight -> a new attention-family op can
borrow a sibling op's util. SOL carries the per-op structural difference.
"""

import numpy as np

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import get_database

SOL = common.DatabaseMode.SOL
SIL = common.DatabaseMode.SILICON
BF = common.KVCacheQuantMode.bfloat16
FBF = common.FMHAQuantMode.bfloat16
GBF = common.GEMMQuantMode.bfloat16


def main():
    db = get_database("b200_sxm", "trtllm", "1.3.0rc10")

    def attn(mode, n, s, b):
        return float(db.query_context_attention(b, s, 0, n, n, BF, FBF, mode, window_size=0, head_size=128))

    def mla(mode, n, s, b):
        return float(db.query_context_mla(b, s, 0, n, BF, FBF, mode))

    def dsa(mode, n, s, b):
        return float(
            db.query_context_dsa_module(b, s, n, BF, FBF, GBF, mode, prefix=0, architecture="GlmMoeDsaForCausalLM")
        )

    ops_map = {"attn": attn, "mla": mla, "dsa": dsa}
    grid_pts = [(n, s, b) for n in (8, 16, 32, 64) for s in (256, 1024, 4096) for b in (1, 4, 16, 64)]

    # collect SOL + silicon + util per op over the shared grid (skip data gaps)
    data = {k: {} for k in ops_map}  # op -> {shape: (sol, sil, util)}
    for name, fn in ops_map.items():
        for sh in grid_pts:
            try:
                sol, sil = fn(SOL, *sh), fn(SIL, *sh)
                if sol > 0 and sil > 0:
                    data[name][sh] = (sol, sil, sol / sil)
            except Exception:
                pass
        us = np.array([v[2] for v in data[name].values()])
        print(f"  {name:>5}: {len(us)} pts, util med {np.median(us):.3f}  [{us.min():.3f}, {us.max():.3f}]")

    print("\n  cross-op transfer  (A util -> predict B silicon), MAPE% over shared shapes")
    print(f"  {'src/dst':>8}" + "".join(f"{b:>9}" for b in ops_map))
    for a in ops_map:
        row = f"  {a:>8}"
        for b in ops_map:
            shared = set(data[a]) & set(data[b])
            if a == b or not shared:
                row += f"{'-':>9}"
                continue
            apes = []
            for sh in shared:
                sol_a, _, util_a = data[a][sh]
                sol_b, sil_b, _ = data[b][sh]
                pred = sol_b / util_a
                apes.append(abs(pred - sil_b) / sil_b)
            row += f"{np.mean(apes) * 100:>8.1f} "
        print(row)


if __name__ == "__main__":
    main()

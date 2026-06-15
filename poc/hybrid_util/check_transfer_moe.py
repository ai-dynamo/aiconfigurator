"""Leave-one-config-out validation of the PRODUCTION cross-shape transfer
(util_empirical.grid_from_reference + ReferenceCandidate + UtilGrid).

For each target MoE config, build reference candidates from the OTHER collected
configs (target excluded), let grid_from_reference pick the nearest sibling and
borrow its util curve, reconstruct latency = SOL_target(nt)/util_ref(nt), and
compare to the target's REAL silicon. This is the production path a new model's
unseen MoE shape would hit. Baseline = fixed SOL/0.4.
"""

import numpy as np
import pandas as pd

from aiconfigurator.sdk.operations import util_empirical as ue

H100 = {"mem_bw": 3350e9, "bf16": 989e12}
QUANT = {"memory": 1, "compute": 2}  # fp8
PATH = "src/aiconfigurator/systems/data/h100_sxm/trtllm/1.3.0rc10/moe_perf.parquet"


def sol_ms(nt, hidden, inter, topk, experts, num_gemms=3):
    total = nt * topk
    ops = total * hidden * inter * num_gemms * 2
    mem = QUANT["memory"] * (
        total * hidden * 2 + total * inter * num_gemms + hidden * inter * num_gemms * min(experts, total)
    )
    return max(ops / (H100["bf16"] * QUANT["compute"]) * 1000.0, mem / H100["mem_bw"] * 1000.0)


def main():
    df = pd.read_parquet(PATH)
    sl = df[
        (df.moe_dtype == "fp8") & (df.moe_tp_size == 1) & (df.moe_ep_size == 1) & (df.distribution == "power_law_1.2")
    ]
    configs = {}  # (topk,experts,hidden,inter) -> {num_tokens: latency}
    for (tk, ne, h, i), g in sl.groupby(["topk", "num_experts", "hidden_size", "inter_size"]):
        configs[(tk, ne, h, i)] = g.drop_duplicates("num_tokens").set_index("num_tokens")["latency"].to_dict()

    print(f"{len(configs)} collected fp8/tp1/ep1 MoE configs\n")
    print(f"  {'target (topk,exp,hid,int)':<28}{'->nearest sibling':<26}{'transfer%':>10}{'const%':>9}")
    tr_all, const_all = [], []
    for tgt, curve in configs.items():
        tk, ne, h, i = tgt

        # build candidates from every OTHER config (leave-one-out)
        def _cands(_tgt=tgt):
            out = []
            for (ctk, cne, ch, ci), ccurve in configs.items():
                if (ctk, cne, ch, ci) == _tgt:
                    continue
                node = {nt: {"latency": lat} for nt, lat in ccurve.items()}
                out.append(
                    ue.ReferenceCandidate(
                        features=(ctk, cne, ch, ci),
                        node=node,
                        sol_fn=(lambda c, _h=ch, _i=ci, _t=ctk, _e=cne: sol_ms(c[0], _h, _i, _t, _e)),
                    )
                )
            return out

        ue._GRID_CACHE.clear()
        grid = ue.grid_from_reference(("loo", tgt), (tk, ne, h, i), _cands, depth=1)
        # find which sibling was picked (for display)
        picked = ue._nearest_candidate((tk, ne, h, i), _cands()).features
        tr, const = [], []
        for nt, actual in curve.items():
            sol_q = sol_ms(nt, h, i, tk, ne)
            pred, _ = ue.estimate(sol_q, (nt,), grid, fallback_scale=0.4)
            tr.append(abs(pred - actual) / actual)
            const.append(abs(sol_q / 0.4 - actual) / actual)
        tr_all += tr
        const_all += const
        print(f"  {tgt!s:<28}{picked!s:<26}{np.mean(tr) * 100:>9.1f}{np.mean(const) * 100:>9.1f}")

    print(
        f"\n  OVERALL transfer MAPE={np.mean(tr_all) * 100:.1f}%  median={np.median(tr_all) * 100:.1f}%  "
        f"|  const MAPE={np.mean(const_all) * 100:.1f}%"
    )


if __name__ == "__main__":
    main()

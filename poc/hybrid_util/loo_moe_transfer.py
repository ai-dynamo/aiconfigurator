"""Leave-one-out accuracy of the MoE cross-shape transfer (the path M3 relies on).

For every collected MoE config (fp8_block, tp1/ep8, b200/trtllm 1.3.0rc10):
  1. ground truth = its real silicon latency at each collected num_tokens point;
  2. transfer estimate = drop this config from the candidate pool, force EMPIRICAL,
     so it borrows the nearest OTHER config's util curve re-scaled by its own SOL;
  3. error = |transfer - silicon| / silicon.

This quantifies how much the (topk, experts, hidden, inter) util-transfer hypothesis
costs — the same mechanism that estimates M3's unseen (topk4, 128, 6144/3072) MoE.
"""

import logging

logging.disable(logging.WARNING)

import numpy as np

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import util_empirical
from aiconfigurator.sdk.operations.moe import MoE
from aiconfigurator.sdk.perf_database import get_database

QM = common.MoEQuantMode.fp8_block
WL = "power_law_1.01"
TP, EP = 1, 8


def latency_of(node, tok):
    pt = node[tok]
    return float(pt["latency"]) if isinstance(pt, dict) else float(pt)


def main():
    db = get_database("b200_sxm", "trtllm", "1.3.0rc10")
    MoE.load_data(db)
    wl_data = db._moe_data[QM][WL]

    configs = []
    for tk in wl_data:
        for ne in wl_data[tk]:
            for hs in wl_data[tk][ne]:
                for isz in wl_data[tk][ne][hs]:
                    node = wl_data[tk][ne][hs].get(isz, {}).get(TP, {}).get(EP)
                    if node:
                        configs.append((tk, ne, hs, isz, node))

    print(f"{'held-out (tk,exp,h,i)':>26} {'borrowed-from':>22} {'MAPE':>7} {'pts'}")
    all_err = []
    for tk, ne, hs, isz, node in configs:
        parent = wl_data[tk][ne][hs][isz][TP]
        saved = parent.pop(EP)  # drop this config from the pool

        # nearest remaining neighbor (for reporting) in normalized-log space
        others = [(t, n, h, i) for (t, n, h, i, _nd) in configs if (t, n, h, i) != (tk, ne, hs, isz)]
        feats = np.log(np.array([(tk, ne, hs, isz)] + others, dtype=float))
        mu, sd = feats.mean(0), feats.std(0)
        sd[sd == 0] = 1
        norm = (feats - mu) / sd
        d = np.linalg.norm(norm[1:] - norm[0], axis=1)
        borrowed = others[int(d.argmin())]

        toks = sorted(saved.keys())
        sample = toks[:: max(1, len(toks) // 6)]
        tok_decode = min(toks, key=lambda t: abs(t - 64))  # M3 decode: MoE sees ~bs=64 tokens
        errs, err_small = [], None
        for tok in [tok_decode] + sample:
            util_empirical._GRID_CACHE.clear()
            if hasattr(db, "clear_runtime_caches"):
                db.clear_runtime_caches()
            est = float(
                db.query_moe(
                    num_tokens=tok,
                    hidden_size=hs,
                    inter_size=isz,
                    topk=tk,
                    num_experts=ne,
                    moe_tp_size=TP,
                    moe_ep_size=EP,
                    quant_mode=QM,
                    workload_distribution=WL,
                    is_context=True,
                    database_mode=common.DatabaseMode.EMPIRICAL,
                )
            )
            e = abs(est - latency_of(saved, tok)) / latency_of(saved, tok)
            if err_small is None:
                err_small = 100 * e  # first iter is toks[0]
            else:
                errs.append(e)

        parent[EP] = saved  # restore
        mape = 100 * float(np.mean(errs))
        all_err.append((mape, err_small, (tk, ne, hs, isz), borrowed))
        print(f"{(tk, ne, hs, isz)!s:>26} {borrowed!s:>22} {mape:>6.1f}% {err_small:>9.1f}% {len(sample)}")

    mapes = [a[0] for a in all_err]
    smalls = [a[1] for a in all_err]
    print(
        f"\nMoE cross-shape transfer LOO MAPE (all tokens): mean {np.mean(mapes):.1f}%  "
        f"median {np.median(mapes):.1f}%  max {np.max(mapes):.1f}%"
    )
    print(f"  at small num_tokens (decode regime ~64): mean {np.mean(smalls):.1f}%  median {np.median(smalls):.1f}%")
    print(
        "(M3 borrows topk6/256/4096/2048 for its topk4/128/6144/3072 MoE — the largest extrapolation; "
        "topk=4 has no same-topk neighbour in the pool at all.)"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{a[2][0]}/{a[2][1]}/{a[2][2]}/{a[2][3]}" for a in all_err]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.bar(x - 0.2, mapes, 0.4, label="MAPE (all num_tokens)", color="#4fa3ff")
    ax.bar(x + 0.2, smalls, 0.4, label="error at small tokens (~64, decode-like)", color="#bc8cff")
    ax.axhline(np.median(mapes), color="#d29922", ls="--", lw=1, label=f"median {np.median(mapes):.0f}%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("transfer error (%)")
    ax.set_title("MoE cross-shape transfer leave-one-out error — b200/trtllm 1.3.0rc10, fp8_block, tp1/ep8")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    import os

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loo_moe_transfer.png")
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()

"""Can we say anything about M3's MoE error without M3 silicon?

We can't measure it directly (no topk4/128/6144/3072 silicon). But we CAN measure
transfer error on the 8 configs that DO have silicon (leave-one-out), and relate it
to the only thing we know about M3's MoE: how far its nearest neighbour is in the
(topk, experts, hidden, inter) normalized-log feature space.

Finding: error vs neighbour-distance correlates 0.85 — but M3's distance (4.07) is
BEYOND every validated case (max 2.52). So M3's MoE transfer is an out-of-support
extrapolation: the fit says ~88%, which is itself unreliable. Honest statement:
M3's MoE term has no validation support and likely errs >=50%. Fix = collect a
closer MoE point (topk4 / 128 experts).
"""

import logging

logging.disable(logging.WARNING)

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import util_empirical
from aiconfigurator.sdk.operations.moe import MoE
from aiconfigurator.sdk.perf_database import get_database

QM = common.MoEQuantMode.fp8_block
WL = "power_law_1.01"
TP, EP = 1, 8
M3 = (4, 128, 6144, 3072)


def main():
    db = get_database("b200_sxm", "trtllm", "1.3.0rc10")
    MoE.load_data(db)
    wl = db._moe_data[QM][WL]
    cfgs = []
    for tk in wl:
        for ne in wl[tk]:
            for hs in wl[tk][ne]:
                for isz in wl[tk][ne][hs]:
                    node = wl[tk][ne][hs].get(isz, {}).get(TP, {}).get(EP)
                    if node:
                        cfgs.append((tk, ne, hs, isz, node))

    feat = lambda c: np.log(np.array(c[:4], dtype=float))
    farr = np.array([feat(c) for c in cfgs])
    mu, sd = farr.mean(0), farr.std(0)
    sd[sd == 0] = 1

    def nn_dist(q):
        qn = (np.log(np.array(q, dtype=float)) - mu) / sd
        ds = [np.linalg.norm((feat(c) - mu) / sd - qn) for c in cfgs if c[:4] != tuple(q)]
        return min(ds)

    def loo_err(c):
        tk, ne, hs, isz, _ = c
        parent = wl[tk][ne][hs][isz][TP]
        saved = parent.pop(EP)
        toks = sorted(saved.keys())
        tok = min(toks, key=lambda t: abs(t - 64))
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
        truth = saved[tok]
        truth = float(truth["latency"] if isinstance(truth, dict) else truth)
        parent[EP] = saved
        return 100 * abs(est - truth) / truth

    dist = np.array([nn_dist(c[:4]) for c in cfgs])
    err = np.array([loo_err(c) for c in cfgs])
    m3d = nn_dist(M3)
    a, b = np.polyfit(dist, err, 1)
    r = np.corrcoef(dist, err)[0, 1]
    print(f"corr(dist,err)={r:.2f}  fit err≈{a:.1f}*d+{b:.1f}  M3 dist={m3d:.2f} -> ~{a * m3d + b:.0f}% (extrapolated)")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.scatter(dist, err, s=55, color="#4fa3ff", zorder=3, label="LOO configs (have silicon)")
    xs = np.linspace(0, m3d + 0.4, 50)
    ax.plot(xs, a * xs + b, color="#d29922", ls="--", lw=1.3, label=f"fit (r={r:.2f})")
    ax.axvspan(dist.max(), m3d + 0.5, color="#f85149", alpha=0.08)
    ax.axvline(m3d, color="#f85149", lw=1.5, label=f"M3 nearest-neighbour dist = {m3d:.1f} (no validation here)")
    ax.scatter(
        [m3d],
        [a * m3d + b],
        marker="*",
        s=240,
        color="#bc8cff",
        zorder=4,
        label=f"M3 extrapolated ≈ {a * m3d + b:.0f}% (unreliable)",
    )
    for c, d, e in zip(cfgs, dist, err, strict=False):
        ax.annotate(
            f"{c[0]}/{c[1]}/{c[2]}/{c[3]}",
            (d, e),
            fontsize=7,
            color="#9aa7b4",
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set(
        xlabel="nearest-neighbour distance (normalized-log feature space)",
        ylabel="decode MoE transfer error (%)",
        title="MoE transfer error vs neighbour distance — M3 is beyond the validated range",
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loo_moe_calibration.png")
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()

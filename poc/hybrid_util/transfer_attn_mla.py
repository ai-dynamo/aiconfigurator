"""Cross-op util transfer, narrowed to ContextAttention <-> ContextMLA, with a
breakdown of WHERE the ~47% error comes from.

util_op = SOL_op / silicon_op (db SOL vs SILICON). For matched (num_heads, s, b),
prefix=0, bf16. We test:
  raw transfer  attn->mla:  pred = SOL_mla / util_attn
  ratio-corrected:          pred = SOL_mla / (util_attn * k),  k = median(util_mla/util_attn)
If the ratio-corrected error is small, the cross-op gap is mostly a SYSTEMATIC
util-level offset (MLA runs at higher SOL-utilisation than MHA), not shape noise.

Outputs a 2x2 figure (util levels, util ratio, error hist, error-by-heads) + stats.
"""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import get_database

SOL, SIL = common.DatabaseMode.SOL, common.DatabaseMode.SILICON
BF, FBF = common.KVCacheQuantMode.bfloat16, common.FMHAQuantMode.bfloat16
HEADS = [8, 16, 32, 64]
SB = [(s, b) for s in (256, 1024, 4096) for b in (1, 4, 16, 64)]


def main():
    db = get_database("b200_sxm", "trtllm", "1.3.0rc10")

    def attn(mode, n, s, b):
        return float(db.query_context_attention(b, s, 0, n, n, BF, FBF, mode, window_size=0, head_size=128))

    def mla(mode, n, s, b):
        return float(db.query_context_mla(b, s, 0, n, BF, FBF, mode))

    # per-point: util_attn, util_mla, sol_attn, sol_mla
    rec = []  # (n, s, b, util_attn, util_mla, sol_attn, sol_mla, sil_attn, sil_mla)
    for n in HEADS:
        for s, b in SB:
            try:
                sa, ma = attn(SOL, n, s, b), attn(SIL, n, s, b)
                sm, mm = mla(SOL, n, s, b), mla(SIL, n, s, b)
                if min(sa, ma, sm, mm) > 0:
                    rec.append((n, s, b, sa / ma, sm / mm, sa, sm, ma, mm))
            except Exception:
                pass

    arr = np.array(rec, dtype=float)
    n_, util_a, util_m = arr[:, 0], arr[:, 3], arr[:, 4]
    sol_m, sil_m = arr[:, 6], arr[:, 8]
    ratio = util_m / util_a
    k = np.median(ratio)

    # transfer attn->mla, raw vs ratio-corrected (signed %)
    err_raw = (sol_m / util_a - sil_m) / sil_m * 100
    err_cor = (sol_m / (util_a * k) - sil_m) / sil_m * 100

    print(f"  n points: {len(arr)}")
    print(f"  util_attn median {np.median(util_a):.3f}   util_mla median {np.median(util_m):.3f}")
    print(f"  util_mla / util_attn:  median k={k:.2f}   range [{ratio.min():.2f}, {ratio.max():.2f}]")
    raw_m, cor_m = np.abs(err_raw).mean(), np.abs(err_cor).mean()
    print(f"  attn->mla MAPE: raw {raw_m:.1f}%  ->  k-corrected {cor_m:.1f}%")

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))

    # (1) util level by num_heads (grouped bar)
    a = ax[0][0]
    ua = [util_a[n_ == h].mean() for h in HEADS]
    um = [util_m[n_ == h].mean() for h in HEADS]
    x = np.arange(len(HEADS))
    a.bar(x - 0.2, ua, 0.4, label="attn", color="#4fa3ff")
    a.bar(x + 0.2, um, 0.4, label="mla", color="#bc8cff")
    a.set_xticks(x)
    a.set_xticklabels(HEADS)
    a.set_xlabel("num_heads")
    a.set_ylabel("mean util")
    a.set_title("util level: MLA consistently higher than MHA")
    a.legend()

    # (2) util ratio histogram (is it ~constant?)
    a = ax[0][1]
    a.hist(ratio, bins=15, color="#3fb950", edgecolor="#10151b")
    a.axvline(k, color="#f85149", ls="--", lw=1.6, label=f"median k={k:.2f}")
    a.set_xlabel("util_mla / util_attn")
    a.set_ylabel("count")
    a.set_title("util ratio (tight band => systematic offset)")
    a.legend()

    # (3) transfer error histogram: raw vs corrected
    a = ax[1][0]
    bins = np.linspace(-100, 100, 41)
    a.hist(
        np.clip(err_raw, -100, 100),
        bins=bins,
        alpha=0.6,
        color="#f85149",
        label=f"raw (MAPE {np.abs(err_raw).mean():.0f}%)",
    )
    a.hist(
        np.clip(err_cor, -100, 100),
        bins=bins,
        alpha=0.7,
        color="#3fb950",
        label=f"ratio-corrected (MAPE {np.abs(err_cor).mean():.0f}%)",
    )
    a.axvline(0, color="#9aa7b4", lw=1)
    a.set_xlabel("attn->mla signed transfer error (%)")
    a.set_ylabel("count")
    a.set_title("error: raw vs after constant-ratio correction")
    a.legend()

    # (4) mean |error| by num_heads: raw vs corrected
    a = ax[1][1]
    er = [np.abs(err_raw[n_ == h]).mean() for h in HEADS]
    ec = [np.abs(err_cor[n_ == h]).mean() for h in HEADS]
    a.bar(x - 0.2, er, 0.4, label="raw", color="#f85149")
    a.bar(x + 0.2, ec, 0.4, label="ratio-corrected", color="#3fb950")
    a.set_xticks(x)
    a.set_xticklabels(HEADS)
    a.set_xlabel("num_heads")
    a.set_ylabel("MAPE %")
    a.set_title("where the error lives (by num_heads)")
    a.legend()

    fig.suptitle("ContextAttention <-> ContextMLA cross-op util transfer (b200/trtllm, prefix=0, bf16)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transfer_attn_mla.png")
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()

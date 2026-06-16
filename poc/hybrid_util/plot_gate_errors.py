"""Plot empirical-vs-silicon SIGNED error distribution per model gate.

Error = (emp - sil)/sil, signed: >0 over-predict, <0 under-predict. x-axis fixed
to [-100, 100]%.
Top row:    histogram of the signed error (distribution + bias direction).
Bottom row: every point's signed error, sorted (tail / outliers).

Reuses run_static_compare.collect() so the numbers match the gate exactly.
"""

import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_static_compare import collect

MODELS = ["llama70b", "glm5", "deepseek_v3"]


def main():
    results = {m: collect(m) for m in MODELS}
    fig, axes = plt.subplots(2, len(MODELS), figsize=(5 * len(MODELS), 7.2))

    bins = np.linspace(-100, 100, 41)  # 5%-wide bins across [-100, 100]
    for j, m in enumerate(MODELS):
        r = results[m]
        e = np.array([a for a, *_ in r["apes"]]) * 100.0  # SIGNED error %
        mape, bias, mx = np.abs(e).mean(), e.mean(), np.abs(e).max()

        # --- histogram (signed distribution) ---
        ax = axes[0][j]
        ax.hist(np.clip(e, -100, 100), bins=bins, color="#4fa3ff", edgecolor="#10151b")
        ax.axvline(0, color="#9aa7b4", lw=1)
        ax.axvline(bias, color="#f85149", ls="--", lw=1.6, label=f"bias {bias:+.1f}%")
        ax.set_xlim(-100, 100)
        ax.set_title(f"{m}  (n={len(e)}, MAPE {mape:.1f}%)", fontsize=11)
        ax.set_xlabel("(emp - sil) / sil  (%)")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)

        # --- per-point signed error, sorted ---
        ax2 = axes[1][j]
        se = np.sort(e)
        ax2.bar(range(len(se)), se, width=1.0, color="#bc8cff")
        ax2.axhline(0, color="#9aa7b4", lw=1)
        ax2.set_ylim(-100, 100)
        ax2.set_title(f"every point sorted · max|e| {mx:.1f}%", fontsize=11)
        ax2.set_xlabel("point (sorted by signed error)")
        ax2.set_ylabel("(emp - sil)/sil  %")

    fig.suptitle("empirical-vs-silicon SIGNED error distribution (util-empirical, pastkv=0)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gate_error_dist.png")
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()

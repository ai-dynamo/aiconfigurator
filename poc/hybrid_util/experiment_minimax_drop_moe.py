"""Drop MiniMax's own MoE slice, then compare run_static SILICON vs EMPIRICAL.

Simulates "MiniMax is a new model we have NO MoE data for". Ground truth =
SILICON with the data present. Then we delete MiniMax's MoE config
(hidden=3072, inter=1536, topk=8, experts=256) from the loaded table and run
EMPIRICAL -- its MoE op now has no own data, so it transfers from the nearest
sibling MoE config (cross-shape transfer). Other ops use their own util.

Shared layer is OFF (db built without database_mode) so the deleted slice is not
refilled from another version -- isolates the cross-shape transfer.

Series:
  empirical_full  : empirical, data present (baseline)
  empirical_drop  : empirical, MiniMax MoE slice removed -> MoE transfers
both compared to silicon_full (truth). Outputs signed-error histograms + stats.
"""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.operations import util_empirical
from aiconfigurator.sdk.perf_database import get_database

MODEL = "MiniMaxAI/MiniMax-M2.5"
SYS, BK, VER = "h100_sxm", "trtllm", "1.3.0rc10"
CFG = dict(topk=8, experts=256, hidden=3072, inter=1536)  # MiniMax MoE slice to drop
BS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SEQ = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]


def drop_minimax_moe(db):
    """Delete the (topk,experts,hidden,inter) MoE config from every quant/dist."""
    removed = 0
    moe = db._moe_data
    for q in list(moe):
        for wl in list(moe[q]):
            node = moe[q][wl]
            try:
                if CFG["inter"] in node[CFG["topk"]][CFG["experts"]][CFG["hidden"]]:
                    del node[CFG["topk"]][CFG["experts"]][CFG["hidden"]][CFG["inter"]]
                    removed += 1
            except (KeyError, TypeError):
                pass
    return removed


def run_grid(backend, model, db, mode):
    db.set_default_database_mode(mode)
    out = {}
    for bs in BS:
        for seq in SEQ:
            rc = config.RuntimeConfig(batch_size=bs, beam_width=1, isl=seq, osl=8, prefix=0)
            try:
                out[(bs, seq)] = backend.run_static_latency_only(model, db, rc, mode="static", stride=8)
            except Exception:
                pass
    return out


def main():
    db = get_database(SYS, BK, VER)  # shared layer OFF
    mc = config.ModelConfig(tp_size=8, moe_tp_size=1, moe_ep_size=8, overwrite_num_layers=8)
    model = get_model(MODEL, mc, backend_name=BK)
    backend = TRTLLMBackend()

    silicon = run_grid(backend, model, db, common.DatabaseMode.SILICON)
    emp_full = run_grid(backend, model, db, common.DatabaseMode.EMPIRICAL)
    n = drop_minimax_moe(db)
    util_empirical._GRID_CACHE.clear()  # else the pre-drop MoE util grid is reused
    db.clear_runtime_caches()  # else query_moe lru returns pre-drop results
    print(f"dropped MiniMax MoE slice from {n} quant/dist nodes")
    emp_drop = run_grid(backend, model, db, common.DatabaseMode.EMPIRICAL)

    def signed_err(pred):
        e = []
        for k, sil in silicon.items():
            if k in pred and sil > 0 and pred[k] > 0:
                e.append((pred[k] - sil) / sil * 100)
        return np.array(e)

    ef, ed = signed_err(emp_full), signed_err(emp_drop)
    print(f"  scored points: silicon={len(silicon)}  emp_full={len(ef)}  emp_drop={len(ed)}")
    for name, e in [("empirical_full (MoE has data)", ef), ("empirical_drop (MoE transferred)", ed)]:
        print(
            f"  {name:<34} MAPE {np.abs(e).mean():5.1f}%  median|e| {np.median(np.abs(e)):5.1f}%  "
            f"p95 {np.percentile(np.abs(e), 95):5.1f}%  max {np.abs(e).max():5.1f}%  bias {e.mean():+5.1f}%"
        )

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    bins = np.linspace(-100, 100, 41)
    for a, e, title in [
        (ax[0], ef, "empirical_full (MoE data present)"),
        (ax[1], ed, "empirical_drop (MoE slice removed -> transfer)"),
    ]:
        a.hist(np.clip(e, -100, 100), bins=bins, color="#4fa3ff", edgecolor="#10151b")
        a.axvline(0, color="#9aa7b4", lw=1)
        a.axvline(e.mean(), color="#f85149", ls="--", lw=1.6, label=f"bias {e.mean():+.1f}%")
        a.set_xlim(-100, 100)
        a.set_title(f"{title}\nMAPE {np.abs(e).mean():.1f}% (n={len(e)})", fontsize=11)
        a.set_xlabel("(empirical - silicon) / silicon  (%)")
        a.set_ylabel("count")
        a.legend(fontsize=8)
    fig.suptitle("MiniMax-M2.5 run_static: empirical vs silicon, MoE data present vs dropped", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "minimax_drop_moe.png")
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()

"""MiniMax-M2.7 (SILICON) vs MiniMax-M3 (HYBRID) on B200, NVFP4.

M2.7 = MOE + full GQA attention, has silicon data -> SILICON. M3 = MoE + MSA
(sparse), no MSA data -> HYBRID (MSA transfers from DSA util). Same hardware
(b200/trtllm), tp8, NVFP4. Reports TTFT (prefill latency) and TPOT (one decode
step) across shapes, and a bar-chart figure for the design doc.
"""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import get_database

SYS, BK, VER = "b200_sxm", "trtllm", "1.3.0rc10"


def build():
    # fp8_block baseline: NVFP4 has no DSA silicon on b200, so M3's MSA transfer
    # would fall back to a constant under NVFP4. fp8_block has DSA data -> the
    # MSA transfer is real, and M2.7 has fp8_block silicon too. Apples-to-apples.
    db = get_database(SYS, BK, VER)
    be = TRTLLMBackend()
    q = dict(
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        moe_quant_mode=common.MoEQuantMode.fp8_block,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
    )
    m27 = get_model(
        "MiniMaxAI/MiniMax-M2.7", config.ModelConfig(tp_size=8, moe_tp_size=1, moe_ep_size=8, **q), backend_name=BK
    )
    m3 = get_model(
        "MiniMaxAI/MiniMax-M3", config.ModelConfig(tp_size=8, moe_tp_size=1, moe_ep_size=8, **q), backend_name=BK
    )
    return db, be, m27, m3


def ttft(be, model, db, mode, bs, isl):
    db.set_default_database_mode(mode)
    s = be.run_static(
        model,
        db,
        config.RuntimeConfig(batch_size=bs, beam_width=1, isl=isl, osl=2, prefix=0),
        mode="static_ctx",
        stride=1,
    )
    return sum(s.get_context_latency_dict().values())


def tpot(be, model, db, mode, bs, kv):
    db.set_default_database_mode(mode)
    s = be.run_static(
        model,
        db,
        config.RuntimeConfig(batch_size=bs, beam_width=1, isl=kv, osl=2, prefix=0),
        mode="static_gen",
        stride=1,
    )
    return sum(s.get_generation_latency_dict().values())


def main():
    db, be, m27, m3 = build()
    sil_m, hyb_m = common.DatabaseMode.SILICON, common.DatabaseMode.HYBRID

    print(f"{'shape':>18}{'M2.7(sil)':>12}{'M3(hyb)':>12}{'M3/M2.7':>9}")
    rows = []
    # TTFT: prefill latency at bs=1 over isl
    for isl in (1024, 2048, 4096, 8192):
        a, b = ttft(be, m27, db, sil_m, 1, isl), ttft(be, m3, db, hyb_m, 1, isl)
        rows.append((f"TTFT isl={isl}", a, b))
        print(f"{'TTFT isl=' + str(isl):>18}{a:>11.1f}{b:>11.1f}{b / a:>9.2f}")
    # TPOT: decode step at bs=64 over kv
    for kv in (2048, 4096, 8192, 16384):
        a, b = tpot(be, m27, db, sil_m, 64, kv), tpot(be, m3, db, hyb_m, 64, kv)
        rows.append((f"TPOT bs64 kv={kv}", a, b))
        print(f"{'TPOT kv=' + str(kv):>18}{a:>11.2f}{b:>11.2f}{b / a:>9.2f}")

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    ttft_rows = [r for r in rows if r[0].startswith("TTFT")]
    tpot_rows = [r for r in rows if r[0].startswith("TPOT")]
    for a, group, title, xl in [
        (ax[0], ttft_rows, "TTFT (prefill, bs=1)", "isl"),
        (ax[1], tpot_rows, "TPOT (decode step, bs=64)", "kv len"),
    ]:
        x = np.arange(len(group))
        labels = [r[0].split("=")[-1] for r in group]
        a.bar(x - 0.2, [r[1] for r in group], 0.4, label="M2.7 (silicon)", color="#4fa3ff")
        a.bar(x + 0.2, [r[2] for r in group], 0.4, label="M3 (hybrid, MSA)", color="#bc8cff")
        a.set_xticks(x)
        a.set_xticklabels(labels)
        a.set_xlabel(xl)
        a.set_ylabel("ms")
        a.set_title(title)
        a.legend(fontsize=8)
    fig.suptitle("MiniMax-M2.7 (silicon) vs M3 (hybrid) — B200, fp8_block, tp8", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_m27_m3.png")
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()

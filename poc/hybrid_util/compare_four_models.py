"""Four-model alignment: DeepSeek-V3.2, GLM-5, MiniMax-M2.7, MiniMax-M3 on B200.

Precision semantics of the sparse-attention module: the *matrix multiplies*
(projection / MoE GEMMs) are FP4 (nvfp4); the attention compute and the indexer
are BF16/FP8. So "nvfp4" here means gemm=nvfp4, moe=nvfp4, fmha=bf16, kv=fp8 —
exactly the `gemm_type=nvfp4, mla_dtype=bfloat16, kv_cache_dtype=fp8` rows.

Per-model policy: try nvfp4. If the model's sparse-attention module has NO nvfp4
rows for its architecture (the transfer/silicon lookup would bottom out at the
constant fallback), drop to fp8_block instead — which has real module data. The
choice is detected from the parquet, not hardcoded, and printed with its reason.

  DeepSeek-V3.2  DeepseekV32ForCausalLM  native DSA      SILICON
  GLM-5          GlmMoeDsaForCausalLM    native DSA      SILICON
  MiniMax-M2.7   MiniMaxM2ForCausalLM    GQA MoE (dense) SILICON
  MiniMax-M3     MiniMaxM3ForCausalLM    MSA -> GlmMoeDsa HYBRID (transfer)
"""

import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import get_database

SYS, BK, VER = "b200_sxm", "trtllm", "1.3.0rc10"
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
    "aiconfigurator",
    "systems",
    "data",
    SYS,
    BK,
    VER,
)


def nvfp4_module_available(arch):
    """True if the DSA sparse-attention module has nvfp4 (fp4-matmul) rows for `arch`."""
    p = os.path.join(DATA_DIR, "dsa_generation_module_perf.parquet")
    if not os.path.exists(p):
        return False
    df = pd.read_parquet(p)
    sub = df[(df["architecture"] == arch) & (df["gemm_type"] == "nvfp4")]
    return len(sub) > 0


def quant_for(model_name, dsa_arch_for_lookup):
    """Pick (label, quant-dict). dsa_arch_for_lookup=None => non-DSA model (nvfp4 always ok)."""
    use_nvfp4 = True
    reason = "non-DSA: gemm/moe nvfp4 + bf16 attn, all present"
    if dsa_arch_for_lookup is not None:
        if nvfp4_module_available(dsa_arch_for_lookup):
            reason = f"{dsa_arch_for_lookup}: nvfp4 module rows present -> real fp4 matmul"
        else:
            use_nvfp4 = False
            reason = f"{dsa_arch_for_lookup}: NO nvfp4 module rows -> would fall back to const, use fp8_block"
    if use_nvfp4:
        q = dict(
            gemm_quant_mode=common.GEMMQuantMode.nvfp4,
            moe_quant_mode=common.MoEQuantMode.nvfp4,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        )
        return "nvfp4", q, reason
    q = dict(
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        moe_quant_mode=common.MoEQuantMode.fp8_block,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
    )
    return "fp8_block", q, reason


# (display, model_name, dsa_arch_for_lookup). All run HYBRID: silicon where the
# exact bracket exists, real-util empirical fallback otherwise (never a hard raise).
MODELS = [
    ("DeepSeek-V3.2", "deepseek-ai/DeepSeek-V3.2", "DeepseekV32ForCausalLM", common.DatabaseMode.HYBRID),
    ("GLM-5", "zai-org/GLM-5", "GlmMoeDsaForCausalLM", common.DatabaseMode.HYBRID),
    ("MiniMax-M2.7", "MiniMaxAI/MiniMax-M2.7", None, common.DatabaseMode.HYBRID),
    ("MiniMax-M3", "MiniMaxAI/MiniMax-M3", "GlmMoeDsaForCausalLM", common.DatabaseMode.HYBRID),
]
COLORS = {"DeepSeek-V3.2": "#3fb950", "GLM-5": "#d29922", "MiniMax-M2.7": "#4fa3ff", "MiniMax-M3": "#bc8cff"}


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
    db = get_database(SYS, BK, VER)
    be = TRTLLMBackend()
    par = dict(tp_size=8, moe_tp_size=1, moe_ep_size=8)

    isls = (1024, 2048, 4096, 8192)
    kvs = (2048, 4096, 8192, 16384)
    rows = {}
    print(f"{'model':>15} {'quant':>10} {'mode':>8}  reason")
    for disp, name, dsa_arch, mode in MODELS:
        label, q, reason = quant_for(name, dsa_arch)
        model = get_model(name, config.ModelConfig(**par, **q), backend_name=BK)
        print(f"{disp:>15} {label:>10} {mode.name:>8}  {reason}")
        ttfts = [ttft(be, model, db, mode, 1, isl) for isl in isls]
        tpots = [tpot(be, model, db, mode, 64, kv) for kv in kvs]
        rows[disp] = (label, ttfts, tpots)

    print(f"\n=== TTFT (prefill, bs=1) ms ===\n{'isl':>8}" + "".join(f"{d:>16}" for d, *_ in MODELS))
    for i, isl in enumerate(isls):
        print(f"{isl:>8}" + "".join(f"{rows[d][1][i]:>16.1f}" for d, *_ in MODELS))
    print(f"\n=== TPOT (decode step, bs=64) ms ===\n{'kv':>8}" + "".join(f"{d:>16}" for d, *_ in MODELS))
    for i, kv in enumerate(kvs):
        print(f"{kv:>8}" + "".join(f"{rows[d][2][i]:>16.2f}" for d, *_ in MODELS))

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for disp, *_ in MODELS:
        label, ttfts, tpots = rows[disp]
        c = COLORS[disp]
        ax[0].plot(isls, ttfts, "-o", color=c, label=f"{disp} ({label})")
        ax[1].plot(kvs, tpots, "-o", color=c, label=f"{disp} ({label})")
    ax[0].set(title="TTFT (prefill, bs=1)", xlabel="isl", ylabel="ms")
    ax[1].set(title="TPOT (decode step, bs=64)", xlabel="kv length", ylabel="ms")
    for a in ax:
        a.legend(fontsize=8)
        a.grid(alpha=0.2)
    fig.suptitle(
        f"DeepSeek-V3.2 / GLM-5 / MiniMax-M2.7 / M3 — {SYS}, {BK} {VER}, tp8 "
        "(nvfp4 where module data exists, else fp8_block)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_four_models.png")
    fig.savefig(out, dpi=130)
    print("\nsaved", out)


if __name__ == "__main__":
    main()

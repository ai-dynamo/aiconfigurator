"""TTFT via run_static for DeepSeek-V3.2 / GLM-5 / MiniMax-M2.7 / M3 on B200.

Pure run_static prefill latency (static_ctx) — a clean model-vs-model TTFT
comparison, not the sglang cookbook's under-load 1580 ms (that number includes
queueing at concurrency 64 and isn't a single run_static prefill). Same precision
policy as compare_four_models: nvfp4 where the architecture has nvfp4 module rows,
else fp8_block.
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
    p = os.path.join(DATA_DIR, "dsa_generation_module_perf.parquet")
    if not os.path.exists(p):
        return False
    df = pd.read_parquet(p)
    return len(df[(df["architecture"] == arch) & (df["gemm_type"] == "nvfp4")]) > 0


def quant_for(dsa_arch):
    # Uniform fp8_block for all four — the common precision with real module data
    # on every architecture (GlmMoeDsa lacks nvfp4 rows). Apples-to-apples.
    q = dict(
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        moe_quant_mode=common.MoEQuantMode.fp8_block,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
    )
    return "fp8_block", q


# (display, model_name, dsa_arch_for_lookup)
MODELS = [
    ("DeepSeek-V3.2", "deepseek-ai/DeepSeek-V3.2", "DeepseekV32ForCausalLM"),
    ("GLM-5", "zai-org/GLM-5", "GlmMoeDsaForCausalLM"),
    ("MiniMax-M2.7", "MiniMaxAI/MiniMax-M2.7", None),
    ("MiniMax-M3", "MiniMaxAI/MiniMax-M3", "GlmMoeDsaForCausalLM"),
]
COLORS = {"DeepSeek-V3.2": "#3fb950", "GLM-5": "#d29922", "MiniMax-M2.7": "#4fa3ff", "MiniMax-M3": "#bc8cff"}


def ttft(be, model, db, bs, isl):
    db.set_default_database_mode(common.DatabaseMode.HYBRID)
    s = be.run_static(
        model,
        db,
        config.RuntimeConfig(batch_size=bs, beam_width=1, isl=isl, osl=2, prefix=0),
        mode="static_ctx",
        stride=1,
    )
    return sum(s.get_context_latency_dict().values())


def main():
    db = get_database(SYS, BK, VER)
    be = TRTLLMBackend()
    par = dict(tp_size=8, moe_tp_size=1, moe_ep_size=8)
    bs = 64  # match cookbook max-concurrency
    isls = (1024, 2048, 4096, 8192)

    rows = {}
    print(f"TTFT via run_static (static_ctx), bs={bs}\n{'model':>15} {'quant':>10}")
    for disp, name, dsa_arch in MODELS:
        label, q = quant_for(dsa_arch)
        model = get_model(name, config.ModelConfig(**par, **q), backend_name=BK)
        rows[disp] = (label, [ttft(be, model, db, bs, isl) for isl in isls])
        print(f"{disp:>15} {label:>10}")

    print(f"\n{'isl':>8}" + "".join(f"{d:>16}" for d, *_ in MODELS))
    for i, isl in enumerate(isls):
        print(f"{isl:>8}" + "".join(f"{rows[d][1][i]:>16.1f}" for d, *_ in MODELS))

    fig, a = plt.subplots(figsize=(7.2, 4.6))
    for disp, *_ in MODELS:
        label, vals = rows[disp]
        a.plot(isls, vals, "-o", color=COLORS[disp], label=f"{disp} ({label})")
    a.set(title=f"TTFT (run_static prefill, bs={bs}) — {SYS}, {BK} {VER}, tp8", xlabel="isl", ylabel="TTFT (ms)")
    a.legend(fontsize=9)
    a.grid(alpha=0.2)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_ttft.png")
    fig.savefig(out, dpi=130)
    print("\nsaved", out)


if __name__ == "__main__":
    main()

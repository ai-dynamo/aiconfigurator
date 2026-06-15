"""Cross-shape MoE transfer: predict model B's MoE latency from model A's
SOL-utilization at the same num_tokens (observation 5).

  util_A(T) = SOL_A(T) / latency_A(T)        # kernel efficiency from model A
  pred_B(T) = SOL_B(T) / util_A(T)           # reconstruct B with B's own SOL

Baseline = today's empirical: latency = SOL_B / 0.4 (moe.py scale_factor 0.4).

Both shapes must exist in the same collected dataset so we have ground-truth
latency_B(T) to score against.
"""

import numpy as np
import pandas as pd

H100 = {"mem_bw": 3350e9, "bfloat16_tc_flops": 989e12}
MOE_SCALE = 0.4
# fp8 MoEQuantMode: memory=1 byte, compute=2
QUANT = {"memory": 1, "compute": 2}

MODELS = {  # name -> (hidden, inter, topk, experts)  [all gated SwiGLU -> 3 gemms]
    "Qwen3-235B-A22B": (4096, 1536, 8, 128),
    "MiniMax-M2.5": (3072, 1536, 8, 256),
    "DeepSeek-V3": (7168, 2048, 8, 256),
    "Qwen3-30B-A3B": (2048, 768, 8, 128),
    "Mixtral-8x22B": (6144, 16384, 2, 8),
}

# curated pairs spanning near -> far shape distance
PAIRS = [
    ("Qwen3-235B-A22B", "MiniMax-M2.5"),  # same topk(8)+inter, diff hidden/experts
    ("DeepSeek-V3", "MiniMax-M2.5"),  # same topk(8)+experts, diff hidden/inter
    ("Qwen3-30B-A3B", "Qwen3-235B-A22B"),  # same topk(8)+experts, diff hidden/inter (sibling family)
    ("Mixtral-8x22B", "Qwen3-235B-A22B"),  # FAR: diff topk(2 vs 8), experts(8 vs 128), inter
    ("Qwen3-235B-A22B", "Mixtral-8x22B"),  # FAR reverse
]


def sol_ms(num_tokens, hidden, inter, topk, experts, tp=1, ep=1, num_gemms=3):
    total = num_tokens * topk
    ops = total * hidden * inter * num_gemms * 2 // ep // tp
    mem_bytes = QUANT["memory"] * (
        total // ep * hidden * 2
        + total // ep * inter * num_gemms // tp
        + hidden * inter * num_gemms // tp * min(experts // ep, total // ep)
    )
    sol_math = ops / (H100["bfloat16_tc_flops"] * QUANT["compute"]) * 1000.0
    sol_mem = mem_bytes / H100["mem_bw"] * 1000.0
    return max(sol_math, sol_mem)


def curve(df, hidden, inter, topk, experts):
    sl = (
        df[
            (df.moe_dtype == "fp8")
            & (df.hidden_size == hidden)
            & (df.inter_size == inter)
            & (df.topk == topk)
            & (df.num_experts == experts)
            & (df.moe_tp_size == 1)
            & (df.moe_ep_size == 1)
            & (df.distribution == "power_law_1.2")
        ]
        .drop_duplicates("num_tokens")
        .sort_values("num_tokens")
    )
    return sl.set_index("num_tokens")["latency"]


def run(path):
    df = pd.read_parquet(path)
    df = df[df["op_name"] == "moe"]
    curves, sols = {}, {}
    for name, (h, i, tk, e) in MODELS.items():
        c = curve(df, h, i, tk, e)
        curves[name] = c
        sols[name] = {t: sol_ms(t, h, i, tk, e) for t in c.index}

    print(f"\n=== {path}")
    for src, dst in PAIRS:
        if True:
            shared = sorted(set(curves[src].index) & set(curves[dst].index))
            ape_tr, ape_const = [], []
            for t in shared:
                lat_dst = curves[dst][t]
                util_src = sols[src][t] / curves[src][t]
                pred_tr = sols[dst][t] / util_src
                pred_const = sols[dst][t] / MOE_SCALE
                ape_tr.append(abs(pred_tr - lat_dst) / lat_dst)
                ape_const.append(abs(pred_const - lat_dst) / lat_dst)
            ape_tr = np.array(ape_tr) * 100
            ape_const = np.array(ape_const) * 100

            def _row(label, a):
                print(f"    {label:<16}{a.mean():>8.2f}{np.median(a):>9.2f}{np.percentile(a, 90):>8.2f}")

            print(f"\n  {src}  -->  {dst}   ({len(shared)} shared num_tokens)")
            print(f"    {'method':<16}{'MAPE%':>8}{'median%':>9}{'p90%':>8}")
            _row("transfer(util)", ape_tr)
            _row("const SOL/0.4", ape_const)


if __name__ == "__main__":
    import sys

    paths = sys.argv[1:] or ["src/aiconfigurator/systems/data/h100_sxm/trtllm/1.3.0rc10/moe_perf.parquet"]
    for p in paths:
        run(p)

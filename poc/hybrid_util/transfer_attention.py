"""Attention cross-(num_heads, head_dim) transfer feasibility (scheme B).

Coordinate = workload (bs, seq) at prefix=0 (pastkv parked). num_heads + head_dim
are NOT coordinates -- they go into SOL. util(bs,seq) = SOL/measured. Transfer:

    pred_target(bs,seq) = SOL_target(bs,seq) / util_source(bs,seq)
                        = measured_source * SOL_target / SOL_source

If util is ~invariant to num_heads / head_dim, transfer should be tight -- which
would mean attention *variants* (different head counts / head dims) can be
modeled by borrowing a sibling's util. Baseline = fixed SOL/0.6 (ctx-attn const).
"""

import numpy as np
import pandas as pd

H100 = {"mem_bw": 3350e9, "bf16": 989e12}
PATH = "src/aiconfigurator/systems/data/h100_sxm/trtllm/1.3.0rc10/context_attention_perf.parquet"
COMPUTE = {"bfloat16": 1, "fp8": 2}  # fmha compute factor
KVMEM = {"bfloat16": 2, "fp8": 1}


def sol_ms(b, s, n, h, n_kv, attn_dtype="bfloat16", kv_dtype="bfloat16"):
    # context attention, prefix=0, window=0 (mirrors ContextAttention.get_sol)
    ops = 2 * b * (s * s) * n * h
    mem = 4 * b * n * s * h + KVMEM[kv_dtype] * b * 2 * n_kv * s * h
    sol_math = ops / H100["bf16"] * 1000.0 / COMPUTE[attn_dtype]
    sol_mem = mem / H100["mem_bw"] * 1000.0
    return max(sol_math, sol_mem)


def load_configs():
    df = pd.read_parquet(PATH)
    df = df[
        (df.attn_dtype == "bfloat16")
        & (df.kv_cache_dtype == "bfloat16")
        & (df.window_size == 0)
        & (df.num_key_value_heads == df.num_heads)
    ]  # MHA
    configs = {}  # (num_heads, head_dim) -> {(b, isl): latency}
    for (n, h), g in df.groupby(["num_heads", "head_dim"]):
        configs[(n, h)] = {(int(r.batch_size), int(r.isl)): float(r.latency) for r in g.itertuples()}
    return configs


def transfer_mape(src, dst, configs):
    """measured_src * SOL_dst/SOL_src vs measured_dst, over shared (b,s)."""
    (ns, hs), (nd, hd) = src, dst
    shared = set(configs[src]) & set(configs[dst])
    apes = []
    for b, s in shared:
        sol_s = sol_ms(b, s, ns, hs, ns)
        sol_d = sol_ms(b, s, nd, hd, nd)
        pred = configs[src][(b, s)] * sol_d / sol_s
        apes.append(abs(pred - configs[dst][(b, s)]) / configs[dst][(b, s)])
    return np.mean(apes) * 100 if apes else float("nan")


def main():
    configs = load_configs()
    print(f"{len(configs)} (num_heads, head_dim) configs: {sorted(configs)}\n")

    print("== transfer across num_heads (head_dim=128) ==")
    for src_n, dst_n in [(16, 32), (32, 16), (8, 64), (64, 8), (16, 64), (24, 32)]:
        if (src_n, 128) in configs and (dst_n, 128) in configs:
            m = transfer_mape((src_n, 128), (dst_n, 128), configs)
            print(f"  heads {src_n:>3} -> {dst_n:<3} : {m:6.1f}%")

    print("\n== transfer across head_dim (same num_heads) ==")
    for n in [8, 16, 32, 64]:
        if (n, 64) in configs and (n, 128) in configs:
            print(
                f"  heads={n}: 64->128 {transfer_mape((n, 64), (n, 128), configs):6.1f}%   "
                f"128->64 {transfer_mape((n, 128), (n, 64), configs):6.1f}%"
            )

    print("\n== leave-one-out: nearest source by (num_heads, head_dim) log-distance ==")
    keys = sorted(configs)
    tr_all, const_all = [], []
    for dst in keys:
        # nearest other config in log (num_heads, head_dim) space
        best, bestd = None, None
        for src in keys:
            if src == dst:
                continue
            d = (np.log(src[0]) - np.log(dst[0])) ** 2 + (np.log(src[1]) - np.log(dst[1])) ** 2
            if bestd is None or d < bestd:
                bestd, best = d, src
        m = transfer_mape(best, dst, configs)
        # const baseline for dst
        nd, hd = dst
        cs = [abs(sol_ms(b, s, nd, hd, nd) / 0.6 - lat) / lat for (b, s), lat in configs[dst].items()]
        cm = np.mean(cs) * 100
        tr_all.append(m)
        const_all.append(cm)
        print(f"  {dst!s:<12} <- {best!s:<12}  transfer {m:6.1f}%   const {cm:6.1f}%")
    print(f"\n  OVERALL transfer MAPE={np.nanmean(tr_all):.1f}%  |  const MAPE={np.mean(const_all):.1f}%")


if __name__ == "__main__":
    main()

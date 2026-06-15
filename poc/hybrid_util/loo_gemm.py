"""Leave-one-out validation: latency-space vs util-space interpolation for GEMM.

Prototype only (not production). Answers: does interpolating SOL-utilization
(util = SOL/latency) and reconstructing latency = SOL/util beat interpolating
raw latency, and beat the fixed empirical scale_factor (0.8)?

Two regimes, both along the m (token) axis for a fixed (quant, n, k) weight
shape -- the dominant real query (gemm.py:452 1-D interp fast path):
  * INTERIOR: hold out an interior m, predict from neighbors.
  * EXTRAPOLATION: hold out the boundary m (min/max), predict by extending.

Methods compared:
  * const   : latency = SOL / 0.8                 (today's empirical)
  * latency : linear interp/extrap of raw latency (today's silicon)
  * util    : interp util, latency = SOL/util_interp; extrap clamps boundary util
"""

import sys

import numpy as np
import pandas as pd

# --- GEMM SOL, mirroring operations/gemm.py:get_sol (units: ms) ---
SPECS = {  # per-system gpu spec, selected by path substring
    "h100_sxm": {"mem_bw": 3350e9, "bfloat16_tc_flops": 989e12, "fp8_tc_flops": 1978e12},
    "b200_sxm": {"mem_bw": 8000e9, "bfloat16_tc_flops": 2250e12, "fp8_tc_flops": 4500e12},
}
MEMBYTES = {"bfloat16": 2, "fp8": 1, "fp8_block": 1}
SCALE_CONST = 0.8


def sol_ms(m, n, k, dt, spec):
    tc = spec["bfloat16_tc_flops"] if dt == "bfloat16" else spec["fp8_tc_flops"]
    sol_math = 2 * m * n * k / tc * 1000.0
    sol_mem = MEMBYTES[dt] * (m * n + m * k + n * k) / spec["mem_bw"] * 1000.0
    return max(sol_math, sol_mem)


def linear_at(xs, ys, xq):
    """Plain 2-point linear through the bracketing/nearest pair (extrapolates)."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    # pick the two points nearest xq (works for interior and boundary)
    order = np.argsort(np.abs(xs - xq))
    i, j = sorted(order[:2])
    x0, x1, y0, y1 = xs[i], xs[j], ys[i], ys[j]
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (xq - x0) / (x1 - x0)


def run(path):
    spec = next((s for key, s in SPECS.items() if key in path), None)
    if spec is None:
        print(f"\n=== {path}\n  SKIP: no SOL spec for this system")
        return
    df = pd.read_parquet(path)
    df = df[df["op_name"] == "gemm"]
    rows = {"interior": [], "extrap": []}  # regime -> list of (method, ape)

    for (dt, n, k), g in df.groupby(["gemm_dtype", "n", "k"]):
        if dt not in MEMBYTES:
            continue
        g = g.drop_duplicates("m").sort_values("m")
        ms = g["m"].to_numpy(float)
        lats = g["latency"].to_numpy(float)
        if len(ms) < 3:
            continue
        sols = np.array([sol_ms(m, n, k, dt, spec) for m in ms])
        utils = sols / lats  # SOL/latency in (0,1]-ish

        for idx in range(len(ms)):
            interior = 0 < idx < len(ms) - 1
            regime = "interior" if interior else "extrap"
            mq, actual, solq = ms[idx], lats[idx], sols[idx]
            keep = np.arange(len(ms)) != idx
            xk, latk, utilk = ms[keep], lats[keep], utils[keep]

            pred_const = solq / SCALE_CONST
            pred_lat = linear_at(xk, latk, mq)
            if interior:
                util_q = linear_at(xk, utilk, mq)
            else:
                # observation 3: extrapolation clamps to the boundary util
                util_q = utilk[np.argmin(np.abs(xk - mq))]
            pred_util = solq / util_q if util_q > 0 else pred_const

            for name, pred in (("const", pred_const), ("latency", pred_lat), ("util", pred_util)):
                if pred > 0:
                    rows[regime].append((name, abs(pred - actual) / actual))

    print(f"\n=== {path}")
    for regime in ("interior", "extrap"):
        data = rows[regime]
        n_pts = len(data) // 3
        print(f"\n  [{regime}]  (~{n_pts} held-out points)")
        print(f"    {'method':<9} {'MAPE%':>8} {'median%':>8} {'p90%':>8}")
        for name in ("const", "latency", "util"):
            apes = np.array([a for nm, a in data if nm == name]) * 100
            if len(apes) == 0:
                continue
            print(f"    {name:<9} {apes.mean():>8.2f} {np.median(apes):>8.2f} {np.percentile(apes, 90):>8.2f}")


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "src/aiconfigurator/systems/data/h100_sxm/trtllm/1.3.0rc10/gemm_perf.parquet",
    ]
    for p in paths:
        run(p)

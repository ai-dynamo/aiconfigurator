# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Classify the pareto-DRIFT entries: discreteness vs real frontier divergence.

For each DRIFT entry, run cli_default under both engines and compare the
user-facing frontier CURVES (throughput tokens/s/gpu vs latency tpot) by
interpolating each curve onto the other's throughput points over the overlapping
range. Small curve gap => frontiers materially equivalent (the row-count DRIFT is
discreteness in point selection). Large gap => a real residual divergence.
"""

import logging

import numpy as np
import pandas as pd

from tools.support_matrix.support_matrix import SupportMatrix, _get_test_constraints

logging.disable(logging.CRITICAL)

ENTRIES = [
    ("Qwen/Qwen3-30B-A3B", "gb200", "vllm", "0.19.0"),
    ("moonshotai/Kimi-K2.5", "h200_sxm", "vllm", "0.14.0"),
    ("moonshotai/Kimi-K2.5", "h200_sxm", "vllm", "0.19.0"),
    ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "gb300", "sglang", "0.5.10"),
]


def frontier_curve(df, x="tokens/s/gpu", y="tpot"):
    d = (
        pd.DataFrame({x: pd.to_numeric(df[x], errors="coerce"), y: pd.to_numeric(df[y], errors="coerce")})
        .dropna()
        .sort_values(x)
    )
    return d[x].to_numpy(dtype=float), d[y].to_numpy(dtype=float)


def max_gap(xa, ya, xb, yb):
    # Compare the two curves on the UNION of both frontiers' x-points within the
    # overlap, not just one curve's x. Sampling only A's x-points is one-sided: a
    # sparser curve can hide divergence that only shows up at the other frontier's
    # x-points. Interpolate BOTH onto the union grid and take the relative gap.
    if len(xa) == 0 or len(xb) == 0:
        return None
    lo = max(xa.min(), xb.min())
    hi = min(xa.max(), xb.max())
    xs = np.unique(np.concatenate([xa, xb]))
    xs = xs[(xs >= lo) & (xs <= hi)]
    if len(xs) < 2:
        return None
    ya_i = np.interp(xs, xa, ya)
    yb_i = np.interp(xs, xb, yb)
    rel = np.abs(ya_i - yb_i) / np.maximum(np.abs(yb_i), 1e-9)
    return float(rel.max() * 100), float(rel.mean() * 100)


# gate metric: envelope extremes (exactly what _compare_frontier_envelope checks)
def ext(df, col, agg):
    v = pd.to_numeric(df[col], errors="coerce").dropna()
    return (v.max() if agg == "max" else v.min()) if len(v) else float("nan")


def _run(model, system, backend, version, be):
    return SupportMatrix._run_mode(
        mode="disagg",
        model=model,
        system=system,
        backend=backend,
        version=version,
        constraints=_get_test_constraints(model),
        engine_step_backend=be,
    )


for model, system, backend, version in ENTRIES:
    tag = f"{model.split('/')[-1]:34s} {system:8s} {version}"
    # Fail fast and keep going: an unsupported/missing combo (or an empty
    # frontier) shouldn't abort the whole sweep. _run_mode may return None or an
    # empty DataFrame; guard before len()/metrics.
    try:
        py = _run(model, system, backend, version, "python")
        ru = _run(model, system, backend, version, "rust")
    except Exception as exc:
        print(f"CLASSIFY {tag}: SKIP ({type(exc).__name__}: {exc})", flush=True)
        continue
    py_n = None if py is None else len(py)
    ru_n = None if ru is None else len(ru)
    if not py_n or not ru_n:
        print(f"CLASSIFY {tag}: SKIP (no frontier — py={py_n} ru={ru_n})", flush=True)
        continue
    line = f"{tag}: py={py_n} ru={ru_n}"

    for col, agg in [("tokens/s/user", "max"), ("tpot", "min"), ("request_latency", "min")]:
        p = ext(py, col, agg)
        r = ext(ru, col, agg)
        rel = abs(p - r) / max(abs(p), abs(r), 1e-9) * 100
        line += f" | {agg}({col}) py={p:.4g} ru={r:.4g} d={rel:.2f}%"
    # smooth curve overlap on the user-facing SLA tradeoff
    xa, ya = frontier_curve(py, "tokens/s/user", "request_latency")
    xb, yb = frontier_curve(ru, "tokens/s/user", "request_latency")
    g = max_gap(xa, ya, xb, yb)
    line += " | reqlat-curve: " + (f"max {g[0]:.2f}% mean {g[1]:.2f}%" if g else "n/a")
    print("CLASSIFY " + line, flush=True)

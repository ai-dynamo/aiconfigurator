#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Test whether per-rank MoE routing skew explains the within-group latency
residual on mixed steps.

Inputs (from a routing_capture run dir):
  routing/routing_rank{R}_part*.npz  -- per-step global logical per-expert counts
                                         [n_steps, num_layers, num_experts] + signature + clean gpu_time_ms
  routing/manifest_rank{R}.txt        -- num_layers/num_experts/top_k/ep_size
  fpm_metrics_detail.csv              -- ground-truth scheduler wall-time per step (optional cross-check)

Method:
  * Per-rank device load = global per-expert counts sliced by the contiguous EP
    map (expert e -> rank e // (num_experts // ep_size)).
  * Per-step skew scalars across the ep_size ranks, aggregated over layers:
      cv     = mean_l std(load_l)/mean(load_l)
      gini   = mean_l gini(load_l)
      maxmean= mean_l max(load_l)/mean(load_l)   (straggler overload; all-to-all gate)
  * Group mixed steps (ctx_tokens>0 & decode_requests>0) by identical aggregate
    inputs (ctx_tokens, ctx_requests, ctx_kv_tokens, decode_requests, round(mean_decode_kv)).
  * Within-group: pooled correlation of skew vs latency residual + variance
    explained (R^2). Verdict: does skew drive the residual?
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd


def _gini(x: np.ndarray) -> float:
    x = np.sort(x.astype(np.float64))
    n = x.size
    s = x.sum()
    if n == 0 or s <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * x).sum()) / (n * s) - (n + 1.0) / n)


def _read_manifest(path: str) -> dict:
    d = {}
    with open(path) as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                d[k] = v
    return d


def load_routing(run_dir: str, rank: int):
    routing_dir = os.path.join(run_dir, "routing")
    man = _read_manifest(os.path.join(routing_dir, f"manifest_rank{rank}.txt"))
    num_layers = int(man["num_layers"])
    num_experts = int(man["num_experts"])
    ep_size = int(man["ep_size"])
    parts = sorted(glob.glob(os.path.join(routing_dir, f"routing_rank{rank}_part*.npz")))
    if not parts:
        raise SystemExit(f"no routing sidecars in {routing_dir} for rank {rank}")
    counts, meta = [], []
    for p in parts:
        z = np.load(p)
        counts.append(z["counts"])
        n = len(z["step"])
        for i in range(n):
            meta.append({
                "step": int(z["step"][i]),
                "gpu_time_ms": float(z["gpu_time_ms"][i]),
                "ctx_tokens": int(z["ctx_tokens"][i]),
                "ctx_requests": int(z["ctx_requests"][i]),
                "ctx_kv_tokens": int(z["ctx_kv_tokens"][i]),
                "decode_requests": int(z["decode_requests"][i]),
                "mean_decode_kv_tokens": float(z["mean_decode_kv_tokens"][i]),
                "total_tokens": int(z["total_tokens"][i]),
            })
    counts = np.concatenate(counts, axis=0)  # [N, L, E]
    df = pd.DataFrame(meta)
    return counts, df, num_layers, num_experts, ep_size


def per_step_skew(counts: np.ndarray, ep_size: int) -> pd.DataFrame:
    N, L, E = counts.shape
    per_rank = E // ep_size
    # [N, L, ep_size] device loads
    loads = counts[:, :, : per_rank * ep_size].reshape(N, L, ep_size, per_rank).sum(axis=3).astype(np.float64)
    mean = loads.mean(axis=2)                       # [N, L]
    std = loads.std(axis=2)                          # [N, L]
    mx = loads.max(axis=2)                           # [N, L]
    active = mean > 0                                # layers with tokens
    with np.errstate(divide="ignore", invalid="ignore"):
        cv_l = np.where(active, std / mean, np.nan)
        mm_l = np.where(active, mx / mean, np.nan)
    cv = np.nanmean(cv_l, axis=1)
    maxmean = np.nanmean(mm_l, axis=1)
    # gini per (step) averaged over layers
    gini = np.empty(N)
    for i in range(N):
        gs = [_gini(loads[i, l]) for l in range(L) if active[i, l]]
        gini[i] = float(np.mean(gs)) if gs else np.nan
    return pd.DataFrame({"cv": cv, "gini": gini, "maxmean": maxmean})


def within_group_analysis(df: pd.DataFrame, lat_col: str, skew_cols, min_n: int = 4, drop_warmup: int = 5):
    df = df.iloc[drop_warmup:].copy()
    mixed = df[(df.ctx_tokens > 0) & (df.decode_requests > 0)].copy()
    mixed["mdk"] = mixed["mean_decode_kv_tokens"].round(0)
    keys = ["ctx_tokens", "ctx_requests", "ctx_kv_tokens", "decode_requests", "mdk"]
    g = mixed.groupby(keys)
    sizes = g.size()
    big_keys = sizes[sizes >= min_n].index
    print(f"\n[{lat_col}] mixed steps: {len(mixed)}; groups(n>={min_n}): {len(big_keys)}")
    # pooled within-group CV of latency
    parts = [grp for _, grp in g if len(grp) >= min_n]
    if not parts:
        print("  no groups large enough; skipping")
        return
    cvs = [(grp[lat_col].std() / grp[lat_col].mean(), len(grp)) for grp in parts]
    pooled_cv = np.sqrt(sum((c**2) * (n - 1) for c, n in cvs) / sum((n - 1) for _, n in cvs))
    print(f"  pooled within-group {lat_col} CV = {pooled_cv:.4f}  (target residual ~0.21)")
    # within-group residuals (demeaned per group), pooled
    lat_res, skew_res = {"lat": []}, {c: [] for c in skew_cols}
    for grp in parts:
        lr = grp[lat_col].values - grp[lat_col].mean()
        lat_res["lat"].append(lr)
        for c in skew_cols:
            skew_res[c].append(grp[c].values - grp[c].mean())
    lat_r = np.concatenate(lat_res["lat"])
    print(f"  pooled within-group residual samples: {len(lat_r)}")
    rng = np.random.default_rng(0)
    for c in skew_cols:
        sr = np.concatenate(skew_res[c])
        m = np.isfinite(lat_r) & np.isfinite(sr)
        if m.sum() < 5 or np.std(sr[m]) == 0:
            print(f"    skew={c:8s}: insufficient/constant")
            continue
        lr, s = lat_r[m], sr[m]
        r = np.corrcoef(lr, s)[0, 1]
        # bootstrap 95% CI on the pooled within-group correlation
        boot = []
        n = len(lr)
        for _ in range(2000):
            idx = rng.integers(0, n, n)
            sd = np.std(s[idx])
            if sd > 0:
                boot.append(np.corrcoef(lr[idx], s[idx])[0, 1])
        lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan))
        print(f"    skew={c:8s}: within-group corr = {r:+.3f}  (95% CI [{lo:+.2f},{hi:+.2f}])  "
              f"R^2={r*r:.3f}  -> explains ~{100*r*r:.0f}% of within-group {lat_col} variance")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--min-n", type=int, default=4)
    ap.add_argument("--drop-warmup", type=int, default=5)
    args = ap.parse_args()

    counts, df, L, E, ep = load_routing(args.run_dir, args.rank)
    print(f"loaded {counts.shape[0]} steps; layers={L} experts={E} ep_size={ep} (experts/rank={E//ep})")
    skew = per_step_skew(counts, ep)
    df = pd.concat([df.reset_index(drop=True), skew.reset_index(drop=True)], axis=1)

    # quick skew shape summary
    mixed = df[(df.ctx_tokens > 0) & (df.decode_requests > 0)]
    print("\n=== per-rank skew shape (mixed steps) ===")
    for c in ["cv", "maxmean", "gini"]:
        s = mixed[c].dropna()
        if len(s):
            print(f"  {c:8s}: mean={s.mean():.3f} p50={s.median():.3f} p95={s.quantile(0.95):.3f} max={s.max():.3f}")

    # 1) self-aligned: clean gpu_time_ms vs skew (primary attribution)
    within_group_analysis(df, "gpu_time_ms", ["maxmean", "cv", "gini"], args.min_n, args.drop_warmup)

    # 2) cross-check: ground-truth wall-time residual exists (FPM detail CSV)
    detail = os.path.join(args.run_dir, "fpm_metrics_detail.csv")
    if os.path.exists(detail):
        d = pd.read_csv(detail)
        d = d[d.dp_rank == args.rank] if "dp_rank" in d.columns else d
        d = d.rename(columns={
            "sum_prefill_tokens": "ctx_tokens", "num_prefill_requests": "ctx_requests",
            "sum_prefill_kv_tokens": "ctx_kv_tokens", "num_decode_requests": "decode_requests",
        })
        d["mean_decode_kv_tokens"] = np.where(
            d["decode_requests"] > 0, d["sum_decode_kv_tokens"] / d["decode_requests"].clip(lower=1), 0.0)
        within_group_analysis(d, "latency_ms", [], args.min_n, args.drop_warmup)

        # 3) order-align sidecar routing to FPM wall-time and attribute (best effort)
        attribute_walltime(df, d, ["maxmean", "cv", "gini"], args.min_n, args.drop_warmup)
    return 0


def attribute_walltime(side: pd.DataFrame, fpm: pd.DataFrame, skew_cols, min_n, drop_warmup):
    """Align the ordered routing sidecar to the ordered FPM wall-time stream by
    matching the aggregate signature sequence, then attribute the WALL-TIME
    within-group residual to skew."""
    sig = ["ctx_tokens", "ctx_requests", "ctx_kv_tokens", "decode_requests"]
    f = fpm.reset_index(drop=True)
    s = side.reset_index(drop=True)
    # exact per-row signature join is ambiguous (repeats); align by sequence order
    # assuming both streams are the same ordered real-step sequence for this rank.
    n = min(len(f), len(s))
    f2 = f.iloc[:n].reset_index(drop=True)
    s2 = s.iloc[:n].reset_index(drop=True)
    match = (f2[sig].values == s2[sig].values).all(axis=1).mean()
    print(f"\n[walltime-attrib] sidecar/FPM order-alignment signature match over first {n} rows = {match:.2%}")
    if match < 0.9:
        print("  alignment weak (<90%); relying on self-aligned gpu_time_ms result above")
        return
    merged = s2.copy()
    merged["latency_ms"] = f2["latency_ms"].values
    within_group_analysis(merged, "latency_ms", skew_cols, min_n, drop_warmup)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Final skew-vs-latency attribution across one or more routing_capture runs.

Three complementary tests of "does per-rank routing skew drive the mixed-step
latency residual":
  1. Skew variability: if skew barely varies step-to-step it cannot explain a
     step-level residual, regardless of correlation.
  2. Coarse-binned within-group: group identical-ish aggregate inputs, pooled
     corr(skew, latency residual) + variance explained.
  3. Regression-residual (the RF+skew question): fit latency ~ aggregate FPM
     features, then test whether skew explains the residual (added R^2).
"""
from __future__ import annotations
import glob, sys
import numpy as np
import pandas as pd


def load(run_dir, rank=0):
    parts = sorted(glob.glob(f"{run_dir}/routing/routing_rank{rank}_part*.npz"))
    rows, counts = [], []
    for p in parts:
        z = np.load(p); counts.append(z["counts"])
        for i in range(len(z["step"])):
            rows.append(dict(gpu_ms=float(z["gpu_time_ms"][i]),
                ctx_tokens=int(z["ctx_tokens"][i]), ctx_requests=int(z["ctx_requests"][i]),
                ctx_kv_tokens=int(z["ctx_kv_tokens"][i]), decode_requests=int(z["decode_requests"][i]),
                mean_decode_kv=float(z["mean_decode_kv_tokens"][i]), total=int(z["total_tokens"][i])))
    counts = np.concatenate(counts); df = pd.DataFrame(rows)
    fpm = pd.read_csv(f"{run_dir}/fpm_metrics_detail.csv")
    fpm = fpm[fpm.dp_rank == 0].reset_index(drop=True)
    n = min(len(fpm), len(df))
    df = df.iloc[:n].copy(); df["wall_ms"] = fpm["latency_ms"].values[:n]
    return df, counts[:n]


def add_skew(df, counts, ep=4):
    E = counts.shape[2]; per = E // ep
    loads = counts[:, :, :per*ep].reshape(counts.shape[0], counts.shape[1], ep, per).sum(3).astype(float)
    mean = loads.mean(2); mx = loads.max(2); std = loads.std(2); act = mean > 0
    df = df.copy()
    df["maxmean"] = np.nanmean(np.where(act, mx/mean, np.nan), axis=1)
    df["cv"] = np.nanmean(np.where(act, std/mean, np.nan), axis=1)
    return df


def main():
    runs = sys.argv[1:] or ["/workspace/repo/aiconfigurator/routing_runs/stageB3"]
    frames = []
    for r in runs:
        df, c = load(r); df = add_skew(df, c); df["run"] = r.split("/")[-1]
        frames.append(df.iloc[10:])  # drop warmup
    df = pd.concat(frames, ignore_index=True)
    mix = df[(df.ctx_tokens > 0) & (df.decode_requests > 0)].copy()
    # drop compile/capture outliers (robust): keep within 5x median gpu time
    med = mix.gpu_ms.median()
    mix = mix[mix.gpu_ms < 5*med]
    print(f"runs={runs}\nmixed steps (post-warmup, outlier-trimmed): {len(mix)}")

    print("\n[1] SKEW VARIABILITY across mixed steps (a near-constant feature cannot drive a residual)")
    for col in ["maxmean", "cv"]:
        s = mix[col]
        print(f"  {col:8s}: mean={s.mean():.3f} std={s.std():.4f}  CV-across-steps={s.std()/s.mean():.4f}  "
              f"range=[{s.min():.3f},{s.max():.3f}]")

    print("\n[2] COARSE-BINNED within-group residual (bin ctx_tokens/512, mean_decode_kv/512)")
    mix["b_ctx"] = (mix.ctx_tokens // 512); mix["b_dr"] = mix.decode_requests
    mix["b_mdk"] = (mix.mean_decode_kv // 512); mix["b_creq"] = mix.ctx_requests.clip(upper=8)
    for lat in ["gpu_ms", "wall_ms"]:
        g = mix.groupby(["b_ctx", "b_creq", "b_dr", "b_mdk"])
        parts = [grp for _, grp in g if len(grp) >= 4]
        if not parts:
            print(f"  {lat}: no bins n>=4"); continue
        lr = np.concatenate([grp[lat].values - grp[lat].mean() for grp in parts])
        cvs = [(grp[lat].std()/grp[lat].mean(), len(grp)) for grp in parts if grp[lat].mean() > 0]
        pooled = np.sqrt(sum(c*c*(n-1) for c, n in cvs)/sum(n-1 for _, n in cvs))
        print(f"  {lat}: {len(parts)} bins, {len(lr)} samples, pooled within-bin CV={pooled:.4f}")
        for col in ["maxmean", "cv"]:
            sr = np.concatenate([grp[col].values - grp[col].mean() for grp in parts])
            m = np.isfinite(lr) & np.isfinite(sr) & (np.abs(sr) >= 0)
            if np.std(sr[m]) == 0:
                print(f"      skew={col}: constant within bins"); continue
            rr = np.corrcoef(lr[m], sr[m])[0, 1]
            print(f"      corr(resid {lat}, {col}) = {rr:+.3f}  R^2={rr*rr:.3f}")

    print("\n[3] REGRESSION-RESIDUAL (RF+skew test): fit latency ~ aggregate FPM features, "
          "does skew explain the residual?")
    feats = ["ctx_tokens", "ctx_requests", "ctx_kv_tokens", "decode_requests", "mean_decode_kv"]
    X = mix[feats].values
    X = np.column_stack([X, X**2, np.ones(len(X))])
    for lat in ["gpu_ms", "wall_ms"]:
        y = mix[lat].values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        base_r2 = 1 - resid.var()/y.var()
        print(f"  {lat}: aggregate-feature R^2={base_r2:.3f}, residual std={resid.std():.2f}ms "
              f"({100*resid.std()/y.mean():.1f}% of mean)")
        for col in ["maxmean", "cv"]:
            sk = mix[col].values
            r = np.corrcoef(resid, sk)[0, 1] if np.std(sk) > 0 else 0.0
            print(f"      corr(residual, {col})={r:+.3f}  ->  skew explains ~{100*r*r:.1f}% of the residual")
    print("\nVERDICT printed above; interpret [1] (skew variance) + [3] (added R^2).")


if __name__ == "__main__":
    main()

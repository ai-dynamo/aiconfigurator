#!/usr/bin/env python3
"""Log-log FPM-vs-AIC layerwise charts for a single model.

Produces three images: one for context (ctx), one for generation (gen), and one
for mixed. For ctx and gen there is a separate subplot per (parallelism, past_kv)
so that only a single query variable is swept along the x axis:

    ctx  ->  x = number of new tokens   (past_kv held fixed per subplot)
    gen  ->  x = decode batch size      (past_kv held fixed per subplot)

For mixed there is one subplot per parallelism (no past_kv split); because a
mixed step has no single swept variable, it is drawn as an AIC-vs-FPM parity
scatter against the y=x reference line.

Each ctx/gen subplot shows three series:
    1. AIC exact   - AIC estimate evaluated at the measured layerwise grid points
    2. AIC interp  - AIC estimate swept densely across x (the interpolated curve)
    3. FPM real    - real collected FPM measurements

Usage:
    .venv/bin/python tools/plot_fpm_vs_aic.py \
        --layerwise runs/layerwise_full_vllm0201_20260615_045248/layerwise.csv \
        --model "Qwen/Qwen3-32B" --out-dir fpm_vs_aic_charts
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path

# Matplotlib needs a writable config dir (the default ~/.config may be read-only
# under sandboxing). Set it before importing matplotlib.
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl_"))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure the repo root is importable so `collector` and `aiconfigurator` resolve.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aiconfigurator.sdk.backends import vllm_backend
from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.perf_database import PerfDatabase
from collector.layerwise.diagnostics.compare_aic_layerwise_fpm import (
    _effective_moe_parallelism,
    _LayerwiseDatabase,
    _model_defaults,
    _prepare_moe_overlay_systems_root,
)
from collector.layerwise.diagnostics.compare_aic_layerwise_fpm_summary import _all_cases

_AIC_EXCEPTIONS = (AssertionError, KeyError, ValueError, Exception)

# Prefill-size buckets for the mixed parity plot, keyed on TOTAL prefill context
# (new ctx_tokens + cached prefix) so long-prompt steps (deep into a 16k prompt)
# land in the high buckets. (lo, hi] in tokens.
_PREFILL_BUCKETS = [(0, 2048), (2048, 4096), (4096, 8192), (8192, 16384), (16384, float("inf"))]
_PREFILL_LABELS = ["<=2k", "2-4k", "4-8k", "8-16k", ">16k"]


def _bucket_colors():
    cmap = plt.get_cmap("viridis")
    n = len(_PREFILL_BUCKETS)
    return [cmap(i / (n - 1)) for i in range(n)]


def _filter_outliers_by_sigma(records, get_fy, sigma: float):
    """Drop records whose error is more than `sigma` std devs from the mean.

    The error metric is the log-ratio log(aic/fpm): it's symmetric for
    multiplicative over/under-prediction and isolates gross disagreements (a
    35x-too-fast point sits ~3.5 in log space vs ~0.1 for a 10% error). Returns
    (kept_records, n_dropped). No assumption that AIC is "correct" — the cut is
    purely on the spread of the observed errors.
    """
    if not sigma or sigma <= 0 or len(records) < 3:
        return list(records), 0
    logs = []
    for r in records:
        f, y = get_fy(r)
        logs.append(math.log(y / f))
    mean = sum(logs) / len(logs)
    std = (sum((v - mean) ** 2 for v in logs) / len(logs)) ** 0.5
    if std <= 0:
        return list(records), 0
    kept = [r for r, v in zip(records, logs, strict=True) if abs(v - mean) <= sigma * std]
    return kept, len(records) - len(kept)


# ----------------------------------------------------------------------------
# AIC evaluation helpers (full per-step latency = sum of all op latencies).
# ----------------------------------------------------------------------------
def _aic_ctx(backend, model, database, rc, new_tokens: int, past_kv: int):
    try:
        latency, _, _ = backend._get_context_step_latency(
            model, database, rc, ctx_tokens=int(new_tokens), ctx_kv_tokens=int(past_kv), ctx_requests=1
        )
        return float(sum(latency.values()))
    except _AIC_EXCEPTIONS:
        return None


def _aic_gen(backend, model, database, rc, batch_size: int, past_kv: int, comm: bool = True):
    # Toggle whether the generation all-reduce is added back for single-GPU-collected
    # layerwise data (off => legacy behavior that drops it). The decode-compute
    # batch calibration is applied inside the SDK, so it is always included here.
    vllm_backend._LAYERWISE_GEN_SINGLE_GPU_COMM = comm
    try:
        latency, _, _ = backend._get_decode_step_latency(
            model, database, rc, batch_size=int(batch_size), past_kv=int(past_kv)
        )
        return float(sum(latency.values()))
    except _AIC_EXCEPTIONS:
        return None


def _aic_mixed(
    backend, model, database, rc, *, ctx_tokens, gen_tokens, mean_decode_kv, ctx_prefix_tokens, ctx_requests
):
    try:
        aic_ms, _, _, _ = backend._get_mix_step_latency(
            model,
            database,
            rc,
            ctx_tokens=int(ctx_tokens),
            gen_tokens=int(gen_tokens),
            isl=round(mean_decode_kv),
            osl=1,
            prefix=int(ctx_prefix_tokens),
            ctx_requests=int(ctx_requests),
        )
        return float(aic_ms)
    except _AIC_EXCEPTIONS:
        return None


def _nearest_in_log(value: float, grid: list[int]) -> int:
    """Nearest grid value to `value` measured in log1p space (handles 0)."""

    def lg(v):
        return math.log1p(max(float(v), 0.0))

    return min(grid, key=lambda g: abs(lg(g) - lg(value)))


def _dense_x(xmin: float, xmax: float, n: int = 60) -> list[int]:
    xmin = max(1.0, float(xmin))
    xmax = max(xmin, float(xmax))
    pts = np.geomspace(xmin, xmax, n)
    return sorted({round(p) for p in pts})


def _raw_collected_points(
    layerwise_df: pd.DataFrame,
    tp: int,
    lw_phase: str,
    sweep_col: str,
    past_kv: int,
    moe_tp: int | None = None,
    ep: int | None = None,
) -> dict[int, float]:
    """Exact full-model latencies straight from the layerwise CSV (no lookup/interp).

    Each row's measured per-chunk latency is scaled to the full model by
    (layer_multiplier / measured_layer_count) and summed across layer types,
    matching the SDK's `_layerwise_detail_scale`. For dense single-type models
    this is just the collected `latency_ms`. For MoE models the same (tp, past_kv)
    has multiple EP/MoE-TP rows, so they MUST be filtered to the one matching the
    subplot -- otherwise the rows get summed across parallelism configs.
    """

    sub = layerwise_df[
        (layerwise_df["phase"] == lw_phase) & (layerwise_df["attn_tp"] == tp) & (layerwise_df["past_kv"] == past_kv)
    ]
    if ep is not None and "ep" in sub.columns:
        sub = sub[sub["ep"] == ep]
    if moe_tp is not None and "moe_tp" in sub.columns:
        sub = sub[sub["moe_tp"] == moe_tp]
    # Hold the non-swept query var fixed so each x maps to one collected point.
    if lw_phase == "ctx":
        sub = sub[sub["batch_size"] == 1]
    else:
        sub = sub[sub["new_tokens"] == 1]

    pts: dict[int, float] = {}
    for x_val, grp in sub.groupby(sweep_col):
        total = 0.0
        for _, row in grp.iterrows():
            mult = float(row.get("layer_multiplier") or 0.0)
            meas = max(float(row.get("measured_layer_count") or 1.0), 1.0)
            scale = (mult / meas) if mult > 0 else 1.0
            total += float(row["latency_ms"]) * scale
        if total > 0:
            pts[int(x_val)] = total
    return pts


# ----------------------------------------------------------------------------
# Per-parallelism setup.
# ----------------------------------------------------------------------------
def _make_model(model_name: str, tp: int, moe_tp: int, ep: int):
    model = _model_defaults(model_name, tp, moe_tp, ep)
    eff_moe_tp, eff_ep = _effective_moe_parallelism(model, moe_tp, ep)
    if (eff_moe_tp, eff_ep) != (moe_tp, ep):
        model = _model_defaults(model_name, tp, eff_moe_tp, eff_ep)
    return model


def _par_label(tp: int, ep: int) -> str:
    return f"tp{tp}" if ep == 1 else f"tp{tp}_ep{ep}"


# ----------------------------------------------------------------------------
# Plot builders.
# ----------------------------------------------------------------------------
def _style_axes(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls=":", alpha=0.4)


def plot_phase_grid(
    *,
    phase: str,
    x_label: str,
    cases,
    model_name: str,
    layerwise_df: pd.DataFrame,
    backend,
    database,
    rc,
    fpm_root: Path,
    out_path: Path,
):
    """Build the ctx or gen image (subplot per parallelism x past_kv).

    For gen, two lines are drawn from ``database``: the uncalibrated decode lookup
    (plum) and the batch-calibrated decode (orange), so the calibration's effect
    is visible against the FPM points.
    """

    is_ctx = phase == "ctx"
    lw_phase = "ctx" if is_ctx else "gen"
    sweep_col = "new_tokens" if is_ctx else "batch_size"

    # Gather the AIC past_kv grid (union across parallelisms) and per-parallelism rows.
    pk_grid = sorted(int(v) for v in layerwise_df[layerwise_df["phase"] == lw_phase]["past_kv"].dropna().unique())
    if not pk_grid:
        print(f"[{phase}] no layerwise rows; skipping")
        return

    par_keys = []  # (tp, moe_tp, ep, fpm_path)
    for case in cases:
        par_keys.append((case.tp, case.moe_tp, case.ep, fpm_root / case.fpm))

    n_rows = len(par_keys)
    n_cols = len(pk_grid)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(max(3.0 * n_cols, 6), max(2.6 * n_rows, 4)), squeeze=False)

    for r, (tp, moe_tp, ep, fpm_path) in enumerate(par_keys):
        model = _make_model(model_name, tp, moe_tp, ep)

        # FPM points for this parallelism, binned to nearest AIC past_kv.
        fpm_by_pk = {pk: {} for pk in pk_grid}  # pk -> {x: [latencies]}
        if fpm_path.is_file():
            fpm = pd.read_csv(fpm_path)
            if is_ctx:
                ctx = fpm[fpm["phase"] == "context"]
                for _, row in ctx.iterrows():
                    x = int(row["ctx_tokens"])
                    pk_val = float(row.get("ctx_kv_tokens", 0) or 0)
                    pk = _nearest_in_log(pk_val, pk_grid)
                    fpm_by_pk[pk].setdefault(x, []).append(float(row["latency_ms"]))
            else:
                dec = fpm[fpm["phase"] == "decode"]
                for _, row in dec.iterrows():
                    x = int(row["decode_requests"])
                    pk_val = float(row.get("mean_decode_kv_tokens", 0) or 0)
                    pk = _nearest_in_log(pk_val, pk_grid)
                    fpm_by_pk[pk].setdefault(x, []).append(float(row["latency_ms"]))

        for c, pk in enumerate(pk_grid):
            ax = axes[r][c]
            _style_axes(ax)

            # 1. Exact collected layerwise points (raw latencies from the CSV,
            #    no lookup/interpolation/smoothing).
            raw_pts = _raw_collected_points(layerwise_df, tp, lw_phase, sweep_col, pk, moe_tp=moe_tp, ep=ep)
            if raw_pts:
                ex_x = sorted(raw_pts)
                ex_y = [raw_pts[x] for x in ex_x]
                ax.scatter(ex_x, ex_y, s=28, color="C0", zorder=3, label="layerwise collected")

            fp = fpm_by_pk.get(pk, {})

            # 2. AIC line. Sweep over the collected x-range AND out to the FPM
            #    x-range so the line spans the red crosses. The portion beyond the
            #    collected data is extrapolated and drawn dashed.
            if raw_pts:
                collected_max = max(raw_pts)
                x_lo = min(raw_pts)
                x_hi = collected_max
                if fp:
                    x_lo = min(x_lo, min(fp))
                    x_hi = max(x_hi, max(fp))
                # 2a. No-comm decode line (gen only): legacy behavior that drops the
                #     tensor-parallel all-reduce, so the lift from adding comm is visible.
                if not is_ctx:
                    cx, cy = [], []
                    for x in _dense_x(x_lo, x_hi):
                        y = _aic_gen(backend, model, database, rc, x, pk, comm=False)
                        if y is not None and y > 0:
                            cx.append(x)
                            cy.append(y)
                    if cx:
                        ax.plot(cx, cy, color="plum", lw=1.8, alpha=0.9, zorder=2, label="AIC no comm")
                solid_x, solid_y, extra_x, extra_y = [], [], [], []
                for x in _dense_x(x_lo, x_hi):
                    y = (
                        _aic_ctx(backend, model, database, rc, x, pk)
                        if is_ctx
                        else _aic_gen(backend, model, database, rc, x, pk, comm=True)
                    )
                    if y is None or y <= 0:
                        continue
                    if x <= collected_max:
                        solid_x.append(x)
                        solid_y.append(y)
                    else:
                        extra_x.append(x)
                        extra_y.append(y)
                main_label = "AIC interp" if is_ctx else "AIC + comm"
                if solid_x:
                    ax.plot(solid_x, solid_y, color="C1", lw=1.2, alpha=0.85, label=main_label)
                if extra_x:
                    # Bridge the gap so the dashed segment connects to the solid one.
                    bx = ([solid_x[-1]] + extra_x) if solid_x else extra_x
                    by = ([solid_y[-1]] + extra_y) if solid_y else extra_y
                    ax.plot(bx, by, color="C1", lw=1.2, alpha=0.85, ls="--", label="AIC extrapolated")

            # 3. FPM real points.
            if fp:
                fx = sorted(fp)
                fy = [float(np.median(fp[x])) for x in fx]
                ax.scatter(fx, fy, s=42, color="C3", marker="x", zorder=4, label="FPM real")

            if r == 0:
                ax.set_title(f"past_kv={pk}", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{_par_label(tp, ep)}\nlatency (ms)", fontsize=9)
            if r == n_rows - 1:
                ax.set_xlabel(x_label, fontsize=8)
            ax.tick_params(labelsize=7)

    # Single shared legend, collected across ALL subplots and deduped (a series
    # like "FPM real" may be absent from the top-left subplot).
    seen: dict[str, object] = {}
    for row_axes in axes:
        for ax in row_axes:
            for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
                seen.setdefault(label, handle)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.suptitle(f"{model_name}  |  {phase.upper()}  |  FPM vs AIC layerwise", fontsize=12, y=0.995)
    if seen:
        fig.legend(
            list(seen.values()),
            list(seen.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=len(seen),
            fontsize=10,
            frameon=True,
        )
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[{phase}] wrote {out_path}  ({n_rows}x{n_cols} subplots)")


def plot_mixed(
    *,
    cases,
    model_name: str,
    layerwise_df: pd.DataFrame,
    backend,
    database,
    rc,
    fpm_root: Path,
    out_path: Path,
    outlier_sigma: float = 3.0,
):
    """Build the mixed image: one AIC-vs-FPM parity subplot per parallelism."""

    par_keys = [(c.tp, c.moe_tp, c.ep, fpm_root / c.fpm) for c in cases]
    n = len(par_keys)
    ncol = min(n, 2)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.0 * ncol, 5.0 * nrow), squeeze=False)
    flat = [axes[i // ncol][i % ncol] for i in range(nrow * ncol)]

    for idx, (tp, moe_tp, ep, fpm_path) in enumerate(par_keys):
        ax = flat[idx]
        _style_axes(ax)
        model = _make_model(model_name, tp, moe_tp, ep)

        # Collect (fpm, aic, ctx_tokens) per mixed step so we can bucket by prefill size.
        recs: list[tuple[float, float, int]] = []
        if fpm_path.is_file():
            fpm = pd.read_csv(fpm_path)
            mixed = fpm[fpm["phase"] == "mixed"]
            for _, row in mixed.iterrows():
                ctx_tokens = int(row["ctx_tokens"])
                ctx_requests = max(int(row["ctx_requests"]), 1)
                ctx_kv_tokens = int(float(row.get("ctx_kv_tokens", 0) or 0))
                ctx_prefix_tokens = round(ctx_kv_tokens / ctx_requests)
                gen_tokens = int(row["decode_requests"])
                mean_decode_kv = float(row["mean_decode_kv_tokens"])
                y = _aic_mixed(
                    backend,
                    model,
                    database,
                    rc,
                    ctx_tokens=ctx_tokens,
                    gen_tokens=gen_tokens,
                    mean_decode_kv=mean_decode_kv,
                    ctx_prefix_tokens=ctx_prefix_tokens,
                    ctx_requests=ctx_requests,
                )
                f = float(row["latency_ms"])
                if y is not None and y > 0 and f > 0:
                    recs.append((f, y, ctx_tokens + ctx_kv_tokens))
        recs, n_dropped = _filter_outliers_by_sigma(recs, lambda r: (r[0], r[1]), outlier_sigma)
        if n_dropped:
            print(
                f"[mixed] {_par_label(tp, ep)}: dropped {n_dropped} outlier(s) "
                f"(>{outlier_sigma:g} sigma log-ratio error)"
            )

        title = f"{_par_label(tp, ep)} mixed"
        if recs:
            lo = min(min(f for f, _, _ in recs), min(a for _, a, _ in recs))
            hi = max(max(f for f, _, _ in recs), max(a for _, a, _ in recs))
            ax.plot([lo, hi], [lo, hi], color="k", ls="--", lw=1.0, zorder=2, label="y = x")
            # One colored series per prefill-size bucket, with per-bucket MAPE.
            print(f"[mixed] {_par_label(tp, ep)} per-prefill-size MAPE:")
            for (b_lo, b_hi), b_label, b_color in zip(_PREFILL_BUCKETS, _PREFILL_LABELS, _bucket_colors(), strict=True):
                pts = [(f, a) for f, a, c in recs if b_lo < c <= b_hi]
                if not pts:
                    continue
                fx = [p[0] for p in pts]
                ay = [p[1] for p in pts]
                bmape = float(np.mean([abs(a / f - 1.0) for f, a in pts]) * 100.0)
                ax.scatter(
                    fx,
                    ay,
                    s=26,
                    color=b_color,
                    alpha=0.75,
                    zorder=3,
                    label=f"ctx {b_label} (n={len(pts)}, {bmape:.0f}%)",
                )
                print(f"    ctx {b_label:>8}: n={len(pts):4d}  MAPE={bmape:5.1f}%")
            mape = float(np.mean([abs(a / f - 1.0) for f, a, _ in recs]) * 100.0)
            title += f"  (n={len(recs)}, MAPE={mape:.1f}%)"
            ax.legend(fontsize=8, title="prefill ctx = new+prefix (tokens)", title_fontsize=8, loc="upper left")
        else:
            ax.text(0.5, 0.5, "no mixed FPM data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("FPM latency (ms)", fontsize=9)
        ax.set_ylabel("AIC latency (ms)", fontsize=9)
        ax.tick_params(labelsize=8)

    for idx in range(n, nrow * ncol):
        flat[idx].axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.suptitle(f"{model_name}  |  MIXED  |  AIC vs FPM parity", fontsize=12, y=0.995)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[mixed] wrote {out_path}  ({n} subplots)")


def plot_mixed_errormap(
    *,
    cases,
    model_name: str,
    backend,
    database,
    rc,
    fpm_root: Path,
    out_path: Path,
    outlier_sigma: float = 3.0,
):
    """Mixed error map: x=ctx (prefill) tokens, y=decode tokens, color=signed error.

    Color is (AIC/FPM - 1): red = AIC over-predicts, blue = AIC under-predicts.
    Reveals where in the (prefill, decode) plane the error concentrates.
    """

    par_keys = [(c.tp, c.moe_tp, c.ep, fpm_root / c.fpm) for c in cases]
    n = len(par_keys)
    ncol = min(n, 2)
    nrow = math.ceil(n / ncol)

    # Collect (ctx_tokens, decode_tokens, signed_err) per parallelism first so the
    # color scale can be shared across subplots.
    per_tp: dict[int, list[tuple[int, int, float]]] = {}
    for idx, (tp, moe_tp, ep, fpm_path) in enumerate(par_keys):
        model = _make_model(model_name, tp, moe_tp, ep)
        # (ctx_tokens, decode_tokens, fpm, aic) so the sigma filter can use f/y.
        raw: list[tuple[int, int, float, float]] = []
        if fpm_path.is_file():
            fpm = pd.read_csv(fpm_path)
            mixed = fpm[fpm["phase"] == "mixed"]
            for _, row in mixed.iterrows():
                ctx_tokens = int(row["ctx_tokens"])
                ctx_requests = max(int(row["ctx_requests"]), 1)
                ctx_kv_tokens = int(float(row.get("ctx_kv_tokens", 0) or 0))
                ctx_prefix_tokens = round(ctx_kv_tokens / ctx_requests)
                decode_tokens = int(row["decode_requests"])
                mean_decode_kv = float(row["mean_decode_kv_tokens"])
                y = _aic_mixed(
                    backend,
                    model,
                    database,
                    rc,
                    ctx_tokens=ctx_tokens,
                    gen_tokens=decode_tokens,
                    mean_decode_kv=mean_decode_kv,
                    ctx_prefix_tokens=ctx_prefix_tokens,
                    ctx_requests=ctx_requests,
                )
                f = float(row["latency_ms"])
                if y is not None and y > 0 and f > 0 and ctx_tokens > 0:
                    raw.append((ctx_tokens, decode_tokens, f, y))
        raw, _ = _filter_outliers_by_sigma(raw, lambda r: (r[2], r[3]), outlier_sigma)
        per_tp[idx] = [(ct, dt, y / f - 1.0) for ct, dt, f, y in raw]

    all_err = [e for recs in per_tp.values() for *_, e in recs]
    vmax = 0.5
    if all_err:
        vmax = float(np.percentile(np.abs(all_err), 95))
        vmax = max(0.1, min(vmax, 1.0))

    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0 * ncol, 5.5 * nrow), squeeze=False)
    flat = [axes[i // ncol][i % ncol] for i in range(nrow * ncol)]
    for idx, (tp, moe_tp, ep, _fpm) in enumerate(par_keys):
        ax = flat[idx]
        recs = per_tp[idx]
        if recs:
            xs = [r[0] for r in recs]
            ys = [r[1] for r in recs]
            cs = [r[2] for r in recs]
            sc = ax.scatter(
                xs,
                ys,
                c=cs,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                s=34,
                alpha=0.85,
                edgecolors="k",
                linewidths=0.2,
            )
            cb = fig.colorbar(sc, ax=ax)
            cb.set_label("AIC/FPM - 1  (red=over, blue=under)", fontsize=8)
            ax.set_xscale("log")
        else:
            ax.text(0.5, 0.5, "no mixed FPM data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{_par_label(tp, ep)} mixed  (color clip ±{vmax * 100:.0f}%)", fontsize=10)
        ax.set_xlabel("ctx tokens (prefill, new+0)", fontsize=9)
        ax.set_ylabel("decode tokens (requests)", fontsize=9)
        ax.grid(True, ls=":", alpha=0.4)
        ax.tick_params(labelsize=8)

    for idx in range(n, nrow * ncol):
        flat[idx].axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.suptitle(f"{model_name}  |  MIXED error map  |  ctx vs decode tokens", fontsize=12, y=0.995)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[mixed] wrote {out_path}  ({n} subplots)")


def plot_allreduce_comparison(*, database, out_path: Path):
    """Per-all-reduce latency vs message size (bytes): standalone vs fused allreduce_rms.

    Plotted against message size in bytes (not tokens) so it is model-agnostic --
    the all-reduce cost is a function of bytes on the wire, not the specific model
    hidden size.
    """

    import aiconfigurator.sdk.common as common

    real_db = getattr(database, "real_database", database)
    system = getattr(real_db, "system", "")
    backend = getattr(real_db, "backend", "")
    version = getattr(real_db, "version", "")
    tps = [2, 4, 8]
    dtype_bytes = 2  # bfloat16
    # message size in elements (= tokens * hidden); fused table was collected at
    # hidden=4096, and the query is driven by message size, so sweep size directly.
    hidden_ref = 4096
    sizes = [hidden_ref * (2**i) for i in range(0, 14)]  # 4096 .. ~33M elements

    def _val(x):
        if x is None:
            return None
        if hasattr(x, "latency"):
            return float(x.latency)
        if isinstance(x, tuple):
            return float(x[0])
        return float(x)

    fig, axes = plt.subplots(1, len(tps), figsize=(5.2 * len(tps), 4.6), squeeze=False)
    for i, tp in enumerate(tps):
        ax = axes[0][i]
        _style_axes(ax)
        sa, fr = [], []
        for size in sizes:
            try:
                sa.append(_val(real_db.query_custom_allreduce(common.CommQuantMode.half, tp, size)))
            except Exception:
                sa.append(None)
            try:
                fr.append(_val(real_db.query_allreduce_rms(common.CommQuantMode.half, tp, size, hidden_ref)))
            except Exception:
                fr.append(None)
        bytes_x = [size * dtype_bytes for size in sizes]
        sx = [b for b, v in zip(bytes_x, sa, strict=True) if v]
        sy = [v for v in sa if v]
        fx = [b for b, v in zip(bytes_x, fr, strict=True) if v]
        fy = [v for v in fr if v]
        if sx:
            ax.plot(sx, sy, "o-", color="C3", lw=1.4, label="custom_allreduce (standalone)")
        if fx:
            ax.plot(fx, fy, "s-", color="C0", lw=1.4, label="allreduce_rms (fused)")
        ax.set_title(f"tp{tp}", fontsize=10)
        ax.set_xlabel("message size (bytes, bf16)", fontsize=9)
        if i == 0:
            ax.set_ylabel("per-all-reduce latency (ms)", fontsize=9)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.suptitle(
        f"{system} {backend} {version}  |  per-all-reduce: standalone custom vs fused allreduce_rms",
        fontsize=12,
        y=0.99,
    )
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[allreduce] wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layerwise", type=Path, required=True, help="Layerwise CSV (AIC measured data).")
    parser.add_argument("--model", default="Qwen/Qwen3-32B", help="Model name as it appears in the layerwise CSV.")
    parser.add_argument("--fpm-root", type=Path, default=Path("fpm_golden_runs"))
    parser.add_argument("--systems-root", default="src/aiconfigurator/systems")
    parser.add_argument("--moe-perf-file", type=Path, default=None, help="MoE overlay (only for MoE models).")
    parser.add_argument("--out-dir", type=Path, default=Path("fpm_vs_aic_charts"))
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=256)
    parser.add_argument("--phases", default="ctx,gen,mixed,allreduce", help="Comma list of phases to plot.")
    parser.add_argument(
        "--mixed-outlier-sigma",
        type=float,
        default=3.0,
        help="Drop mixed points whose log(AIC/FPM) error is more than this many std devs "
        "from the mean error (per parallelism). 0 disables filtering.",
    )
    parser.add_argument(
        "--repair-decode-kv-above",
        type=int,
        default=8192,
        help="Treat collected decode rows for --model with past_kv >= this as corrupt and "
        "linearly extrapolate them from past_kv 2048/4096. 0 disables the repair.",
    )
    args = parser.parse_args()

    if not args.layerwise.is_file():
        raise SystemExit(f"Layerwise CSV not found: {args.layerwise}")

    cases = [c for c in _all_cases() if c.model == args.model]
    if not cases:
        raise SystemExit(
            f"No comparison cases found for model {args.model!r}. "
            "Known models: " + ", ".join(sorted({c.model for c in _all_cases()}))
        )
    # Stable ordering by parallelism.
    cases = sorted(cases, key=lambda c: (c.tp, c.ep, c.moe_tp))

    systems_root = args.systems_root
    if args.moe_perf_file is not None:
        if not args.moe_perf_file.is_file():
            raise SystemExit(f"MoE perf file not found: {args.moe_perf_file}")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        systems_root = _prepare_moe_overlay_systems_root(
            systems_root=systems_root,
            moe_perf_file=args.moe_perf_file,
            output=args.out_dir / "overlay.csv",
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    layerwise_df = pd.read_csv(args.layerwise)
    layerwise_df = layerwise_df[layerwise_df["model"] == args.model]
    if layerwise_df.empty:
        raise SystemExit(f"No layerwise rows for model {args.model!r} in {args.layerwise}")

    real_db = PerfDatabase("b300_sxm", "vllm", "0.20.1", systems_root=systems_root)
    # `database` applies the high-KV decode repair (linear extrapolation from
    # past_kv 2048/4096), so the decode lookup is corruption-free. The gen chart's
    # calibrated vs uncalibrated lines are then both computed from this DB.
    repair_above = (
        args.repair_decode_kv_above if args.repair_decode_kv_above and args.repair_decode_kv_above > 0 else None
    )
    database = _LayerwiseDatabase(
        args.layerwise,
        real_db,
        repair_decode_kv_above=repair_above,
        repair_decode_models=(args.model,) if repair_above else (),
        repair_decode_anchor_kvs=(2048, 4096),
    )
    backend = VLLMBackend()
    rc = RuntimeConfig(
        vllm_max_num_batched_tokens=args.vllm_max_num_batched_tokens,
        vllm_max_num_seqs=args.vllm_max_num_seqs,
    )
    vllm_backend._USE_LAYERWISE = True

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    if "ctx" in phases:
        plot_phase_grid(
            phase="ctx",
            x_label="new tokens",
            cases=cases,
            model_name=args.model,
            layerwise_df=layerwise_df,
            backend=backend,
            database=database,
            rc=rc,
            fpm_root=args.fpm_root,
            out_path=args.out_dir / "fpm_vs_aic_ctx.png",
        )
    if "gen" in phases:
        plot_phase_grid(
            phase="gen",
            x_label="decode batch size",
            cases=cases,
            model_name=args.model,
            layerwise_df=layerwise_df,
            backend=backend,
            database=database,
            rc=rc,
            fpm_root=args.fpm_root,
            out_path=args.out_dir / "fpm_vs_aic_gen.png",
        )
    if "mixed" in phases:
        plot_mixed(
            cases=cases,
            model_name=args.model,
            layerwise_df=layerwise_df,
            backend=backend,
            database=database,
            rc=rc,
            fpm_root=args.fpm_root,
            out_path=args.out_dir / "fpm_vs_aic_mixed.png",
            outlier_sigma=args.mixed_outlier_sigma,
        )
        plot_mixed_errormap(
            cases=cases,
            model_name=args.model,
            backend=backend,
            database=database,
            rc=rc,
            fpm_root=args.fpm_root,
            out_path=args.out_dir / "fpm_vs_aic_mixed_errormap.png",
            outlier_sigma=args.mixed_outlier_sigma,
        )
    if "allreduce" in phases:
        plot_allreduce_comparison(
            database=database,
            out_path=args.out_dir / "allreduce_custom_vs_fused.png",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

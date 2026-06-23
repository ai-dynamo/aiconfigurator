#!/usr/bin/env python3
"""Zoom one (model, parallelism) mixed-step case and plot FPM latency against
every available step-level variable, to see what actually drives mixed latency.

Motivation: on the MoE mixed parity chart, FPM latency clusters and is nearly
flat in scheduled context tokens, while AIC rises with ctx tokens. The standard
parity plot has no axis that exposes which variable (if any) latency tracks. This
tool facets latency vs each variable so the driver (or its absence) is visible.

Key handling:
- Warmup quarantine: points are colored by stream index; the first ``--drop-warmup``
  steps (cold start / first CUDA-graph capture) are drawn as red rings and can be
  excluded from the printed correlations. On thin runs the largest latencies are
  cold-start artifacts, not MoE physics.
- isl/osl are NOT available at step granularity. ``ctx_kv_tokens`` (prefill KV
  already cached) and ``mean_decode_kv_tokens`` (decode context length) are used
  as proxies and labeled as such.
- Optional AIC overlay: pass ``--layerwise`` to overlay AIC's mixed prediction
  (reuses the chart tool's ``_aic_mixed``); without it only FPM is plotted.

Example:
    uv run python collector/layerwise/diagnostics/plot_mixed_latency_drivers.py \
      --fpm fpm_golden_runs/fpm_upfront_qwen36_moe_full_once_20260613_201336/tp4_ep4_past4096/fpm_metrics_phase.csv \
      --model "Qwen/Qwen3.6-35B-A3B" --tp 4 --ep 4 --drop-warmup 5 \
      --out fpm_vs_aic_charts_qwen36/mixed_latency_drivers_tp4_ep4.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Step-level variables to facet against, with human labels. isl/osl are not
# available per step; the kv columns are the closest proxies.
VARIABLES: list[tuple[str, str]] = [
    ("ctx_tokens", "scheduled prefill tokens (this step)"),
    ("decode_requests", "decode batch size"),
    ("ctx_requests", "# prefill requests in step"),
    ("ctx_kv_tokens", "prefill KV already cached  [ISL proxy]"),
    ("mean_decode_kv_tokens", "decode context length  [ISL+gen proxy]"),
]


def _load_mixed(fpm_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(fpm_csv)
    m = df[df["phase"] == "mixed"].copy().reset_index(drop=True)
    m["stream_index"] = m.index
    return m


def _aic_overlay(args, mixed: pd.DataFrame) -> list[float] | None:
    """Best-effort AIC mixed prediction per FPM row. Returns None if unavailable."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
        from plot_fpm_vs_aic import _aic_mixed, _LayerwiseDatabase, _make_model  # type: ignore

        from aiconfigurator.sdk.backends import vllm_backend
        from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend
        from aiconfigurator.sdk.config import RuntimeConfig
        from aiconfigurator.sdk.perf_database import PerfDatabase
    except Exception as exc:  # pragma: no cover - overlay is optional
        print(f"[aic] overlay unavailable ({exc}); plotting FPM only")
        return None

    real_db = PerfDatabase("b300_sxm", "vllm", "0.20.1", systems_root=args.systems_root)
    database = _LayerwiseDatabase(args.layerwise, real_db)
    backend = VLLMBackend()
    rc = RuntimeConfig(
        vllm_max_num_batched_tokens=args.vllm_max_num_batched_tokens,
        vllm_max_num_seqs=args.vllm_max_num_seqs,
    )
    vllm_backend._USE_LAYERWISE = True
    model = _make_model(args.model, args.tp, args.moe_tp, args.ep)
    preds: list[float] = []
    for _, row in mixed.iterrows():
        ctx_requests = max(int(row["ctx_requests"]), 1)
        ctx_kv = int(float(row.get("ctx_kv_tokens", 0) or 0))
        try:
            preds.append(
                _aic_mixed(
                    backend,
                    model,
                    database,
                    rc,
                    ctx_tokens=int(row["ctx_tokens"]),
                    gen_tokens=int(row["decode_requests"]),
                    mean_decode_kv=float(row["mean_decode_kv_tokens"]),
                    ctx_prefix_tokens=round(ctx_kv / ctx_requests),
                    ctx_requests=ctx_requests,
                )
            )
        except Exception:
            preds.append(float("nan"))
    return preds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fpm", type=Path, required=True, help="Path to fpm_metrics_phase.csv")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--moe-tp", type=int, default=1)
    ap.add_argument("--ep", type=int, default=1)
    ap.add_argument("--drop-warmup", type=int, default=0, help="Exclude first K steps from printed stats (cold start).")
    ap.add_argument("--layerwise", type=Path, default=None, help="Layerwise CSV to overlay AIC prediction (optional).")
    ap.add_argument("--systems-root", default="src/aiconfigurator/systems")
    ap.add_argument("--vllm-max-num-batched-tokens", type=int, default=2048)
    ap.add_argument("--vllm-max-num-seqs", type=int, default=128, help="Match the FPM run (golden qwen36 = 128).")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    mixed = _load_mixed(args.fpm)
    if mixed.empty:
        raise SystemExit(f"No mixed rows in {args.fpm}")
    aic = _aic_overlay(args, mixed) if args.layerwise else None

    kept = mixed.iloc[args.drop_warmup :]
    print(f"n_mixed={len(mixed)} (dropping first {args.drop_warmup} for stats -> n={len(kept)})")
    lat = kept["latency_ms"]
    print(f"latency mean={lat.mean():.2f} ms  CV={lat.std() / lat.mean():.3f}")
    for col, _ in VARIABLES:
        print(f"  corr(latency, {col}) = {kept['latency_ms'].corr(kept[col]):.2f}")

    n = len(VARIABLES)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.2 * nrow), squeeze=False)
    flat = [axes[i // ncol][i % ncol] for i in range(nrow * ncol)]
    warm = mixed["stream_index"] < args.drop_warmup

    for idx, (col, label) in enumerate(VARIABLES):
        ax = flat[idx]
        sc = ax.scatter(
            mixed[col], mixed["latency_ms"], c=mixed["stream_index"], cmap="viridis", s=28, alpha=0.85, label="FPM"
        )
        if warm.any():
            ax.scatter(
                mixed[col][warm],
                mixed["latency_ms"][warm],
                s=90,
                facecolors="none",
                edgecolors="red",
                linewidths=1.3,
                label=f"warmup (first {args.drop_warmup})",
            )
        if aic is not None:
            ax.scatter(mixed[col], aic, c="orange", marker="x", s=30, alpha=0.8, label="AIC pred")
        r = kept["latency_ms"].corr(kept[col])
        ax.set_title(f"{label}\ncorr(latency)={r:.2f}", fontsize=9)
        ax.set_xlabel(col)
        ax.set_ylabel("latency_ms")
        ax.grid(True, ls=":", alpha=0.4)
        if idx == 0:
            ax.legend(fontsize=7, loc="best")
    for j in range(n, len(flat)):
        flat[j].axis("off")
    cbar = fig.colorbar(sc, ax=axes, shrink=0.6, location="right")
    cbar.set_label("stream index (step order)")
    fig.suptitle(f"{args.model} | tp{args.tp} moe_tp{args.moe_tp} ep{args.ep} | mixed latency drivers", fontsize=12)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

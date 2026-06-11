#!/usr/bin/env python3
"""Plot AIC-vs-FPM comparison rows with mixed-step token histograms.

The comparison CSVs are produced by ``compare_aic_layerwise_fpm.py``. This
script is intentionally diagnostic-only: it visualizes an existing comparison
without re-running AIC or FPM.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_phase(path: Path | None, label: str) -> pd.DataFrame:
    """Load a comparison CSV and attach a display group label."""

    if path is None:
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["plot_group"] = label
    return frame


def _phase_rows(frame: pd.DataFrame, phases: set[str]) -> pd.DataFrame:
    """Return rows whose normalized phase belongs to ``phases``."""

    if frame.empty or "phase" not in frame:
        return pd.DataFrame()
    phase = frame["phase"].astype(str).str.lower()
    return frame[phase.isin(phases)].copy()


def _plot_latency_scatter(ax: plt.Axes, frame: pd.DataFrame, title: str) -> None:
    """Plot AIC latency against FPM latency for one phase group."""

    if frame.empty:
        ax.set_title(f"{title}: no rows")
        ax.axis("off")
        return

    x = frame["fpm_ms"].astype(float)
    y = frame["aic_ms"].astype(float)
    err = frame["error_pct"].astype(float)
    colors = np.where(err.abs() <= 5.0, "#2ca25f", np.where(err.abs() <= 20.0, "#f0ad4e", "#d95f02"))
    ax.scatter(x, y, c=colors, s=28, alpha=0.8, edgecolors="none")

    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    padding = max((hi - lo) * 0.05, 1.0)
    ax.plot([lo - padding, hi + padding], [lo - padding, hi + padding], color="#555555", linewidth=1)
    ax.set_xlim(lo - padding, hi + padding)
    ax.set_ylim(lo - padding, hi + padding)
    ax.set_xlabel("FPM latency (ms)")
    ax.set_ylabel("AIC latency (ms)")
    ax.set_title(f"{title}: n={len(frame)}, MAPE={err.abs().mean():.1f}%")
    ax.grid(True, alpha=0.25)


def _hist_bins(values: pd.Series) -> np.ndarray | int:
    """Choose readable histogram bins for token counts."""

    clean = values.dropna().astype(float)
    if clean.empty:
        return 10
    max_value = float(clean.max())
    if max_value <= 64:
        return np.arange(-0.5, max_value + 1.5, 1.0)
    return min(40, max(8, int(np.sqrt(len(clean)) * 2)))


def _plot_token_hist(ax: plt.Axes, mixed: pd.DataFrame, column: str, title: str, xlabel: str) -> None:
    """Plot a histogram for one mixed-step token-count column."""

    if mixed.empty or column not in mixed:
        ax.set_title(f"{title}: no rows")
        ax.axis("off")
        return

    values = mixed[column].dropna().astype(float)
    ax.hist(values, bins=_hist_bins(values), color="#4c78a8", alpha=0.85)
    ax.set_title(f"{title}: n={len(values)}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("rows")
    ax.grid(True, axis="y", alpha=0.25)


def _plot_error_hist(ax: plt.Axes, mixed: pd.DataFrame) -> None:
    """Plot the mixed-step error distribution."""

    if mixed.empty:
        ax.set_title("Mixed error: no rows")
        ax.axis("off")
        return

    errors = mixed["error_pct"].dropna().astype(float)
    ax.hist(errors, bins=min(50, max(10, int(np.sqrt(len(errors)) * 2))), color="#8172b2", alpha=0.85)
    ax.axvline(0.0, color="#333333", linewidth=1)
    ax.set_title(f"Mixed error: median={errors.median():.1f}%, MAPE={errors.abs().mean():.1f}%")
    ax.set_xlabel("AIC error (%)")
    ax.set_ylabel("rows")
    ax.grid(True, axis="y", alpha=0.25)


def plot_comparison(
    context: pd.DataFrame,
    decode: pd.DataFrame,
    mixed_source: pd.DataFrame,
    output: Path,
    title: str,
) -> None:
    """Create the comparison figure and save PNG/SVG variants."""

    context_rows = _phase_rows(context, {"ctx", "context"})
    decode_rows = _phase_rows(decode, {"gen", "decode"})
    mixed_rows = _phase_rows(mixed_source, {"mixed"})

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    _plot_latency_scatter(axes[0, 0], context_rows, "Context")
    _plot_latency_scatter(axes[0, 1], decode_rows, "Decode")
    _plot_latency_scatter(axes[0, 2], mixed_rows, "Mixed")
    _plot_token_hist(axes[1, 0], mixed_rows, "ctx_tokens", "Mixed ctx tokens", "context tokens")
    _plot_token_hist(axes[1, 1], mixed_rows, "decode_requests", "Mixed decode tokens", "decode tokens")
    _plot_error_hist(axes[1, 2], mixed_rows)

    fig.suptitle(title, fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=160)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    """Parse CLI arguments and write the requested plot."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-csv", type=Path, required=True)
    parser.add_argument("--decode-csv", type=Path, required=True)
    parser.add_argument("--mixed-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output path stem; .png and .svg are written.")
    parser.add_argument("--title", default="FPM vs AIC layerwise")
    args = parser.parse_args()

    plot_comparison(
        _load_phase(args.context_csv, "context"),
        _load_phase(args.decode_csv, "decode"),
        _load_phase(args.mixed_csv, "mixed"),
        args.output,
        args.title,
    )


if __name__ == "__main__":
    main()

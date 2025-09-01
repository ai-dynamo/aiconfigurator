# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def _split(spec: str):
    metric, stat = (spec.split("::", 1) + ["avg"])[:2]
    return metric, stat

class ParetoPlot:
    name = "pareto"

    def __init__(self, x_metric: str, y_metric: str, *,
                 merge: bool = False, num_gpus: int | None = None,
                 plot_label: str | None = None, show_cc_label: bool = True,
                 expand_x: bool = False):
        self._x_spec = x_metric
        self._y_spec = y_metric
        self._merge = merge
        self._num_gpus = num_gpus
        self._override_label = plot_label
        self.show_cc_label = show_cc_label
        self._expand_x = expand_x
        self._series: list[tuple[str, pd.DataFrame]] = []
        self._optimal_points: list[tuple[str, pd.DataFrame]] = []

    def add_series(self, label: str, df: pd.DataFrame):
        self._series.append((label, df))

    def add_optimal_point(self, label: str, df: pd.DataFrame):
        """Add optimal configuration point to be plotted with special markers."""
        self._optimal_points.append((label, df))

    def _col(self, df: pd.DataFrame, spec: str):
        m, s = _split(spec)
        col = f"{m}_{s}"
        if col not in df.columns and s != "avg":
            col = f"{m}_avg"
        return df[col]

    @staticmethod
    def _pareto_front(pts: np.ndarray) -> np.ndarray:
        keep = []
        for p in pts:
            if not np.any(np.all(pts >= p, axis=1) & np.any(pts > p, axis=1)):
                keep.append(p)
        keep = np.array(keep)
        return keep[np.argsort(keep[:, 0])]

    def _merge_series(self) -> pd.DataFrame:
        return pd.concat([df for _, df in self._series], ignore_index=True)

    def render(self, ax: plt.Axes, **opts):
        x_label_override = opts.get("x_label")
        y_label_override = opts.get("y_label")

        if self._merge:
            merged_df = self._merge_series()
            label = self._override_label or "merged"
            self._series = [(label, merged_df)]

        all_x_vals = pd.Series(dtype=float)

        for label, df in self._series:
            if self._override_label:
                label = self._override_label

            x_vals = self._col(df, self._x_spec)
            y_vals = self._col(df, self._y_spec)
            if self._num_gpus and self._num_gpus > 0:
                y_vals = y_vals / self._num_gpus

            all_x_vals = pd.concat([all_x_vals, x_vals])

            scat = ax.scatter(x_vals, y_vals, label=label, s=20)
            colour = scat.get_facecolor()[0]

            front = self._pareto_front(np.column_stack([x_vals, y_vals]))
            ax.plot(front[:, 0], front[:, 1], "-", linewidth=1, color=colour)

            if self.show_cc_label:
                for xv, yv, tag in zip(x_vals, y_vals, df.get("load_label", [""] * len(df))):
                    if tag:
                        ax.annotate(tag, (xv, yv), textcoords="offset points",
                                    xytext=(2, 2), fontsize=6, ha="left", va="bottom")

        # Plot optimal configuration points with special markers
        for label, df in self._optimal_points:
            if df.empty:
                continue
                
            x_vals = self._col(df, self._x_spec)
            y_vals = self._col(df, self._y_spec)
            # NOTE: Optimal points from pareto CSV already have per-GPU values, 
            # so we don't divide by num_gpus again to avoid double normalization

            all_x_vals = pd.concat([all_x_vals, x_vals])

            # Plot with distinctive markers (star shape, larger size, different color)
            ax.scatter(x_vals, y_vals, label=f"{label} (Optimal)", 
                      marker='*', s=150, c='red', edgecolors='black', linewidth=1, zorder=10)
            
            # Add labels for optimal points
            for xv, yv in zip(x_vals, y_vals):
                ax.annotate(f"Optimal\n{label}", (xv, yv), textcoords="offset points",
                           xytext=(5, 5), fontsize=8, ha="left", va="bottom", 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        if self._expand_x and not all_x_vals.empty:
            min_x = 0.0
            max_x = all_x_vals.max()
            span = max_x - all_x_vals.min()
            ax.set_xlim(min_x, max_x + span)

        ax.xaxis.set_major_locator(MaxNLocator(nbins=15))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=15))

        ax.set_xlabel(x_label_override or self._x_spec)
        default_y = self._y_spec + (" (per gpu)" if self._num_gpus else "")
        ax.set_ylabel(y_label_override or default_y)
        ax.set_title(opts.get("title", "Pareto"))
        ax.grid(True, alpha=0.3)
        ax.legend()

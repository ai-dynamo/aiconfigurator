# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
from bokeh.plotting import figure, save
from bokeh.models import ColumnDataSource, HoverTool, DatetimeTickFormatter
from bokeh.palettes import Category10, Category20, Turbo256
from bokeh.resources import INLINE


@dataclass
class TimeSeriesSpec:
    """
    Generic timeseries line spec (extensible for future metrics).
    - name: legend label
    - data: a DataFrame
    - x: column name for datetime-like values (pd.Timestamp or seconds epoch float)
    - y: column name for numeric values
    - color: optional hex color
    """
    name: str
    data: pd.DataFrame
    x: str
    y: str
    color: Optional[str] = None


def _choose_palette(n: int) -> List[str]:
    if n <= 10:
        return list(Category10[10][:n])
    if n <= 20:
        return list(Category20[20][:n])
    # For many lines, sample Turbo256
    step = max(1, len(Turbo256) // max(1, n))
    return [Turbo256[(i * step) % len(Turbo256)] for i in range(n)]


def _ensure_datetime(series: pd.Series) -> pd.Series:
    """
    Convert numeric epoch seconds or already-datetime series to pandas datetime (ns).
    If float seconds are provided (e.g., 0.01s spacing), convert with unit='s'.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, unit="s", errors="coerce")


def plot_gpu_timeseries(
    csv_path: Path | str,
    output_html_path: Path | str,
    title: str = "GPU Utilization (%)",
    *,
    y_label: str = "util %",
    extra_series: Optional[List[TimeSeriesSpec]] = None,
) -> Path:
    """
    Render a Bokeh HTML with one line per GPU showing utilization (%).
    CSV columns: timestamp (seconds), gpu_index (int), util_percent (0..100), mem_used_mb (int)
    """
    csv_path = Path(csv_path)
    output_html_path = Path(output_html_path)
    extra_series = extra_series or []

    df = pd.read_csv(csv_path)
    if not {"timestamp", "gpu_index", "util_percent"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: timestamp, gpu_index, util_percent")

    df["ts"] = _ensure_datetime(df["timestamp"])
    df = df.dropna(subset=["ts"]).sort_values(["gpu_index", "ts"])
    df["series"] = df["gpu_index"].apply(lambda g: f"GPU{g}")
    df["value"] = df["util_percent"]

    p = figure(
        x_axis_type="datetime",
        title=title,
        sizing_mode="stretch_width",
        height=380,
        tools="pan,wheel_zoom,box_zoom,reset,save,hover",
        active_scroll="wheel_zoom",
    )
    p.xaxis.formatter = DatetimeTickFormatter(
        seconds="%H:%M:%S.%3N", minutes="%H:%M:%S", hours="%m-%d %H:%M", days="%m-%d"
    )
    p.yaxis.axis_label = y_label
    p.grid.grid_line_alpha = 0.3

    gpus = sorted(df["gpu_index"].unique().tolist())
    colors = _choose_palette(len(gpus))

    for i, gpu in enumerate(gpus):
        sub = df[df["gpu_index"] == gpu]
        source = ColumnDataSource(
            dict(
                ts=sub["ts"],
                value=sub["value"],
                gpu=[f"GPU{gpu}"] * len(sub),
                series=[f"GPU{gpu}"] * len(sub),
            )
        )
        p.line(
            x="ts", y="value", source=source,
            line_width=2, color=colors[i],
            legend_label=f"GPU{gpu}", name=f"GPU{gpu}",
        )
        p.scatter(
            x="ts", y="value", source=source,
            marker="circle", size=3, alpha=0.55, color=colors[i],
            legend_label=f"GPU{gpu}", name=f"GPU{gpu}",
        )

    for j, spec in enumerate(extra_series):
        d = spec.data.copy()
        d["__ts"] = _ensure_datetime(d[spec.x])
        d = d.dropna(subset=["__ts"]).sort_values("__ts")
        color = spec.color or _choose_palette(len(extra_series))[j % max(1, len(extra_series))]
        src = ColumnDataSource(dict(ts=d["__ts"], value=d[spec.y], series=[spec.name] * len(d)))
        p.line(x="ts", y="value", source=src, line_width=2, color=color,
               legend_label=spec.name, name=spec.name)
        p.scatter(x="ts", y="value", source=src, marker="circle", size=3, alpha=0.55,
                  color=color, legend_label=spec.name, name=spec.name)

    hover = p.select_one(HoverTool)
    hover.tooltips = [
        ("time", "@ts{%F %T.%3N}"),
        ("series", "@series"),
        ("value", "@value{0.0}"),
    ]
    hover.formatters = {"@ts": "datetime"}
    hover.mode = "vline"

    p.legend.click_policy = "hide"
    p.legend.location = "top_left"

    # Save as HTML (inline resources)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    save(p, filename=str(output_html_path), title=title, resources=INLINE)
    return output_html_path


def plot_gpu_timeseries_bokeh_from_df(
    df: pd.DataFrame,
    output_html_path: Path | str,
    title: str = "GPU Utilization (%)",
    *,
    y_label: str = "util %",
    extra_series: Optional[List[TimeSeriesSpec]] = None,
) -> Path:
    """
    Same as above but takes an in-memory DataFrame with required columns.
    """
    tmp_csv = Path(output_html_path).with_suffix(".tmp.csv")
    df.to_csv(tmp_csv, index=False)
    try:
        return plot_gpu_timeseries(tmp_csv, output_html_path, title, y_label=y_label, extra_series=extra_series)
    finally:
        if tmp_csv.exists():
            try:
                tmp_csv.unlink()
            except Exception:
                pass

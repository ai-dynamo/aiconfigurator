#!/usr/bin/env python3
"""Reproduce AIC native validation for GPT-OSS-120B prefill FPM telemetry.

Inputs are aggregate Dynamo FPM rows with:
wall_time_s,prefill_tokens,prefill_kv_tokens,prefill_requests,...

The AIC runtime treats each aggregate prefill row as a homogeneous batch:
new_tokens_per_req = sum_prefill_tokens / num_prefill_requests
prefix_per_req = sum_prefill_kv_tokens / num_prefill_requests
using integer division in Rust.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize, PowerNorm


DEFAULT_CONFIG = {
    "schema_version": 1,
    "model_name": "openai/gpt-oss-120b",
    "system_name": "b200_sxm",
    "backend": "trtllm",
    # 1.3.0rc15 is marker-only in this checkout; 1.3.0rc10 is the latest complete DB.
    "backend_version": "1.3.0rc10",
    "tp_size": 1,
    "pp_size": 1,
    "moe_tp_size": 1,
    "moe_ep_size": 1,
    "attention_dp_size": 1,
    "weight_dtype": "nvfp4",
    "moe_dtype": "w4a8_mxfp4_mxfp8",
    "activation_dtype": "fp8",
    "kv_cache_dtype": "fp8",
    "kv_block_size": None,
    "nextn": None,
    "nextn_accept_rates": None,
    "extra": {},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(".codex-artifacts/trtllm_b200_gpt_oss_120b_mixed_fp4_fpms.csv"),
    )
    parser.add_argument("--outdir", type=Path, default=Path(".codex-artifacts"))
    parser.add_argument("--aiconfigurator-src", type=Path, default=Path("aiconfigurator/src"))
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, int | float]]:
    rows: list[dict[str, int | float]] = []
    with path.open(newline="") as f:
        for i, row in enumerate(csv.DictReader(f), 1):
            parsed = {
                key: int(value) if key != "wall_time_s" else float(value)
                for key, value in row.items()
            }
            parsed["row_id"] = i
            rows.append(parsed)
    return rows


def bucket_index(value: float, lo: float, hi: float, buckets_per_axis: int) -> int:
    if hi <= lo:
        return 0
    idx = int((value - lo) / (hi - lo) * buckets_per_axis)
    return max(0, min(idx, buckets_per_axis - 1))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def var_pop(values: list[float]) -> float:
    if not values:
        return float("nan")
    avg = mean(values)
    return sum((value - avg) ** 2 for value in values) / len(values)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def make_fpm(row: dict[str, int | float]) -> dict[str, object]:
    return {
        "version": 1,
        "wall_time": row["wall_time_s"],
        "scheduled_requests": {
            "num_prefill_requests": row["prefill_requests"],
            "sum_prefill_tokens": row["prefill_tokens"],
            "sum_prefill_kv_tokens": row["prefill_kv_tokens"],
            "num_decode_requests": 0,
            "sum_decode_kv_tokens": 0,
        },
        "queued_requests": {
            "num_prefill_requests": 0,
            "sum_prefill_tokens": row["queued_prefill_tokens"],
            "num_decode_requests": 0,
            "sum_decode_kv_tokens": row["queued_decode_kv_tokens"],
        },
    }


def run_predictions(rows: list[dict[str, int | float]], aiconfigurator_src: Path) -> list[dict[str, float]]:
    sys.path.insert(0, str(aiconfigurator_src))
    from aiconfigurator.sdk.rust_engine_step import RustForwardPassPerfModel

    model = RustForwardPassPerfModel.from_native(DEFAULT_CONFIG, {})
    print("AIC diagnostics:", model.diagnostics())

    compute_values = [float(row["prefill_tokens"]) for row in rows]
    read_values = [float(row["prefill_kv_tokens"]) for row in rows]
    compute_lo, compute_hi = min(compute_values), max(compute_values)
    read_lo, read_hi = min(read_values), max(read_values)
    buckets_per_axis = 8

    predictions: list[dict[str, float]] = []
    for row in rows:
        aic_ms = model.estimate_forward_pass_time_ms([make_fpm(row)])
        observed_ms = float(row["wall_time_s"]) * 1000.0
        diff_pct = (float(aic_ms) - observed_ms) / observed_ms * 100.0
        requests = float(row["prefill_requests"])
        predictions.append(
            {
                **row,
                "observed_ms": observed_ms,
                "aic_pred_ms": float(aic_ms),
                "diff_pct": diff_pct,
                "compute_bucket_x": bucket_index(float(row["prefill_tokens"]), compute_lo, compute_hi, buckets_per_axis),
                "read_bucket_y": bucket_index(float(row["prefill_kv_tokens"]), read_lo, read_hi, buckets_per_axis),
                "avg_compute_tokens_per_req": float(row["prefill_tokens"]) / requests if requests else 0.0,
                "avg_read_tokens_per_req": float(row["prefill_kv_tokens"]) / requests if requests else 0.0,
            }
        )
    return predictions


def write_predictions(predictions: list[dict[str, float]], path: Path) -> None:
    fields = [
        "row_id",
        "wall_time_s",
        "observed_ms",
        "aic_pred_ms",
        "diff_pct",
        "prefill_tokens",
        "prefill_kv_tokens",
        "prefill_requests",
        "avg_compute_tokens_per_req",
        "avg_read_tokens_per_req",
        "queued_prefill_tokens",
        "compute_bucket_x",
        "read_bucket_y",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in predictions:
            writer.writerow({field: row[field] for field in fields})


def write_summary(predictions: list[dict[str, float]], path: Path) -> None:
    compute_values = [float(row["prefill_tokens"]) for row in predictions]
    read_values = [float(row["prefill_kv_tokens"]) for row in predictions]
    compute_lo, compute_hi = min(compute_values), max(compute_values)
    read_lo, read_hi = min(read_values), max(read_values)
    buckets_per_axis = 8

    cells: dict[tuple[int, int], list[dict[str, float]]] = defaultdict(list)
    for row in predictions:
        cells[(int(row["read_bucket_y"]), int(row["compute_bucket_x"]))].append(row)

    fields = [
        "read_bucket_y",
        "compute_bucket_x",
        "read_lo",
        "read_hi",
        "compute_lo",
        "compute_hi",
        "count",
        "observed_ms_mean",
        "aic_ms_mean",
        "diff_pct_mean",
        "diff_pct_var",
        "diff_pct_std",
        "diff_pct_p50",
        "diff_pct_p90",
        "diff_pct_min",
        "diff_pct_max",
        "prefill_requests_mean",
        "avg_compute_tokens_per_req_mean",
        "avg_read_tokens_per_req_mean",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for iy in range(buckets_per_axis):
            read_start = read_lo + (read_hi - read_lo) * iy / buckets_per_axis
            read_end = read_lo + (read_hi - read_lo) * (iy + 1) / buckets_per_axis if iy < buckets_per_axis - 1 else read_hi
            for ix in range(buckets_per_axis):
                compute_start = compute_lo + (compute_hi - compute_lo) * ix / buckets_per_axis
                compute_end = compute_lo + (compute_hi - compute_lo) * (ix + 1) / buckets_per_axis if ix < buckets_per_axis - 1 else compute_hi
                rows = cells.get((iy, ix), [])
                diffs = [float(row["diff_pct"]) for row in rows]
                observed = [float(row["observed_ms"]) for row in rows]
                predicted = [float(row["aic_pred_ms"]) for row in rows]
                summary = {
                    "read_bucket_y": iy,
                    "compute_bucket_x": ix,
                    "read_lo": read_start,
                    "read_hi": read_end,
                    "compute_lo": compute_start,
                    "compute_hi": compute_end,
                    "count": len(rows),
                    "observed_ms_mean": mean(observed) if rows else "",
                    "aic_ms_mean": mean(predicted) if rows else "",
                    "diff_pct_mean": mean(diffs) if rows else "",
                    "diff_pct_var": var_pop(diffs) if rows else "",
                    "diff_pct_std": math.sqrt(var_pop(diffs)) if rows else "",
                    "diff_pct_p50": percentile(diffs, 0.5) if rows else "",
                    "diff_pct_p90": percentile(diffs, 0.9) if rows else "",
                    "diff_pct_min": min(diffs) if rows else "",
                    "diff_pct_max": max(diffs) if rows else "",
                    "prefill_requests_mean": mean([float(row["prefill_requests"]) for row in rows]) if rows else "",
                    "avg_compute_tokens_per_req_mean": mean([float(row["avg_compute_tokens_per_req"]) for row in rows]) if rows else "",
                    "avg_read_tokens_per_req_mean": mean([float(row["avg_read_tokens_per_req"]) for row in rows]) if rows else "",
                }
                writer.writerow(summary)


def load_summary(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]:
    buckets_per_axis = 8
    count = np.zeros((buckets_per_axis, buckets_per_axis), dtype=float)
    mean_diff = np.full((buckets_per_axis, buckets_per_axis), np.nan, dtype=float)
    var_diff = np.full((buckets_per_axis, buckets_per_axis), np.nan, dtype=float)
    compute_ranges: list[tuple[float, float]] = [(0.0, 0.0)] * buckets_per_axis
    read_ranges: list[tuple[float, float]] = [(0.0, 0.0)] * buckets_per_axis

    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            iy = int(row["read_bucket_y"])
            ix = int(row["compute_bucket_x"])
            row_count = int(row["count"])
            count[iy, ix] = row_count
            compute_ranges[ix] = (float(row["compute_lo"]), float(row["compute_hi"]))
            read_ranges[iy] = (float(row["read_lo"]), float(row["read_hi"]))
            if row_count:
                mean_diff[iy, ix] = float(row["diff_pct_mean"])
                var_diff[iy, ix] = float(row["diff_pct_var"])
    return count, mean_diff, var_diff, compute_ranges, read_ranges


def plot_heatmaps(summary_path: Path, outdir: Path) -> None:
    count, mean_diff, var_diff, compute_ranges, read_ranges = load_summary(summary_path)
    buckets_per_axis = 8
    xlabels = [f"x{i}\n{lo:.0f}-{hi:.0f}" for i, (lo, hi) in enumerate(compute_ranges)]
    ylabels = [f"y{i}\n{lo:.0f}-{hi:.0f}" for i, (lo, hi) in enumerate(read_ranges)]

    def cmap_with_bad(name: str):
        cmap = plt.get_cmap(name).copy()
        cmap.set_bad("white")
        return cmap

    def setup_axis(ax, title: str) -> None:
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_xlabel("compute tokens: prefill_tokens")
        ax.set_ylabel("read tokens: prefill_kv_tokens")
        ax.set_xticks(range(buckets_per_axis), xlabels, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(buckets_per_axis), ylabels, fontsize=8)
        ax.set_xlim(-0.5, buckets_per_axis - 0.5)
        ax.set_ylim(buckets_per_axis - 0.5, -0.5)
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, buckets_per_axis, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, buckets_per_axis, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)

    def text_color(cmap, norm, value: float) -> str:
        red, green, blue, _ = cmap(norm(float(value)))
        luminance = 0.299 * red + 0.587 * green + 0.114 * blue
        return "white" if luminance < 0.45 else "#1f2933"

    def valid_value(value) -> bool:
        return not np.ma.is_masked(value) and np.isfinite(float(value))

    def annotate(ax, data, fmt, cmap, norm) -> None:
        for iy in range(buckets_per_axis):
            for ix in range(buckets_per_axis):
                value = data[iy, ix]
                if not valid_value(value):
                    continue
                numeric = float(value)
                ax.text(
                    ix,
                    iy,
                    fmt(numeric),
                    ha="center",
                    va="center",
                    color=text_color(cmap, norm, numeric),
                    fontsize=8,
                    fontweight="medium",
                )

    def render_one(path: Path, data, *, title: str, cmap_name: str, norm, cbar_label: str, fmt) -> None:
        fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
        cmap = cmap_with_bad(cmap_name)
        image = ax.imshow(data, cmap=cmap, norm=norm)
        setup_axis(ax, title)
        annotate(ax, data, fmt, cmap, norm)
        cbar = fig.colorbar(image, ax=ax, shrink=0.82)
        cbar.set_label(cbar_label)
        fig.savefig(path, dpi=180)
        plt.close(fig)

    count_data = np.ma.masked_where(count <= 0, count)
    count_norm = LogNorm(vmin=1, vmax=max(1, count.max()))
    mean_norm = Normalize(vmin=-100, vmax=100)
    var_norm = PowerNorm(gamma=0.35, vmin=0, vmax=np.nanmax(var_diff))

    outputs = {
        "count": outdir / "trtllm_b200_gpt_oss_120b_heatmap_count.png",
        "mean": outdir / "trtllm_b200_gpt_oss_120b_heatmap_diff_pct_mean.png",
        "var": outdir / "trtllm_b200_gpt_oss_120b_heatmap_diff_pct_var.png",
        "combined": outdir / "trtllm_b200_gpt_oss_120b_heatmaps_combined.png",
    }
    render_one(
        outputs["count"],
        count_data,
        title="FPM count per 2D bucket (log color scale)",
        cmap_name="Blues",
        norm=count_norm,
        cbar_label="count",
        fmt=lambda value: f"{int(value)}",
    )
    render_one(
        outputs["mean"],
        mean_diff,
        title="AIC native validation: mean diff%",
        cmap_name="RdBu_r",
        norm=mean_norm,
        cbar_label="(AIC ms - observed ms) / observed ms * 100",
        fmt=lambda value: f"{value:.1f}%",
    )
    render_one(
        outputs["var"],
        var_diff,
        title="AIC native validation: variance of diff% (power color scale)",
        cmap_name="magma",
        norm=var_norm,
        cbar_label="population variance of diff%",
        fmt=lambda value: f"{value:.1f}",
    )

    fig, axes = plt.subplots(1, 3, figsize=(24, 7.5), constrained_layout=True)
    configs = [
        (count_data, "Blues", count_norm, "Count (log scale)", "count", lambda value: f"{int(value)}"),
        (mean_diff, "RdBu_r", mean_norm, "Mean diff%", "mean diff%", lambda value: f"{value:.1f}%"),
        (var_diff, "magma", var_norm, "Variance of diff%", "var(diff%)", lambda value: f"{value:.1f}"),
    ]
    for ax, (data, cmap_name, norm, title, label, fmt) in zip(axes, configs):
        cmap = cmap_with_bad(cmap_name)
        image = ax.imshow(data, cmap=cmap, norm=norm)
        setup_axis(ax, title)
        annotate(ax, data, fmt, cmap, norm)
        cbar = fig.colorbar(image, ax=ax, shrink=0.74)
        cbar.set_label(label)
    fig.suptitle(
        "GPT-OSS-120B / B200 SXM / TRT-LLM AIC Native Validation by 2D Prefill Bucket",
        fontsize=15,
    )
    fig.savefig(outputs["combined"], dpi=180)
    plt.close(fig)

    for name, path in outputs.items():
        print(f"{name}: {path.resolve()}")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.input)
    predictions = run_predictions(rows, args.aiconfigurator_src)

    predictions_path = args.outdir / "trtllm_b200_gpt_oss_120b_aic_native_predictions.csv"
    summary_path = args.outdir / "trtllm_b200_gpt_oss_120b_aic_native_2d_cell_diff_summary.csv"
    write_predictions(predictions, predictions_path)
    write_summary(predictions, summary_path)
    plot_heatmaps(summary_path, args.outdir)

    diffs = [float(row["diff_pct"]) for row in predictions]
    observed = [float(row["observed_ms"]) for row in predictions]
    predicted = [float(row["aic_pred_ms"]) for row in predictions]
    print(
        "overall:",
        f"n={len(predictions)}",
        f"observed_ms_mean={mean(observed):.3f}",
        f"aic_ms_mean={mean(predicted):.3f}",
        f"diff_pct_mean={mean(diffs):.3f}",
        f"diff_pct_var={var_pop(diffs):.3f}",
        f"diff_pct_p50={percentile(diffs, 0.5):.3f}",
        f"diff_pct_p90={percentile(diffs, 0.9):.3f}",
    )


if __name__ == "__main__":
    main()

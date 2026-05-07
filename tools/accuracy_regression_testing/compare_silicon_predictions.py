#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import csv
import importlib
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

REQUIRED_COLUMNS = (
    "mode",
    "model_path",
    "backend",
    "system",
    "silicon_ttft_ms",
    "silicon_tpot_ms",
    "old_predicted_ttft_ms",
    "new_predicted_ttft_ms",
    "old_predicted_tpot_ms",
    "new_predicted_tpot_ms",
)
Metric = tuple[str, str, str, str]
GroupSpec = tuple[str, Callable[[dict[str, str]], str]]
METRICS: tuple[Metric, ...] = (
    ("TTFT", "silicon_ttft_ms", "old_predicted_ttft_ms", "new_predicted_ttft_ms"),
    ("TPOT", "silicon_tpot_ms", "old_predicted_tpot_ms", "new_predicted_tpot_ms"),
)
SMALL_DELTA_LINE_THRESHOLD = 0.1


def _mape(rows: list[dict[str, str]], actual: str, predicted: str) -> float:
    return sum(_ape(row, actual, predicted) for row in rows) / len(rows)


def _mape_improvement(rows: list[dict[str, str]], actual: str, old: str, new: str) -> float:
    return _mape(rows, actual, old) - _mape(rows, actual, new)


def _ape(row: dict[str, str], actual: str, predicted: str) -> float:
    actual_value = float(row[actual])
    return abs(float(row[predicted]) - actual_value) / actual_value * 100


def _group_by(
    rows: list[dict[str, str]], key_fn: Callable[[dict[str, str]], str]
) -> OrderedDict[str, list[dict[str, str]]]:
    groups = OrderedDict()
    for row in rows:
        groups.setdefault(key_fn(row), []).append(row)
    return groups


def _model_name(row: dict[str, str]) -> str:
    return row["model_path"].split("/", maxsplit=1)[1] if "/" in row["model_path"] else row["model_path"]


def _backend_mode(row: dict[str, str]) -> str:
    return f"{row['backend']}-{row['mode']}"


def _exact_config(row: dict[str, str]) -> str:
    return f"{_model_name(row)}|{row['system']}|{row['backend']}|{row['mode']}"


GROUP_SPECS: tuple[GroupSpec, ...] = (
    ("model", _model_name),
    ("hardware", lambda row: row["system"]),
    ("backend-mode", _backend_mode),
)
SUMMARY_GROUP_SPECS: tuple[GroupSpec, ...] = (
    ("mode", lambda row: row["mode"]),
    *GROUP_SPECS,
    ("model-hardware-backend-mode", _exact_config),
)


def _summary_row(partition: str, rows: list[dict[str, str]]) -> dict[str, str]:
    comparable_rows = _comparable_rows(rows)
    ttft_improvement = (
        f"{_mape_improvement(comparable_rows, 'silicon_ttft_ms', 'old_predicted_ttft_ms', 'new_predicted_ttft_ms'):.6f}"
        if comparable_rows
        else ""
    )
    tpot_improvement = (
        f"{_mape_improvement(comparable_rows, 'silicon_tpot_ms', 'old_predicted_tpot_ms', 'new_predicted_tpot_ms'):.6f}"
        if comparable_rows
        else ""
    )
    return {
        "partition": partition,
        "num_samples_added": str(_prediction_count_delta(rows)),
        "ttft_mape_improvement_%": ttft_improvement,
        "tpot_mape_improvement_%": tpot_improvement,
    }


def _ape_improvement(row: dict[str, str], actual: str, old: str, new: str) -> float:
    return _ape(row, actual, old) - _ape(row, actual, new)


def _sample_change_counts(rows: list[dict[str, str]], actual: str, old: str, new: str) -> tuple[int, int]:
    improved = increased = 0
    for row in rows:
        improvement = _ape_improvement(row, actual, old, new)
        improved += improvement > 0
        increased += improvement < 0
    return improved, increased


def _has_prediction(row: dict[str, str], prefix: str) -> bool:
    return bool(row.get(f"{prefix}_predicted_ttft_ms") and row.get(f"{prefix}_predicted_tpot_ms"))


def _prediction_count_delta(rows: list[dict[str, str]]) -> int:
    return sum(_has_prediction(row, "new") for row in rows) - sum(_has_prediction(row, "old") for row in rows)


def _prediction_counts(rows: list[dict[str, str]]) -> tuple[int, int]:
    return sum(_has_prediction(row, "old") for row in rows), sum(_has_prediction(row, "new") for row in rows)


def _comparable_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if _has_prediction(row, "old") and _has_prediction(row, "new")]


def _colors(values: list[float]) -> list[str]:
    return ["green" if value > 0 else "red" if value < 0 else "black" for value in values]


def _color_group(value: float) -> str:
    return "green" if value > 0 else "red" if value < 0 else "black"


def _set_category_labels(ax, labels: list[str]) -> None:
    rotate = len(labels) > 4 or any(len(label) > 12 for label in labels)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45 if rotate else 0, ha="right" if rotate else "center", fontsize=11)


def _plot_error_range_bars(
    ax,
    improvements: list[float],
    old_errors: list[float],
    new_errors: list[float],
    labels: list[str] | None = None,
    width: float = 1.0,
) -> None:
    x = list(range(len(improvements)))
    for idx, improvement, old_error, new_error, color in zip(
        x, improvements, old_errors, new_errors, _colors(improvements), strict=True
    ):
        lower = min(old_error, new_error)
        height = abs(old_error - new_error)
        if improvement == 0:
            ax.hlines(new_error, idx - width / 2, idx + width / 2, color=color, linewidth=2)
        else:
            ax.bar(idx, height, bottom=lower, color=color, alpha=0.75, width=width)
            if abs(improvement) < SMALL_DELTA_LINE_THRESHOLD:
                ax.hlines(new_error, idx - width / 2, idx + width / 2, color="black", linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.margins(y=0.15)
    ax.grid(axis="y", alpha=0.3)
    if labels is not None:
        _set_category_labels(ax, labels)


def _plot_ape_improvements(ax, rows: list[dict[str, str]], title: str, actual: str, old: str, new: str) -> None:
    points = [(_color_group(_ape_improvement(row, actual, old, new)), _ape(row, actual, new)) for row in rows]
    color_order = {"green": 0, "red": 1, "black": 2}
    points = sorted(points, key=lambda item: (color_order[item[0]], item[1]))
    x = list(range(len(points)))
    colors = [color for color, _ in points]
    new_errors = [new_error for _, new_error in points]
    ax.scatter(x, new_errors, c=colors, s=10, alpha=0.8)
    section_start = 0
    ymax = max(new_errors + [0.0])
    for color in ("green", "red", "black"):
        count = colors.count(color)
        if count:
            section_end = section_start + count
            ax.text(
                (section_start + section_end - 1) / 2,
                ymax * 1.03,
                f"n = {count}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
            if section_end < len(points):
                ax.axvline(section_end - 0.5, color="black", linestyle="--", linewidth=0.8)
            section_start = section_end
    ax.margins(y=0.15)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel("row")
    ax.set_ylabel("New APE (%)")


def _plot_mape_improvements(
    ax,
    title: str,
    actual: str,
    old: str,
    new: str,
    labels: list[str],
    groups: list[list[dict[str, str]]],
    sample_groups: list[list[dict[str, str]]],
) -> None:
    improvements = [_mape_improvement(group, actual, old, new) for group in groups]
    old_mapes = [_mape(group, actual, old) for group in groups]
    new_mapes = [_mape(group, actual, new) for group in groups]
    _plot_error_range_bars(ax, improvements, old_mapes, new_mapes, labels, width=0.65)
    ymax = max(old_mapes + new_mapes + [0.0])
    for idx, (old_mape, new_mape, group) in enumerate(zip(old_mapes, new_mapes, sample_groups, strict=True)):
        old_count, new_count = _prediction_counts(group)
        samples_added = new_count - old_count
        text_bg_color = "green" if samples_added > 0 else "red" if samples_added < 0 else "white"
        text_bg_alpha = 0.3 if samples_added else 0.0
        ax.text(
            idx,
            max(old_mape, new_mape) + ymax * 0.03,
            f"n: {old_count} → {new_count}" if old_count != new_count else f"n: {old_count}",
            ha="center",
            va="bottom",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": text_bg_color,
                "alpha": text_bg_alpha,
                "edgecolor": "none",
            },
        )
    ax.set_title(title)
    ax.set_ylabel("MAPE (%)")


def create_plot(rows: list[dict[str, str]], output: Path) -> None:
    plt = importlib.import_module("matplotlib.pyplot")
    patch = importlib.import_module("matplotlib.patches").Patch
    plt.rcParams.update(
        {
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, axes = plt.subplots(4, 2, figsize=(16, 18), constrained_layout=True)
    fig.suptitle("AIC Prediction Accuracy Change", fontsize=22, fontweight="bold")
    comparable_rows = _comparable_rows(rows)
    grouped = []
    for name, key_fn in GROUP_SPECS:
        comparable_groups = _group_by(comparable_rows, key_fn)
        full_groups = _group_by(rows, key_fn)
        grouped.append((name, comparable_groups, full_groups))
    grouped_ymax = (
        max(
            max(_mape(group_rows, actual, old), _mape(group_rows, actual, new))
            for _, actual, old, new in METRICS
            for _, grouped_rows, _ in grouped
            for group_rows in grouped_rows.values()
        )
        * 1.25
    )

    for col_idx, (metric, actual, old, new) in enumerate(METRICS):
        _plot_ape_improvements(axes[0][col_idx], comparable_rows, f"{metric}: New APE by row", actual, old, new)
        for row_idx, (name, grouped_rows, full_grouped_rows) in enumerate(grouped, start=1):
            labels = list(grouped_rows.keys())
            _plot_mape_improvements(
                axes[row_idx][col_idx],
                f"{metric}: Old/New MAPE by {name}",
                actual,
                old,
                new,
                labels,
                list(grouped_rows.values()),
                [full_grouped_rows[label] for label in labels],
            )
            axes[row_idx][col_idx].set_ylim(0, grouped_ymax)

    legend_handles = [
        patch(color="green", label="improvement"),
        patch(color="red", label="regression"),
        patch(color="black", label="no change"),
    ]
    for ax in axes.flat:
        ax.legend(handles=legend_handles)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def read_valid_rows(input_path: Path) -> list[dict[str, str]]:
    with input_path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing required columns: {', '.join(sorted(missing))}")
        return [
            row
            for row in reader
            if all(
                row[column] != ""
                for column in ("mode", "model_path", "backend", "system", "silicon_ttft_ms", "silicon_tpot_ms")
            )
            and float(row["silicon_ttft_ms"])
            and float(row["silicon_tpot_ms"])
        ]


def write_summary(rows: list[dict[str, str]], output_path: Path) -> None:
    summary_rows = [_summary_row("all", rows)]
    for _, key_fn in SUMMARY_GROUP_SPECS:
        summary_rows.extend(
            _summary_row(partition, group_rows)
            for partition, group_rows in _group_by(rows, key_fn).items()
            if _comparable_rows(group_rows)
        )
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "partition",
                "num_samples_added",
                "ttft_mape_improvement_%",
                "tpot_mape_improvement_%",
            ),
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("comparison_summary.csv"))
    parser.add_argument("--plot-output", type=Path, default=Path("comparison_plot.png"))
    args = parser.parse_args()

    rows = read_valid_rows(args.input)
    if not rows:
        raise SystemExit("No valid rows found.")

    write_summary(rows, args.output)
    comparable_rows = _comparable_rows(rows)
    if not comparable_rows:
        raise SystemExit("No comparable rows found.")
    create_plot(rows, args.plot_output)


if __name__ == "__main__":
    main()

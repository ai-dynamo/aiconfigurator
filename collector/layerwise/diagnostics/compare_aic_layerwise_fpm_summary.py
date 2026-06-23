#!/usr/bin/env python3
"""Print an AIC layerwise-vs-FPM summary table for golden vLLM runs."""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import shutil
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:  # pragma: no cover - supports both module and direct-file execution.
    from .compare_aic_layerwise_fpm import (
        _load_fpm_max_num_seqs,
        _merge_layerwise_csvs,
        _resolve_auto_max_num_batched_tokens,
        compare,
    )
except ImportError:  # pragma: no cover
    from compare_aic_layerwise_fpm import (
        _load_fpm_max_num_seqs,
        _merge_layerwise_csvs,
        _resolve_auto_max_num_batched_tokens,
        compare,
    )


@dataclass(frozen=True)
class Case:
    """One FPM golden comparison case."""

    name: str
    model: str
    tp: int
    moe_tp: int
    ep: int
    fpm: str
    needs_moe_overlay: bool = False
    group: str = "primary"


PRIMARY_CASES: tuple[Case, ...] = (
    Case(
        "qwen32_tp1",
        "Qwen/Qwen3-32B",
        1,
        1,
        1,
        "fpm_upfront_qwen32_full_once_20260613_194007/tp1_ep1_past4096/fpm_metrics_phase.csv",
    ),
    Case(
        "qwen32_tp2",
        "Qwen/Qwen3-32B",
        2,
        1,
        1,
        "fpm_upfront_qwen32_full_once_20260613_194007/tp2_ep1_past4096/fpm_metrics_phase.csv",
    ),
    Case(
        "qwen32_tp4",
        "Qwen/Qwen3-32B",
        4,
        1,
        1,
        "fpm_upfront_qwen32_full_once_20260613_194007/tp4_ep1_past4096/fpm_metrics_phase.csv",
    ),
    Case(
        "qwen32_tp8",
        "Qwen/Qwen3-32B",
        8,
        1,
        1,
        "fpm_upfront_qwen32_full_once_20260613_194007/tp8_ep1_past4096/fpm_metrics_phase.csv",
    ),
    Case(
        "qwen36_tp1_ep1",
        "Qwen/Qwen3.6-35B-A3B",
        1,
        1,
        1,
        "fpm_upfront_qwen36_moe_full_once_20260613_201336/tp1_ep1_past4096/fpm_metrics_phase.csv",
        needs_moe_overlay=True,
    ),
    Case(
        "qwen36_tp2_ep2",
        "Qwen/Qwen3.6-35B-A3B",
        2,
        1,
        2,
        "fpm_upfront_qwen36_moe_full_once_20260613_201336/tp2_ep2_past4096/fpm_metrics_phase.csv",
        needs_moe_overlay=True,
    ),
    Case(
        "qwen36_tp4_ep4",
        "Qwen/Qwen3.6-35B-A3B",
        4,
        1,
        4,
        "fpm_upfront_qwen36_moe_full_once_20260613_201336/tp4_ep4_past4096/fpm_metrics_phase.csv",
        needs_moe_overlay=True,
    ),
    Case(
        "qwen36_tp8_ep8",
        "Qwen/Qwen3.6-35B-A3B",
        8,
        1,
        8,
        "fpm_upfront_qwen36_moe_full_once_20260613_201336/tp8_ep8_past4096/fpm_metrics_phase.csv",
        needs_moe_overlay=True,
    ),
    Case(
        "dsv4_tp1_ep4",
        "deepseek-ai/DeepSeek-V4-Flash",
        1,
        1,
        4,
        "fpm_dsv4_flash_tp1_ep4_dp4_official_20260614_023637/fpm_metrics_phase.csv",
        needs_moe_overlay=True,
    ),
    Case(
        "dsv4_tp2_ep4",
        "deepseek-ai/DeepSeek-V4-Flash",
        2,
        1,
        4,
        "fpm_dsv4_flash_tp2_tp4_ep4_official_20260614_070344/tp2_ep4_past4096/fpm_metrics_phase.csv",
        needs_moe_overlay=True,
    ),
    Case(
        "dsv4_tp4_ep4",
        "deepseek-ai/DeepSeek-V4-Flash",
        4,
        1,
        4,
        "fpm_dsv4_flash_tp2_tp4_ep4_official_20260614_070344/tp4_ep4_past4096/fpm_metrics_phase.csv",
        needs_moe_overlay=True,
    ),
)

QWEN36_TP_ONLY_CASES: tuple[Case, ...] = (
    Case(
        "qwen36_tp2_ep1",
        "Qwen/Qwen3.6-35B-A3B",
        2,
        2,
        1,
        "fpm_upfront_qwen36_moe_full_once_20260613_201336/tp2_ep1_past4096/fpm_metrics_phase.csv",
        needs_moe_overlay=True,
        group="qwen36_tp_only",
    ),
    Case(
        "qwen36_tp4_ep1",
        "Qwen/Qwen3.6-35B-A3B",
        4,
        4,
        1,
        "fpm_upfront_qwen36_moe_full_once_20260613_201336/tp4_ep1_past4096/fpm_metrics_phase.csv",
        needs_moe_overlay=True,
        group="qwen36_tp_only",
    ),
)

QWEN36_GAP_FILL_CASES: tuple[Case, ...] = (
    Case(
        "qwen36_tp1_ep2",
        "Qwen/Qwen3.6-35B-A3B",
        1,
        1,
        2,
        "fpm_gap_qwen36_dp_ep_official_20260614_025420/tp1_dp_ep/tp1_ep2_past4096/fpm_metrics_phase.csv",
        needs_moe_overlay=True,
        group="qwen36_gap_fill",
    ),
    Case(
        "qwen36_tp1_ep4",
        "Qwen/Qwen3.6-35B-A3B",
        1,
        1,
        4,
        "fpm_gap_qwen36_dp_ep_official_20260614_025420/tp1_dp_ep/tp1_ep4_past4096/fpm_metrics_phase.csv",
        needs_moe_overlay=True,
        group="qwen36_gap_fill",
    ),
    Case(
        "qwen36_tp1_ep8",
        "Qwen/Qwen3.6-35B-A3B",
        1,
        1,
        8,
        "fpm_gap_qwen36_dp_ep_official_20260614_025420/tp1_dp_ep/tp1_ep8_past4096/fpm_metrics_phase.csv",
        needs_moe_overlay=True,
        group="qwen36_gap_fill",
    ),
    Case(
        "qwen36_tp2_ep4",
        "Qwen/Qwen3.6-35B-A3B",
        2,
        1,
        4,
        "fpm_gap_qwen36_dp_ep_official_20260614_025420/tp2_dp_ep/tp2_ep4_past4096/fpm_metrics_phase.csv",
        needs_moe_overlay=True,
        group="qwen36_gap_fill",
    ),
    Case(
        "qwen36_tp2_ep8",
        "Qwen/Qwen3.6-35B-A3B",
        2,
        1,
        8,
        "fpm_gap_qwen36_dp_ep_official_20260614_025420/tp2_dp_ep/tp2_ep8_past4096/fpm_metrics_phase.csv",
        needs_moe_overlay=True,
        group="qwen36_gap_fill",
    ),
    Case(
        "qwen36_tp4_ep8",
        "Qwen/Qwen3.6-35B-A3B",
        4,
        1,
        8,
        "fpm_gap_qwen36_dp_ep_official_20260614_025420/tp4_ep8/fpm_metrics_phase.csv",
        needs_moe_overlay=True,
        group="qwen36_gap_fill",
    ),
)


def _all_cases() -> tuple[Case, ...]:
    return PRIMARY_CASES + QWEN36_TP_ONLY_CASES + QWEN36_GAP_FILL_CASES


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layerwise", type=Path, help="Layerwise CSV to evaluate.")
    parser.add_argument(
        "--layerwise-overlay",
        type=Path,
        action="append",
        default=[],
        help=(
            "Append a diagnostic layerwise CSV before comparison. May be repeated. "
            "Useful for explicit physical TP/EP probe rows without modifying canonical data."
        ),
    )
    parser.add_argument(
        "--layerwise-overlay-clear-max-num-seqs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Clear max_num_seqs on overlay rows so default decode lookups can prefer them. "
            "Use only for diagnostic overlays that intentionally replace the default envelope."
        ),
    )
    parser.add_argument(
        "--allow-physical-layerwise",
        action="store_true",
        help="Set AIC_LAYERWISE_ALLOW_PHYSICAL_GPUS=1 for diagnostic multi-GPU layerwise overlays.",
    )
    parser.add_argument("--fpm-root", type=Path, default=Path("fpm_golden_runs"))
    parser.add_argument(
        "--moe-perf-file",
        type=Path,
        default=Path("runs/combined_moe_ops_overlay_with_eager_20260613/moe_perf.txt"),
        help="MoE op overlay used for MoE model AIC estimates.",
    )
    parser.add_argument("--systems-root", default="src/aiconfigurator/systems")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run one named case. May be repeated. Use --list-cases to see names.",
    )
    parser.add_argument(
        "--include-qwen36-tp-only",
        action="store_true",
        help=(
            "Compare Qwen3.6 TP-only MoE cases from the upfront FPM run. "
            "These are included by default unless --primary-only is set."
        ),
    )
    parser.add_argument(
        "--include-qwen36-gap-fill",
        action="store_true",
        help=(
            "Compare the later Qwen3.6 TP/EP gap-fill FPM cases. "
            "These are included by default unless --primary-only is set."
        ),
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Only compare the original primary case set unless extra include flags are passed.",
    )
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional directory for raw comparison CSVs.")
    parser.add_argument("--keep-output-dir", action="store_true", help="Keep temp raw comparison CSVs.")
    parser.add_argument("--aggregation", choices=("median", "mean", "trimmed_mean"), default="trimmed_mean")
    parser.add_argument("--decode-match", choices=("exact", "nearest", "pooled"), default="pooled")
    parser.add_argument("--decode-past-kv", type=int, default=4096)
    parser.add_argument("--decode-osl", type=int, default=2)
    parser.add_argument("--max-decode-kv-distance", type=float, default=4.0)
    parser.add_argument("--decode-pool-forward-window", type=float, default=6.0)
    parser.add_argument("--fpm-workload-segment", default="sweep")
    parser.add_argument(
        "--vllm-max-num-batched-tokens",
        default="auto",
        help=(
            "vLLM scheduler max_num_batched_tokens for runtime metadata. "
            "Defaults to FPM metadata/observed rows capped by the largest context shape in --layerwise."
        ),
    )
    parser.add_argument(
        "--vllm-max-num-seqs",
        default="none",
        help=(
            "vLLM scheduler max_num_seqs for runtime metadata. Defaults to none so summaries can use "
            "layerwise decode rows collected with a larger max_num_seqs than the FPM metadata. Pass auto "
            "for a strict FPM-metadata lookup."
        ),
    )
    parser.add_argument("--moe-workload-distribution", default="power_law")
    parser.add_argument(
        "--include-mixed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include mixed prefill+decode FPM scheduler rows.",
    )
    parser.add_argument("--include-per-ops", action="store_true")
    parser.add_argument(
        "--shape-breakdown",
        choices=("none", "aggregate", "case"),
        default="none",
        help=(
            "Append an error breakdown by phase, token bucket, and batch bucket. "
            "aggregate groups all successful cases; case keeps each case separate."
        ),
    )
    parser.add_argument(
        "--shape-breakdown-bins",
        choices=("power2", "exact"),
        default="power2",
        help="Token/batch bucket style for --shape-breakdown.",
    )
    parser.add_argument(
        "--shape-breakdown-min-rows",
        type=int,
        default=1,
        help="Minimum row count required to print a shape-breakdown bucket.",
    )
    parser.add_argument(
        "--mixed-mode",
        choices=("workload", "clean"),
        default="workload",
        help=(
            "Mixed-row comparison policy. workload compares real FPM mixed scheduler ticks; "
            "clean applies conservative chunk/pathology filters."
        ),
    )
    parser.add_argument(
        "--filter-pathological-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter continuation-context FPM rows that are below a plausible latency floor.",
    )
    parser.add_argument(
        "--filter-pathological-decode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter isolated high-latency FPM decode rows before bin matching.",
    )
    parser.add_argument(
        "--filter-pathological-mixed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override mixed pathology filtering. Defaults off in workload mode and on in clean mode.",
    )
    parser.add_argument(
        "--filter-mixed-below-decode-floor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Filter mixed rows with nonzero context whose FPM latency is below comparable decode-only rows. "
            "Pass --no-filter-mixed-below-decode-floor for the raw workload view."
        ),
    )
    parser.add_argument(
        "--filter-nonterminal-mixed-chunks",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override nonterminal mixed chunk filtering. Defaults off in workload mode and on in clean mode.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Do not suppress comparator stdout.")
    return parser


def _selected_cases(args: argparse.Namespace) -> list[Case]:
    if args.case:
        by_name = {case.name: case for case in _all_cases()}
        missing = [name for name in args.case if name not in by_name]
        if missing:
            choices = ", ".join(sorted(by_name))
            raise SystemExit(f"Unknown case(s): {', '.join(missing)}. Choices: {choices}")
        return [by_name[name] for name in args.case]

    if args.primary_only:
        cases = list(PRIMARY_CASES)
    else:
        cases = list(_all_cases())

    if args.primary_only and args.include_qwen36_tp_only:
        cases.extend(QWEN36_TP_ONLY_CASES)
    if args.primary_only and args.include_qwen36_gap_fill:
        cases.extend(QWEN36_GAP_FILL_CASES)
    return cases


def _resolve_auto_int(
    raw: str,
    *,
    layerwise_csv: Path,
    fpm_csv: Path,
    case: Case,
    workload_segment: str | None,
    kind: str,
) -> int | None:
    if raw == "auto":
        if kind == "max_num_batched_tokens":
            return _resolve_auto_max_num_batched_tokens(
                layerwise_csv=layerwise_csv,
                fpm_csv=fpm_csv,
                model_name=case.model,
                tp=case.tp,
                moe_tp=case.moe_tp,
                ep=case.ep,
                workload_segment=workload_segment,
            )
        if kind == "max_num_seqs":
            return _load_fpm_max_num_seqs(fpm_csv)
        raise ValueError(kind)
    if raw in ("", "none", "None"):
        return None
    return int(raw)


def _mape(rows: Iterable[dict[str, object]], *, phase: str | None = None) -> float | None:
    values: list[float] = []
    for row in rows:
        if phase is not None and row.get("phase") != phase:
            continue
        try:
            error_pct = float(row["error_pct"])  # type: ignore[index]
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(error_pct):
            values.append(abs(error_pct))
    if not values:
        return None
    return statistics.mean(values)


def _bias(rows: Iterable[dict[str, object]], *, phase: str | None = None) -> float | None:
    values: list[float] = []
    for row in rows:
        if phase is not None and row.get("phase") != phase:
            continue
        try:
            error_pct = float(row["error_pct"])  # type: ignore[index]
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(error_pct):
            values.append(error_pct)
    if not values:
        return None
    return statistics.mean(values)


def _phase_count(rows: Iterable[dict[str, object]], phase: str) -> int:
    return sum(1 for row in rows if row.get("phase") == phase)


def _format_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}%"


def _print_case_list() -> None:
    print(f"{'case':24} {'group':18} {'model':32} {'tp':>2} {'moe':>3} {'ep':>2}")
    for case in _all_cases():
        print(
            f"{case.name:24} {case.group:18} {case.model:32} "
            f"{case.tp:2d} {case.moe_tp:3d} {case.ep:2d}"
        )


def _print_summary(results: list[tuple[Case, list[dict[str, object]], str | None]]) -> None:
    print(f"{'case':24} {'rows':>5} {'all':>8} {'ctx':>8} {'gen':>8} {'mixed':>8} status")
    successful_rows: list[dict[str, object]] = []
    for case, rows, error in results:
        if error is not None:
            print(f"{case.name:24} {0:5d} {'-':>8} {'-':>8} {'-':>8} {'-':>8} FAILED: {error}")
            continue
        successful_rows.extend(rows)
        print(
            f"{case.name:24} {len(rows):5d} "
            f"{_format_pct(_mape(rows)):>8} "
            f"{_format_pct(_mape(rows, phase='ctx')):>8} "
            f"{_format_pct(_mape(rows, phase='gen')):>8} "
            f"{_format_pct(_mape(rows, phase='mixed')):>8} ok"
        )

    if not successful_rows:
        return

    print()
    print(f"{'aggregate':24} {'rows':>5} {'all':>8} {'ctx':>8} {'gen':>8} {'mixed':>8}")
    print(
        f"{'all_cases':24} {len(successful_rows):5d} "
        f"{_format_pct(_mape(successful_rows)):>8} "
        f"{_format_pct(_mape(successful_rows, phase='ctx')):>8} "
        f"{_format_pct(_mape(successful_rows, phase='gen')):>8} "
        f"{_format_pct(_mape(successful_rows, phase='mixed')):>8}"
    )


def _row_float(row: dict[str, object], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _parse_shape_number(shape: str, prefix: str) -> float | None:
    marker = prefix
    start = shape.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = start
    while end < len(shape) and (shape[end].isdigit() or shape[end] == "."):
        end += 1
    if end == start:
        return None
    try:
        return float(shape[start:end])
    except ValueError:
        return None


def _bucket(value: float | None, *, bins: str) -> tuple[str, float]:
    if value is None:
        return "-", -1.0
    if bins == "exact":
        if abs(value - round(value)) < 1e-6:
            label = str(int(round(value)))
        else:
            label = f"{value:.3f}".rstrip("0").rstrip(".")
        return label, value
    if value <= 0:
        return "0", 0.0
    upper = 2 ** math.ceil(math.log2(value))
    return f"<={int(upper)}", float(upper)


def _shape_axes(row: dict[str, object]) -> tuple[float | None, float | None]:
    phase = str(row.get("phase") or "")
    shape = str(row.get("shape") or "")
    if phase == "ctx":
        return _row_float(row, "ctx_tokens"), _row_float(row, "ctx_requests")
    if phase == "gen":
        tokens = (
            _row_float(row, "aic_decode_past_kv")
            or _row_float(row, "fpm_representative_decode_kv")
            or _parse_shape_number(shape, "past")
        )
        batch = _row_float(row, "decode_requests") or _parse_shape_number(shape, "bs")
        return tokens, batch
    if phase == "mixed":
        return _row_float(row, "ctx_tokens"), _row_float(row, "decode_requests")
    return None, None


def _print_shape_breakdown(
    results: list[tuple[Case, list[dict[str, object]], str | None]],
    *,
    scope: str,
    bins: str,
    min_rows: int,
) -> None:
    grouped: dict[tuple[str, str, str, str], tuple[float, float, list[dict[str, object]]]] = {}
    for case, rows, error in results:
        if error is not None:
            continue
        case_key = "all_cases" if scope == "aggregate" else case.name
        for row in rows:
            phase = str(row.get("phase") or "")
            tokens, batch = _shape_axes(row)
            token_label, token_sort = _bucket(tokens, bins=bins)
            batch_label, batch_sort = _bucket(batch, bins=bins)
            key = (case_key, phase, token_label, batch_label)
            if key not in grouped:
                grouped[key] = (token_sort, batch_sort, [])
            grouped[key][2].append(row)

    visible = [
        (case_key, phase, token_label, batch_label, token_sort, batch_sort, rows)
        for (case_key, phase, token_label, batch_label), (token_sort, batch_sort, rows) in grouped.items()
        if len(rows) >= max(1, min_rows)
    ]
    if not visible:
        return

    print()
    print(
        "shape_breakdown "
        f"(bins={bins}; token=ctx_tokens for ctx/mixed, decode past-KV for gen; "
        "batch=ctx_requests for ctx, decode_requests for gen/mixed)"
    )
    if scope == "case":
        print(f"{'case':24} {'phase':6} {'token_bin':>10} {'batch_bin':>10} {'rows':>5} {'mape':>8} {'bias':>8}")
    else:
        print(f"{'phase':6} {'token_bin':>10} {'batch_bin':>10} {'rows':>5} {'mape':>8} {'bias':>8}")

    phase_order = {"ctx": 0, "gen": 1, "mixed": 2}
    for case_key, phase, token_label, batch_label, token_sort, batch_sort, rows in sorted(
        visible,
        key=lambda item: (
            item[0],
            phase_order.get(item[1], 99),
            item[4],
            item[5],
            item[2],
            item[3],
        ),
    ):
        if scope == "case":
            print(
                f"{case_key:24} {phase:6} {token_label:>10} {batch_label:>10} "
                f"{len(rows):5d} {_format_pct(_mape(rows)):>8} {_format_pct(_bias(rows)):>8}"
            )
        else:
            print(
                f"{phase:6} {token_label:>10} {batch_label:>10} "
                f"{len(rows):5d} {_format_pct(_mape(rows)):>8} {_format_pct(_bias(rows)):>8}"
            )


def _compare_case(
    *,
    case: Case,
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict[str, object]]:
    fpm_csv = args.fpm_root / case.fpm
    if not fpm_csv.is_file():
        raise FileNotFoundError(fpm_csv)
    moe_perf_file = args.moe_perf_file if case.needs_moe_overlay else None
    if moe_perf_file is not None and not moe_perf_file.is_file():
        raise FileNotFoundError(moe_perf_file)

    max_num_batched_tokens = _resolve_auto_int(
        args.vllm_max_num_batched_tokens,
        layerwise_csv=args.layerwise,
        fpm_csv=fpm_csv,
        case=case,
        workload_segment=args.fpm_workload_segment,
        kind="max_num_batched_tokens",
    )
    max_num_seqs = _resolve_auto_int(
        args.vllm_max_num_seqs,
        layerwise_csv=args.layerwise,
        fpm_csv=fpm_csv,
        case=case,
        workload_segment=args.fpm_workload_segment,
        kind="max_num_seqs",
    )
    filter_pathological_mixed = args.filter_pathological_mixed
    if filter_pathological_mixed is None:
        filter_pathological_mixed = args.mixed_mode == "clean"
    filter_nonterminal_mixed_chunks = args.filter_nonterminal_mixed_chunks
    if filter_nonterminal_mixed_chunks is None:
        filter_nonterminal_mixed_chunks = args.mixed_mode == "clean"
    return compare(
        layerwise_csv=args.layerwise,
        fpm_csv=fpm_csv,
        model_name=case.model,
        tp=case.tp,
        moe_tp=case.moe_tp,
        ep=case.ep,
        moe_workload_distribution=args.moe_workload_distribution,
        output=output_dir / f"{case.name}.csv",
        filtered_output=output_dir / f"{case.name}_filtered_rows.csv",
        aggregation=args.aggregation,
        decode_past_kv=args.decode_past_kv,
        decode_osl=args.decode_osl,
        decode_match=args.decode_match,
        max_decode_kv_distance=args.max_decode_kv_distance,
        decode_pool_forward_window=args.decode_pool_forward_window,
        include_mixed=args.include_mixed,
        vllm_max_num_batched_tokens=max_num_batched_tokens,
        vllm_max_num_seqs=max_num_seqs,
        filter_pathological_context=args.filter_pathological_context,
        pathological_context_min_continuation_ctx_tokens=128,
        pathological_context_continuation_min_latency_ms=5.0,
        pathological_context_peer_min_count=3,
        pathological_context_high_latency_factor=3.0,
        filter_pathological_decode=args.filter_pathological_decode,
        pathological_decode_peer_kv_window=8.0,
        pathological_decode_peer_batch_window=2,
        pathological_decode_min_peer_count=1,
        pathological_decode_latency_factor=5.0,
        pathological_decode_min_latency_ms=20.0,
        filter_pathological_mixed=filter_pathological_mixed,
        filter_mixed_below_decode_floor=args.filter_mixed_below_decode_floor,
        filter_nonterminal_mixed_chunks=filter_nonterminal_mixed_chunks,
        fpm_workload_segment=args.fpm_workload_segment,
        pathological_mixed_tiny_ctx_tokens=320,
        pathological_mixed_min_ctx_tokens=128,
        pathological_mixed_peer_ctx_fraction=0.05,
        pathological_mixed_peer_ctx_min_window=512,
        pathological_mixed_min_peer_count=3,
        pathological_mixed_latency_fraction=0.60,
        pathological_mixed_high_latency_factor=1.2,
        pathological_mixed_decode_spike_window=5,
        include_per_ops=args.include_per_ops,
        systems_root=args.systems_root,
        moe_perf_file=moe_perf_file,
    )


def main() -> int:
    args = _build_parser().parse_args()
    if args.list_cases:
        _print_case_list()
        return 0

    if args.layerwise is None:
        raise SystemExit("--layerwise is required unless --list-cases is used")
    if not args.layerwise.is_file():
        raise SystemExit(f"Layerwise CSV does not exist: {args.layerwise}")
    for overlay_csv in args.layerwise_overlay:
        if not overlay_csv.is_file():
            raise SystemExit(f"Layerwise overlay CSV does not exist: {overlay_csv}")
    if args.allow_physical_layerwise:
        os.environ["AIC_LAYERWISE_ALLOW_PHYSICAL_GPUS"] = "1"

    selected_cases = _selected_cases(args)
    output_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="aic_fpm_summary_"))
    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_output = args.output_dir is None and not args.keep_output_dir
    if args.layerwise_overlay:
        args.layerwise = _merge_layerwise_csvs(
            args.layerwise,
            args.layerwise_overlay,
            output_dir / "layerwise_with_overlays.csv",
            clear_overlay_max_num_seqs=args.layerwise_overlay_clear_max_num_seqs,
        )

    results: list[tuple[Case, list[dict[str, object]], str | None]] = []
    try:
        for case in selected_cases:
            try:
                if args.verbose:
                    rows = _compare_case(case=case, args=args, output_dir=output_dir)
                else:
                    with open(Path(output_dir) / f"{case.name}.log", "w") as log_f:
                        with contextlib.redirect_stdout(log_f), contextlib.redirect_stderr(log_f):
                            rows = _compare_case(case=case, args=args, output_dir=output_dir)
                results.append((case, rows, None))
            except Exception as exc:  # noqa: BLE001 - print all case failures in one table.
                if args.fail_fast:
                    raise
                results.append((case, [], f"{type(exc).__name__}: {exc}"))
        _print_summary(results)
        if args.shape_breakdown != "none":
            _print_shape_breakdown(
                results,
                scope=args.shape_breakdown,
                bins=args.shape_breakdown_bins,
                min_rows=args.shape_breakdown_min_rows,
            )
        if not cleanup_output:
            print()
            print(f"raw_outputs: {output_dir}")
        return 1 if any(error for _, _, error in results) else 0
    finally:
        if cleanup_output:
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

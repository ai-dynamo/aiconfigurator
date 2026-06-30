#!/usr/bin/env python3
"""Compare layerwise rows against FPM phase rows with explicit decode KV bins.

This diagnostic tool avoids pooling decode rows across a KV window by default.
A layerwise decode row with ``past_kv=4096`` represents one decode step, so the
matching FPM row should have the same ``mean_decode_kv_tokens`` when available.
If an FPM export uses a different convention, ``--decode-kv-offset`` can shift
the matched FPM target explicitly. If the FPM workload only produced a nearby
mean KV for a multi-request batch, the output labels that row as a nearest match
and reports the distance. An explicit pooled mode is available for comparing
against a short steady decode window after prefill.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _float_or_none(value: Any) -> float | None:
    """Parse a float, returning None for missing values."""

    if value in (None, ""):
        return None
    return float(value)


def _int_or_none(value: Any) -> int | None:
    """Parse an integer through float syntax used by some CSV exports."""

    parsed = _float_or_none(value)
    if parsed is None:
        return None
    return int(parsed)


def _aggregate(values: list[float], mode: str) -> float:
    """Aggregate repeated latency samples."""

    if not values:
        raise ValueError("cannot aggregate an empty sample set")
    if mode == "median":
        return float(statistics.median(values))
    if mode == "mean":
        return float(statistics.fmean(values))
    if mode == "trimmed_mean":
        if len(values) < 3:
            return float(statistics.fmean(values))
        return float(statistics.fmean(sorted(values)[1:-1]))
    raise ValueError(f"unsupported aggregation: {mode}")


def _entry_scale(row: dict[str, str]) -> float:
    """Return the representative-layer scale encoded in a layerwise row."""

    multiplier = _float_or_none(row.get("layer_multiplier")) or 0.0
    if multiplier <= 0.0:
        return 1.0
    measured = max(_float_or_none(row.get("measured_layer_count")) or 1.0, 1.0)
    return multiplier / measured


def load_layerwise_rows(
    path: Path,
    *,
    model_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Load layerwise rows and combine representative duplicates per shape."""

    combined: dict[tuple[str, int, int, int, str, int, int, int], dict[str, Any]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            phase = str(row.get("phase", "")).lower()
            if phase not in {"ctx", "gen"}:
                continue
            model = str(row.get("model", ""))
            if model_filter and model != model_filter:
                continue
            batch_size = _int_or_none(row.get("batch_size"))
            new_tokens = _int_or_none(row.get("new_tokens") or row.get("seq_len_q"))
            past_kv = _int_or_none(row.get("past_kv") or row.get("seq_len_kv_cache")) or 0
            latency = _float_or_none(row.get("latency_ms"))
            if batch_size is None or new_tokens is None or latency is None:
                continue
            attn_tp = _int_or_none(row.get("attn_tp")) or 1
            moe_tp = _int_or_none(row.get("moe_tp")) or 1
            ep = _int_or_none(row.get("ep")) or 1
            key = (phase, batch_size, new_tokens, past_kv, model, attn_tp, moe_tp, ep)
            entry = combined.setdefault(
                key,
                {
                    "model": model,
                    "attn_tp": attn_tp,
                    "moe_tp": moe_tp,
                    "ep": ep,
                    "phase": phase,
                    "batch_size": batch_size,
                    "new_tokens": new_tokens,
                    "past_kv": past_kv,
                    "layerwise_ms": 0.0,
                    "layerwise_rows": 0,
                },
            )
            entry["layerwise_ms"] += latency * _entry_scale(row)
            entry["layerwise_rows"] += 1
    return [combined[key] for key in sorted(combined)]


def load_fpm_bins(path: Path) -> tuple[dict[tuple[int, int], list[float]], dict[tuple[int, float], list[float]]]:
    """Load FPM context and decode latency bins from a phase CSV."""

    context: dict[tuple[int, int], list[float]] = defaultdict(list)
    decode: dict[tuple[int, float], list[float]] = defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            phase = str(row.get("phase", "")).lower()
            latency = _float_or_none(row.get("latency_ms"))
            if latency is None:
                continue
            if phase == "context":
                ctx_tokens = _int_or_none(row.get("ctx_tokens"))
                ctx_requests = _int_or_none(row.get("ctx_requests")) or 1
                if ctx_tokens is not None:
                    context[(ctx_requests, ctx_tokens)].append(latency)
            elif phase == "decode":
                decode_requests = _int_or_none(row.get("decode_requests"))
                mean_kv = _float_or_none(row.get("mean_decode_kv_tokens"))
                if decode_requests is not None and mean_kv is not None:
                    decode[(decode_requests, mean_kv)].append(latency)
    return context, decode


def _match_decode_bin(
    decode_bins: dict[tuple[int, float], list[float]],
    *,
    batch_size: int,
    target_past_kv: int,
    decode_kv_offset: float,
    mode: str,
    max_distance: float,
    pool_forward_window: float,
) -> tuple[str, list[float], str] | None:
    """Return the selected FPM decode KV bin for a target layerwise row."""

    target_fpm_kv = float(target_past_kv) + decode_kv_offset
    exact_key = (batch_size, target_fpm_kv)
    if mode != "pooled" and exact_key in decode_bins:
        return f"{target_fpm_kv:.3f}", decode_bins[exact_key], "exact"
    if mode == "exact":
        return None
    if mode == "pooled":
        lower = target_fpm_kv
        upper = lower + pool_forward_window
        pooled: list[tuple[float, list[float]]] = [
            (kv, values) for (bs, kv), values in decode_bins.items() if bs == batch_size and lower <= kv <= upper
        ]
        if not pooled:
            return None
        pooled.sort(key=lambda item: item[0])
        values = [latency for _, samples in pooled for latency in samples]
        first_kv = pooled[0][0]
        last_kv = pooled[-1][0]
        if first_kv == last_kv:
            label = f"{first_kv:.3f}"
        else:
            label = f"{first_kv:.3f}..{last_kv:.3f}"
        return label, values, "pooled"

    candidates = [(abs(kv - target_fpm_kv), kv, values) for (bs, kv), values in decode_bins.items() if bs == batch_size]
    if not candidates:
        return None
    distance, kv, values = min(candidates, key=lambda item: (item[0], item[1]))
    if distance > max_distance:
        return None
    return f"{kv:.3f}", values, "nearest"


def compare_layerwise_to_fpm(
    layerwise_path: Path,
    fpm_path: Path,
    *,
    aggregation: str,
    decode_match: str,
    max_decode_kv_distance: float,
    decode_pool_forward_window: float,
    decode_kv_offset: float = 0.0,
    model_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Build comparison rows from one layerwise CSV and one FPM phase CSV."""

    context_bins, decode_bins = load_fpm_bins(fpm_path)
    rows = []
    for layerwise in load_layerwise_rows(layerwise_path, model_filter=model_filter):
        phase = layerwise["phase"]
        fpm_values: list[float] | None = None
        fpm_decode_kv = ""
        fpm_match = ""
        if phase == "ctx":
            fpm_values = context_bins.get((layerwise["batch_size"], layerwise["new_tokens"]))
            fpm_match = "exact" if fpm_values else ""
        else:
            matched = _match_decode_bin(
                decode_bins,
                batch_size=layerwise["batch_size"],
                target_past_kv=layerwise["past_kv"],
                decode_kv_offset=decode_kv_offset,
                mode=decode_match,
                max_distance=max_decode_kv_distance,
                pool_forward_window=decode_pool_forward_window,
            )
            if matched:
                fpm_decode_kv, fpm_values, fpm_match = matched
        if not fpm_values:
            continue
        fpm_ms = _aggregate(fpm_values, aggregation)
        error_pct = ((layerwise["layerwise_ms"] / fpm_ms) - 1.0) * 100.0
        rows.append(
            {
                **layerwise,
                "fpm_ms": fpm_ms,
                "fpm_samples": len(fpm_values),
                "fpm_decode_kv": fpm_decode_kv,
                "fpm_match": fpm_match,
                "error_pct": error_pct,
                "abs_error_pct": abs(error_pct),
            }
        )
    return rows


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layerwise", type=Path, required=True, help="Layerwise CSV path.")
    parser.add_argument("--fpm", type=Path, required=True, help="FPM phase CSV path.")
    parser.add_argument("--output", type=Path, required=True, help="Output comparison CSV path.")
    parser.add_argument("--model", default=None, help="Optional model filter for multi-model layerwise CSVs.")
    parser.add_argument(
        "--aggregation",
        choices=("median", "mean", "trimmed_mean"),
        default="trimmed_mean",
        help="FPM repeat aggregation.",
    )
    parser.add_argument(
        "--decode-match",
        choices=("exact", "nearest", "pooled"),
        default="nearest",
        help="Decode KV bin matching. Pooled uses a forward KV window after the layerwise past_kv.",
    )
    parser.add_argument(
        "--max-decode-kv-distance",
        type=float,
        default=4.0,
        help="Maximum mean-KV distance accepted for nearest decode matching.",
    )
    parser.add_argument(
        "--decode-pool-forward-window",
        type=float,
        default=6.0,
        help="Forward KV window used by --decode-match pooled.",
    )
    parser.add_argument(
        "--decode-kv-offset",
        type=float,
        default=0.0,
        help="Offset added to layerwise past_kv when matching FPM mean decode KV.",
    )
    return parser


def main() -> None:
    """Run the comparison CLI."""

    args = _build_parser().parse_args()
    rows = compare_layerwise_to_fpm(
        args.layerwise,
        args.fpm,
        aggregation=args.aggregation,
        decode_match=args.decode_match,
        max_decode_kv_distance=args.max_decode_kv_distance,
        decode_pool_forward_window=args.decode_pool_forward_window,
        decode_kv_offset=args.decode_kv_offset,
        model_filter=args.model,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "attn_tp",
        "moe_tp",
        "ep",
        "phase",
        "batch_size",
        "new_tokens",
        "past_kv",
        "layerwise_ms",
        "layerwise_rows",
        "fpm_ms",
        "fpm_samples",
        "fpm_decode_kv",
        "fpm_match",
        "error_pct",
        "abs_error_pct",
    ]
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare AIC vLLM layerwise predictions against FPM phase rows.

This diagnostic is intentionally narrow: it uses an explicit layerwise CSV as
the layerwise database, reuses the repo's real communication/MoE tables, and
calls ``VLLMBackend`` scheduler-step estimators directly for FPM-comparable
shapes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from aiconfigurator.sdk import common, interpolation
from aiconfigurator.sdk.backends import vllm_backend
from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.operations.layerwise import (
    _MAX_NUM_BATCHED_INDEX_KEY,
    _MAX_NUM_BATCHED_MODE_INDEX_KEY,
    _MAX_NUM_BATCHED_PARALLEL_INDEX_KEY,
    _MAX_NUM_BATCHED_PARALLEL_MODE_INDEX_KEY,
    _MODE_INDEX_KEY,
    _PARALLEL_INDEX_KEY,
    _PARALLEL_MODE_INDEX_KEY,
    _interpolate_metric_2d,
    _interpolated_layer_scale_metadata,
    _representative_components,
    _uniform_bool_metric,
    _uniform_float_metric,
    _uniform_str_metric,
    load_layerwise_data,
)
from aiconfigurator.sdk.perf_database import PerfDatabase, PerfDataNotAvailableError
from aiconfigurator.sdk.utils import get_model_config_from_model_path

MIXED_PER_OP_FIELDS = [
    "mixed_layerwise_context_combined",
    "mixed_layerwise_context_tp_allreduce",
    "mixed_layerwise_decode_delta",
    "mixed_layerwise_ep_high_decode_floor",
    "mixed_moe",
    "mixed_moe_tp_allreduce",
    "mixed_moe_ep_alltoall",
    "mixed_moe_router",
    "mixed_moe_shared_expert",
]

CONTEXT_PER_OP_FIELDS = [
    "context_layerwise",
    "context_tp_allreduce",
    "context_scheduler_overhead",
    "context_moe_tp_allreduce",
    "context_moe_ep_alltoall",
    "context_moe",
    "context_moe_router",
    "context_moe_shared_expert",
    "context_moe_scheduler_overhead",
    "context_moe_scheduler_residual",
]

GENERATION_PER_OP_FIELDS = [
    "generation_layerwise",
    "generation_tp_allreduce",
    "generation_moe_tp_allreduce",
    "generation_moe_ep_alltoall",
    "generation_tp_allreduce_rms",
    "generation_moe",
    "generation_moe_router",
    "generation_moe_shared_expert",
    "generation_moe_scheduler_overhead",
    "generation_moe_scheduler_residual",
]


def _replace_path(path: Path) -> None:
    """Remove a generated overlay path if it already exists."""

    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)


def _symlink_or_copy(source: Path, destination: Path) -> None:
    """Link ``destination`` to ``source``, falling back to copy on unsupported filesystems."""

    _replace_path(destination)
    source = source.resolve()
    try:
        destination.symlink_to(source, target_is_directory=source.is_dir())
    except OSError:
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def _read_perf_frame(path: Path):
    """Read a CSV/TXT/parquet perf file into a pandas DataFrame."""

    import pandas as pd

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _prepare_moe_overlay_systems_root(
    *,
    systems_root: str,
    moe_perf_file: Path,
    output: Path,
    system: str = "b300_sxm",
    backend: str = "vllm",
    version: str = "0.20.1",
) -> str:
    """Create a generated systems root with a run-local MoE table overlaid."""

    import pandas as pd

    base_root = Path(systems_root)
    if not moe_perf_file.is_file():
        raise FileNotFoundError(f"MoE perf file does not exist: {moe_perf_file}")

    overlay_root = output.parent / f"{output.stem}_systems_overlay"
    _replace_path(overlay_root)
    overlay_root.mkdir(parents=True, exist_ok=True)

    shutil.copy2(base_root / f"{system}.yaml", overlay_root / f"{system}.yaml")

    base_system_data = base_root / "data" / system
    overlay_system_data = overlay_root / "data" / system
    overlay_system_data.mkdir(parents=True, exist_ok=True)

    for child in base_system_data.iterdir():
        destination = overlay_system_data / child.name
        if child.name == backend:
            destination.mkdir(exist_ok=True)
            continue
        _symlink_or_copy(child, destination)

    base_backend_root = base_system_data / backend
    overlay_backend_root = overlay_system_data / backend
    overlay_backend_root.mkdir(parents=True, exist_ok=True)
    for child in base_backend_root.iterdir():
        destination = overlay_backend_root / child.name
        if child.name == version:
            destination.mkdir(exist_ok=True)
            continue
        _symlink_or_copy(child, destination)

    base_version_root = base_backend_root / version
    overlay_version_root = overlay_backend_root / version
    overlay_version_root.mkdir(parents=True, exist_ok=True)
    for child in base_version_root.iterdir():
        if child.name in {"moe_perf.parquet", "moe_perf.txt"}:
            continue
        _symlink_or_copy(child, overlay_version_root / child.name)

    base_moe = base_version_root / "moe_perf.parquet"
    frames = []
    if base_moe.is_file():
        frames.append(_read_perf_frame(base_moe))
    frames.append(_read_perf_frame(moe_perf_file))
    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged.to_csv(overlay_version_root / "moe_perf.txt", index=False)
    return str(overlay_root)


def _add_mixed_per_op_columns(
    row: dict[str, Any],
    per_ops: dict[str, float],
    per_ops_source: dict[str, str],
) -> None:
    """Add stable mixed-step per-op latency/source columns to a comparison row."""

    for op_name in MIXED_PER_OP_FIELDS:
        row[f"aic_op_{op_name}"] = float(per_ops.get(op_name, 0.0))
        row[f"aic_source_{op_name}"] = str(per_ops_source.get(op_name, ""))


def _add_context_per_op_columns(
    row: dict[str, Any],
    per_ops: dict[str, float],
    per_ops_source: dict[str, str],
) -> None:
    """Add stable context per-op latency/source columns to a comparison row."""

    for op_name in CONTEXT_PER_OP_FIELDS:
        row[f"aic_op_{op_name}"] = float(per_ops.get(op_name, 0.0))
        row[f"aic_source_{op_name}"] = str(per_ops_source.get(op_name, ""))


def _add_generation_per_op_columns(
    row: dict[str, Any],
    per_ops: dict[str, float],
    per_ops_source: dict[str, str],
) -> None:
    """Add stable generation per-op latency/source columns to a comparison row."""

    for op_name in GENERATION_PER_OP_FIELDS:
        row[f"aic_op_{op_name}"] = float(per_ops.get(op_name, 0.0))
        row[f"aic_source_{op_name}"] = str(per_ops_source.get(op_name, ""))


def _trimmed_mean(values: list[float]) -> float:
    """Return a trimmed mean, dropping one min and max when possible."""

    if len(values) < 3:
        return float(statistics.fmean(values))
    return float(statistics.fmean(sorted(values)[1:-1]))


def _aggregate(values: list[float], mode: str) -> float:
    """Aggregate latency samples."""

    if mode == "median":
        return float(statistics.median(values))
    if mode == "mean":
        return float(statistics.fmean(values))
    if mode == "trimmed_mean":
        return _trimmed_mean(values)
    raise ValueError(f"unsupported aggregation: {mode}")


def _decode_pathology_reasons(
    rows: list[dict[str, str]],
    *,
    peer_kv_window: float,
    peer_batch_window: int,
    min_peer_count: int,
    latency_factor: float,
    min_latency_ms: float,
) -> dict[int, str]:
    """Return FPM-only pathology reasons for isolated high-latency decode rows."""

    parsed: list[tuple[int, int, float, float]] = []
    for index, row in enumerate(rows):
        if str(row.get("phase", "")).lower() != "decode":
            continue
        decode_requests = int(float(row.get("decode_requests") or 0))
        mean_kv = float(row.get("mean_decode_kv_tokens") or 0.0)
        latency_ms = float(row.get("latency_ms") or 0.0)
        parsed.append((index, decode_requests, mean_kv, latency_ms))

    reasons: dict[int, str] = {}
    for index, decode_requests, mean_kv, latency_ms in parsed:
        if latency_ms < min_latency_ms:
            continue
        peers = [
            peer_latency
            for peer_index, peer_decode_requests, peer_kv, peer_latency in parsed
            if peer_index != index
            and peer_decode_requests == decode_requests
            and abs(peer_kv - mean_kv) <= peer_kv_window
            and peer_latency > 0.0
        ]
        if len(peers) < min_peer_count:
            peers = [
                peer_latency
                for peer_index, peer_decode_requests, peer_kv, peer_latency in parsed
                if peer_index != index
                and abs(peer_decode_requests - decode_requests) <= peer_batch_window
                and abs(peer_kv - mean_kv) <= peer_kv_window
                and peer_latency > 0.0
            ]
        if len(peers) < min_peer_count:
            continue
        peer_median = float(statistics.median(peers))
        if latency_ms > peer_median * latency_factor:
            reasons[index] = (
                "decode_latency_above_peer_envelope:"
                f"latency_ms={latency_ms:.3f},peer_median_ms={peer_median:.3f},"
                f"decode_requests={decode_requests},mean_kv={mean_kv:.3f},peer_count={len(peers)}"
            )

    segment_start: int | None = None
    for index, row in enumerate(rows + [{"phase": ""}]):
        phase = str(row.get("phase", "")).lower()
        if phase == "decode":
            if segment_start is None:
                segment_start = index
            continue
        if segment_start is None:
            continue

        prev_row = rows[segment_start - 1] if segment_start > 0 else {}
        prev_phase = str(prev_row.get("phase", "")).lower()
        prev_ctx_tokens = int(float(prev_row.get("ctx_tokens") or 0))
        if prev_phase == "mixed" and prev_ctx_tokens > 0:
            for decode_index in range(segment_start, index):
                decode_row = rows[decode_index]
                reasons.setdefault(
                    decode_index,
                    "decode_segment_after_prefill:"
                    f"prev_phase={prev_phase},prev_ctx_tokens={prev_ctx_tokens},"
                    f"decode_requests={decode_row.get('decode_requests', '')},"
                    f"mean_kv={float(decode_row.get('mean_decode_kv_tokens') or 0.0):.3f}",
                )
        segment_start = None
    return reasons


def _context_pathology_reasons(
    rows: list[dict[str, str]],
    *,
    min_continuation_ctx_tokens: int,
    continuation_min_latency_ms: float,
    peer_min_count: int = 3,
    high_latency_factor: float = 3.0,
    filter_nonterminal_chunks: bool = True,
    nonterminal_chunk_lookahead: int = 3,
) -> dict[int, str]:
    """Return pathology reasons for context rows that are not clean targets."""

    reasons: dict[int, str] = {}
    parsed: list[tuple[int, int, int, int, float]] = []
    for index, row in enumerate(rows):
        if str(row.get("phase", "")).lower() != "context":
            continue
        ctx_tokens = int(float(row.get("ctx_tokens") or 0))
        ctx_requests = int(float(row.get("ctx_requests") or 1))
        ctx_kv_tokens = int(float(row.get("ctx_kv_tokens") or 0))
        latency_ms = float(row.get("latency_ms") or 0.0)
        parsed.append((index, ctx_requests, ctx_tokens, ctx_kv_tokens, latency_ms))
        if index == 0 and ctx_tokens > 0 and ctx_kv_tokens == 0 and latency_ms > 0.0:
            following_latencies: list[float] = []
            for next_index in range(1, min(len(rows), 6)):
                next_row = rows[next_index]
                next_phase = str(next_row.get("phase", "")).lower()
                if next_phase not in {"context", "mixed"}:
                    continue
                next_ctx_tokens = int(float(next_row.get("ctx_tokens") or 0))
                next_latency_ms = float(next_row.get("latency_ms") or 0.0)
                if next_ctx_tokens <= 0 or next_latency_ms <= 0.0:
                    continue
                if next_ctx_tokens >= max(ctx_tokens * 2, min_continuation_ctx_tokens):
                    following_latencies.append(next_latency_ms)
            if following_latencies:
                following_median = float(statistics.median(following_latencies))
                if following_median > 0.0 and latency_ms > following_median * high_latency_factor:
                    reasons[index] = (
                        "context_segment_start_above_following_envelope:"
                        f"latency_ms={latency_ms:.3f},following_median_ms={following_median:.3f},"
                        f"ctx_tokens={ctx_tokens},ctx_requests={ctx_requests},"
                        f"ctx_kv_tokens={ctx_kv_tokens},peer_count={len(following_latencies)}"
                    )
                    continue
        if 0 < ctx_tokens < min_continuation_ctx_tokens and ctx_kv_tokens > 0:
            reasons[index] = (
                "context_tiny_continuation_tail:"
                f"ctx_tokens={ctx_tokens},ctx_requests={ctx_requests},"
                f"ctx_kv_tokens={ctx_kv_tokens},latency_ms={latency_ms:.3f}"
            )
            continue
        if (
            ctx_kv_tokens > 0
            and ctx_tokens >= min_continuation_ctx_tokens
            and latency_ms < continuation_min_latency_ms
        ):
            reasons[index] = (
                "context_continuation_below_latency_floor:"
                f"latency_ms={latency_ms:.3f},ctx_tokens={ctx_tokens},"
                f"ctx_kv_tokens={ctx_kv_tokens},floor_ms={continuation_min_latency_ms:.3f}"
            )
            continue

        if not filter_nonterminal_chunks or ctx_tokens <= 0:
            continue
        chunk_kv_tokens = ctx_kv_tokens + ctx_tokens * max(ctx_requests, 1)
        for next_index in range(index + 1, min(len(rows), index + 1 + max(nonterminal_chunk_lookahead, 0))):
            next_row = rows[next_index]
            if str(next_row.get("phase", "")).lower() not in {"context", "mixed"}:
                continue
            next_ctx_kv_tokens = int(float(next_row.get("ctx_kv_tokens") or 0))
            if next_ctx_kv_tokens == chunk_kv_tokens:
                reasons[index] = (
                    "context_nonterminal_prefill_chunk:"
                    f"ctx_tokens={ctx_tokens},ctx_requests={ctx_requests},"
                    f"next_row_index={next_index},next_ctx_kv_tokens={next_ctx_kv_tokens}"
                )
                break
    for index, ctx_requests, ctx_tokens, ctx_kv_tokens, latency_ms in parsed:
        if index in reasons or latency_ms <= 0.0:
            continue
        all_same_shape_peers = [
            peer_latency
            for peer_index, peer_ctx_requests, peer_ctx_tokens, peer_ctx_kv_tokens, peer_latency in parsed
            if peer_index != index
            and peer_ctx_requests == ctx_requests
            and peer_ctx_tokens == ctx_tokens
            and peer_ctx_kv_tokens == ctx_kv_tokens
            and peer_latency > 0.0
        ]
        peers = [
            peer_latency
            for peer_index, peer_ctx_requests, peer_ctx_tokens, peer_ctx_kv_tokens, peer_latency in parsed
            if peer_index != index
            and peer_index not in reasons
            and peer_ctx_requests == ctx_requests
            and peer_ctx_tokens == ctx_tokens
            and peer_ctx_kv_tokens == ctx_kv_tokens
            and peer_latency > 0.0
        ]
        if (
            ctx_kv_tokens > 0
            and len(all_same_shape_peers) >= peer_min_count
            and len(peers) < peer_min_count
        ):
            peer_median = float(statistics.median(all_same_shape_peers))
            if peer_median < continuation_min_latency_ms and latency_ms >= continuation_min_latency_ms:
                reasons[index] = (
                    "context_continuation_isolated_after_low_floor:"
                    f"latency_ms={latency_ms:.3f},peer_median_ms={peer_median:.3f},"
                    f"ctx_tokens={ctx_tokens},ctx_requests={ctx_requests},"
                    f"ctx_kv_tokens={ctx_kv_tokens},peer_count={len(all_same_shape_peers)}"
                )
                continue
        if len(peers) < peer_min_count:
            continue
        peer_median = float(statistics.median(peers))
        if peer_median > 0.0 and latency_ms > peer_median * high_latency_factor:
            reasons[index] = (
                "context_latency_above_peer_envelope:"
                f"latency_ms={latency_ms:.3f},peer_median_ms={peer_median:.3f},"
                f"ctx_tokens={ctx_tokens},ctx_requests={ctx_requests},"
                f"ctx_kv_tokens={ctx_kv_tokens},peer_count={len(peers)}"
            )
    return reasons


def _context_workload_transition_reasons(
    rows: list[dict[str, str]],
    support_rows: list[dict[str, str]],
    *,
    high_latency_factor: float = 1.20,
) -> dict[int, str]:
    """Return singleton real-context rows that look like workload-transition overhead."""

    context_indices = [
        index
        for index, row in enumerate(rows)
        if str(row.get("phase", "")).lower() == "context"
    ]
    if len(context_indices) != 1:
        return {}

    index = context_indices[0]
    row = rows[index]
    ctx_tokens = int(float(row.get("ctx_tokens") or 0))
    ctx_requests = int(float(row.get("ctx_requests") or 1))
    ctx_kv_tokens = int(float(row.get("ctx_kv_tokens") or 0))
    latency_ms = float(row.get("latency_ms") or 0.0)
    if ctx_tokens <= 0 or ctx_requests != 1 or ctx_kv_tokens != 0 or latency_ms <= 0.0:
        return {}

    row_segment = str(row.get("workload_segment") or "")
    row_counter_id = str(row.get("counter_id") or "")
    min_peer_ctx_tokens = max(1, int(ctx_tokens * 0.5))
    max_peer_ctx_tokens = max(ctx_tokens + 64, int(ctx_tokens * 2.0))
    peer_latencies: list[float] = []
    for support_row in support_rows:
        if str(support_row.get("phase", "")).lower() != "context":
            continue
        if row_counter_id and row_counter_id == str(support_row.get("counter_id") or ""):
            continue
        if row_segment and row_segment == str(support_row.get("workload_segment") or ""):
            continue
        peer_ctx_requests = int(float(support_row.get("ctx_requests") or 1))
        peer_ctx_kv_tokens = int(float(support_row.get("ctx_kv_tokens") or 0))
        peer_ctx_tokens = int(float(support_row.get("ctx_tokens") or 0))
        peer_latency_ms = float(support_row.get("latency_ms") or 0.0)
        if peer_ctx_requests != ctx_requests or peer_ctx_kv_tokens != 0 or peer_latency_ms <= 0.0:
            continue
        if min_peer_ctx_tokens <= peer_ctx_tokens <= max_peer_ctx_tokens:
            peer_latencies.append(peer_latency_ms)

    if not peer_latencies:
        return {}
    peer_median = float(statistics.median(peer_latencies))
    if peer_median <= 0.0 or latency_ms <= peer_median * high_latency_factor:
        return {}
    return {
        index: (
            "context_singleton_workload_transition_above_support_envelope:"
            f"latency_ms={latency_ms:.3f},support_median_ms={peer_median:.3f},"
            f"ctx_tokens={ctx_tokens},ctx_requests={ctx_requests},"
            f"ctx_kv_tokens={ctx_kv_tokens},support_peer_count={len(peer_latencies)}"
        )
    }


def _load_fpm(
    path: Path,
    *,
    workload_segment: str | None = "sweep",
    filter_pathological_context: bool = False,
    pathological_context_min_continuation_ctx_tokens: int = 128,
    pathological_context_continuation_min_latency_ms: float = 5.0,
    pathological_context_peer_min_count: int = 3,
    pathological_context_high_latency_factor: float = 3.0,
    filter_pathological_decode: bool = False,
    pathological_decode_peer_kv_window: float = 8.0,
    pathological_decode_peer_batch_window: int = 2,
    pathological_decode_min_peer_count: int = 1,
    pathological_decode_latency_factor: float = 5.0,
    pathological_decode_min_latency_ms: float = 20.0,
) -> tuple[dict[tuple[int, int, int], list[float]], dict[tuple[int, float], list[float]], list[dict[str, Any]]]:
    """Load context/decode bins from an FPM phase CSV."""

    context: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    decode: dict[tuple[int, float], list[float]] = defaultdict(list)
    filtered_rows: list[dict[str, Any]] = []
    rows = _read_fpm_rows(path, workload_segment=workload_segment)
    decode_pathology_reasons = (
        _decode_pathology_reasons(
            rows,
            peer_kv_window=pathological_decode_peer_kv_window,
            peer_batch_window=pathological_decode_peer_batch_window,
            min_peer_count=pathological_decode_min_peer_count,
            latency_factor=pathological_decode_latency_factor,
            min_latency_ms=pathological_decode_min_latency_ms,
        )
        if filter_pathological_decode
        else {}
    )
    context_pathology_reasons = (
        _context_pathology_reasons(
            rows,
            min_continuation_ctx_tokens=pathological_context_min_continuation_ctx_tokens,
            continuation_min_latency_ms=pathological_context_continuation_min_latency_ms,
            peer_min_count=pathological_context_peer_min_count,
            high_latency_factor=pathological_context_high_latency_factor,
        )
        if filter_pathological_context
        else {}
    )
    if filter_pathological_context and _normalized_workload_segment(workload_segment) is not None:
        support_rows = _read_fpm_rows(path, workload_segment=None)
        context_pathology_reasons.update(
            _context_workload_transition_reasons(rows, support_rows)
        )
    context_rows = rows
    if rows and str(rows[0].get("phase", "")).lower() == "context":
        context_rows = []
        for row in rows:
            if str(row.get("phase", "")).lower() != "context":
                break
            context_rows.append(row)
    for index, row in enumerate(context_rows):
        if str(row.get("phase", "")).lower() != "context":
            continue
        pathology_reason = context_pathology_reasons.get(index)
        if pathology_reason:
            filtered_rows.append({
                "row_index": index,
                "phase": row.get("phase", ""),
                "counter_id": row.get("counter_id", ""),
                "reason": pathology_reason,
                "latency_ms": row.get("latency_ms", ""),
                "ctx_tokens": row.get("ctx_tokens", ""),
                "ctx_requests": row.get("ctx_requests", ""),
                "ctx_kv_tokens": row.get("ctx_kv_tokens", ""),
                "decode_requests": row.get("decode_requests", ""),
                "mean_decode_kv_tokens": row.get("mean_decode_kv_tokens", ""),
            })
            continue
        ctx_requests = int(row["ctx_requests"])
        ctx_tokens = int(row["ctx_tokens"])
        ctx_kv_tokens = int(float(row.get("ctx_kv_tokens") or 0))
        if ctx_requests <= 0 or ctx_tokens <= 0:
            continue
        ctx_prefix_tokens = round(ctx_kv_tokens / max(ctx_requests, 1))
        context[(ctx_requests, ctx_tokens, ctx_prefix_tokens)].append(float(row["latency_ms"]))

    for index, row in enumerate(rows):
        pathology_reason = decode_pathology_reasons.get(index)
        if pathology_reason:
            filtered_rows.append({
                "row_index": index,
                "phase": row.get("phase", ""),
                "counter_id": row.get("counter_id", ""),
                "reason": pathology_reason,
                "latency_ms": row.get("latency_ms", ""),
                "ctx_tokens": row.get("ctx_tokens", ""),
                "ctx_requests": row.get("ctx_requests", ""),
                "ctx_kv_tokens": row.get("ctx_kv_tokens", ""),
                "decode_requests": row.get("decode_requests", ""),
                "mean_decode_kv_tokens": row.get("mean_decode_kv_tokens", ""),
            })
            continue
        latency = float(row["latency_ms"])
        if row["phase"] == "decode":
            decode[(int(row["decode_requests"]), float(row["mean_decode_kv_tokens"]))].append(latency)
    return context, decode, filtered_rows


def _normalized_workload_segment(value: str | None) -> str | None:
    """Return a normalized workload-segment filter."""

    if value is None:
        return None
    normalized = value.strip()
    if not normalized or normalized.lower() in {"all", "none"}:
        return None
    return normalized


def _read_fpm_rows(path: Path, *, workload_segment: str | None = "sweep") -> list[dict[str, str]]:
    """Read FPM phase rows, filtering by workload segment when the CSV provides it."""

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    segment = _normalized_workload_segment(workload_segment)
    if segment is None or not rows:
        return rows
    if "workload_segment" not in rows[0]:
        return rows
    if not any(row.get("workload_segment") for row in rows):
        return rows
    return [row for row in rows if row.get("workload_segment") == segment]


def _write_filtered_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write filtered FPM rows and reasons for auditability."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "row_index",
        "phase",
        "counter_id",
        "reason",
        "latency_ms",
        "ctx_tokens",
        "ctx_requests",
        "ctx_kv_tokens",
        "decode_requests",
        "mean_decode_kv_tokens",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_fpm_max_num_batched_tokens(
    fpm_csv: Path,
    *,
    workload_segment: str | None = "sweep",
) -> int | None:
    """Return the effective FPM context-token budget for AIC chunking.

    vLLM's serialized ``scheduler_config.max_num_batched_tokens`` is normally
    the right value. Some architectures can still report phase rows whose
    scheduled context-token count is larger than that metadata value. In that
    case the FPM row is the more authoritative measurement boundary for this
    diagnostic, so use the observed scheduler budget as a lower bound.
    """

    candidates = [
        fpm_csv.parent / "vllm_metadata.json",
        fpm_csv.parent / "effective_vllm_config.json",
    ]
    keys = (
        "scheduler_config.max_num_batched_tokens",
        "max_num_batched_tokens",
    )
    metadata_value: int | None = None
    metadata_is_fpm_artifact = False
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        stack: list[Any] = [data]
        while stack:
            value = stack.pop()
            if not isinstance(value, dict):
                continue
            for key in keys:
                raw = value.get(key)
                if raw not in (None, ""):
                    metadata_value = int(raw)
                    break
            if metadata_value is not None:
                break
            stack.extend(value.values())
        if metadata_value is not None:
            metadata_is_fpm_artifact = path.name == "vllm_metadata.json" and data.get("artifact_kind") == "fpm"
            break

    observed_value = _infer_observed_fpm_context_budget(fpm_csv, workload_segment=workload_segment)
    if metadata_value is None:
        deployment_value = _load_dynamo_runtime_max_num_batched_tokens(fpm_csv.parent)
        if deployment_value is not None:
            return deployment_value
        return observed_value
    if metadata_is_fpm_artifact:
        return metadata_value
    deployment_value = _load_dynamo_runtime_max_num_batched_tokens(fpm_csv.parent)
    if deployment_value is not None:
        return deployment_value
    if observed_value is None:
        return metadata_value
    return max(metadata_value, observed_value)


def _load_dynamo_runtime_max_num_batched_tokens(run_dir: Path) -> int | None:
    """Return Dynamo model-card runtime context budget when present."""

    discovery = run_dir / "discovery"
    if not discovery.exists():
        return None
    values: set[int] = set()
    for path in discovery.rglob("*"):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        stack: list[Any] = [data]
        while stack:
            value = stack.pop()
            if not isinstance(value, dict):
                continue
            runtime_config = value.get("runtime_config")
            if isinstance(runtime_config, dict):
                raw = runtime_config.get("max_num_batched_tokens")
                if raw not in (None, ""):
                    values.add(int(raw))
            stack.extend(value.values())
    if not values:
        return None
    if len(values) > 1:
        return max(values)
    return values.pop()


def _infer_observed_fpm_context_budget(
    fpm_csv: Path,
    *,
    workload_segment: str | None = "sweep",
) -> int | None:
    """Infer a lower-bound single-request context budget from FPM phase rows."""

    if not fpm_csv.exists():
        return None
    observed = 0
    for row in _read_fpm_rows(fpm_csv, workload_segment=workload_segment):
        phase = str(row.get("phase", "")).lower()
        if phase not in {"context", "mixed"}:
            continue
        ctx_tokens = int(float(row.get("ctx_tokens") or 0))
        ctx_requests = int(float(row.get("ctx_requests") or 0))
        ctx_kv_tokens = int(float(row.get("ctx_kv_tokens") or 0))
        if ctx_tokens <= 0 or ctx_requests != 1:
            continue
        # Prefix-zero rows expose the initial scheduled chunk. Continuation
        # rows with a nonzero prefix still prove the scheduler accepted that
        # many new context tokens for one request in one FPM iteration.
        # Multi-request rows report aggregate scheduled context tokens and
        # are not a valid per-request chunk size for AIC context queries.
        observed = max(observed, ctx_tokens)
        if ctx_kv_tokens == 0:
            observed = max(observed, ctx_tokens)
    return observed or None


def _mixed_pathology_reasons(
    rows: list[dict[str, str]],
    *,
    peer_rows: list[dict[str, str]] | None = None,
    decode_rows: list[dict[str, str]] | None = None,
    tiny_ctx_tokens: int,
    min_ctx_tokens: int,
    peer_ctx_fraction: float,
    peer_ctx_min_window: int,
    min_peer_count: int,
    latency_fraction: float,
    high_latency_factor: float,
    decode_floor_peer_kv_window: float = 512.0,
    decode_floor_peer_batch_window: int = 1,
    decode_floor_min_peer_count: int = 3,
    decode_floor_latency_fraction: float = 0.95,
) -> dict[int, str]:
    """Return FPM-only pathology reasons for impossible mixed rows.

    The filter compares each large-context mixed row against neighboring mixed
    rows with similar scheduled context-token counts. This keeps raw FPM data
    intact while preventing a few scheduler-accounting artifacts from
    dominating AIC-vs-FPM charts.
    """

    def _parse_mixed_rows(source_rows: list[dict[str, str]]) -> list[tuple[int, float, int, float, float, int, float]]:
        parsed_rows: list[tuple[int, float, int, float, float, int, float]] = []
        for index, row in enumerate(source_rows):
            ctx_tokens = float(row.get("ctx_tokens") or 0.0)
            ctx_requests = int(float(row.get("ctx_requests") or 0.0))
            ctx_kv_tokens = float(row.get("ctx_kv_tokens") or 0.0)
            decode_requests = int(float(row.get("decode_requests") or 0.0))
            mean_decode_kv = float(row.get("mean_decode_kv_tokens") or 0.0)
            latency_ms = float(row.get("latency_ms") or 0.0)
            parsed_rows.append(
                (index, ctx_tokens, ctx_requests, latency_ms, ctx_kv_tokens, decode_requests, mean_decode_kv)
            )
        return parsed_rows

    parsed = _parse_mixed_rows(rows)
    peer_parsed = parsed if peer_rows is None else _parse_mixed_rows(peer_rows)
    parsed_decode: list[tuple[int, float, float]] = []
    for row in decode_rows or []:
        parsed_decode.append((
            int(float(row.get("decode_requests") or 0.0)),
            float(row.get("mean_decode_kv_tokens") or 0.0),
            float(row.get("latency_ms") or 0.0),
        ))

    reasons: dict[int, str] = {}
    shape_latencies: dict[tuple[float, int, float, int, float], list[tuple[int, float]]] = defaultdict(list)
    for index, ctx_tokens, ctx_requests, latency_ms, ctx_kv_tokens, decode_requests, mean_decode_kv in peer_parsed:
        shape_key = (ctx_tokens, ctx_requests, ctx_kv_tokens, decode_requests, round(mean_decode_kv, 3))
        shape_latencies[shape_key].append((index, latency_ms))

    for position, parsed_row in enumerate(parsed):
        index, ctx_tokens, ctx_requests, latency_ms, ctx_kv_tokens, decode_requests, mean_decode_kv = parsed_row
        queued_ctx_tokens = float(rows[index].get("queued_ctx_tokens") or 0.0)
        queued_ctx_requests = int(float(rows[index].get("queued_ctx_requests") or 0.0))
        if queued_ctx_requests > 0 and queued_ctx_tokens > 0.0 and ctx_tokens > 0.0:
            reasons[index] = (
                "mixed_nonterminal_queued_context:"
                f"ctx_tokens={ctx_tokens:.0f},ctx_requests={ctx_requests},"
                f"queued_ctx_tokens={queued_ctx_tokens:.0f},queued_ctx_requests={queued_ctx_requests},"
                f"latency_ms={latency_ms:.3f}"
            )
            continue
        if 0 < ctx_tokens < tiny_ctx_tokens and ctx_kv_tokens > 0.0 and decode_requests > 0:
            reasons[index] = (
                "mixed_tiny_continuation_tail:"
                f"ctx_tokens={ctx_tokens:.0f},ctx_kv_tokens={ctx_kv_tokens:.0f},"
                f"decode_requests={decode_requests},latency_ms={latency_ms:.3f}"
            )
            continue
        if decode_requests > 0 and parsed_decode:
            mean_decode_kv = float(rows[index].get("mean_decode_kv_tokens") or 0.0)
            decode_peers = [
                peer_latency
                for peer_requests, peer_kv, peer_latency in parsed_decode
                if abs(peer_requests - decode_requests) <= decode_floor_peer_batch_window
                and abs(peer_kv - mean_decode_kv) <= decode_floor_peer_kv_window
                and peer_latency > 0.0
            ]
            if len(decode_peers) >= decode_floor_min_peer_count:
                decode_median = float(statistics.median(decode_peers))
                if latency_ms < decode_median * decode_floor_latency_fraction:
                    reasons[index] = (
                        "mixed_latency_below_decode_floor:"
                        f"latency_ms={latency_ms:.3f},decode_median_ms={decode_median:.3f},"
                        f"decode_requests={decode_requests},mean_kv={mean_decode_kv:.3f},"
                        f"peer_count={len(decode_peers)}"
                    )
                    continue
        shape_key = (ctx_tokens, ctx_requests, ctx_kv_tokens, decode_requests, round(mean_decode_kv, 3))
        same_shape_peer_latencies = [
            peer_latency
            for peer_index, peer_latency in shape_latencies.get(shape_key, [])
            if peer_index != index and peer_latency > 0.0
        ]
        if same_shape_peer_latencies:
            same_shape_median = float(statistics.median(same_shape_peer_latencies))
            if latency_ms > same_shape_median * high_latency_factor:
                reasons[index] = (
                    "mixed_latency_above_same_shape_peer:"
                    f"latency_ms={latency_ms:.3f},peer_median_ms={same_shape_median:.3f},"
                    f"ctx_tokens={ctx_tokens:.0f},ctx_requests={ctx_requests},"
                    f"ctx_kv_tokens={ctx_kv_tokens:.0f},decode_requests={decode_requests},"
                    f"mean_kv={mean_decode_kv:.3f},peer_count={len(same_shape_peer_latencies)}"
                )
                continue
        if 0 < position < len(parsed) - 1 and decode_requests > 0:
            _, _, _, prev_latency, _, prev_decode_requests, prev_mean_decode_kv = parsed[position - 1]
            _, _, _, next_latency, _, next_decode_requests, next_mean_decode_kv = parsed[position + 1]
            continuous_decode_window = (
                abs(prev_decode_requests - decode_requests) <= 1
                and abs(next_decode_requests - decode_requests) <= 1
                and abs((prev_mean_decode_kv + 1.0) - mean_decode_kv) <= 0.25
                and abs((mean_decode_kv + 1.0) - next_mean_decode_kv) <= 0.25
            )
            adjacent_envelope_ms = max(prev_latency, next_latency)
            if (
                continuous_decode_window
                and ctx_tokens >= min_ctx_tokens
                and adjacent_envelope_ms > 0.0
                and latency_ms > adjacent_envelope_ms * high_latency_factor
            ):
                reasons[index] = (
                    "mixed_latency_above_adjacent_sequence_envelope:"
                    f"latency_ms={latency_ms:.3f},adjacent_envelope_ms={adjacent_envelope_ms:.3f},"
                    f"ctx_tokens={ctx_tokens:.0f},decode_requests={decode_requests},"
                    f"mean_kv={mean_decode_kv:.3f}"
                )
                continue
        below_peer_envelope_floor = ctx_tokens < min_ctx_tokens
        if below_peer_envelope_floor and not (
            ctx_tokens > 0
            and ctx_kv_tokens == 0
            and ctx_requests == 1
            and decode_requests > 0
        ):
            continue
        window = max(float(peer_ctx_min_window), abs(ctx_tokens) * peer_ctx_fraction)
        peers = [
            peer_latency
            for _, peer_ctx_tokens, _, peer_latency, _, _, _ in peer_parsed
            if abs(peer_ctx_tokens - ctx_tokens) <= window and peer_latency > 0.0
        ]
        if len(peers) >= min_peer_count:
            peer_median = float(statistics.median(peers))
            if not below_peer_envelope_floor and latency_ms < peer_median * latency_fraction:
                reasons[index] = (
                    "mixed_latency_below_peer_envelope:"
                    f"latency_ms={latency_ms:.3f},peer_median_ms={peer_median:.3f},"
                    f"ctx_tokens={ctx_tokens:.0f},peer_count={len(peers)}"
                )
            elif latency_ms > peer_median * high_latency_factor:
                reasons[index] = (
                    "mixed_latency_above_peer_envelope:"
                    f"latency_ms={latency_ms:.3f},peer_median_ms={peer_median:.3f},"
                    f"ctx_tokens={ctx_tokens:.0f},peer_count={len(peers)}"
                )
    return reasons


def _decode_spike_adjacent_mixed_reasons(
    rows: list[dict[str, str]],
    *,
    window: int,
    peer_kv_window: float,
    peer_batch_window: int,
    min_peer_count: int,
    latency_factor: float,
    min_latency_ms: float,
) -> dict[str, str]:
    """Return mixed counter ids adjacent to isolated high-latency decode spikes."""

    if window <= 0:
        return {}
    decode_reasons = _decode_pathology_reasons(
        rows,
        peer_kv_window=peer_kv_window,
        peer_batch_window=peer_batch_window,
        min_peer_count=min_peer_count,
        latency_factor=latency_factor,
        min_latency_ms=min_latency_ms,
    )
    spike_indices = [
        index
        for index, reason in decode_reasons.items()
        if reason.startswith("decode_latency_above_peer_envelope:")
    ]
    if not spike_indices:
        return {}

    reasons: dict[str, str] = {}
    for index, row in enumerate(rows):
        if str(row.get("phase", "")).lower() != "mixed":
            continue
        counter_id = str(row.get("counter_id", ""))
        if not counter_id:
            continue
        adjacent_spikes = [
            spike_index
            for spike_index in spike_indices
            if abs(index - spike_index) <= window
        ]
        if not adjacent_spikes:
            continue
        nearest_spike = min(adjacent_spikes, key=lambda spike_index: abs(index - spike_index))
        reasons[counter_id] = (
            "mixed_adjacent_to_decode_latency_spike:"
            f"mixed_row_index={index},decode_row_index={nearest_spike},"
            f"distance={abs(index - nearest_spike)},decode_reason={decode_reasons[nearest_spike]}"
        )
    return reasons


def _nonterminal_mixed_chunk_counter_ids(rows: list[dict[str, str]], *, lookahead: int = 3) -> set[str]:
    """Return mixed-row counter ids whose context continues in a following row."""

    counter_ids: set[str] = set()
    for index, row in enumerate(rows):
        if str(row.get("phase", "")).lower() != "mixed":
            continue
        ctx_tokens = int(float(row.get("ctx_tokens") or 0))
        ctx_requests = int(float(row.get("ctx_requests") or 0))
        ctx_kv_tokens = int(float(row.get("ctx_kv_tokens") or 0))
        if ctx_tokens <= 0 or ctx_requests <= 0:
            continue
        # FPM reports mixed ctx_tokens/ctx_kv_tokens as aggregate scheduled
        # tokens across the active partial-prefill requests. Advance by the
        # aggregate token count; multiplying by request count misses multi-
        # request chunk chains.
        next_prefix = ctx_kv_tokens + ctx_tokens
        for next_index in range(index + 1, min(len(rows), index + 1 + max(lookahead, 0))):
            next_row = rows[next_index]
            if str(next_row.get("phase", "")).lower() not in {"context", "mixed"}:
                continue
            next_ctx_kv_tokens = int(float(next_row.get("ctx_kv_tokens") or 0))
            if next_ctx_kv_tokens == next_prefix:
                counter_ids.add(str(row.get("counter_id", "")))
                break
    return counter_ids


def _mixed_chunk_sequences(
    rows: list[dict[str, str]],
    *,
    blocked_counter_ids: set[str] | None = None,
    lookahead: int = 3,
) -> list[list[str]]:
    """Return contiguous mixed prefill chunk chains by counter id.

    Nonterminal mixed prefill chunks are not standalone target shapes. When the
    continuation is also a mixed row, compare the sum of the scheduler steps
    instead of comparing only the terminal tail.
    """

    blocked_counter_ids = blocked_counter_ids or set()
    counter_to_index = {
        str(row.get("counter_id", "")): index
        for index, row in enumerate(rows)
        if str(row.get("phase", "")).lower() == "mixed" and str(row.get("counter_id", ""))
    }
    consumed: set[str] = set()
    sequences: list[list[str]] = []
    for index, row in enumerate(rows):
        if str(row.get("phase", "")).lower() != "mixed":
            continue
        counter_id = str(row.get("counter_id", ""))
        if not counter_id or counter_id in consumed or counter_id in blocked_counter_ids:
            continue
        chain = [counter_id]
        current_index = index
        current_row = row
        while True:
            ctx_tokens = int(float(current_row.get("ctx_tokens") or 0))
            ctx_requests = int(float(current_row.get("ctx_requests") or 0))
            ctx_kv_tokens = int(float(current_row.get("ctx_kv_tokens") or 0))
            if ctx_tokens <= 0 or ctx_requests <= 0:
                break
            next_prefix = ctx_kv_tokens + ctx_tokens
            next_counter_id = ""
            next_row: dict[str, str] | None = None
            for next_index in range(current_index + 1, min(len(rows), current_index + 1 + max(lookahead, 0))):
                candidate = rows[next_index]
                if str(candidate.get("phase", "")).lower() != "mixed":
                    continue
                candidate_counter_id = str(candidate.get("counter_id", ""))
                candidate_prefix = int(float(candidate.get("ctx_kv_tokens") or 0))
                if candidate_counter_id and candidate_prefix == next_prefix:
                    next_counter_id = candidate_counter_id
                    next_row = candidate
                    current_index = next_index
                    break
            if (
                not next_counter_id
                or next_row is None
                or next_counter_id in blocked_counter_ids
                or next_counter_id in consumed
                or next_counter_id not in counter_to_index
            ):
                break
            chain.append(next_counter_id)
            current_row = next_row
        if len(chain) > 1:
            sequences.append(chain)
            consumed.update(chain)
    return sequences


def _should_aggregate_mixed_chunk_sequence(
    backend: VLLMBackend,
    model: common.BaseModel,
    sequence_rows: list[dict[str, str]],
) -> bool:
    """Return whether a mixed chunk chain should be compared as one sequence."""

    if backend._layerwise_has_subquadratic_context_attention(model):
        return True

    # Dense chunk continuations are not standalone scheduler targets; compare
    # the complete mixed chain. Non-subquadratic MoE high-decode real-workload
    # chains can be better represented by individual scheduler envelopes. The
    # repeated Qwen sweep pathology is the low-decode prefill chain, where
    # separate chunk rows split one scheduler target into an under-predicted
    # first chunk and an over-predicted continuation.
    topk = int(getattr(model, "_topk", 0) or 0)
    if topk <= 0:
        return True
    max_decode_requests = max(
        int(float(row.get("decode_requests") or 0))
        for row in sequence_rows
    )
    return max_decode_requests <= topk


def _queue_adjacent_mixed_chunk_reasons(
    rows: list[dict[str, str]],
    *,
    lookbehind: int = 2,
    lookahead: int = 2,
) -> dict[str, str]:
    """Return mixed rows adjacent to queued-context drain/fill scheduler chunks."""

    reasons: dict[str, str] = {}
    for index, row in enumerate(rows):
        if str(row.get("phase", "")).lower() != "mixed":
            continue
        counter_id = str(row.get("counter_id", ""))
        if not counter_id:
            continue
        ctx_tokens = int(float(row.get("ctx_tokens") or 0))
        decode_requests = int(float(row.get("decode_requests") or 0))
        if ctx_tokens <= 0 or decode_requests <= 0:
            continue
        next_rows = rows[index + 1 : min(len(rows), index + 1 + max(lookahead, 0))]
        if any(
            float(next_row.get("queued_ctx_tokens") or 0.0) > 0.0
            and int(float(next_row.get("queued_ctx_requests") or 0.0)) > 0
            for next_row in next_rows
        ):
            reasons[counter_id] = "mixed_nonterminal_before_queued_context"
            continue
        previous_rows = rows[max(0, index - max(lookbehind, 0)) : index]
        if any(
            float(previous_row.get("queued_ctx_tokens") or 0.0) > 0.0
            and int(float(previous_row.get("queued_ctx_requests") or 0.0)) > 0
            for previous_row in previous_rows
        ):
            reasons[counter_id] = "mixed_nonterminal_queue_drain_tail"
    return reasons


def _mixed_shape_key(row: dict[str, str]) -> tuple[int, int, int, int, float]:
    """Return the FPM scheduler shape key for a mixed row."""

    ctx_tokens = int(float(row.get("ctx_tokens") or 0))
    ctx_requests = int(float(row.get("ctx_requests") or 0))
    ctx_kv_tokens = int(float(row.get("ctx_kv_tokens") or 0))
    decode_requests = int(float(row.get("decode_requests") or 0))
    mean_decode_kv = round(float(row.get("mean_decode_kv_tokens") or 0.0), 3)
    return ctx_tokens, ctx_requests, ctx_kv_tokens, decode_requests, mean_decode_kv


def _aggregate_mixed_rows(
    rows: list[dict[str, str]],
    *,
    aggregation: str,
) -> list[dict[str, Any]]:
    """Aggregate repeated mixed FPM measurements by scheduler shape."""

    grouped: dict[tuple[int, int, int, int, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[_mixed_shape_key(row)].append(row)

    aggregated: list[dict[str, Any]] = []
    for key, samples in sorted(grouped.items()):
        ctx_tokens, ctx_requests, ctx_kv_tokens, decode_requests, mean_decode_kv = key
        latencies = [float(row["latency_ms"]) for row in samples]
        representative = dict(samples[0])
        representative["ctx_tokens"] = str(ctx_tokens)
        representative["ctx_requests"] = str(ctx_requests)
        representative["ctx_kv_tokens"] = str(ctx_kv_tokens)
        representative["decode_requests"] = str(decode_requests)
        representative["mean_decode_kv_tokens"] = f"{mean_decode_kv:.3f}"
        representative["latency_ms"] = str(_aggregate(latencies, aggregation))
        representative["_fpm_samples"] = len(samples)
        representative["_counter_ids"] = ",".join(str(row.get("counter_id", "")) for row in samples)
        aggregated.append(representative)
    return aggregated


def _match_decode(
    decode: dict[tuple[int, float], list[float]],
    batch_size: int,
    past_kv: int,
    mode: str,
    max_distance: float,
    pool_forward_window: float,
) -> tuple[str, list[float], str, int] | None:
    """Return the requested decode bin/window for a batch and KV target."""

    exact = (batch_size, float(past_kv))
    if mode != "pooled" and exact in decode:
        return f"{float(past_kv):.3f}", decode[exact], "exact", past_kv
    if mode == "exact":
        return None
    if mode == "pooled":
        lower = float(past_kv)
        upper = lower + pool_forward_window
        pooled = [
            (kv, values)
            for (bs, kv), values in decode.items()
            if bs == batch_size and lower <= kv <= upper
        ]
        if not pooled:
            return None
        pooled.sort(key=lambda item: item[0])
        values = [latency for _, samples in pooled for latency in samples]
        kv_values = [kv for kv, samples in pooled for _ in samples]
        first_kv = pooled[0][0]
        last_kv = pooled[-1][0]
        label = f"{first_kv:.3f}" if first_kv == last_kv else f"{first_kv:.3f}..{last_kv:.3f}"
        representative_kv = round(statistics.median(kv_values))
        return label, values, "pooled", representative_kv
    candidates = [
        (abs(kv - float(past_kv)), kv, values)
        for (bs, kv), values in decode.items()
        if bs == batch_size
    ]
    if not candidates:
        return None
    distance, kv, values = min(candidates, key=lambda item: (item[0], item[1]))
    if distance > max_distance:
        return None
    return f"{kv:.3f}", values, "nearest", round(kv)


def _nearest_available_generation_kv(
    layerwise_data: dict[str, Any],
    *,
    model: str,
    tp_size: int,
    requested_kv: int,
    max_distance: float,
) -> int | None:
    """Return the nearest collected layerwise decode KV for an AIC query."""

    try:
        model_data = layerwise_data[model.lower()]["GEN"][tp_size]
    except KeyError:
        return None

    available: set[int] = set()
    for batch_data in model_data.values():
        if not isinstance(batch_data, dict):
            continue
        for seq_len in batch_data:
            try:
                available.add(round(float(seq_len)))
            except (TypeError, ValueError):
                continue
    if not available:
        return None
    nearest = min(available, key=lambda seq_len: (abs(seq_len - requested_kv), seq_len))
    if abs(nearest - requested_kv) > max_distance:
        return None
    return nearest


class _Config:
    """Minimal model config consumed by VLLMBackend phase estimators."""

    def __init__(
        self,
        *,
        tp_size: int,
        moe_tp_size: int,
        moe_ep_size: int,
        workload_distribution: str = "power_law",
        gemm_quant_mode: common.GEMMQuantMode = common.GEMMQuantMode.bfloat16,
        moe_quant_mode: common.MoEQuantMode | None = None,
        kvcache_quant_mode: common.KVCacheQuantMode = common.KVCacheQuantMode.bfloat16,
        fmha_quant_mode: common.FMHAQuantMode = common.FMHAQuantMode.bfloat16,
        nextn: int = 0,
        nextn_accept_rates: list[float] | None = None,
        extra_params: Any | None = None,
    ):
        self.tp_size = tp_size
        self.pp_size = 1
        self.moe_tp_size = moe_tp_size
        self.moe_ep_size = moe_ep_size
        moe_width = int(moe_tp_size) * int(moe_ep_size)
        self.attention_dp_size = max(1, moe_width // int(tp_size)) if moe_width % int(tp_size) == 0 else 1
        self.gemm_quant_mode = gemm_quant_mode
        self.moe_quant_mode = moe_quant_mode
        self.kvcache_quant_mode = kvcache_quant_mode
        self.fmha_quant_mode = fmha_quant_mode
        self.workload_distribution = workload_distribution
        self.moe_backend = None
        self.enable_eplb = False
        self.nextn = nextn
        self.nextn_accept_rates = nextn_accept_rates or []
        self.extra_params = extra_params


class _Model:
    """Minimal model object consumed by VLLMBackend phase estimators."""

    def __init__(
        self,
        *,
        model_path: str,
        tp_size: int,
        moe_tp_size: int,
        moe_ep_size: int,
        num_layers: int,
        hidden_size: int,
        workload_distribution: str = "power_law",
        topk: int = 0,
        num_experts: int = 0,
        moe_inter_size: int = 0,
        shared_expert_inter_size: int = 0,
        gemm_quant_mode: common.GEMMQuantMode = common.GEMMQuantMode.bfloat16,
        moe_quant_mode: common.MoEQuantMode | None = None,
        kvcache_quant_mode: common.KVCacheQuantMode = common.KVCacheQuantMode.bfloat16,
        fmha_quant_mode: common.FMHAQuantMode = common.FMHAQuantMode.bfloat16,
        nextn: int = 0,
        nextn_accept_rates: list[float] | None = None,
        extra_params: Any | None = None,
    ):
        self.model_path = model_path
        if moe_quant_mode is None and num_experts > 0:
            moe_quant_mode = common.MoEQuantMode.bfloat16
        self.config = _Config(
            tp_size=tp_size,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
            workload_distribution=workload_distribution,
            gemm_quant_mode=gemm_quant_mode,
            moe_quant_mode=moe_quant_mode,
            kvcache_quant_mode=kvcache_quant_mode,
            fmha_quant_mode=fmha_quant_mode,
            nextn=nextn,
            nextn_accept_rates=nextn_accept_rates,
            extra_params=extra_params,
        )
        self._num_layers = num_layers
        self._hidden_size = hidden_size
        self._nextn = nextn
        self._nextn_accept_rates = nextn_accept_rates or []
        self._topk = topk
        self._num_experts = num_experts
        self._moe_inter_size = moe_inter_size
        self._shared_expert_inter_size = shared_expert_inter_size
        self.extra_params = extra_params


class _LayerwiseDatabase:
    """Adapter that serves supplied layerwise rows plus real comm/MoE tables."""

    def __init__(self, layerwise_csv: Path, real_database: PerfDatabase):
        self.layerwise = load_layerwise_data(str(layerwise_csv))
        self.real_database = real_database
        self._extracted_metrics_cache: dict[Any, Any] = {}

    def query_layerwise_detail(
        self,
        model: str,
        phase: str,
        tp_size: int,
        batch_size: int,
        seq_len: int,
        seq_len_kv_cache: int = 0,
        moe_weight_mode: str | None = None,
        max_num_batched_tokens: int | None = None,
        moe_tp_size: int | None = None,
        moe_ep_size: int | None = None,
    ) -> dict[str, Any]:
        """Return an exact layerwise detail row."""

        phase_key = phase.upper()
        model_key = model.lower()
        parallel_requested = moe_tp_size is not None or moe_ep_size is not None
        parallel_fallback_ep_size: int | None = None
        if parallel_requested:
            query_moe_tp = int(moe_tp_size or 1)
            query_ep = int(moe_ep_size or 1)

            def _select_parallel_family(family: dict | None) -> tuple[dict | None, int | None]:
                if not family:
                    return None, None
                if query_ep in family and family[query_ep]:
                    return family[query_ep], None
                candidates = [
                    (int(candidate_ep), candidate_data)
                    for candidate_ep, candidate_data in family.items()
                    if int(candidate_ep) != query_ep and candidate_data
                ]
                if not candidates:
                    return None, None
                candidates.sort(
                    key=lambda item: (
                        abs(math.log2(max(float(item[0]), 1.0) / max(float(query_ep), 1.0))),
                        item[0],
                    )
                )
                return candidates[0][1], candidates[0][0]

            try:
                if phase_key == "CTX" and max_num_batched_tokens is not None and moe_weight_mode:
                    max_key = int(max_num_batched_tokens)
                    parallel_family = self.layerwise[_MAX_NUM_BATCHED_PARALLEL_MODE_INDEX_KEY][model_key][phase_key][
                        tp_size
                    ][str(moe_weight_mode)][max_key][query_moe_tp]
                elif phase_key == "CTX" and max_num_batched_tokens is not None:
                    max_key = int(max_num_batched_tokens)
                    parallel_family = self.layerwise[_MAX_NUM_BATCHED_PARALLEL_INDEX_KEY][model_key][phase_key][
                        tp_size
                    ][max_key][query_moe_tp]
                elif moe_weight_mode:
                    parallel_family = self.layerwise[_PARALLEL_MODE_INDEX_KEY][model_key][phase_key][tp_size][
                        str(moe_weight_mode)
                    ][query_moe_tp]
                else:
                    parallel_family = self.layerwise[_PARALLEL_INDEX_KEY][model_key][phase_key][tp_size][query_moe_tp]
                model_data, parallel_fallback_ep_size = _select_parallel_family(parallel_family)
            except KeyError:
                model_data = None
        else:
            model_data = None
        if model_data:
            pass
        elif phase_key == "CTX" and max_num_batched_tokens is not None:
            max_key = int(max_num_batched_tokens)
            if moe_weight_mode:
                mode_index = self.layerwise.get(_MAX_NUM_BATCHED_MODE_INDEX_KEY, {})
                try:
                    model_data = mode_index[model_key][phase_key][tp_size][str(moe_weight_mode)][max_key]
                except KeyError:
                    model_data = self.layerwise[_MAX_NUM_BATCHED_INDEX_KEY][model_key][phase_key][tp_size][max_key]
            else:
                model_data = self.layerwise[_MAX_NUM_BATCHED_INDEX_KEY][model_key][phase_key][tp_size][max_key]
        elif moe_weight_mode:
            mode_index = self.layerwise.get(_MODE_INDEX_KEY, {})
            try:
                model_data = mode_index[model_key][phase_key][tp_size][str(moe_weight_mode)]
            except KeyError:
                model_data = self.layerwise[model_key][phase_key][tp_size]
        else:
            model_data = self.layerwise[model_key][phase_key][tp_size]
        if phase_key == "CTX":
            if seq_len in model_data and seq_len_kv_cache in model_data[seq_len]:
                detail = self._normalize_detail(model_data[seq_len][seq_len_kv_cache])
                detail["query_seq_len_q"] = float(seq_len)
                detail["query_seq_len_kv_cache"] = float(seq_len_kv_cache)
                if parallel_fallback_ep_size is not None:
                    detail["parallel_fallback_moe_ep_size"] = float(parallel_fallback_ep_size)
                    detail["requested_moe_ep_size"] = float(moe_ep_size or 1)
                return detail
            if len(model_data) < 2:
                raise KeyError((model, phase, tp_size, batch_size, seq_len, seq_len_kv_cache))
            is_nonzero_prefix_context = seq_len_kv_cache > 0
            result = interpolation.interp_2d_linear(
                seq_len,
                seq_len_kv_cache,
                model_data,
                self._extracted_metrics_cache,
            )
            result["rms_latency"] = _interpolate_metric_2d(
                seq_len,
                seq_len_kv_cache,
                model_data,
                "rms_latency",
                self._extracted_metrics_cache,
            )
        elif batch_size in model_data and seq_len in model_data[batch_size]:
            detail = model_data[batch_size][seq_len]
            if phase_key == "GEN":
                detail = self._robust_generation_detail(model_data, batch_size, seq_len)
            detail = self._normalize_detail(detail)
            detail["query_seq_len_q"] = float(seq_len)
            detail["query_seq_len_kv_cache"] = float(seq_len_kv_cache)
            if parallel_fallback_ep_size is not None:
                detail["parallel_fallback_moe_ep_size"] = float(parallel_fallback_ep_size)
                detail["requested_moe_ep_size"] = float(moe_ep_size or 1)
            return detail
        else:
            is_nonzero_prefix_context = False
            result = interpolation.interp_2d_linear(
                batch_size,
                seq_len,
                model_data,
                self._extracted_metrics_cache,
            )
            result["rms_latency"] = _interpolate_metric_2d(
                batch_size,
                seq_len,
                model_data,
                "rms_latency",
                self._extracted_metrics_cache,
            )
        result["includes_moe"] = _uniform_bool_metric(model_data, "includes_moe")
        result["layer_type"] = _uniform_str_metric(model_data, "layer_type")
        result["layer_index"] = _uniform_float_metric(model_data, "layer_index")
        scale_metadata = _interpolated_layer_scale_metadata(model_data)
        if scale_metadata is not None:
            result["measured_layer_count"], result["layer_multiplier"] = scale_metadata
        elif not is_nonzero_prefix_context:
            result["measured_layer_count"] = _uniform_float_metric(model_data, "measured_layer_count", 1.0)
            result["layer_multiplier"] = _uniform_float_metric(model_data, "layer_multiplier")
        result["max_num_batched_tokens"] = _uniform_float_metric(model_data, "max_num_batched_tokens")
        result["physical_gpus"] = _uniform_float_metric(model_data, "physical_gpus")
        result["latency_source"] = _uniform_str_metric(model_data, "latency_source")
        result["components"] = _representative_components(model_data)
        result["query_seq_len_q"] = float(seq_len)
        result["query_seq_len_kv_cache"] = float(seq_len_kv_cache)
        if parallel_fallback_ep_size is not None:
            result["parallel_fallback_moe_ep_size"] = float(parallel_fallback_ep_size)
            result["requested_moe_ep_size"] = float(moe_ep_size or 1)
        return self._normalize_detail(result)

    def _robust_generation_detail(self, model_data: dict, batch_size: int, seq_len: int) -> Any:
        """Suppress isolated high scheduler-envelope decode outliers.

        Layerwise GEN scheduler rows are full-step timing samples. A single
        noisy batch-size point can otherwise dominate FPM comparisons even when
        adjacent batch sizes for the same KV shape are stable.
        """

        entry = model_data[batch_size][seq_len]
        if not isinstance(entry, dict):
            return entry
        latency_source = str(entry.get("latency_source") or "")
        if latency_source not in {"schedule_to_update", "worker_wall", "fpm_wall"}:
            return entry
        try:
            latency = float(entry["latency"])
        except (KeyError, TypeError, ValueError):
            return entry
        if latency <= 0.0:
            return entry

        points: list[tuple[int, float]] = []
        for candidate_batch, seq_data in model_data.items():
            if candidate_batch == batch_size or not isinstance(seq_data, dict) or seq_len not in seq_data:
                continue
            candidate = seq_data[seq_len]
            if not isinstance(candidate, dict):
                continue
            if str(candidate.get("latency_source") or "") != latency_source:
                continue
            if str(candidate.get("moe_weight_mode") or "") != str(entry.get("moe_weight_mode") or ""):
                continue
            try:
                candidate_latency = float(candidate["latency"])
            except (KeyError, TypeError, ValueError):
                continue
            if candidate_latency > 0.0:
                points.append((int(candidate_batch), candidate_latency))
        if len(points) < 2:
            return entry

        lower = sorted((point for point in points if point[0] < batch_size), reverse=True)
        higher = sorted(point for point in points if point[0] > batch_size)
        neighbors = (lower[:1] + higher[:2]) if not lower else (lower[:1] + higher[:1])
        if len(neighbors) < 2:
            return entry
        neighbor_median = float(statistics.median(latency for _, latency in neighbors))
        if neighbor_median <= 0.0:
            return entry
        if latency <= neighbor_median * 1.25 or latency - neighbor_median <= 0.5:
            return entry

        smoothed = dict(entry)
        smoothed["latency"] = neighbor_median
        smoothed["diagnostic_smoothed_from_latency"] = latency
        return smoothed

    def _normalize_detail(self, result: Any) -> dict[str, Any]:
        """Return layerwise detail fields matching ``PerfDatabase`` output."""

        if not isinstance(result, dict):
            result = {"latency": float(result), "energy": 0.0}
        out: dict[str, Any] = {
            "latency": float(result["latency"]),
            "energy": float(result.get("energy", 0.0)),
            "rms_latency": float(result.get("rms_latency", 0.0)),
            "rms_kernel_count": float(result.get("rms_kernel_count", 0.0)),
            "includes_moe": bool(result.get("includes_moe", False)),
        }
        if result.get("layer_type") not in (None, ""):
            out["layer_type"] = str(result["layer_type"])
        for metric in (
            "layer_index",
            "measured_layer_count",
            "layer_multiplier",
            "physical_gpus",
            "max_num_batched_tokens",
            "seq_len_q",
            "seq_len_kv_cache",
            "query_seq_len_q",
            "query_seq_len_kv_cache",
        ):
            if result.get(metric) not in (None, ""):
                out[metric] = float(result[metric])
        for metric in ("latency_source", "measurement_mode", "attribution_target", "vllm_config_hash"):
            if result.get(metric) not in (None, ""):
                out[metric] = str(result[metric])
        if result.get("moe_weight_mode") not in (None, ""):
            out["moe_weight_mode"] = str(result["moe_weight_mode"])
        if result.get("parallel_fallback_moe_ep_size") not in (None, ""):
            out["parallel_fallback_moe_ep_size"] = float(result["parallel_fallback_moe_ep_size"])
        if result.get("requested_moe_ep_size") not in (None, ""):
            out["requested_moe_ep_size"] = float(result["requested_moe_ep_size"])
        if isinstance(result.get("components"), list):
            out["components"] = [dict(component) for component in result["components"] if isinstance(component, dict)]
        return out

    def query_custom_allreduce(self, *args: Any, **kwargs: Any) -> Any:
        """Proxy TP allreduce queries to the real database."""

        return self.real_database.query_custom_allreduce(*args, **kwargs)

    def query_allreduce_rms(self, *args: Any, **kwargs: Any) -> Any:
        """Proxy fused allreduce+RMS queries to the real database."""

        return self.real_database.query_allreduce_rms(*args, **kwargs)

    def query_nccl(self, *args: Any, **kwargs: Any) -> Any:
        """Proxy NCCL collective queries to the real database."""

        return self.real_database.query_nccl(*args, **kwargs)

    def query_moe(self, *args: Any, **kwargs: Any) -> Any:
        """Proxy MoE op queries to the real database."""

        return self.real_database.query_moe(*args, **kwargs)

    def query_gemm(self, *args: Any, **kwargs: Any) -> Any:
        """Proxy GEMM queries to the real database."""

        return self.real_database.query_gemm(*args, **kwargs)

    def query_compute_scale(self, *args: Any, **kwargs: Any) -> Any:
        """Proxy GEMM scale queries to the real database."""

        return self.real_database.query_compute_scale(*args, **kwargs)

    def query_scale_matrix(self, *args: Any, **kwargs: Any) -> Any:
        """Proxy GEMM scale-matrix queries to the real database."""

        return self.real_database.query_scale_matrix(*args, **kwargs)

    def query_mem_op(self, *args: Any, **kwargs: Any) -> Any:
        """Proxy analytical memory-operation queries to the real database."""

        return self.real_database.query_mem_op(*args, **kwargs)


def _model_defaults(model: str, tp: int, moe_tp: int, ep: int, *, workload_distribution: str = "power_law") -> _Model:
    """Return known model metadata for current diagnostics."""

    if model == "Qwen/Qwen3-32B":
        return _Model(
            model_path=model,
            tp_size=tp,
            moe_tp_size=moe_tp,
            moe_ep_size=ep,
            num_layers=64,
            hidden_size=5120,
            workload_distribution=workload_distribution,
        )
    if model == "Qwen/Qwen3.6-35B-A3B":
        return _Model(
            model_path=model,
            tp_size=tp,
            moe_tp_size=moe_tp,
            moe_ep_size=ep,
            num_layers=40,
            hidden_size=2048,
            workload_distribution=workload_distribution,
            topk=8,
            num_experts=256,
            moe_inter_size=512,
            shared_expert_inter_size=0,
        )
    if model == "deepseek-ai/DeepSeek-V4-Flash":
        model_info = get_model_config_from_model_path(model)
        return _Model(
            model_path=model,
            tp_size=tp,
            moe_tp_size=moe_tp,
            moe_ep_size=ep,
            num_layers=int(model_info["layers"]),
            hidden_size=int(model_info["hidden_size"]),
            workload_distribution=workload_distribution,
            topk=int(model_info["topk"]),
            num_experts=int(model_info["num_experts"]),
            moe_inter_size=int(model_info["moe_inter_size"]),
            shared_expert_inter_size=2048,
            gemm_quant_mode=common.GEMMQuantMode.fp8_block,
            moe_quant_mode=common.MoEQuantMode.w4a8_mxfp4_mxfp8,
            kvcache_quant_mode=common.KVCacheQuantMode.fp8,
            fmha_quant_mode=common.FMHAQuantMode.fp8,
            nextn=0,
            nextn_accept_rates=[],
            extra_params=model_info["extra_params"],
        )
    raise ValueError(f"unknown model defaults for {model!r}")


def _effective_moe_parallelism(model: _Model, requested_moe_tp: int, requested_ep: int) -> tuple[int, int]:
    """Return the MoE parallelism that is meaningful for this model."""

    if model._num_experts == 0:
        return 1, 1
    return requested_moe_tp, requested_ep


def compare(
    *,
    layerwise_csv: Path,
    fpm_csv: Path,
    model_name: str,
    tp: int,
    moe_tp: int,
    ep: int,
    moe_workload_distribution: str,
    output: Path,
    filtered_output: Path | None,
    aggregation: str,
    decode_past_kv: int,
    decode_osl: int,
    decode_match: str,
    max_decode_kv_distance: float,
    decode_pool_forward_window: float,
    include_mixed: bool,
    vllm_max_num_batched_tokens: int | None,
    filter_pathological_context: bool,
    pathological_context_min_continuation_ctx_tokens: int,
    pathological_context_continuation_min_latency_ms: float,
    pathological_context_peer_min_count: int,
    pathological_context_high_latency_factor: float,
    filter_pathological_decode: bool,
    pathological_decode_peer_kv_window: float,
    pathological_decode_peer_batch_window: int,
    pathological_decode_min_peer_count: int,
    pathological_decode_latency_factor: float,
    pathological_decode_min_latency_ms: float,
    filter_pathological_mixed: bool,
    filter_nonterminal_mixed_chunks: bool,
    fpm_workload_segment: str | None,
    pathological_mixed_tiny_ctx_tokens: int,
    pathological_mixed_min_ctx_tokens: int,
    pathological_mixed_peer_ctx_fraction: float,
    pathological_mixed_peer_ctx_min_window: int,
    pathological_mixed_min_peer_count: int,
    pathological_mixed_latency_fraction: float,
    pathological_mixed_high_latency_factor: float,
    pathological_mixed_decode_spike_window: int,
    include_per_ops: bool = False,
    systems_root: str = "src/aiconfigurator/systems",
    moe_perf_file: Path | None = None,
) -> list[dict[str, Any]]:
    """Write AIC-vs-FPM comparison rows."""

    if moe_perf_file is not None:
        systems_root = _prepare_moe_overlay_systems_root(
            systems_root=systems_root,
            moe_perf_file=moe_perf_file,
            output=output,
        )
    real_db = PerfDatabase("b300_sxm", "vllm", "0.20.1", systems_root=systems_root)
    database = _LayerwiseDatabase(layerwise_csv, real_db)
    if model_name.lower() not in database.layerwise:
        raise PerfDataNotAvailableError(f"Model {model_name!r} not found in layerwise data {layerwise_csv}")
    model = _model_defaults(model_name, tp, moe_tp, ep, workload_distribution=moe_workload_distribution)
    effective_moe_tp, effective_ep = _effective_moe_parallelism(model, moe_tp, ep)
    if (effective_moe_tp, effective_ep) != (moe_tp, ep):
        print(
            "compare_aic_layerwise_fpm: "
            f"model {model_name!r} is dense; using moe_tp={effective_moe_tp}, ep={effective_ep} "
            f"instead of requested moe_tp={moe_tp}, ep={ep}.",
            file=sys.stderr,
        )
        moe_tp, ep = effective_moe_tp, effective_ep
        model = _model_defaults(model_name, tp, moe_tp, ep, workload_distribution=moe_workload_distribution)
    backend = VLLMBackend()
    runtime_config = RuntimeConfig(vllm_max_num_batched_tokens=vllm_max_num_batched_tokens)
    old_use_layerwise = vllm_backend._USE_LAYERWISE
    vllm_backend._USE_LAYERWISE = True
    context, decode, filtered_rows = _load_fpm(
        fpm_csv,
        workload_segment=fpm_workload_segment,
        filter_pathological_context=filter_pathological_context,
        pathological_context_min_continuation_ctx_tokens=pathological_context_min_continuation_ctx_tokens,
        pathological_context_continuation_min_latency_ms=pathological_context_continuation_min_latency_ms,
        pathological_context_peer_min_count=pathological_context_peer_min_count,
        pathological_context_high_latency_factor=pathological_context_high_latency_factor,
        filter_pathological_decode=filter_pathological_decode,
        pathological_decode_peer_kv_window=pathological_decode_peer_kv_window,
        pathological_decode_peer_batch_window=pathological_decode_peer_batch_window,
        pathological_decode_min_peer_count=pathological_decode_min_peer_count,
        pathological_decode_latency_factor=pathological_decode_latency_factor,
        pathological_decode_min_latency_ms=pathological_decode_min_latency_ms,
    )
    rows: list[dict[str, Any]] = []
    try:
        for (batch_size, ctx_tokens, ctx_prefix_tokens), samples in sorted(context.items()):
            if batch_size != 1:
                continue
            fpm_ms = _aggregate(samples, aggregation)
            try:
                latency, _, sources = backend._get_context_step_latency(
                    model,
                    database,
                    runtime_config,
                    ctx_tokens=ctx_tokens,
                    ctx_kv_tokens=ctx_prefix_tokens * batch_size,
                    ctx_requests=batch_size,
                )
            except (AssertionError, KeyError, PerfDataNotAvailableError, ValueError):
                continue
            aic_ms = float(sum(latency.values()))
            shape = f"ctx{ctx_tokens}"
            if ctx_prefix_tokens > 0:
                shape = f"{shape}_prefix{ctx_prefix_tokens}"
            output_row = {
                "model": model_name,
                "tp": tp,
                "moe_tp": moe_tp,
                "ep": ep,
                "fpm_workload_segment": _normalized_workload_segment(fpm_workload_segment) or "all",
                "phase": "ctx",
                "shape": shape,
                "fpm_ms": fpm_ms,
                "aic_ms": aic_ms,
                "error_pct": ((aic_ms / fpm_ms) - 1.0) * 100.0,
                "fpm_samples": len(samples),
                "fpm_match": "exact",
                "ctx_tokens": ctx_tokens,
                "ctx_requests": batch_size,
                "ctx_kv_tokens": ctx_prefix_tokens * batch_size,
                "ctx_prefix_tokens": ctx_prefix_tokens,
            }
            if include_per_ops:
                _add_context_per_op_columns(output_row, latency, sources)
            rows.append(output_row)
        for batch_size in sorted({bs for bs, _ in decode}):
            matched = _match_decode(
                decode,
                batch_size,
                decode_past_kv,
                decode_match,
                max_decode_kv_distance,
                decode_pool_forward_window,
            )
            if matched is None:
                continue
            kv_label, samples, match, fpm_representative_kv = matched
            fpm_ms = _aggregate(samples, aggregation)
            aic_past_kv = _nearest_available_generation_kv(
                database.layerwise,
                model=model_name,
                tp_size=tp,
                requested_kv=fpm_representative_kv,
                max_distance=max(
                    max_decode_kv_distance,
                    decode_pool_forward_window if decode_match == "pooled" else 0.0,
                ),
            )
            if aic_past_kv is None:
                continue
            try:
                latency, _, sources = backend._get_decode_step_latency(
                    model,
                    database,
                    runtime_config,
                    batch_size=batch_size,
                    past_kv=aic_past_kv,
                )
            except (AssertionError, KeyError, PerfDataNotAvailableError, ValueError):
                continue
            aic_ms = float(sum(latency.values()))
            output_row = {
                "model": model_name,
                "tp": tp,
                "moe_tp": moe_tp,
                "ep": ep,
                "fpm_workload_segment": _normalized_workload_segment(fpm_workload_segment) or "all",
                "phase": "gen",
                "shape": f"bs{batch_size}_past{aic_past_kv}",
                "fpm_ms": fpm_ms,
                "aic_ms": aic_ms,
                "error_pct": ((aic_ms / fpm_ms) - 1.0) * 100.0,
                "fpm_samples": len(samples),
                "fpm_match": f"{match}:{kv_label}",
                "fpm_representative_decode_kv": fpm_representative_kv,
                "aic_decode_past_kv": aic_past_kv,
            }
            if include_per_ops:
                _add_generation_per_op_columns(output_row, latency, sources)
            rows.append(output_row)
        if include_mixed:
            phase_rows = _read_fpm_rows(fpm_csv, workload_segment=fpm_workload_segment)
            mixed_rows = [fpm_row for fpm_row in phase_rows if fpm_row["phase"] == "mixed"]
            decode_rows = [fpm_row for fpm_row in phase_rows if fpm_row["phase"] == "decode"]
            support_mixed_rows: list[dict[str, str]] | None = None
            support_decode_rows = decode_rows
            if _normalized_workload_segment(fpm_workload_segment) is not None:
                support_phase_rows = _read_fpm_rows(fpm_csv, workload_segment=None)
                support_mixed_rows = [fpm_row for fpm_row in support_phase_rows if fpm_row["phase"] == "mixed"]
                support_decode_rows = [fpm_row for fpm_row in support_phase_rows if fpm_row["phase"] == "decode"]
            nonterminal_mixed_counter_ids = _nonterminal_mixed_chunk_counter_ids(phase_rows)
            queue_adjacent_mixed_reasons = _queue_adjacent_mixed_chunk_reasons(phase_rows)
            decode_spike_adjacent_mixed_reasons = (
                _decode_spike_adjacent_mixed_reasons(
                    phase_rows,
                    window=pathological_mixed_decode_spike_window,
                    peer_kv_window=pathological_decode_peer_kv_window,
                    peer_batch_window=pathological_decode_peer_batch_window,
                    min_peer_count=pathological_decode_min_peer_count,
                    latency_factor=pathological_decode_latency_factor,
                    min_latency_ms=pathological_decode_min_latency_ms,
                )
                if filter_pathological_mixed
                else {}
            )
            pathology_reasons = (
                _mixed_pathology_reasons(
                    mixed_rows,
                    peer_rows=support_mixed_rows,
                    decode_rows=support_decode_rows,
                    tiny_ctx_tokens=pathological_mixed_tiny_ctx_tokens,
                    min_ctx_tokens=pathological_mixed_min_ctx_tokens,
                    peer_ctx_fraction=pathological_mixed_peer_ctx_fraction,
                    peer_ctx_min_window=pathological_mixed_peer_ctx_min_window,
                    min_peer_count=pathological_mixed_min_peer_count,
                    latency_fraction=pathological_mixed_latency_fraction,
                    high_latency_factor=pathological_mixed_high_latency_factor,
                )
                if filter_pathological_mixed
                else {}
            )
            mixed_row_by_counter_id = {
                str(fpm_row.get("counter_id", "")): fpm_row
                for fpm_row in mixed_rows
                if str(fpm_row.get("counter_id", ""))
            }
            blocked_sequence_counter_ids = set(queue_adjacent_mixed_reasons)
            blocked_sequence_counter_ids.update(decode_spike_adjacent_mixed_reasons)
            for row_index in pathology_reasons:
                blocked_sequence_counter_ids.add(str(mixed_rows[row_index].get("counter_id", "")))
            sequence_rows_by_first_counter_id: dict[str, list[dict[str, str]]] = {}
            sequence_counter_ids: set[str] = set()
            aggregate_mixed_chunk_sequences = filter_nonterminal_mixed_chunks
            if aggregate_mixed_chunk_sequences:
                for sequence in _mixed_chunk_sequences(
                    phase_rows,
                    blocked_counter_ids=blocked_sequence_counter_ids,
                ):
                    sequence_rows = [mixed_row_by_counter_id.get(counter_id) for counter_id in sequence]
                    if any(sequence_row is None for sequence_row in sequence_rows):
                        continue
                    typed_sequence_rows = [
                        sequence_row for sequence_row in sequence_rows if sequence_row is not None
                    ]
                    if not _should_aggregate_mixed_chunk_sequence(backend, model, typed_sequence_rows):
                        continue
                    sequence_counter_ids.update(sequence)
                    sequence_rows_by_first_counter_id[sequence[0]] = typed_sequence_rows

            def _predict_mixed_fpm_row(
                fpm_row: dict[str, str],
            ) -> tuple[
                float,
                float,
                int,
                int,
                int,
                int,
                int,
                float,
                dict[str, float],
                dict[str, str],
            ]:
                fpm_ms = float(fpm_row["latency_ms"])
                ctx_tokens = int(fpm_row["ctx_tokens"])
                ctx_requests = int(fpm_row["ctx_requests"])
                ctx_kv_tokens = int(float(fpm_row.get("ctx_kv_tokens") or 0))
                ctx_prefix_tokens = round(ctx_kv_tokens / max(ctx_requests, 1))
                gen_tokens = int(fpm_row["decode_requests"])
                mean_decode_kv = float(fpm_row["mean_decode_kv_tokens"])
                aic_ms, _, per_ops, per_ops_source = backend._get_mix_step_latency(
                    model,
                    database,
                    runtime_config,
                    ctx_tokens=ctx_tokens,
                    gen_tokens=gen_tokens,
                    isl=round(mean_decode_kv),
                    osl=1,
                    prefix=ctx_prefix_tokens,
                    ctx_requests=ctx_requests,
                )
                return (
                    fpm_ms,
                    float(aic_ms),
                    ctx_tokens,
                    ctx_requests,
                    ctx_kv_tokens,
                    ctx_prefix_tokens,
                    gen_tokens,
                    mean_decode_kv,
                    per_ops,
                    per_ops_source,
                )

            comparable_mixed_rows: list[dict[str, str]] = []
            for row_index, fpm_row in enumerate(mixed_rows):
                mixed_counter_id = str(fpm_row.get("counter_id", ""))
                if mixed_counter_id in sequence_counter_ids:
                    continue
                is_nonterminal_mixed = mixed_counter_id in nonterminal_mixed_counter_ids
                if filter_nonterminal_mixed_chunks and is_nonterminal_mixed:
                    filtered_rows.append({
                        "row_index": row_index,
                        "phase": fpm_row.get("phase", ""),
                        "counter_id": fpm_row.get("counter_id", ""),
                        "reason": "mixed_nonterminal_prefill_chunk",
                        "latency_ms": fpm_row.get("latency_ms", ""),
                        "ctx_tokens": fpm_row.get("ctx_tokens", ""),
                        "ctx_requests": fpm_row.get("ctx_requests", ""),
                        "ctx_kv_tokens": fpm_row.get("ctx_kv_tokens", ""),
                        "decode_requests": fpm_row.get("decode_requests", ""),
                        "mean_decode_kv_tokens": fpm_row.get("mean_decode_kv_tokens", ""),
                    })
                    continue
                queue_adjacent_reason = queue_adjacent_mixed_reasons.get(mixed_counter_id)
                if filter_nonterminal_mixed_chunks and queue_adjacent_reason is not None:
                    filtered_rows.append({
                        "row_index": row_index,
                        "phase": fpm_row.get("phase", ""),
                        "counter_id": fpm_row.get("counter_id", ""),
                        "reason": queue_adjacent_reason,
                        "latency_ms": fpm_row.get("latency_ms", ""),
                        "ctx_tokens": fpm_row.get("ctx_tokens", ""),
                        "ctx_requests": fpm_row.get("ctx_requests", ""),
                        "ctx_kv_tokens": fpm_row.get("ctx_kv_tokens", ""),
                        "decode_requests": fpm_row.get("decode_requests", ""),
                        "mean_decode_kv_tokens": fpm_row.get("mean_decode_kv_tokens", ""),
                    })
                    continue
                decode_spike_adjacent_reason = decode_spike_adjacent_mixed_reasons.get(mixed_counter_id)
                if decode_spike_adjacent_reason is not None:
                    filtered_rows.append({
                        "row_index": row_index,
                        "phase": fpm_row.get("phase", ""),
                        "counter_id": fpm_row.get("counter_id", ""),
                        "reason": decode_spike_adjacent_reason,
                        "latency_ms": fpm_row.get("latency_ms", ""),
                        "ctx_tokens": fpm_row.get("ctx_tokens", ""),
                        "ctx_requests": fpm_row.get("ctx_requests", ""),
                        "ctx_kv_tokens": fpm_row.get("ctx_kv_tokens", ""),
                        "decode_requests": fpm_row.get("decode_requests", ""),
                        "mean_decode_kv_tokens": fpm_row.get("mean_decode_kv_tokens", ""),
                    })
                    continue
                if row_index in pathology_reasons:
                    filtered_rows.append({
                        "row_index": row_index,
                        "phase": fpm_row.get("phase", ""),
                        "counter_id": fpm_row.get("counter_id", ""),
                        "reason": pathology_reasons[row_index],
                        "latency_ms": fpm_row.get("latency_ms", ""),
                        "ctx_tokens": fpm_row.get("ctx_tokens", ""),
                        "ctx_requests": fpm_row.get("ctx_requests", ""),
                        "ctx_kv_tokens": fpm_row.get("ctx_kv_tokens", ""),
                        "decode_requests": fpm_row.get("decode_requests", ""),
                        "mean_decode_kv_tokens": fpm_row.get("mean_decode_kv_tokens", ""),
                    })
                    continue
                comparable_mixed_rows.append(fpm_row)
            for fpm_row in _aggregate_mixed_rows(comparable_mixed_rows, aggregation=aggregation):
                try:
                    (
                        fpm_ms,
                        aic_ms,
                        ctx_tokens,
                        ctx_requests,
                        ctx_kv_tokens,
                        ctx_prefix_tokens,
                        gen_tokens,
                        mean_decode_kv,
                        per_ops,
                        per_ops_source,
                    ) = _predict_mixed_fpm_row(fpm_row)
                except (AssertionError, KeyError, PerfDataNotAvailableError, ValueError):
                    continue
                output_row = {
                    "model": model_name,
                    "tp": tp,
                    "moe_tp": moe_tp,
                    "ep": ep,
                    "fpm_workload_segment": _normalized_workload_segment(fpm_workload_segment) or "all",
                    "phase": "mixed",
                    "shape": f"ctx{ctx_tokens}_gen{gen_tokens}_kv{mean_decode_kv:.3f}",
                    "fpm_ms": fpm_ms,
                    "aic_ms": float(aic_ms),
                    "error_pct": ((float(aic_ms) / fpm_ms) - 1.0) * 100.0,
                    "fpm_samples": int(fpm_row.get("_fpm_samples", 1)),
                    "fpm_match": f"aggregated:{aggregation}",
                    "counter_id": fpm_row.get("_counter_ids", fpm_row.get("counter_id", "")),
                    "ctx_tokens": ctx_tokens,
                    "ctx_requests": ctx_requests,
                    "ctx_kv_tokens": ctx_kv_tokens,
                    "ctx_prefix_tokens": ctx_prefix_tokens,
                    "decode_requests": gen_tokens,
                    "mean_decode_kv_tokens": mean_decode_kv,
                    "mixed_nonterminal_chunk": False,
                }
                if include_per_ops:
                    _add_mixed_per_op_columns(output_row, per_ops, per_ops_source)
                rows.append(output_row)
            for first_counter_id, sequence_rows in sequence_rows_by_first_counter_id.items():
                fpm_ms = 0.0
                aic_ms = 0.0
                total_ctx_tokens = 0
                max_ctx_requests = 0
                first_ctx_kv_tokens = 0
                first_ctx_prefix_tokens = 0
                last_gen_tokens = 0
                last_mean_decode_kv = 0.0
                summed_per_ops: dict[str, float] = defaultdict(float)
                summed_per_ops_source: dict[str, str] = {}
                try:
                    for component_index, fpm_row in enumerate(sequence_rows):
                        (
                            component_fpm_ms,
                            component_aic_ms,
                            ctx_tokens,
                            ctx_requests,
                            ctx_kv_tokens,
                            ctx_prefix_tokens,
                            gen_tokens,
                            mean_decode_kv,
                            per_ops,
                            per_ops_source,
                        ) = _predict_mixed_fpm_row(fpm_row)
                        fpm_ms += component_fpm_ms
                        aic_ms += component_aic_ms
                        total_ctx_tokens += ctx_tokens
                        max_ctx_requests = max(max_ctx_requests, ctx_requests)
                        if component_index == 0:
                            first_ctx_kv_tokens = ctx_kv_tokens
                            first_ctx_prefix_tokens = ctx_prefix_tokens
                        last_gen_tokens = gen_tokens
                        last_mean_decode_kv = mean_decode_kv
                        for op_name, op_ms in per_ops.items():
                            summed_per_ops[op_name] += float(op_ms)
                        for op_name, op_source in per_ops_source.items():
                            if not op_source:
                                continue
                            existing_source = summed_per_ops_source.get(op_name)
                            if existing_source is None:
                                summed_per_ops_source[op_name] = op_source
                            elif existing_source != op_source:
                                summed_per_ops_source[op_name] = "mixed"
                except (AssertionError, KeyError, PerfDataNotAvailableError, ValueError):
                    continue
                counter_ids = ",".join(str(fpm_row.get("counter_id", "")) for fpm_row in sequence_rows)
                output_row = {
                    "model": model_name,
                    "tp": tp,
                    "moe_tp": moe_tp,
                    "ep": ep,
                    "fpm_workload_segment": _normalized_workload_segment(fpm_workload_segment) or "all",
                    "phase": "mixed",
                    "shape": (
                        f"ctx{total_ctx_tokens}_chunks{len(sequence_rows)}"
                        f"_gen{last_gen_tokens}_kv{last_mean_decode_kv:.3f}"
                    ),
                    "fpm_ms": fpm_ms,
                    "aic_ms": aic_ms,
                    "error_pct": ((aic_ms / fpm_ms) - 1.0) * 100.0,
                    "fpm_samples": sum(int(fpm_row.get("_fpm_samples", 1)) for fpm_row in sequence_rows),
                    "fpm_match": f"sequence:{len(sequence_rows)}",
                    "counter_id": counter_ids,
                    "ctx_tokens": total_ctx_tokens,
                    "ctx_requests": max_ctx_requests,
                    "ctx_kv_tokens": first_ctx_kv_tokens,
                    "ctx_prefix_tokens": first_ctx_prefix_tokens,
                    "decode_requests": last_gen_tokens,
                    "mean_decode_kv_tokens": last_mean_decode_kv,
                    "mixed_nonterminal_chunk": False,
                }
                if include_per_ops:
                    _add_mixed_per_op_columns(output_row, summed_per_ops, summed_per_ops_source)
                rows.append(output_row)
    finally:
        vllm_backend._USE_LAYERWISE = old_use_layerwise

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "tp",
        "moe_tp",
        "ep",
        "fpm_workload_segment",
        "phase",
        "shape",
        "fpm_ms",
        "aic_ms",
        "error_pct",
        "fpm_samples",
        "fpm_match",
        "counter_id",
        "ctx_tokens",
        "ctx_requests",
        "ctx_kv_tokens",
        "ctx_prefix_tokens",
        "decode_requests",
        "mean_decode_kv_tokens",
        "fpm_representative_decode_kv",
        "aic_decode_past_kv",
        "mixed_nonterminal_chunk",
    ]
    if include_per_ops:
        for op_name in CONTEXT_PER_OP_FIELDS:
            fields.append(f"aic_op_{op_name}")
            fields.append(f"aic_source_{op_name}")
        for op_name in GENERATION_PER_OP_FIELDS:
            fields.append(f"aic_op_{op_name}")
            fields.append(f"aic_source_{op_name}")
        for op_name in MIXED_PER_OP_FIELDS:
            fields.append(f"aic_op_{op_name}")
            fields.append(f"aic_source_{op_name}")
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if filtered_output is not None:
        _write_filtered_rows(filtered_output, filtered_rows)
    return rows


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layerwise", type=Path, required=True)
    parser.add_argument("--fpm", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--systems-root", default="src/aiconfigurator/systems")
    parser.add_argument(
        "--moe-perf-file",
        type=Path,
        help=(
            "Optional run-local moe_perf CSV/TXT/parquet to overlay on top of --systems-root. "
            "Use this when op-level MoE data was collected separately from layerwise data."
        ),
    )
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--moe-tp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument(
        "--moe-workload-distribution",
        default="power_law",
        help="MoE workload distribution for AIC queries, for example power_law or sampled_zipf_1.2.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--filtered-output",
        type=Path,
        default=None,
        help="Filtered-row audit CSV. Defaults to OUTPUT with _filtered_rows suffix.",
    )
    parser.add_argument("--aggregation", choices=("median", "mean", "trimmed_mean"), default="trimmed_mean")
    parser.add_argument("--decode-past-kv", type=int, default=4096)
    parser.add_argument("--decode-osl", type=int, default=2)
    parser.add_argument(
        "--decode-match",
        choices=("exact", "nearest", "pooled"),
        default="nearest",
        help="Decode KV matching mode. Pooled compares against a forward steady-state KV window.",
    )
    parser.add_argument("--max-decode-kv-distance", type=float, default=4.0)
    parser.add_argument("--decode-pool-forward-window", type=float, default=6.0)
    parser.add_argument(
        "--fpm-workload-segment",
        default="sweep",
        help=(
            "Filter FPM rows by workload_segment when that column is populated. "
            "Defaults to sweep for static layerwise comparison; use all to include every segment."
        ),
    )
    parser.add_argument(
        "--vllm-max-num-batched-tokens",
        default="auto",
        help="vLLM scheduler max_num_batched_tokens for runtime metadata. Defaults to FPM metadata when present.",
    )
    parser.add_argument(
        "--filter-pathological-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter continuation-context FPM rows that are below a plausible latency floor.",
    )
    parser.add_argument("--pathological-context-min-continuation-ctx-tokens", type=int, default=128)
    parser.add_argument("--pathological-context-continuation-min-latency-ms", type=float, default=5.0)
    parser.add_argument("--pathological-context-peer-min-count", type=int, default=3)
    parser.add_argument("--pathological-context-high-latency-factor", type=float, default=3.0)
    parser.add_argument(
        "--filter-pathological-decode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter isolated high-latency FPM decode rows before bin matching.",
    )
    parser.add_argument("--pathological-decode-peer-kv-window", type=float, default=8.0)
    parser.add_argument("--pathological-decode-peer-batch-window", type=int, default=2)
    parser.add_argument("--pathological-decode-min-peer-count", type=int, default=1)
    parser.add_argument("--pathological-decode-latency-factor", type=float, default=5.0)
    parser.add_argument("--pathological-decode-min-latency-ms", type=float, default=20.0)
    parser.add_argument(
        "--include-mixed",
        action="store_true",
        help="Also compare mixed prefill+decode FPM scheduler rows with _get_mix_step_latency.",
    )
    parser.add_argument(
        "--filter-pathological-mixed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter mixed FPM rows whose latency is far below nearby large-context mixed rows.",
    )
    parser.add_argument(
        "--filter-nonterminal-mixed-chunks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Filter mixed rows that are intermediate chunked-prefill iterations. "
            "These rows do not represent complete target iteration shapes."
        ),
    )
    parser.add_argument("--pathological-mixed-tiny-ctx-tokens", type=int, default=320)
    parser.add_argument("--pathological-mixed-min-ctx-tokens", type=int, default=128)
    parser.add_argument("--pathological-mixed-peer-ctx-fraction", type=float, default=0.05)
    parser.add_argument("--pathological-mixed-peer-ctx-min-window", type=int, default=512)
    parser.add_argument("--pathological-mixed-min-peer-count", type=int, default=3)
    parser.add_argument("--pathological-mixed-latency-fraction", type=float, default=0.60)
    parser.add_argument("--pathological-mixed-high-latency-factor", type=float, default=1.2)
    parser.add_argument(
        "--pathological-mixed-decode-spike-window",
        type=int,
        default=5,
        help="Filter mixed rows within this many FPM rows of an isolated high-latency decode spike.",
    )
    parser.add_argument(
        "--include-per-ops",
        action="store_true",
        help="Include context, generation, and mixed-step AIC per-op latency/source columns for diagnostics.",
    )
    return parser


def main() -> None:
    """Run the diagnostic comparison."""

    args = _build_parser().parse_args()
    if args.vllm_max_num_batched_tokens == "auto":
        max_num_batched_tokens = _load_fpm_max_num_batched_tokens(
            args.fpm,
            workload_segment=args.fpm_workload_segment,
        )
    elif args.vllm_max_num_batched_tokens in ("", "none", "None"):
        max_num_batched_tokens = None
    else:
        max_num_batched_tokens = int(args.vllm_max_num_batched_tokens)
    filtered_output = args.filtered_output
    if filtered_output is None:
        filtered_output = args.output.with_name(f"{args.output.stem}_filtered_rows{args.output.suffix}")
    compare(
        layerwise_csv=args.layerwise,
        fpm_csv=args.fpm,
        model_name=args.model,
        tp=args.tp,
        moe_tp=args.moe_tp,
        ep=args.ep,
        moe_workload_distribution=args.moe_workload_distribution,
        output=args.output,
        filtered_output=filtered_output,
        aggregation=args.aggregation,
        decode_past_kv=args.decode_past_kv,
        decode_osl=args.decode_osl,
        decode_match=args.decode_match,
        max_decode_kv_distance=args.max_decode_kv_distance,
        decode_pool_forward_window=args.decode_pool_forward_window,
        include_mixed=args.include_mixed,
        vllm_max_num_batched_tokens=max_num_batched_tokens,
        filter_pathological_context=args.filter_pathological_context,
        pathological_context_min_continuation_ctx_tokens=args.pathological_context_min_continuation_ctx_tokens,
        pathological_context_continuation_min_latency_ms=args.pathological_context_continuation_min_latency_ms,
        pathological_context_peer_min_count=args.pathological_context_peer_min_count,
        pathological_context_high_latency_factor=args.pathological_context_high_latency_factor,
        filter_pathological_decode=args.filter_pathological_decode,
        pathological_decode_peer_kv_window=args.pathological_decode_peer_kv_window,
        pathological_decode_peer_batch_window=args.pathological_decode_peer_batch_window,
        pathological_decode_min_peer_count=args.pathological_decode_min_peer_count,
        pathological_decode_latency_factor=args.pathological_decode_latency_factor,
        pathological_decode_min_latency_ms=args.pathological_decode_min_latency_ms,
        filter_pathological_mixed=args.filter_pathological_mixed,
        filter_nonterminal_mixed_chunks=args.filter_nonterminal_mixed_chunks,
        fpm_workload_segment=args.fpm_workload_segment,
        pathological_mixed_tiny_ctx_tokens=args.pathological_mixed_tiny_ctx_tokens,
        pathological_mixed_min_ctx_tokens=args.pathological_mixed_min_ctx_tokens,
        pathological_mixed_peer_ctx_fraction=args.pathological_mixed_peer_ctx_fraction,
        pathological_mixed_peer_ctx_min_window=args.pathological_mixed_peer_ctx_min_window,
        pathological_mixed_min_peer_count=args.pathological_mixed_min_peer_count,
        pathological_mixed_latency_fraction=args.pathological_mixed_latency_fraction,
        pathological_mixed_high_latency_factor=args.pathological_mixed_high_latency_factor,
        pathological_mixed_decode_spike_window=args.pathological_mixed_decode_spike_window,
        include_per_ops=args.include_per_ops,
        systems_root=args.systems_root,
        moe_perf_file=args.moe_perf_file,
    )


if __name__ == "__main__":
    main()

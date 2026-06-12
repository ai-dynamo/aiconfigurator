#!/usr/bin/env python3
"""Compare AIC vLLM layerwise predictions against FPM phase rows.

This diagnostic is intentionally narrow: it uses an explicit layerwise CSV as
the layerwise database, reuses the repo's real communication/MoE tables, and
calls ``VLLMBackend`` phase estimators directly for FPM-comparable shapes.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from aiconfigurator.sdk import common, interpolation
from aiconfigurator.sdk.backends import vllm_backend
from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.operations.layerwise import (
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
    "mixed_moe_router",
    "mixed_moe_shared_expert",
]

CONTEXT_PER_OP_FIELDS = [
    "context_layerwise",
    "context_tp_allreduce",
    "context_moe_tp_allreduce",
    "context_moe_ep_alltoall",
    "context_moe",
    "context_moe_router",
    "context_moe_shared_expert",
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
    filter_nonterminal_chunks: bool = True,
    nonterminal_chunk_lookahead: int = 3,
) -> dict[int, str]:
    """Return pathology reasons for context rows that are not clean targets."""

    reasons: dict[int, str] = {}
    for index, row in enumerate(rows):
        if str(row.get("phase", "")).lower() != "context":
            continue
        ctx_tokens = int(float(row.get("ctx_tokens") or 0))
        ctx_requests = int(float(row.get("ctx_requests") or 1))
        ctx_kv_tokens = int(float(row.get("ctx_kv_tokens") or 0))
        latency_ms = float(row.get("latency_ms") or 0.0)
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

        if not filter_nonterminal_chunks or ctx_kv_tokens > 0 or ctx_tokens <= 0:
            continue
        chunk_kv_tokens = ctx_tokens * max(ctx_requests, 1)
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
    return reasons


def _load_fpm(
    path: Path,
    *,
    filter_pathological_context: bool = False,
    pathological_context_min_continuation_ctx_tokens: int = 128,
    pathological_context_continuation_min_latency_ms: float = 5.0,
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
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
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
        )
        if filter_pathological_context
        else {}
    )
    context_rows = rows
    if rows and str(rows[0].get("phase", "")).lower() == "context":
        context_rows = []
        for row in rows:
            if str(row.get("phase", "")).lower() != "context":
                break
            context_rows.append(row)
    consumed_context_indexes: set[int] = set()
    for index, row in enumerate(context_rows):
        if index in consumed_context_indexes:
            continue
        if str(row.get("phase", "")).lower() != "context":
            continue
        ctx_requests = int(row["ctx_requests"])
        ctx_tokens = int(row["ctx_tokens"])
        ctx_kv_tokens = int(float(row.get("ctx_kv_tokens") or 0))
        if ctx_requests <= 0 or ctx_tokens <= 0:
            continue
        ctx_prefix_tokens = round(ctx_kv_tokens / max(ctx_requests, 1))
        total_ctx_tokens = ctx_tokens
        total_latency = float(row["latency_ms"])
        consumed_context_indexes.add(index)
        expected_next_kv_tokens = ctx_kv_tokens + ctx_tokens * ctx_requests
        next_index = index + 1
        while next_index < len(context_rows):
            next_row = context_rows[next_index]
            if str(next_row.get("phase", "")).lower() != "context":
                break
            next_ctx_requests = int(next_row["ctx_requests"])
            next_ctx_kv_tokens = int(float(next_row.get("ctx_kv_tokens") or 0))
            if next_ctx_requests != ctx_requests or next_ctx_kv_tokens != expected_next_kv_tokens:
                break
            next_ctx_tokens = int(next_row["ctx_tokens"])
            total_ctx_tokens += next_ctx_tokens
            total_latency += float(next_row["latency_ms"])
            consumed_context_indexes.add(next_index)
            expected_next_kv_tokens += next_ctx_tokens * ctx_requests
            next_index += 1
        context[(ctx_requests, total_ctx_tokens, ctx_prefix_tokens)].append(total_latency)

    for index, row in enumerate(rows):
        pathology_reason = decode_pathology_reasons.get(index) or context_pathology_reasons.get(index)
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


def _load_fpm_max_num_batched_tokens(fpm_csv: Path) -> int | None:
    """Return the effective FPM context-token budget for AIC chunking.

    vLLM's serialized ``scheduler_config.max_num_batched_tokens`` is normally
    the right value. Some architectures can still report phase rows whose
    scheduled context-token count is larger than that metadata value. In that
    case the FPM row is the more authoritative measurement boundary for this
    diagnostic, so use the observed scheduler budget as a lower bound.
    """

    candidates = [
        fpm_csv.parent / "effective_vllm_config.json",
        fpm_csv.parent / "vllm_metadata.json",
    ]
    keys = (
        "scheduler_config.max_num_batched_tokens",
        "max_num_batched_tokens",
    )
    metadata_value: int | None = None
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
            break

    observed_value = _infer_observed_fpm_context_budget(fpm_csv)
    if metadata_value is None:
        return observed_value
    if observed_value is None:
        return metadata_value
    return max(metadata_value, observed_value)


def _infer_observed_fpm_context_budget(fpm_csv: Path) -> int | None:
    """Infer a lower-bound context scheduler budget from FPM phase rows."""

    if not fpm_csv.exists():
        return None
    observed = 0
    with fpm_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            phase = str(row.get("phase", "")).lower()
            if phase not in {"context", "mixed"}:
                continue
            ctx_tokens = int(float(row.get("ctx_tokens") or 0))
            ctx_kv_tokens = int(float(row.get("ctx_kv_tokens") or 0))
            if ctx_tokens <= 0:
                continue
            # Prefix-zero rows expose the initial scheduled chunk. Continuation
            # rows with a nonzero prefix still prove the scheduler accepted that
            # many new context tokens in one FPM iteration.
            observed = max(observed, ctx_tokens)
            if ctx_kv_tokens == 0:
                observed = max(observed, ctx_tokens)
    return observed or None


def _mixed_pathology_reasons(
    rows: list[dict[str, str]],
    *,
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

    parsed: list[tuple[int, float, float, float, int, float]] = []
    for index, row in enumerate(rows):
        ctx_tokens = float(row.get("ctx_tokens") or 0.0)
        ctx_kv_tokens = float(row.get("ctx_kv_tokens") or 0.0)
        decode_requests = int(float(row.get("decode_requests") or 0.0))
        mean_decode_kv = float(row.get("mean_decode_kv_tokens") or 0.0)
        latency_ms = float(row.get("latency_ms") or 0.0)
        parsed.append((index, ctx_tokens, latency_ms, ctx_kv_tokens, decode_requests, mean_decode_kv))
    parsed_decode: list[tuple[int, float, float]] = []
    for row in decode_rows or []:
        parsed_decode.append((
            int(float(row.get("decode_requests") or 0.0)),
            float(row.get("mean_decode_kv_tokens") or 0.0),
            float(row.get("latency_ms") or 0.0),
        ))

    reasons: dict[int, str] = {}
    for position, (index, ctx_tokens, latency_ms, ctx_kv_tokens, decode_requests, mean_decode_kv) in enumerate(parsed):
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
        if 0 < position < len(parsed) - 1 and decode_requests > 0:
            _, _, prev_latency, _, prev_decode_requests, prev_mean_decode_kv = parsed[position - 1]
            _, _, next_latency, _, next_decode_requests, next_mean_decode_kv = parsed[position + 1]
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
        if ctx_tokens < min_ctx_tokens:
            continue
        window = max(float(peer_ctx_min_window), abs(ctx_tokens) * peer_ctx_fraction)
        peers = [
            peer_latency
            for _, peer_ctx_tokens, peer_latency, _, _, _ in parsed
            if abs(peer_ctx_tokens - ctx_tokens) <= window and peer_latency > 0.0
        ]
        if len(peers) >= min_peer_count:
            peer_median = float(statistics.median(peers))
            if latency_ms < peer_median * latency_fraction:
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
        next_prefix = ctx_kv_tokens + ctx_tokens * ctx_requests
        for next_index in range(index + 1, min(len(rows), index + 1 + max(lookahead, 0))):
            next_row = rows[next_index]
            if str(next_row.get("phase", "")).lower() not in {"context", "mixed"}:
                continue
            next_ctx_kv_tokens = int(float(next_row.get("ctx_kv_tokens") or 0))
            if next_ctx_kv_tokens == next_prefix:
                counter_ids.add(str(row.get("counter_id", "")))
                break
    return counter_ids


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
        self.attention_dp_size = 1
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
    ) -> dict[str, Any]:
        """Return an exact layerwise detail row."""

        phase_key = phase.upper()
        if moe_weight_mode:
            mode_index = self.layerwise.get("__mode_index__", {})
            try:
                model_data = mode_index[model.lower()][phase_key][tp_size][str(moe_weight_mode)]
            except KeyError:
                model_data = self.layerwise[model.lower()][phase_key][tp_size]
        else:
            model_data = self.layerwise[model.lower()][phase_key][tp_size]
        if phase_key == "CTX":
            if seq_len in model_data and seq_len_kv_cache in model_data[seq_len]:
                return self._normalize_detail(model_data[seq_len][seq_len_kv_cache])
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
            return self._normalize_detail(model_data[batch_size][seq_len])
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
        return self._normalize_detail(result)

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
        ):
            if result.get(metric) not in (None, ""):
                out[metric] = float(result[metric])
        for metric in ("latency_source", "measurement_mode", "attribution_target", "vllm_config_hash"):
            if result.get(metric) not in (None, ""):
                out[metric] = str(result[metric])
        if result.get("moe_weight_mode") not in (None, ""):
            out["moe_weight_mode"] = str(result["moe_weight_mode"])
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
            shared_expert_inter_size=512,
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
    filter_pathological_decode: bool,
    pathological_decode_peer_kv_window: float,
    pathological_decode_peer_batch_window: int,
    pathological_decode_min_peer_count: int,
    pathological_decode_latency_factor: float,
    pathological_decode_min_latency_ms: float,
    filter_pathological_mixed: bool,
    filter_nonterminal_mixed_chunks: bool,
    pathological_mixed_tiny_ctx_tokens: int,
    pathological_mixed_min_ctx_tokens: int,
    pathological_mixed_peer_ctx_fraction: float,
    pathological_mixed_peer_ctx_min_window: int,
    pathological_mixed_min_peer_count: int,
    pathological_mixed_latency_fraction: float,
    pathological_mixed_high_latency_factor: float,
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
    backend = VLLMBackend()
    runtime_config = RuntimeConfig(vllm_max_num_batched_tokens=vllm_max_num_batched_tokens)
    old_use_layerwise = vllm_backend._USE_LAYERWISE
    vllm_backend._USE_LAYERWISE = True
    context, decode, filtered_rows = _load_fpm(
        fpm_csv,
        filter_pathological_context=filter_pathological_context,
        pathological_context_min_continuation_ctx_tokens=pathological_context_min_continuation_ctx_tokens,
        pathological_context_continuation_min_latency_ms=pathological_context_continuation_min_latency_ms,
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
                latency, _, sources = backend._run_context_phase(
                    model,
                    database,
                    runtime_config,
                    batch_size=batch_size,
                    isl=ctx_tokens + ctx_prefix_tokens,
                    prefix=ctx_prefix_tokens,
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
                latency, _, sources = backend._run_generation_phase(
                    model,
                    database,
                    runtime_config,
                    batch_size=batch_size,
                    beam_width=1,
                    isl=aic_past_kv,
                    osl=decode_osl,
                    stride=32,
                )
            except (AssertionError, KeyError, PerfDataNotAvailableError, ValueError):
                continue
            aic_ms = float(sum(latency.values()))
            output_row = {
                "model": model_name,
                "tp": tp,
                "moe_tp": moe_tp,
                "ep": ep,
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
            with fpm_csv.open(newline="") as f:
                phase_rows = list(csv.DictReader(f))
            mixed_rows = [fpm_row for fpm_row in phase_rows if fpm_row["phase"] == "mixed"]
            decode_rows = [fpm_row for fpm_row in phase_rows if fpm_row["phase"] == "decode"]
            nonterminal_mixed_counter_ids = _nonterminal_mixed_chunk_counter_ids(phase_rows)
            pathology_reasons = (
                _mixed_pathology_reasons(
                    mixed_rows,
                    decode_rows=decode_rows,
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
            for row_index, fpm_row in enumerate(mixed_rows):
                is_nonterminal_mixed = str(fpm_row.get("counter_id", "")) in nonterminal_mixed_counter_ids
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
                fpm_ms = float(fpm_row["latency_ms"])
                ctx_tokens = int(fpm_row["ctx_tokens"])
                ctx_requests = int(fpm_row["ctx_requests"])
                ctx_kv_tokens = int(float(fpm_row.get("ctx_kv_tokens") or 0))
                ctx_prefix_tokens = round(ctx_kv_tokens / max(ctx_requests, 1))
                gen_tokens = int(fpm_row["decode_requests"])
                mean_decode_kv = float(fpm_row["mean_decode_kv_tokens"])
                try:
                    aic_ms, _, per_ops, per_ops_source = backend._get_mix_step_latency(
                        model,
                        database,
                        runtime_config,
                        ctx_tokens=ctx_tokens,
                        gen_tokens=gen_tokens,
                        isl=round(mean_decode_kv),
                        osl=1,
                        prefix=ctx_prefix_tokens,
                    )
                except (AssertionError, KeyError, PerfDataNotAvailableError, ValueError):
                    continue
                output_row = {
                    "model": model_name,
                    "tp": tp,
                    "moe_tp": moe_tp,
                    "ep": ep,
                    "phase": "mixed",
                    "shape": f"ctx{ctx_tokens}_gen{gen_tokens}_kv{mean_decode_kv:.3f}",
                    "fpm_ms": fpm_ms,
                    "aic_ms": float(aic_ms),
                    "error_pct": ((float(aic_ms) / fpm_ms) - 1.0) * 100.0,
                    "fpm_samples": 1,
                    "fpm_match": f"row:{fpm_row.get('counter_id', '')}",
                    "counter_id": fpm_row.get("counter_id", ""),
                    "ctx_tokens": ctx_tokens,
                    "ctx_requests": ctx_requests,
                    "ctx_kv_tokens": ctx_kv_tokens,
                    "ctx_prefix_tokens": ctx_prefix_tokens,
                    "decode_requests": gen_tokens,
                    "mean_decode_kv_tokens": mean_decode_kv,
                    "mixed_nonterminal_chunk": is_nonterminal_mixed,
                }
                if include_per_ops:
                    _add_mixed_per_op_columns(output_row, per_ops, per_ops_source)
                rows.append(output_row)
    finally:
        vllm_backend._USE_LAYERWISE = old_use_layerwise

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "tp",
        "moe_tp",
        "ep",
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
        "--vllm-max-num-batched-tokens",
        default="auto",
        help="vLLM scheduler max_num_batched_tokens for AIC chunking. Defaults to FPM metadata when present.",
    )
    parser.add_argument(
        "--filter-pathological-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter continuation-context FPM rows that are below a plausible latency floor.",
    )
    parser.add_argument("--pathological-context-min-continuation-ctx-tokens", type=int, default=128)
    parser.add_argument("--pathological-context-continuation-min-latency-ms", type=float, default=5.0)
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
    parser.add_argument("--pathological-mixed-tiny-ctx-tokens", type=int, default=256)
    parser.add_argument("--pathological-mixed-min-ctx-tokens", type=int, default=128)
    parser.add_argument("--pathological-mixed-peer-ctx-fraction", type=float, default=0.05)
    parser.add_argument("--pathological-mixed-peer-ctx-min-window", type=int, default=512)
    parser.add_argument("--pathological-mixed-min-peer-count", type=int, default=3)
    parser.add_argument("--pathological-mixed-latency-fraction", type=float, default=0.25)
    parser.add_argument("--pathological-mixed-high-latency-factor", type=float, default=1.35)
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
        max_num_batched_tokens = _load_fpm_max_num_batched_tokens(args.fpm)
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
        filter_pathological_decode=args.filter_pathological_decode,
        pathological_decode_peer_kv_window=args.pathological_decode_peer_kv_window,
        pathological_decode_peer_batch_window=args.pathological_decode_peer_batch_window,
        pathological_decode_min_peer_count=args.pathological_decode_min_peer_count,
        pathological_decode_latency_factor=args.pathological_decode_latency_factor,
        pathological_decode_min_latency_ms=args.pathological_decode_min_latency_ms,
        filter_pathological_mixed=args.filter_pathological_mixed,
        filter_nonterminal_mixed_chunks=args.filter_nonterminal_mixed_chunks,
        pathological_mixed_tiny_ctx_tokens=args.pathological_mixed_tiny_ctx_tokens,
        pathological_mixed_min_ctx_tokens=args.pathological_mixed_min_ctx_tokens,
        pathological_mixed_peer_ctx_fraction=args.pathological_mixed_peer_ctx_fraction,
        pathological_mixed_peer_ctx_min_window=args.pathological_mixed_peer_ctx_min_window,
        pathological_mixed_min_peer_count=args.pathological_mixed_min_peer_count,
        pathological_mixed_latency_fraction=args.pathological_mixed_latency_fraction,
        pathological_mixed_high_latency_factor=args.pathological_mixed_high_latency_factor,
        include_per_ops=args.include_per_ops,
        systems_root=args.systems_root,
        moe_perf_file=args.moe_perf_file,
    )


if __name__ == "__main__":
    main()

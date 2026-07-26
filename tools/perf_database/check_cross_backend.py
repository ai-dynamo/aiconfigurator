# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Two-layer cross-backend consistency check of perf-database perf tables.

For every (system, op_file) present in >= 2 backends, join the latest version
of each backend on the shape key and emit findings on two layers:

  Layer 1 — ANOMALY (data-validity, actionable):
    - `nonpositive_latency`: rows whose latency is <= 0 — a classic collector
                           corruption mode (timer failure, unit bug). Counted
                           per (backend, kernel_source) and excluded from the
                           statistical checks below. Tables whose latency is a
                           difference/calibration value (see
                           _DELTA_LATENCY_OP_FILES) are exempt.
    - `pair_outlier`     : a shape point whose cross-backend latency ratio
                           deviates from its *local baseline* by more than
                           --anomaly-factor (default 3x). The baseline is
                           hierarchical — shape bucket, then series (non-sweep
                           shape columns), then op-level median — so
                           legitimate shape-dependent framework gaps are
                           absorbed and what's left is likely a bad
                           measurement in one of the two tables. A single
                           pair cannot tell WHICH side is bad (one side too
                           slow and the other too fast look identical), so
                           each outlier names both sides as candidates and
                           attribution is decided by corroboration: a side
                           that deviates against >= 2 distinct reference
                           backends is reported as the likely suspect.
    - `region_deviation` : a whole shape bucket whose median ratio deviates
                           from the op-level median by more than the anomaly
                           factor. Not a stray point — a whole region of the
                           joined table deviates from how these two backends
                           normally compare (bad sweep, degenerate kernel
                           path, wrong unit).
    - `mono_violation`   : within one backend, latency drops by more than
                           (1 - --mono-tolerance) while a sweep dimension
                           (batch_size / isl / m / step) grows with all other
                           shape columns fixed. Points below --noise-floor
                           latency are exempt (timer noise regime; the floor
                           is auto-scaled for microsecond-unit tables).

  Layer 2 — GAP (framework-difference, informational):
    - per shape-bucket median latency ratios between backend pairs, annotated
      with each side's kernel_source (e.g. fa3 vs flash_attention vs triton).
      Buckets whose median ratio exceeds --gap-factor (default 1.5x) are
      surfaced as significant-but-plausible framework differences, not data
      errors.

Backend pairs whose latest tables disagree on shape columns are NOT force-
joined: extra columns that are constant in their table are dropped (noted in
the report); otherwise the pair is skipped and reported as `schema_mismatch`.

Shape-key convention follows check_kernel_source.py: every column that is not
a meta column ({framework, version, device, op_name, kernel_source}) and not
a latency column is part of the shape key. Sweep columns get log2-bucketed to
form the local-baseline bucket key.

Usage:
    python3 tools/perf_database/check_cross_backend.py \\
        --data-root aic-core/src/aiconfigurator_core/systems/data \\
        --systems h200_sxm \\
        --out-md   $TMPDIR/cross-backend-check.md \\
        --out-json $TMPDIR/cross-backend-check.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns that never participate in the shape key (same set as check_kernel_source.py).
_META_COLUMNS = {"framework", "version", "device", "op_name", "kernel_source"}

# Latency-like columns. The first one found in the header is used as the
# metric; ALL of them are excluded from the shape key (some comm tables carry
# two latency measurements side by side).
_LATENCY_COLUMNS_PRIORITY = (
    "latency",
    "avg_ms",
    "combine_avg_t_us",
    "dispatch_avg_t_us",
)

# Shape columns treated as sweep dimensions: log2-bucketed for the local
# baseline, and checked for latency monotonicity within a backend.
_SWEEP_COLUMNS = ("batch_size", "isl", "m", "num_tokens", "step")

# Backend directory names to skip — framework-agnostic by construction.
_SKIP_BACKEND_DIRS = {"nccl", "oneccl"}

# Legacy top-level backend dirs (see check_kernel_source.py for the sync note).
_LEGACY_BACKEND_DIRS = {"trtllm", "sglang", "vllm"}

# Tables whose latency is a DIFFERENCE or calibration value, where <= 0 is
# semantically valid — exempt from the nonpositive_latency check (their <= 0
# rows are still excluded from the log-ratio statistics, which need positives):
#   - computescale: latency = dynamic - static quant pass
#     (collector/trtllm/collect_computescale.py:78 stores it unclamped, so
#     negatives are expected; sglang/vllm clamp to 0.0)
#   - dsv4_csa_topk_calib / glm5_topk_module: flat/top_last score_mode row
#     pairs consumed as DELTA = flat - top_last by perf_database
#     (collector/sglang/deepseekv4_sparse_modules.py:233); sub-resolution
#     kernels legitimately record 0.0
_DELTA_LATENCY_OP_FILES = {
    "computescale_perf.parquet",
    "dsv4_csa_topk_calib_perf.parquet",
    "glm5_topk_module_perf.parquet",
}

_PRE_RELEASE_TAGS = {"rc", "a", "b", "c", "alpha", "beta", "dev", "pre", "preview"}


def _version_key(version: str) -> tuple:
    """Sortable key for backend version strings like '1.3.0rc10' or '0.5.6.post2'.

    Every element is a (rank, number, text) triple. A terminator element with
    rank 1 is appended so that a release compares HIGHER than its own
    pre-release ('1.0.0' > '1.0.0rc4' — the rc's next element has rank 0) and
    LOWER than its post-release ('0.5.6.post2' > '0.5.6' — rank 2 > 1).
    """
    parts = re.findall(r"(\d+|[a-zA-Z]+)", version)
    key: list[tuple[int, int, str]] = []
    for p in parts:
        if p.isdigit():
            key.append((1, int(p), ""))
        else:
            rank = 0 if p.lower() in _PRE_RELEASE_TAGS else 2
            key.append((rank, 0, p.lower()))
    key.append((1, -1, ""))
    return tuple(key)


def _iter_op_tables(data_root: Path) -> Iterable[tuple[str, str, str, str, Path]]:
    """Yield (system, backend, version, op_file, path) across legacy and
    family-first layouts. Only *.parquet / *.txt tables are yielded."""
    for system_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        backend_dirs: list[tuple[str, Path]] = []
        for entry in sorted(p for p in system_dir.iterdir() if p.is_dir()):
            if entry.name in _SKIP_BACKEND_DIRS:
                continue
            if entry.name in _LEGACY_BACKEND_DIRS:
                backend_dirs.append((entry.name, entry))
            else:  # family dir
                backend_dirs.extend(
                    (b.name, b)
                    for b in sorted(p for p in entry.iterdir() if p.is_dir())
                    if b.name not in _SKIP_BACKEND_DIRS
                )
        for backend, backend_dir in backend_dirs:
            for version_dir in sorted(p for p in backend_dir.iterdir() if p.is_dir()):
                for path in sorted(itertools.chain(version_dir.glob("*.parquet"), version_dir.glob("*.txt"))):
                    if path.name in ("INCOMPLETE.txt",):
                        continue
                    yield system_dir.name, backend, version_dir.name, path.name, path


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _pick_latency_column(columns: Iterable[str]) -> str | None:
    for candidate in _LATENCY_COLUMNS_PRIORITY:
        if candidate in columns:
            return candidate
    return None


def _log2_bucket(value) -> object:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return value
    if v <= 0 or math.isnan(v):
        return value
    return math.floor(math.log2(v))


@dataclass
class OpTable:
    """One backend's latest-version table for a (system, op_file), reduced to
    min latency per shape key (best kernel wins; its kernel_source is kept)."""

    backend: str
    version: str
    shape_cols: list[str]
    frame: pd.DataFrame  # shape_cols + latency + kernel_source
    kernel_sources: list[str]
    # 1000.0 for microsecond-unit latency columns (*_us), 1.0 for ms tables.
    noise_scale: float

    @property
    def label(self) -> str:
        return f"{self.backend}/{self.version}"


def _load_op_table(path: Path, backend: str, version: str) -> tuple[OpTable | None, dict[str, int]]:
    """Returns (table, nonpositive-latency row counts by kernel_source).

    Nonpositive rows are excluded from the returned frame but reported so the
    caller can emit them as Layer-1 anomalies instead of silently cleaning
    them up.
    """
    df = _read_table(path)
    latency_col = _pick_latency_column(df.columns)
    if latency_col is None:
        logger.warning("No latency column in %s; skipping", path)
        return None, {}
    if "kernel_source" not in df.columns:
        df["kernel_source"] = "<unknown>"
    nonpositive = df[df[latency_col].notna() & (df[latency_col] <= 0)]
    npos_by_ks = {str(k): int(v) for k, v in nonpositive["kernel_source"].value_counts().items()}
    shape_cols = [c for c in df.columns if c not in _META_COLUMNS and c not in _LATENCY_COLUMNS_PRIORITY]
    df = df[df[latency_col] > 0]
    if df.empty or not shape_cols:
        return None, npos_by_ks
    df = df.rename(columns={latency_col: "latency"})
    # Min latency per shape key, keeping the winning kernel_source.
    idx = df.groupby(shape_cols, dropna=False)["latency"].idxmin()
    reduced = df.loc[idx, [*shape_cols, "latency", "kernel_source"]].reset_index(drop=True)
    table = OpTable(
        backend=backend,
        version=version,
        shape_cols=shape_cols,
        frame=reduced,
        kernel_sources=sorted(df["kernel_source"].astype(str).unique()),
        noise_scale=1000.0 if latency_col.endswith("_us") else 1.0,
    )
    return table, npos_by_ks


def _join_cols(df: pd.DataFrame, cols: list[str], bucket_sweeps: bool) -> pd.Series:
    parts = []
    for c in cols:
        col = df[c]
        if bucket_sweeps and c in _SWEEP_COLUMNS:
            col = col.map(_log2_bucket)
        parts.append(c + "=" + col.astype(str))
    out = parts[0]
    for p in parts[1:]:
        out = out + "|" + p
    return out


def _categorical_cols(df: pd.DataFrame, shape_cols: list[str]) -> list[str]:
    """Shape columns that are string-valued (dtype labels etc.) — used as the
    cluster key so point anomalies aggregate into reviewable findings."""
    return [c for c in shape_cols if not pd.api.types.is_numeric_dtype(df[c])]


def _align_shape_columns(a: OpTable, b: OpTable) -> tuple[list[str], pd.DataFrame, pd.DataFrame, list[str]] | None:
    """Reconcile the two tables' shape columns before joining.

    Extra columns that are single-valued in their table are dropped (noted).
    If either side has a genuinely varying extra column, the pair cannot be
    joined shape-for-shape — return None so the caller reports the mismatch
    instead of producing a many-to-many join across different shapes.
    """
    notes: list[str] = []
    frames = {id(a): a.frame, id(b): b.frame}
    for own, other in ((a, b), (b, a)):
        extra = [c for c in own.shape_cols if c not in other.shape_cols]
        for col in extra:
            values = frames[id(own)][col]
            if values.nunique(dropna=False) == 1:
                notes.append(f"{own.label}: dropped constant column {col}={values.iloc[0]!r}")
                frames[id(own)] = frames[id(own)].drop(columns=[col])
            else:
                return None
    shape_cols = [c for c in a.shape_cols if c in b.shape_cols]
    if not shape_cols:
        return None
    return shape_cols, frames[id(a)], frames[id(b)], notes


def _check_pair(
    system: str,
    op_file: str,
    a: OpTable,
    b: OpTable,
    anomaly_factor: float,
    gap_factor: float,
    min_bucket_points: int,
) -> tuple[list[dict], list[dict]]:
    """Compare two backends' tables. Returns (anomalies, gaps)."""
    pair = f"{b.label} vs {a.label}"
    aligned = _align_shape_columns(a, b)
    if aligned is None:
        gap = {
            "kind": "schema_mismatch",
            "system": system,
            "op_file": op_file,
            "pair": pair,
            "shape_cols_a": a.shape_cols,
            "shape_cols_b": b.shape_cols,
        }
        return [], [gap]
    shape_cols, frame_a, frame_b, align_notes = aligned
    merged = frame_a.merge(frame_b, on=shape_cols, suffixes=("_a", "_b"))
    if merged.empty:
        return [], []

    merged["log_ratio"] = np.log(merged["latency_b"] / merged["latency_a"])
    merged["bucket"] = _join_cols(merged, shape_cols, bucket_sweeps=True)
    series_cols = [c for c in shape_cols if c not in _SWEEP_COLUMNS]
    merged["series"] = _join_cols(merged, series_cols, bucket_sweeps=False) if series_cols else ""

    # Hierarchical baseline: bucket -> series -> op-level median. Each level
    # only applies when it has enough points to be a baseline at all.
    global_med = merged["log_ratio"].median()
    bucket_med = merged.groupby("bucket")["log_ratio"].transform("median")
    bucket_n = merged.groupby("bucket")["log_ratio"].transform("size")
    series_med = merged.groupby("series")["log_ratio"].transform("median")
    series_n = merged.groupby("series")["log_ratio"].transform("size")
    baseline = pd.Series(global_med, index=merged.index)
    baseline = series_med.where(series_n >= min_bucket_points, baseline)
    baseline = bucket_med.where(bucket_n >= min_bucket_points, baseline)
    merged["deviation"] = np.exp((merged["log_ratio"] - baseline).abs())
    merged["baseline"] = baseline

    cat_cols = _categorical_cols(merged, shape_cols)
    log_anomaly = math.log(anomaly_factor)
    log_gap = math.log(gap_factor)

    # ---- Layer 1: per-point outliers vs. local baseline --------------------
    # A single pair cannot attribute the fault: if the ratio is above the
    # baseline, either b is too slow or a is too fast. Both sides are named
    # as candidates; cluster_suspects() resolves attribution by corroboration.
    anomalies: list[dict] = []
    flagged = merged[merged["deviation"] > anomaly_factor]
    for row in flagged.itertuples():
        above = row.log_ratio > row.baseline
        candidates = [
            {"side": b.label if above else a.label, "kind": "slower_than_expected"},
            {"side": a.label if above else b.label, "kind": "faster_than_expected"},
        ]
        shape = {c: _jsonable(getattr(row, c)) for c in shape_cols}
        anomalies.append(
            {
                "kind": "pair_outlier",
                "system": system,
                "op_file": op_file,
                "pair": pair,
                "shape": shape,
                "shape_sig": "|".join(f"{k}={v}" for k, v in shape.items()),
                "cat_shape": {c: shape[c] for c in cat_cols},
                "latency_a": float(row.latency_a),
                "latency_b": float(row.latency_b),
                "kernel_source_a": str(row.kernel_source_a),
                "kernel_source_b": str(row.kernel_source_b),
                "ratio": float(np.exp(row.log_ratio)),
                "local_baseline_ratio": float(np.exp(row.baseline)),
                "deviation": float(row.deviation),
                "candidates": candidates,
            }
        )

    # ---- Bucket-level stats: region deviations (L1) + gaps (L2) ------------
    gaps: list[dict] = []
    grp = merged.groupby("bucket").agg(
        median_log_ratio=("log_ratio", "median"),
        points=("log_ratio", "size"),
        ks_a=("kernel_source_a", lambda s: ",".join(sorted(set(s.astype(str))))),
        ks_b=("kernel_source_b", lambda s: ",".join(sorted(set(s.astype(str))))),
    )
    grp = grp[grp["points"] >= min_bucket_points]

    # A region deviates when its median is far from how these two backends
    # normally compare on this op (the op-level median) — NOT when the
    # absolute gap is large, which would flag every bucket of a legitimately
    # slower backend.
    rel = (grp["median_log_ratio"] - global_med).abs()
    regions = grp[rel > log_anomaly]
    for bucket, row in regions.sort_values(
        "median_log_ratio", key=lambda s: (s - global_med).abs(), ascending=False
    ).iterrows():
        anomalies.append(
            {
                "kind": "region_deviation",
                "system": system,
                "op_file": op_file,
                "pair": pair,
                "bucket": bucket,
                "median_ratio": float(np.exp(row["median_log_ratio"])),
                "op_median_ratio": float(np.exp(global_med)),
                "rel_deviation": float(np.exp(abs(row["median_log_ratio"] - global_med))),
                "points": int(row["points"]),
                "kernel_source_a": row["ks_a"],
                "kernel_source_b": row["ks_b"],
            }
        )

    sig = grp[(grp["median_log_ratio"].abs() > log_gap) & (rel <= log_anomaly)]
    for bucket, row in sig.sort_values("median_log_ratio", key=lambda s: s.abs(), ascending=False).iterrows():
        gaps.append(
            {
                "kind": "framework_gap",
                "system": system,
                "op_file": op_file,
                "pair": pair,
                "bucket": bucket,
                "median_ratio": float(np.exp(row["median_log_ratio"])),
                "points": int(row["points"]),
                "kernel_source_a": row["ks_a"],
                "kernel_source_b": row["ks_b"],
            }
        )

    # Pair-level headline (always emitted, even when under the gap threshold).
    gaps.append(
        {
            "kind": "pair_summary",
            "system": system,
            "op_file": op_file,
            "pair": pair,
            "joined_points": len(merged),
            "median_ratio": float(np.exp(global_med)),
            "p05_ratio": float(np.exp(merged["log_ratio"].quantile(0.05))),
            "p95_ratio": float(np.exp(merged["log_ratio"].quantile(0.95))),
            "gap_buckets": len(sig),
            "region_deviation_buckets": len(regions),
            "total_buckets": len(grp),
            "kernel_sources_a": a.kernel_sources,
            "kernel_sources_b": b.kernel_sources,
            "align_notes": align_notes,
        }
    )
    return anomalies, gaps


def _check_monotonicity(system: str, op_file: str, t: OpTable, mono_tolerance: float, noise_floor: float) -> list[dict]:
    """Flag latency drops > (1 - mono_tolerance) while one sweep column grows
    and all other shape columns stay fixed. Points below the noise floor
    (scaled to the table's latency unit) are exempt — sub-noise timings
    jitter without meaning."""
    findings: list[dict] = []
    floor = noise_floor * t.noise_scale
    sweep_present = [c for c in t.shape_cols if c in _SWEEP_COLUMNS]
    for sweep in sweep_present:
        others = [c for c in t.shape_cols if c != sweep]
        df = t.frame
        vals = pd.to_numeric(df[sweep], errors="coerce")
        sub = df[vals.notna()].assign(_sweep=vals[vals.notna()])
        if sub.empty:
            continue
        sub = sub.sort_values("_sweep")
        grouped = sub.groupby(others, dropna=False) if others else [((), sub)]
        for key, series in grouped:
            lat = series["latency"].to_numpy()
            sw = series["_sweep"].to_numpy()
            if len(lat) < 2:
                continue
            drop = lat[1:] / lat[:-1]
            bad = np.nonzero((drop < mono_tolerance) & (lat[:-1] >= floor))[0]
            for i in bad:
                fixed = dict(zip(others, key if isinstance(key, tuple) else (key,), strict=False))
                findings.append(
                    {
                        "kind": "mono_violation",
                        "system": system,
                        "op_file": op_file,
                        "backend": t.backend,
                        "version": t.version,
                        "sweep_col": sweep,
                        "fixed_shape": {k: _jsonable(v) for k, v in fixed.items()},
                        "sweep_from": _jsonable(sw[i]),
                        "sweep_to": _jsonable(sw[i + 1]),
                        "latency_from": float(lat[i]),
                        "latency_to": float(lat[i + 1]),
                        "drop_ratio": float(drop[i]),
                        "kernel_source": str(series["kernel_source"].iloc[i + 1]),
                    }
                )
    return findings


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def cluster_suspects(anomalies: list[dict]) -> tuple[list[dict], list[dict]]:
    """Aggregate pair_outlier points into reviewable clusters.

    Every outlier names both sides as candidates. Points are grouped per
    (system, op_file, candidate side, categorical shape columns); a candidate
    that deviates against >= 2 distinct reference backends is corroborated —
    that side's data is the common factor and the likely suspect. Outliers
    none of whose candidates are corroborated are grouped per pair as
    'undetermined' (a two-backend mismatch cannot be attributed).

    Returns (corroborated_clusters, undetermined_clusters).
    """
    by_candidate: dict[tuple, list[tuple[dict, str]]] = defaultdict(list)
    for a in anomalies:
        if a["kind"] != "pair_outlier":
            continue
        cat_sig = "|".join(f"{k}={v}" for k, v in a["cat_shape"].items())
        for cand in a["candidates"]:
            by_candidate[(a["system"], a["op_file"], cand["side"], cat_sig)].append((a, cand["kind"]))

    def _other_side(point: dict, side: str) -> str:
        left, _, right = point["pair"].partition(" vs ")
        return right if left == side else left

    corroborated: list[dict] = []
    attributed_ids: set[int] = set()
    for (system, op_file, side, _cat_sig), entries in by_candidate.items():
        refs = sorted({_other_side(p, side) for p, _ in entries})
        if len(refs) < 2:
            continue
        points = [p for p, _ in entries]
        deviations = sorted(p["deviation"] for p in points)
        worst = max(points, key=lambda p: p["deviation"])
        kinds = Counter(kind for _, kind in entries)
        corroborated.append(
            {
                "system": system,
                "op_file": op_file,
                "suspect": side,
                "suspect_kind": kinds.most_common(1)[0][0],
                "categorical_shape": worst["cat_shape"],
                "points": len({p["shape_sig"] for p in points}),
                "reference_backends": refs,
                "deviation_min": deviations[0],
                "deviation_max": deviations[-1],
                "example_shape": worst["shape"],
                "example": {
                    "pair": worst["pair"],
                    "latency_a": worst["latency_a"],
                    "latency_b": worst["latency_b"],
                    "ratio": worst["ratio"],
                    "local_baseline_ratio": worst["local_baseline_ratio"],
                },
            }
        )
        attributed_ids.update(id(p) for p in points)
    corroborated.sort(key=lambda c: -c["points"])

    undetermined_by_key: dict[tuple, list[dict]] = defaultdict(list)
    for a in anomalies:
        if a["kind"] != "pair_outlier" or id(a) in attributed_ids:
            continue
        cat_sig = "|".join(f"{k}={v}" for k, v in a["cat_shape"].items())
        undetermined_by_key[(a["system"], a["op_file"], a["pair"], cat_sig)].append(a)
    undetermined: list[dict] = []
    for (system, op_file, pair, _cat_sig), points in undetermined_by_key.items():
        deviations = sorted(p["deviation"] for p in points)
        worst = max(points, key=lambda p: p["deviation"])
        undetermined.append(
            {
                "system": system,
                "op_file": op_file,
                "pair": pair,
                "categorical_shape": worst["cat_shape"],
                "points": len({p["shape_sig"] for p in points}),
                "deviation_min": deviations[0],
                "deviation_max": deviations[-1],
                "example_shape": worst["shape"],
                "example": {
                    "ratio": worst["ratio"],
                    "local_baseline_ratio": worst["local_baseline_ratio"],
                },
            }
        )
    undetermined.sort(key=lambda c: -c["points"])
    return corroborated, undetermined


def cluster_mono(anomalies: list[dict]) -> list[dict]:
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for a in anomalies:
        if a["kind"] != "mono_violation":
            continue
        by_key[(a["system"], a["op_file"], a["backend"], a["version"], a["kernel_source"], a["sweep_col"])].append(a)
    clusters = []
    for (system, op_file, backend, version, ks, sweep), points in by_key.items():
        worst = min(points, key=lambda p: p["drop_ratio"])
        clusters.append(
            {
                "system": system,
                "op_file": op_file,
                "backend": f"{backend}/{version}",
                "kernel_source": ks,
                "sweep_col": sweep,
                "points": len(points),
                "worst_drop": worst["drop_ratio"],
                "example": worst,
            }
        )
    clusters.sort(key=lambda c: -c["points"])
    return clusters


def run_checks(
    data_root: Path,
    systems: list[str] | None,
    backends: list[str] | None,
    op_files: list[str] | None,
    anomaly_factor: float,
    gap_factor: float,
    mono_tolerance: float,
    min_bucket_points: int,
    noise_floor: float,
) -> tuple[list[dict], list[dict]]:
    """Returns (anomalies, gaps) across the selected slice of the data tree."""
    # (system, op_file) -> backend -> [(version, path)]
    tables: dict[tuple[str, str], dict[str, list[tuple[str, Path]]]] = {}
    for system, backend, version, op_file, path in _iter_op_tables(data_root):
        if systems and system not in systems:
            continue
        if backends and backend not in backends:
            continue
        if op_files and op_file not in op_files:
            continue
        tables.setdefault((system, op_file), {}).setdefault(backend, []).append((version, path))

    anomalies: list[dict] = []
    gaps: list[dict] = []
    for (system, op_file), by_backend in sorted(tables.items()):
        # Latest version per backend.
        loaded: list[OpTable] = []
        for backend, versions in sorted(by_backend.items()):
            version, path = max(versions, key=lambda vp: _version_key(vp[0]))
            table, npos_by_ks = _load_op_table(path, backend, version)
            if npos_by_ks and op_file not in _DELTA_LATENCY_OP_FILES:
                anomalies.append(
                    {
                        "kind": "nonpositive_latency",
                        "system": system,
                        "op_file": op_file,
                        "backend": backend,
                        "version": version,
                        "rows": sum(npos_by_ks.values()),
                        "by_kernel_source": npos_by_ks,
                    }
                )
            if table is not None:
                loaded.append(table)
        if not loaded:
            continue

        for t in loaded:
            anomalies.extend(_check_monotonicity(system, op_file, t, mono_tolerance, noise_floor))

        if len(loaded) < 2:
            continue
        for a, b in itertools.combinations(loaded, 2):
            pair_anoms, pair_gaps = _check_pair(system, op_file, a, b, anomaly_factor, gap_factor, min_bucket_points)
            anomalies.extend(pair_anoms)
            gaps.extend(pair_gaps)
        logger.info("%s/%s: %d backends compared", system, op_file, len(loaded))
    return anomalies, gaps


def _fmt_shape(shape: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in shape.items() if v not in ("", None))


def render_markdown(anomalies: list[dict], gaps: list[dict], max_rows: int) -> str:
    lines: list[str] = ["# Cross-backend consistency report\n"]

    suspects, undetermined = cluster_suspects(anomalies)
    mono_clusters = cluster_mono(anomalies)
    regions = [a for a in anomalies if a["kind"] == "region_deviation"]
    nonpositive = [a for a in anomalies if a["kind"] == "nonpositive_latency"]
    mismatches = [g for g in gaps if g["kind"] == "schema_mismatch"]
    fw_gaps = [g for g in gaps if g["kind"] == "framework_gap"]
    summaries = [g for g in gaps if g["kind"] == "pair_summary"]
    n_outliers = sum(1 for a in anomalies if a["kind"] == "pair_outlier")
    n_mono = sum(1 for a in anomalies if a["kind"] == "mono_violation")

    lines.append("## Layer 1 — anomalies (suspected invalid data)\n")
    lines.append(f"- Nonpositive-latency rows: **{sum(a['rows'] for a in nonpositive)}** in {len(nonpositive)} tables")
    lines.append(
        f"- Cross-backend point outliers: **{n_outliers}** "
        f"({len(suspects)} corroborated suspect clusters, {len(undetermined)} undetermined clusters)"
    )
    lines.append(f"- Whole-region deviations (bucket median far from op median): **{len(regions)}**")
    lines.append(f"- Monotonicity violations: **{n_mono}** in **{len(mono_clusters)}** clusters\n")

    if nonpositive:
        lines.append("### Nonpositive-latency rows\n")
        lines.append("| system | op_file | backend | rows | by kernel_source |")
        lines.append("|---|---|---|---|---|")
        for a in sorted(nonpositive, key=lambda x: -x["rows"])[:max_rows]:
            by_ks = ", ".join(f"{k}: {v}" for k, v in a["by_kernel_source"].items())
            lines.append(f"| {a['system']} | {a['op_file']} | {a['backend']}/{a['version']} | {a['rows']} | {by_ks} |")
        lines.append("")

    if suspects:
        lines.append(f"### Corroborated suspects (top {min(max_rows, len(suspects))})\n")
        lines.append(
            "The same side deviates against two or more reference backends — that side's "
            "data is the common factor. `suspect_kind` says how it deviates from the local "
            "baseline (too slow or too fast; anomalously FAST usually means a broken measurement).\n"
        )
        lines.append("| system | op_file | suspect | kind | dtype cols | points | refs | deviation | worst example |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for c in suspects[:max_rows]:
            ex = c["example"]
            lines.append(
                f"| {c['system']} | {c['op_file']} | **{c['suspect']}** | {c['suspect_kind']} | "
                f"{_fmt_shape(c['categorical_shape'])} | {c['points']} | {', '.join(c['reference_backends'])} | "
                f"{c['deviation_min']:.1f}-{c['deviation_max']:.1f}x | "
                f"{_fmt_shape(c['example_shape'])}: ratio {ex['ratio']:.2f} "
                f"vs baseline {ex['local_baseline_ratio']:.2f} |"
            )
        lines.append("")

    if undetermined:
        lines.append(f"### Undetermined pair mismatches (top {min(max_rows, len(undetermined))})\n")
        lines.append(
            "Only one reference backend exists for these points, so the deviating side "
            "cannot be attributed — one of the two tables is off.\n"
        )
        lines.append("| system | op_file | pair | dtype cols | points | deviation | worst example |")
        lines.append("|---|---|---|---|---|---|---|")
        for c in undetermined[:max_rows]:
            ex = c["example"]
            lines.append(
                f"| {c['system']} | {c['op_file']} | {c['pair']} | {_fmt_shape(c['categorical_shape'])} | "
                f"{c['points']} | {c['deviation_min']:.1f}-{c['deviation_max']:.1f}x | "
                f"{_fmt_shape(c['example_shape'])}: ratio {ex['ratio']:.2f} "
                f"vs baseline {ex['local_baseline_ratio']:.2f} |"
            )
        lines.append("")

    if regions:
        lines.append(f"### Whole-region deviations (top {min(max_rows, len(regions))})\n")
        lines.append("| system | op_file | pair | bucket | median ratio | op median | rel deviation | points |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in sorted(regions, key=lambda x: -x["rel_deviation"])[:max_rows]:
            lines.append(
                f"| {r['system']} | {r['op_file']} | {r['pair']} | `{r['bucket']}` | "
                f"{r['median_ratio']:.2f} | {r['op_median_ratio']:.2f} | {r['rel_deviation']:.1f}x | {r['points']} |"
            )
        lines.append("")

    if mono_clusters:
        lines.append(f"### Monotonicity violation clusters (top {min(max_rows, len(mono_clusters))})\n")
        lines.append("| system | op_file | backend | kernel_source | sweep | points | worst drop | worst example |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for c in mono_clusters[:max_rows]:
            ex = c["example"]
            lines.append(
                f"| {c['system']} | {c['op_file']} | {c['backend']} | `{c['kernel_source']}` | "
                f"{c['sweep_col']} | {c['points']} | {c['worst_drop']:.2f} | "
                f"{_fmt_shape(ex['fixed_shape'])}: {ex['sweep_col']} {ex['sweep_from']}→{ex['sweep_to']}, "
                f"lat {ex['latency_from']:.4g}→{ex['latency_to']:.4g} |"
            )
        lines.append("")

    if mismatches:
        lines.append("### Skipped comparisons (shape schema mismatch)\n")
        lines.append("| system | op_file | pair | shape_cols_a | shape_cols_b |")
        lines.append("|---|---|---|---|---|")
        for m in mismatches[:max_rows]:
            lines.append(
                f"| {m['system']} | {m['op_file']} | {m['pair']} | "
                f"{', '.join(m['shape_cols_a'])} | {', '.join(m['shape_cols_b'])} |"
            )
        lines.append("")

    lines.append("## Layer 2 — framework gaps (informational)\n")
    lines.append(
        "Ratios are `latency_b / latency_a` for the pair `b vs a`; a ratio < 1 means "
        "backend b is faster. Different kernel_source values on the two sides mean the "
        "gap likely reflects a kernel implementation difference, not bad data.\n"
    )
    if summaries:
        lines.append("### Pair summaries\n")
        lines.append(
            "| system | op_file | pair | points | median | p05 | p95 | gap/region/total buckets | "
            "kernels_a | kernels_b |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for s in sorted(summaries, key=lambda x: (x["system"], x["op_file"], x["pair"])):
            lines.append(
                f"| {s['system']} | {s['op_file']} | {s['pair']} | {s['joined_points']} | "
                f"{s['median_ratio']:.2f} | {s['p05_ratio']:.2f} | {s['p95_ratio']:.2f} | "
                f"{s['gap_buckets']}/{s['region_deviation_buckets']}/{s['total_buckets']} | "
                f"`{', '.join(s['kernel_sources_a'])}` | `{', '.join(s['kernel_sources_b'])}` |"
            )
        lines.append("")

    if fw_gaps:
        lines.append(f"### Significant gap buckets (top {min(max_rows, len(fw_gaps))} by |median ratio|)\n")
        lines.append("| system | op_file | pair | bucket | median ratio | points | ks_a → ks_b |")
        lines.append("|---|---|---|---|---|---|---|")
        for g in sorted(fw_gaps, key=lambda x: -abs(math.log(x["median_ratio"])))[:max_rows]:
            lines.append(
                f"| {g['system']} | {g['op_file']} | {g['pair']} | `{g['bucket']}` | "
                f"{g['median_ratio']:.2f} | {g['points']} | "
                f"`{g['kernel_source_a']}` → `{g['kernel_source_b']}` |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("aic-core/src/aiconfigurator_core/systems/data"),
        help="Root of the systems/data tree.",
    )
    parser.add_argument("--systems", nargs="*", default=None, help="Restrict to these systems.")
    parser.add_argument("--backends", nargs="*", default=None, help="Restrict to these backends.")
    parser.add_argument("--op-files", nargs="*", default=None, help="Restrict to these op file basenames.")
    parser.add_argument(
        "--anomaly-factor",
        type=float,
        default=3.0,
        help="Layer 1: flag points deviating from the local baseline by more than this factor.",
    )
    parser.add_argument(
        "--gap-factor",
        type=float,
        default=1.5,
        help="Layer 2: report buckets whose median cross-backend ratio exceeds this factor.",
    )
    parser.add_argument(
        "--mono-tolerance",
        type=float,
        default=0.7,
        help="Layer 1: flag latency drops below this ratio while a sweep dimension grows.",
    )
    parser.add_argument(
        "--noise-floor",
        type=float,
        default=0.03,
        help="Latency in ms below which monotonicity noise is ignored (auto-scaled for us-unit tables).",
    )
    parser.add_argument(
        "--min-bucket-points",
        type=int,
        default=5,
        help="Minimum joined points for a bucket/series to serve as a local baseline.",
    )
    parser.add_argument("--max-report-rows", type=int, default=50, help="Row cap per markdown table.")
    parser.add_argument("--out-md", type=Path, default=None, help="Write the Markdown report.")
    parser.add_argument("--out-json", type=Path, default=None, help="Write all findings as JSON.")
    parser.add_argument(
        "--fail-on-anomalies",
        action="store_true",
        help="Exit nonzero when Layer 1 anomalies are found (CI gate mode).",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    if args.anomaly_factor <= 1.0:
        parser.error("--anomaly-factor must be > 1")
    if args.gap_factor <= 1.0:
        parser.error("--gap-factor must be > 1")
    if args.gap_factor > args.anomaly_factor:
        parser.error("--gap-factor must be <= --anomaly-factor (the gap band sits below the anomaly threshold)")
    if not 0.0 < args.mono_tolerance <= 1.0:
        parser.error("--mono-tolerance must be in (0, 1]")

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")

    anomalies, gaps = run_checks(
        data_root=args.data_root,
        systems=args.systems,
        backends=args.backends,
        op_files=args.op_files,
        anomaly_factor=args.anomaly_factor,
        gap_factor=args.gap_factor,
        mono_tolerance=args.mono_tolerance,
        min_bucket_points=args.min_bucket_points,
        noise_floor=args.noise_floor,
    )

    suspects, undetermined = cluster_suspects(anomalies)
    mono_clusters = cluster_mono(anomalies)
    regions = [a for a in anomalies if a["kind"] == "region_deviation"]
    nonpositive = [a for a in anomalies if a["kind"] == "nonpositive_latency"]

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(
                {
                    "anomalies": anomalies,
                    "gaps": gaps,
                    "suspect_clusters": suspects,
                    "undetermined_clusters": undetermined,
                    "mono_clusters": mono_clusters,
                },
                indent=2,
            )
        )
        logger.info("wrote %s", args.out_json)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_markdown(anomalies, gaps, args.max_report_rows))
        logger.info("wrote %s", args.out_md)

    n_outliers = sum(1 for a in anomalies if a["kind"] == "pair_outlier")
    n_mono = sum(1 for a in anomalies if a["kind"] == "mono_violation")
    n_gap_buckets = sum(1 for g in gaps if g["kind"] == "framework_gap")
    n_mismatch = sum(1 for g in gaps if g["kind"] == "schema_mismatch")
    print(
        f"\nLayer 1: {sum(a['rows'] for a in nonpositive)} nonpositive-latency rows, "
        f"{n_outliers} point outliers ({len(suspects)} corroborated suspects, "
        f"{len(undetermined)} undetermined), {len(regions)} region deviations, "
        f"{n_mono} mono violations in {len(mono_clusters)} clusters"
    )
    print(
        f"Layer 2: {n_gap_buckets} significant gap buckets across "
        f"{sum(1 for g in gaps if g['kind'] == 'pair_summary')} backend pairs"
        + (f"; {n_mismatch} pairs skipped (schema mismatch)" if n_mismatch else "")
    )
    for c in suspects[:10]:
        print(
            f"  SUSPECT? {c['system']}/{c['op_file']}: {c['suspect']} ({c['suspect_kind']}) "
            f"{_fmt_shape(c['categorical_shape'])} — {c['points']} points, "
            f"deviation {c['deviation_min']:.1f}-{c['deviation_max']:.1f}x, "
            f"refs: {', '.join(c['reference_backends'])}"
        )

    if args.fail_on_anomalies and (nonpositive or n_outliers or n_mono or regions):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

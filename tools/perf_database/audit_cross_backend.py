# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Two-layer cross-backend validity audit of perf-database perf tables.

For every (system, op_file) present in >= 2 backends, join the latest version
of each backend on the shape key and emit findings on two layers:

  Layer 1 — ANOMALY (data-validity, actionable):
    - `pair_outlier`     : a shape point whose cross-backend latency ratio
                           deviates from its *local baseline* by more than
                           --anomaly-factor (default 3x). The baseline is
                           hierarchical — shape bucket, then series (non-sweep
                           shape columns), then op-level median — so
                           legitimate shape-dependent framework gaps are
                           absorbed and what's left is likely a bad
                           measurement in one of the two tables.
    - `region_deviation` : an entire shape bucket whose *median* ratio exceeds
                           --anomaly-factor. Not a stray point — a whole
                           region of one backend's table deviates (bad sweep,
                           degenerate kernel path, wrong unit).
    - `mono_violation`   : within one backend, latency drops by more than
                           (1 - --mono-tolerance) while a sweep dimension
                           (batch_size / isl / m / step) grows with all other
                           shape columns fixed. Points below --noise-floor
                           latency are exempt (timer noise regime).

    Point anomalies are clustered by (pair, kernel_source pair, categorical
    shape columns) for reporting, and clusters where the same slow side
    deviates against >= 2 distinct reference backends are marked as a
    corroborated culprit ("vllm/0.24.0 is the common factor").

  Layer 2 — GAP (framework-difference, informational):
    - per shape-bucket median latency ratios between backend pairs, annotated
      with each side's kernel_source (e.g. fa3 vs flash_attention vs triton).
      Buckets whose median ratio exceeds --gap-factor (default 1.5x) but stay
      below --anomaly-factor are surfaced as significant-but-plausible
      framework differences, not data errors.

Shape-key convention follows audit_kernel_source.py: every column that is not
a meta column ({framework, version, device, op_name, kernel_source}) and not
the latency column is part of the shape key. Sweep columns get log2-bucketed
to form the local-baseline bucket key.

Usage:
    python3 tools/perf_database/audit_cross_backend.py \\
        --data-root aic-core/src/aiconfigurator_core/systems/data \\
        --systems h200_sxm \\
        --out-md   $TMPDIR/cross-backend-audit.md \\
        --out-json $TMPDIR/cross-backend-audit.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns that never participate in the shape key (same set as audit_kernel_source.py).
_META_COLUMNS = {"framework", "version", "device", "op_name", "kernel_source"}

# Latency-like columns. The first one found in the header is used as the metric.
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

# Legacy top-level backend dirs (see audit_kernel_source.py for the sync note).
_LEGACY_BACKEND_DIRS = {"trtllm", "sglang", "vllm"}


def _version_key(version: str) -> tuple:
    """Sortable key for backend version strings like '1.3.0rc10' or '0.5.6.post2'."""
    parts = re.findall(r"(\d+|[a-zA-Z]+)", version)
    key = []
    for p in parts:
        if p.isdigit():
            key.append((1, int(p)))
        else:
            # Pre-release tags (rc, a, b) sort before the release; post after.
            key.append((0 if p.lower() in ("rc", "a", "b", "alpha", "beta", "dev") else 2, p.lower()))
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

    @property
    def label(self) -> str:
        return f"{self.backend}/{self.version}"


def _load_op_table(path: Path, backend: str, version: str) -> OpTable | None:
    df = _read_table(path)
    latency_col = _pick_latency_column(df.columns)
    if latency_col is None:
        logger.warning("No latency column in %s; skipping", path)
        return None
    shape_cols = [c for c in df.columns if c not in _META_COLUMNS and c != latency_col]
    df = df[df[latency_col] > 0]
    if df.empty or not shape_cols:
        return None
    df = df.rename(columns={latency_col: "latency"})
    if "kernel_source" not in df.columns:
        df["kernel_source"] = "<unknown>"
    # Min latency per shape key, keeping the winning kernel_source.
    idx = df.groupby(shape_cols, dropna=False)["latency"].idxmin()
    reduced = df.loc[idx, [*shape_cols, "latency", "kernel_source"]].reset_index(drop=True)
    return OpTable(
        backend=backend,
        version=version,
        shape_cols=shape_cols,
        frame=reduced,
        kernel_sources=sorted(df["kernel_source"].astype(str).unique()),
    )


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


def _audit_pair(
    system: str,
    op_file: str,
    a: OpTable,
    b: OpTable,
    anomaly_factor: float,
    gap_factor: float,
    min_bucket_points: int,
) -> tuple[list[dict], list[dict]]:
    """Compare two backends' tables. Returns (anomalies, gaps)."""
    shape_cols = [c for c in a.shape_cols if c in b.shape_cols]
    if not shape_cols:
        return [], []
    merged = a.frame.merge(b.frame, on=shape_cols, suffixes=("_a", "_b"))
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

    pair = f"{b.label} vs {a.label}"
    cat_cols = _categorical_cols(merged, shape_cols)
    log_anomaly = math.log(anomaly_factor)
    log_gap = math.log(gap_factor)

    # ---- Layer 1: per-point outliers vs. local baseline --------------------
    anomalies: list[dict] = []
    flagged = merged[merged["deviation"] > anomaly_factor]
    for _, row in flagged.iterrows():
        slow_is_b = row["log_ratio"] > row["baseline"]
        anomalies.append(
            {
                "kind": "pair_outlier",
                "system": system,
                "op_file": op_file,
                "pair": pair,
                "shape": {c: _jsonable(row[c]) for c in shape_cols},
                "latency_a": float(row["latency_a"]),
                "latency_b": float(row["latency_b"]),
                "kernel_source_a": str(row["kernel_source_a"]),
                "kernel_source_b": str(row["kernel_source_b"]),
                "ratio": float(np.exp(row["log_ratio"])),
                "local_baseline_ratio": float(np.exp(row["baseline"])),
                "deviation": float(row["deviation"]),
                "slow_side": b.label if slow_is_b else a.label,
                "reference_side": a.label if slow_is_b else b.label,
                "slow_kernel_source": str(row["kernel_source_b"] if slow_is_b else row["kernel_source_a"]),
                "cluster_key": "|".join(
                    [op_file, b.label if slow_is_b else a.label] + [f"{c}={row[c]}" for c in cat_cols]
                ),
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

    regions = grp[grp["median_log_ratio"].abs() > log_anomaly]
    for bucket, row in regions.sort_values("median_log_ratio", key=lambda s: s.abs(), ascending=False).iterrows():
        slow_is_b = row["median_log_ratio"] > 0
        anomalies.append(
            {
                "kind": "region_deviation",
                "system": system,
                "op_file": op_file,
                "pair": pair,
                "bucket": bucket,
                "median_ratio": float(np.exp(row["median_log_ratio"])),
                "points": int(row["points"]),
                "kernel_source_a": row["ks_a"],
                "kernel_source_b": row["ks_b"],
                "slow_side": b.label if slow_is_b else a.label,
                "reference_side": a.label if slow_is_b else b.label,
                "slow_kernel_source": row["ks_b"] if slow_is_b else row["ks_a"],
            }
        )

    sig = grp[(grp["median_log_ratio"].abs() > log_gap) & (grp["median_log_ratio"].abs() <= log_anomaly)]
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
        }
    )
    return anomalies, gaps


def _audit_monotonicity(system: str, op_file: str, t: OpTable, mono_tolerance: float, noise_floor: float) -> list[dict]:
    """Flag latency drops > (1 - mono_tolerance) while one sweep column grows
    and all other shape columns stay fixed. Points below noise_floor latency
    are exempt — sub-noise timings jitter without meaning."""
    findings: list[dict] = []
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
            bad = np.nonzero((drop < mono_tolerance) & (lat[:-1] >= noise_floor))[0]
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


def cluster_pair_outliers(anomalies: list[dict]) -> list[dict]:
    """Aggregate pair_outlier points into reviewable clusters keyed by
    (op_file, slow side, categorical shape columns). A cluster corroborated by
    >= 2 distinct reference backends marks its slow side as the likely culprit."""
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for a in anomalies:
        if a["kind"] != "pair_outlier":
            continue
        by_key[(a["system"], a["cluster_key"])].append(a)

    clusters: list[dict] = []
    for (system, key), points in by_key.items():
        refs = sorted({p["reference_side"] for p in points})
        ratios = sorted(p["ratio"] for p in points)
        worst = max(points, key=lambda p: p["deviation"])
        clusters.append(
            {
                "system": system,
                "op_file": points[0]["op_file"],
                "slow_side": points[0]["slow_side"],
                "slow_kernel_source": ",".join(sorted({p["slow_kernel_source"] for p in points})),
                "categorical_shape": {k: v for k, v in worst["shape"].items() if isinstance(v, str)},
                "points": len(points),
                "reference_backends": refs,
                "corroborated": len(refs) >= 2,
                "ratio_min": ratios[0],
                "ratio_max": ratios[-1],
                "worst_deviation": worst["deviation"],
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
    clusters.sort(key=lambda c: (not c["corroborated"], -c["points"]))
    return clusters


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


def audit(
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
            table = _load_op_table(path, backend, version)
            if table is not None:
                loaded.append(table)
        if not loaded:
            continue

        for t in loaded:
            anomalies.extend(_audit_monotonicity(system, op_file, t, mono_tolerance, noise_floor))

        if len(loaded) < 2:
            continue
        for a, b in itertools.combinations(loaded, 2):
            pair_anoms, pair_gaps = _audit_pair(system, op_file, a, b, anomaly_factor, gap_factor, min_bucket_points)
            anomalies.extend(pair_anoms)
            gaps.extend(pair_gaps)
        logger.info("%s/%s: %d backends compared", system, op_file, len(loaded))
    return anomalies, gaps


def _fmt_shape(shape: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in shape.items() if v not in ("", None))


def render_markdown(anomalies: list[dict], gaps: list[dict], max_rows: int) -> str:
    lines: list[str] = ["# Cross-backend validity audit\n"]

    outlier_clusters = cluster_pair_outliers(anomalies)
    mono_clusters = cluster_mono(anomalies)
    regions = [a for a in anomalies if a["kind"] == "region_deviation"]
    fw_gaps = [g for g in gaps if g["kind"] == "framework_gap"]
    summaries = [g for g in gaps if g["kind"] == "pair_summary"]
    n_outliers = sum(1 for a in anomalies if a["kind"] == "pair_outlier")
    n_mono = sum(1 for a in anomalies if a["kind"] == "mono_violation")

    lines.append("## Layer 1 — anomalies (suspected invalid data)\n")
    lines.append(f"- Cross-backend point outliers: **{n_outliers}** in **{len(outlier_clusters)}** clusters")
    lines.append(f"- Whole-region deviations (bucket median beyond anomaly factor): **{len(regions)}**")
    lines.append(f"- Monotonicity violations: **{n_mono}** in **{len(mono_clusters)}** clusters\n")

    if outlier_clusters:
        lines.append(f"### Outlier clusters (top {min(max_rows, len(outlier_clusters))}; corroborated first)\n")
        lines.append(
            "A cluster groups outlier points by (op, slow side, dtype-like columns). "
            "`corroborated=yes` means the same slow side deviates against two or more "
            "reference backends — that side's data is the likely culprit.\n"
        )
        lines.append(
            "| system | op_file | slow side (culprit?) | slow kernel | dtype cols | points | refs | "
            "corroborated | ratio range | worst example |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for c in outlier_clusters[:max_rows]:
            ex = c["example"]
            lines.append(
                f"| {c['system']} | {c['op_file']} | **{c['slow_side']}** | `{c['slow_kernel_source']}` | "
                f"{_fmt_shape(c['categorical_shape'])} | {c['points']} | {', '.join(c['reference_backends'])} | "
                f"{'**yes**' if c['corroborated'] else 'no'} | "
                f"{c['ratio_min']:.2f}-{c['ratio_max']:.2f} | "
                f"{_fmt_shape(c['example_shape'])}: {ex['ratio']:.1f}x vs baseline {ex['local_baseline_ratio']:.2f} |"
            )
        lines.append("")

    if regions:
        lines.append(f"### Whole-region deviations (top {min(max_rows, len(regions))})\n")
        lines.append("| system | op_file | pair | bucket | median ratio | points | slow side | slow kernel |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in sorted(regions, key=lambda x: -abs(math.log(x["median_ratio"])))[:max_rows]:
            lines.append(
                f"| {r['system']} | {r['op_file']} | {r['pair']} | `{r['bucket']}` | "
                f"{r['median_ratio']:.2f} | {r['points']} | **{r['slow_side']}** | `{r['slow_kernel_source']}` |"
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
        help="Latency (same unit as the table, typically ms) below which monotonicity noise is ignored.",
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

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")

    anomalies, gaps = audit(
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

    outlier_clusters = cluster_pair_outliers(anomalies)
    mono_clusters = cluster_mono(anomalies)
    regions = [a for a in anomalies if a["kind"] == "region_deviation"]

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(
                {
                    "anomalies": anomalies,
                    "gaps": gaps,
                    "outlier_clusters": outlier_clusters,
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
    corroborated = [c for c in outlier_clusters if c["corroborated"]]
    print(
        f"\nLayer 1: {n_outliers} point outliers in {len(outlier_clusters)} clusters "
        f"({len(corroborated)} corroborated), {len(regions)} region deviations, "
        f"{n_mono} mono violations in {len(mono_clusters)} clusters"
    )
    print(
        f"Layer 2: {n_gap_buckets} significant gap buckets across "
        f"{sum(1 for g in gaps if g['kind'] == 'pair_summary')} backend pairs"
    )
    for c in corroborated[:10]:
        print(
            f"  CULPRIT? {c['system']}/{c['op_file']}: {c['slow_side']} "
            f"[{c['slow_kernel_source']}] {_fmt_shape(c['categorical_shape'])} — "
            f"{c['points']} points, ratio {c['ratio_min']:.1f}-{c['ratio_max']:.1f}x, "
            f"refs: {', '.join(c['reference_backends'])}"
        )

    if args.fail_on_anomalies and (n_outliers or n_mono or regions):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

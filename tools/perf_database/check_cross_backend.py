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
    - `spike_violation`  : within one backend, a point higher than BOTH sweep
                           neighbors by more than --spike-factor — a bad
                           measurement (jitter, preemption, missing warmup)
                           needing no reference framework.
    - `below_sol`        : gemm latency below the speed-of-light bound
                           computed from the system's gpu spec (peak flops /
                           HBM bandwidth; the bandwidth term only applies to
                           working sets beyond L2 hot-cache reach).
                           Physically impossible — definitive without any
                           reference framework.
    - `machine_op_deviation`: hint (not gated) — a (system, backend, op)
                           whose latency level is off relative to how that
                           system compares to its peers on its other ops
                           (hardware factor removed); points at a
                           collection-environment problem on that machine.

  Layer 2 — GAP (framework-difference, informational):
    - `systematic_offset`: a backend pair whose op-level median ratio deviates
      beyond --offset-factor (default 1.15x) in the SAME direction on every
      covered system (>= --min-offset-systems). A shape-local kernel weakness
      moves with shape and hardware; a uniform multiplier reproduced across
      systems is the fingerprint of a framework-level collection or
      configuration difference (eager mode, missing torch.compile, disabled
      CUDA graphs) rather than of the shapes.
    - per-pair summaries: joined points, median/p05/p95 ratio, and both
      sides' kernel_source lists (e.g. fa3 vs flash_attention vs triton) so
      implementation differences are distinguishable from bad data.
    - `kernel_choice_cost`: within one framework's table, kernels measured on
      the same shapes as a faster alternative — the price of that backend
      selection.
    - attribution: offsets and suspects are labeled `kernel_choice` (the two
      sides run different kernel families, normalized via
      collector/kernel_source_backends.yaml) or `harness_config` (same kernel
      family — the gap comes from the wrapper/config around it).

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


_KERNEL_COST_FACTOR = 1.5


def _load_op_table(path: Path, backend: str, version: str) -> tuple[OpTable | None, dict[str, int], list[dict]]:
    """Returns (table, nonpositive-latency row counts by kernel_source,
    within-framework kernel-choice costs).

    Nonpositive rows are excluded from the returned frame but reported so the
    caller can emit them as Layer-1 anomalies instead of silently cleaning
    them up. Kernel-choice costs quantify, for tables that measured several
    kernel_sources on the same shapes, how much slower each kernel's median is
    than the per-shape best — the price of picking that backend.
    """
    df = _read_table(path)
    latency_col = _pick_latency_column(df.columns)
    if latency_col is None:
        logger.warning("No latency column in %s; skipping", path)
        return None, {}, []
    if "kernel_source" not in df.columns:
        df["kernel_source"] = "<unknown>"
    nonpositive = df[df[latency_col].notna() & (df[latency_col] <= 0)]
    npos_by_ks = {str(k): int(v) for k, v in nonpositive["kernel_source"].value_counts().items()}
    shape_cols = [c for c in df.columns if c not in _META_COLUMNS and c not in _LATENCY_COLUMNS_PRIORITY]
    df = df[df[latency_col] > 0]
    if df.empty or not shape_cols:
        return None, npos_by_ks, []
    df = df.rename(columns={latency_col: "latency"})
    kernel_costs: list[dict] = []
    if df["kernel_source"].nunique() > 1:
        penalty = df["latency"] / df.groupby(shape_cols, dropna=False)["latency"].transform("min")
        agg = penalty.groupby(df["kernel_source"].astype(str)).agg(["median", "max", "size"])
        for ks, row in agg.iterrows():
            if row["median"] > _KERNEL_COST_FACTOR:
                kernel_costs.append(
                    {
                        "kernel_source": ks,
                        "median_penalty": float(row["median"]),
                        "max_penalty": float(row["max"]),
                        "rows": int(row["size"]),
                    }
                )
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
    return table, npos_by_ks, kernel_costs


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

    # Pair-level headline (always emitted).
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
            "region_deviation_buckets": len(regions),
            "total_buckets": len(grp),
            "kernel_sources_a": a.kernel_sources,
            "kernel_sources_b": b.kernel_sources,
            "align_notes": align_notes,
        }
    )
    return anomalies, gaps


def _check_curve(
    system: str, op_file: str, t: OpTable, mono_tolerance: float, spike_factor: float, noise_floor: float
) -> list[dict]:
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
            fixed = dict(zip(others, key if isinstance(key, tuple) else (key,), strict=False))
            fixed_shape = {k: _jsonable(v) for k, v in fixed.items()}
            drop = lat[1:] / lat[:-1]
            bad = np.nonzero((drop < mono_tolerance) & (lat[:-1] >= floor))[0]
            for i in bad:
                findings.append(
                    {
                        "kind": "mono_violation",
                        "system": system,
                        "op_file": op_file,
                        "backend": t.backend,
                        "version": t.version,
                        "sweep_col": sweep,
                        "fixed_shape": fixed_shape,
                        "sweep_from": _jsonable(sw[i]),
                        "sweep_to": _jsonable(sw[i + 1]),
                        "latency_from": float(lat[i]),
                        "latency_to": float(lat[i + 1]),
                        "ratio": float(drop[i]),
                        "kernel_source": str(series["kernel_source"].iloc[i + 1]),
                    }
                )
            # Upward spikes: a point higher than BOTH neighbors along the
            # sweep curve is a bad measurement (jitter, preemption, missing
            # warmup) — no reference framework needed. Latency should vary
            # smoothly (stair-steps included) with the sweep dimension.
            if len(lat) >= 3:
                neighbors = np.maximum(lat[:-2], lat[2:])
                spikes = np.nonzero((lat[1:-1] > neighbors * spike_factor) & (neighbors >= floor))[0]
                for i in spikes:
                    findings.append(
                        {
                            "kind": "spike_violation",
                            "system": system,
                            "op_file": op_file,
                            "backend": t.backend,
                            "version": t.version,
                            "sweep_col": sweep,
                            "fixed_shape": fixed_shape,
                            "sweep_at": _jsonable(sw[i + 1]),
                            "latency": float(lat[i + 1]),
                            "neighbor_latency": float(neighbors[i]),
                            "ratio": float(lat[i + 1] / neighbors[i]),
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


# Measured latency below this fraction of the speed-of-light bound is flagged
# as physically impossible (small margin for spec rounding).
_SOL_MARGIN = 0.98

# The HBM-bandwidth term of the bound only applies when the working set is
# safely beyond L2 capacity: microbenchmarks re-run the same tensors, so a
# working set that fits in L2 (up to ~126MB on Blackwell) can legitimately
# beat HBM speed-of-light on hot cache. Below this size only the compute
# bound (unbeatable at any cache level) is enforced.
_SOL_MIN_WORKING_SET_BYTES = 512e6

# gemm_dtype -> (peak-flops spec key, activation bytes/elt, weight bytes/elt).
# int8_wo computes in bf16 (weight-only quant); nvfp4 needs fp4_tc_flops
# (absent on pre-Blackwell specs -> dtype skipped there).
_GEMM_SOL_DTYPES = {
    "bfloat16": ("bfloat16_tc_flops", 2.0, 2.0),
    "fp8": ("fp8_tc_flops", 1.0, 1.0),
    "fp8_block": ("fp8_tc_flops", 1.0, 1.0),
    "int8_wo": ("bfloat16_tc_flops", 2.0, 1.0),
    "nvfp4": ("fp4_tc_flops", 0.5, 0.5),
}

_SPEC_CACHE: dict[str, dict | None] = {}


def _load_gpu_spec(spec_root: Path, system: str) -> dict | None:
    if system not in _SPEC_CACHE:
        try:
            import yaml

            _SPEC_CACHE[system] = yaml.safe_load((spec_root / f"{system}.yaml").read_text()).get("gpu")
        except (OSError, AttributeError):
            logger.info("no gpu spec for %s under %s; SOL check skipped", system, spec_root)
            _SPEC_CACHE[system] = None
    return _SPEC_CACHE[system]


def _check_gemm_sol(system: str, op_file: str, t: OpTable, spec: dict) -> tuple[list[dict], list[dict]]:
    """Speed-of-light lower bound for gemm tables.

    SOL = max(compute time at theoretical peak flops, memory time at peak
    bandwidth) — a strict physical bound needing no reference framework.
    Measured latency below it is definitively invalid (Layer 1 `below_sol`);
    the per-dtype efficiency distribution (SOL / measured) doubles as an
    environment-health signal per system (Layer 2 `sol_efficiency`).
    """
    df = t.frame
    if not {"m", "n", "k", "gemm_dtype"}.issubset(df.columns):
        return [], []
    mem_bw = spec.get("mem_bw")
    if not mem_bw:
        return [], []
    anomalies: list[dict] = []
    efficiencies: list[dict] = []
    for dtype, sub in df.groupby("gemm_dtype"):
        entry = _GEMM_SOL_DTYPES.get(str(dtype))
        if entry is None:
            continue
        flops_key, a_bytes, w_bytes = entry
        flops = spec.get(flops_key)
        if not flops:
            continue
        m = sub["m"].to_numpy(dtype=float)
        n = sub["n"].to_numpy(dtype=float)
        k = sub["k"].to_numpy(dtype=float)
        compute_s = 2.0 * m * n * k / flops
        working_set = m * k * a_bytes + n * k * w_bytes + m * n * a_bytes
        memory_s = np.where(working_set >= _SOL_MIN_WORKING_SET_BYTES, working_set / mem_bw, 0.0)
        sol = np.maximum(compute_s, memory_s) * 1000.0 * t.noise_scale  # table units
        lat = sub["latency"].to_numpy(dtype=float)
        ratio = lat / sol
        below = ratio < _SOL_MARGIN
        if below.any():
            worst = int(np.argmin(ratio))
            anomalies.append(
                {
                    "kind": "below_sol",
                    "system": system,
                    "op_file": op_file,
                    "backend": t.backend,
                    "version": t.version,
                    "gemm_dtype": str(dtype),
                    "points": int(below.sum()),
                    "worst_fraction_of_sol": float(ratio[worst]),
                    "example_shape": {"m": int(m[worst]), "n": int(n[worst]), "k": int(k[worst])},
                    "example_latency": float(lat[worst]),
                    "example_sol": float(sol[worst]),
                }
            )
        efficiencies.append(
            {
                "kind": "sol_efficiency",
                "system": system,
                "op_file": op_file,
                "backend": t.backend,
                "version": t.version,
                "gemm_dtype": str(dtype),
                "points": len(sub),
                "median_efficiency": float(np.median(1.0 / ratio)),
                "p95_efficiency": float(np.quantile(1.0 / ratio, 0.95)),
            }
        )
    return anomalies, efficiencies


def load_kernel_map(path: Path) -> dict[tuple[str, str], str]:
    """Load collector/kernel_source_backends.yaml: (framework, kernel_source)
    -> runtime backend name (fa3, triton, flashinfer, ...). Returns {} when
    the file is unavailable — attribution then falls back to raw label names."""
    try:
        import yaml

        doc = yaml.safe_load(path.read_text())
        return {(m["framework"], m["kernel_source"]): m["backend"] for m in doc.get("mappings", [])}
    except (OSError, KeyError, TypeError) as exc:
        logger.warning("kernel map %s unavailable (%s); attribution uses raw kernel_source names", path, exc)
        return {}


def _normalize_kernels(kmap: dict[tuple[str, str], str], framework: str, kernel_sources) -> set[str]:
    out: set[str] = set()
    for entry in kernel_sources:
        for ks in str(entry).split(","):
            ks = ks.strip()
            if ks:
                out.add(kmap.get((framework, ks), ks.lower()))
    return out


# Normalized backend names that don't identify a kernel (framework-internal
# dispatch, placeholders) — comparisons against them can't be attributed.
_OPAQUE_KERNELS = {"trtllm_internal", "default", "unverified"}


def _attribution(slow_kernels: set[str], reference_kernels: set[str]) -> str:
    """Same normalized kernel family on both sides -> the gap comes from the
    harness/config around the kernel; disjoint families -> the frameworks run
    different kernels, pointing at kernel-backend selection. Sides made up of
    opaque labels only cannot be attributed."""
    if not slow_kernels or not reference_kernels:
        return "unknown"
    if slow_kernels <= _OPAQUE_KERNELS or reference_kernels <= _OPAQUE_KERNELS:
        return "unknown"
    return "harness_config" if slow_kernels & reference_kernels else "kernel_choice"


def cluster_suspects(anomalies: list[dict], kmap: dict[tuple[str, str], str]) -> tuple[list[dict], list[dict]]:
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
        slow_ks: set[str] = set()
        ref_ks: set[str] = set()
        for point in points:
            b_label, _, a_label = point["pair"].partition(" vs ")
            if side == b_label:
                slow_ks |= _normalize_kernels(kmap, side.split("/")[0], [point["kernel_source_b"]])
                ref_ks |= _normalize_kernels(kmap, a_label.split("/")[0], [point["kernel_source_a"]])
            else:
                slow_ks |= _normalize_kernels(kmap, side.split("/")[0], [point["kernel_source_a"]])
                ref_ks |= _normalize_kernels(kmap, b_label.split("/")[0], [point["kernel_source_b"]])
        corroborated.append(
            {
                "system": system,
                "op_file": op_file,
                "suspect": side,
                "suspect_kind": kinds.most_common(1)[0][0],
                "attribution": _attribution(slow_ks, ref_ks),
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


def detect_systematic_offsets(
    gaps: list[dict], offset_factor: float, min_systems: int, kmap: dict[tuple[str, str], str]
) -> list[dict]:
    """Find backend pairs whose op-level median ratio deviates in the SAME
    direction on every covered system.

    A shape-local kernel weakness moves with the shape and the hardware; a
    configuration/collection difference (missing torch.compile, eager mode,
    disabled CUDA graphs) multiplies every measurement uniformly — so a
    near-constant median offset reproduced across >= min_systems systems is
    the fingerprint of a framework-level collection issue, not of the shapes.
    Versions may differ per system; grouping is by backend name.
    """
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for s in gaps:
        if s["kind"] != "pair_summary":
            continue
        b_label, _, a_label = s["pair"].partition(" vs ")
        groups[(s["op_file"], a_label.split("/")[0], b_label.split("/")[0])].append(s)

    log_thresh = math.log(offset_factor)
    offsets: list[dict] = []
    for (op_file, backend_a, backend_b), entries in groups.items():
        if len(entries) < min_systems:
            continue
        logs = sorted(math.log(e["median_ratio"]) for e in entries)
        if logs[0] > log_thresh:
            slow, reference = backend_b, backend_a
        elif logs[-1] < -log_thresh:
            slow, reference = backend_a, backend_b
        else:
            continue
        mid = logs[len(logs) // 2]
        slow_is_b = slow == backend_b
        slow_ks = _normalize_kernels(
            kmap, slow, [ks for e in entries for ks in (e["kernel_sources_b"] if slow_is_b else e["kernel_sources_a"])]
        )
        ref_ks = _normalize_kernels(
            kmap,
            reference,
            [ks for e in entries for ks in (e["kernel_sources_a"] if slow_is_b else e["kernel_sources_b"])],
        )
        offsets.append(
            {
                "kind": "systematic_offset",
                "op_file": op_file,
                "slow_backend": slow,
                "reference_backend": reference,
                "systems": sorted(e["system"] for e in entries),
                "overall_median_ratio": float(np.exp(abs(mid))),
                "median_ratio_by_system": {
                    e["system"]: round(float(np.exp(abs(math.log(e["median_ratio"])))), 2) for e in entries
                },
                "kernel_sources_slow": sorted(slow_ks),
                "kernel_sources_reference": sorted(ref_ks),
                "attribution": _attribution(slow_ks, ref_ks),
            }
        )
    offsets.sort(key=lambda o: (-len(o["systems"]), -o["overall_median_ratio"]))
    return offsets


def cluster_curve_findings(anomalies: list[dict], kind: str) -> list[dict]:
    """Cluster mono_violation / spike_violation points per
    (system, op_file, backend, kernel_source, sweep_col)."""
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for a in anomalies:
        if a["kind"] != kind:
            continue
        by_key[(a["system"], a["op_file"], a["backend"], a["version"], a["kernel_source"], a["sweep_col"])].append(a)
    clusters = []
    for (system, op_file, backend, version, ks, sweep), points in by_key.items():
        # Worst = biggest latency drop for mono, biggest spike for spikes.
        worst = (
            min(points, key=lambda p: p["ratio"]) if kind == "mono_violation" else max(points, key=lambda p: p["ratio"])
        )
        clusters.append(
            {
                "system": system,
                "op_file": op_file,
                "backend": f"{backend}/{version}",
                "kernel_source": ks,
                "sweep_col": sweep,
                "points": len(points),
                "worst_ratio": worst["ratio"],
                "example": worst,
            }
        )
    clusters.sort(key=lambda c: -c["points"])
    return clusters


def detect_machine_fingerprint(
    fp_cache: dict[tuple[str, str], dict[str, tuple[str, pd.Series]]],
    factor: float,
    min_ops: int = 5,
    min_shared_shapes: int = 50,
) -> list[dict]:
    """Flag (system, backend, op) combinations whose latency level is off
    relative to how that system compares to its peers on OTHER ops.

    For each (op, backend) covered by >= 3 systems, shapes are aligned across
    systems and each system's median log-deviation from the cross-system
    per-shape median is computed. Subtracting the system's own median
    deviation across ops removes the hardware factor (an H200 is uniformly
    faster than an A100); what remains is op-specific: one op collected badly
    on one machine. CAUTION: this is a hint, not a verdict — version skew
    between systems and differing compute/memory hardware ratios both add
    spread, so findings need human triage and are excluded from the
    --fail-on-anomalies gate.
    """
    dev: dict[tuple[str, str], dict[str, float]] = {}
    versions: dict[tuple[str, str, str], str] = {}
    for (op_file, backend), by_system in fp_cache.items():
        if len(by_system) < 3:
            continue
        frame = pd.DataFrame({sys: series for sys, (_ver, series) in by_system.items()})
        frame = frame[frame.notna().sum(axis=1) >= 3]
        if len(frame) < min_shared_shapes:
            continue
        log_frame = np.log(frame)
        centered = log_frame.sub(log_frame.median(axis=1), axis=0)
        for system, value in centered.median(axis=0).dropna().items():
            dev.setdefault((backend, system), {})[op_file] = float(value)
            versions[(backend, system, op_file)] = by_system[system][0]

    findings: list[dict] = []
    log_factor = math.log(factor)
    for (backend, system), by_op in dev.items():
        if len(by_op) < min_ops:
            continue
        ordered = sorted(by_op.values())
        hardware_offset = ordered[len(ordered) // 2]
        for op_file, value in by_op.items():
            residual = value - hardware_offset
            if abs(residual) > log_factor:
                findings.append(
                    {
                        "kind": "machine_op_deviation",
                        "system": system,
                        "op_file": op_file,
                        "backend": backend,
                        "version": versions[(backend, system, op_file)],
                        "direction": "slow" if residual > 0 else "fast",
                        "rel_ratio": float(np.exp(abs(residual))),
                        "ops_compared": len(by_op),
                    }
                )
    findings.sort(key=lambda f: -f["rel_ratio"])
    return findings


def run_checks(
    data_root: Path,
    systems: list[str] | None,
    backends: list[str] | None,
    op_files: list[str] | None,
    anomaly_factor: float,
    mono_tolerance: float,
    spike_factor: float,
    min_bucket_points: int,
    noise_floor: float,
    spec_root: Path | None = None,
    fingerprint_factor: float | None = 2.0,
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
    # (op_file, backend) -> system -> (version, latency Series indexed by shape hash)
    fp_cache: dict[tuple[str, str], dict[str, tuple[str, pd.Series]]] = {}
    for (system, op_file), by_backend in sorted(tables.items()):
        # Latest version per backend.
        loaded: list[OpTable] = []
        for backend, versions in sorted(by_backend.items()):
            version, path = max(versions, key=lambda vp: _version_key(vp[0]))
            table, npos_by_ks, kernel_costs = _load_op_table(path, backend, version)
            for cost in kernel_costs:
                gaps.append(
                    {
                        "kind": "kernel_choice_cost",
                        "system": system,
                        "op_file": op_file,
                        "backend": backend,
                        "version": version,
                        **cost,
                    }
                )
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
            anomalies.extend(_check_curve(system, op_file, t, mono_tolerance, spike_factor, noise_floor))
            if fingerprint_factor:
                sig_hash = pd.util.hash_pandas_object(
                    _join_cols(t.frame, t.shape_cols, bucket_sweeps=False), index=False
                )
                series = pd.Series(t.frame["latency"].to_numpy(), index=sig_hash.to_numpy())
                fp_cache.setdefault((op_file, t.backend), {})[system] = (
                    t.version,
                    series[~series.index.duplicated()],
                )
            if op_file == "gemm_perf.parquet" and spec_root is not None:
                spec = _load_gpu_spec(spec_root, system)
                if spec:
                    sol_anoms, sol_effs = _check_gemm_sol(system, op_file, t, spec)
                    anomalies.extend(sol_anoms)
                    gaps.extend(sol_effs)

        if len(loaded) < 2:
            continue
        for a, b in itertools.combinations(loaded, 2):
            pair_anoms, pair_gaps = _check_pair(system, op_file, a, b, anomaly_factor, min_bucket_points)
            anomalies.extend(pair_anoms)
            gaps.extend(pair_gaps)
        logger.info("%s/%s: %d backends compared", system, op_file, len(loaded))
    if fingerprint_factor:
        anomalies.extend(detect_machine_fingerprint(fp_cache, fingerprint_factor))
    return anomalies, gaps


def _fmt_shape(shape: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in shape.items() if v not in ("", None))


def render_markdown(
    anomalies: list[dict],
    gaps: list[dict],
    offsets: list[dict],
    suspects: list[dict],
    undetermined: list[dict],
    max_rows: int,
) -> str:
    lines: list[str] = ["# Cross-backend consistency report\n"]

    mono_clusters = cluster_curve_findings(anomalies, "mono_violation")
    spike_clusters = cluster_curve_findings(anomalies, "spike_violation")
    regions = [a for a in anomalies if a["kind"] == "region_deviation"]
    nonpositive = [a for a in anomalies if a["kind"] == "nonpositive_latency"]
    mismatches = [g for g in gaps if g["kind"] == "schema_mismatch"]
    summaries = [g for g in gaps if g["kind"] == "pair_summary"]
    kernel_costs = [g for g in gaps if g["kind"] == "kernel_choice_cost"]
    below_sol = [a for a in anomalies if a["kind"] == "below_sol"]
    sol_effs = [g for g in gaps if g["kind"] == "sol_efficiency"]
    machine = [a for a in anomalies if a["kind"] == "machine_op_deviation"]
    n_outliers = sum(1 for a in anomalies if a["kind"] == "pair_outlier")
    n_mono = sum(1 for a in anomalies if a["kind"] == "mono_violation")
    n_spike = sum(1 for a in anomalies if a["kind"] == "spike_violation")

    lines.append("## Layer 1 — anomalies (suspected invalid data)\n")
    lines.append(f"- Nonpositive-latency rows: **{sum(a['rows'] for a in nonpositive)}** in {len(nonpositive)} tables")
    lines.append(
        f"- Cross-backend point outliers: **{n_outliers}** "
        f"({len(suspects)} corroborated suspect clusters, {len(undetermined)} undetermined clusters)"
    )
    lines.append(f"- Whole-region deviations (bucket median far from op median): **{len(regions)}**")
    lines.append(f"- Monotonicity violations: **{n_mono}** in **{len(mono_clusters)}** clusters")
    lines.append(f"- Curve spikes (within-framework): **{n_spike}** in **{len(spike_clusters)}** clusters")
    lines.append(
        f"- Below speed-of-light points (physically impossible): "
        f"**{sum(a['points'] for a in below_sol)}** in {len(below_sol)} groups"
    )
    lines.append(f"- Machine-fingerprint deviations (hint, not gated): **{len(machine)}**\n")

    if nonpositive:
        lines.append("### Nonpositive-latency rows\n")
        lines.append("| system | op_file | backend | rows | by kernel_source |")
        lines.append("|---|---|---|---|---|")
        for a in sorted(nonpositive, key=lambda x: -x["rows"])[:max_rows]:
            by_ks = ", ".join(f"{k}: {v}" for k, v in a["by_kernel_source"].items())
            lines.append(f"| {a['system']} | {a['op_file']} | {a['backend']}/{a['version']} | {a['rows']} | {by_ks} |")
        lines.append("")

    if below_sol:
        lines.append("### Below speed-of-light (physically impossible measurements)\n")
        lines.append(
            "Latency below max(peak-flops compute time, peak-bandwidth memory time) — "
            "no kernel can be this fast; the measurement did not run what the table claims.\n"
        )
        lines.append("| system | op_file | backend | dtype | points | worst x of SOL | worst example |")
        lines.append("|---|---|---|---|---|---|---|")
        for a in sorted(below_sol, key=lambda x: x["worst_fraction_of_sol"])[:max_rows]:
            lines.append(
                f"| {a['system']} | {a['op_file']} | {a['backend']}/{a['version']} | {a['gemm_dtype']} | "
                f"{a['points']} | {a['worst_fraction_of_sol']:.2f} | "
                f"{_fmt_shape(a['example_shape'])}: {a['example_latency']:.4g} vs SOL {a['example_sol']:.4g} |"
            )
        lines.append("")

    if suspects:
        lines.append(f"### Corroborated suspects (top {min(max_rows, len(suspects))})\n")
        lines.append(
            "The same side deviates against two or more reference backends — that side's "
            "data is the common factor. `suspect_kind` says how it deviates from the local "
            "baseline (too slow or too fast; anomalously FAST usually means a broken measurement).\n"
        )
        lines.append(
            "| system | op_file | suspect | kind | attribution | dtype cols | points | refs | "
            "deviation | worst example |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for c in suspects[:max_rows]:
            ex = c["example"]
            lines.append(
                f"| {c['system']} | {c['op_file']} | **{c['suspect']}** | {c['suspect_kind']} | {c['attribution']} | "
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
                f"{c['sweep_col']} | {c['points']} | {c['worst_ratio']:.2f} | "
                f"{_fmt_shape(ex['fixed_shape'])}: {ex['sweep_col']} {ex['sweep_from']}→{ex['sweep_to']}, "
                f"lat {ex['latency_from']:.4g}→{ex['latency_to']:.4g} |"
            )
        lines.append("")

    if spike_clusters:
        lines.append(f"### Curve spike clusters (top {min(max_rows, len(spike_clusters))})\n")
        lines.append(
            "A point higher than BOTH sweep neighbors — a bad measurement (jitter, "
            "preemption, missing warmup) needing no reference framework.\n"
        )
        lines.append("| system | op_file | backend | kernel_source | sweep | points | worst spike | worst example |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for c in spike_clusters[:max_rows]:
            ex = c["example"]
            lines.append(
                f"| {c['system']} | {c['op_file']} | {c['backend']} | `{c['kernel_source']}` | "
                f"{c['sweep_col']} | {c['points']} | {c['worst_ratio']:.1f}x | "
                f"{_fmt_shape(ex['fixed_shape'])}: {ex['sweep_col']}={ex['sweep_at']}, "
                f"lat {ex['latency']:.4g} vs neighbors {ex['neighbor_latency']:.4g} |"
            )
        lines.append("")

    if machine:
        lines.append(f"### Machine-fingerprint deviations (top {min(max_rows, len(machine))}; hint only)\n")
        lines.append(
            "The system's latency level on this op is off relative to how the same system "
            "compares to its peers on its OTHER ops (hardware factor removed). Version skew "
            "and compute/memory hardware ratios add spread — triage before acting.\n"
        )
        lines.append("| system | backend | op_file | direction | rel ratio | ops compared |")
        lines.append("|---|---|---|---|---|---|")
        for a in machine[:max_rows]:
            lines.append(
                f"| {a['system']} | {a['backend']}/{a['version']} | {a['op_file']} | "
                f"{a['direction']} | {a['rel_ratio']:.2f}x | {a['ops_compared']} |"
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
    if offsets:
        lines.append("### Systematic cross-system offsets\n")
        lines.append(
            "The same backend pair shows a same-direction median offset on EVERY covered "
            "system. A shape-local kernel weakness moves with shape and hardware; a uniform "
            "multiplier reproduced across systems is the fingerprint of a framework-level "
            "collection/configuration difference (eager mode, missing torch.compile, "
            "disabled CUDA graphs).\n"
        )
        lines.append(
            "| op_file | slow backend | reference | systems | median offset | attribution | per-system | slow kernels |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for o in offsets[:max_rows]:
            per_sys = ", ".join(f"{k}:{v}" for k, v in sorted(o["median_ratio_by_system"].items()))
            lines.append(
                f"| {o['op_file']} | **{o['slow_backend']}** | {o['reference_backend']} | "
                f"{len(o['systems'])} | {o['overall_median_ratio']:.2f}x | {o['attribution']} | {per_sys} | "
                f"`{', '.join(o['kernel_sources_slow'])}` |"
            )
        lines.append("")

    if kernel_costs:
        lines.append(f"### Within-framework kernel choice cost (top {min(max_rows, len(kernel_costs))})\n")
        lines.append(
            "For tables that measured several kernel_sources on the same shapes: how much "
            "slower each kernel's median is than the per-shape best — the price of that "
            "backend selection inside one framework.\n"
        )
        lines.append("| system | op_file | backend | kernel_source | median penalty | max | rows |")
        lines.append("|---|---|---|---|---|---|---|")
        for c in sorted(kernel_costs, key=lambda x: -x["median_penalty"])[:max_rows]:
            lines.append(
                f"| {c['system']} | {c['op_file']} | {c['backend']}/{c['version']} | `{c['kernel_source']}` | "
                f"{c['median_penalty']:.2f}x | {c['max_penalty']:.1f}x | {c['rows']} |"
            )
        lines.append("")

    if sol_effs:
        lines.append("### gemm speed-of-light efficiency (environment health)\n")
        lines.append(
            "Median achieved fraction of the physical bound per (system, backend, dtype). "
            "A system whose efficiencies sit well below its peers on the same hardware "
            "generation points at a collection-environment problem, not at the kernels.\n"
        )
        lines.append("| system | backend | dtype | points | median eff | p95 eff |")
        lines.append("|---|---|---|---|---|---|")
        for e in sorted(sol_effs, key=lambda x: (x["system"], x["backend"], x["gemm_dtype"]))[: max_rows * 2]:
            lines.append(
                f"| {e['system']} | {e['backend']}/{e['version']} | {e['gemm_dtype']} | "
                f"{e['points']} | {e['median_efficiency']:.2f} | {e['p95_efficiency']:.2f} |"
            )
        lines.append("")

    if summaries:
        lines.append("### Pair summaries\n")
        lines.append(
            "| system | op_file | pair | points | median | p05 | p95 | region/total buckets | kernels_a | kernels_b |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for s in sorted(summaries, key=lambda x: (x["system"], x["op_file"], x["pair"])):
            lines.append(
                f"| {s['system']} | {s['op_file']} | {s['pair']} | {s['joined_points']} | "
                f"{s['median_ratio']:.2f} | {s['p05_ratio']:.2f} | {s['p95_ratio']:.2f} | "
                f"{s['region_deviation_buckets']}/{s['total_buckets']} | "
                f"`{', '.join(s['kernel_sources_a'])}` | `{', '.join(s['kernel_sources_b'])}` |"
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
        "--mono-tolerance",
        type=float,
        default=0.7,
        help="Layer 1: flag latency drops below this ratio while a sweep dimension grows.",
    )
    parser.add_argument(
        "--spike-factor",
        type=float,
        default=3.0,
        help="Layer 1: flag points higher than both sweep neighbors by more than this factor.",
    )
    parser.add_argument(
        "--fingerprint-factor",
        type=float,
        default=2.0,
        help="Machine-fingerprint hint threshold (0 disables): flag a (system, backend, op) whose "
        "hardware-factor-corrected latency level deviates beyond this factor from the system's "
        "own level on other ops.",
    )
    parser.add_argument(
        "--systems-spec-root",
        type=Path,
        default=Path("aic-core/src/aiconfigurator_core/systems"),
        help="Directory of <system>.yaml gpu specs used for the gemm speed-of-light bound.",
    )
    parser.add_argument(
        "--kernel-map",
        type=Path,
        default=Path("collector/kernel_source_backends.yaml"),
        help="kernel_source -> runtime backend translation table, used to attribute gaps to "
        "kernel choice vs harness/config differences.",
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
    parser.add_argument(
        "--offset-factor",
        type=float,
        default=1.15,
        help="Layer 2: flag a backend pair whose op-level median deviates beyond this factor "
        "in the same direction on every covered system (systematic offset).",
    )
    parser.add_argument(
        "--min-offset-systems",
        type=int,
        default=3,
        help="Minimum number of systems a pair must cover to qualify as a systematic offset.",
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
    if not 0.0 < args.mono_tolerance <= 1.0:
        parser.error("--mono-tolerance must be in (0, 1]")
    if args.offset_factor <= 1.0:
        parser.error("--offset-factor must be > 1")
    if args.spike_factor <= 1.0:
        parser.error("--spike-factor must be > 1")

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")

    anomalies, gaps = run_checks(
        data_root=args.data_root,
        systems=args.systems,
        backends=args.backends,
        op_files=args.op_files,
        anomaly_factor=args.anomaly_factor,
        mono_tolerance=args.mono_tolerance,
        spike_factor=args.spike_factor,
        min_bucket_points=args.min_bucket_points,
        noise_floor=args.noise_floor,
        spec_root=args.systems_spec_root,
        fingerprint_factor=args.fingerprint_factor or None,
    )

    kmap = load_kernel_map(args.kernel_map)
    suspects, undetermined = cluster_suspects(anomalies, kmap)
    mono_clusters = cluster_curve_findings(anomalies, "mono_violation")
    spike_clusters = cluster_curve_findings(anomalies, "spike_violation")
    offsets = detect_systematic_offsets(gaps, args.offset_factor, args.min_offset_systems, kmap)
    regions = [a for a in anomalies if a["kind"] == "region_deviation"]
    nonpositive = [a for a in anomalies if a["kind"] == "nonpositive_latency"]
    below_sol = [a for a in anomalies if a["kind"] == "below_sol"]

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
                    "spike_clusters": spike_clusters,
                    "systematic_offsets": offsets,
                },
                indent=2,
            )
        )
        logger.info("wrote %s", args.out_json)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_markdown(anomalies, gaps, offsets, suspects, undetermined, args.max_report_rows))
        logger.info("wrote %s", args.out_md)

    n_outliers = sum(1 for a in anomalies if a["kind"] == "pair_outlier")
    n_mono = sum(1 for a in anomalies if a["kind"] == "mono_violation")
    n_spike = sum(1 for a in anomalies if a["kind"] == "spike_violation")
    n_mismatch = sum(1 for g in gaps if g["kind"] == "schema_mismatch")
    print(
        f"\nLayer 1: {sum(a['rows'] for a in nonpositive)} nonpositive-latency rows, "
        f"{n_outliers} point outliers ({len(suspects)} corroborated suspects, "
        f"{len(undetermined)} undetermined), {len(regions)} region deviations, "
        f"{n_mono} mono violations in {len(mono_clusters)} clusters, "
        f"{n_spike} spikes in {len(spike_clusters)} clusters, "
        f"{sum(a['points'] for a in below_sol)} below-SOL points in {len(below_sol)} groups"
    )
    for a in sorted(below_sol, key=lambda x: x["worst_fraction_of_sol"])[:5]:
        print(
            f"  BELOW-SOL! {a['system']}/{a['op_file']}: {a['backend']}/{a['version']} {a['gemm_dtype']} — "
            f"{a['points']} points, worst at {a['worst_fraction_of_sol']:.2f}x of physical bound "
            f"({_fmt_shape(a['example_shape'])}: {a['example_latency']:.4g} vs SOL {a['example_sol']:.4g})"
        )
    machine = [a for a in anomalies if a["kind"] == "machine_op_deviation"]
    for a in machine[:5]:
        print(
            f"  MACHINE? {a['system']} {a['backend']}/{a['version']} {a['op_file']}: "
            f"{a['rel_ratio']:.2f}x {a['direction']} vs the system's own level on "
            f"{a['ops_compared']} other ops"
        )
    print(
        f"Layer 2: {len(offsets)} systematic offsets across "
        f"{sum(1 for g in gaps if g['kind'] == 'pair_summary')} backend pairs"
        + (f"; {n_mismatch} pairs skipped (schema mismatch)" if n_mismatch else "")
    )
    for o in offsets[: args.max_report_rows if args.max_report_rows < 10 else 10]:
        print(
            f"  OFFSET? {o['op_file']}: {o['slow_backend']} is ~{o['overall_median_ratio']:.2f}x slower than "
            f"{o['reference_backend']} on all {len(o['systems'])} covered systems "
            f"({o['attribution']}) [{', '.join(o['kernel_sources_slow'])}]"
        )
    for c in suspects[:10]:
        print(
            f"  SUSPECT? {c['system']}/{c['op_file']}: {c['suspect']} ({c['suspect_kind']}) "
            f"{_fmt_shape(c['categorical_shape'])} — {c['points']} points, "
            f"deviation {c['deviation_min']:.1f}-{c['deviation_max']:.1f}x, "
            f"refs: {', '.join(c['reference_backends'])}"
        )

    if args.fail_on_anomalies and (nonpositive or below_sol or n_outliers or n_mono or n_spike or regions):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

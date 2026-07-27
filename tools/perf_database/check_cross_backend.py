# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Multi-reference sanity checks of perf-database perf tables.

Different backends legitimately differ — this tool does NOT demand
consistency. It flags SUSPICIOUS measurements by checking each table against
several independent reference frames (sibling backends' local baseline,
sibling systems, the curve's own neighboring shapes, physical speed-of-light
bounds) and asks a human to double-check them; nothing is blocked by default.

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
joined: extra columns that are constant in their table are dropped (recorded
in the pair-summary align_notes); otherwise the pair is skipped and reported as `schema_mismatch`.

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
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backend_facts import load_backend_map, translate
from check_kernel_source import _META_COLUMNS, _iter_data_files

logger = logging.getLogger(__name__)

# op_name is metadata for the sibling scanner but NOT for this checker:
# several tables store multiple logical operations in one file (e.g.
# mla_bmm_perf's mla_gen_pre/mla_gen_post) and their shapes must never be
# compared across op_name values.
_SHAPE_EXCLUDED_META = _META_COLUMNS - {"op_name"}

# Single latency columns, tried in order.
_SINGLE_LATENCY_COLUMNS = ("latency", "avg_ms")

# Multi-component schemas whose runtime consumer SUMS the components — the
# checker must audit the same effective latency the consumer sees:
#   - wideep_deepep_ll: combine + dispatch
#   - wideep_deepep_normal: notify + transmit for both phases
_LATENCY_COMPONENT_SETS = (
    ("combine_avg_t_us", "dispatch_avg_t_us"),
    ("combine_notify_us", "combine_transmit_us", "dispatch_notify_us", "dispatch_transmit_us"),
)

# Derived output metrics that must never enter the shape key.
_DERIVED_METRIC_SUFFIXES = ("_bandwidth_gbps",)


class _SchemaUnsupportedError(Exception):
    """No recognized latency column or component set in the table."""


def _resolve_latency(df: pd.DataFrame) -> tuple[pd.Series, str, set[str]]:
    """Returns (effective latency, unit label column, all latency-ish columns
    to exclude from the shape key)."""
    excluded = set(_SINGLE_LATENCY_COLUMNS) | {c for s in _LATENCY_COMPONENT_SETS for c in s}
    excluded |= {c for c in df.columns if c.endswith(_DERIVED_METRIC_SUFFIXES)}
    for col in _SINGLE_LATENCY_COLUMNS:
        if col in df.columns:
            return df[col], col, excluded
    for components in _LATENCY_COMPONENT_SETS:
        if all(c in df.columns for c in components):
            return df[list(components)].sum(axis=1), components[0], excluded
    raise _SchemaUnsupportedError(f"no recognized latency column among {sorted(df.columns)}")


# Shape columns treated as sweep dimensions: log2-bucketed for the local
# baseline, and checked for latency monotonicity within a backend.
_SWEEP_COLUMNS = ("batch_size", "isl", "m", "num_tokens", "step")

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

# --fail-on-anomalies trips when a kind's tally EXCEEDS its allowance below
# (override per kind with --gate-threshold KIND=N). Tallies: nonpositive rows
# and below-SOL points are summed; other kinds count findings. Rationale for
# the nonzero defaults: point outliers / regions / curve violations carry
# statistical noise, and even below-SOL tolerates a few points of spec
# rounding — a gate should fire on a PATTERN, not a stray point.
# machine_op_deviation is a hint by design and stays ungated.
_GATE_THRESHOLDS = {
    "unreadable_table": 0,
    "nonpositive_latency": 0,
    "below_sol": 10,
    "pair_outlier": 100,
    "region_deviation": 20,
    "mono_violation": 100,
    "spike_violation": 50,
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
    """(system, backend, version, op_file, path) via the sibling tool's dual-layout walker."""
    for system, backend, version, path in _iter_data_files(data_root):
        yield system, backend, version, path.name, path


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


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
    latency, unit_col, latency_excluded = _resolve_latency(df)
    if "kernel_source" not in df.columns:
        df["kernel_source"] = "<unknown>"
    # A missing label must not hide a corrupt row (NaN groups vanish from
    # value_counts by default).
    df["kernel_source"] = df["kernel_source"].fillna("<unknown>")
    nonpositive = df[latency.notna() & (latency <= 0)]
    npos_by_ks = {str(k): int(v) for k, v in nonpositive["kernel_source"].value_counts().items()}
    shape_cols = [c for c in df.columns if c not in _SHAPE_EXCLUDED_META and c not in latency_excluded]
    keep = latency > 0
    df = df[keep].drop(columns=[c for c in latency_excluded if c in df.columns]).assign(latency=latency[keep])
    if df.empty or not shape_cols:
        return None, npos_by_ks, []
    kernel_costs: list[dict] = []
    shape_gb = df.groupby(shape_cols, dropna=False)["latency"]
    if df["kernel_source"].nunique() > 1:
        penalty = df["latency"] / shape_gb.transform("min")
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
    idx = shape_gb.idxmin()
    reduced = df.loc[idx, [*shape_cols, "latency", "kernel_source"]].reset_index(drop=True)
    table = OpTable(
        backend=backend,
        version=version,
        shape_cols=shape_cols,
        frame=reduced,
        kernel_sources=sorted(df["kernel_source"].astype(str).unique()),
        noise_scale=1000.0 if unit_col.endswith("_us") else 1.0,
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

    def _drop_constant_extras(own: OpTable, other: OpTable) -> pd.DataFrame | None:
        frame = own.frame
        for col in (c for c in own.shape_cols if c not in other.shape_cols):
            values = frame[col]
            if values.nunique(dropna=False) != 1:
                return None
            notes.append(f"{own.label}: dropped constant column {col}={values.iloc[0]!r}")
            frame = frame.drop(columns=[col])
        return frame

    frame_a = _drop_constant_extras(a, b)
    frame_b = _drop_constant_extras(b, a)
    shape_cols = [c for c in a.shape_cols if c in b.shape_cols]
    if frame_a is None or frame_b is None or not shape_cols:
        return None
    return shape_cols, frame_a, frame_b, notes


def _check_pair(
    system: str, op_file: str, a: OpTable, b: OpTable, anomaly_factor: float, min_bucket_points: int
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
    bucket_gb = merged.groupby("bucket")["log_ratio"]
    series_gb = merged.groupby("series")["log_ratio"]
    bucket_med = bucket_gb.transform("median")
    bucket_n = bucket_gb.transform("size")
    series_med = series_gb.transform("median")
    series_n = series_gb.transform("size")
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
        slow, fast = (b.label, a.label) if row.log_ratio > row.baseline else (a.label, b.label)
        shape = {c: _jsonable(getattr(row, c)) for c in shape_cols}
        anomalies.append(
            {
                "kind": "pair_outlier",
                "system": system,
                "op_file": op_file,
                "pair": pair,
                "shape": shape,
                "cat_shape": {c: shape[c] for c in cat_cols},
                "latency_a": float(row.latency_a),
                "latency_b": float(row.latency_b),
                "kernel_source_a": str(row.kernel_source_a),
                "kernel_source_b": str(row.kernel_source_b),
                "ratio": float(np.exp(row.log_ratio)),
                "local_baseline_ratio": float(np.exp(row.baseline)),
                "deviation": float(row.deviation),
                "candidates": [
                    {"side": slow, "kind": "slower_than_expected"},
                    {"side": fast, "kind": "faster_than_expected"},
                ],
            }
        )

    # ---- Bucket-level stats: region deviations (L1) + gaps (L2) ------------
    gaps: list[dict] = []
    grp = bucket_gb.agg(median_log_ratio="median", points="size")
    grp = grp[grp["points"] >= min_bucket_points]

    # A region deviates when its median is far from how these two backends
    # normally compare on this op (the op-level median) — NOT when the
    # absolute gap is large, which would flag every bucket of a legitimately
    # slower backend.
    rel = (grp["median_log_ratio"] - global_med).abs()
    for bucket, row in grp[rel > log_anomaly].iterrows():
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
            "region_deviation_buckets": int((rel > log_anomaly).sum()),
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
            base = {
                "system": system,
                "op_file": op_file,
                "backend": t.backend,
                "version": t.version,
                "sweep_col": sweep,
                "fixed_shape": {k: _jsonable(v) for k, v in fixed.items()},
            }
            drop = lat[1:] / lat[:-1]
            bad = np.nonzero((drop < mono_tolerance) & (lat[:-1] >= floor))[0]
            for i in bad:
                findings.append(
                    {
                        "kind": "mono_violation",
                        **base,
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
                            **base,
                            "sweep_at": _jsonable(sw[i + 1]),
                            "latency": float(lat[i + 1]),
                            "neighbor_latency": float(neighbors[i]),
                            "ratio": float(lat[i + 1] / neighbors[i]),
                            "kernel_source": str(series["kernel_source"].iloc[i + 1]),
                        }
                    )
    return findings


def _backends_of(pair: str) -> str:
    b_label, _, a_label = pair.partition(" vs ")
    return "+".join(sorted((a_label.split("/")[0], b_label.split("/")[0])))


def _finding_key(a: dict) -> str:
    """Stable identity of a gated finding for baseline/ratchet comparison.

    Deliberately version-free (a backend version bump must not turn every
    known finding into a 'new' one) and shape-anchored (the same bad point
    keeps the same key across runs)."""
    kind = a["kind"]
    if kind == "pair_outlier":
        return f"{kind}|{a['system']}|{a['op_file']}|{_backends_of(a['pair'])}|{_sig(a['shape'])}"
    if kind == "region_deviation":
        return f"{kind}|{a['system']}|{a['op_file']}|{_backends_of(a['pair'])}|{a['bucket']}"
    if kind in ("mono_violation", "spike_violation"):
        at = a.get("sweep_at", f"{a.get('sweep_from')}>{a.get('sweep_to')}")
        return (
            f"{kind}|{a['system']}|{a['op_file']}|{a['backend']}|{a['kernel_source']}|"
            f"{a['sweep_col']}|{_sig(a['fixed_shape'])}|{at}"
        )
    # Group-level kinds: nonpositive_latency, below_sol, unreadable_table.
    return f"{kind}|{a['system']}|{a['op_file']}|{a['backend']}|{a.get('gemm_dtype', '')}"


# Group-level kinds whose finding aggregates a COUNT: the baseline must
# remember the count, not just the key, or growth inside a known group would
# be invisible to the ratchet.
_GROUP_COUNT_FIELDS = {"nonpositive_latency": "rows", "below_sol": "points"}
_BASELINE_SCHEMA = 2


def snapshot_baseline(gated: list[dict]) -> dict:
    keys: set[str] = set()
    counts: dict[str, int] = {}
    for a in gated:
        key = _key_hash(_finding_key(a))
        if a["kind"] in _GROUP_COUNT_FIELDS:
            counts[key] = counts.get(key, 0) + a[_GROUP_COUNT_FIELDS[a["kind"]]]
        else:
            keys.add(key)
    return {"schema": _BASELINE_SCHEMA, "keys": sorted(keys), "counts": dict(sorted(counts.items()))}


def evaluate_gate(
    gated: list[dict], thresholds: dict[str, int], baseline: dict | None
) -> tuple[dict[str, int], Counter, int]:
    """Returns (breaches, new-finding tallies per kind, suppressed count).

    With a baseline, point-level findings are suppressed by key membership;
    group-level findings are suppressed only up to the baselined COUNT — a
    known group that grows contributes its excess as new."""
    known_keys = set(baseline["keys"]) if baseline else set()
    known_counts = baseline.get("counts", {}) if baseline else {}
    tallies: Counter = Counter()
    suppressed = 0
    for a in gated:
        kind = a["kind"]
        key = _key_hash(_finding_key(a))
        if kind in _GROUP_COUNT_FIELDS:
            current = a[_GROUP_COUNT_FIELDS[kind]]
            known = known_counts.get(key, 0)
            suppressed += min(current, known)
            if current > known:
                tallies[kind] += current - known
        elif key in known_keys:
            suppressed += 1
        else:
            tallies[kind] += 1
    return {k: n for k, n in tallies.items() if n > thresholds[k]}, tallies, suppressed


def _key_hash(key: str) -> str:
    import hashlib

    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _sig(d: dict, sep: str = "|") -> str:
    return sep.join(f"{k}={v}" for k, v in d.items())


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


def load_kernel_map(path: Path) -> list[dict]:
    """collector/kernel_source_backends.yaml mapping entries, via the
    backend_facts loader/translator (honors per-entry `match:` conditions).
    Returns [] when unavailable — attribution then uses raw label names."""
    try:
        return load_backend_map(path)
    except (OSError, KeyError, TypeError) as exc:
        logger.warning("kernel map %s unavailable (%s); attribution uses raw kernel_source names", path, exc)
        return []


def _normalize_kernels(
    kmap: list[dict], framework: str, op_file: str, kernel_sources, axes: dict | None = None
) -> set[str]:
    """Normalize kernel_source labels via the registry. The registry keys ops
    by STEM (gemm_perf, not gemm_perf.parquet) and its conditional entries
    match on axis values (gemm_dtype, ...). Labels whose only registry entries
    are conditional and whose conditions cannot be evaluated here resolve to
    an `unresolved:` marker so attribution reports `unknown` rather than
    guessing."""
    stem = op_file.rsplit(".", 1)[0]
    axis_map = {k: str(v) for k, v in (axes or {}).items()}
    out: set[str] = set()
    for entry in kernel_sources:
        for ks in str(entry).split(","):
            ks = ks.strip()
            if not ks:
                continue
            backend = translate(kmap, framework, stem, axis_map, ks)
            if backend is not None:
                out.add(backend)
            elif any(m["framework"] == framework and m["kernel_source"] == ks for m in kmap):
                out.add(f"unresolved:{ks}")
            else:
                out.add(ks.lower())
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

    def _opaque(kernels: set[str]) -> bool:
        return all(k in _OPAQUE_KERNELS or k.startswith("unresolved:") for k in kernels)

    if _opaque(slow_kernels) or _opaque(reference_kernels):
        return "unknown"
    return "harness_config" if slow_kernels & reference_kernels else "kernel_choice"


def cluster_suspects(anomalies: list[dict], kmap: list[dict]) -> tuple[list[dict], list[dict]]:
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
        cat_sig = _sig(a["cat_shape"])
        for cand in a["candidates"]:
            by_candidate[(a["system"], a["op_file"], cand["side"], cat_sig)].append((a, cand["kind"]))

    def _other_side(point: dict, side: str) -> str:
        left, _, right = point["pair"].partition(" vs ")
        return right if left == side else left

    def _cluster_common(points: list[dict]) -> dict:
        deviations = sorted(p["deviation"] for p in points)
        worst = max(points, key=lambda p: p["deviation"])
        return {
            "system": worst["system"],
            "op_file": worst["op_file"],
            "categorical_shape": worst["cat_shape"],
            "points": len({_sig(p["shape"]) for p in points}),
            "deviation_min": deviations[0],
            "deviation_max": deviations[-1],
            "example_shape": worst["shape"],
            "example": {"ratio": worst["ratio"], "local_baseline_ratio": worst["local_baseline_ratio"]},
        }

    corroborated: list[dict] = []
    attributed_ids: set[int] = set()
    for (system, op_file, side, _cat_sig), entries in by_candidate.items():
        refs = sorted({_other_side(p, side) for p, _ in entries})
        if len(refs) < 2:
            continue
        points = [p for p, _ in entries]
        kinds = Counter(kind for _, kind in entries)
        slow_ks: set[str] = set()
        ref_ks: set[str] = set()
        for point in points:
            b_label, _, a_label = point["pair"].partition(" vs ")
            slow_sfx, ref_label = ("_b", a_label) if side == b_label else ("_a", b_label)
            ref_sfx = "_a" if slow_sfx == "_b" else "_b"
            axes = point["cat_shape"]
            slow_ks |= _normalize_kernels(kmap, side.split("/")[0], op_file, [point[f"kernel_source{slow_sfx}"]], axes)
            ref_ks |= _normalize_kernels(
                kmap, ref_label.split("/")[0], op_file, [point[f"kernel_source{ref_sfx}"]], axes
            )
        corroborated.append(
            {
                **_cluster_common(points),
                "suspect": side,
                "suspect_kind": kinds.most_common(1)[0][0],
                "attribution": _attribution(slow_ks, ref_ks),
                "reference_backends": refs,
            }
        )
        attributed_ids.update(id(p) for p in points)
    corroborated.sort(key=lambda c: -c["points"])

    undetermined_by_key: dict[tuple, list[dict]] = defaultdict(list)
    for a in anomalies:
        if a["kind"] != "pair_outlier" or id(a) in attributed_ids:
            continue
        undetermined_by_key[(a["system"], a["op_file"], a["pair"], _sig(a["cat_shape"]))].append(a)
    undetermined: list[dict] = []
    for (_system, _op_file, pair, _cat_sig), points in undetermined_by_key.items():
        undetermined.append({**_cluster_common(points), "pair": pair})
    undetermined.sort(key=lambda c: -c["points"])
    return corroborated, undetermined


def detect_systematic_offsets(
    gaps: list[dict], offset_factor: float, min_systems: int, dispersion: float, kmap: list[dict]
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
        # A systematic (configuration-level) offset must be a NEAR-CONSTANT
        # multiplier: same-direction 1.2x/4x/20x is not one. Require the
        # per-system medians to sit within `dispersion` of each other.
        spread = float(np.exp(logs[-1] - logs[0]))
        if spread > dispersion:
            continue
        mid = logs[len(logs) // 2]
        slow_sfx, ref_sfx = ("_b", "_a") if slow == backend_b else ("_a", "_b")
        slow_ks = _normalize_kernels(
            kmap, slow, op_file, [ks for e in entries for ks in e[f"kernel_sources{slow_sfx}"]]
        )
        ref_ks = _normalize_kernels(
            kmap, reference, op_file, [ks for e in entries for ks in e[f"kernel_sources{ref_sfx}"]]
        )
        offsets.append(
            {
                "kind": "systematic_offset",
                "op_file": op_file,
                "slow_backend": slow,
                "reference_backend": reference,
                "systems": sorted(e["system"] for e in entries),
                "overall_median_ratio": float(np.exp(abs(mid))),
                "spread": round(spread, 2),
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
        # Series already hold log-latency (float32), so no log pass here.
        frame = pd.DataFrame({sys: series for sys, (_ver, series) in by_system.items()})
        frame = frame[frame.notna().sum(axis=1) >= 3]
        if len(frame) < min_shared_shapes:
            continue
        centered = frame.sub(frame.median(axis=1), axis=0)
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
            try:
                table, npos_by_ks, kernel_costs = _load_op_table(path, backend, version)
            except _SchemaUnsupportedError as exc:
                # Deliberately informational, not gated: a new op family's
                # schema must be added to _SINGLE_LATENCY_COLUMNS /
                # _LATENCY_COMPONENT_SETS before this tool can audit it, and
                # the report says so instead of silently skipping.
                gaps.append(
                    {
                        "kind": "unsupported_schema",
                        "system": system,
                        "op_file": op_file,
                        "backend": backend,
                        "version": version,
                        "error": str(exc)[:200],
                    }
                )
                logger.warning("unsupported schema %s: %s", path, exc)
                continue
            except Exception as exc:
                anomalies.append(
                    {
                        "kind": "unreadable_table",
                        "system": system,
                        "op_file": op_file,
                        "backend": backend,
                        "version": version,
                        "error": f"{type(exc).__name__}: {exc}"[:200],
                    }
                )
                logger.warning("unreadable table %s: %s", path, exc)
                continue
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
                sig_hash = pd.util.hash_pandas_object(t.frame[t.shape_cols], index=False)
                series = pd.Series(np.log(t.frame["latency"].to_numpy(dtype=np.float32)), index=sig_hash.to_numpy())
                fp_cache.setdefault((op_file, t.backend), {})[system] = (t.version, series[~series.index.duplicated()])
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
    return _sig({k: v for k, v in shape.items() if v not in ("", None)}, sep=", ")


def derive_views(anomalies: list[dict], gaps: list[dict]) -> dict:
    """Per-kind slices and clusters, derived ONCE and shared by the markdown
    report, the JSON dump, the console summary, and the CI gate."""

    def kind(items: list[dict], k: str) -> list[dict]:
        return [x for x in items if x["kind"] == k]

    return {
        "unreadable": kind(anomalies, "unreadable_table"),
        "nonpositive": kind(anomalies, "nonpositive_latency"),
        "below_sol": kind(anomalies, "below_sol"),
        "regions": kind(anomalies, "region_deviation"),
        "machine": kind(anomalies, "machine_op_deviation"),
        "n_outliers": len(kind(anomalies, "pair_outlier")),
        "mono_clusters": cluster_curve_findings(anomalies, "mono_violation"),
        "spike_clusters": cluster_curve_findings(anomalies, "spike_violation"),
        "n_mono": len(kind(anomalies, "mono_violation")),
        "n_spike": len(kind(anomalies, "spike_violation")),
        "mismatches": kind(gaps, "schema_mismatch"),
        "unsupported": kind(gaps, "unsupported_schema"),
        "summaries": kind(gaps, "pair_summary"),
        "kernel_costs": kind(gaps, "kernel_choice_cost"),
        "sol_effs": kind(gaps, "sol_efficiency"),
    }


def synthesize_problems(views: dict, offsets: list[dict], suspects: list[dict], top: int = 15) -> list[dict]:
    """Merge all detectors' evidence into per-(backend, op_file) problems,
    ranked by evidence strength — the triage list a human reads first.

    A problem's score grows with the number of independent detectors that
    corroborate it, the number of systems it reproduces on, and its size.
    """
    problems: dict[tuple[str, str], dict] = {}

    def add(backend: str, op_file: str, system: str | None, detector: str, evidence: str, worst: float = 1.0):
        prob = problems.setdefault(
            (backend, op_file),
            {"backend": backend, "op_file": op_file, "systems": set(), "detectors": {}, "worst": 1.0},
        )
        if system:
            prob["systems"].add(system)
        prob["detectors"].setdefault(detector, evidence)
        prob["worst"] = max(prob["worst"], worst)

    # Cross-system aggregation of per-system suspect clusters.
    by_suspect: dict[tuple, list[dict]] = defaultdict(list)
    for c in suspects:
        by_suspect[(c["suspect"].split("/")[0], c["op_file"], _sig(c["categorical_shape"]))].append(c)
    for (backend, op_file, _cat), cs in by_suspect.items():
        worst = max(c["deviation_max"] for c in cs)
        att = Counter(c["attribution"] for c in cs).most_common(1)[0][0]
        add(
            backend,
            op_file,
            None,
            "suspect",
            f"corroborated suspect on {len({c['system'] for c in cs})} system(s) "
            f"({_fmt_shape(cs[0]['categorical_shape'])}, {att}): "
            f"{sum(c['points'] for c in cs)} pts, dev up to {worst:.0f}x",
            worst,
        )
        for c in cs:
            problems[(backend, op_file)]["systems"].add(c["system"])
    for o in offsets:
        add(
            o["slow_backend"],
            o["op_file"],
            None,
            "offset",
            f"uniform {o['overall_median_ratio']:.2f}x offset vs {o['reference_backend']} "
            f"on all {len(o['systems'])} systems ({o['attribution']})",
            o["overall_median_ratio"],
        )
        problems[(o["slow_backend"], o["op_file"])]["systems"].update(o["systems"])
    for a in views["machine"]:
        add(
            a["backend"],
            a["op_file"],
            a["system"],
            "machine",
            f"machine-level {a['rel_ratio']:.1f}x {a['direction']} on {a['system']} "
            f"(vs its own level on {a['ops_compared']} ops)",
            a["rel_ratio"],
        )
    for kind_key, label in (("mono_clusters", "mono drops"), ("spike_clusters", "spikes")):
        agg: dict[tuple[str, str], int] = defaultdict(int)
        for c in views[kind_key]:
            agg[(c["backend"].split("/")[0], c["op_file"])] += c["points"]
        for (backend, op_file), pts in agg.items():
            if pts >= 100:
                add(backend, op_file, None, kind_key, f"{pts} {label} along sweep curves")
    for a in views["nonpositive"]:
        add(a["backend"], a["op_file"], a["system"], "nonpositive", f"{a['rows']} nonpositive-latency rows")
    for a in views["unreadable"]:
        add(a["backend"], a["op_file"], a["system"], "unreadable", f"unreadable table: {a['error']}", 100.0)
    for a in views["below_sol"]:
        add(
            a["backend"],
            a["op_file"],
            a["system"],
            "below_sol",
            f"{a['points']} points below the physical bound (worst {a['worst_fraction_of_sol']:.2f}x of SOL)",
        )

    ranked = sorted(
        problems.values(),
        key=lambda p: -(len(p["detectors"]) * 10 + len(p["systems"]) * 2 + math.log10(p["worst"] + 1)),
    )
    return ranked[:top]


def _problem_lines(problems: list[dict]) -> list[str]:
    return [
        f"{i}. **{p['backend']} · {p['op_file']}** ({len(p['systems'])} systems, "
        f"{len(p['detectors'])} detector(s)) — " + "; ".join(p["detectors"].values())
        for i, p in enumerate(problems, 1)
    ]


def _md_section(
    lines: list[str], title: str, headers: str, rows: list[str], note: str | None = None, top: int | None = None
) -> None:
    """Append one report section. `headers` is a pipe-separated header line;
    each row is the matching pipe-separated cell string."""
    if not rows:
        return
    suffix = f" (top {min(top, len(rows))})" if top else ""
    lines.append(f"### {title}{suffix}\n")
    if note:
        lines.append(note + "\n")
    lines.append(f"| {headers} |")
    lines.append("|" + "---|" * (headers.count("|") + 1))
    lines.extend(f"| {r} |" for r in rows[: top or len(rows)])
    lines.append("")


def render_markdown(
    views: dict, offsets: list[dict], suspects: list[dict], undetermined: list[dict], max_rows: int
) -> str:
    v = views
    lines: list[str] = ["# Perf data sanity report\n"]

    problems = synthesize_problems(v, offsets, suspects)
    if problems:
        lines.append("## Triage — start here\n")
        lines.append(
            "Every detector's evidence merged per (backend, op) and ranked by strength "
            "(independent detectors, systems reproduced on, magnitude). The tables below "
            "are the appendix.\n"
        )
        lines.extend(_problem_lines(problems))
        lines.append("")

    lines.append("## Layer 1 — anomalies (suspected invalid data)\n")
    lines.append(
        f"- Nonpositive-latency rows: **{sum(a['rows'] for a in v['nonpositive'])}** in {len(v['nonpositive'])} tables"
    )
    lines.append(
        f"- Cross-backend point outliers: **{v['n_outliers']}** "
        f"({len(suspects)} corroborated suspect clusters, {len(undetermined)} undetermined clusters)"
    )
    lines.append(f"- Whole-region deviations (bucket median far from op median): **{len(v['regions'])}**")
    lines.append(f"- Monotonicity violations: **{v['n_mono']}** in **{len(v['mono_clusters'])}** clusters")
    lines.append(f"- Curve spikes (within-framework): **{v['n_spike']}** in **{len(v['spike_clusters'])}** clusters")
    lines.append(
        f"- Below speed-of-light points (physically impossible): "
        f"**{sum(a['points'] for a in v['below_sol'])}** in {len(v['below_sol'])} groups"
    )
    lines.append(f"- Machine-fingerprint deviations (hint, not gated): **{len(v['machine'])}**")
    lines.append(f"- Unreadable tables: **{len(v['unreadable'])}**\n")

    _md_section(
        lines,
        "Unreadable tables (corrupt or truncated files)",
        "system | op_file | backend | error",
        [f"{a['system']} | {a['op_file']} | {a['backend']}/{a['version']} | {a['error']}" for a in v["unreadable"]],
        top=max_rows,
    )
    _md_section(
        lines,
        "Nonpositive-latency rows",
        "system | op_file | backend | rows | by kernel_source",
        [
            f"{a['system']} | {a['op_file']} | {a['backend']}/{a['version']} | {a['rows']} | "
            + ", ".join(f"{k}: {n}" for k, n in a["by_kernel_source"].items())
            for a in sorted(v["nonpositive"], key=lambda x: -x["rows"])
        ],
        top=max_rows,
    )
    _md_section(
        lines,
        "Below speed-of-light (physically impossible measurements)",
        "system | op_file | backend | dtype | points | worst x of SOL | worst example",
        [
            f"{a['system']} | {a['op_file']} | {a['backend']}/{a['version']} | {a['gemm_dtype']} | "
            f"{a['points']} | {a['worst_fraction_of_sol']:.2f} | "
            f"{_fmt_shape(a['example_shape'])}: {a['example_latency']:.4g} vs SOL {a['example_sol']:.4g}"
            for a in sorted(v["below_sol"], key=lambda x: x["worst_fraction_of_sol"])
        ],
        note=(
            "Latency below max(peak-flops compute time, peak-bandwidth memory time) — "
            "no kernel can be this fast; the measurement did not run what the table claims."
        ),
        top=max_rows,
    )
    _md_section(
        lines,
        "Corroborated suspects",
        "system | op_file | suspect | kind | attribution | dtype cols | points | refs | deviation | worst example",
        [
            f"{c['system']} | {c['op_file']} | **{c['suspect']}** | {c['suspect_kind']} | {c['attribution']} | "
            f"{_fmt_shape(c['categorical_shape'])} | {c['points']} | {', '.join(c['reference_backends'])} | "
            f"{c['deviation_min']:.1f}-{c['deviation_max']:.1f}x | "
            f"{_fmt_shape(c['example_shape'])}: ratio {c['example']['ratio']:.2f} "
            f"vs baseline {c['example']['local_baseline_ratio']:.2f}"
            for c in suspects
        ],
        note=(
            "The same side deviates against two or more reference backends — that side's "
            "data is the common factor. `suspect_kind` says how it deviates from the local "
            "baseline (too slow or too fast; anomalously FAST usually means a broken measurement)."
        ),
        top=max_rows,
    )
    _md_section(
        lines,
        "Undetermined pair mismatches",
        "system | op_file | pair | dtype cols | points | deviation | worst example",
        [
            f"{c['system']} | {c['op_file']} | {c['pair']} | {_fmt_shape(c['categorical_shape'])} | "
            f"{c['points']} | {c['deviation_min']:.1f}-{c['deviation_max']:.1f}x | "
            f"{_fmt_shape(c['example_shape'])}: ratio {c['example']['ratio']:.2f} "
            f"vs baseline {c['example']['local_baseline_ratio']:.2f}"
            for c in undetermined
        ],
        note=(
            "Only one reference backend exists for these points, so the deviating side "
            "cannot be attributed — one of the two tables is off."
        ),
        top=max_rows,
    )
    _md_section(
        lines,
        "Whole-region deviations",
        "system | op_file | pair | bucket | median ratio | op median | rel deviation | points",
        [
            f"{r['system']} | {r['op_file']} | {r['pair']} | `{r['bucket']}` | {r['median_ratio']:.2f} | "
            f"{r['op_median_ratio']:.2f} | {r['rel_deviation']:.1f}x | {r['points']}"
            for r in sorted(v["regions"], key=lambda x: -x["rel_deviation"])
        ],
        top=max_rows,
    )
    _md_section(
        lines,
        "Monotonicity violation clusters",
        "system | op_file | backend | kernel_source | sweep | points | worst drop | worst example",
        [
            f"{c['system']} | {c['op_file']} | {c['backend']} | `{c['kernel_source']}` | {c['sweep_col']} | "
            f"{c['points']} | {c['worst_ratio']:.2f} | "
            f"{_fmt_shape(c['example']['fixed_shape'])}: {c['example']['sweep_col']} "
            f"{c['example']['sweep_from']}→{c['example']['sweep_to']}, "
            f"lat {c['example']['latency_from']:.4g}→{c['example']['latency_to']:.4g}"
            for c in v["mono_clusters"]
        ],
        top=max_rows,
    )
    _md_section(
        lines,
        "Curve spike clusters",
        "system | op_file | backend | kernel_source | sweep | points | worst spike | worst example",
        [
            f"{c['system']} | {c['op_file']} | {c['backend']} | `{c['kernel_source']}` | {c['sweep_col']} | "
            f"{c['points']} | {c['worst_ratio']:.1f}x | "
            f"{_fmt_shape(c['example']['fixed_shape'])}: {c['example']['sweep_col']}={c['example']['sweep_at']}, "
            f"lat {c['example']['latency']:.4g} vs neighbors {c['example']['neighbor_latency']:.4g}"
            for c in v["spike_clusters"]
        ],
        note=(
            "A point higher than BOTH sweep neighbors — a bad measurement (jitter, "
            "preemption, missing warmup) needing no reference framework."
        ),
        top=max_rows,
    )
    _md_section(
        lines,
        "Machine-fingerprint deviations",
        "system | backend | op_file | direction | rel ratio | ops compared",
        [
            f"{a['system']} | {a['backend']}/{a['version']} | {a['op_file']} | {a['direction']} | "
            f"{a['rel_ratio']:.2f}x | {a['ops_compared']}"
            for a in v["machine"]
        ],
        note=(
            "The system's latency level on this op is off relative to how the same system "
            "compares to its peers on its OTHER ops (hardware factor removed). Version skew "
            "and compute/memory hardware ratios add spread — triage before acting."
        ),
        top=max_rows,
    )
    _md_section(
        lines,
        "Unsupported schemas (not audited)",
        "system | op_file | backend | error",
        [f"{a['system']} | {a['op_file']} | {a['backend']}/{a['version']} | {a['error']}" for a in v["unsupported"]],
        note=(
            "No recognized latency column or component set — add the schema to "
            "_SINGLE_LATENCY_COLUMNS / _LATENCY_COMPONENT_SETS to bring these tables under audit."
        ),
        top=max_rows,
    )
    _md_section(
        lines,
        "Skipped comparisons (shape schema mismatch)",
        "system | op_file | pair | shape_cols_a | shape_cols_b",
        [
            f"{m['system']} | {m['op_file']} | {m['pair']} | "
            f"{', '.join(m['shape_cols_a'])} | {', '.join(m['shape_cols_b'])}"
            for m in v["mismatches"]
        ],
        top=max_rows,
    )

    lines.append("## Layer 2 — framework gaps (informational)\n")
    lines.append(
        "Ratios are `latency_b / latency_a` for the pair `b vs a`; a ratio < 1 means "
        "backend b is faster. Different kernel_source values on the two sides mean the "
        "gap likely reflects a kernel implementation difference, not bad data.\n"
    )
    _md_section(
        lines,
        "Systematic cross-system offsets",
        "op_file | slow backend | reference | systems | median offset | attribution | per-system | slow kernels",
        [
            f"{o['op_file']} | **{o['slow_backend']}** | {o['reference_backend']} | {len(o['systems'])} | "
            f"{o['overall_median_ratio']:.2f}x | {o['attribution']} | "
            + ", ".join(f"{k}:{r}" for k, r in sorted(o["median_ratio_by_system"].items()))
            + f" | `{', '.join(o['kernel_sources_slow'])}`"
            for o in offsets
        ],
        note=(
            "The same backend pair shows a same-direction median offset on EVERY covered "
            "system. A shape-local kernel weakness moves with shape and hardware; a uniform "
            "multiplier reproduced across systems is the fingerprint of a framework-level "
            "collection/configuration difference (eager mode, missing torch.compile, "
            "disabled CUDA graphs)."
        ),
        top=max_rows,
    )
    _md_section(
        lines,
        "Within-framework kernel choice cost",
        "system | op_file | backend | kernel_source | median penalty | max | rows",
        [
            f"{c['system']} | {c['op_file']} | {c['backend']}/{c['version']} | `{c['kernel_source']}` | "
            f"{c['median_penalty']:.2f}x | {c['max_penalty']:.1f}x | {c['rows']}"
            for c in sorted(v["kernel_costs"], key=lambda x: -x["median_penalty"])
        ],
        note=(
            "For tables that measured several kernel_sources on the same shapes: how much "
            "slower each kernel's median is than the per-shape best — the price of that "
            "backend selection inside one framework."
        ),
        top=max_rows,
    )
    _md_section(
        lines,
        "gemm speed-of-light efficiency (environment health)",
        "system | backend | dtype | points | median eff | p95 eff",
        [
            f"{e['system']} | {e['backend']}/{e['version']} | {e['gemm_dtype']} | {e['points']} | "
            f"{e['median_efficiency']:.2f} | {e['p95_efficiency']:.2f}"
            for e in sorted(v["sol_effs"], key=lambda x: (x["system"], x["backend"], x["gemm_dtype"]))
        ],
        note=(
            "Median achieved fraction of the physical bound per (system, backend, dtype). "
            "A system whose efficiencies sit well below its peers on the same hardware "
            "generation points at a collection-environment problem, not at the kernels."
        ),
        top=max_rows * 2,
    )
    _md_section(
        lines,
        "Pair summaries",
        "system | op_file | pair | points | median | p05 | p95 | region/total buckets | kernels_a | kernels_b",
        [
            f"{s['system']} | {s['op_file']} | {s['pair']} | {s['joined_points']} | {s['median_ratio']:.2f} | "
            f"{s['p05_ratio']:.2f} | {s['p95_ratio']:.2f} | "
            f"{s['region_deviation_buckets']}/{s['total_buckets']} | "
            f"`{', '.join(s['kernel_sources_a'])}` | `{', '.join(s['kernel_sources_b'])}`"
            for s in sorted(v["summaries"], key=lambda x: (x["system"], x["op_file"], x["pair"]))
        ],
    )

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
    parser.add_argument(
        "--offset-dispersion",
        type=float,
        default=2.0,
        help="Systematic offsets require per-system medians within this factor of each other "
        "(a same-direction but wildly varying gap is not a configuration-level offset).",
    )
    parser.add_argument("--max-report-rows", type=int, default=50, help="Row cap per markdown table.")
    parser.add_argument("--out-md", type=Path, default=None, help="Write the Markdown report.")
    parser.add_argument("--out-json", type=Path, default=None, help="Write all findings as JSON.")
    parser.add_argument(
        "--fail-on-anomalies",
        action="store_true",
        help="Exit nonzero when any gated kind's tally exceeds its threshold (CI gate mode).",
    )
    parser.add_argument(
        "--gate-threshold",
        action="append",
        default=[],
        metavar="KIND=N",
        help=f"Override a gate allowance, repeatable (defaults: {_GATE_THRESHOLDS}).",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Ratchet mode: known-finding fingerprints; the gate tallies only findings NOT in this file.",
    )
    parser.add_argument(
        "--write-baseline",
        type=Path,
        default=None,
        help="Write the current gated findings' fingerprints to this file (the ratchet snapshot).",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    if args.anomaly_factor <= 1.0:
        parser.error("--anomaly-factor must be > 1")
    if not 0.0 < args.mono_tolerance <= 1.0:
        parser.error("--mono-tolerance must be in (0, 1]")
    if args.offset_factor <= 1.0:
        parser.error("--offset-factor must be > 1")
    if args.offset_dispersion <= 1.0:
        parser.error("--offset-dispersion must be > 1")
    if args.spike_factor <= 1.0:
        parser.error("--spike-factor must be > 1")
    gate_thresholds = dict(_GATE_THRESHOLDS)
    for spec in args.gate_threshold:
        kind, _, n = spec.partition("=")
        if kind not in gate_thresholds or not n.isdigit():
            parser.error(f"--gate-threshold expects KIND=N with KIND in {sorted(gate_thresholds)}")
        gate_thresholds[kind] = int(n)

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
    offsets = detect_systematic_offsets(gaps, args.offset_factor, args.min_offset_systems, args.offset_dispersion, kmap)
    v = derive_views(anomalies, gaps)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(
                {
                    "anomalies": anomalies,
                    "gaps": gaps,
                    "suspect_clusters": suspects,
                    "undetermined_clusters": undetermined,
                    "mono_clusters": v["mono_clusters"],
                    "spike_clusters": v["spike_clusters"],
                    "systematic_offsets": offsets,
                }
            )
        )
        logger.info("wrote %s", args.out_json)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_markdown(v, offsets, suspects, undetermined, args.max_report_rows))
        logger.info("wrote %s", args.out_md)

    print(
        f"\nLayer 1: {sum(a['rows'] for a in v['nonpositive'])} nonpositive-latency rows, "
        f"{v['n_outliers']} point outliers ({len(suspects)} corroborated suspects, "
        f"{len(undetermined)} undetermined), {len(v['regions'])} region deviations, "
        f"{v['n_mono']} mono violations in {len(v['mono_clusters'])} clusters, "
        f"{v['n_spike']} spikes in {len(v['spike_clusters'])} clusters, "
        f"{sum(a['points'] for a in v['below_sol'])} below-SOL points in {len(v['below_sol'])} groups"
    )
    for line in _problem_lines(synthesize_problems(v, offsets, suspects, top=10)):
        print("  " + line.replace("**", ""))
    for a in sorted(v["below_sol"], key=lambda x: x["worst_fraction_of_sol"])[:5]:
        print(
            f"  BELOW-SOL! {a['system']}/{a['op_file']}: {a['backend']}/{a['version']} {a['gemm_dtype']} — "
            f"{a['points']} points, worst at {a['worst_fraction_of_sol']:.2f}x of physical bound "
            f"({_fmt_shape(a['example_shape'])}: {a['example_latency']:.4g} vs SOL {a['example_sol']:.4g})"
        )
    for a in v["machine"][:5]:
        print(
            f"  MACHINE? {a['system']} {a['backend']}/{a['version']} {a['op_file']}: "
            f"{a['rel_ratio']:.2f}x {a['direction']} vs the system's own level on "
            f"{a['ops_compared']} other ops"
        )
    print(
        f"Layer 2: {len(offsets)} systematic offsets across {len(v['summaries'])} backend pairs"
        + (f"; {len(v['mismatches'])} pairs skipped (schema mismatch)" if v["mismatches"] else "")
    )
    for o in offsets[: min(args.max_report_rows, 10)]:
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

    gated = [a for a in anomalies if a["kind"] in gate_thresholds]
    if args.write_baseline:
        args.write_baseline.parent.mkdir(parents=True, exist_ok=True)
        args.write_baseline.write_text(json.dumps(snapshot_baseline(gated)))
        print(f"\nBaseline written: {len(gated)} findings -> {args.write_baseline}")

    if args.fail_on_anomalies:
        baseline = None
        if args.baseline:
            if args.baseline.exists():
                baseline = json.loads(args.baseline.read_text())
                if baseline.get("schema") != _BASELINE_SCHEMA:
                    parser.error(f"baseline schema {baseline.get('schema')} != {_BASELINE_SCHEMA}; regenerate it")
            else:
                logger.warning("baseline %s not found; gating ALL findings", args.baseline)
        breaches, tallies, suppressed = evaluate_gate(gated, gate_thresholds, baseline)
        if baseline:
            print(f"\nBaseline: {suppressed} known findings suppressed; {sum(tallies.values())} new")
        if breaches:
            print(
                "GATE FAILED — new-finding tallies over allowance: "
                + ", ".join(f"{k}: {n} (allowed {gate_thresholds[k]})" for k, n in sorted(breaches.items()))
            )
            raise SystemExit(1)
        print("GATE OK — all tallies within allowance")


if __name__ == "__main__":
    main()

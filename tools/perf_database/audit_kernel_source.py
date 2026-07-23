# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Audit cross-framework shareability of perf-database perf tables.

For every (system, op_file, kernel_source) triple, compute:
  - which (framework, version) pairs contributed rows
  - how many shape keys overlap between frameworks
  - latency divergence stats across overlapping shapes
  - within-framework cross-version row duplication (dedup target)

Tier classification:
  - `shared`           : kernel_source is a non-empty named kernel
                         (not "default")
  - `shared_fallback`  : kernel_source == "default" (framework-implicit,
                         low-fidelity placeholder)

Rows whose kernel_source is blank/`<unknown>` are skipped during audit (and
logged) — they have no name to key inheritance on, and the current corpus has
zero such rows.

The script emits three artifacts:
  - JSON of per-group raw stats
  - Markdown audit table
  - YAML manifest consumed by the SDK loader

Usage:
    python3 tools/perf_database/audit_kernel_source.py \\
        --data-root src/aiconfigurator/systems/data \\
        --out-json $TMPDIR/op-kernel-sources.json \\
        --out-md   docs/perf_database/op-kernel-sources.md \\
        --out-manifest src/aiconfigurator/systems/op_kernel_source_manifest.yaml

The manifest lives under src/aiconfigurator/systems/ so the SDK loader
(src/aiconfigurator/sdk/perf_database.py) reads it as package data and decides,
per (op_file, kernel_source), which sibling backend/version directories the
active backend may inherit from. No perf data is moved or rewritten on disk.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)

# Columns that never participate in the shape key.
_META_COLUMNS = {"framework", "version", "device", "op_name", "kernel_source"}

# Latency-like columns. The first one found in the header is used as the metric.
_LATENCY_COLUMNS_PRIORITY = (
    "latency",
    "avg_ms",
    "combine_avg_t_us",
    "dispatch_avg_t_us",
)

# Files to skip entirely (markers, already-shared layers, irregular formats).
# reuse.yaml/collection_meta.yaml (Collector V3 structured markers) never match
# the *.parquet/*.txt glob in _iter_data_files below, so listing them here is
# defensive/documentation-only, not currently load-bearing.
_SKIP_FILE_BASENAMES = {"INCOMPLETE.txt", "reuse.yaml", "collection_meta.yaml"}

# Backend directory names to skip — these are framework-agnostic by construction.
_SKIP_BACKEND_DIRS = {"nccl", "oneccl"}

# Legacy top-level backend dirs. Family-first layout (Collector V3) treats any
# other first-level directory under a system dir as a family dir containing
# <backend>/<version> subtrees. Keep this set textually identical to the
# CANONICAL _KNOWN_BACKEND_DIRS in
# aic-core/src/aiconfigurator_core/sdk/operations/base.py minus
# _SKIP_BACKEND_DIRS (a deliberate 3-entry variant: consumer backends only,
# no comm pseudo-backends; base.py lists every copy that must stay in sync).
_LEGACY_BACKEND_DIRS = {"trtllm", "sglang", "vllm"}


def classify_tier(kernel_source: str) -> str:
    """Classify a kernel_source string into one of the two tiers.

    Caller must filter out blank/`<unknown>` kernel_source rows before calling —
    they have no name to key inheritance on and aren't represented in any tier.
    """
    if kernel_source == "default":
        return "shared_fallback"
    return "shared"


@dataclass
class GroupStats:
    """Stats for a single (system, op_file, kernel_source) triple."""

    system: str
    op_file: str
    kernel_source: str
    # (framework, version) -> row count
    rows_by_fw_version: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    # shape_key -> {framework: latency}; latency is the LAST seen value (versions overwrite).
    latency_by_shape_framework: dict[tuple, dict[str, float]] = field(default_factory=dict)
    # shape_key -> {(framework, version): latency} for cross-version dedup analysis.
    latency_by_shape_fw_version: dict[tuple, dict[tuple[str, str], float]] = field(default_factory=dict)
    total_raw_rows: int = 0

    def record_row(self, framework: str, version: str, shape_key: tuple, latency: float) -> None:
        self.rows_by_fw_version[(framework, version)] += 1
        self.latency_by_shape_framework.setdefault(shape_key, {})[framework] = latency
        self.latency_by_shape_fw_version.setdefault(shape_key, {})[(framework, version)] = latency
        self.total_raw_rows += 1

    @property
    def frameworks(self) -> set[str]:
        return {fw for fw, _ in self.rows_by_fw_version}

    @property
    def rows_by_framework(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for (fw, _ver), n in self.rows_by_fw_version.items():
            out[fw] += n
        return dict(out)

    def _within_framework_dedup_count(self) -> int:
        """How many rows are *exact* duplicates within the same framework across versions.

        For each (shape_key, framework), if multiple versions recorded the same
        latency, all but one are dedup-able. If they recorded different latencies,
        none are dedup-able (the version dimension is meaningful for that shape).
        """
        dedup_count = 0
        for by_fw_ver in self.latency_by_shape_fw_version.values():
            per_fw_latencies: dict[str, list[float]] = defaultdict(list)
            for (fw, _ver), lat in by_fw_ver.items():
                per_fw_latencies[fw].append(lat)
            for latencies in per_fw_latencies.values():
                if len(latencies) <= 1:
                    continue
                # Count exact duplicates only (within float tolerance).
                seen: list[float] = []
                for lat in latencies:
                    if any(abs(lat - s) < 1e-9 for s in seen):
                        dedup_count += 1
                    else:
                        seen.append(lat)
        return dedup_count

    def summary(self) -> dict:
        """Compute divergence + dedup summary for this group."""
        overlap_pct_diffs: list[float] = []
        overlapping_shapes = 0
        for by_fw in self.latency_by_shape_framework.values():
            if len(by_fw) < 2:
                continue
            overlapping_shapes += 1
            latencies = list(by_fw.values())
            mean = sum(latencies) / len(latencies)
            if mean <= 0:
                continue
            spread = (max(latencies) - min(latencies)) / mean * 100.0
            overlap_pct_diffs.append(spread)

        median_div = statistics.median(overlap_pct_diffs) if overlap_pct_diffs else None
        p95_div = (
            statistics.quantiles(overlap_pct_diffs, n=20)[-1]
            if len(overlap_pct_diffs) >= 20
            else (max(overlap_pct_diffs) if overlap_pct_diffs else None)
        )

        return {
            "system": self.system,
            "op_file": self.op_file,
            "kernel_source": self.kernel_source,
            "tier": classify_tier(self.kernel_source),
            "frameworks": sorted(self.frameworks),
            "rows_by_framework": self.rows_by_framework,
            "total_raw_rows": self.total_raw_rows,
            "total_shape_keys": len(self.latency_by_shape_framework),
            "overlapping_shape_keys": overlapping_shapes,
            "within_framework_dedup_rows": self._within_framework_dedup_count(),
            "median_pct_divergence": median_div,
            "p95_pct_divergence": p95_div,
            "max_pct_divergence": max(overlap_pct_diffs) if overlap_pct_diffs else None,
        }


def _pick_latency_column(header: list[str]) -> str | None:
    for candidate in _LATENCY_COLUMNS_PRIORITY:
        if candidate in header:
            return candidate
    return None


def _build_shape_key(row: dict[str, str], header: list[str], latency_col: str) -> tuple:
    keys = [c for c in header if c not in _META_COLUMNS and c != latency_col]
    return tuple((c, row.get(c, "")) for c in keys)


def _iter_backend_dirs(system_dir: Path) -> Iterable[tuple[str, Path]]:
    """Yield (backend, backend_path) for every backend dir under a system dir,
    across both the legacy (<backend>/<version>) and family-first
    (<family>/<backend>/<version>) layouts. `_SKIP_BACKEND_DIRS` entries are
    excluded at whichever level they appear (top-level or inside a family dir).
    """
    for entry in sorted(system_dir.iterdir()):
        if not entry.is_dir() or entry.name in _SKIP_BACKEND_DIRS:
            continue
        if entry.name in _LEGACY_BACKEND_DIRS:
            yield entry.name, entry
        else:  # family dir
            for backend_dir in sorted(entry.iterdir()):
                if not backend_dir.is_dir() or backend_dir.name in _SKIP_BACKEND_DIRS:
                    continue
                yield backend_dir.name, backend_dir


def _iter_data_files(data_root: Path) -> Iterable[tuple[str, str, str, Path]]:
    """Yield (system, backend, version, path) for every perf data table, across
    both the legacy and family-first (Collector V3) tree layouts."""
    for system_dir in sorted(data_root.iterdir()):
        if not system_dir.is_dir():
            continue
        for backend, backend_dir in _iter_backend_dirs(system_dir):
            for version_dir in sorted(backend_dir.iterdir()):
                if not version_dir.is_dir():
                    continue
                paths = sorted([*version_dir.glob("*.parquet"), *version_dir.glob("*.txt")])
                for path in paths:
                    if path.name in _SKIP_FILE_BASENAMES:
                        continue
                    yield system_dir.name, backend, version_dir.name, path


@dataclass
class _FileAuditResult:
    """Per-file output of `_audit_one_file`. Records are flat so the merge
    step is just a loop over `record_row` calls — same as the serial path."""

    # (system, op_file, kernel_source, framework, version, shape_key, latency)
    records: list[tuple[str, str, str, str, str, tuple, float]] = field(default_factory=list)
    rows_scanned: int = 0
    rows_skipped: int = 0
    rows_unnamed_kernel_source: int = 0
    unnamed_examples: list[tuple[str, str]] = field(default_factory=list)


def _read_perf_rows(path: Path) -> tuple[list[str], list[dict]]:
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        header = table.schema.names
        rows = [{key: "" if value is None else value for key, value in row.items()} for row in table.to_pylist()]
        return header, rows

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def _audit_one_file(system: str, backend: str, version: str, path: Path) -> _FileAuditResult:
    """Parse one perf data file and emit the records it contributes. Pure-ish: only the
    `logger.warning` for missing latency columns escapes."""
    out = _FileAuditResult()
    header, rows = _read_perf_rows(path)
    latency_col = _pick_latency_column(header)
    if latency_col is None:
        logger.warning("No latency column found in %s; skipping", path)
        return out

    for row in rows:
        out.rows_scanned += 1
        framework = (row.get("framework") or backend).lower()
        row_version = row.get("version") or version
        raw_ks = row.get("kernel_source")
        kernel_source = (raw_ks or "").strip()
        if not kernel_source:
            out.rows_unnamed_kernel_source += 1
            if len(out.unnamed_examples) < 5:
                out.unnamed_examples.append((str(path), repr(raw_ks)))
            continue
        latency_raw = row.get(latency_col)
        try:
            latency = float(latency_raw) if latency_raw not in (None, "") else None
        except ValueError:
            latency = None
        if latency is None or latency <= 0:
            out.rows_skipped += 1
            continue

        shape_key = _build_shape_key(row, header, latency_col)
        out.records.append((system, path.name, kernel_source, framework, row_version, shape_key, latency))
    return out


_AUDIT_THREADS = 4


def audit(data_root: Path) -> dict[tuple[str, str, str], GroupStats]:
    """Walk the data tree and accumulate per-group stats.

    Files are parsed in a `_AUDIT_THREADS`-wide thread pool; the merge into
    `groups` runs serially on the main thread, which is cheap enough not to
    need locks.
    """
    groups: dict[tuple[str, str, str], GroupStats] = {}
    rows_scanned = 0
    rows_skipped = 0
    rows_unnamed_kernel_source = 0
    unnamed_examples: list[tuple[str, str]] = []  # (path, kernel_source-as-seen)

    # Materialize the file list up front so the progress bar can show a total
    # and the walk doesn't appear stuck on slow disks. The list is small
    # (~hundreds).
    file_list = list(_iter_data_files(data_root))
    total_files = len(file_list)

    with ThreadPoolExecutor(max_workers=_AUDIT_THREADS) as pool:
        futures = [pool.submit(_audit_one_file, *args) for args in file_list]
        pbar = tqdm(
            as_completed(futures),
            desc=f"audit {data_root}",
            unit="file",
            total=total_files,
        )
        for fut in pbar:
            res = fut.result()
            rows_scanned += res.rows_scanned
            rows_skipped += res.rows_skipped
            rows_unnamed_kernel_source += res.rows_unnamed_kernel_source
            for path_str, raw_ks in res.unnamed_examples:
                if len(unnamed_examples) < 5:
                    unnamed_examples.append((path_str, raw_ks))
            for system, op_file, kernel_source, framework, row_version, shape_key, latency in res.records:
                key = (system, op_file, kernel_source)
                group = groups.get(key)
                if group is None:
                    group = GroupStats(system=system, op_file=op_file, kernel_source=kernel_source)
                    groups[key] = group
                group.record_row(framework, row_version, shape_key, latency)
            pbar.set_postfix(rows=rows_scanned, groups=len(groups))

    logger.info(
        "audit: %d files, %d rows, %d skipped, %d groups",
        total_files,
        rows_scanned,
        rows_skipped,
        len(groups),
    )
    if rows_unnamed_kernel_source:
        # Unexpected with the current corpus: every row should carry a named or
        # 'default' kernel_source. If this fires, decide whether to backfill or
        # add a new tier — don't silently drop more rows.
        logger.warning(
            "Skipped %d rows with blank/unknown kernel_source. Examples: %s",
            rows_unnamed_kernel_source,
            unnamed_examples,
        )
    return groups


def render_markdown(summaries: list[dict]) -> str:
    """Render a per-(op_file) markdown table of tier classification + divergence stats."""
    by_op: dict[str, list[dict]] = defaultdict(list)
    for s in summaries:
        by_op[s["op_file"]].append(s)

    # Top-level totals.
    total_rows = sum(s["total_raw_rows"] for s in summaries)
    dedup_rows = sum(s["within_framework_dedup_rows"] for s in summaries)
    tier_counts: dict[str, int] = defaultdict(int)
    tier_rows: dict[str, int] = defaultdict(int)
    for s in summaries:
        tier_counts[s["tier"]] += 1
        tier_rows[s["tier"]] += s["total_raw_rows"]

    lines: list[str] = ["# Shareability audit\n"]
    lines.append(
        "Classifies every `(system, op_file, kernel_source)` triple in the perf database into one of two tiers:\n"
    )
    lines.append(
        "- **`shared`** — named kernel_source. The SDK loader inherits these rows from sibling backend/version "
        "directories (cross-version and cross-backend) when the database is loaded in HYBRID mode.\n"
        "- **`shared_fallback`** — `kernel_source = default`. Framework-implicit, low-fidelity. "
        "Inherited alongside `shared` rows in HYBRID mode (HYBRID already accepts coarser fallbacks).\n"
        "\n"
        "Rows with a blank/`<unknown>` kernel_source are skipped during audit (the current corpus has none).\n"
    )

    lines.append("## Headline numbers\n")
    lines.append(f"- Total rows scanned: **{total_rows:,}**")
    pct = dedup_rows / max(total_rows, 1) * 100
    lines.append(f"- Within-framework cross-version dedup-able rows: **{dedup_rows:,}** (~{pct:.1f}%)")
    lines.append("- Tier distribution (groups / rows):")
    for tier in ("shared", "shared_fallback"):
        lines.append(f"  - `{tier}`: {tier_counts[tier]} groups · {tier_rows[tier]:,} rows")
    lines.append("")

    for op_file in sorted(by_op):
        lines.append(f"\n## `{op_file}`\n")
        lines.append(
            "| system | kernel_source | tier | frameworks | rows_per_fw | overlap_keys | "
            "dedup rows | median % | p95 % | max % |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        rows = sorted(by_op[op_file], key=lambda s: (s["system"], s["kernel_source"]))
        for s in rows:
            rows_per_fw = ", ".join(f"{fw}:{n}" for fw, n in sorted(s["rows_by_framework"].items()))
            frameworks = ", ".join(s["frameworks"])

            def _fmt(val: float | None) -> str:
                return "—" if val is None else f"{val:.1f}"

            lines.append(
                f"| {s['system']} | `{s['kernel_source']}` | {s['tier']} | {frameworks} | "
                f"{rows_per_fw} | {s['overlapping_shape_keys']} / {s['total_shape_keys']} | "
                f"{s['within_framework_dedup_rows']} | "
                f"{_fmt(s['median_pct_divergence'])} | "
                f"{_fmt(s['p95_pct_divergence'])} | "
                f"{_fmt(s['max_pct_divergence'])} |"
            )

    # Appendix: every distinct kernel_source seen in the corpus, grouped by tier.
    # Aggregated across (system, op_file) so a kernel_source that shows up in
    # multiple ops/systems appears once with the union of its frameworks.
    lines.append("\n## Appendix: all kernel sources\n")
    lines.append(
        "Each row is one distinct `kernel_source` value seen in the corpus, with the union of "
        "frameworks, op files, and systems it appears in. Tier is determined by the kernel_source "
        "name alone, so a single kernel_source has one tier across the whole corpus.\n"
    )
    by_ks: dict[str, dict] = {}
    for s in summaries:
        ks = s["kernel_source"]
        agg = by_ks.setdefault(
            ks,
            {
                "kernel_source": ks,
                "tier": s["tier"],
                "frameworks": set(),
                "op_files": set(),
                "systems": set(),
                "total_raw_rows": 0,
            },
        )
        agg["frameworks"].update(s["frameworks"])
        agg["op_files"].add(s["op_file"])
        agg["systems"].add(s["system"])
        agg["total_raw_rows"] += s["total_raw_rows"]

    for tier in ("shared", "shared_fallback"):
        entries = sorted(
            (a for a in by_ks.values() if a["tier"] == tier),
            key=lambda a: a["kernel_source"].lower(),
        )
        if not entries:
            continue
        lines.append(f"\n### `{tier}` ({len(entries)} kernel sources)\n")
        lines.append("| kernel_source | frameworks | op files | systems | rows |")
        lines.append("|---|---|---|---|---|")
        for a in entries:
            lines.append(
                f"| `{a['kernel_source']}` | {', '.join(sorted(a['frameworks']))} | "
                f"{', '.join(sorted(a['op_files']))} | {', '.join(sorted(a['systems']))} | "
                f"{a['total_raw_rows']:,} |"
            )
    return "\n".join(lines) + "\n"


def render_manifest(summaries: list[dict]) -> str:
    """Render the YAML manifest consumed by the SDK loader.

    Aggregates per-(op_file, kernel_source) across systems — same kernel_source
    on different systems shares a tier.
    """
    by_pair: dict[tuple[str, str], dict] = {}
    for s in summaries:
        key = (s["op_file"], s["kernel_source"])
        agg = by_pair.setdefault(
            key,
            {
                "op_file": s["op_file"],
                "kernel_source": s["kernel_source"],
                "tier": s["tier"],
                "systems": set(),
                "frameworks": set(),
                "total_raw_rows": 0,
                "within_framework_dedup_rows": 0,
                "median_pct_divergence": [],
            },
        )
        agg["systems"].add(s["system"])
        agg["frameworks"].update(s["frameworks"])
        agg["total_raw_rows"] += s["total_raw_rows"]
        agg["within_framework_dedup_rows"] += s["within_framework_dedup_rows"]
        if s["median_pct_divergence"] is not None:
            agg["median_pct_divergence"].append(s["median_pct_divergence"])

    lines: list[str] = [
        "# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        "# Perf-database op-kernel-source manifest.",
        "#",
        "# What this is:",
        "#   The runtime contract for cross-backend / cross-version measurement reuse.",
        "#   For every distinct (op_file, kernel_source) pair seen across the data tree,",
        "#   it records which backends produce rows under that kernel and which tier",
        "#   the kernel belongs to ('shared' or 'shared_fallback').",
        "#",
        "# How it's used:",
        "#   src/aiconfigurator/sdk/perf_database.py reads this file at PerfDatabase",
        "#   construction time. When the database is loaded in HYBRID mode, the loader",
        "#   walks sibling <sys>/<framework>/<version>/<op_file> directories and inherits",
        "#   rows tagged with the kernel_sources this manifest declares the active backend",
        "#   may consume — keeping the active backend's own rows on key conflict. Both",
        "#   `tier=shared` and `tier=shared_fallback` (kernel_source=default,",
        "#   framework-implicit) rows are inherited; HYBRID already accepts coarser",
        "#   fallbacks, so they are not gated separately.",
        "#",
        "# How to regenerate:",
        "#   Generated by tools/perf_database/audit_kernel_source.py from the data tree —",
        "#   do not hand-edit. Re-run whenever a perf table under src/aiconfigurator/systems/data/",
        "#   is added, removed, or has its kernel_source values changed:",
        "#",
        "#     python3 tools/perf_database/audit_kernel_source.py \\",
        "#         --data-root src/aiconfigurator/systems/data \\",
        "#         --out-manifest src/aiconfigurator/systems/op_kernel_source_manifest.yaml",
        "#",
        "# Schema (per group):",
        "#   op_file:                     perf table basename, e.g. gemm_perf.parquet",
        "#   kernel_source:               kernel name as it appears in the perf table's kernel_source column",
        "#   tier:                        'shared' (named, high-fidelity) or",
        "#                                'shared_fallback' (default, framework-implicit)",
        "#   systems:                     systems where rows for this (op_file, kernel_source) exist",
        "#   frameworks:                  backends that produce rows (= the inheritance whitelist)",
        "#   total_raw_rows:              row count across all systems and frameworks",
        "#   within_framework_dedup_rows: rows that duplicate prior versions within the same framework",
        "#   median_pct_divergence:       optional; cross-framework latency spread for overlapping shapes",
        "groups:",
    ]
    for (_op, _ks), agg in sorted(by_pair.items()):
        med = statistics.median(agg["median_pct_divergence"]) if agg["median_pct_divergence"] else None
        lines.append(f"  - op_file: {agg['op_file']}")
        lines.append(f"    kernel_source: {agg['kernel_source']!r}")
        lines.append(f"    tier: {agg['tier']}")
        lines.append(f"    systems: [{', '.join(sorted(agg['systems']))}]")
        lines.append(f"    frameworks: [{', '.join(sorted(agg['frameworks']))}]")
        lines.append(f"    total_raw_rows: {agg['total_raw_rows']}")
        lines.append(f"    within_framework_dedup_rows: {agg['within_framework_dedup_rows']}")
        if med is not None:
            lines.append(f"    median_pct_divergence: {med:.2f}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("src/aiconfigurator/systems/data"),
        help="Root of the systems/data tree.",
    )
    parser.add_argument("--out-json", type=Path, default=None, help="Write per-group summaries as JSON.")
    parser.add_argument("--out-md", type=Path, default=None, help="Write a Markdown audit table.")
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=None,
        help="Write a YAML manifest consumed by the SDK loader.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")

    groups = audit(args.data_root)
    summaries = [g.summary() for g in groups.values()]

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summaries, indent=2, sort_keys=True))
        logger.info("wrote %s", args.out_json)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_markdown(summaries))
        logger.info("wrote %s", args.out_md)
    if args.out_manifest:
        args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.out_manifest.write_text(render_manifest(summaries))
        logger.info("wrote %s", args.out_manifest)

    # Console summary.
    tier_counts: dict[str, int] = defaultdict(int)
    tier_rows: dict[str, int] = defaultdict(int)
    for s in summaries:
        tier_counts[s["tier"]] += 1
        tier_rows[s["tier"]] += s["total_raw_rows"]
    total_rows = sum(s["total_raw_rows"] for s in summaries)
    dedup_rows = sum(s["within_framework_dedup_rows"] for s in summaries)
    print(f"\nTotal rows: {total_rows:,}")
    print(
        f"Within-framework cross-version dedup-able rows: {dedup_rows:,} ({dedup_rows / max(total_rows, 1) * 100:.1f}%)"
    )
    print("Tier distribution (groups / rows):")
    for tier in ("shared", "shared_fallback"):
        print(f"  {tier:18}{tier_counts[tier]:>5} groups · {tier_rows[tier]:>10,} rows")


if __name__ == "__main__":
    main()

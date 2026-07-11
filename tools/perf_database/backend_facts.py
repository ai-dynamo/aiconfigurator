# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Bootstrap and audit `collector/op_backend_facts.yaml`.

The registry records, for every (op table, framework, version, system,
fact-axis slice), which kernel backend(s) the op runs on — the vocabulary is
the `kernel_source` labels the collectors write. The registry itself is
collector-owned reference data: after the initial bootstrap it is maintained
deliberately (a collector-upgrade PR states the new version's backend defaults
by editing it), NOT regenerated from whatever a collection run produced.

This tool has exactly two jobs:

  - bootstrap (default): derive the registry from the perf database. Used once
    to seed the file, and afterwards only to draft entries for a new
    framework version/system from freshly collected data — the diff is then
    reviewed like any hand edit.
  - `--check`: audit the committed registry against the perf database labels.
    Drift means either the data is mislabeled or the registry is stale; both
    must be resolved explicitly. Exits non-zero on drift.

Fact axes are the columns kernel dispatch actually depends on: the op identity
inside a shared table (`op_name`, `phase`, `architecture`) plus the
precision/quant columns (`attn_dtype`, `kv_cache_dtype`, `moe_dtype`, ...).
Shape columns (batch/seq/heads/...) are deliberately excluded: a slice whose
backend flips with shape shows up with multiple kernel_sources and is a fact
to investigate, not to average away.

Usage:
    python3 tools/perf_database/backend_facts.py                # bootstrap/draft
    python3 tools/perf_database/backend_facts.py --check        # drift audit
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from audit_kernel_source import _iter_data_files

logger = logging.getLogger(__name__)

# Columns that identify the op / dispatch slice, in render order. Only the
# subset present in a table's header participates in that table's fact key.
# `op_name`, `phase`, `architecture` are identity axes (kernel_source is known
# to vary with them); the rest are the precision/quant axes.
_FACT_COLUMNS = (
    "op_name",
    "architecture",
    "phase",
    "attn_dtype",
    "kv_cache_dtype",
    "mla_dtype",
    "gemm_type",
    "gemm_dtype",
    "moe_dtype",
    "kernel_dtype",
    "bmm_dtype",
    "quant_dtype",
    "quant_type",
    "allreduce_dtype",
    "nccl_dtype",
)


@dataclass(frozen=True)
class FactKey:
    op_file: str  # table basename without extension, e.g. "gemm_perf"
    framework: str
    version: str
    system: str
    axis_values: tuple[tuple[str, str], ...]  # ((column, value), ...) in _FACT_COLUMNS order


def _scan_parquet(path: Path, axis_cols: list[str]) -> Counter:
    """Group one parquet table by (axis columns, kernel_source) → row count."""
    import pyarrow.parquet as pq

    cols = [*axis_cols, "kernel_source"]
    table = pq.read_table(path, columns=cols)
    grouped = table.group_by(cols).aggregate([([], "count_all")])
    counts: Counter = Counter()
    for row in grouped.to_pylist():
        key = tuple(str(row[c]) if row[c] is not None else "" for c in cols)
        counts[key] += row["count_all"]
    return counts


def _scan_txt(path: Path, axis_cols: list[str]) -> Counter:
    counts: Counter = Counter()
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = tuple(row.get(c) or "" for c in [*axis_cols, "kernel_source"])
            counts[key] += 1
    return counts


def _read_header(path: Path) -> list[str]:
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_schema(path).names
    with path.open(newline="") as f:
        return next(csv.reader(f), [])


def scan(data_root: Path) -> tuple[dict[FactKey, set[str]], dict[str, list[str]]]:
    """Walk the data tree and accumulate observed kernel_sources per fact key.

    Returns (facts, axes_by_op): `facts` maps FactKey -> {kernel_source, ...};
    `axes_by_op` maps op_file -> the union of fact columns seen in its headers.
    """
    facts: dict[FactKey, set[str]] = defaultdict(set)
    axes_by_op: dict[str, list[str]] = {}
    skipped_no_kernel_source: list[str] = []

    for system, backend, version, path in _iter_data_files(data_root):
        header = _read_header(path)
        if "kernel_source" not in header:
            skipped_no_kernel_source.append(str(path))
            continue
        axis_cols = [c for c in _FACT_COLUMNS if c in header]
        op_file = path.name.rsplit(".", 1)[0]
        seen = axes_by_op.setdefault(op_file, [])
        for c in axis_cols:
            if c not in seen:
                seen.append(c)

        counts = _scan_parquet(path, axis_cols) if path.suffix.lower() == ".parquet" else _scan_txt(path, axis_cols)
        for key in counts:
            *axis_vals, kernel_source = key
            kernel_source = kernel_source.strip() or "<unnamed>"
            fact_key = FactKey(
                op_file=op_file,
                framework=backend,
                version=version,
                system=system,
                axis_values=tuple(zip(axis_cols, axis_vals, strict=True)),
            )
            facts[fact_key].add(kernel_source)

    # Keep axis order canonical per op even when headers differ across files.
    for op_file, cols in axes_by_op.items():
        axes_by_op[op_file] = [c for c in _FACT_COLUMNS if c in cols]
    if skipped_no_kernel_source:
        logger.warning(
            "Skipped %d files without a kernel_source column: %s",
            len(skipped_no_kernel_source),
            skipped_no_kernel_source[:5],
        )
    return dict(facts), axes_by_op


def _fact_entries(facts: dict[FactKey, set[str]], axes_by_op: dict[str, list[str]]) -> dict[str, list[dict]]:
    """Reshape into the per-op sorted entry lists the registry stores."""
    by_op: dict[str, list[dict]] = defaultdict(list)
    for key, sources in facts.items():
        axis_map = dict(key.axis_values)
        entry = {
            "framework": key.framework,
            "version": key.version,
            "system": key.system,
            **{c: axis_map.get(c, "") for c in axes_by_op[key.op_file]},
            "kernel_sources": sorted(sources),
        }
        by_op[key.op_file].append(entry)
    for op_file, entries in by_op.items():
        entries.sort(key=lambda e: tuple(str(e[c]) for c in ("framework", "version", "system", *axes_by_op[op_file])))
    return dict(sorted(by_op.items()))


def _yaml_scalar(value) -> str:
    return json.dumps(value)  # JSON string quoting is valid YAML


def render_yaml(by_op: dict[str, list[dict]], axes_by_op: dict[str, list[str]]) -> str:
    lines = [
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        "# Op-backend facts registry: for every (op table, framework, version,",
        "# system, fact-axis slice), the kernel backend(s) the op runs on, in the",
        "# vocabulary of the `kernel_source` labels the collectors write.",
        "#",
        "# Layer contract (see .claude/rules/collector/layer_permissions.md):",
        "# this is REFERENCE data for the collection harness — backend expectations",
        "# to pin and validate against. It never gates whether a case runs and it",
        "# must not grow match/skip rules.",
        "#",
        "# Ownership: bootstrapped from the perf database by",
        "# tools/perf_database/backend_facts.py (2026-07-11). From then on it is",
        "# maintained deliberately: a collector-upgrade PR that changes backend",
        "# dispatch updates the affected entries (the tool can draft them from",
        "# freshly collected data; review the diff like a hand edit). Audit with:",
        "#",
        "#   python3 tools/perf_database/backend_facts.py --check",
        "#",
        "# Entry semantics:",
        "#   kernel_sources: [one]      - the backend the op runs on for this slice",
        "#   kernel_sources: [several]  - multi-backend sweep (e.g. MoE per-quant",
        "#                                families, eager+graph) or a shape-gated",
        "#                                dispatch boundary inside the slice",
        "#   kernel_sources: [default]  - the collector did not name the kernel",
        "#                                (framework-implicit; low fidelity)",
        "schema_version: 1",
        "ops:",
    ]
    for op_file, entries in by_op.items():
        axes = axes_by_op[op_file]
        lines.append(f"  - op_file: {op_file}")
        lines.append(f"    axes: [{', '.join(axes)}]")
        lines.append("    facts:")
        for e in entries:
            ks = ", ".join(_yaml_scalar(k) for k in e["kernel_sources"])
            fields = [f"{c}: {_yaml_scalar(e[c])}" for c in ("framework", "version", "system", *axes)]
            lines.append(f"      - {{{', '.join(fields)}, kernel_sources: [{ks}]}}")
    return "\n".join(lines) + "\n"


def _flatten(doc: dict) -> dict[tuple, tuple[str, ...]]:
    """Flatten a registry document into {slice key: kernel_sources} for diffing."""
    flat: dict[tuple, tuple[str, ...]] = {}
    for op in doc["ops"]:
        axes = op["axes"]
        for e in op["facts"]:
            key = (op["op_file"], e["framework"], str(e["version"]), e["system"]) + tuple(
                str(e.get(c, "")) for c in axes
            )
            flat[key] = tuple(e["kernel_sources"])
    return flat


def check(registry_path: Path, by_op: dict[str, list[dict]], axes_by_op: dict[str, list[str]]) -> list[str]:
    """Compare the committed registry against facts derived from the data tree.

    Returns human-readable drift lines (empty = in sync).
    """
    import yaml

    committed = _flatten(yaml.safe_load(registry_path.read_text()))
    derived = _flatten(
        {"ops": [{"op_file": op, "axes": axes_by_op[op], "facts": entries} for op, entries in by_op.items()]}
    )

    drift: list[str] = []
    for key in sorted(derived.keys() - committed.keys(), key=str):
        drift.append(f"data-not-in-registry: {key} -> {list(derived[key])}")
    for key in sorted(committed.keys() - derived.keys(), key=str):
        drift.append(f"registry-not-in-data: {key} -> {list(committed[key])}")
    for key in sorted(derived.keys() & committed.keys(), key=str):
        if derived[key] != committed[key]:
            drift.append(f"mismatch: {key} registry={list(committed[key])} data={list(derived[key])}")
    return drift


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("src/aiconfigurator/systems/data"))
    parser.add_argument("--registry", type=Path, default=Path("collector/op_backend_facts.yaml"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="Audit the committed registry against the perf database instead of writing it.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")

    facts, axes_by_op = scan(args.data_root)
    by_op = _fact_entries(facts, axes_by_op)
    total = sum(len(v) for v in by_op.values())

    if args.check:
        drift = check(args.registry, by_op, axes_by_op)
        if drift:
            for line in drift[:50]:
                print(line)
            if len(drift) > 50:
                print(f"... and {len(drift) - 50} more")
            print(f"\nDRIFT: {len(drift)} differences between {args.registry} and {args.data_root}")
            sys.exit(1)
        print(f"OK: {args.registry} matches {args.data_root} ({total:,} fact slices)")
        return

    args.registry.parent.mkdir(parents=True, exist_ok=True)
    args.registry.write_text(render_yaml(by_op, axes_by_op))
    logger.info("wrote %s", args.registry)
    print(f"Fact slices: {total:,}")


if __name__ == "__main__":
    main()

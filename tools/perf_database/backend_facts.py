# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Derive the op-backend facts record from the perf database.

For every (op table, framework, version, system, fact-axis slice) observed in
the data tree, record which kernel backend(s) the rows were collected on — the
`kernel_source` column. Fact axes are the columns kernel dispatch actually
depends on: the op identity inside a shared table (`op_name`, `phase`,
`architecture`) plus the precision/quant columns (`attn_dtype`,
`kv_cache_dtype`, `moe_dtype`, ...). Shape columns (batch/seq/heads/...) are
deliberately excluded: a slice whose backend flips with shape shows up as
`multi` and is a fact to investigate, not to average away.

This is a *record of observations*, not a claim about framework internals:
`kernel_source` is only as truthful as the collector that wrote it. SGLang
collectors pin and label the serving-default backend per SM; vLLM defers to
vLLM's own selector; TRT-LLM writes coarse static tags (`torch_flow`,
`default`) because its unified backend dispatches internally. Slices whose
only label is `default` are flagged `default_only` (framework-implicit,
low-fidelity) rather than presented as a named backend.

Statuses:
  - `single`       : exactly one named kernel_source — the observed default.
  - `multi`        : more than one kernel_source — either an intentional
                     multi-backend sweep (e.g. MoE per-quant backend families,
                     WideEP fa3+flashinfer) or a shape-dependent dispatch
                     boundary. Row counts per kernel_source are kept.
  - `default_only` : only the placeholder `default` label — the collector did
                     not name the kernel.

Usage (regenerate the checked-in record after any data change):
    python3 tools/perf_database/backend_facts.py \\
        --data-root src/aiconfigurator/systems/data \\
        --out-yaml docs/perf_database/op-backend-facts.yaml \\
        --out-md   docs/perf_database/op-backend-facts.md
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


def classify_status(kernel_sources: dict[str, int]) -> str:
    named = [ks for ks in kernel_sources if ks != "default"]
    if not named:
        return "default_only"
    if len(kernel_sources) == 1:
        return "single"
    return "multi"


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


def scan(data_root: Path) -> tuple[dict[FactKey, dict[str, int]], dict[str, list[str]]]:
    """Walk the data tree and accumulate kernel_source counts per fact key.

    Returns (facts, axes_by_op): `facts` maps FactKey -> {kernel_source: rows};
    `axes_by_op` maps op_file -> the union of fact columns seen in its headers.
    """
    facts: dict[FactKey, Counter] = defaultdict(Counter)
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
        for key, n in counts.items():
            *axis_vals, kernel_source = key
            kernel_source = kernel_source.strip()
            if not kernel_source:
                kernel_source = "<unnamed>"
            fact_key = FactKey(
                op_file=op_file,
                framework=backend,
                version=version,
                system=system,
                axis_values=tuple(zip(axis_cols, axis_vals, strict=True)),
            )
            facts[fact_key][kernel_source] += n

    # Keep axis order canonical per op even when headers differ across files.
    for op_file, cols in axes_by_op.items():
        axes_by_op[op_file] = [c for c in _FACT_COLUMNS if c in cols]
    if skipped_no_kernel_source:
        logger.warning(
            "Skipped %d files without a kernel_source column: %s",
            len(skipped_no_kernel_source),
            skipped_no_kernel_source[:5],
        )
    return {k: dict(v) for k, v in facts.items()}, axes_by_op


def _fact_entries(facts: dict[FactKey, dict[str, int]], axes_by_op: dict[str, list[str]]) -> dict[str, list[dict]]:
    """Reshape into per-op sorted entry lists shared by both renderers."""
    by_op: dict[str, list[dict]] = defaultdict(list)
    for key, sources in facts.items():
        axis_map = dict(key.axis_values)
        entry = {
            "framework": key.framework,
            "version": key.version,
            "system": key.system,
            **{c: axis_map.get(c, "") for c in axes_by_op[key.op_file]},
            "kernel_sources": dict(sorted(sources.items())),
            "status": classify_status(sources),
        }
        by_op[key.op_file].append(entry)
    for op_file, entries in by_op.items():
        entries.sort(key=lambda e: tuple(str(e[c]) for c in ("framework", "version", "system", *axes_by_op[op_file])))
    return dict(sorted(by_op.items()))


def _yaml_scalar(value) -> str:
    if isinstance(value, int):
        return str(value)
    return json.dumps(value)  # JSON string quoting is valid YAML


def render_yaml(by_op: dict[str, list[dict]], axes_by_op: dict[str, list[str]]) -> str:
    lines = [
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        "# Op-backend facts: for every (op table, framework, version, system,",
        "# fact-axis slice) in the perf database, the kernel backend(s) the rows",
        "# were collected on (`kernel_source`) with row counts, and a status:",
        "#   single       - one named backend (the observed default)",
        "#   multi        - multiple backends in the slice (multi-backend sweep or",
        "#                  shape-dependent dispatch; see per-backend row counts)",
        "#   default_only - collector wrote only the placeholder 'default' label",
        "#",
        "# Generated by tools/perf_database/backend_facts.py from the data tree -",
        "# do not hand-edit. Regenerate after any perf-data change:",
        "#",
        "#   python3 tools/perf_database/backend_facts.py",
        "#",
        "schema_version: 1",
        "ops:",
    ]
    for op_file, entries in by_op.items():
        axes = axes_by_op[op_file]
        lines.append(f"  - op_file: {op_file}")
        lines.append(f"    axes: [{', '.join(axes)}]")
        lines.append("    facts:")
        for e in entries:
            ks = ", ".join(f"{_yaml_scalar(k)}: {n}" for k, n in e["kernel_sources"].items())
            fields = [f"{c}: {_yaml_scalar(e[c])}" for c in ("framework", "version", "system", *axes)]
            lines.append(f"      - {{{', '.join(fields)}, kernel_sources: {{{ks}}}, status: {e['status']}}}")
    return "\n".join(lines) + "\n"


def render_markdown(by_op: dict[str, list[dict]], axes_by_op: dict[str, list[str]]) -> str:
    total = sum(len(v) for v in by_op.values())
    status_counts: Counter = Counter(e["status"] for entries in by_op.values() for e in entries)

    lines = [
        "# Op-backend facts\n",
        "For every `(op table, framework, version, system, fact-axis slice)` in the perf",
        "database: which kernel backend(s) (`kernel_source`) the rows were collected on.",
        "Generated by `tools/perf_database/backend_facts.py` — do not hand-edit; see the",
        "tool docstring for status semantics and caveats (TRT-LLM labels are coarse).\n",
        "## Headline numbers\n",
        f"- Fact slices: **{total:,}**",
    ]
    for status in ("single", "multi", "default_only"):
        lines.append(f"  - `{status}`: {status_counts.get(status, 0):,}")
    lines.append("")

    for op_file, entries in by_op.items():
        axes = axes_by_op[op_file]
        lines.append(f"\n## `{op_file}`\n")
        lines.append(f"| framework | version | system | {' | '.join(axes)} | kernel_source(s) | status |")
        lines.append("|---" * (5 + len(axes)) + "|")
        for e in entries:
            sources = e["kernel_sources"]
            if len(sources) == 1:
                ks = f"`{next(iter(sources))}`"
            else:
                ks = ", ".join(f"`{k}` ({n})" for k, n in sources.items())
            axis_cells = " | ".join(str(e[c]) for c in axes)
            lines.append(f"| {e['framework']} | {e['version']} | {e['system']} | {axis_cells} | {ks} | {e['status']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("src/aiconfigurator/systems/data"))
    parser.add_argument("--out-yaml", type=Path, default=Path("docs/perf_database/op-backend-facts.yaml"))
    parser.add_argument("--out-md", type=Path, default=Path("docs/perf_database/op-backend-facts.md"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")

    facts, axes_by_op = scan(args.data_root)
    by_op = _fact_entries(facts, axes_by_op)

    args.out_yaml.parent.mkdir(parents=True, exist_ok=True)
    args.out_yaml.write_text(render_yaml(by_op, axes_by_op))
    logger.info("wrote %s", args.out_yaml)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(by_op, axes_by_op))
    logger.info("wrote %s", args.out_md)

    status_counts: Counter = Counter(e["status"] for entries in by_op.values() for e in entries)
    total = sum(len(v) for v in by_op.values())
    print(f"\nFact slices: {total:,}")
    for status in ("single", "multi", "default_only"):
        print(f"  {status:14}{status_counts.get(status, 0):>7,}")


if __name__ == "__main__":
    main()

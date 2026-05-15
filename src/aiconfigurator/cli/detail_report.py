# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from aiconfigurator.sdk.inference_summary import InferenceSummary

if TYPE_CHECKING:
    from aiconfigurator.cli.api import EstimateResult

_DETAIL_SECTIONS = ("summary", "memory", "time", "energy", "source")


def _parse_detail(detail: str | set[str] | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(detail, str):
        tokens = [t.strip() for t in detail.split(",") if t.strip()]
    else:
        tokens = [str(t).strip() for t in detail if str(t).strip()]
    if not tokens:
        tokens = ["summary"]

    expanded: set[str] = set()
    for tok in tokens:
        if tok == "all":
            expanded.update(_DETAIL_SECTIONS)
        elif tok in _DETAIL_SECTIONS:
            expanded.add(tok)
        else:
            raise ValueError(f"Unknown detail section {tok!r}. Allowed: {', '.join(_DETAIL_SECTIONS)}, all.")
    return [section for section in _DETAIL_SECTIONS if section in expanded]


def detail_requests_time(detail: str | set[str] | list[str] | tuple[str, ...]) -> bool:
    """Return whether a detail selector includes the time section."""
    return "time" in _parse_detail(detail)


def format_estimate_detail(
    result: EstimateResult,
    sol_result: EstimateResult | None = None,
    *,
    detail: str | set[str] | list[str] | tuple[str, ...] = "summary",
    width: int = 80,
    top_n_ops: int = 12,
) -> str:
    """Format detail sections for any estimate mode, with optional SOL comparison."""
    sections = _parse_detail(detail)
    out: list[str] = []

    for section in sections:
        section_lines: list[str] = []
        if section == "time":
            section_lines = _format_time_detail(result, sol_result, width=width, top_n_ops=top_n_ops)
        elif result.summary is not None:
            section_lines = result.summary.format_detail_report(
                detail=section,
                width=width,
                top_n_ops=top_n_ops,
            ).splitlines()
        elif section == "summary":
            section_lines = _format_raw_summary(result)

        if not section_lines:
            continue
        if out:
            out.append("")
        out.extend(section_lines)

    return "\n".join(out)


def _format_raw_summary(result: EstimateResult) -> list[str]:
    raw = result.raw or {}
    lines = ["Performance Summary"]
    for label, key in (
        ("ttft", "ttft"),
        ("tpot", "tpot"),
        ("request latency", "request_latency"),
        ("throughput", "tokens/s"),
        ("seq/s", "seq/s"),
    ):
        value = raw.get(key)
        if value is None:
            continue
        unit = " ms" if key in {"ttft", "tpot", "request_latency"} else ""
        lines.append(f"  {label:<16s} {float(value):>12.3f}{unit}")
    return lines


def _format_time_detail(
    result: EstimateResult,
    sol_result: EstimateResult | None,
    *,
    width: int,
    top_n_ops: int,
) -> list[str]:
    lines: list[str] = []
    metric_lines = _format_latency_metrics(result, sol_result)
    if metric_lines:
        lines.extend(metric_lines)

    scheduling = (result.per_ops_data or {}).get("scheduling")
    if scheduling:
        if lines:
            lines.append("")
        num_mix = float(scheduling.get("num_mix_steps", 0.0) or 0.0)
        num_genonly = float(scheduling.get("num_genonly_steps", 0.0) or 0.0)
        lines.append(f"Scheduling: {num_mix:.0f} mix steps + {num_genonly:.0f} gen-only steps")

    sections = _time_sections(result, sol_result)
    for title, ops, sol_ops, sources in sections:
        if lines:
            lines.append("")
        lines.append(_section_header(title, ops, sol_ops))
        lines.extend(
            _format_op_rows(
                ops,
                sol_ops=sol_ops,
                source_dict=sources,
                top_n=top_n_ops,
                bar_width=max(12, width - 68) if sol_ops else max(16, width - 40),
            )
        )

    if not lines:
        return ["Time Breakdown", "  <no latency data>"]
    return lines


def _format_latency_metrics(result: EstimateResult, sol_result: EstimateResult | None) -> list[str]:
    raw = result.raw or {}
    sol_raw = sol_result.raw if sol_result is not None else {}
    rows: list[str] = []
    for label, key in (
        ("ttft", "ttft"),
        ("tpot", "tpot"),
        ("request latency", "request_latency"),
    ):
        latency = float(raw.get(key, 0.0) or 0.0)
        sol_latency = float(sol_raw.get(key, 0.0) or 0.0) if sol_raw else 0.0
        if latency <= 0.0 and sol_latency <= 0.0:
            continue
        rows.append(_format_latency_row(label, latency, sol_latency if sol_raw else None))
    return ["Latency Summary", *rows] if rows else []


def _format_latency_row(label: str, latency: float, sol_latency: float | None) -> str:
    if sol_latency is None:
        return f"  {label:<16s} {latency:>10.3f} ms"
    sol_pct = sol_latency / latency * 100.0 if latency > 0.0 else 0.0
    return f"  {label:<16s} {latency:>10.3f} ms  SOL {sol_latency:>10.3f} ms  {sol_pct:>6.1f}% SOL/time"


def _time_sections(
    result: EstimateResult,
    sol_result: EstimateResult | None,
) -> list[tuple[str, dict[str, float], dict[str, float] | None, dict[str, str] | None]]:
    if result.summary is not None:
        static_sections = _static_time_sections(result, sol_result)
        if static_sections:
            return static_sections

    per_ops_data = result.per_ops_data or {}
    sol_per_ops_data = sol_result.per_ops_data if sol_result is not None else None
    per_ops_source = result.per_ops_source or {}
    sections: list[tuple[str, dict[str, float], dict[str, float] | None, dict[str, str] | None]] = []

    for key, title in (
        ("mix_step", "Mix Step"),
        ("genonly_step", "Gen-Only Step"),
        ("prefill", "Prefill (static_ctx)"),
        ("decode", "Decode (static_gen)"),
    ):
        ops = per_ops_data.get(key)
        if not ops:
            continue
        sol_ops = sol_per_ops_data.get(key) if sol_per_ops_data else None
        sources = per_ops_source.get(key) if per_ops_source else None
        sections.append((title, dict(ops), dict(sol_ops) if sol_ops else None, dict(sources) if sources else None))

    return sections


def _static_time_sections(
    result: EstimateResult,
    sol_result: EstimateResult | None,
) -> list[tuple[str, dict[str, float], dict[str, float] | None, dict[str, str] | None]]:
    summary = result.summary
    if summary is None:
        return []

    sol_summary = sol_result.summary if sol_result is not None else None
    sections: list[tuple[str, dict[str, float], dict[str, float] | None, dict[str, str] | None]] = []

    ctx = summary.get_context_latency_dict()
    if ctx:
        sol_ctx = sol_summary.get_context_latency_dict() if sol_summary is not None else None
        sections.append(
            (
                "Context phase",
                dict(ctx),
                dict(sol_ctx) if sol_ctx else None,
                dict(summary.get_context_source_dict() or {}),
            )
        )

    gen = summary.get_generation_latency_dict()
    if gen:
        sol_gen = sol_summary.get_generation_latency_dict() if sol_summary is not None else None
        sections.append(
            (
                "Generation phase",
                dict(gen),
                dict(sol_gen) if sol_gen else None,
                dict(summary.get_generation_source_dict() or {}),
            )
        )

    return sections


def _section_header(title: str, ops: dict[str, float], sol_ops: dict[str, float] | None) -> str:
    total = sum(float(v) for v in ops.values())
    header = f"{title} (total = {total:.3f} ms"
    if sol_ops:
        sol_total = sum(float(v) for v in sol_ops.values())
        sol_pct = sol_total / total * 100.0 if total > 0.0 else 0.0
        header += f", SOL = {sol_total:.3f} ms, SOL/time = {sol_pct:.1f}%"
    header += ")"
    return header


def _format_op_rows(
    latency_dict: dict[str, float],
    *,
    sol_ops: dict[str, float] | None,
    source_dict: dict[str, str] | None,
    top_n: int,
    bar_width: int,
) -> list[str]:
    items = [(op, float(lat)) for op, lat in latency_dict.items() if float(lat) > 0.0]
    if not items:
        return ["  <no measurable per-op data>"]

    total = sum(lat for _, lat in items)
    if total <= 0.0:
        return ["  <no measurable per-op data>"]

    shown = items[:top_n]
    rest = items[top_n:]
    name_w = min(32, max(len(op) for op, _ in shown))

    lines = [_format_op_row(op, lat, total, name_w, bar_width, sol_ops, source_dict) for op, lat in shown]

    if rest:
        rest_name = f"... (others, {len(rest)} items)"
        rest_lat = sum(lat for _, lat in rest)
        rest_sol = sum(float(sol_ops.get(op, 0.0) or 0.0) for op, _ in rest) if sol_ops else None
        lines.append(_format_op_row(rest_name, rest_lat, total, name_w, bar_width, None, None, sol_latency=rest_sol))
    return lines


def _format_op_row(
    op: str,
    latency: float,
    total: float,
    name_w: int,
    bar_width: int,
    sol_ops: dict[str, float] | None,
    source_dict: dict[str, str] | None,
    *,
    sol_latency: float | None = None,
) -> str:
    pct = latency / total * 100.0 if total > 0.0 else 0.0
    bar = InferenceSummary._ascii_bar(latency / total if total > 0.0 else 0.0, width=bar_width)
    sol_suffix = ""
    if sol_ops is not None or sol_latency is not None:
        sol_value = sol_latency if sol_latency is not None else float(sol_ops.get(op, 0.0) or 0.0)
        sol_pct = sol_value / latency * 100.0 if latency > 0.0 else 0.0
        sol_suffix = f"  SOL {sol_value:>10.3f} ms  {sol_pct:>6.1f}% SOL/time"
    src_suffix = ""
    if source_dict:
        src = source_dict.get(op)
        if src:
            src_suffix = f" [{src}]"
    return f"  {op:<{name_w}s}  {latency:>10.3f} ms  {pct:>5.1f}%{sol_suffix}  {bar}{src_suffix}"

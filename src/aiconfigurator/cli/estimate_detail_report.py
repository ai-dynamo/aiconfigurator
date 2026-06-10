# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI-side detail report formatting for estimate results.

The formatter intentionally lives outside InferenceSummary so report rendering
stays a CLI post-processing concern. It reads InferenceSummary internals here
because the summary object is the data boundary for this report.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiconfigurator.cli.api import EstimateResult
    from aiconfigurator.sdk.inference_summary import InferenceSummary

DetailSelector = str | set[str] | list[str] | tuple[str, ...]

DETAIL_SECTIONS = ("summary", "memory", "time", "energy", "source")


def parse_detail_sections(detail: DetailSelector) -> list[str]:
    if isinstance(detail, str):
        tokens = [t.strip() for t in detail.split(",") if t.strip()]
    else:
        tokens = [str(t).strip() for t in detail if str(t).strip()]
    if not tokens:
        tokens = ["summary"]

    expanded: set[str] = set()
    for tok in tokens:
        if tok == "all":
            expanded.update(DETAIL_SECTIONS)
        elif tok in DETAIL_SECTIONS:
            expanded.add(tok)
        else:
            raise ValueError(f"Unknown detail section {tok!r}. Allowed: {', '.join(DETAIL_SECTIONS)}, all.")
    return [section for section in DETAIL_SECTIONS if section in expanded]


def detail_requests_time(detail: DetailSelector) -> bool:
    """Return whether a detail selector includes the time section."""
    return "time" in parse_detail_sections(detail)


def format_summary_detail_report(
    summary: InferenceSummary,
    *,
    detail: DetailSelector = "summary",
    width: int = 80,
    top_n_ops: int = 12,
) -> str:
    """Format detail sections from an InferenceSummary as CLI post-processing."""
    sections = parse_detail_sections(detail)
    bar_width = max(16, width - 40)

    out: list[str] = []
    for section in sections:
        if out:
            out.append("")
        if section == "summary":
            out.extend(_format_summary_section(summary))
        elif section == "memory":
            out.extend(_format_memory_section(summary, bar_width=bar_width))
        elif section == "time":
            out.extend(_format_summary_time_section(summary, bar_width=bar_width, top_n=top_n_ops))
        elif section == "energy":
            out.extend(_format_energy_section(summary, bar_width=bar_width, top_n=top_n_ops))
        elif section == "source":
            out.extend(_format_source_section(summary))
    return "\n".join(out)


def format_estimate_detail_report(
    result: EstimateResult,
    sol_result: EstimateResult | None = None,
    *,
    detail: DetailSelector = "summary",
    width: int = 80,
    top_n_ops: int = 12,
) -> str:
    """Format detail sections for any estimate mode, with optional SOL comparison."""
    sections = parse_detail_sections(detail)
    out: list[str] = []

    for section in sections:
        section_lines: list[str] = []
        if section == "time":
            section_lines = _format_time_detail(result, sol_result, width=width, top_n_ops=top_n_ops)
        elif result.summary is not None and (
            section == "memory" or result.mode in ("static", "static_ctx", "static_gen")
        ):
            section_lines = format_summary_detail_report(
                result.summary,
                detail=section,
                width=width,
                top_n_ops=top_n_ops,
            ).splitlines()
        elif section == "source":
            section_lines = _format_raw_source_from_per_ops(result.per_ops_source or {})
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
        ("encoder latency", "encoder_latency"),
        ("encoder memory", "encoder_memory"),
        ("encoder memory", "(e)memory"),
        ("throughput", "tokens/s"),
        ("seq/s", "seq/s"),
    ):
        value = raw.get(key)
        if value is None:
            continue
        if key in {"encoder_latency", "encoder_memory", "(e)memory"} and float(value) <= 0.0:
            continue
        if key in {"ttft", "tpot", "request_latency", "encoder_latency"}:
            unit = " ms"
        elif key in {"encoder_memory", "(e)memory"}:
            unit = " GB"
        else:
            unit = ""
        lines.append(f"  {label:<16s} {float(value):>12.3f}{unit}")
    return lines


def _ascii_bar(fraction: float, width: int = 40) -> str:
    frac = max(0.0, min(1.0, float(fraction)))
    filled = round(frac * width)
    return "█" * filled + "░" * (width - filled)


def _format_op_bars(
    latency_dict: dict[str, float],
    *,
    top_n: int,
    bar_width: int,
    unit: str = "ms",
    source_dict: dict[str, str] | None = None,
) -> list[str]:
    if not latency_dict:
        return ["  <no data>"]

    items = [(op, float(lat)) for op, lat in latency_dict.items() if lat > 0.0]
    if not items:
        return ["  <no measurable per-op data>"]

    total = sum(lat for _, lat in items)
    if total <= 0.0:
        return ["  <no measurable per-op data>"]

    shown = items[:top_n]
    rest = items[top_n:]
    name_w = min(32, max(len(op) for op, _ in shown))
    lines: list[str] = []
    for op, lat in shown:
        pct = lat / total * 100.0
        bar = _ascii_bar(lat / total, width=bar_width)
        src_tag = ""
        if source_dict is not None:
            src = source_dict.get(op)
            if src:
                src_tag = f" [{src}]"
        lines.append(f"  {op:<{name_w}s}  {lat:>10.3f} {unit}  {pct:>5.1f}%  {bar}{src_tag}")
    if rest:
        rest_lat = sum(lat for _, lat in rest)
        rest_pct = rest_lat / total * 100.0
        bar = _ascii_bar(rest_lat / total, width=bar_width)
        lines.append(
            f"  {'... (others, ' + str(len(rest)) + ' items)':<{name_w}s}  "
            f"{rest_lat:>10.3f} {unit}  {rest_pct:>5.1f}%  {bar}"
        )
    return lines


def _format_summary_section(summary: InferenceSummary) -> list[str]:
    enc_ms = sum(summary._encoder_latency_dict.values())
    ctx_ms = sum(summary._context_latency_dict.values())
    gen_ms = sum(summary._generation_latency_dict.values())
    total_ms = enc_ms + ctx_ms + gen_ms
    rc = summary._runtime_config
    lines = ["Performance Summary"]
    lines.append(f"  total latency       {total_ms:>12.3f} ms")
    if enc_ms > 0:
        lines.append(f"  encoder            {enc_ms:>12.3f} ms")
    if ctx_ms > 0 and enc_ms > 0:
        lines.append(f"  context            {ctx_ms:>12.3f} ms")
        lines.append(f"  ttft               {enc_ms + ctx_ms:>12.3f} ms")
    elif ctx_ms > 0:
        lines.append(f"  context (TTFT)      {ctx_ms:>12.3f} ms")
    if gen_ms > 0:
        tpot = 0.0
        if rc is not None and rc.osl is not None and rc.osl > 1:
            tpot = gen_ms / (rc.osl - 1)
        lines.append(f"  generation          {gen_ms:>12.3f} ms")
        if tpot > 0:
            lines.append(f"  tpot                {tpot:>12.3f} ms")
    if summary._summary_df is not None and len(summary._summary_df) > 0:
        row = summary._summary_df.iloc[0]
        if "tokens/s" in row.index:
            lines.append(f"  throughput          {row['tokens/s']:>12.2f} tokens/s")
        if "seq/s" in row.index:
            lines.append(f"  throughput          {row['seq/s']:>12.3f} seq/s")
    if summary._is_oom:
        lines.append("  ⚠ OOM: estimated memory exceeds GPU capacity")
    elif summary._is_kv_cache_oom:
        lines.append("  ⚠ KV cache budget exceeded under free_gpu_memory_fraction")
    return lines


def _format_memory_section(summary: InferenceSummary, bar_width: int) -> list[str]:
    if not summary._memory:
        return ["Memory Layout", "  <no memory data>"]

    cap_gib: float | None = None
    if summary._mem_capacity_bytes:
        cap_gib = summary._mem_capacity_bytes / (1 << 30)

    lines: list[str] = []
    header = "Memory Layout"
    if cap_gib is not None:
        header += f" (capacity {cap_gib:.2f} GiB)"
    lines.append(header)

    order = ["weights", "kvcache", "activations", "nccl", "others"]
    breakdown_keys = [k for k in order if k in summary._memory and k != "total"]
    for k in summary._memory:
        if k != "total" and k not in breakdown_keys:
            breakdown_keys.append(k)

    total_gib = float(summary._memory.get("total", 0.0))
    denom = cap_gib if cap_gib is not None else max(total_gib, 1e-9)
    name_w = max((len(k) for k in breakdown_keys), default=10)

    for key in breakdown_keys:
        gib = float(summary._memory[key])
        frac = gib / denom if denom > 0 else 0.0
        lines.append(f"  {key:<{name_w}s}  {gib:>8.3f} GiB  {_ascii_bar(frac, width=bar_width)}  {frac * 100.0:>5.1f}%")

    lines.append("  " + "-" * (name_w + 8 + 3 + bar_width + 8))
    total_frac = total_gib / denom if denom > 0 else 0.0
    free_suffix = ""
    if cap_gib is not None:
        free_gib = max(0.0, cap_gib - total_gib)
        free_suffix = f"  (free {free_gib:.3f} GiB)"
    lines.append(
        f"  {'total':<{name_w}s}  {total_gib:>8.3f} GiB  "
        f"{_ascii_bar(total_frac, width=bar_width)}  {total_frac * 100.0:>5.1f}%{free_suffix}"
    )

    if summary._encoder_memory:
        lines.append("")
        lines.append("Encoder Memory (included in prefill worker)")
        enc_total = float(summary._encoder_memory.get("total", 0.0) or 0.0)
        enc_denom = max(enc_total, 1e-9)
        for key, value in summary._encoder_memory.items():
            gib = float(value)
            frac = gib / enc_denom if key != "total" else 1.0
            lines.append(
                f"  {key:<{name_w}s}  {gib:>8.3f} GiB  {_ascii_bar(frac, width=bar_width)}  {frac * 100.0:>5.1f}%"
            )

    if summary._kv_bytes_per_seq is not None and summary._kv_bytes_per_seq > 0:
        kv_per_seq_gib = summary._kv_bytes_per_seq / (1 << 30)
        seq_len_str = f" (seq_len={summary._kv_seq_len_used})" if summary._kv_seq_len_used is not None else ""
        lines.append("")
        lines.append(f"  kvcache/seq  {kv_per_seq_gib:>8.4f} GiB{seq_len_str}")
        if cap_gib is not None:
            kv_total_gib = float(summary._memory.get("kvcache", 0.0))
            non_kv_gib = total_gib - kv_total_gib
            free_for_kv_gib = max(0.0, cap_gib - non_kv_gib)
            if kv_per_seq_gib > 0:
                max_bs = int(free_for_kv_gib // kv_per_seq_gib)
                extra = ""
                if summary._free_gpu_memory_fraction is not None:
                    eff_frac = (
                        summary._free_gpu_memory_fraction
                        * (1.0 - summary._kv_cache_reserved_fraction)
                        * (1.0 - summary._kv_cache_tolerance)
                    )
                    max_bs_frac = int((free_for_kv_gib * eff_frac) // kv_per_seq_gib)
                    extra = f"  /  {max_bs_frac} (under free_gpu_memory_fraction={summary._free_gpu_memory_fraction:g})"
                lines.append(f"  max batch (KV-bound, same isl/osl) ≈ {max_bs}{extra}")
                lines.append("    note: ignores activation growth with batch; treat as an upper bound.")
    return lines


def _format_summary_time_section(summary: InferenceSummary, bar_width: int, top_n: int) -> list[str]:
    lines: list[str] = []
    enc_total = sum(summary._encoder_latency_dict.values())
    ctx_total = sum(summary._context_latency_dict.values())
    gen_total = sum(summary._generation_latency_dict.values())

    if enc_total > 0:
        lines.append(f"Encoder phase (total = {enc_total:.3f} ms)")
        lines.extend(
            _format_op_bars(
                summary._encoder_latency_dict,
                top_n=top_n,
                bar_width=bar_width,
                unit="ms",
                source_dict=summary._encoder_source_dict or None,
            )
        )
    if ctx_total > 0:
        if enc_total > 0:
            lines.append("")
        ctx_label = "TTFT(+encoder)" if enc_total > 0 else "TTFT"
        lines.append(f"Context phase ({ctx_label} = {ctx_total:.3f} ms)")
        lines.extend(
            _format_op_bars(
                summary._context_latency_dict,
                top_n=top_n,
                bar_width=bar_width,
                unit="ms",
                source_dict=summary._context_source_dict or None,
            )
        )
    if gen_total > 0:
        if enc_total > 0 or ctx_total > 0:
            lines.append("")
        lines.append(f"Generation phase (total = {gen_total:.3f} ms)")
        lines.extend(
            _format_op_bars(
                summary._generation_latency_dict,
                top_n=top_n,
                bar_width=bar_width,
                unit="ms",
                source_dict=summary._generation_source_dict or None,
            )
        )
    if enc_total == 0 and ctx_total == 0 and gen_total == 0:
        lines.append("Time Breakdown")
        lines.append("  <no per-op latency data>")
    return lines


def _format_energy_section(summary: InferenceSummary, bar_width: int, top_n: int) -> list[str]:
    enc = summary._encoder_energy_wms_dict or {}
    ctx = summary._context_energy_wms_dict or {}
    gen = summary._generation_energy_wms_dict or {}
    if not enc and not ctx and not gen:
        return ["Energy Breakdown", "  <no energy data>"]

    lines: list[str] = []
    if enc:
        total_enc = sum(enc.values())
        lines.append(f"Encoder energy (total = {total_enc:.3f} W·ms, avg P = {summary._encoder_power_avg:.1f} W)")
        lines.extend(_format_op_bars(enc, top_n=top_n, bar_width=bar_width, unit="W·ms"))
    if ctx:
        if enc:
            lines.append("")
        total_ctx = sum(ctx.values())
        lines.append(f"Context energy (total = {total_ctx:.3f} W·ms, avg P = {summary._context_power_avg:.1f} W)")
        lines.extend(_format_op_bars(ctx, top_n=top_n, bar_width=bar_width, unit="W·ms"))
    if gen:
        if enc or ctx:
            lines.append("")
        total_gen = sum(gen.values())
        lines.append(f"Generation energy (total = {total_gen:.3f} W·ms, avg P = {summary._generation_power_avg:.1f} W)")
        lines.extend(_format_op_bars(gen, top_n=top_n, bar_width=bar_width, unit="W·ms"))
    return lines


def _format_source_section(summary: InferenceSummary) -> list[str]:
    enc = summary._encoder_source_dict or {}
    ctx = summary._context_source_dict or {}
    gen = summary._generation_source_dict or {}
    if not enc and not ctx and not gen:
        return ["Data Source Breakdown", "  <no source data>"]

    lines = ["Data Source Breakdown (per-op)"]
    if enc:
        lines.append(f"  encoder     {_summarize_source_dict(enc)}")
        for op, src in sorted(enc.items()):
            lines.append(f"    {op:<30s} {src}")
    if ctx:
        lines.append(f"  context     {_summarize_source_dict(ctx)}")
        for op, src in sorted(ctx.items()):
            lines.append(f"    {op:<30s} {src}")
    if gen:
        lines.append(f"  generation  {_summarize_source_dict(gen)}")
        for op, src in sorted(gen.items()):
            lines.append(f"    {op:<30s} {src}")
    return lines


def _format_raw_source_from_per_ops(per_ops_source: dict) -> list[str]:
    if not per_ops_source:
        return ["Data Source Breakdown", "  <no source data>"]

    phase_titles = {
        "encoder": "Encoder (colocated prefill)",
        "mix_step": "Mix Step",
        "genonly_step": "Gen-Only Step",
        "prefill": "Prefill (static_ctx)",
        "decode": "Decode (static_gen)",
    }
    ordered_keys = [key for key in phase_titles if per_ops_source.get(key)]
    ordered_keys.extend(sorted(key for key in per_ops_source if key not in phase_titles and per_ops_source.get(key)))
    if not ordered_keys:
        return ["Data Source Breakdown", "  <no source data>"]

    lines = ["Data Source Breakdown (per-op)"]
    for key in ordered_keys:
        sources = dict(per_ops_source.get(key) or {})
        title = phase_titles.get(key, key)
        lines.append(f"  {title:<20s} {_summarize_source_dict(sources)}")
        for op, src in sorted(sources.items()):
            lines.append(f"    {op:<30s} {src}")
    return lines


def _summarize_source_dict(d: dict[str, str]) -> str:
    counts: dict[str, int] = {}
    for v in d.values():
        counts[v] = counts.get(v, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])
    return ", ".join(f"{k}={v}" for k, v in ordered)


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
    if not rows:
        return []
    header = f"  {'metric':<16s} {'latency':>13s}"
    if sol_raw:
        header += f"  {'SOL':>13s}  {'SOL%':>7s}"
    return ["Latency Summary", header, *rows]


def _format_latency_row(label: str, latency: float, sol_latency: float | None) -> str:
    if sol_latency is None:
        return f"  {label:<16s} {latency:>10.3f} ms"
    sol_pct = sol_latency / latency * 100.0 if latency > 0.0 else 0.0
    return f"  {label:<16s} {latency:>10.3f} ms  {sol_latency:>10.3f} ms  {sol_pct:>6.1f}%"


def _time_sections(
    result: EstimateResult,
    sol_result: EstimateResult | None,
) -> list[tuple[str, dict[str, float], dict[str, float] | None, dict[str, str] | None]]:
    if result.summary is not None and result.mode in ("static", "static_ctx", "static_gen"):
        static_sections = _static_time_sections(result, sol_result)
        if static_sections:
            return static_sections

    per_ops_data = result.per_ops_data or {}
    sol_per_ops_data = sol_result.per_ops_data if sol_result is not None else None
    per_ops_source = result.per_ops_source or {}
    sections: list[tuple[str, dict[str, float], dict[str, float] | None, dict[str, str] | None]] = []

    for key, title in (
        ("encoder", "Encoder (colocated prefill)"),
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

    enc = summary.get_encoder_latency_dict()
    if enc:
        sol_enc = sol_summary.get_encoder_latency_dict() if sol_summary is not None else None
        sections.append(
            (
                "Encoder phase",
                dict(enc),
                dict(sol_enc) if sol_enc else None,
                dict(summary.get_encoder_source_dict() or {}),
            )
        )

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
        header += f", SOL = {sol_total:.3f} ms, SOL% = {sol_pct:.1f}%"
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
    show_source = source_dict is not None

    header = f"  {'op':<{name_w}s}  {'latency':>13s}"
    if sol_ops:
        header += f"  {'SOL':>13s}  {'SOL%':>7s}"
    share_w = bar_width + 8
    header += f"  {'share (%)':<{share_w}s}"
    if show_source:
        header += "  source"
    lines = [
        header,
        *[
            _format_op_row(op, lat, total, name_w, bar_width, sol_ops, source_dict, show_source=show_source)
            for op, lat in shown
        ],
    ]

    if rest:
        rest_name = f"... (others, {len(rest)} items)"
        rest_lat = sum(lat for _, lat in rest)
        rest_sol = sum(float(sol_ops.get(op, 0.0) or 0.0) for op, _ in rest) if sol_ops else None
        lines.append(
            _format_op_row(
                rest_name,
                rest_lat,
                total,
                name_w,
                bar_width,
                None,
                source_dict,
                sol_latency=rest_sol,
                show_source=show_source,
            )
        )
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
    show_source: bool = False,
) -> str:
    pct = latency / total * 100.0 if total > 0.0 else 0.0
    bar = _ascii_bar(latency / total if total > 0.0 else 0.0, width=bar_width)
    share = f"{bar}  {pct:>5.1f}%"
    share_w = bar_width + 8
    sol_suffix = ""
    if sol_ops is not None or sol_latency is not None:
        sol_value = sol_latency if sol_latency is not None else float(sol_ops.get(op, 0.0) or 0.0)
        sol_pct = sol_value / latency * 100.0 if latency > 0.0 else 0.0
        sol_suffix = f"  {sol_value:>10.3f} ms  {sol_pct:>6.1f}%"
    src_suffix = ""
    if show_source:
        src = source_dict.get(op)
        src_suffix = f"  [{src}]" if src else "  "
    return f"  {op:<{name_w}s}  {latency:>10.3f} ms{sol_suffix}  {share:<{share_w}s}{src_suffix}"

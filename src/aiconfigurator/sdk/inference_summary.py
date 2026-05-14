# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pandas as pd

from aiconfigurator.sdk.config import RuntimeConfig

logger = logging.getLogger(__name__)


class InferenceSummary:
    """
    InferecneSummary to hold results of inference with energy tracking.

    Attributes:
        runtime_config: runtime config
        memory: memory breakdown
        context_latency_dict: latency breakdown for context (ms)
        generation_latency_dict: latency breakdown for generation (ms)
        context_energy_wms_dict: energy breakdown for context (W·ms)
        generation_energy_wms_dict: energy breakdown for generation (W·ms)
        summary_df: summary dataframe

    Units:
        - latency: milliseconds (ms)
        - energy: watt-milliseconds (W·ms) = millijoules (mJ)
        - power: watts (W) - derived from energy/latency

    Methods:
        set_memory_and_check_oom: set memory and check oom (+ optional kv cache budget check)
        set_oom: set oom
        check_oom: check oom
        set_kv_cache_oom: set kv cache oom
        check_kv_cache_oom: check kv cache oom
        set_context_latency_dict: set context latency dict
        set_generation_latency_dict: set generation latency dict
        get_context_latency_dict: get context latency dict
        get_generation_latency_dict: get generation latency dict
        set_context_energy_wms_dict: set context energy dict
        set_generation_energy_wms_dict: set generation energy dict
        get_context_energy_wms_dict: get context energy dict
        get_generation_energy_wms_dict: get generation energy dict
        set_kv_per_seq: stash per-sequence KV cache footprint context (for capacity probing)
        get_kv_per_seq: get per-sequence KV cache footprint context
        get_mem_capacity_bytes: get the GPU memory capacity captured by set_memory_and_check_oom
        get_static_info: legacy 4-tuple breakdown text used by webapp Static Tab
        format_detail_report: new flexible breakdown report (preferred for new call sites)
        set_summary_df: set summary dataframe
        get_summary_df: get summary dataframe
    """

    def __init__(self, runtime_config: RuntimeConfig) -> None:
        """
        Initialize inference summary.
        """
        self._runtime_config = runtime_config

        # raw data dict
        self._memory = {}
        self._context_latency_dict = {}  # ms
        self._generation_latency_dict = {}  # ms
        self._context_energy_wms_dict = {}  # RENAMED from _context_power_dict, W·ms
        self._generation_energy_wms_dict = {}  # RENAMED from _generation_power_dict, W·ms
        # Per-op data source ("silicon", "empirical", or "mixed") populated by
        # base_backend phase helpers from PerformanceResult.source.
        self._context_source_dict: dict[str, str] = {}
        self._generation_source_dict: dict[str, str] = {}
        self._is_oom = None
        self._is_kv_cache_oom = False

        # NEW: Store computed power averages
        self._context_power_avg = 0.0
        self._generation_power_avg = 0.0
        self._e2e_power_avg = 0.0

        # summary dataframe
        self._summary_df = None

        # cached result dict for efficient batch operations
        self._result_dict = None

        # per-ops latency breakdown (populated by run_agg or run_disagg)
        self._per_ops_data: dict | None = None
        # per-ops data source breakdown, parallel to _per_ops_data: same key
        # structure but values are "silicon" / "empirical" / "mixed" strings.
        self._per_ops_source: dict | None = None

        # Capacity probing context. Populated by set_memory_and_check_oom
        # (capacity) and by backends running static-mode estimation (kv per seq).
        # Used by format_detail_report() to compute capacity-% / headroom /
        # max-batch-size estimates. These are best-effort: agg/disagg flows
        # may leave them unset, in which case the report degrades gracefully.
        self._mem_capacity_bytes: int | None = None
        self._free_gpu_memory_fraction: float | None = None
        self._kv_cache_reserved_fraction: float = 0.0
        self._kv_cache_tolerance: float = 0.0
        self._kv_bytes_per_seq: float | None = None
        self._kv_seq_len_used: int | None = None

    def set_memory_and_check_oom(
        self,
        memory_dict: dict,
        mem_capacity: int,
        free_gpu_memory_fraction: float | None = None,
        kv_cache_reserved_fraction: float = 0.0,
        kv_cache_tolerance: float = 0.0,
    ) -> None:
        """
        Set memory and check oom.

        *memory_dict* should reflect the actual runtime memory layout
        (e.g. kvcache computed with ``max_seq_len``, activations with
        ``max_num_tokens``).

        When *free_gpu_memory_fraction* is not ``None``, also performs the
        KV cache budget check using the same *memory_dict*.
        """
        self._memory = memory_dict
        self._is_oom = self._memory["total"] >= (mem_capacity / (1 << 30))
        self._is_kv_cache_oom = False
        self._mem_capacity_bytes = mem_capacity
        self._free_gpu_memory_fraction = free_gpu_memory_fraction
        self._kv_cache_reserved_fraction = kv_cache_reserved_fraction
        self._kv_cache_tolerance = kv_cache_tolerance
        if free_gpu_memory_fraction is not None:
            self._check_and_set_kv_cache_oom(
                mem_capacity,
                free_gpu_memory_fraction,
                kv_cache_reserved_fraction,
                kv_cache_tolerance,
            )

    def _check_and_set_kv_cache_oom(
        self,
        mem_capacity: int,
        free_gpu_memory_fraction: float,
        kv_cache_reserved_fraction: float,
        kv_cache_tolerance: float,
    ) -> None:
        """Check whether the KV cache exceeds the fraction-based memory budget.

        Uses ``self._memory`` (set by :meth:`set_memory_and_check_oom`).

        Equivalent to the inflation formula
        ``kv / (frac*(1-res)*(1-tol)) + non_kv >= capacity`` rewritten as
        ``kv > (capacity - non_kv) * frac * (1-res) * (1-tol)``.
        """
        self._is_kv_cache_oom = False
        if self._is_oom:
            return
        mem_cap_gib = mem_capacity / (1 << 30)
        kv_gib = self._memory.get("kvcache", 0.0)
        non_kv_gib = self._memory["total"] - kv_gib
        kv_budget = (
            (mem_cap_gib - non_kv_gib)
            * free_gpu_memory_fraction
            * (1 - kv_cache_reserved_fraction)
            * (1 - kv_cache_tolerance)
        )
        self._is_kv_cache_oom = kv_gib > kv_budget

    def set_oom(self, is_oom: bool) -> None:
        """
        Set oom.
        """
        self._is_oom = is_oom

    def set_context_latency_dict(self, context_latency_dict: dict) -> None:
        """
        Set context latency dict.
        """
        self._context_latency_dict = context_latency_dict

    def set_generation_latency_dict(self, generation_latency_dict: dict) -> None:
        """
        Set generation latency dict.
        """
        self._generation_latency_dict = generation_latency_dict

    def get_context_latency_dict(self) -> dict:
        """
        Get context latency dict.
        """
        return self._context_latency_dict

    def get_generation_latency_dict(self) -> dict:
        """
        Get generation latency dict.
        """
        return self._generation_latency_dict

    # NEW: Energy dict accessors (explicit _wms naming for clarity)
    def set_context_energy_wms_dict(self, energy_wms_dict: dict[str, float]) -> None:
        """
        Set context energy dict (units: W·ms).

        Args:
            energy_wms_dict: Dict of operation -> energy in watt-milliseconds (W·ms).
                            Note: 1 W·ms = 1 millijoule (mJ).
        """
        self._context_energy_wms_dict = energy_wms_dict

    def set_generation_energy_wms_dict(self, energy_wms_dict: dict[str, float]) -> None:
        """
        Set generation energy dict (units: W·ms).

        Args:
            energy_wms_dict: Dict of operation -> energy in watt-milliseconds (W·ms).
        """
        self._generation_energy_wms_dict = energy_wms_dict

    def get_context_energy_wms_dict(self) -> dict[str, float]:
        """
        Returns dict of operation -> energy in watt-milliseconds (W·ms).

        Note: 1 W·ms = 1 millijoule (mJ). To convert to joules: divide by 1000.
        """
        return self._context_energy_wms_dict

    def get_generation_energy_wms_dict(self) -> dict[str, float]:
        """
        Returns dict of operation -> energy in watt-milliseconds (W·ms).
        """
        return self._generation_energy_wms_dict

    # Alias accessors (for less verbose code)
    def get_context_energy_dict(self) -> dict[str, float]:
        """Alias for get_context_energy_wms_dict() - returns energy in W·ms"""
        return self._context_energy_wms_dict

    def get_generation_energy_dict(self) -> dict[str, float]:
        """Alias for get_generation_energy_wms_dict() - returns energy in W·ms"""
        return self._generation_energy_wms_dict

    # NEW: Power average accessors
    def set_context_power_avg(self, power_avg: float) -> None:
        """Set context phase average power (watts)."""
        self._context_power_avg = power_avg

    def set_generation_power_avg(self, power_avg: float) -> None:
        """Set generation phase average power (watts)."""
        self._generation_power_avg = power_avg

    def set_e2e_power_avg(self, power_avg: float) -> None:
        """Set end-to-end average power (watts)."""
        self._e2e_power_avg = power_avg

    def get_context_power_avg(self) -> float:
        """Get context phase average power (watts)."""
        return self._context_power_avg

    def get_generation_power_avg(self) -> float:
        """Get generation phase average power (watts)."""
        return self._generation_power_avg

    def get_e2e_power_avg(self) -> float:
        """Get end-to-end average power (watts)."""
        return self._e2e_power_avg

    def has_sufficient_power_data(self, threshold: float = 0.9) -> bool:
        """
        Check if power data coverage is sufficient for reliable power estimation.

        Args:
            threshold: Minimum ratio of latency with non-zero energy to total latency (default 0.9)

        Returns:
            bool: True if latency with non-zero energy >= threshold * total latency
        """
        # Calculate total latency
        total_latency = sum(self._context_latency_dict.values()) + sum(self._generation_latency_dict.values())

        if total_latency == 0:
            return False

        # Calculate latency from operations with non-zero energy
        latency_with_energy = 0.0
        for op_name, latency in self._context_latency_dict.items():
            if self._context_energy_wms_dict.get(op_name, 0.0) > 0:
                latency_with_energy += latency

        for op_name, latency in self._generation_latency_dict.items():
            if self._generation_energy_wms_dict.get(op_name, 0.0) > 0:
                latency_with_energy += latency

        # Check if coverage meets threshold
        coverage_ratio = latency_with_energy / total_latency
        return coverage_ratio >= threshold

    def check_oom(self) -> bool:
        """
        Check if total memory usage exceeds GPU capacity.

        Returns True when ``weights + activations + kvcache + overhead >=
        gpu_capacity``.  This is the *absolute* capacity check.

        A separate :meth:`check_kv_cache_oom` exists for the *relative*
        budget check, i.e. whether the KV cache portion alone exceeds the
        ``free_gpu_memory_fraction``-based budget that the serving runtime
        reserves for KV cache.
        """
        if self._is_oom is None:
            logger.warning("WARNING: memory status is not set")
        return self._is_oom

    def set_kv_cache_oom(self, is_kv_cache_oom: bool) -> None:
        """
        Set kv cache oom.
        """
        self._is_kv_cache_oom = is_kv_cache_oom

    def check_kv_cache_oom(self) -> bool:
        """
        Check kv cache oom.
        """
        return self._is_kv_cache_oom

    def get_memory(self) -> dict:
        """
        Get memory breakdown dict (keys: total, weights, activations, kvcache, nccl, others).
        """
        return self._memory

    def get_static_info(self) -> tuple[str, str, str, str]:
        """
        Get static info.
        """

        def get_latency_and_breakdown_percentage_string_helper(metrics: dict) -> tuple[float, str]:
            breakdown_string = ""
            latency = 0
            for op, op_latency in metrics.items():
                latency += op_latency

            breakdown_string += f"total                      ({latency:>10.5f} ms)\n"
            for op, op_latency in metrics.items():
                breakdown_string += f"{op:<25}   {op_latency:>10.3f} ms {int(op_latency / latency * 100):>5}%\n"
            return latency, breakdown_string

        context_latency, context_latency_string = get_latency_and_breakdown_percentage_string_helper(
            self._context_latency_dict
        )
        generation_latency, generation_latency_string = get_latency_and_breakdown_percentage_string_helper(
            self._generation_latency_dict
        )

        assert self._summary_df is not None, "summary df is not set"

        # summary string for display
        perf_info = "Performance Summary:\n"
        perf_info += f"total latency        {(context_latency + generation_latency):>17.5f} ms\n"
        perf_info += f"context latency (ttft):{context_latency:>16.5f} ms\n"
        if generation_latency != 0:
            perf_info += f"generation latency:{generation_latency:>19.5f} ms\n"
            perf_info += (
                f"throughput {self._summary_df.loc[0, 'tokens/s']:.2f} tokens/s, tpot "
                f"{self._summary_df.loc[0, 'tpot']:.3f} ms\n"
            )
        context_info = "Context breakdown:\n" + context_latency_string
        generation_info = "Generation breakdown:\n" + generation_latency_string

        mem_info = "\nMemory Usage: \n"
        for item, memory_usage in self._memory.items():
            mem_info += f"{item:29} {memory_usage:>8.3f} GiB\n"

        return perf_info, mem_info, context_info, generation_info

    # ----- Modern multi-section breakdown report -----

    # Allowed section names accepted by format_detail_report.
    _DETAIL_SECTIONS = ("summary", "memory", "time", "energy", "source")

    @staticmethod
    def _parse_detail(detail: str | set[str] | list[str] | tuple[str, ...]) -> list[str]:
        """Normalize ``detail`` input into an ordered list of section names.

        Accepts:
            - "summary" / "memory" / "time" / "energy" / "source" / "all"
            - "summary,memory,time" (comma-separated string)
            - iterable of the above tokens

        Unknown tokens raise ValueError. Order of returned sections follows
        :pyattr:`_DETAIL_SECTIONS` so the printed report has a stable layout
        regardless of the user's input order.
        """
        if isinstance(detail, str):
            tokens = [t.strip() for t in detail.split(",") if t.strip()]
        else:
            tokens = [str(t).strip() for t in detail if str(t).strip()]
        if not tokens:
            tokens = ["summary"]

        expanded: set[str] = set()
        for tok in tokens:
            if tok == "all":
                expanded.update(InferenceSummary._DETAIL_SECTIONS)
            elif tok in InferenceSummary._DETAIL_SECTIONS:
                expanded.add(tok)
            else:
                raise ValueError(
                    f"Unknown detail section {tok!r}. Allowed: {', '.join(InferenceSummary._DETAIL_SECTIONS)}, all."
                )
        return [s for s in InferenceSummary._DETAIL_SECTIONS if s in expanded]

    @staticmethod
    def _ascii_bar(fraction: float, width: int = 40) -> str:
        """Render a single horizontal bar of width ``width`` cells.

        ``fraction`` is clipped to [0, 1]. Uses block characters ``█`` (filled)
        and ``░`` (empty). Returns exactly ``width`` characters.
        """
        frac = max(0.0, min(1.0, float(fraction)))
        filled = round(frac * width)
        return "█" * filled + "░" * (width - filled)

    @staticmethod
    def _format_op_bars(
        latency_dict: dict[str, float],
        *,
        top_n: int,
        bar_width: int,
        unit: str = "ms",
        source_dict: dict[str, str] | None = None,
    ) -> list[str]:
        """Format per-op latency dict as ASCII bar lines, preserving op order.

        Operations are listed in the dict's natural insertion order — which
        ``_run_context_phase`` / ``_run_generation_phase`` populate by iterating
        ``model.context_ops`` / ``model.generation_ops`` — so readers see the
        breakdown in the same sequence the model defines the forward pass.

        Operations with zero or near-zero latency are skipped. If the dict has
        more than ``top_n`` measurable entries, the trailing entries (still in
        insertion order) are folded into a single ``... (others, k items)`` line
        as a safety cap to keep very-long lists compact.
        """
        if not latency_dict:
            return ["  <no data>"]

        # Preserve insertion order (matches model.{context,generation}_ops order).
        items = [(op, float(lat)) for op, lat in latency_dict.items() if lat > 0.0]
        if not items:
            return ["  <no measurable per-op data>"]

        total = sum(lat for _, lat in items)
        if total <= 0.0:
            return ["  <no measurable per-op data>"]

        lines: list[str] = []
        shown = items[:top_n]
        rest = items[top_n:]
        name_w = min(32, max(len(op) for op, _ in shown))
        for op, lat in shown:
            pct = lat / total * 100.0
            bar = InferenceSummary._ascii_bar(lat / total, width=bar_width)
            src_tag = ""
            if source_dict is not None:
                src = source_dict.get(op)
                if src:
                    src_tag = f" [{src}]"
            lines.append(f"  {op:<{name_w}s}  {lat:>10.3f} {unit}  {pct:>5.1f}%  {bar}{src_tag}")
        if rest:
            rest_lat = sum(lat for _, lat in rest)
            rest_pct = rest_lat / total * 100.0
            bar = InferenceSummary._ascii_bar(rest_lat / total, width=bar_width)
            lines.append(
                f"  {'... (others, ' + str(len(rest)) + ' items)':<{name_w}s}  "
                f"{rest_lat:>10.3f} {unit}  {rest_pct:>5.1f}%  {bar}"
            )
        return lines

    def _format_summary_section(self) -> list[str]:
        ctx_ms = sum(self._context_latency_dict.values())
        gen_ms = sum(self._generation_latency_dict.values())
        total_ms = ctx_ms + gen_ms
        rc = self._runtime_config
        lines = ["Performance Summary"]
        lines.append(f"  total latency       {total_ms:>12.3f} ms")
        if ctx_ms > 0:
            lines.append(f"  context (TTFT)      {ctx_ms:>12.3f} ms")
        if gen_ms > 0:
            tpot = 0.0
            if rc is not None and rc.osl is not None and rc.osl > 1:
                tpot = gen_ms / (rc.osl - 1)
            lines.append(f"  generation          {gen_ms:>12.3f} ms")
            if tpot > 0:
                lines.append(f"  tpot                {tpot:>12.3f} ms")
        if self._summary_df is not None and len(self._summary_df) > 0:
            row = self._summary_df.iloc[0]
            if "tokens/s" in row.index:
                lines.append(f"  throughput          {row['tokens/s']:>12.2f} tokens/s")
            if "seq/s" in row.index:
                lines.append(f"  throughput          {row['seq/s']:>12.3f} seq/s")
        if self._is_oom:
            lines.append("  ⚠ OOM: estimated memory exceeds GPU capacity")
        elif self._is_kv_cache_oom:
            lines.append("  ⚠ KV cache budget exceeded under free_gpu_memory_fraction")
        return lines

    def _format_memory_section(self, bar_width: int) -> list[str]:
        if not self._memory:
            return ["Memory Layout", "  <no memory data>"]

        cap_gib: float | None = None
        if self._mem_capacity_bytes:
            cap_gib = self._mem_capacity_bytes / (1 << 30)

        lines: list[str] = []
        header = "Memory Layout"
        if cap_gib is not None:
            header += f" (capacity {cap_gib:.2f} GiB)"
        lines.append(header)

        # Order the sub-items deterministically; total/others are special.
        order = ["weights", "kvcache", "activations", "nccl", "others"]
        breakdown_keys = [k for k in order if k in self._memory and k != "total"]
        # Include any extra (unexpected) keys at the end except total.
        for k in self._memory:
            if k != "total" and k not in breakdown_keys:
                breakdown_keys.append(k)

        # Denominator for the bars: prefer capacity, fall back to the total
        # so the relative shape is still readable when capacity is unknown.
        total_gib = float(self._memory.get("total", 0.0))
        denom = cap_gib if cap_gib is not None else max(total_gib, 1e-9)

        name_w = max((len(k) for k in breakdown_keys), default=10)
        for key in breakdown_keys:
            gib = float(self._memory[key])
            frac = gib / denom if denom > 0 else 0.0
            bar = self._ascii_bar(frac, width=bar_width)
            pct = frac * 100.0
            lines.append(f"  {key:<{name_w}s}  {gib:>8.3f} GiB  {bar}  {pct:>5.1f}%")

        lines.append("  " + "-" * (name_w + 8 + 3 + bar_width + 8))
        total_frac = total_gib / denom if denom > 0 else 0.0
        # ``free`` is mathematically just ``capacity - total`` (and free% = 100% - total%);
        # we append it on the total line so the absolute remaining GiB is readable at a
        # glance without forcing the reader to subtract, while avoiding a redundant
        # second bar.
        free_suffix = ""
        if cap_gib is not None:
            free_gib = max(0.0, cap_gib - total_gib)
            free_suffix = f"  (free {free_gib:.3f} GiB)"
        lines.append(
            f"  {'total':<{name_w}s}  {total_gib:>8.3f} GiB  "
            f"{self._ascii_bar(total_frac, width=bar_width)}  {total_frac * 100.0:>5.1f}%{free_suffix}"
        )

        # Per-sequence + max-BS section (only when we have KV-per-seq context).
        if self._kv_bytes_per_seq is not None and self._kv_bytes_per_seq > 0:
            kv_per_seq_gib = self._kv_bytes_per_seq / (1 << 30)
            seq_len_str = f" (seq_len={self._kv_seq_len_used})" if self._kv_seq_len_used is not None else ""
            lines.append("")
            lines.append(f"  kvcache/seq  {kv_per_seq_gib:>8.4f} GiB{seq_len_str}")
            if cap_gib is not None:
                kv_total_gib = float(self._memory.get("kvcache", 0.0))
                non_kv_gib = total_gib - kv_total_gib
                free_for_kv_gib = max(0.0, cap_gib - non_kv_gib)
                if kv_per_seq_gib > 0:
                    max_bs = int(free_for_kv_gib // kv_per_seq_gib)
                    extra = ""
                    if self._free_gpu_memory_fraction is not None:
                        eff_frac = (
                            self._free_gpu_memory_fraction
                            * (1.0 - self._kv_cache_reserved_fraction)
                            * (1.0 - self._kv_cache_tolerance)
                        )
                        max_bs_frac = int((free_for_kv_gib * eff_frac) // kv_per_seq_gib)
                        extra = (
                            f"  /  {max_bs_frac} (under free_gpu_memory_fraction={self._free_gpu_memory_fraction:g})"
                        )
                    lines.append(f"  max batch (KV-bound, same isl/osl) ≈ {max_bs}{extra}")
                    lines.append("    note: ignores activation growth with batch; treat as an upper bound.")
        return lines

    def _format_time_section(self, bar_width: int, top_n: int) -> list[str]:
        lines: list[str] = []
        ctx_total = sum(self._context_latency_dict.values())
        gen_total = sum(self._generation_latency_dict.values())

        if ctx_total > 0:
            lines.append(f"Context phase (TTFT = {ctx_total:.3f} ms)")
            lines.extend(
                self._format_op_bars(
                    self._context_latency_dict,
                    top_n=top_n,
                    bar_width=bar_width,
                    unit="ms",
                    source_dict=self._context_source_dict or None,
                )
            )
        if gen_total > 0:
            if ctx_total > 0:
                lines.append("")
            lines.append(f"Generation phase (total = {gen_total:.3f} ms)")
            lines.extend(
                self._format_op_bars(
                    self._generation_latency_dict,
                    top_n=top_n,
                    bar_width=bar_width,
                    unit="ms",
                    source_dict=self._generation_source_dict or None,
                )
            )
        if ctx_total == 0 and gen_total == 0:
            lines.append("Time Breakdown")
            lines.append("  <no per-op latency data>")
        return lines

    def _format_energy_section(self, bar_width: int, top_n: int) -> list[str]:
        ctx = self._context_energy_wms_dict or {}
        gen = self._generation_energy_wms_dict or {}
        if not ctx and not gen:
            return ["Energy Breakdown", "  <no energy data>"]
        lines: list[str] = []
        if ctx:
            total_ctx = sum(ctx.values())
            lines.append(f"Context energy (total = {total_ctx:.3f} W·ms, avg P = {self._context_power_avg:.1f} W)")
            lines.extend(self._format_op_bars(ctx, top_n=top_n, bar_width=bar_width, unit="W·ms"))
        if gen:
            if ctx:
                lines.append("")
            total_gen = sum(gen.values())
            lines.append(
                f"Generation energy (total = {total_gen:.3f} W·ms, avg P = {self._generation_power_avg:.1f} W)"
            )
            lines.extend(self._format_op_bars(gen, top_n=top_n, bar_width=bar_width, unit="W·ms"))
        return lines

    def _format_source_section(self) -> list[str]:
        ctx = self._context_source_dict or {}
        gen = self._generation_source_dict or {}
        if not ctx and not gen:
            return ["Data Source Breakdown", "  <no source data>"]

        def _summarize(d: dict[str, str]) -> str:
            counts: dict[str, int] = {}
            for v in d.values():
                counts[v] = counts.get(v, 0) + 1
            ordered = sorted(counts.items(), key=lambda kv: -kv[1])
            return ", ".join(f"{k}={v}" for k, v in ordered)

        lines = ["Data Source Breakdown (per-op)"]
        if ctx:
            lines.append(f"  context     {_summarize(ctx)}")
            for op, src in sorted(ctx.items()):
                lines.append(f"    {op:<30s} {src}")
        if gen:
            lines.append(f"  generation  {_summarize(gen)}")
            for op, src in sorted(gen.items()):
                lines.append(f"    {op:<30s} {src}")
        return lines

    def format_detail_report(
        self,
        detail: str | set[str] | list[str] | tuple[str, ...] = "summary",
        width: int = 80,
        top_n_ops: int = 12,
    ) -> str:
        """Return a formatted multi-section breakdown string.

        Args:
            detail: Which sections to include. Accepts a single section name,
                a comma-separated string, an iterable of names, or ``"all"``.
                Valid section names: ``summary``, ``memory``, ``time``,
                ``energy``, ``source``. Default is ``"summary"``.
            width: Total render width hint; controls bar widths. Default 80.
            top_n_ops: Maximum number of per-op rows shown in time / energy
                sections before folding into ``... (others)``. Default 12.

        Returns:
            A single multi-line string ready for ``print()``.

        See also:
            :meth:`get_static_info` — the legacy 4-tuple variant used by the
            webapp Static Tab. This method is a superset and is preferred for
            new call sites; the legacy method is kept for backward compatibility.
        """
        sections = self._parse_detail(detail)
        # Bars get whatever's left after the longest label + numeric columns.
        bar_width = max(16, width - 40)

        out: list[str] = []
        for section in sections:
            if out:
                out.append("")
            if section == "summary":
                out.extend(self._format_summary_section())
            elif section == "memory":
                out.extend(self._format_memory_section(bar_width=bar_width))
            elif section == "time":
                out.extend(self._format_time_section(bar_width=bar_width, top_n=top_n_ops))
            elif section == "energy":
                out.extend(self._format_energy_section(bar_width=bar_width, top_n=top_n_ops))
            elif section == "source":
                out.extend(self._format_source_section())
        return "\n".join(out)

    def set_summary_df(self, summary_df: pd.DataFrame) -> None:
        """
        Set summary dataframe.
        """
        self._summary_df = summary_df

    def get_summary_df(self) -> pd.DataFrame:
        """
        Get summary dataframe.
        """
        if self._summary_df is None:
            logger.warning("WARNING: summary df is not set")
        return self._summary_df

    def set_per_ops_data(self, per_ops_data: dict) -> None:
        """Set per-operation latency breakdown data from run_agg."""
        self._per_ops_data = per_ops_data

    def get_per_ops_data(self) -> dict | None:
        """Get per-operation latency breakdown data (populated by run_agg)."""
        return self._per_ops_data

    def set_per_ops_source(self, per_ops_source: dict) -> None:
        """Set per-operation data-source breakdown ("silicon"/"empirical"/"mixed")."""
        self._per_ops_source = per_ops_source

    def get_per_ops_source(self) -> dict | None:
        """Get per-operation data-source breakdown, parallel to per_ops_data."""
        return self._per_ops_source

    def set_context_source_dict(self, context_source_dict: dict) -> None:
        """Set the per-op data source dict for the context (prefill) phase."""
        self._context_source_dict = context_source_dict

    def get_context_source_dict(self) -> dict:
        """Get the per-op data source dict for the context (prefill) phase."""
        return self._context_source_dict

    def set_generation_source_dict(self, generation_source_dict: dict) -> None:
        """Set the per-op data source dict for the generation (decode) phase."""
        self._generation_source_dict = generation_source_dict

    def get_generation_source_dict(self) -> dict:
        """Get the per-op data source dict for the generation (decode) phase."""
        return self._generation_source_dict

    # --- Capacity / KV-per-seq probing context (used by format_detail_report) ---

    def set_kv_per_seq(self, kv_bytes_per_seq: float, seq_len_used: int) -> None:
        """Stash per-sequence KV cache footprint context for capacity probing.

        Args:
            kv_bytes_per_seq: KV cache bytes consumed by a single sequence on
                one GPU at the seq length actually used for memory estimation.
            seq_len_used: The seq length used (typically ``isl + beam_width * osl``,
                or ``max_seq_len`` when provided by the backend).
        """
        self._kv_bytes_per_seq = float(kv_bytes_per_seq)
        self._kv_seq_len_used = int(seq_len_used)

    def get_kv_per_seq(self) -> tuple[float | None, int | None]:
        """Return the (kv_bytes_per_seq, seq_len_used) pair, or (None, None) if unset."""
        return self._kv_bytes_per_seq, self._kv_seq_len_used

    def get_mem_capacity_bytes(self) -> int | None:
        """Return the GPU memory capacity (bytes) captured by set_memory_and_check_oom, or None."""
        return self._mem_capacity_bytes

    def set_result_dict(self, result_dict: dict) -> None:
        """
        Set the cached result dict for efficient batch operations.
        """
        self._result_dict = result_dict

    def get_result_dict(self) -> dict | None:
        """
        Get the result as a dict. Returns cached dict if available,
        otherwise extracts from the first row of the summary DataFrame.
        """
        if self._result_dict is not None:
            return self._result_dict

        # Fallback: create from DataFrame if not cached
        if self._summary_df is not None and len(self._summary_df) > 0:
            return self._summary_df.iloc[0].to_dict()
        return None

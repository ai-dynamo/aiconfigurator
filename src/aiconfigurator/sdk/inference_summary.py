# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
        set_memory_and_check_oom: set memory and check oom
        set_oom: set oom
        set_context_latency_dict: set context latency dict
        set_generation_latency_dict: set generation latency dict
        get_context_latency_dict: get context latency dict
        get_generation_latency_dict: get generation latency dict
        set_context_energy_wms_dict: set context energy dict
        set_generation_energy_wms_dict: set generation energy dict
        get_context_energy_wms_dict: get context energy dict
        get_generation_energy_wms_dict: get generation energy dict
        check_oom: check oom
        get_static_info: get static info for static mode print
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
        self._is_oom = None

        # NEW: Store computed power averages
        self._context_power_avg = 0.0
        self._generation_power_avg = 0.0
        self._e2e_power_avg = 0.0

        # summary dataframe
        self._summary_df = None

        # cached result dict for efficient batch operations
        self._result_dict = None

    def set_memory_and_check_oom(self, memory_dict: dict, mem_capacity: int) -> None:
        """
        Set memory and check oom.
        """
        self._memory = memory_dict
        self._is_oom = self._memory["total"] >= (mem_capacity / (1 << 30))

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
        Check oom.
        """
        if self._is_oom is None:
            logger.warning("WARNING: memory status is not set")
        return self._is_oom

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

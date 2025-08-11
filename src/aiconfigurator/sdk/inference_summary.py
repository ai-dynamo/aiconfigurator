# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import pandas as pd
from aiconfigurator.sdk.config import RuntimeConfig
logger = logging.getLogger(__name__)


class InferenceSummary(object):
    '''
    InferecneSummary to hold results of inference

    Attributes:
        runtime_config: runtime config
        memory: memory breakdown
        context_latency_dict: latency breakdown for context
        generation_latency_dict: latency breakdown for generation
        summary_df: summary dataframe

    Methods:
        set_memory_and_check_oom: set memory and check oom
        set_oom: set oom
        set_context_latency_dict: set context latency dict
        set_generation_latency_dict: set generation latency dict
        get_context_latency_dict: get context latency dict
        get_generation_latency_dict: get generation latency dict
        check_oom: check oom
        get_static_info: get static info for static mode print
        set_summary_df: set summary dataframe
        get_summary_df: get summary dataframe
    '''
    def __init__(self, runtime_config: RuntimeConfig) -> None:
        """
        Initialize inference summary.
        """
        self._runtime_config = runtime_config

        # raw data dict        
        self._memory = {}
        self._context_latency_dict = {}
        self._generation_latency_dict = {}
        self._is_oom = None

        # summary dataframe
        self._summary_df = None        

    def set_memory_and_check_oom(self, memory_dict: dict, mem_capacity: int) -> None:
        """
        Set memory and check oom.
        """
        self._memory = memory_dict
        self._is_oom = self._memory['total'] >= (mem_capacity / (1<<30))

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

    def check_oom(self) -> bool:
        """
        Check oom.
        """
        if self._is_oom is None:
            logger.warning('WARNING: memory status is not set')     
        return self._is_oom

    def get_static_info(self) -> tuple[str, str, str, str]:
        """
        Get static info.
        """
        def get_latency_and_breakdown_percentage_string_helper(metrics: dict) -> tuple[float, str]:
            breakdown_string = ''
            latency = 0
            for op, op_latency in metrics.items():
                latency += op_latency

            breakdown_string += f'total                      ({latency:>10.5f} ms)\n'
            for op, op_latency in metrics.items():
                breakdown_string += f'{op:<25}   {op_latency:>10.3f} ms {int(op_latency/latency*100):>5}%\n'
            return latency, breakdown_string
        context_latency, context_latency_string = get_latency_and_breakdown_percentage_string_helper(self._context_latency_dict)
        generation_latency, generation_latency_string = get_latency_and_breakdown_percentage_string_helper(self._generation_latency_dict)
        
        assert self._summary_df is not None, 'summary df is not set'

        # summary string for display
        perf_info = 'Performance Summary:\n'
        perf_info += f'total latency        {(context_latency+generation_latency):>17.5f} ms\n'
        perf_info += f'context latency (ttft):{context_latency:>16.5f} ms\n'
        if generation_latency !=0:
            perf_info += f'generation latency:{generation_latency:>19.5f} ms\n'
            perf_info += f'throughput {self._summary_df.loc[0, "tokens/s"]:.2f} tokens/s, tpot {self._summary_df.loc[0, "tpot"]:.3f} ms\n'
        context_info = 'Context breakdown:\n'+context_latency_string
        generation_info = 'Generation breakdown:\n'+generation_latency_string

        mem_info = '\nMemory Usage: \n'
        for item, memory_usage in self._memory.items():
            mem_info += f'{item:29} {memory_usage:>8.3f} GiB\n'

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
            logger.warning('WARNING: summary df is not set')                
        return self._summary_df

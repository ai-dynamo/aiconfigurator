# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import pandas as pd

from aiconfigurator.sdk import common
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.inference_summary import InferenceSummary
from aiconfigurator.sdk.models import BaseModel
from aiconfigurator.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)


class BaseBackend(ABC):
    """
    Base class for all backends.
    All backends should inherit from this class and implement the abstract methods.
    All backends should implement the following methods:

    Attributes:

    Methods:
        run_static: this is common for all backends. It's implemented in this class.
            If there might be some backend-specific logic, it should be implemented in the subclass.
        run_agg: this is backend-specific. It should be implemented in the subclass.
        find_best_agg_result_under_constraints: this is backend-specific.
            It should be implemented in the subclass.
        _get_memory_usage: this is backend-specific. It should be implemented in the subclass.
    """

    def run_static(
        self,
        model: BaseModel,
        database: PerfDatabase,
        runtime_config: RuntimeConfig,
        mode: str,
        stride: int = 32,
        latency_correction_scale: float = 1.0,
    ) -> InferenceSummary:
        """
        Run the static inference.

        Args:
            model (BaseModel): the model to run inference
            database (PerfDatabase): the database to run inference
            runtime_config (RuntimeConfig): the runtime config
            mode (str): the mode to run inference, static, static_ctx, static_gen
            stride (int): the stride is used to accelerate the estimation, for a give osl,
                will only computes the i, i+stride, i+2*stride, ... step, default is 32.
            latency_correction_scale (float): the correction scale to adjust the latency,
                default is 1.0.
                corrected latency = latency * latency_correction_scale
        """

        def _run_context(batch_size: int, isl: int, prefix) -> tuple[dict[str, float], dict[str, float]]:
            """
            Run context phase.
            
            Returns:
                tuple: (context_latency_dict, context_power_dict)
            """
            context_latency_dict = defaultdict(float)
            context_power_dict = defaultdict(float)  # NEW

            # isl is corrected based on prefix.
            # Please handle the real logic in your context attention related operations.
            isl = isl - prefix
            if isl <= 0:
                raise ValueError(f"isl must be greater than 0 after removing prefix, but got {isl}")

            for op in model.context_ops:
                # query latency and store the latency
                x = batch_size * isl if "logits_gemm" not in op._name else batch_size
                result = op.query(database, x=x, batch_size=batch_size, beam_width=1, s=isl, prefix=prefix)
                
                # Extract latency (automatic, it's a float)
                latency = float(result)
                context_latency_dict[op._name] += latency
                
                # Extract power and weight by latency
                # This gives us sum(power * time) for weighted average calculation
                power = getattr(result, 'power', 0.0)  # Safe: defaults to 0 if attribute missing
                context_power_dict[op._name] += power * latency

            return context_latency_dict, context_power_dict

        def _run_generation(batch_size: int, beam_width: int, isl: int, osl: int, stride: int) -> tuple[dict[str, float], dict[str, float]]:
            """
            Run generation phase.
            
            Returns:
                tuple: (generation_latency_dict, generation_power_dict)
            """
            # mtp/speculative decoding correction
            batch_size = batch_size * (model._nextn + 1)

            generation_latency_dict = defaultdict(float)
            generation_power_dict = defaultdict(float)  # NEW

            for i in range(0, osl - 1, stride):
                latency_dict = defaultdict(float)
                power_dict = defaultdict(float)  # NEW
                
                for op in model.generation_ops:
                    result = op.query(
                        database,
                        x=batch_size * beam_width,
                        batch_size=batch_size,
                        beam_width=beam_width,
                        s=isl + i + 1,
                    )
                    
                    # Extract latency and power
                    latency_dict[op._name] += float(result)
                    power_dict[op._name] += getattr(result, 'power', 0.0)

                # usually stride, but might be less at the end
                repeat_count = min(stride, osl - 1 - i)

                for op in latency_dict.keys():
                    # Latency is cumulative - multiply by repeat_count  
                    generation_latency_dict[op] += latency_dict[op] * repeat_count
                    # Power is instantaneous (watts) - weight by latency*repeat_count when accumulating
                    # This gives us sum(power * time) which we'll divide by sum(time) for weighted average
                    generation_power_dict[op] += power_dict[op] * latency_dict[op] * repeat_count

            return generation_latency_dict, generation_power_dict

        summary = InferenceSummary(runtime_config)
        batch_size, beam_width, isl, osl, prefix = (
            runtime_config.batch_size,
            runtime_config.beam_width,
            runtime_config.isl,
            runtime_config.osl,
            runtime_config.prefix,
        )

        # Execute phases (MODIFIED to unpack power dicts)
        context_latency_dict, context_power_dict = {}, {}
        generation_latency_dict, generation_power_dict = {}, {}
        
        if mode == "static_ctx":
            context_latency_dict, context_power_dict = _run_context(batch_size, isl, prefix)
            memory = self._get_memory_usage(model, database, batch_size, beam_width, isl, 1)
        elif mode == "static_gen":
            generation_latency_dict, generation_power_dict = _run_generation(batch_size, beam_width, isl, osl, stride)
            memory = self._get_memory_usage(
                model,
                database,
                batch_size,
                beam_width,
                isl,
                osl,
                num_tokens=batch_size * beam_width,
            )  # for gen only, all kvcache is needed.
        else:
            context_latency_dict, context_power_dict = _run_context(batch_size, isl, prefix)
            generation_latency_dict, generation_power_dict = _run_generation(batch_size, beam_width, isl, osl, stride)
            memory = self._get_memory_usage(model, database, batch_size, beam_width, isl, osl)

        if latency_correction_scale != 1.0:
            logger.debug(f"latency_correction_scale: {latency_correction_scale} is applied to the latency")
            for op in context_latency_dict:
                context_latency_dict[op] *= latency_correction_scale
            for op in generation_latency_dict:
                generation_latency_dict[op] *= latency_correction_scale

        # Calculate total latencies
        context_latency = sum(context_latency_dict.values())
        generation_latency = sum(generation_latency_dict.values())
        
        # NEW: Calculate weighted average power
        # Formula: sum(power * latency) / sum(latency)
        context_power_avg = 0.0
        if context_latency > 0:
            # context_power_dict contains sum(power * latency)
            # context_latency contains sum(latency)
            # So: avg_power = sum(power * latency) / sum(latency)
            weighted_power_sum = sum(context_power_dict[op] for op in context_latency_dict.keys())
            context_power_avg = weighted_power_sum / context_latency
        
        generation_power_avg = 0.0
        if generation_latency > 0:
            # generation_power_dict contains sum(power * latency * repeat_count)
            # generation_latency contains sum(latency * repeat_count)
            # So: avg_power = sum(power * latency) / sum(latency)
            weighted_power_sum = sum(generation_power_dict[op] for op in generation_latency_dict.keys())
            generation_power_avg = weighted_power_sum / generation_latency
        
        # E2E weighted average power
        total_latency = context_latency + generation_latency
        e2e_power_avg = 0.0
        if total_latency > 0:
            e2e_power_avg = (
                context_power_avg * context_latency + 
                generation_power_avg * generation_latency
            ) / total_latency

        bs = batch_size
        global_bs = bs * model.config.attention_dp_size
        concurrency = global_bs
        ttft = context_latency
        tpot = 0.0 if osl <= 1 else generation_latency / (osl - 1)
        num_generated_tokens = max(osl - 1, 0)
        request_latency = ttft + tpot * num_generated_tokens
        if request_latency == 0.0:
            request_latency = context_latency + generation_latency
        request_rate = 0.0
        seq_s = (
            0.0 if request_latency == 0.0 else global_bs / request_latency * 1000 * model.config.pp_size
        )  # handle statc_gen only with osl==1, scale by pp
        seq_s_gpu = seq_s / model.config.tp_size / model.config.pp_size / model.config.attention_dp_size
        tokens_s = seq_s * osl if mode != "static_gen" else seq_s * (osl - 1)
        if mode == "static_ctx":
            tokens_s = seq_s * 1  # only first token
        tokens_s_gpu = tokens_s / model.config.tp_size / model.config.pp_size / model.config.attention_dp_size
        tokens_s_user = 0.0 if tpot == 0.0 else 1000.0 / tpot
        tp = model.config.tp_size
        pp = model.config.pp_size
        dp = model.config.attention_dp_size
        moe_tp = model.config.moe_tp_size
        moe_ep = model.config.moe_ep_size
        num_total_gpus = tp * pp * dp
        parallel = f"tp{tp}pp{pp}dp{dp}etp{moe_tp}ep{moe_ep}"
        gemm = model.config.gemm_quant_mode.name
        kvcache = model.config.kvcache_quant_mode.name
        fmha = model.config.fmha_quant_mode.name
        moe = model.config.moe_quant_mode.name
        comm = model.config.comm_quant_mode.name
        mem = memory["total"]

        data = [
            [
                model.model_name,
                isl,
                osl,
                prefix,
                concurrency,
                request_rate,
                bs,
                global_bs,
                ttft,
                tpot,
                seq_s,
                seq_s_gpu,
                tokens_s,
                tokens_s_gpu,
                tokens_s_user,
                request_latency,
                context_latency,
                generation_latency,
                num_total_gpus,
                tp,
                pp,
                dp,
                moe_tp,
                moe_ep,
                parallel,
                gemm,
                kvcache,
                fmha,
                moe,
                comm,
                mem,
                database.backend,
                database.version,
                database.system,
                e2e_power_avg,  # NEW: E2E weighted average power in watts
            ]
        ]

        summary_df = pd.DataFrame(data, columns=common.ColumnsStatic).round(3)

        summary.set_context_latency_dict(context_latency_dict)
        summary.set_generation_latency_dict(generation_latency_dict)
        summary.set_context_power_dict(context_power_dict)  # NEW
        summary.set_generation_power_dict(generation_power_dict)  # NEW
        summary.set_memory_and_check_oom(memory, database.system_spec["gpu"]["mem_capacity"])
        summary.set_summary_df(summary_df)

        return summary

    def _get_ctx_tokens_list_for_agg_sweep(
        self,
        isl: int,
        ctx_stride: int,
        enable_chunked_prefill: bool,
        max_normal_ctx_tokens: int = 8192,
        max_ctx_tokens_multiple_of_isl: int = 2,
        max_ctx_tokens_small_search_steps: int = 16,
        max_ctx_tokens_search_steps: int = 8,
    ) -> list[int]:
        """
        Generate a list of num_context_tokens to sweep for agg inference.

        Args:
            isl: Target input sequence length during inference.
            ctx_stride: Default stride for context_tokens to sweep, ignored if enable_chunked_prefill is True.
            enable_chunked_prefill: Whether the inference framework will have chunked_prefill enabled.
            max_normal_ctx_tokens: boundary at which to increase the stride for faster sweeping.
            max_ctx_tokens_multiple_of_isl: Maximum multiple of isl to consider for ctx tokens.
            max_ctx_tokens_small_search_steps: Maximum search steps under max_normal_ctx_tokens.
            max_ctx_tokens_large_search_steps: Maximum search steps over max_normal_ctx_tokens.
        Returns:
            Sorted list of num_context_tokens to sweep.
        """

        # Largest ctx_tokens to consider for sweeping.
        max_ctx_tokens = max(max_normal_ctx_tokens, isl * max_ctx_tokens_multiple_of_isl)

        # Sweep stride under max_normal_ctx_tokens.
        ctx_stride = max(ctx_stride, max_normal_ctx_tokens // max_ctx_tokens_small_search_steps)

        # Sweep stride once ctx_tokens is larger than max_normal_ctx_tokens.
        ctx_stride_large = max(
            1024,
            ctx_stride,
            max_ctx_tokens // max_ctx_tokens_search_steps,
        )

        if not enable_chunked_prefill:
            new_ctx_stride = max(isl, ctx_stride)
            new_ctx_stride_large = int(np.ceil(ctx_stride_large / isl) * isl)
            logger.debug(
                f"enable_chunked_prefill is off, override ctx_stride: from {ctx_stride} to {new_ctx_stride}, "
                f"ctx_stride_large: from {ctx_stride_large} to {new_ctx_stride_large}"
            )
            ctx_stride = new_ctx_stride
            ctx_stride_large = new_ctx_stride_large

        # prepare ctx_tokens_list
        ctx_tokens_list = []
        ctx_tokens = 0
        while True:
            if ctx_tokens < max_normal_ctx_tokens:
                ctx_tokens += ctx_stride
            else:
                ctx_tokens += ctx_stride_large

            if ctx_tokens > max_ctx_tokens:
                break

            ctx_tokens_list.append(ctx_tokens)

        # add those just match the multiple of isl
        for i in range(1, max_ctx_tokens_multiple_of_isl + 1):
            ctx_tokens = isl * i
            if ctx_tokens not in ctx_tokens_list:
                ctx_tokens_list.append(ctx_tokens)
        ctx_tokens_list.sort()
        return ctx_tokens_list

    @abstractmethod
    def run_agg(
        self, model: BaseModel, database: PerfDatabase, runtime_config: RuntimeConfig, **kwargs
    ) -> InferenceSummary:
        """
        Run the agg inference.
        """
        pass

    @abstractmethod
    def find_best_agg_result_under_constraints(
        self, model: BaseModel, database: PerfDatabase, runtime_config: RuntimeConfig, **kwargs
    ) -> InferenceSummary:
        """
        Find the best agg result under constraints.
        """
        pass

    @abstractmethod
    def _get_memory_usage(
        self,
        model: BaseModel,
        database: PerfDatabase,
        batch_size: int,
        beam_width: int,
        isl: int,
        osl: int,
        num_tokens: int = 0,
    ) -> dict[str, float]:
        """
        Get the memory usage of the backend.
        """
        pass

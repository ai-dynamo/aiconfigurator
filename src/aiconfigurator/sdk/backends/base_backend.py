# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections import defaultdict
import copy
from aiconfigurator.sdk.inference_summary import InferenceSummary
import pandas as pd
from aiconfigurator.sdk import common
from aiconfigurator.sdk.models import BaseModel
from aiconfigurator.sdk.perf_database import PerfDatabase
from aiconfigurator.sdk.config import RuntimeConfig

class BaseBackend(ABC):
    """
    Base class for all backends.
    All backends should inherit from this class and implement the abstract methods.
    All backends should implement the following methods:

    Attributes:

    Methods:
        run_static: this is common for all backends. It's implemented in this class. If there might be some backend-specific logic, it should be implemented in the subclass.
        run_ifb: this is backend-specific. It should be implemented in the subclass.
        find_best_ifb_result_under_constraints: this is backend-specific. It should be implemented in the subclass.
        _get_memory_usage: this is backend-specific. It should be implemented in the subclass.
    """
    def run_static(self, 
                   model: BaseModel, 
                   database: PerfDatabase, 
                   runtime_config: RuntimeConfig, 
                   mode: str, 
                   stride: int = 32) -> InferenceSummary:
        """
        Run the static inference.
        """
        def _run_context(batch_size: int, isl: int) -> dict[str, float]:
            context_latency_dict = defaultdict(float)

            for op in model.context_ops:
                #query latency and store the latency
                x = batch_size*isl if 'logits_gemm' not in op._name else batch_size
                latency = op.query(database, x=x, batch_size=batch_size, beam_width=1, s=isl)
                context_latency_dict[op._name] += latency

            return context_latency_dict

        def _run_generation(batch_size: int, beam_width: int, isl: int, osl: int, stride: int) -> dict[str, float]:
            # mtp/speculative decoding correction
            batch_size = batch_size*(model._nextn+1)

            latencies = []
            cached_latency_dict = None
            for i in range(osl-1):

                if i%stride != 0:
                    latencies.append(copy.deepcopy(cached_latency_dict))
                    continue

                latency_dict = defaultdict(float)
                for op in model.generation_ops:
                    latency = op.query(database, x=batch_size*beam_width, batch_size=batch_size, beam_width=beam_width, s=isl+i+1)
                    latency_dict[op._name] += latency
                cached_latency_dict = latency_dict

                latencies.append(latency_dict)
        
            generation_latency_dict = {}
            if len(latencies) > 0:
                for key in latencies[0].keys():
                    generation_latency_dict[key] = 0.0
                    for latency_dict in latencies:
                        generation_latency_dict[key] += latency_dict[key]
            
            return generation_latency_dict

        summary = InferenceSummary(runtime_config)
        batch_size, beam_width, isl, osl = runtime_config.batch_size, runtime_config.beam_width, runtime_config.isl, runtime_config.osl
        
        context_latency_dict, generation_latency_dict = {}, {}
        if mode == 'static_ctx':
            context_latency_dict = _run_context(batch_size, isl)
            memory = self._get_memory_usage(model, database, batch_size, beam_width, isl, 1)
        elif mode == 'static_gen':
            generation_latency_dict = _run_generation(batch_size, beam_width, isl, osl, stride)
            memory = self._get_memory_usage(model, database, batch_size, beam_width, isl, osl, num_tokens=batch_size*beam_width) # for gen only, all kvcache is needed.
        else:   
            context_latency_dict = _run_context(batch_size, isl)
            generation_latency_dict = _run_generation(batch_size, beam_width, isl, osl, stride)
            memory = self._get_memory_usage(model, database, batch_size, beam_width, isl, osl)

        context_latency, generation_latency = 0.0, 0.0
        for op, op_latency in context_latency_dict.items():
            context_latency += op_latency
        for op, op_latency in generation_latency_dict.items():
            generation_latency += op_latency

        bs = batch_size        
        global_bs = bs * model.config.attention_dp_size
        concurrency = global_bs
        latency = context_latency + generation_latency
        request_rate = 0.0
        ttft = context_latency
        tpot = 0.0 if osl <= 1 else generation_latency / (osl-1)
        seq_s = 0.0 if latency == 0.0 else global_bs / latency * 1000 * model.config.pp_size # handle statc_gen only with osl==1, scale by pp
        seq_s_gpu = seq_s / model.config.tp_size / model.config.pp_size / model.config.attention_dp_size
        tokens_s = seq_s * osl if mode != 'static_gen' else seq_s * (osl-1)
        if mode == 'static_ctx':
            tokens_s = seq_s * 1 # only first token
        tokens_s_gpu = tokens_s / model.config.tp_size / model.config.pp_size / model.config.attention_dp_size
        tokens_s_user = 0.0 if tpot == 0.0 else 1000.0 / tpot
        tp = model.config.tp_size
        pp = model.config.pp_size
        dp = model.config.attention_dp_size
        moe_tp = model.config.moe_tp_size
        moe_ep = model.config.moe_ep_size
        num_total_gpus = tp*pp*dp
        parallel = f'tp{tp}pp{pp}dp{dp}etp{moe_tp}ep{moe_ep}'
        gemm = model.config.gemm_quant_mode.name
        kvcache = model.config.kvcache_quant_mode.name
        fmha = model.config.fmha_quant_mode.name
        moe = model.config.moe_quant_mode.name
        comm = model.config.comm_quant_mode.name
        mem = memory['total']
        
        
        data = [[model.model_name, isl, osl, \
                 concurrency, request_rate, bs, global_bs, \
                 ttft, tpot, seq_s, seq_s_gpu, tokens_s, tokens_s_gpu, tokens_s_user, latency, context_latency, generation_latency, \
                 num_total_gpus, \
                 tp, pp, dp, moe_tp, moe_ep, parallel, \
                 gemm, kvcache, fmha, moe, comm, \
                 mem,
                 database.backend, database.version, database.system]]

        summary_df = pd.DataFrame(data, columns=common.ColumnsStatic).round(3)

        summary.set_context_latency_dict(context_latency_dict)
        summary.set_generation_latency_dict(generation_latency_dict)
        summary.set_memory_and_check_oom(memory, database.system_spec['gpu']['mem_capacity'])
        summary.set_summary_df(summary_df)

        return summary

    @abstractmethod
    def run_ifb(self, 
                model: BaseModel, 
                database: PerfDatabase, 
                runtime_config: RuntimeConfig, 
                **kwargs) -> InferenceSummary:
        """
        Run the IFB inference.
        """
        pass

    @abstractmethod
    def find_best_ifb_result_under_constraints(self, 
                                               model: BaseModel, 
                                               database: PerfDatabase, 
                                               runtime_config: RuntimeConfig, 
                                               **kwargs) -> InferenceSummary:
        """
        Find the best IFB result under constraints.
        """
        pass

    @abstractmethod
    def _get_memory_usage(self, 
                          model: BaseModel, 
                          database: PerfDatabase, 
                          batch_size: int, 
                          beam_width: int, 
                          isl: int, 
                          osl: int, num_tokens: int = 0) -> dict[str, float]:
        """
        Get the memory usage of the backend.
        """
        pass